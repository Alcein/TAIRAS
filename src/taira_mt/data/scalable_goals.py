"""Build target-free category goals from frozen relation statistics."""

from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from pydantic import Field, model_validator

from taira_mt.data.goal_construction import CategoryRelation, NormalizedCatalogRow
from taira_mt.data.numeric_goal_specs import (
    POPULARITY_MAPPING_VERSION,
    GoalFamily,
    NumericGoalSpec,
    popularity_percentiles,
)
from taira_mt.models import (
    AttributeValue,
    BeliefState,
    CatalogItem,
    Constraint,
    ConstraintImportance,
    HiddenGoal,
    PublicConstraint,
    StrictModel,
    UserPersona,
)
from taira_mt.pilot.schemas import CandidatePoolRecord
from taira_mt.provenance import content_sha256
from taira_mt.recommender import CandidatePool, CatalogIndex, rank_catalog_from_belief
from taira_mt.state import DomainSlotOntology, SlotDefinition, initialize_belief

AUTOMATIC_BUILDER_VERSION = "target_free_category_v2"
AUTOMATIC_POOL_RECIPE = "public_visible_90_plus_far_10_v2"
VISIBLE_GROUP_SPLIT_RECIPE = "unseen_visible_group_80_10_10_v1"
VISIBLE_GROUP_ID_SCHEMA_VERSION = "public_visible_group_v2"


class GoalDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class InteractionScenario(str, Enum):
    CLARIFY = "clarify"
    READY = "ready"
    REPAIR = "repair"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    CALIBRATION = "calibration"
    TEST = "test"


class CandidateGoalLabel(str, Enum):
    ACCEPTABLE = "acceptable"
    HIDDEN_CATEGORY_NEGATIVE = "hidden_category_negative"
    NUMERIC_BOUNDARY_NEGATIVE = "numeric_boundary_negative"
    NUMERIC_MISSING = "numeric_missing"
    SOFT_NEGATIVE = "soft_negative"
    FAR_NEGATIVE = "far_negative"


class CandidateGoalStratum(str, Enum):
    HIDDEN_CATEGORY_NEGATIVE = "hidden_category_negative"
    PRICE_BELOW_MINIMUM = "price_below_minimum"
    PRICE_ABOVE_MAXIMUM = "price_above_maximum"
    RATING_BELOW_MINIMUM = "rating_below_minimum"
    POPULARITY_BELOW_MINIMUM = "popularity_below_minimum"
    NUMERIC_MISSING = "numeric_missing"
    SOFT_NEGATIVE = "soft_negative"
    FAR_NEGATIVE = "far_negative"


class AutomaticPublicGoal(StrictModel):
    case_id: str
    domain: str
    split: DatasetSplit
    scenario: InteractionScenario
    difficulty: GoalDifficulty
    initial_request: str
    observed_constraints: dict[str, PublicConstraint]
    ontology_id: str
    candidate_pool_id: str
    goal_family: GoalFamily = GoalFamily.C
    numeric_spec_id: str | None = None
    split_group_id: str | None = None
    split_strategy: str = "legacy_relation_hash_v1"


class CandidatePrivateLabel(StrictModel):
    item_id: str
    label: CandidateGoalLabel
    utility: float = Field(ge=0.0, le=1.0)
    acceptable: bool | None = None
    strata: list[CandidateGoalStratum] = Field(default_factory=list)
    violated_hard_slots: list[str] = Field(default_factory=list)
    missed_soft_slots: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def fill_legacy_acceptability(self) -> CandidatePrivateLabel:
        expected = self.label == CandidateGoalLabel.ACCEPTABLE
        if self.acceptable is None:
            self.acceptable = expected
        elif self.acceptable != expected and self.label in {
            CandidateGoalLabel.ACCEPTABLE,
            CandidateGoalLabel.HIDDEN_CATEGORY_NEGATIVE,
            CandidateGoalLabel.FAR_NEGATIVE,
        }:
            raise ValueError("legacy candidate label disagrees with acceptable flag")
        if len(self.strata) != len(set(self.strata)):
            raise ValueError("candidate strata must be unique")
        return self


class AutomaticPrivateGoal(StrictModel):
    case_id: str
    relation_id: str
    hidden_goal: HiddenGoal
    persona: UserPersona
    visible_tag: str
    hidden_tag: str
    # A base builder still rejects a goal with no full-catalog acceptable item.
    # Persona expansion, however, intentionally retains stricter persona
    # variants even when their threshold leaves no acceptable catalog item.
    # Keeping those no-solution variants is necessary for an honest
    # abstention/abandonment evaluation rather than silently filtering them.
    full_feasible_count: int = Field(ge=0)
    full_feasible_set_hash: str
    pool_labels: list[CandidatePrivateLabel]
    pool_acceptable_item_ids: list[str]
    goal_family: GoalFamily = GoalFamily.C
    numeric_spec: NumericGoalSpec | None = None
    public_pool_seed: int | None = None
    pool_replica: int = Field(
        default=0,
        ge=0,
        description="Legacy v1 read-compatibility field; v2 construction never uses it.",
    )
    construction_version: str = "target_free_category_v1"

    @model_validator(mode="after")
    def validate_numeric_family(self) -> AutomaticPrivateGoal:
        if self.goal_family == GoalFamily.C and self.numeric_spec is not None:
            raise ValueError("category-only goal cannot carry a numeric spec")
        if self.goal_family != GoalFamily.C:
            if self.numeric_spec is None or self.numeric_spec.family != self.goal_family:
                raise ValueError("numeric goal family must agree with its private spec")
        return self


@dataclass(frozen=True)
class BuiltAutomaticGoal:
    public_goal: AutomaticPublicGoal
    private_goal: AutomaticPrivateGoal
    ontology: DomainSlotOntology
    initial_belief: BeliefState
    catalog_index: CatalogIndex
    candidate_pool: CandidatePool
    pool_record: CandidatePoolRecord


def relation_difficulty(relation: CategoryRelation) -> GoalDifficulty:
    prevalence = relation.hidden_given_visible
    if prevalence >= 0.20:
        return GoalDifficulty.EASY
    if prevalence >= 0.10:
        return GoalDifficulty.MEDIUM
    return GoalDifficulty.HARD


def visible_group_split(domain: str, visible_tag: str) -> DatasetSplit:
    """Hash a public visible-category group into one unseen-group split."""

    key = f"{VISIBLE_GROUP_SPLIT_RECIPE}|{domain}|{visible_tag}"
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return DatasetSplit.TRAIN
    if bucket < 90:
        return DatasetSplit.DEV
    return DatasetSplit.TEST


def visible_group_id(domain: str, visible_tag: str) -> str:
    """Return a content-addressed public group identifier for split auditing."""

    payload = {
        "group_id_schema_version": VISIBLE_GROUP_ID_SCHEMA_VERSION,
        "domain": domain,
        "visible_tag": visible_tag,
    }
    return f"visible-group-{content_sha256(payload)[:16]}"


def goal_split(relation: CategoryRelation) -> DatasetSplit:
    """Return the split shared by all hidden goals under one visible group."""

    return visible_group_split(relation.domain, relation.visible_tag)


def _stable_take(rows: list[NormalizedCatalogRow], count: int, key: str) -> list[NormalizedCatalogRow]:
    return heapq.nsmallest(
        count,
        rows,
        key=lambda row: (hashlib.sha256(f"{key}|{row.item_id}".encode()).hexdigest(), row.item_id),
    )


def catalog_item_from_row(
    row: NormalizedCatalogRow,
    popularity_by_item: dict[str, float] | None = None,
) -> CatalogItem:
    attributes: dict[str, AttributeValue] = {
        "broad_category": list(row.category_tags),
        "specific_category": list(row.category_tags),
        "category_tags": list(row.category_tags),
    }
    if row.price is not None:
        attributes["price"] = row.price
    if row.rating is not None and row.rating <= 5.0:
        attributes["rating"] = row.rating
    if row.ranking is not None:
        attributes["ranking"] = row.ranking
    if popularity_by_item is not None and row.item_id in popularity_by_item:
        attributes["popularity_percentile"] = popularity_by_item[row.item_id]
    return CatalogItem(item_id=row.item_id, title=row.title, attributes=attributes)


def build_visible_candidate_pool(
    rows: list[NormalizedCatalogRow],
    *,
    domain: str,
    metadata_sha256: str,
    visible_tag: str,
    public_pool_seed: int,
) -> tuple[CatalogIndex, CandidatePool, CandidatePoolRecord]:
    """Build a pool whose definition cannot access any hidden-goal field."""

    visible_rows = [row for row in rows if visible_tag in row.category_tags]
    far_rows = [row for row in rows if visible_tag not in row.category_tags]
    if len(visible_rows) < 90 or len(far_rows) < 10:
        raise ValueError("visible pool family has insufficient catalog support")
    family_key = f"{AUTOMATIC_POOL_RECIPE}|{domain}|{visible_tag}|{public_pool_seed}|{metadata_sha256}"
    selected = [
        *_stable_take(visible_rows, 90, f"{family_key}|visible"),
        *_stable_take(far_rows, 10, f"{family_key}|far"),
    ]
    selected = sorted(
        selected,
        key=lambda row: (hashlib.sha256(f"{family_key}|order|{row.item_id}".encode()).hexdigest(), row.item_id),
    )
    popularity_by_item = popularity_percentiles(visible_rows)
    items = [catalog_item_from_row(row, popularity_by_item) for row in selected]
    recipe_config = {
        "builder_version": AUTOMATIC_BUILDER_VERSION,
        "visible_tag": visible_tag,
        "public_pool_seed": public_pool_seed,
        "visible_count": 90,
        "far_count": 10,
        "popularity_mapping_version": POPULARITY_MAPPING_VERSION,
    }
    index = CatalogIndex(items, domain=domain, source_sha256=metadata_sha256)
    pool = index.pool(recipe_id=AUTOMATIC_POOL_RECIPE, recipe_config=recipe_config)
    record = CandidatePoolRecord(pool_id=pool.pool_id, **pool.identity_payload)
    return index, pool, record


def _domain_noun(domain: str) -> str:
    return {
        "amazon_beauty": "beauty product",
        "amazon_clothing": "clothing item",
        "amazon_music": "music release",
    }[domain]


def fixed_domain_ontology(domain: str) -> DomainSlotOntology:
    """Return one public ontology per domain, independent of hidden family."""

    common = [
        SlotDefinition(name="broad_category", askable=False, priority=10),
        SlotDefinition(name="specific_category", aliases=["subtype"], askable=True, priority=40),
        SlotDefinition(name="price", askable=True, priority=30),
    ]
    if domain == "amazon_music":
        extra = SlotDefinition(name="rating", askable=True, priority=20)
    elif domain in {"amazon_beauty", "amazon_clothing"}:
        extra = SlotDefinition(name="popularity_percentile", askable=True, priority=20)
    else:
        raise KeyError(f"unsupported automatic-goal domain: {domain}")
    return DomainSlotOntology(domain=domain, slots=[*common, extra])


def build_automatic_category_goal(
    relation: CategoryRelation,
    scenario: InteractionScenario,
    rows: list[NormalizedCatalogRow],
    metadata_sha256: str,
    public_pool_seed: int,
    persona: UserPersona | None = None,
) -> BuiltAutomaticGoal:
    split = goal_split(relation)
    difficulty = relation_difficulty(relation)
    index, pool, pool_record = build_visible_candidate_pool(
        rows,
        domain=relation.domain,
        metadata_sha256=metadata_sha256,
        visible_tag=relation.visible_tag,
        public_pool_seed=public_pool_seed,
    )
    labels: list[CandidatePrivateLabel] = []
    acceptable: list[str] = []
    for item_id in pool.item_ids:
        tags = index.get(item_id).attributes["category_tags"]
        assert isinstance(tags, list)
        if relation.visible_tag not in tags:
            label = CandidateGoalLabel.FAR_NEGATIVE
            utility = 0.0
        elif relation.hidden_tag in tags:
            label = CandidateGoalLabel.ACCEPTABLE
            utility = 1.0
            acceptable.append(item_id)
        else:
            label = CandidateGoalLabel.HIDDEN_CATEGORY_NEGATIVE
            utility = 0.1
        labels.append(CandidatePrivateLabel(item_id=item_id, label=label, utility=utility))
    label_counts = Counter(label.label for label in labels)
    if not 5 <= len(acceptable) <= 35:
        raise ValueError("POOL_POSITIVE_OUT_OF_RANGE")
    if label_counts[CandidateGoalLabel.HIDDEN_CATEGORY_NEGATIVE] < 10:
        raise ValueError("MISSING_HIDDEN_CATEGORY_NEGATIVES")
    if label_counts[CandidateGoalLabel.FAR_NEGATIVE] != 10:
        raise ValueError("INVALID_FAR_NEGATIVE_COUNT")
    persona = persona or UserPersona(
        persona_id=f"normal-{relation.relation_id}-{scenario.value}",
        patience_turns=6,
        cooperativeness=1.0,
        acceptance_threshold=0.8,
        question_cost=0.04,
    )
    case_signature = {
        "builder": AUTOMATIC_BUILDER_VERSION,
        "relation_id": relation.relation_id,
        "scenario": scenario.value,
        "pool_id": pool.pool_id,
        "persona_id": persona.persona_id,
    }
    case_id = f"auto-{relation.domain.removeprefix('amazon_')}-{content_sha256(case_signature)[:16]}"
    noun = _domain_noun(relation.domain)
    if scenario == InteractionScenario.READY:
        initial_request = f"I'm looking for a {noun} in {relation.visible_tag}, specifically {relation.hidden_tag}."
    else:
        initial_request = f"I'm looking for a {noun} in {relation.visible_tag}."
    specific_public = scenario == InteractionScenario.READY
    constraints = [
        Constraint(
            slot="broad_category",
            value=relation.visible_tag,
            importance=ConstraintImportance.HARD,
            initially_revealed=True,
        ),
        Constraint(
            slot="specific_category",
            value=relation.hidden_tag,
            importance=ConstraintImportance.HARD,
            initially_revealed=specific_public,
        ),
    ]
    goal = HiddenGoal(
        goal_id=f"goal-{case_id}",
        user_id=f"synthetic-user-{relation.relation_id}",
        domain=relation.domain,
        initial_request=initial_request,
        constraints=constraints,
        target_item_ids=[],
    )
    ontology = fixed_domain_ontology(relation.domain)
    observed: dict[str, AttributeValue] = {"broad_category": relation.visible_tag}
    if specific_public:
        observed["specific_category"] = relation.hidden_tag
    belief = initialize_belief(observed, ontology, candidate_pool=pool, patience_prior=5)
    if scenario == InteractionScenario.REPAIR:
        ranked_items = rank_catalog_from_belief(belief, pool)
        first_slate = [item.item_id for item in ranked_items[:2]]
        if any(item_id in acceptable for item_id in first_slate):
            raise ValueError("REPAIR_PUBLIC_SLATE_DOES_NOT_TRIGGER_REJECTION")

    full_feasible = sorted(
        row.item_id
        for row in rows
        if relation.visible_tag in row.category_tags and relation.hidden_tag in row.category_tags
    )
    public = AutomaticPublicGoal(
        case_id=case_id,
        domain=relation.domain,
        split=split,
        scenario=scenario,
        difficulty=difficulty,
        initial_request=initial_request,
        observed_constraints=observed,
        ontology_id=f"{relation.domain}-category-v1",
        candidate_pool_id=pool.pool_id,
        split_group_id=visible_group_id(relation.domain, relation.visible_tag),
        split_strategy=VISIBLE_GROUP_SPLIT_RECIPE,
    )
    private = AutomaticPrivateGoal(
        case_id=case_id,
        relation_id=relation.relation_id,
        hidden_goal=goal,
        persona=persona,
        visible_tag=relation.visible_tag,
        hidden_tag=relation.hidden_tag,
        full_feasible_count=len(full_feasible),
        full_feasible_set_hash=content_sha256(full_feasible),
        pool_labels=labels,
        pool_acceptable_item_ids=acceptable,
        public_pool_seed=public_pool_seed,
        construction_version=AUTOMATIC_BUILDER_VERSION,
    )
    return BuiltAutomaticGoal(public, private, ontology, belief, index, pool, pool_record)


def _scenario_schedule(count: int) -> list[InteractionScenario]:
    clarify_count = round(count * 0.4)
    ready_count = round(count * 0.3)
    repair_count = count - clarify_count - ready_count
    schedule = (
        [InteractionScenario.CLARIFY] * clarify_count
        + [InteractionScenario.READY] * ready_count
        + [InteractionScenario.REPAIR] * repair_count
    )
    indexed = list(enumerate(schedule))
    indexed.sort(key=lambda pair: hashlib.sha256(f"scenario_schedule|{pair[1].value}|{pair[0]}".encode()).hexdigest())
    return [scenario for _, scenario in indexed]


def select_automatic_category_goals(
    relations: list[CategoryRelation],
    rows: list[NormalizedCatalogRow],
    metadata_sha256: str,
    count: int,
    root_seed: int,
) -> tuple[list[BuiltAutomaticGoal], dict[str, int]]:
    if count < 3:
        raise ValueError("automatic selection requires at least three goals")
    quotas = {
        GoalDifficulty.EASY: count // 3,
        GoalDifficulty.MEDIUM: count // 3,
        GoalDifficulty.HARD: count - 2 * (count // 3),
    }
    scenario_schedule = _scenario_schedule(count)
    selected: list[BuiltAutomaticGoal] = []
    rejection_counts: Counter[str] = Counter()
    visible_usage: Counter[str] = Counter()
    hidden_usage: Counter[str] = Counter()
    for difficulty in (GoalDifficulty.EASY, GoalDifficulty.MEDIUM, GoalDifficulty.HARD):
        candidates = [relation for relation in relations if relation_difficulty(relation) == difficulty]
        candidates.sort(key=lambda relation: hashlib.sha256(f"{root_seed}|{relation.relation_id}".encode()).hexdigest())
        target = quotas[difficulty]
        for relation in candidates:
            if sum(goal.public_goal.difficulty == difficulty for goal in selected) >= target:
                break
            if visible_usage[relation.visible_tag] >= 5:
                rejection_counts["VISIBLE_TAG_CAP"] += 1
                continue
            if hidden_usage[relation.hidden_tag] >= 3:
                rejection_counts["HIDDEN_TAG_CAP"] += 1
                continue
            scenario = scenario_schedule[len(selected)]
            try:
                built = build_automatic_category_goal(
                    relation,
                    scenario,
                    rows,
                    metadata_sha256,
                    public_pool_seed=root_seed,
                )
            except ValueError as exc:
                rejection_counts[str(exc)] += 1
                continue
            selected.append(built)
            visible_usage[relation.visible_tag] += 1
            hidden_usage[relation.hidden_tag] += 1
        if sum(goal.public_goal.difficulty == difficulty for goal in selected) < target:
            raise RuntimeError(f"could not fill {difficulty.value} quota")
    if len(selected) != count:
        raise RuntimeError(f"expected {count} automatic goals, built {len(selected)}")
    return sorted(selected, key=lambda goal: goal.public_goal.case_id), dict(sorted(rejection_counts.items()))
