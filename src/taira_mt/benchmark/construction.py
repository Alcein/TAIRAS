"""Zero-API construction of the first real-user symbolic benchmark pilot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from taira_mt.benchmark.constraint_surface import (
    CATALOG_EQUIVALENT_CONSTRAINT_SURFACE_VERSION,
    LEGACY_CONSTRAINT_SURFACE_POLICY_VERSION,
    catalog_equivalent_constraint_surface,
)
from taira_mt.benchmark.environment import DeterministicEvolvingUserEnvironment
from taira_mt.benchmark.evidence import (
    EvidenceStrength,
    HistoryNumericEvidence,
    SequenceEvidenceProfile,
)
from taira_mt.benchmark.oracle import BENCHMARK_UTILITY_VERSION, IndependentBenchmarkOracle
from taira_mt.benchmark.schemas import (
    EXPLANATION_GUARD_SIMULATOR_VERSION,
    SIMULATOR_VERSION,
    AmbiguityOption,
    BenchmarkCasePair,
    BenchmarkConstraint,
    BenchmarkUserAct,
    CandidateFeasibility,
    CooperationMode,
    DialoguePhaseV1,
    DisclosurePlan,
    EvolutionPlan,
    FeedbackSpecificity,
    GoalVersion,
    InitialQueryFrame,
    IntentCompleteness,
    IntentStability,
    LatentGoalV1,
    OracleSpecV1,
    PrivateBenchmarkCase,
    PrivateInitialUserState,
    PublicBenchmarkCase,
    RuntimeUserState,
    ScenarioFamily,
    SourceEvidence,
    TransitionEvent,
    TransitionTrigger,
    UserBehaviorSpec,
    benchmark_content_sha256,
)
from taira_mt.benchmark.source import (
    BenchmarkSourceAudit,
    DomainSourceAudit,
    EligibleUserSource,
    SourceAuditConfig,
    SourceAuditResult,
    audit_benchmark_sources,
)
from taira_mt.constraints.rendering import render_constraint
from taira_mt.data.goal_construction import (
    CategoryRelation,
    NormalizedCatalogRow,
    discover_category_relations,
    load_normalized_catalog,
)
from taira_mt.data.numeric_goal_specs import popularity_percentiles
from taira_mt.data.relation_review import (
    CategoryRelationReview,
    filter_publishable_relations,
    load_relation_review,
    relation_review_sha256,
)
from taira_mt.data.scalable_goals import catalog_item_from_row
from taira_mt.models import (
    CatalogItem,
    ConstraintImportance,
    ConstraintOperator,
    ConstraintPredicate,
    StrictModel,
)
from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.recommender.catalog import CandidatePool, CatalogIndex

SYMBOLIC_PILOT_BUILDER_VERSION = "user_symbolic_pilot_v1"
USER_POOL_RECIPE_VERSION = "user_visible_60_far_10_v1"
NUMERIC_MISSING_POLICY_VERSION = "numeric_missing_tristate_coverage_v1"
LEGACY_CONFLICT_POLICY_VERSION = "individually_impossible_numeric_v0"
JOINT_CONFLICT_POLICY_VERSION = "catalog_plausible_joint_infeasibility_v1"


class SymbolicPilotConfig(StrictModel):
    builder_version: str = SYMBOLIC_PILOT_BUILDER_VERSION
    public_pool_seed: int = Field(default=20260720, ge=0)
    visible_pool_size: int = Field(default=60, ge=20)
    far_pool_size: int = Field(default=10, ge=5)
    direct_per_domain: int = Field(default=4, ge=0)
    dynamic_per_family_per_domain: int = Field(default=3, ge=1)
    max_cases_per_relation_per_domain: int = Field(default=2, ge=1)
    minimum_acceptable_items: int = Field(default=2, ge=2)
    patience_budget: int = Field(default=6, ge=2, le=20)
    numeric_missing_policy_version: str = NUMERIC_MISSING_POLICY_VERSION
    minimum_target_numeric_coverage: float = Field(default=0.8, ge=0.0, le=1.0)
    maximum_target_numeric_missing: int = Field(default=2, ge=0)
    conflict_policy_version: str = LEGACY_CONFLICT_POLICY_VERSION
    minimum_conflict_numeric_only_items: int = Field(default=2, ge=1)
    minimum_conflict_price: float = Field(default=1.0, gt=0.0)
    minimum_popularity_top_percent: float = Field(default=1.0, gt=0.0, le=100.0)
    multistage_dynamic_goals: bool = False
    post_update_explanation_gate: bool = False
    constraint_surface_policy_version: str = (
        CATALOG_EQUIVALENT_CONSTRAINT_SURFACE_VERSION
    )

    @model_validator(mode="after")
    def validate_version(self) -> SymbolicPilotConfig:
        if self.builder_version != SYMBOLIC_PILOT_BUILDER_VERSION:
            raise ValueError(f"unsupported pilot builder version: {self.builder_version}")
        if self.numeric_missing_policy_version != NUMERIC_MISSING_POLICY_VERSION:
            raise ValueError(
                f"unsupported numeric missing policy: {self.numeric_missing_policy_version}"
            )
        if self.conflict_policy_version not in {
            LEGACY_CONFLICT_POLICY_VERSION,
            JOINT_CONFLICT_POLICY_VERSION,
        }:
            raise ValueError(
                f"unsupported conflict construction policy: {self.conflict_policy_version}"
            )
        if self.constraint_surface_policy_version not in {
            LEGACY_CONSTRAINT_SURFACE_POLICY_VERSION,
            CATALOG_EQUIVALENT_CONSTRAINT_SURFACE_VERSION,
        }:
            raise ValueError(
                "unsupported constraint surface policy: "
                f"{self.constraint_surface_policy_version}"
            )
        return self

    @property
    def per_domain_count(self) -> int:
        return self.direct_per_domain + 4 * self.dynamic_per_family_per_domain

    def quota(self, scenario: ScenarioFamily) -> int:
        if scenario == ScenarioFamily.DIRECT:
            return self.direct_per_domain
        return self.dynamic_per_family_per_domain


class PilotCandidatePoolRecord(StrictModel):
    case_id: str = Field(min_length=1)
    pool_id: str = Field(min_length=64, max_length=64)
    domain: str = Field(min_length=1)
    catalog_source_sha256: str = Field(min_length=64, max_length=64)
    retrieval_recipe_id: str = Field(min_length=1)
    retrieval_recipe_config: dict[str, Any]
    ordered_items: list[CatalogItem] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pool(self) -> PilotCandidatePoolRecord:
        item_ids = [item.item_id for item in self.ordered_items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("serialized candidate pool contains duplicate items")
        index = CatalogIndex(
            self.ordered_items,
            domain=self.domain,
            source_sha256=self.catalog_source_sha256,
        )
        pool = index.pool(
            item_ids,
            recipe_id=self.retrieval_recipe_id,
            recipe_config=self.retrieval_recipe_config,
        )
        if pool.pool_id != self.pool_id:
            raise ValueError("serialized candidate pool identity does not replay")
        return self

    def materialize(self) -> CandidatePool:
        index = CatalogIndex(
            self.ordered_items,
            domain=self.domain,
            source_sha256=self.catalog_source_sha256,
        )
        return index.pool(
            [item.item_id for item in self.ordered_items],
            recipe_id=self.retrieval_recipe_id,
            recipe_config=self.retrieval_recipe_config,
        )


class ConstructionAttempt(StrictModel):
    domain: str
    scenario_family: ScenarioFamily
    pseudonymous_user_id: str
    relation_id: str
    selected: bool
    reason: str
    case_id: str | None = None


class SymbolicPilotSummary(StrictModel):
    build_id: str
    case_count: int = Field(ge=0)
    domain_counts: dict[str, int]
    scenario_counts: dict[str, int]
    domain_scenario_counts: dict[str, int]
    distinct_user_count: int = Field(ge=0)
    distinct_relation_count: int = Field(ge=0)
    construction_failure_counts: dict[str, int]


class PilotArtifactFile(StrictModel):
    relative_path: str
    sha256: str = Field(min_length=64, max_length=64)
    byte_size: int = Field(ge=0)
    row_count: int | None = Field(default=None, ge=0)


class SymbolicPilotManifest(StrictModel):
    schema_version: str = SYMBOLIC_PILOT_BUILDER_VERSION
    artifact_role: str = "zero_api_user_symbolic_pilot"
    build_id: str
    config: SymbolicPilotConfig
    source_audit_id: str = Field(min_length=64, max_length=64)
    relation_review_sha256: str = Field(min_length=64, max_length=64)
    public_cases_sha256: str = Field(min_length=64, max_length=64)
    private_cases_sha256: str = Field(min_length=64, max_length=64)
    candidate_pools_sha256: str = Field(min_length=64, max_length=64)
    files: list[PilotArtifactFile]


@dataclass(frozen=True)
class BuiltSymbolicCase:
    pair: BenchmarkCasePair
    candidate_pool_record: PilotCandidatePoolRecord
    relation_id: str
    pseudonymous_user_id: str
    sequence_evidence_profile_id: str | None = None
    history_numeric_slot: str | None = None
    history_category_reinforced: bool = False
    numeric_plan: _NumericPlan | None = None


@dataclass(frozen=True)
class SymbolicPilotBuildResult:
    build_id: str
    config: SymbolicPilotConfig
    source_audit: BenchmarkSourceAudit
    relation_review_sha256: str
    cases: tuple[BuiltSymbolicCase, ...]
    attempts: tuple[ConstructionAttempt, ...]
    summary: SymbolicPilotSummary


def _stable_key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _stable_take(rows: list[NormalizedCatalogRow], count: int, key: str) -> list[NormalizedCatalogRow]:
    return sorted(rows, key=lambda row: (_stable_key(key, row.item_id), row.item_id))[:count]


def _broad_request_phrase(domain: str, visible_category: str) -> str:
    """Use a domain-safe head noun without guessing the still-hidden subtype."""

    head = {
        "amazon_beauty": "a product",
        "amazon_clothing": "an item",
        "amazon_music": "a release",
    }[domain]
    return f"{head} from the {visible_category} category"


def _display_tag(
    relation: CategoryRelation,
    field: str,
    review: CategoryRelationReview,
) -> str:
    source = getattr(relation, field)
    alias = next(
        (
            item.display_value
            for item in review.display_aliases
            if item.relation_id == relation.relation_id and item.field == field
        ),
        None,
    )
    return alias or source


def _candidate_pool(
    *,
    rows: list[NormalizedCatalogRow],
    domain: str,
    metadata_sha256: str,
    visible_tag: str,
    source: EligibleUserSource,
    config: SymbolicPilotConfig,
) -> CandidatePool:
    excluded = {
        source.anchor_item_id,
        *source.joined_history_item_ids,
        *source.missing_history_item_ids,
    }
    visible_rows = [
        row for row in rows if visible_tag in row.category_tags and row.item_id not in excluded
    ]
    far_rows = [
        row for row in rows if visible_tag not in row.category_tags and row.item_id not in excluded
    ]
    if len(visible_rows) < config.visible_pool_size or len(far_rows) < config.far_pool_size:
        raise ValueError("POOL_SOURCE_CAPACITY")
    case_pool_key = content_sha256(
        {
            "version": USER_POOL_RECIPE_VERSION,
            "source_sequence_sha256": source.source_sequence_sha256,
            "domain": domain,
            "visible_tag": visible_tag,
            "public_pool_seed": config.public_pool_seed,
        }
    )
    selected = [
        *_stable_take(visible_rows, config.visible_pool_size, f"{case_pool_key}|visible"),
        *_stable_take(far_rows, config.far_pool_size, f"{case_pool_key}|far"),
    ]
    selected.sort(key=lambda row: (_stable_key(case_pool_key, "order", row.item_id), row.item_id))
    popularity = popularity_percentiles(visible_rows)
    items = [catalog_item_from_row(row, popularity) for row in selected]
    recipe_config = {
        "builder_version": SYMBOLIC_PILOT_BUILDER_VERSION,
        "visible_tag": visible_tag,
        "public_pool_seed": config.public_pool_seed,
        "case_pool_key": case_pool_key,
        "source_exclusion_sha256": content_sha256(sorted(excluded)),
        "visible_count": config.visible_pool_size,
        "far_count": config.far_pool_size,
    }
    index = CatalogIndex(items, domain=domain, source_sha256=metadata_sha256)
    pool = index.pool(recipe_id=USER_POOL_RECIPE_VERSION, recipe_config=recipe_config)
    if excluded.intersection(pool.item_ids):
        raise AssertionError("source anchor/history leaked into the candidate pool")
    return pool


def _category_constraints(
    relation: CategoryRelation,
    review: CategoryRelationReview,
    *,
    constraint_surface_policy_version: str,
) -> tuple[BenchmarkConstraint, BenchmarkConstraint]:
    visible_display = _display_tag(relation, "visible_tag", review)
    hidden_display = _display_tag(relation, "hidden_tag", review)
    return (
        BenchmarkConstraint(
            constraint_id="c-visible-category",
            slot="broad_category",
            predicate=ConstraintPredicate(
                operator=ConstraintOperator.EQ,
                value=relation.visible_tag,
                importance=ConstraintImportance.HARD,
            ),
            surface_form=f"I am looking within {visible_display}",
            evidence_kind="reviewed_category_relation",
        ),
        BenchmarkConstraint(
            constraint_id="c-specific-category",
            slot="specific_category",
            predicate=ConstraintPredicate(
                operator=ConstraintOperator.EQ,
                value=relation.hidden_tag,
                importance=ConstraintImportance.HARD,
            ),
            surface_form=(
                f"I need {hidden_display}"
                if constraint_surface_policy_version
                == CATALOG_EQUIVALENT_CONSTRAINT_SURFACE_VERSION
                else f"More specifically, I need {hidden_display}"
            ),
            evidence_kind="held_out_anchor_category",
        ),
    )


def _materialize_goal_version(
    version: int,
    constraints: list[BenchmarkConstraint],
    pool: CandidatePool,
) -> GoalVersion:
    provisional = GoalVersion(
        version=version,
        constraints=constraints,
        acceptable_item_ids=[],
        feasible_item_count=0,
        acceptance_threshold=0.8,
    )
    truth = IndependentBenchmarkOracle().recompute_goal_truth(provisional, pool)
    return provisional.model_copy(
        update={
            "feasible_item_count": len(truth.feasible_item_ids),
            "acceptable_item_ids": list(truth.acceptable_item_ids),
        }
    )


@dataclass(frozen=True)
class _NumericPlan:
    slot: str
    operator: ConstraintOperator
    relaxed_value: float
    impossible_value: float
    relaxed_surface: str
    impossible_surface: str
    history_evidence_slot: str | None = None
    target_item_count: int = 0
    observed_item_count: int = 0
    missing_item_count: int = 0
    coverage: float = 0.0
    minimum_required_coverage: float = 0.8
    maximum_allowed_missing: int = 2
    conflict_policy_version: str = LEGACY_CONFLICT_POLICY_VERSION
    conflict_numeric_only_passing_count: int = 0
    conflict_joint_passing_count: int = 0
    relaxed_joint_passing_count: int = 0


def _passes_numeric(value: float, operator: ConstraintOperator, threshold: float) -> bool:
    if operator == ConstraintOperator.LTE:
        return value <= threshold
    if operator == ConstraintOperator.GTE:
        return value >= threshold
    raise ValueError(f"unsupported numeric conflict operator: {operator}")


def _catalog_plausible_joint_conflict_value(
    pool: CandidatePool,
    hidden_tag: str,
    slot: str,
    operator: ConstraintOperator,
    target_observed: list[float],
    *,
    minimum_numeric_only_items: int,
    minimum_conflict_price: float,
    minimum_popularity_top_percent: float,
) -> tuple[float, int]:
    """Choose a real catalog boundary that conflicts only with the target subtype."""

    visible_tag = pool.recipe_config.get("visible_tag")
    if not isinstance(visible_tag, str) or not visible_tag:
        raise ValueError("JOINT_CONFLICT_POOL_LACKS_VISIBLE_TAG")
    visible_values: list[float] = []
    for item in pool.remaining_items():
        tags = item.attributes.get("category_tags") or item.attributes.get("broad_category") or []
        value = item.attributes.get(slot)
        if (
            visible_tag in tags
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            visible_values.append(float(value))
    if not visible_values:
        raise ValueError("JOINT_CONFLICT_HAS_NO_VISIBLE_NUMERIC_EVIDENCE")

    if operator == ConstraintOperator.LTE:
        target_boundary = min(target_observed)
        candidates = sorted(
            {
                value
                for value in visible_values
                if minimum_conflict_price <= value < target_boundary
            },
            reverse=True,
        )
    else:
        target_boundary = max(target_observed)
        upper_bound = 5.0 if slot == "rating" else 1.0
        if slot == "popularity_percentile":
            upper_bound = min(upper_bound, 1.0 - minimum_popularity_top_percent / 100.0)
        candidates = sorted(
            {value for value in visible_values if target_boundary < value <= upper_bound}
        )
    for candidate in candidates:
        numeric_only_count = sum(
            _passes_numeric(value, operator, candidate) for value in visible_values
        )
        if numeric_only_count >= minimum_numeric_only_items:
            return candidate, numeric_only_count
    raise ValueError("NO_CATALOG_PLAUSIBLE_JOINT_CONFLICT_BOUNDARY")


def _numeric_plan(
    pool: CandidatePool,
    hidden_tag: str,
    *,
    require_impossible_price: bool,
    history_evidence: HistoryNumericEvidence | None = None,
    minimum_target_numeric_coverage: float = 0.8,
    maximum_target_numeric_missing: int = 2,
    conflict_policy_version: str = LEGACY_CONFLICT_POLICY_VERSION,
    minimum_conflict_numeric_only_items: int = 2,
    minimum_conflict_price: float = 1.0,
    minimum_popularity_top_percent: float = 1.0,
    constraint_surface_policy_version: str = (
        CATALOG_EQUIVALENT_CONSTRAINT_SURFACE_VERSION
    ),
) -> _NumericPlan:
    positive_items = [
        item
        for item in pool.remaining_items()
        if hidden_tag in (item.attributes.get("specific_category") or [])
    ]
    preferred = (
        [(history_evidence.slot, _operator_for_evidence(history_evidence))]
        if history_evidence is not None
        else [
            ("price", ConstraintOperator.LTE),
            ("rating", ConstraintOperator.GTE),
            ("popularity_percentile", ConstraintOperator.GTE),
        ]
    )
    coverage_failures: list[str] = []
    for slot, operator in preferred:
        observed = sorted(
            float(value)
            for item in positive_items
            if isinstance((value := item.attributes.get(slot)), (int, float))
            and not isinstance(value, bool)
        )
        target_count = len(positive_items)
        observed_count = len(observed)
        missing_count = target_count - observed_count
        coverage = observed_count / target_count if target_count else 0.0
        if (
            coverage < minimum_target_numeric_coverage
            or missing_count > maximum_target_numeric_missing
        ):
            coverage_failures.append(
                f"{slot}:{observed_count}/{target_count}:missing={missing_count}"
            )
            continue
        if len(observed) < 4:
            continue
        if history_evidence is not None:
            relaxed = (
                history_evidence.upper_quartile
                if operator == ConstraintOperator.LTE
                else history_evidence.lower_quartile
            )
            passing = (
                sum(value <= relaxed for value in observed)
                if operator == ConstraintOperator.LTE
                else sum(value >= relaxed for value in observed)
            )
            if not 2 <= passing < len(observed):
                continue
        else:
            thresholds = sorted(set(observed))
            candidates: list[tuple[float, int]] = []
            for threshold in thresholds:
                if operator == ConstraintOperator.LTE:
                    passing = sum(value <= threshold for value in observed)
                else:
                    passing = sum(value >= threshold for value in observed)
                if 2 <= passing < len(observed):
                    candidates.append((threshold, passing))
            if not candidates:
                continue
            relaxed, _ = min(
                candidates,
                key=lambda pair: (abs(pair[1] / len(observed) - 0.5), pair[0]),
            )
        conflict_numeric_only_passing_count = 0
        if require_impossible_price and conflict_policy_version == JOINT_CONFLICT_POLICY_VERSION:
            try:
                impossible, conflict_numeric_only_passing_count = (
                    _catalog_plausible_joint_conflict_value(
                        pool,
                        hidden_tag,
                        slot,
                        operator,
                        observed,
                        minimum_numeric_only_items=minimum_conflict_numeric_only_items,
                        minimum_conflict_price=minimum_conflict_price,
                        minimum_popularity_top_percent=minimum_popularity_top_percent,
                    )
                )
            except ValueError as exc:
                coverage_failures.append(f"{slot}:{exc}")
                continue
        elif operator == ConstraintOperator.LTE:
            impossible = max(0.0, min(observed) - max(1.0, min(observed) * 0.1))
            if require_impossible_price and impossible < 0.0:
                continue
        else:
            impossible = max(observed) + (0.1 if slot == "rating" else 0.01)
        relaxed_predicate = ConstraintPredicate(
            operator=operator,
            value=relaxed,
            importance=ConstraintImportance.HARD,
        )
        impossible_predicate = ConstraintPredicate(
            operator=operator,
            value=impossible,
            importance=ConstraintImportance.HARD,
        )
        if (
            constraint_surface_policy_version
            == CATALOG_EQUIVALENT_CONSTRAINT_SURFACE_VERSION
        ):
            relaxed_surface = catalog_equivalent_constraint_surface(
                pool,
                slot,
                relaxed_predicate,
            )
            impossible_surface = catalog_equivalent_constraint_surface(
                pool,
                slot,
                impossible_predicate,
            )
        else:
            relaxed_surface = render_constraint(slot, relaxed_predicate)
            impossible_surface = render_constraint(slot, impossible_predicate)
        return _NumericPlan(
            slot=slot,
            operator=operator,
            relaxed_value=relaxed,
            impossible_value=impossible,
            relaxed_surface=relaxed_surface,
            impossible_surface=impossible_surface,
            history_evidence_slot=(history_evidence.slot if history_evidence is not None else None),
            target_item_count=target_count,
            observed_item_count=observed_count,
            missing_item_count=missing_count,
            coverage=coverage,
            minimum_required_coverage=minimum_target_numeric_coverage,
            maximum_allowed_missing=maximum_target_numeric_missing,
            conflict_policy_version=conflict_policy_version,
            conflict_numeric_only_passing_count=conflict_numeric_only_passing_count,
            conflict_joint_passing_count=sum(
                _passes_numeric(value, operator, impossible) for value in observed
            ),
            relaxed_joint_passing_count=(
                sum(_passes_numeric(value, operator, relaxed) for value in observed)
            ),
        )
    if coverage_failures:
        raise ValueError(f"NUMERIC_COVERAGE_GATE_FAILED:{','.join(coverage_failures)}")
    if history_evidence is not None:
        raise ValueError("HISTORY_NUMERIC_NOT_NONDEGENERATE")
    raise ValueError("NO_NONDEGENERATE_NUMERIC_PLAN")


def _operator_for_evidence(evidence: HistoryNumericEvidence) -> ConstraintOperator:
    if evidence.slot == "price":
        return ConstraintOperator.LTE
    if evidence.slot in {"rating", "popularity_percentile"}:
        return ConstraintOperator.GTE
    raise ValueError(f"unsupported history evidence slot: {evidence.slot}")


def _history_numeric_plan(
    pool: CandidatePool,
    hidden_tag: str,
    profile: SequenceEvidenceProfile | None,
    *,
    require_impossible: bool,
    minimum_target_numeric_coverage: float,
    maximum_target_numeric_missing: int,
    conflict_policy_version: str,
    minimum_conflict_numeric_only_items: int,
    minimum_conflict_price: float,
    minimum_popularity_top_percent: float,
    constraint_surface_policy_version: str,
) -> _NumericPlan:
    if profile is None:
        raise ValueError("MISSING_SEQUENCE_EVIDENCE_PROFILE")
    reliable = [
        item
        for item in profile.numeric_evidence
        if item.strength in {EvidenceStrength.MODERATE, EvidenceStrength.STRONG}
    ]
    order = {"price": 0, "rating": 1, "popularity_percentile": 2}
    reliable.sort(key=lambda item: (order.get(item.slot, 99), item.slot))
    failures: list[str] = []
    for evidence in reliable:
        try:
            return _numeric_plan(
                pool,
                hidden_tag,
                require_impossible_price=require_impossible,
                history_evidence=evidence,
                minimum_target_numeric_coverage=minimum_target_numeric_coverage,
                maximum_target_numeric_missing=maximum_target_numeric_missing,
                conflict_policy_version=conflict_policy_version,
                minimum_conflict_numeric_only_items=minimum_conflict_numeric_only_items,
                minimum_conflict_price=minimum_conflict_price,
                minimum_popularity_top_percent=minimum_popularity_top_percent,
                constraint_surface_policy_version=constraint_surface_policy_version,
            )
        except ValueError as exc:
            failures.append(f"{evidence.slot}:{exc}")
    if not reliable:
        raise ValueError("NO_RELIABLE_HISTORY_NUMERIC_EVIDENCE")
    raise ValueError(f"NO_USABLE_HISTORY_NUMERIC_EVIDENCE:{','.join(failures)}")


def _numeric_constraint(plan: _NumericPlan, *, impossible: bool) -> BenchmarkConstraint:
    value = plan.impossible_value if impossible else plan.relaxed_value
    surface = plan.impossible_surface if impossible else plan.relaxed_surface
    return BenchmarkConstraint(
        constraint_id="c-tradeoff",
        slot=plan.slot,
        predicate=ConstraintPredicate(
            operator=plan.operator,
            value=value,
            importance=ConstraintImportance.HARD,
        ),
        surface_form=surface,
        evidence_kind=(
            "catalog_plausible_joint_conflict_boundary"
            if impossible and plan.conflict_policy_version == JOINT_CONFLICT_POLICY_VERSION
            else "candidate_positive_numeric_boundary"
        ),
    )


def _initial_factors(
    scenario: ScenarioFamily,
    goal: GoalVersion,
    public_constraint_ids: list[str],
) -> tuple[DialoguePhaseV1, IntentCompleteness, CandidateFeasibility]:
    hard_ids = {
        constraint.constraint_id
        for constraint in goal.constraints
        if constraint.predicate.importance == ConstraintImportance.HARD
    }
    if scenario == ScenarioFamily.AMBIGUOUS:
        completeness = IntentCompleteness.AMBIGUOUS
    elif hard_ids.issubset(public_constraint_ids):
        completeness = IntentCompleteness.SUFFICIENT
    else:
        completeness = IntentCompleteness.INCOMPLETE
    if goal.feasible_item_count == 0:
        feasibility = CandidateFeasibility.EMPTY
        phase = DialoguePhaseV1.NEGOTIATE
    elif goal.feasible_item_count <= 10:
        feasibility = CandidateFeasibility.NARROW
        phase = DialoguePhaseV1.READY if completeness == IntentCompleteness.SUFFICIENT else DialoguePhaseV1.ELICIT
    else:
        feasibility = CandidateFeasibility.BROAD
        phase = DialoguePhaseV1.READY if completeness == IntentCompleteness.SUFFICIENT else DialoguePhaseV1.ELICIT
    return phase, completeness, feasibility


def _initial_query(
    scenario: ScenarioFamily,
    relation: CategoryRelation,
    review: CategoryRelationReview,
    numeric_constraint: BenchmarkConstraint | None,
) -> str:
    visible = _display_tag(relation, "visible_tag", review)
    hidden = _display_tag(relation, "hidden_tag", review)
    if scenario == ScenarioFamily.DIRECT:
        return f"I'm looking for {hidden}."
    if scenario == ScenarioFamily.PROGRESSIVE:
        broad = _broad_request_phrase(relation.domain, visible)
        return f"I'm looking for {broad}, but I could use help narrowing down the exact type."
    if scenario == ScenarioFamily.AMBIGUOUS:
        broad = _broad_request_phrase(relation.domain, visible)
        return f"I need {broad} for a particular use, but several subtypes seem plausible."
    if scenario == ScenarioFamily.REVISION:
        return f"I'm looking for {hidden}."
    if numeric_constraint is None:
        raise ValueError("conflict query requires a numeric constraint")
    return f"I'm looking for {hidden}. {numeric_constraint.surface_form}"


def build_symbolic_case(
    *,
    source: EligibleUserSource,
    source_audit: DomainSourceAudit,
    relation: CategoryRelation,
    scenario: ScenarioFamily,
    rows: list[NormalizedCatalogRow],
    sibling_relations: list[CategoryRelation],
    review: CategoryRelationReview,
    config: SymbolicPilotConfig,
    sequence_evidence_profile: SequenceEvidenceProfile | None = None,
    require_history_numeric: bool = False,
    case_builder_version: str = SYMBOLIC_PILOT_BUILDER_VERSION,
) -> BuiltSymbolicCase:
    if source.domain != relation.domain or source_audit.domain != relation.domain:
        raise ValueError("source/relation/domain audit mismatch")
    anchor = next((row for row in rows if row.item_id == source.anchor_item_id), None)
    if anchor is None:
        raise ValueError("ANCHOR_NOT_IN_NORMALIZED_CATALOG")
    if relation.visible_tag not in anchor.category_tags or relation.hidden_tag not in anchor.category_tags:
        raise ValueError("ANCHOR_DOES_NOT_SUPPORT_RELATION")
    if sequence_evidence_profile is not None:
        if sequence_evidence_profile.pseudonymous_user_id != source.pseudonymous_user_id:
            raise ValueError("SEQUENCE_EVIDENCE_USER_MISMATCH")
        if sequence_evidence_profile.source_sequence_sha256 != source.source_sequence_sha256:
            raise ValueError("SEQUENCE_EVIDENCE_SOURCE_MISMATCH")

    pool = _candidate_pool(
        rows=rows,
        domain=relation.domain,
        metadata_sha256=source_audit.metadata_source.sha256,
        visible_tag=relation.visible_tag,
        source=source,
        config=config,
    )
    visible_constraint, specific_constraint = _category_constraints(
        relation,
        review,
        constraint_surface_policy_version=config.constraint_surface_policy_version,
    )
    category_constraints = [visible_constraint, specific_constraint]
    category_goal = _materialize_goal_version(0, category_constraints, pool)
    if len(category_goal.acceptable_item_ids) < config.minimum_acceptable_items:
        raise ValueError("INSUFFICIENT_CATEGORY_ACCEPTABLE_SET")

    versions: list[GoalVersion]
    public_ids: list[str]
    hidden_order: list[str]
    ambiguity_options: list[AmbiguityOption] = []
    selected_ambiguity: str | None = None
    events: list[TransitionEvent] = []
    numeric_for_query: BenchmarkConstraint | None = None
    plan: _NumericPlan | None = None

    if scenario == ScenarioFamily.DIRECT:
        versions = [category_goal]
        public_ids = ["c-visible-category", "c-specific-category"]
        hidden_order = []
    elif scenario == ScenarioFamily.PROGRESSIVE:
        if config.multistage_dynamic_goals:
            plan = _history_numeric_plan(
                pool,
                relation.hidden_tag,
                sequence_evidence_profile,
                require_impossible=False,
                minimum_target_numeric_coverage=config.minimum_target_numeric_coverage,
                maximum_target_numeric_missing=config.maximum_target_numeric_missing,
                conflict_policy_version=config.conflict_policy_version,
                minimum_conflict_numeric_only_items=config.minimum_conflict_numeric_only_items,
                minimum_conflict_price=config.minimum_conflict_price,
                minimum_popularity_top_percent=config.minimum_popularity_top_percent,
                constraint_surface_policy_version=(
                    config.constraint_surface_policy_version
                ),
            )
            tradeoff = _numeric_constraint(plan, impossible=False)
            category_goal = _materialize_goal_version(
                0,
                [*category_constraints, tradeoff],
                pool,
            )
            if len(category_goal.acceptable_item_ids) < config.minimum_acceptable_items:
                raise ValueError("INSUFFICIENT_MULTISTAGE_ACCEPTABLE_SET")
        versions = [category_goal]
        public_ids = ["c-visible-category"]
        hidden_order = [
            "c-specific-category",
            *(["c-tradeoff"] if config.multistage_dynamic_goals else []),
        ]
    elif scenario == ScenarioFamily.AMBIGUOUS:
        alternatives = [
            candidate
            for candidate in sibling_relations
            if candidate.hidden_tag != relation.hidden_tag
        ]
        if not alternatives:
            raise ValueError("NO_AMBIGUITY_SIBLING")
        alternatives.sort(key=lambda item: _stable_key(source.source_sequence_sha256, item.relation_id))
        selected_ambiguity = f"interpretation-{relation.relation_id}"
        ambiguity_options = [
            AmbiguityOption(
                option_id=selected_ambiguity,
                label=_display_tag(relation, "hidden_tag", review),
                constraint_ids=["c-specific-category"],
            ),
            *[
                AmbiguityOption(
                    option_id=f"interpretation-{candidate.relation_id}",
                    label=_display_tag(candidate, "hidden_tag", review),
                )
                for candidate in alternatives[:2]
            ],
        ]
        if config.multistage_dynamic_goals:
            plan = _history_numeric_plan(
                pool,
                relation.hidden_tag,
                sequence_evidence_profile,
                require_impossible=False,
                minimum_target_numeric_coverage=config.minimum_target_numeric_coverage,
                maximum_target_numeric_missing=config.maximum_target_numeric_missing,
                conflict_policy_version=config.conflict_policy_version,
                minimum_conflict_numeric_only_items=config.minimum_conflict_numeric_only_items,
                minimum_conflict_price=config.minimum_conflict_price,
                minimum_popularity_top_percent=config.minimum_popularity_top_percent,
                constraint_surface_policy_version=(
                    config.constraint_surface_policy_version
                ),
            )
            tradeoff = _numeric_constraint(plan, impossible=False)
            category_goal = _materialize_goal_version(
                0,
                [*category_constraints, tradeoff],
                pool,
            )
            if len(category_goal.acceptable_item_ids) < config.minimum_acceptable_items:
                raise ValueError("INSUFFICIENT_MULTISTAGE_ACCEPTABLE_SET")
        versions = [category_goal]
        public_ids = ["c-visible-category"]
        hidden_order = [
            "c-specific-category",
            *(["c-tradeoff"] if config.multistage_dynamic_goals else []),
        ]
        events = [
            TransitionEvent(
                event_id="event-resolve-occasion",
                trigger=TransitionTrigger.ASK_INTERPRETATION,
                trigger_ref="occasion_interpretation",
                from_goal_version=0,
                to_goal_version=0,
                reveal_constraint_ids=["c-specific-category"],
                response_act=BenchmarkUserAct.CLARIFY,
            )
        ]
    elif scenario == ScenarioFamily.REVISION:
        plan = (
            _history_numeric_plan(
                pool,
                relation.hidden_tag,
                sequence_evidence_profile,
                require_impossible=False,
                minimum_target_numeric_coverage=config.minimum_target_numeric_coverage,
                maximum_target_numeric_missing=config.maximum_target_numeric_missing,
                conflict_policy_version=config.conflict_policy_version,
                minimum_conflict_numeric_only_items=config.minimum_conflict_numeric_only_items,
                minimum_conflict_price=config.minimum_conflict_price,
                minimum_popularity_top_percent=config.minimum_popularity_top_percent,
                constraint_surface_policy_version=(
                    config.constraint_surface_policy_version
                ),
            )
            if require_history_numeric
            else _numeric_plan(
                pool,
                relation.hidden_tag,
                require_impossible_price=False,
                minimum_target_numeric_coverage=config.minimum_target_numeric_coverage,
                maximum_target_numeric_missing=config.maximum_target_numeric_missing,
                conflict_policy_version=config.conflict_policy_version,
                minimum_conflict_numeric_only_items=config.minimum_conflict_numeric_only_items,
                minimum_conflict_price=config.minimum_conflict_price,
                minimum_popularity_top_percent=config.minimum_popularity_top_percent,
                constraint_surface_policy_version=(
                    config.constraint_surface_policy_version
                ),
            )
        )
        tightened = _numeric_constraint(plan, impossible=False)
        revised_goal = _materialize_goal_version(1, [*category_constraints, tightened], pool)
        if len(revised_goal.acceptable_item_ids) < config.minimum_acceptable_items:
            raise ValueError("INSUFFICIENT_REVISED_ACCEPTABLE_SET")
        if set(revised_goal.acceptable_item_ids) == set(category_goal.acceptable_item_ids):
            raise ValueError("REVISION_DOES_NOT_CHANGE_ACCEPTABLE_SET")
        versions = [category_goal, revised_goal]
        public_ids = ["c-visible-category", "c-specific-category"]
        hidden_order = []
        events = [
            TransitionEvent(
                event_id="event-tighten-after-good-slate",
                trigger=TransitionTrigger.SATISFACTORY_RECOMMENDATION,
                from_goal_version=0,
                to_goal_version=1,
                reveal_constraint_ids=["c-tradeoff"],
                target_phase=DialoguePhaseV1.REPAIR,
                resulting_stability=IntentStability.TIGHTENED,
                response_act=BenchmarkUserAct.CLARIFY,
            ),
            TransitionEvent(
                event_id="event-tighten-after-bad-slate",
                trigger=TransitionTrigger.UNSATISFACTORY_RECOMMENDATION,
                from_goal_version=0,
                to_goal_version=1,
                reveal_constraint_ids=["c-tradeoff"],
                target_phase=DialoguePhaseV1.REPAIR,
                resulting_stability=IntentStability.TIGHTENED,
                response_act=BenchmarkUserAct.REJECT,
            ),
        ]
        if config.post_update_explanation_gate:
            events.append(
                TransitionEvent(
                    event_id="event-explain-after-revision",
                    trigger=TransitionTrigger.SATISFACTORY_RECOMMENDATION,
                    from_goal_version=1,
                    to_goal_version=1,
                    target_phase=DialoguePhaseV1.FEEDBACK,
                    response_act=BenchmarkUserAct.REQUEST_EXPLANATION,
                )
            )
    elif scenario == ScenarioFamily.CONFLICT:
        plan = (
            _history_numeric_plan(
                pool,
                relation.hidden_tag,
                sequence_evidence_profile,
                require_impossible=True,
                minimum_target_numeric_coverage=config.minimum_target_numeric_coverage,
                maximum_target_numeric_missing=config.maximum_target_numeric_missing,
                conflict_policy_version=config.conflict_policy_version,
                minimum_conflict_numeric_only_items=config.minimum_conflict_numeric_only_items,
                minimum_conflict_price=config.minimum_conflict_price,
                minimum_popularity_top_percent=config.minimum_popularity_top_percent,
                constraint_surface_policy_version=(
                    config.constraint_surface_policy_version
                ),
            )
            if require_history_numeric
            else _numeric_plan(
                pool,
                relation.hidden_tag,
                require_impossible_price=True,
                minimum_target_numeric_coverage=config.minimum_target_numeric_coverage,
                maximum_target_numeric_missing=config.maximum_target_numeric_missing,
                conflict_policy_version=config.conflict_policy_version,
                minimum_conflict_numeric_only_items=config.minimum_conflict_numeric_only_items,
                minimum_conflict_price=config.minimum_conflict_price,
                minimum_popularity_top_percent=config.minimum_popularity_top_percent,
                constraint_surface_policy_version=(
                    config.constraint_surface_policy_version
                ),
            )
        )
        impossible = _numeric_constraint(plan, impossible=True)
        relaxed = _numeric_constraint(plan, impossible=False)
        conflict_goal = _materialize_goal_version(0, [*category_constraints, impossible], pool)
        relaxed_goal = _materialize_goal_version(1, [*category_constraints, relaxed], pool)
        if conflict_goal.feasible_item_count != 0:
            raise ValueError("CONFLICT_GOAL_IS_NOT_EMPTY")
        if config.conflict_policy_version == JOINT_CONFLICT_POLICY_VERSION:
            if (
                plan.conflict_numeric_only_passing_count
                < config.minimum_conflict_numeric_only_items
            ):
                raise ValueError("CONFLICT_NUMERIC_REQUIREMENT_IS_NOT_CATALOG_PLAUSIBLE")
            if plan.conflict_joint_passing_count != 0:
                raise ValueError("CONFLICT_TARGET_SUBTYPE_STILL_PASSES_NUMERIC_BOUNDARY")
        if len(relaxed_goal.acceptable_item_ids) < config.minimum_acceptable_items:
            raise ValueError("RELAXATION_DOES_NOT_RECOVER_CANDIDATES")
        versions = [conflict_goal, relaxed_goal]
        public_ids = ["c-visible-category", "c-specific-category", "c-tradeoff"]
        hidden_order = []
        numeric_for_query = impossible
        events = [
            TransitionEvent(
                event_id="event-disclose-feasible-tradeoff",
                trigger=TransitionTrigger.ASK_CONSTRAINT,
                trigger_ref="c-tradeoff",
                from_goal_version=0,
                to_goal_version=1,
                reveal_constraint_ids=["c-tradeoff"],
                target_phase=DialoguePhaseV1.READY,
                resulting_stability=IntentStability.RELAXED,
                response_act=BenchmarkUserAct.INFORM,
            ),
            TransitionEvent(
                event_id="event-accept-tradeoff-relaxation",
                trigger=TransitionTrigger.ALLOWED_RELAXATION,
                trigger_ref="c-tradeoff",
                from_goal_version=0,
                to_goal_version=1,
                reveal_constraint_ids=["c-tradeoff"],
                target_phase=DialoguePhaseV1.READY,
                resulting_stability=IntentStability.RELAXED,
                response_act=BenchmarkUserAct.AGREE_RELAXATION,
            )
        ]
        if config.post_update_explanation_gate:
            events.append(
                TransitionEvent(
                    event_id="event-explain-after-relaxation",
                    trigger=TransitionTrigger.SATISFACTORY_RECOMMENDATION,
                    from_goal_version=1,
                    to_goal_version=1,
                    target_phase=DialoguePhaseV1.FEEDBACK,
                    response_act=BenchmarkUserAct.REQUEST_EXPLANATION,
                )
            )
    else:  # pragma: no cover - exhaustive enum guard
        raise AssertionError(scenario)

    initial_goal = versions[0]
    phase, completeness, feasibility = _initial_factors(scenario, initial_goal, public_ids)
    query = _initial_query(scenario, relation, review, numeric_for_query)
    case_signature = {
        "builder_version": case_builder_version,
        "source_sequence_sha256": source.source_sequence_sha256,
        "relation_id": relation.relation_id,
        "scenario": scenario.value,
        "pool_id": pool.pool_id,
        "utility_version": BENCHMARK_UTILITY_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "numeric_missing_policy_version": config.numeric_missing_policy_version,
    }
    case_id = f"user-case-{content_sha256(case_signature)[:24]}"
    style_signature = {
        **case_signature,
        "simulator_version": EXPLANATION_GUARD_SIMULATOR_VERSION,
    }
    style_case_id = f"user-case-{content_sha256(style_signature)[:24]}"
    style_id = ["concise", "conversational", "direct", "deliberative"][
        int(_stable_key(style_case_id, "style")[:8], 16) % 4
    ]
    public = PublicBenchmarkCase(
        case_id=case_id,
        domain=relation.domain,
        initial_query=query,
        candidate_pool_id=pool.pool_id,
        split_group_id=f"relation-{relation.relation_id}",
    )
    initial_state = PrivateInitialUserState(
        scenario_family=scenario,
        latent_goal=LatentGoalV1(
            target_type=_display_tag(relation, "hidden_tag", review),
            versions=versions,
        ),
        disclosure_plan=DisclosurePlan(
            initially_public=public_ids,
            hidden_order=hidden_order,
            ambiguity_options=ambiguity_options,
            selected_ambiguity_option_id=selected_ambiguity,
        ),
        evolution_plan=EvolutionPlan(events=events),
        behavior=UserBehaviorSpec(
            patience_budget=config.patience_budget,
            cooperation_mode=(
                CooperationMode.DIRECT
                if scenario in {ScenarioFamily.DIRECT, ScenarioFamily.REVISION, ScenarioFamily.CONFLICT}
                else CooperationMode.PROGRESSIVE
            ),
            feedback_specificity=FeedbackSpecificity.CONSTRAINT,
            language_style_id=style_id,
        ),
        runtime_init=RuntimeUserState(
            phase=phase,
            completeness=completeness,
            feasibility=feasibility,
            disclosed_constraint_ids=public_ids,
            patience_remaining=config.patience_budget,
        ),
        oracle_spec=OracleSpecV1(
            utility_version=BENCHMARK_UTILITY_VERSION,
            candidate_pool_id=pool.pool_id,
        ),
    )
    private = PrivateBenchmarkCase(
        case_id=case_id,
        public_case_sha256=benchmark_content_sha256(public),
        source_evidence=SourceEvidence(
            pseudonymous_user_id=source.pseudonymous_user_id,
            source_sequence_sha256=source.source_sequence_sha256,
            anchor_item_id=source.anchor_item_id,
            joined_history_item_ids=source.joined_history_item_ids,
            missing_history_item_ids=source.missing_history_item_ids,
            metadata_source_sha256=source_audit.metadata_source.sha256,
            user_source_sha256=source_audit.user_source.sha256,
        ),
        query_frame=InitialQueryFrame(
            intent_phrase=f"find {_display_tag(relation, 'hidden_tag', review)}",
            public_constraint_ids=public_ids,
            style_id=style_id,
            renderer_version="canonical_template_v1",
        ),
        initial_state=initial_state,
    )
    pair = BenchmarkCasePair(public=public, private=private)
    pool_record = PilotCandidatePoolRecord(
        case_id=case_id,
        pool_id=pool.pool_id,
        domain=relation.domain,
        catalog_source_sha256=pool.index.source_sha256,
        retrieval_recipe_id=pool.recipe_id,
        retrieval_recipe_config=pool.recipe_config,
        ordered_items=[pool.index.get(item_id) for item_id in pool.item_ids],
    )
    DeterministicEvolvingUserEnvironment(pair, pool_record.materialize())
    return BuiltSymbolicCase(
        pair=pair,
        candidate_pool_record=pool_record,
        relation_id=relation.relation_id,
        pseudonymous_user_id=source.pseudonymous_user_id,
        sequence_evidence_profile_id=(
            sequence_evidence_profile.profile_id
            if sequence_evidence_profile is not None
            else None
        ),
        history_numeric_slot=(
            plan.history_evidence_slot if plan is not None else None
        ),
        history_category_reinforced=(
            relation.hidden_tag
            in set(
                sequence_evidence_profile.anchor_reinforced_category_tags
                if sequence_evidence_profile is not None
                else []
            )
        ),
        numeric_plan=plan,
    )


def _scenario_order() -> tuple[ScenarioFamily, ...]:
    return (
        ScenarioFamily.CONFLICT,
        ScenarioFamily.REVISION,
        ScenarioFamily.AMBIGUOUS,
        ScenarioFamily.PROGRESSIVE,
        ScenarioFamily.DIRECT,
    )


def build_symbolic_pilot(
    *,
    source_root: Path,
    review_path: Path,
    config: SymbolicPilotConfig | None = None,
) -> SymbolicPilotBuildResult:
    config = config or SymbolicPilotConfig()
    source_root = source_root.resolve()
    review = load_relation_review(review_path)
    review_sha256 = relation_review_sha256(review)
    source_result: SourceAuditResult = audit_benchmark_sources(source_root, SourceAuditConfig())
    audit_by_domain = {domain.domain: domain for domain in source_result.audit.domains}
    users_by_domain: dict[str, list[EligibleUserSource]] = defaultdict(list)
    for source in source_result.eligible_users:
        users_by_domain[source.domain].append(source)

    built: list[BuiltSymbolicCase] = []
    attempts: list[ConstructionAttempt] = []
    selected_users: set[str] = set()
    relation_counts: Counter[tuple[str, str]] = Counter()

    for domain in sorted(users_by_domain):
        metadata_path = source_root / domain / "metadata.csv"
        rows, _ = load_normalized_catalog(metadata_path)
        row_by_id = {row.item_id: row for row in rows}
        discovered = discover_category_relations(
            rows,
            domain,
            metadata_path,
            source_root,
        )
        relations = filter_publishable_relations(discovered.relations, review)
        relations_by_visible: dict[str, list[CategoryRelation]] = defaultdict(list)
        for relation in relations:
            relations_by_visible[relation.visible_tag].append(relation)

        user_relations: dict[str, list[CategoryRelation]] = {}
        for source in users_by_domain[domain]:
            anchor = row_by_id.get(source.anchor_item_id)
            if anchor is None:
                continue
            tags = set(anchor.category_tags)
            matches = [
                relation
                for relation in relations
                if relation.visible_tag in tags and relation.hidden_tag in tags
            ]
            matches.sort(key=lambda item: _stable_key(source.source_sequence_sha256, item.relation_id))
            if matches:
                user_relations[source.pseudonymous_user_id] = matches

        for scenario in _scenario_order():
            quota = config.quota(scenario)
            scenario_selected = 0
            ordered_users = sorted(
                users_by_domain[domain],
                key=lambda source: _stable_key(
                    config.public_pool_seed,
                    domain,
                    scenario.value,
                    source.source_sequence_sha256,
                ),
            )
            for source in ordered_users:
                if scenario_selected >= quota:
                    break
                if source.pseudonymous_user_id in selected_users:
                    continue
                matching = user_relations.get(source.pseudonymous_user_id, [])
                for relation in matching:
                    count_key = (domain, relation.relation_id)
                    if relation_counts[count_key] >= config.max_cases_per_relation_per_domain:
                        continue
                    try:
                        case = build_symbolic_case(
                            source=source,
                            source_audit=audit_by_domain[domain],
                            relation=relation,
                            scenario=scenario,
                            rows=rows,
                            sibling_relations=relations_by_visible[relation.visible_tag],
                            review=review,
                            config=config,
                        )
                    except (AssertionError, KeyError, ValueError) as exc:
                        attempts.append(
                            ConstructionAttempt(
                                domain=domain,
                                scenario_family=scenario,
                                pseudonymous_user_id=source.pseudonymous_user_id,
                                relation_id=relation.relation_id,
                                selected=False,
                                reason=str(exc),
                            )
                        )
                        continue
                    built.append(case)
                    selected_users.add(source.pseudonymous_user_id)
                    relation_counts[count_key] += 1
                    scenario_selected += 1
                    attempts.append(
                        ConstructionAttempt(
                            domain=domain,
                            scenario_family=scenario,
                            pseudonymous_user_id=source.pseudonymous_user_id,
                            relation_id=relation.relation_id,
                            selected=True,
                            reason="SELECTED",
                            case_id=case.pair.public.case_id,
                        )
                    )
                    break
            if scenario_selected != quota:
                raise RuntimeError(
                    f"PILOT_QUOTA_SHORTFALL:{domain}:{scenario.value}:{scenario_selected}/{quota}"
                )

    built.sort(key=lambda item: item.pair.public.case_id)
    public_payload = [item.pair.public.model_dump(mode="json") for item in built]
    private_payload = [item.pair.private.model_dump(mode="json") for item in built]
    pool_payload = [item.candidate_pool_record.model_dump(mode="json") for item in built]
    identity = {
        "builder_version": config.builder_version,
        "config": config.model_dump(mode="json"),
        "source_audit_id": source_result.audit.audit_id,
        "relation_review_sha256": review_sha256,
        "public_cases_sha256": content_sha256(public_payload),
        "private_cases_sha256": content_sha256(private_payload),
        "candidate_pools_sha256": content_sha256(pool_payload),
    }
    build_id = f"symbolic-pilot-{content_sha256(identity)[:16]}"
    failure_counts = Counter(attempt.reason for attempt in attempts if not attempt.selected)
    summary = SymbolicPilotSummary(
        build_id=build_id,
        case_count=len(built),
        domain_counts=dict(sorted(Counter(item.pair.public.domain for item in built).items())),
        scenario_counts=dict(
            sorted(
                Counter(
                    item.pair.private.initial_state.scenario_family.value
                    for item in built
                ).items()
            )
        ),
        domain_scenario_counts=dict(
            sorted(
                Counter(
                    f"{item.pair.public.domain}:{item.pair.private.initial_state.scenario_family.value}"
                    for item in built
                ).items()
            )
        ),
        distinct_user_count=len({item.pseudonymous_user_id for item in built}),
        distinct_relation_count=len({item.relation_id for item in built}),
        construction_failure_counts=dict(sorted(failure_counts.items())),
    )
    return SymbolicPilotBuildResult(
        build_id=build_id,
        config=config,
        source_audit=source_result.audit,
        relation_review_sha256=review_sha256,
        cases=tuple(built),
        attempts=tuple(attempts),
        summary=summary,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_idempotently(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different artifact file: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(canonical_json(record) + "\n" for record in records)


def write_symbolic_pilot_artifact(
    result: SymbolicPilotBuildResult,
    output_root: Path,
) -> Path:
    run_dir = output_root / result.build_id
    run_dir.mkdir(parents=True, exist_ok=True)
    public_records = [item.pair.public.model_dump(mode="json") for item in result.cases]
    private_records = [item.pair.private.model_dump(mode="json") for item in result.cases]
    pool_records = [item.candidate_pool_record.model_dump(mode="json") for item in result.cases]
    attempt_records = [item.model_dump(mode="json") for item in result.attempts]
    outputs = {
        "cases.public.jsonl": _jsonl(public_records),
        "cases.private.jsonl": _jsonl(private_records),
        "candidate_pools.public.jsonl": _jsonl(pool_records),
        "construction_ledger.private.jsonl": _jsonl(attempt_records),
        "summary.json": json.dumps(result.summary.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        "source_audit.json": json.dumps(
            result.source_audit.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    }
    for filename, text in outputs.items():
        _write_idempotently(run_dir / filename, text)

    files = [
        PilotArtifactFile(
            relative_path=filename,
            sha256=_file_sha256(run_dir / filename),
            byte_size=(run_dir / filename).stat().st_size,
            row_count=(text.count("\n") if filename.endswith(".jsonl") else None),
        )
        for filename, text in sorted(outputs.items())
    ]
    manifest = SymbolicPilotManifest(
        build_id=result.build_id,
        config=result.config,
        source_audit_id=result.source_audit.audit_id,
        relation_review_sha256=result.relation_review_sha256,
        public_cases_sha256=content_sha256(public_records),
        private_cases_sha256=content_sha256(private_records),
        candidate_pools_sha256=content_sha256(pool_records),
        files=files,
    )
    _write_idempotently(
        run_dir / "manifest.json",
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
    )
    return run_dir
