"""History-grounded symbolic development benchmark for the evolving user simulator."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from taira_mt.benchmark.constraint_surface import (
    CATALOG_EQUIVALENT_CONSTRAINT_SURFACE_VERSION,
    LEGACY_CONSTRAINT_SURFACE_POLICY_VERSION,
)
from taira_mt.benchmark.construction import (
    JOINT_CONFLICT_POLICY_VERSION,
    LEGACY_CONFLICT_POLICY_VERSION,
    NUMERIC_MISSING_POLICY_VERSION,
    BuiltSymbolicCase,
    ConstructionAttempt,
    PilotArtifactFile,
    SymbolicPilotConfig,
    _scenario_order,
    _stable_key,
    build_symbolic_case,
)
from taira_mt.benchmark.evidence import (
    EvidenceStrength,
    SequenceEvidenceManifest,
    SequenceEvidenceProfile,
    load_sequence_evidence_artifact,
)
from taira_mt.benchmark.schemas import ScenarioFamily
from taira_mt.benchmark.source import (
    BenchmarkSourceAudit,
    EligibleUserSource,
    SourceAuditConfig,
    audit_benchmark_sources,
)
from taira_mt.data.goal_construction import (
    CategoryRelation,
    discover_category_relations,
    load_normalized_catalog,
)
from taira_mt.data.relation_review import (
    filter_publishable_relations,
    load_relation_review,
    relation_review_sha256,
)
from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json, content_sha256

LEGACY_DEVELOPMENT_BENCHMARK_VERSION = "user_development_benchmark_v1"
JOINT_CONFLICT_DEVELOPMENT_BENCHMARK_VERSION = "user_development_benchmark_v2"
MULTISTAGE_DEVELOPMENT_BENCHMARK_VERSION = (
    "user_development_benchmark_v3_multistage"
)
CATALOG_SURFACE_DEVELOPMENT_BENCHMARK_VERSION = (
    "user_development_benchmark_v4_catalog_equivalent_surface"
)
DEVELOPMENT_BENCHMARK_VERSION = (
    "user_development_benchmark_v5_preference_grounded_explanation"
)
LEGACY_DEVELOPMENT_CASE_IDENTITY_VERSION = LEGACY_DEVELOPMENT_BENCHMARK_VERSION
MULTISTAGE_DEVELOPMENT_CASE_IDENTITY_VERSION = (
    MULTISTAGE_DEVELOPMENT_BENCHMARK_VERSION
)
# Surface-policy revisions preserve the semantic user/scenario identity.  Full
# artifact hashes still change, while case-level splits and language styles stay
# stable across lexical releases.
DEVELOPMENT_CASE_IDENTITY_VERSION = MULTISTAGE_DEVELOPMENT_CASE_IDENTITY_VERSION


class DevelopmentBenchmarkConfig(StrictModel):
    builder_version: str = DEVELOPMENT_BENCHMARK_VERSION
    public_pool_seed: int = Field(default=20260722, ge=0)
    visible_pool_size: int = Field(default=60, ge=20)
    far_pool_size: int = Field(default=10, ge=5)
    direct_per_domain: int = Field(default=16, ge=1)
    dynamic_per_family_per_domain: int = Field(default=16, ge=1)
    max_cases_per_relation_per_domain: int = Field(default=3, ge=1)
    minimum_acceptable_items: int = Field(default=2, ge=2)
    patience_budget: int = Field(default=6, ge=2, le=20)
    require_reliable_history_numeric_for_dynamic_tradeoff: bool = True
    numeric_missing_policy_version: str = NUMERIC_MISSING_POLICY_VERSION
    minimum_target_numeric_coverage: float = Field(default=0.8, ge=0.0, le=1.0)
    maximum_target_numeric_missing: int = Field(default=2, ge=0)
    conflict_policy_version: str = JOINT_CONFLICT_POLICY_VERSION
    minimum_conflict_numeric_only_items: int = Field(default=2, ge=1)
    minimum_conflict_price: float = Field(default=1.0, gt=0.0)
    minimum_popularity_top_percent: float = Field(default=1.0, gt=0.0, le=100.0)
    case_identity_version: str = DEVELOPMENT_CASE_IDENTITY_VERSION
    multistage_dynamic_goals: bool = True
    post_update_explanation_gate: bool = True
    constraint_surface_policy_version: str = (
        CATALOG_EQUIVALENT_CONSTRAINT_SURFACE_VERSION
    )

    @model_validator(mode="before")
    @classmethod
    def fill_legacy_policy(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        builder_version = payload.get("builder_version")
        if builder_version == LEGACY_DEVELOPMENT_BENCHMARK_VERSION:
            payload.setdefault("conflict_policy_version", LEGACY_CONFLICT_POLICY_VERSION)
        if builder_version in {
            LEGACY_DEVELOPMENT_BENCHMARK_VERSION,
            JOINT_CONFLICT_DEVELOPMENT_BENCHMARK_VERSION,
        }:
            payload.setdefault(
                "case_identity_version", LEGACY_DEVELOPMENT_CASE_IDENTITY_VERSION
            )
            payload.setdefault("multistage_dynamic_goals", False)
            payload.setdefault("post_update_explanation_gate", False)
            payload.setdefault(
                "constraint_surface_policy_version",
                LEGACY_CONSTRAINT_SURFACE_POLICY_VERSION,
            )
        elif builder_version == MULTISTAGE_DEVELOPMENT_BENCHMARK_VERSION:
            payload.setdefault(
                "case_identity_version",
                MULTISTAGE_DEVELOPMENT_CASE_IDENTITY_VERSION,
            )
            payload.setdefault("multistage_dynamic_goals", True)
            payload.setdefault("post_update_explanation_gate", True)
            payload.setdefault(
                "constraint_surface_policy_version",
                LEGACY_CONSTRAINT_SURFACE_POLICY_VERSION,
            )
        elif builder_version == CATALOG_SURFACE_DEVELOPMENT_BENCHMARK_VERSION:
            payload.setdefault(
                "case_identity_version",
                DEVELOPMENT_CASE_IDENTITY_VERSION,
            )
            payload.setdefault("multistage_dynamic_goals", True)
            payload.setdefault("post_update_explanation_gate", True)
            payload.setdefault(
                "constraint_surface_policy_version",
                CATALOG_EQUIVALENT_CONSTRAINT_SURFACE_VERSION,
            )
        return payload

    @model_validator(mode="after")
    def validate_version(self) -> DevelopmentBenchmarkConfig:
        if self.builder_version not in {
            LEGACY_DEVELOPMENT_BENCHMARK_VERSION,
            JOINT_CONFLICT_DEVELOPMENT_BENCHMARK_VERSION,
            MULTISTAGE_DEVELOPMENT_BENCHMARK_VERSION,
            CATALOG_SURFACE_DEVELOPMENT_BENCHMARK_VERSION,
            DEVELOPMENT_BENCHMARK_VERSION,
        }:
            raise ValueError(f"unsupported development benchmark version: {self.builder_version}")
        if self.numeric_missing_policy_version != NUMERIC_MISSING_POLICY_VERSION:
            raise ValueError(
                f"unsupported numeric missing policy: {self.numeric_missing_policy_version}"
            )
        expected_conflict_policy = (
            LEGACY_CONFLICT_POLICY_VERSION
            if self.builder_version == LEGACY_DEVELOPMENT_BENCHMARK_VERSION
            else JOINT_CONFLICT_POLICY_VERSION
        )
        if self.conflict_policy_version != expected_conflict_policy:
            raise ValueError(
                "development benchmark version and conflict policy version differ"
            )
        expected_identity = {
            LEGACY_DEVELOPMENT_BENCHMARK_VERSION: (
                LEGACY_DEVELOPMENT_CASE_IDENTITY_VERSION
            ),
            JOINT_CONFLICT_DEVELOPMENT_BENCHMARK_VERSION: (
                LEGACY_DEVELOPMENT_CASE_IDENTITY_VERSION
            ),
            MULTISTAGE_DEVELOPMENT_BENCHMARK_VERSION: (
                MULTISTAGE_DEVELOPMENT_CASE_IDENTITY_VERSION
            ),
            CATALOG_SURFACE_DEVELOPMENT_BENCHMARK_VERSION: (
                DEVELOPMENT_CASE_IDENTITY_VERSION
            ),
            DEVELOPMENT_BENCHMARK_VERSION: DEVELOPMENT_CASE_IDENTITY_VERSION,
        }[self.builder_version]
        if self.case_identity_version != expected_identity:
            raise ValueError("development builder and case identity versions differ")
        expected_multistage = self.builder_version in {
            MULTISTAGE_DEVELOPMENT_BENCHMARK_VERSION,
            CATALOG_SURFACE_DEVELOPMENT_BENCHMARK_VERSION,
            DEVELOPMENT_BENCHMARK_VERSION,
        }
        if self.multistage_dynamic_goals != expected_multistage:
            raise ValueError("development builder and multistage-goal policy differ")
        if self.post_update_explanation_gate != expected_multistage:
            raise ValueError("development builder and explanation-gate policy differ")
        expected_surface_policy = (
            CATALOG_EQUIVALENT_CONSTRAINT_SURFACE_VERSION
            if self.builder_version
            in {
                CATALOG_SURFACE_DEVELOPMENT_BENCHMARK_VERSION,
                DEVELOPMENT_BENCHMARK_VERSION,
            }
            else LEGACY_CONSTRAINT_SURFACE_POLICY_VERSION
        )
        if self.constraint_surface_policy_version != expected_surface_policy:
            raise ValueError("development builder and constraint-surface policy differ")
        return self

    @property
    def per_domain_count(self) -> int:
        return self.direct_per_domain + 4 * self.dynamic_per_family_per_domain

    def quota(self, scenario: ScenarioFamily) -> int:
        if scenario == ScenarioFamily.DIRECT:
            return self.direct_per_domain
        return self.dynamic_per_family_per_domain

    def pilot_config(self) -> SymbolicPilotConfig:
        return SymbolicPilotConfig(
            public_pool_seed=self.public_pool_seed,
            visible_pool_size=self.visible_pool_size,
            far_pool_size=self.far_pool_size,
            direct_per_domain=self.direct_per_domain,
            dynamic_per_family_per_domain=self.dynamic_per_family_per_domain,
            max_cases_per_relation_per_domain=self.max_cases_per_relation_per_domain,
            minimum_acceptable_items=self.minimum_acceptable_items,
            patience_budget=self.patience_budget,
            numeric_missing_policy_version=self.numeric_missing_policy_version,
            minimum_target_numeric_coverage=self.minimum_target_numeric_coverage,
            maximum_target_numeric_missing=self.maximum_target_numeric_missing,
            conflict_policy_version=self.conflict_policy_version,
            minimum_conflict_numeric_only_items=self.minimum_conflict_numeric_only_items,
            minimum_conflict_price=self.minimum_conflict_price,
            minimum_popularity_top_percent=self.minimum_popularity_top_percent,
            multistage_dynamic_goals=self.multistage_dynamic_goals,
            post_update_explanation_gate=self.post_update_explanation_gate,
            constraint_surface_policy_version=(
                self.constraint_surface_policy_version
            ),
        )


class ConstraintEvidenceOrigin(str, Enum):
    HELD_OUT_ANCHOR = "held_out_anchor_catalog"
    HISTORY_SEQUENCE = "history_sequence_descriptive"
    CONTROLLED_SCENARIO = "precommitted_controlled_scenario"


class ConstraintEvidenceLink(StrictModel):
    goal_version: int = Field(ge=0)
    constraint_id: str = Field(min_length=1)
    slot: str = Field(min_length=1)
    origins: list[ConstraintEvidenceOrigin] = Field(min_length=1)
    evidence_strength: EvidenceStrength | None = None
    interpretation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_origins(self) -> ConstraintEvidenceLink:
        if len(self.origins) != len(set(self.origins)):
            raise ValueError("constraint evidence origins must be unique")
        if self.evidence_strength is not None and ConstraintEvidenceOrigin.HISTORY_SEQUENCE not in self.origins:
            raise ValueError("evidence strength is only defined for history evidence")
        return self


class NumericCoverageAudit(StrictModel):
    slot: str = Field(min_length=1)
    target_item_count: int = Field(ge=1)
    observed_item_count: int = Field(ge=0)
    missing_item_count: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    minimum_required_coverage: float = Field(ge=0.0, le=1.0)
    maximum_allowed_missing: int = Field(ge=0)
    policy_version: str = NUMERIC_MISSING_POLICY_VERSION

    @model_validator(mode="after")
    def validate_counts_and_gate(self) -> NumericCoverageAudit:
        if self.observed_item_count + self.missing_item_count != self.target_item_count:
            raise ValueError("numeric coverage counts do not sum to the target set")
        expected = self.observed_item_count / self.target_item_count
        if abs(self.coverage - expected) > 1e-12:
            raise ValueError("numeric coverage ratio differs from its counts")
        if self.coverage < self.minimum_required_coverage:
            raise ValueError("numeric coverage audit does not pass its registered gate")
        if self.missing_item_count > self.maximum_allowed_missing:
            raise ValueError("numeric missing count exceeds its registered gate")
        if self.policy_version != NUMERIC_MISSING_POLICY_VERSION:
            raise ValueError(f"unsupported numeric missing policy: {self.policy_version}")
        return self


class ConflictPlausibilityAudit(StrictModel):
    policy_version: str = JOINT_CONFLICT_POLICY_VERSION
    slot: str = Field(min_length=1)
    requested_value: float
    relaxed_value: float
    numeric_only_passing_count: int = Field(ge=1)
    joint_target_passing_count: int = Field(ge=0)
    relaxed_target_passing_count: int = Field(ge=1)
    requested_surface: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_joint_infeasibility(self) -> ConflictPlausibilityAudit:
        if self.policy_version != JOINT_CONFLICT_POLICY_VERSION:
            raise ValueError("unsupported conflict plausibility audit policy")
        if self.joint_target_passing_count != 0:
            raise ValueError("joint conflict audit must have zero target-subtype matches")
        if self.slot == "rating" and not 0.0 <= self.requested_value <= 5.0:
            raise ValueError("conflict rating request exceeds the catalog rating scale")
        if self.slot == "price" and self.requested_value <= 0.0:
            raise ValueError("conflict price request must remain positive")
        if self.slot == "popularity_percentile" and not 0.0 <= self.requested_value < 1.0:
            raise ValueError("conflict popularity request must remain inside the percentile scale")
        return self


class DevelopmentCaseEvidence(StrictModel):
    case_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    source_sequence_sha256: str = Field(min_length=64, max_length=64)
    relation_id: str = Field(min_length=1)
    scenario_family: ScenarioFamily
    history_category_reinforced: bool
    history_numeric_slot: str | None = None
    numeric_coverage: NumericCoverageAudit | None = None
    conflict_plausibility: ConflictPlausibilityAudit | None = None
    constraint_links: list[ConstraintEvidenceLink] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_links(self) -> DevelopmentCaseEvidence:
        keys = [(item.goal_version, item.constraint_id) for item in self.constraint_links]
        if len(keys) != len(set(keys)):
            raise ValueError("case evidence cannot repeat a goal-version constraint")
        has_numeric_slot = self.history_numeric_slot is not None
        has_numeric_coverage = self.numeric_coverage is not None
        if has_numeric_slot != has_numeric_coverage:
            raise ValueError("history numeric slot and coverage evidence must appear together")
        if has_numeric_slot:
            if self.scenario_family == ScenarioFamily.DIRECT:
                raise ValueError("direct controls cannot bind hidden numeric evidence")
            assert self.numeric_coverage is not None
            if self.numeric_coverage.slot != self.history_numeric_slot:
                raise ValueError("numeric coverage slot differs from history numeric slot")
        if self.scenario_family in {ScenarioFamily.REVISION, ScenarioFamily.CONFLICT}:
            if not has_numeric_slot:
                raise ValueError("dynamic trade-off cases require history numeric evidence")
        if self.conflict_plausibility is not None:
            if self.scenario_family != ScenarioFamily.CONFLICT:
                raise ValueError("only conflict cases may carry a plausibility audit")
            if self.conflict_plausibility.slot != self.history_numeric_slot:
                raise ValueError("conflict plausibility slot differs from history numeric slot")
        return self


class DevelopmentBenchmarkSummary(StrictModel):
    build_id: str
    case_count: int = Field(ge=0)
    domain_counts: dict[str, int]
    scenario_counts: dict[str, int]
    domain_scenario_counts: dict[str, int]
    distinct_user_count: int = Field(ge=0)
    distinct_relation_count: int = Field(ge=0)
    history_category_reinforced_count: int = Field(ge=0)
    history_numeric_slot_counts: dict[str, int]
    minimum_numeric_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_numeric_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    total_numeric_missing_items: int = Field(default=0, ge=0)
    joint_conflict_case_count: int = Field(default=0, ge=0)
    minimum_conflict_numeric_only_passing_count: int | None = Field(default=None, ge=1)
    mean_conflict_numeric_only_passing_count: float | None = Field(default=None, ge=1.0)
    minimum_relaxed_target_passing_count: int | None = Field(default=None, ge=1)
    multistage_dynamic_case_count: int = Field(default=0, ge=0)
    post_update_explanation_case_count: int = Field(default=0, ge=0)
    initial_hidden_constraint_count_distribution: dict[str, int] = Field(
        default_factory=dict
    )
    construction_failure_counts: dict[str, int]


class DevelopmentBenchmarkManifest(StrictModel):
    schema_version: str = DEVELOPMENT_BENCHMARK_VERSION
    artifact_role: str = "history_grounded_symbolic_development_benchmark"
    build_id: str
    config: DevelopmentBenchmarkConfig
    source_audit_id: str = Field(min_length=64, max_length=64)
    sequence_evidence_build_id: str
    sequence_evidence_profiles_sha256: str = Field(min_length=64, max_length=64)
    relation_review_sha256: str = Field(min_length=64, max_length=64)
    public_cases_sha256: str = Field(min_length=64, max_length=64)
    private_cases_sha256: str = Field(min_length=64, max_length=64)
    candidate_pools_sha256: str = Field(min_length=64, max_length=64)
    case_evidence_sha256: str = Field(min_length=64, max_length=64)
    files: list[PilotArtifactFile]

    @model_validator(mode="after")
    def validate_schema_version(self) -> DevelopmentBenchmarkManifest:
        if self.schema_version != self.config.builder_version:
            raise ValueError("development manifest and config versions differ")
        return self


@dataclass(frozen=True)
class DevelopmentBenchmarkBuildResult:
    build_id: str
    config: DevelopmentBenchmarkConfig
    source_audit: BenchmarkSourceAudit
    evidence_manifest: SequenceEvidenceManifest
    relation_review_sha256: str
    cases: tuple[BuiltSymbolicCase, ...]
    evidence: tuple[DevelopmentCaseEvidence, ...]
    attempts: tuple[ConstructionAttempt, ...]
    summary: DevelopmentBenchmarkSummary


def _reliable_numeric(profile: SequenceEvidenceProfile, slot: str) -> EvidenceStrength:
    item = next(
        (
            value
            for value in profile.numeric_evidence
            if value.slot == slot
            and value.strength in {EvidenceStrength.MODERATE, EvidenceStrength.STRONG}
        ),
        None,
    )
    if item is None:
        raise ValueError("selected history numeric evidence is not reliable")
    return item.strength


def _case_evidence(
    case: BuiltSymbolicCase,
    profile: SequenceEvidenceProfile,
) -> DevelopmentCaseEvidence:
    links: list[ConstraintEvidenceLink] = []
    for goal in case.pair.private.initial_state.latent_goal.versions:
        for constraint in goal.constraints:
            if constraint.constraint_id in {"c-visible-category", "c-specific-category"}:
                origins = [ConstraintEvidenceOrigin.HELD_OUT_ANCHOR]
                strength = None
                interpretation = "target category grounded by the held-out anchor catalog tags"
                category = next(
                    (item for item in profile.category_evidence if item.tag == constraint.predicate.value),
                    None,
                )
                if category is not None:
                    origins.append(ConstraintEvidenceOrigin.HISTORY_SEQUENCE)
                    strength = category.strength
                    interpretation = (
                        "anchor-grounded target category with repeated descriptive history support"
                    )
            elif constraint.constraint_id == "c-tradeoff":
                if case.pair.private.initial_state.scenario_family == ScenarioFamily.CONFLICT and goal.version == 0:
                    origins = [ConstraintEvidenceOrigin.CONTROLLED_SCENARIO]
                    strength = None
                    interpretation = (
                        "precommitted aspirational conflict used to study explicit negotiation"
                    )
                else:
                    origins = [ConstraintEvidenceOrigin.HISTORY_SEQUENCE]
                    if case.history_numeric_slot is None:
                        raise ValueError("history-grounded trade-off is missing its evidence slot")
                    strength = _reliable_numeric(profile, case.history_numeric_slot)
                    interpretation = (
                        "threshold taken from the fixed descriptive history interquartile profile"
                    )
            else:
                raise ValueError(f"unregistered development constraint: {constraint.constraint_id}")
            links.append(
                ConstraintEvidenceLink(
                    goal_version=goal.version,
                    constraint_id=constraint.constraint_id,
                    slot=constraint.slot,
                    origins=origins,
                    evidence_strength=strength,
                    interpretation=interpretation,
                )
            )
    conflict_plausibility = None
    if (
        case.pair.private.initial_state.scenario_family == ScenarioFamily.CONFLICT
        and case.numeric_plan is not None
        and case.numeric_plan.conflict_policy_version == JOINT_CONFLICT_POLICY_VERSION
    ):
        initial_tradeoff = next(
            item
            for item in case.pair.private.initial_state.latent_goal.version(0).constraints
            if item.constraint_id == "c-tradeoff"
        )
        conflict_plausibility = ConflictPlausibilityAudit(
            slot=case.numeric_plan.slot,
            requested_value=case.numeric_plan.impossible_value,
            relaxed_value=case.numeric_plan.relaxed_value,
            numeric_only_passing_count=case.numeric_plan.conflict_numeric_only_passing_count,
            joint_target_passing_count=case.numeric_plan.conflict_joint_passing_count,
            relaxed_target_passing_count=case.numeric_plan.relaxed_joint_passing_count,
            requested_surface=initial_tradeoff.surface_form,
        )
    return DevelopmentCaseEvidence(
        case_id=case.pair.public.case_id,
        profile_id=profile.profile_id,
        source_sequence_sha256=profile.source_sequence_sha256,
        relation_id=case.relation_id,
        scenario_family=case.pair.private.initial_state.scenario_family,
        history_category_reinforced=case.history_category_reinforced,
        history_numeric_slot=case.history_numeric_slot,
        numeric_coverage=(
            NumericCoverageAudit(
                slot=case.numeric_plan.slot,
                target_item_count=case.numeric_plan.target_item_count,
                observed_item_count=case.numeric_plan.observed_item_count,
                missing_item_count=case.numeric_plan.missing_item_count,
                coverage=case.numeric_plan.coverage,
                minimum_required_coverage=case.numeric_plan.minimum_required_coverage,
                maximum_allowed_missing=case.numeric_plan.maximum_allowed_missing,
            )
            if case.numeric_plan is not None
            else None
        ),
        conflict_plausibility=conflict_plausibility,
        constraint_links=links,
    )


def build_development_benchmark(
    *,
    source_root: Path,
    review_path: Path,
    sequence_evidence_dir: Path,
    config: DevelopmentBenchmarkConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> DevelopmentBenchmarkBuildResult:
    config = config or DevelopmentBenchmarkConfig()
    source_root = source_root.resolve()
    review = load_relation_review(review_path)
    review_sha256 = relation_review_sha256(review)
    evidence_manifest, profiles = load_sequence_evidence_artifact(sequence_evidence_dir)
    profile_by_user = {item.pseudonymous_user_id: item for item in profiles}
    source_result = audit_benchmark_sources(source_root, SourceAuditConfig())
    if source_result.audit.audit_id != evidence_manifest.source_audit_id:
        raise ValueError("sequence evidence and current source audit IDs differ")
    if set(profile_by_user) != {
        item.pseudonymous_user_id for item in source_result.eligible_users
    }:
        raise ValueError("sequence evidence does not cover exactly the eligible users")

    audit_by_domain = {item.domain: item for item in source_result.audit.domains}
    users_by_domain: dict[str, list[EligibleUserSource]] = defaultdict(list)
    for source in source_result.eligible_users:
        users_by_domain[source.domain].append(source)

    built: list[BuiltSymbolicCase] = []
    evidence_records: list[DevelopmentCaseEvidence] = []
    attempts: list[ConstructionAttempt] = []
    selected_users: set[str] = set()
    relation_counts: Counter[tuple[str, str]] = Counter()
    pilot_config = config.pilot_config()

    for domain in sorted(users_by_domain):
        metadata_path = source_root / domain / "metadata.csv"
        rows, _ = load_normalized_catalog(metadata_path)
        row_by_id = {row.item_id: row for row in rows}
        discovered = discover_category_relations(rows, domain, metadata_path, source_root)
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
            selected_for_scenario = 0
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
                if selected_for_scenario >= quota:
                    break
                if source.pseudonymous_user_id in selected_users:
                    continue
                profile = profile_by_user[source.pseudonymous_user_id]
                for relation in user_relations.get(source.pseudonymous_user_id, []):
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
                            config=pilot_config,
                            sequence_evidence_profile=profile,
                            require_history_numeric=(
                                config.require_reliable_history_numeric_for_dynamic_tradeoff
                                and scenario
                                in {
                                    ScenarioFamily.PROGRESSIVE,
                                    ScenarioFamily.AMBIGUOUS,
                                    ScenarioFamily.REVISION,
                                    ScenarioFamily.CONFLICT,
                                }
                            ),
                            case_builder_version=config.case_identity_version,
                        )
                        case_evidence = _case_evidence(case, profile)
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
                    evidence_records.append(case_evidence)
                    selected_users.add(source.pseudonymous_user_id)
                    relation_counts[count_key] += 1
                    selected_for_scenario += 1
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
            if selected_for_scenario != quota:
                raise RuntimeError(
                    f"DEVELOPMENT_QUOTA_SHORTFALL:{domain}:{scenario.value}:"
                    f"{selected_for_scenario}/{quota}"
                )
            if progress is not None:
                progress(
                    f"{domain}/{scenario.value}: selected "
                    f"{selected_for_scenario}/{quota} cases."
                )

    by_case = {item.case_id: item for item in evidence_records}
    if config.multistage_dynamic_goals:
        expected_multistage = [
            item
            for item in built
            if item.pair.private.initial_state.scenario_family
            in {ScenarioFamily.PROGRESSIVE, ScenarioFamily.AMBIGUOUS}
        ]
        if any(item.numeric_plan is None for item in expected_multistage):
            raise AssertionError("multistage progressive/ambiguous case lacks numeric goal")
    built.sort(key=lambda item: item.pair.public.case_id)
    evidence_records = [by_case[item.pair.public.case_id] for item in built]
    public_payload = [item.pair.public.model_dump(mode="json") for item in built]
    private_payload = [item.pair.private.model_dump(mode="json") for item in built]
    pool_payload = [item.candidate_pool_record.model_dump(mode="json") for item in built]
    evidence_payload = [item.model_dump(mode="json") for item in evidence_records]
    identity = {
        "builder_version": config.builder_version,
        "config": config.model_dump(mode="json"),
        "source_audit_id": source_result.audit.audit_id,
        "sequence_evidence_build_id": evidence_manifest.build_id,
        "relation_review_sha256": review_sha256,
        "public_cases_sha256": content_sha256(public_payload),
        "private_cases_sha256": content_sha256(private_payload),
        "candidate_pools_sha256": content_sha256(pool_payload),
        "case_evidence_sha256": content_sha256(evidence_payload),
    }
    build_id = f"development-benchmark-{content_sha256(identity)[:16]}"
    failure_counts = Counter(item.reason for item in attempts if not item.selected)
    numeric_audits = [
        item.numeric_coverage for item in evidence_records if item.numeric_coverage is not None
    ]
    conflict_audits = [
        item.conflict_plausibility
        for item in evidence_records
        if item.conflict_plausibility is not None
    ]
    summary = DevelopmentBenchmarkSummary(
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
                    f"{item.pair.public.domain}:"
                    f"{item.pair.private.initial_state.scenario_family.value}"
                    for item in built
                ).items()
            )
        ),
        distinct_user_count=len({item.pseudonymous_user_id for item in built}),
        distinct_relation_count=len({item.relation_id for item in built}),
        history_category_reinforced_count=sum(item.history_category_reinforced for item in built),
        history_numeric_slot_counts=dict(
            sorted(Counter(item.history_numeric_slot for item in built if item.history_numeric_slot).items())
        ),
        minimum_numeric_coverage=(
            min(item.coverage for item in numeric_audits) if numeric_audits else None
        ),
        mean_numeric_coverage=(
            sum(item.coverage for item in numeric_audits) / len(numeric_audits)
            if numeric_audits
            else None
        ),
        total_numeric_missing_items=sum(item.missing_item_count for item in numeric_audits),
        joint_conflict_case_count=len(conflict_audits),
        minimum_conflict_numeric_only_passing_count=(
            min(item.numeric_only_passing_count for item in conflict_audits)
            if conflict_audits
            else None
        ),
        mean_conflict_numeric_only_passing_count=(
            sum(item.numeric_only_passing_count for item in conflict_audits)
            / len(conflict_audits)
            if conflict_audits
            else None
        ),
        minimum_relaxed_target_passing_count=(
            min(item.relaxed_target_passing_count for item in conflict_audits)
            if conflict_audits
            else None
        ),
        multistage_dynamic_case_count=sum(
            item.numeric_plan is not None
            and item.pair.private.initial_state.scenario_family
            in {ScenarioFamily.PROGRESSIVE, ScenarioFamily.AMBIGUOUS}
            for item in built
        ),
        post_update_explanation_case_count=sum(
            any(
                event.response_act is not None
                and event.response_act.value == "request_explanation"
                for event in item.pair.private.initial_state.evolution_plan.events
            )
            for item in built
        ),
        initial_hidden_constraint_count_distribution=dict(
            sorted(
                Counter(
                    str(len(item.pair.private.initial_state.disclosure_plan.hidden_order))
                    for item in built
                ).items()
            )
        ),
        construction_failure_counts=dict(sorted(failure_counts.items())),
    )
    return DevelopmentBenchmarkBuildResult(
        build_id=build_id,
        config=config,
        source_audit=source_result.audit,
        evidence_manifest=evidence_manifest,
        relation_review_sha256=review_sha256,
        cases=tuple(built),
        evidence=tuple(evidence_records),
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
    return "".join(canonical_json(item) + "\n" for item in records)


def write_development_benchmark_artifact(
    result: DevelopmentBenchmarkBuildResult,
    output_root: Path,
) -> Path:
    run_dir = output_root / result.build_id
    run_dir.mkdir(parents=True, exist_ok=True)
    public = [item.pair.public.model_dump(mode="json") for item in result.cases]
    private = [item.pair.private.model_dump(mode="json") for item in result.cases]
    pools = [item.candidate_pool_record.model_dump(mode="json") for item in result.cases]
    evidence = [item.model_dump(mode="json") for item in result.evidence]
    attempts = [item.model_dump(mode="json") for item in result.attempts]
    outputs = {
        "cases.public.jsonl": _jsonl(public),
        "cases.private.jsonl": _jsonl(private),
        "candidate_pools.public.jsonl": _jsonl(pools),
        "case_evidence.private.jsonl": _jsonl(evidence),
        "construction_ledger.private.jsonl": _jsonl(attempts),
        "summary.json": json.dumps(
            result.summary.model_dump(mode="json"), ensure_ascii=False, indent=2
        )
        + "\n",
        "source_audit.json": json.dumps(
            result.source_audit.model_dump(mode="json"), ensure_ascii=False, indent=2
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
    manifest = DevelopmentBenchmarkManifest(
        schema_version=result.config.builder_version,
        build_id=result.build_id,
        config=result.config,
        source_audit_id=result.source_audit.audit_id,
        sequence_evidence_build_id=result.evidence_manifest.build_id,
        sequence_evidence_profiles_sha256=result.evidence_manifest.profiles_sha256,
        relation_review_sha256=result.relation_review_sha256,
        public_cases_sha256=content_sha256(public),
        private_cases_sha256=content_sha256(private),
        candidate_pools_sha256=content_sha256(pools),
        case_evidence_sha256=content_sha256(evidence),
        files=files,
    )
    _write_idempotently(
        run_dir / "manifest.json",
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
    )
    return run_dir
