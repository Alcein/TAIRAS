"""Public features and evaluator-private labels for DSUS snapshots."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, field_validator, model_validator

from taira_mt.benchmark.schemas import (
    BenchmarkActionType,
    BenchmarkUserAct,
    CandidateFeasibility,
    DialoguePhaseV1,
    IntentCompleteness,
    IntentStability,
    ScenarioFamily,
)
from taira_mt.models import ConstraintPredicate, StrictModel

DSUS_DATASET_VERSION = "dsus_evolving_intent_development_v3_public_corrective_expert"


class DSUSDevelopmentSplit(str, Enum):
    TRAIN = "train"
    DEV = "dev"


class DSUSBehaviorPolicy(str, Enum):
    CORRECTIVE_EXPERT = "public_information_corrective_expert"
    ALWAYS_ASK = "always_ask"
    ALWAYS_RECOMMEND = "always_recommend"
    DETERMINISTIC_RANDOM = "deterministic_random"
    PUBLIC_HEURISTIC = "public_heuristic"


class PatienceRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SatisfactionBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DecisionObligation(str, Enum):
    ELICIT_PREFERENCE = "elicit_preference"
    RESOLVE_AMBIGUITY = "resolve_ambiguity"
    RECOMMEND = "recommend"
    GROUND_EXPLANATION = "ground_explanation"
    NEGOTIATE_TRADEOFF = "negotiate_tradeoff"
    TERMINATE = "terminate"


class DSUSDatasetConfig(StrictModel):
    builder_version: str = DSUS_DATASET_VERSION
    split_seed: str = "dsus-development-split-v1"
    dev_cases_per_domain_scenario: int = Field(default=4, ge=1, le=8)
    behavior_policies: list[DSUSBehaviorPolicy] = Field(
        default_factory=lambda: list(DSUSBehaviorPolicy)
    )
    policy_root_seed: int = Field(default=20260721, ge=0)
    maximum_behavior_turns: int = Field(default=6, ge=2, le=12)
    include_public_catalog_summary: bool = True

    @model_validator(mode="after")
    def validate_version(self) -> DSUSDatasetConfig:
        if self.builder_version != DSUS_DATASET_VERSION:
            raise ValueError(f"unsupported DSUS dataset version: {self.builder_version}")
        if len(self.behavior_policies) != len(set(self.behavior_policies)):
            raise ValueError("DSUS behavior policies must be unique")
        if DSUSBehaviorPolicy.CORRECTIVE_EXPERT not in self.behavior_policies:
            raise ValueError("DSUS data requires the public-information corrective expert")
        return self


class PublicConstraintObservation(StrictModel):
    slot: str
    predicate: ConstraintPredicate
    surface_form: str


class PublicDSUSFeature(StrictModel):
    """Minimal online DSUS input assembled from Agent-visible interaction memory.

    Unlike :class:`DSUSStateFeature`, this contract deliberately has no case,
    episode, split, renderer, behavior-policy, or semantic-state identifier.
    It is the only feature shape accepted by an online Agent after the
    development-data adapter has been removed.
    """

    domain: str
    turn_index: int = Field(ge=0)
    serialized_dialogue: str = Field(min_length=1)
    public_constraints: list[PublicConstraintObservation]
    public_verified_candidate_count: int = Field(ge=0)
    public_unknown_candidate_count: int = Field(ge=0)
    asked_question_targets: list[str] = Field(default_factory=list)
    rejected_item_count: int = Field(ge=0)
    last_system_action_type: BenchmarkActionType | None = None

    @field_validator("asked_question_targets")
    @classmethod
    def unique_question_targets(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("public asked-question targets must be unique")
        return value

    @model_validator(mode="after")
    def validate_public_constraint_slots(self) -> PublicDSUSFeature:
        slots = [item.slot for item in self.public_constraints]
        if len(slots) != len(set(slots)):
            raise ValueError("public constraint observations must have unique slots")
        return self

    def model_inputs(self) -> dict[str, object]:
        """Return the exact safe payload for an online state estimator."""

        return self.model_dump(mode="json")


class DSUSStateFeature(StrictModel):
    """One public decision snapshot; metadata is excluded by ``model_inputs``."""

    schema_version: str = DSUS_DATASET_VERSION
    state_id: str = Field(min_length=64, max_length=64)
    semantic_state_id: str = Field(min_length=64, max_length=64)
    episode_id: str
    behavior_policy: DSUSBehaviorPolicy
    case_id: str
    split_group_id: str
    split: DSUSDevelopmentSplit
    domain: str
    turn_index: int = Field(ge=0)
    language_renderer_id: str
    serialized_dialogue: str
    public_constraints: list[PublicConstraintObservation]
    public_verified_candidate_count: int = Field(ge=0)
    public_unknown_candidate_count: int = Field(ge=0)
    asked_question_targets: list[str] = Field(default_factory=list)
    rejected_item_count: int = Field(ge=0)
    last_system_action_type: BenchmarkActionType | None = None

    @field_validator("asked_question_targets")
    @classmethod
    def unique_question_targets(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("public asked-question targets must be unique")
        return value

    def model_inputs(self) -> dict[str, object]:
        return self.online_feature().model_inputs()

    def online_feature(self) -> PublicDSUSFeature:
        """Drop development-only lineage before an online prediction call."""

        return PublicDSUSFeature(
            domain=self.domain,
            turn_index=self.turn_index,
            serialized_dialogue=self.serialized_dialogue,
            public_constraints=self.public_constraints,
            public_verified_candidate_count=self.public_verified_candidate_count,
            public_unknown_candidate_count=self.public_unknown_candidate_count,
            asked_question_targets=list(self.asked_question_targets),
            rejected_item_count=self.rejected_item_count,
            last_system_action_type=self.last_system_action_type,
        )


class DSUSStateLabel(StrictModel):
    """Private training target aligned to one public DSUS feature record."""

    schema_version: str = DSUS_DATASET_VERSION
    state_id: str = Field(min_length=64, max_length=64)
    semantic_state_id: str = Field(min_length=64, max_length=64)
    case_id: str
    behavior_policy: DSUSBehaviorPolicy
    split: DSUSDevelopmentSplit
    scenario_family: ScenarioFamily
    state_archetype_id: str = Field(min_length=64, max_length=64)
    intent_completeness: IntentCompleteness
    intent_stability: IntentStability
    candidate_feasibility: CandidateFeasibility
    dialogue_phase: DialoguePhaseV1
    last_user_act: BenchmarkUserAct | None = None
    explanation_pending: bool
    patience_remaining_ratio: float = Field(ge=0.0, le=1.0)
    patience_risk: PatienceRisk
    satisfaction: float = Field(ge=0.0, le=1.0)
    satisfaction_band: SatisfactionBand
    decision_obligation: DecisionObligation
    optimal_action_type: BenchmarkActionType
    optimal_question_target: str | None = None

    @model_validator(mode="after")
    def validate_decision_target(self) -> DSUSStateLabel:
        expected = {
            DecisionObligation.ELICIT_PREFERENCE: BenchmarkActionType.ASK,
            DecisionObligation.RESOLVE_AMBIGUITY: BenchmarkActionType.ASK,
            DecisionObligation.RECOMMEND: BenchmarkActionType.RECOMMEND,
            DecisionObligation.GROUND_EXPLANATION: BenchmarkActionType.EXPLAIN,
            DecisionObligation.NEGOTIATE_TRADEOFF: BenchmarkActionType.ASK,
            DecisionObligation.TERMINATE: BenchmarkActionType.STOP,
        }[self.decision_obligation]
        if self.optimal_action_type != expected:
            raise ValueError("decision obligation and optimal action differ")
        if (self.optimal_action_type == BenchmarkActionType.ASK) != (
            self.optimal_question_target is not None
        ):
            raise ValueError("only optimal ASK states carry a question target")
        return self


class DSUSPosterior(StrictModel):
    """Public-model posterior consumed by Pattern matching and CIV.

    All categorical distributions are model outputs; the evaluator-private
    ``DSUSStateLabel`` is never embedded in an online feature record.  The
    posterior is deliberately factorized so that a downstream Pattern can use
    uncertainty about, for example, feasibility without pretending that every
    factor is known with equal confidence.
    """

    intent_completeness: dict[str, float]
    intent_stability: dict[str, float]
    candidate_feasibility: dict[str, float]
    dialogue_phase: dict[str, float]
    last_user_act: dict[str, float]
    explanation_pending: dict[str, float]
    patience_risk: dict[str, float]
    satisfaction_band: dict[str, float]
    patience_remaining_ratio: float = Field(ge=0.0, le=1.0)
    satisfaction: float = Field(ge=0.0, le=1.0)

    @field_validator(
        "intent_completeness",
        "intent_stability",
        "candidate_feasibility",
        "dialogue_phase",
        "last_user_act",
        "explanation_pending",
        "patience_risk",
        "satisfaction_band",
    )
    @classmethod
    def validate_probability_map(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("DSUS posterior distributions must be non-empty")
        if any(not 0.0 <= probability <= 1.0 for probability in value.values()):
            raise ValueError("DSUS posterior probabilities must lie in [0, 1]")
        if abs(sum(value.values()) - 1.0) > 1e-6:
            raise ValueError("DSUS posterior probabilities must sum to one")
        return value

    @staticmethod
    def maximum_probability(distribution: dict[str, float]) -> tuple[str, float]:
        """Return a deterministic MAP value and its confidence."""

        return min(distribution.items(), key=lambda item: (-item[1], item[0]))


class DSUSSplitRecord(StrictModel):
    split_group_id: str
    split: DSUSDevelopmentSplit
    case_ids: list[str]
    domain_scenario_counts: dict[str, int]


class DSUSDevelopmentSummary(StrictModel):
    build_id: str
    parent_benchmark_build_id: str
    parent_language_bank_id: str
    case_count: int = Field(ge=1)
    episode_count: int = Field(ge=1)
    state_count: int = Field(ge=1)
    unique_semantic_state_count: int = Field(ge=1)
    semantic_duplicate_count: int = Field(ge=0)
    state_archetype_count: int = Field(ge=1)
    split_case_counts: dict[str, int]
    split_state_counts: dict[str, int]
    split_group_counts: dict[str, int]
    state_domain_scenario_split_counts: dict[str, int]
    behavior_policy_episode_counts: dict[str, int]
    behavior_policy_state_counts: dict[str, int]
    behavior_policy_success_rates: dict[str, float]
    optimal_action_counts: dict[str, int]
    decision_obligation_counts: dict[str, int]
    dialogue_phase_counts: dict[str, int]
    intent_stability_counts: dict[str, int]
    candidate_feasibility_counts: dict[str, int]
    group_overlap_count: int = Field(ge=0)
    all_corrective_expert_episodes_successful: bool
    feature_label_join_exact: bool
    runtime_external_calls: bool = False


class DSUSArtifactFile(StrictModel):
    relative_path: str
    sha256: str = Field(min_length=64, max_length=64)
    byte_size: int = Field(ge=0)
    line_count: int | None = Field(default=None, ge=0)


class DSUSDevelopmentManifest(StrictModel):
    schema_version: str = DSUS_DATASET_VERSION
    artifact_role: str = "decision_sufficient_user_state_development_data"
    build_id: str
    config: DSUSDatasetConfig
    freeze_id: str
    parent_benchmark_build_id: str
    parent_benchmark_manifest_sha256: str = Field(min_length=64, max_length=64)
    parent_language_bank_id: str
    parent_language_bank_manifest_sha256: str = Field(min_length=64, max_length=64)
    language_renderer_id: str
    features_sha256: str = Field(min_length=64, max_length=64)
    labels_sha256: str = Field(min_length=64, max_length=64)
    episodes_sha256: str = Field(min_length=64, max_length=64)
    split_registry_sha256: str = Field(min_length=64, max_length=64)
    source_tree_sha256: str = Field(min_length=64, max_length=64)
    outputs: list[DSUSArtifactFile]
