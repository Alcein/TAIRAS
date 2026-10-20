"""Strict public/private contracts for the zero-API engineering pilot."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from taira_mt.models import (
    BeliefState,
    EvaluationTrace,
    HiddenGoal,
    PublicConstraint,
    StrictModel,
    SystemAction,
    TerminationReason,
    UserObservation,
    UserPersona,
)
from taira_mt.provenance import content_sha256


class PilotScenario(str, Enum):
    CLARIFY = "clarify"
    READY = "ready"
    REPAIR = "repair"


class SourceFileRecord(StrictModel):
    domain: str
    kind: str
    relative_path: str
    sha256: str
    row_count: int = Field(ge=0)


class ConstructionProvenance(StrictModel):
    builder_version: str
    anchor_item_id: str
    anchor_title: str
    raw_user_id: str
    source_query_relative_path: str
    source_query_sha256: str
    excluded_history_item_ids: list[str]
    positive_count_in_source: int = Field(ge=0)
    near_negative_count_in_source: int = Field(ge=0)
    far_negative_count_in_source: int = Field(ge=0)


class CandidatePoolRecord(StrictModel):
    pool_id: str
    domain: str
    catalog_source_sha256: str
    retrieval_recipe_id: str
    retrieval_recipe_config: dict[str, Any]
    ordered_item_ids: list[str]

    @model_validator(mode="after")
    def validate_identity(self) -> CandidatePoolRecord:
        payload = {
            "domain": self.domain,
            "catalog_source_sha256": self.catalog_source_sha256,
            "retrieval_recipe_id": self.retrieval_recipe_id,
            "retrieval_recipe_config": self.retrieval_recipe_config,
            "ordered_item_ids": self.ordered_item_ids,
        }
        if self.pool_id != content_sha256(payload):
            raise ValueError("candidate pool ID does not match its canonical definition")
        if not self.ordered_item_ids or len(self.ordered_item_ids) != len(set(self.ordered_item_ids)):
            raise ValueError("candidate pool requires unique ordered item IDs")
        return self


class PublicCaseSpec(StrictModel):
    case_id: str
    domain: str
    scenario: PilotScenario
    initial_request: str
    observed_constraints: dict[str, PublicConstraint]
    ontology_id: str
    candidate_pool_id: str


class PrivateCaseSpec(StrictModel):
    case_id: str
    hidden_goal: HiddenGoal
    persona: UserPersona
    acceptable_item_ids: list[str]
    provenance: ConstructionProvenance


class PatternDecisionTrace(StrictModel):
    pattern_id: str
    pattern_version: str
    pattern_library_hash: str
    belief_before_hash: str
    applicable_pattern_ids: list[str]
    selected_action_type: str
    instantiated_information_target: str | None = None


class PublicTurnRecord(StrictModel):
    turn_index: int
    belief_before: BeliefState
    pattern_decision: PatternDecisionTrace
    system_action: SystemAction
    public_user_text: str
    policy_observation: UserObservation
    belief_after: BeliefState


class PublicEpisode(StrictModel):
    schema_version: str = "0.1.0"
    episode_id: str
    seed: int
    case_id: str
    trajectory_id: str | None = None
    domain: str
    public_initial_request: str
    candidate_pool_id: str
    pattern_library_hash: str
    repair_context_id: str | None = None
    repair_context_public_hash: str | None = None
    turns: list[PublicTurnRecord]

    @model_validator(mode="after")
    def validate_turns(self) -> PublicEpisode:
        if [turn.turn_index for turn in self.turns] != list(range(len(self.turns))):
            raise ValueError("public turn indices must be contiguous")
        if (self.repair_context_id is None) != (self.repair_context_public_hash is None):
            raise ValueError("repair context ID and public hash must be present together")
        return self


class PrivateTurnRecord(StrictModel):
    turn_index: int
    public_record_hash: str
    reward: float
    evaluation_trace: EvaluationTrace


class PrivateEpisode(StrictModel):
    schema_version: str = "0.1.0"
    episode_id: str
    case_id: str
    trajectory_id: str | None = None
    goal_id: str
    persona_id: str
    repair_context_id: str | None = None
    repair_context_public_hash: str | None = None
    turns: list[PrivateTurnRecord]
    termination_reason: TerminationReason
    success: bool
    metrics: dict[str, Any]

    @model_validator(mode="after")
    def validate_context_identity(self) -> PrivateEpisode:
        if (self.repair_context_id is None) != (self.repair_context_public_hash is None):
            raise ValueError("repair context ID and public hash must be present together")
        return self


class PilotFailure(StrictModel):
    case_id: str
    error_type: str
    message: str


class PilotSummary(StrictModel):
    case_count: int
    episode_count: int
    failure_count: int
    success_count: int
    domain_counts: dict[str, int]
    scenario_counts: dict[str, int]
    action_counts: dict[str, int]
    pattern_counts: dict[str, int]
    counterfactual_state_count: int = 0
    counterfactual_branch_count: int = 0
    external_calls: bool = False


class OutputFileRecord(StrictModel):
    relative_path: str
    sha256: str
    line_count: int = Field(ge=0)


class PilotRunManifest(StrictModel):
    schema_version: str = "0.1.0"
    run_id: str
    config_hash: str
    root_seed: int
    episode_seed_derivation: str
    external_calls: bool = False
    pattern_library_hash: str
    source_files: list[SourceFileRecord]
    output_files: list[OutputFileRecord]
    case_count: int
    episode_count: int
    failure_count: int
