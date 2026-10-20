"""Public feature and private label schemas for counterfactual supervision."""

from __future__ import annotations

from pydantic import Field, model_validator

from taira_mt.dialogue_policy.cost_aware import PatternInstantiation
from taira_mt.models import BeliefState, StrictModel, SystemAction, UserAct
from taira_mt.provenance import content_sha256


class CounterfactualRewardConfig(StrictModel):
    horizon: int = Field(default=2, ge=1, le=8)
    rollout_replicates: int = Field(default=4, ge=1, le=64)
    discount: float = Field(default=0.9, ge=0.0, le=1.0)
    acceptance_bonus: float = Field(default=1.0, ge=0.0)
    abandonment_penalty: float = Field(default=1.0, ge=0.0)
    hard_violation_penalty: float = Field(default=0.5, ge=0.0)
    turn_cost: float = Field(default=0.02, ge=0.0)
    ask_cost: float = Field(default=0.04, ge=0.0)
    repair_cost: float = Field(default=0.02, ge=0.0)
    stop_cost: float = Field(default=0.0, ge=0.0)
    ask_margin: float = 0.0
    readiness_temperature: float = Field(default=0.2, gt=0.0)


class CounterfactualPublicState(StrictModel):
    belief_hash: str
    candidate_pool_id: str
    pattern_library_hash: str
    tracker_name: str
    instantiator_version: str
    belief: BeliefState
    candidates: list[PatternInstantiation]

    @model_validator(mode="after")
    def validate_public_identity(self) -> CounterfactualPublicState:
        if self.belief_hash != content_sha256(self.belief.model_dump(mode="json")):
            raise ValueError("belief_hash does not match counterfactual belief")
        for candidate in self.candidates:
            if candidate.candidate_pool_id != self.candidate_pool_id:
                raise ValueError("candidate instantiation references a different pool")
            if candidate.pattern_library_hash != self.pattern_library_hash:
                raise ValueError("candidate instantiation references a different Pattern library")
            if candidate.instantiator_version != self.instantiator_version:
                raise ValueError("candidate instantiation version mismatch")
        return self


class BranchStepTrace(StrictModel):
    step_index: int
    action: SystemAction
    true_user_act: UserAct
    satisfaction_delta: float
    acceptance_component: float
    abandonment_component: float
    hard_violation_component: float
    turn_cost_component: float
    action_cost_component: float
    undiscounted_reward: float
    discounted_reward: float


class BranchRollout(StrictModel):
    instantiation_id: str
    rollout_seed: int
    steps: list[BranchStepTrace]
    discounted_return: float
    accepted: bool
    abandoned: bool
    hard_violation: bool


class CounterfactualActionLabel(StrictModel):
    instantiation_id: str
    pattern_id: str
    action_type: str
    q_mean: float
    q_std: float = Field(ge=0.0)
    accept_probability: float = Field(ge=0.0, le=1.0)
    abandon_probability: float = Field(ge=0.0, le=1.0)
    hard_violation_probability: float = Field(ge=0.0, le=1.0)
    rollouts: list[BranchRollout]


class PairwisePreference(StrictModel):
    preferred_instantiation_id: str
    dispreferred_instantiation_id: str
    value_margin: float = Field(ge=0.0)


class CounterfactualDecisionLabel(StrictModel):
    belief_hash: str
    candidate_pool_id: str
    root_seed: int
    reward_config_hash: str
    simulator_checkpoint_digest: str = Field(min_length=64, max_length=64)
    action_labels: list[CounterfactualActionLabel]
    best_instantiation_id: str
    best_pattern_id: str
    regret_by_instantiation: dict[str, float]
    pairwise_preferences: list[PairwisePreference]
    net_voc_by_ask_instantiation: dict[str, float]
    ask_comparison_available: bool
    should_ask: bool
    readiness_label: float = Field(ge=0.0, le=1.0)


class CounterfactualPublicRecord(StrictModel):
    schema_version: str = "counterfactual_record_v3"
    state_id: str
    dataset_build_id: str
    rollout_run_id: str
    episode_id: str
    case_id: str
    trajectory_id: str | None = None
    turn_index: int = Field(ge=0)
    repair_context_id: str | None = None
    repair_context_public_hash: str | None = None
    state: CounterfactualPublicState

    @model_validator(mode="after")
    def validate_state_turn(self) -> CounterfactualPublicRecord:
        if self.turn_index != self.state.belief.turn_index:
            raise ValueError("counterfactual record turn does not match belief turn")
        if (self.repair_context_id is None) != (self.repair_context_public_hash is None):
            raise ValueError("repair context ID and public hash must be present together")
        return self


class CounterfactualPrivateRecord(StrictModel):
    schema_version: str = "counterfactual_record_v3"
    state_id: str
    dataset_build_id: str
    rollout_run_id: str
    episode_id: str
    case_id: str
    trajectory_id: str | None = None
    turn_index: int = Field(ge=0)
    repair_context_id: str | None = None
    repair_context_public_hash: str | None = None
    label: CounterfactualDecisionLabel
