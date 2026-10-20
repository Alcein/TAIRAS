"""Strict public/private contracts for the fixed multi-turn user environment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from taira_mt.models import ConstraintPredicate, OracleAssessment, StrictModel, normalize_slot_name
from taira_mt.provenance import content_sha256

BENCHMARK_CASE_SCHEMA_VERSION = "user_benchmark_case_v1"
BENCHMARK_ACTION_SCHEMA_VERSION = "user_benchmark_action_v1"
LEGACY_BENCHMARK_TRACE_SCHEMA_VERSION = "user_benchmark_trace_v1"
BENCHMARK_TRACE_SCHEMA_VERSION = "user_benchmark_trace_v2_language_surface"
LEGACY_SIMULATOR_VERSION = "evolving_intent_fsm_v1"
EXPLANATION_GUARD_SIMULATOR_VERSION = "evolving_intent_fsm_v2_explanation_guard"
SIMULATOR_VERSION = "evolving_intent_fsm_v3_preference_grounded_explanation"


def _unique(values: list[str], *, field_name: str) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique values")
    return values


def benchmark_content_sha256(value: StrictModel | dict[str, Any] | list[Any]) -> str:
    """Hash a benchmark record with the canonical project serializer."""

    payload = value.model_dump(mode="json") if isinstance(value, StrictModel) else value
    return content_sha256(payload)


class ScenarioFamily(str, Enum):
    DIRECT = "direct"
    PROGRESSIVE = "progressive"
    AMBIGUOUS = "ambiguous"
    REVISION = "revision"
    CONFLICT = "conflict"


class DialoguePhaseV1(str, Enum):
    ELICIT = "elicit"
    READY = "ready"
    FEEDBACK = "feedback"
    REPAIR = "repair"
    NEGOTIATE = "negotiate"
    TERMINAL = "terminal"


class IntentCompleteness(str, Enum):
    INCOMPLETE = "incomplete"
    AMBIGUOUS = "ambiguous"
    SUFFICIENT = "sufficient"


class IntentStability(str, Enum):
    STABLE = "stable"
    CORRECTED = "corrected"
    TIGHTENED = "tightened"
    RELAXED = "relaxed"


class CandidateFeasibility(str, Enum):
    BROAD = "broad"
    NARROW = "narrow"
    EMPTY = "empty"


class BenchmarkActionType(str, Enum):
    ASK = "ask"
    RECOMMEND = "recommend"
    EXPLAIN = "explain"
    PROPOSE_RELAX = "propose_relax"
    STOP = "stop"


class BenchmarkUserAct(str, Enum):
    INFORM = "inform"
    CLARIFY = "clarify"
    REMIND = "remind"
    NO_PREFERENCE = "no_preference"
    ACCEPT = "accept"
    REJECT = "reject"
    REQUEST_EXPLANATION = "request_explanation"
    ACKNOWLEDGE = "acknowledge"
    AGREE_RELAXATION = "agree_relaxation"
    REFUSE_RELAXATION = "refuse_relaxation"
    CORRECT_INVALID_ACTION = "correct_invalid_action"
    ABANDON = "abandon"
    GRACEFUL_END = "graceful_end"


class TransitionTrigger(str, Enum):
    ASK_CONSTRAINT = "ask_constraint"
    ASK_INTERPRETATION = "ask_interpretation"
    UNSATISFACTORY_RECOMMENDATION = "unsatisfactory_recommendation"
    SATISFACTORY_RECOMMENDATION = "satisfactory_recommendation"
    ALLOWED_RELAXATION = "allowed_relaxation"
    TURN_REACHED = "turn_reached"


class CooperationMode(str, Enum):
    DIRECT = "direct"
    PROGRESSIVE = "progressive"
    GUARDED = "guarded"


class FeedbackSpecificity(str, Enum):
    BRIEF = "brief"
    CONSTRAINT = "constraint"
    EXPLANATORY = "explanatory"


class TerminationKind(str, Enum):
    ACCEPTED = "accepted"
    GRACEFUL_NO_SOLUTION = "graceful_no_solution"
    ABANDONED = "abandoned"
    MAX_TURNS = "max_turns"
    ONGOING = "ongoing"


class BenchmarkConstraint(StrictModel):
    constraint_id: str = Field(min_length=1)
    slot: str = Field(min_length=1)
    predicate: ConstraintPredicate
    surface_form: str = Field(min_length=1)
    evidence_kind: str = Field(min_length=1)

    @field_validator("slot", mode="before")
    @classmethod
    def normalize_slot(cls, value: str) -> str:
        return normalize_slot_name(value)


class GoalVersion(StrictModel):
    version: int = Field(ge=0)
    constraints: list[BenchmarkConstraint] = Field(min_length=1)
    acceptable_item_ids: list[str] = Field(default_factory=list)
    feasible_item_count: int = Field(ge=0)
    acceptance_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    @field_validator("acceptable_item_ids")
    @classmethod
    def validate_acceptable_ids(cls, value: list[str]) -> list[str]:
        return _unique(value, field_name="acceptable_item_ids")

    @model_validator(mode="after")
    def validate_goal_version(self) -> GoalVersion:
        constraint_ids = [constraint.constraint_id for constraint in self.constraints]
        slots = [constraint.slot for constraint in self.constraints]
        _unique(constraint_ids, field_name="constraint IDs")
        _unique(slots, field_name="constraint slots")
        if len(self.acceptable_item_ids) > self.feasible_item_count:
            raise ValueError("acceptable item count cannot exceed feasible item count")
        if self.feasible_item_count == 0 and self.acceptable_item_ids:
            raise ValueError("an infeasible goal version cannot have acceptable items")
        return self

    def constraint_ids(self) -> set[str]:
        return {constraint.constraint_id for constraint in self.constraints}


class LatentGoalV1(StrictModel):
    target_type: str = Field(min_length=1)
    initial_version: int = 0
    versions: list[GoalVersion] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_versions(self) -> LatentGoalV1:
        indices = [version.version for version in self.versions]
        if indices != list(range(len(indices))):
            raise ValueError("goal versions must be contiguous and ordered from zero")
        if self.initial_version != 0:
            raise ValueError("v1 benchmark goals must begin at version zero")
        return self

    def version(self, index: int) -> GoalVersion:
        try:
            return self.versions[index]
        except IndexError as exc:
            raise KeyError(f"unknown goal version: {index}") from exc

    def all_constraint_ids(self) -> set[str]:
        return {
            constraint.constraint_id
            for version in self.versions
            for constraint in version.constraints
        }


class AmbiguityOption(StrictModel):
    option_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    constraint_ids: list[str] = Field(default_factory=list)

    @field_validator("constraint_ids")
    @classmethod
    def validate_constraint_ids(cls, value: list[str]) -> list[str]:
        return _unique(value, field_name="ambiguity constraint IDs")


class DisclosurePlan(StrictModel):
    initially_public: list[str] = Field(default_factory=list)
    hidden_order: list[str] = Field(default_factory=list)
    ambiguity_options: list[AmbiguityOption] = Field(default_factory=list)
    selected_ambiguity_option_id: str | None = None

    @field_validator("initially_public", "hidden_order")
    @classmethod
    def validate_unique_constraint_ids(cls, value: list[str]) -> list[str]:
        return _unique(value, field_name="disclosure constraint IDs")

    @model_validator(mode="after")
    def validate_disclosure_plan(self) -> DisclosurePlan:
        if set(self.initially_public).intersection(self.hidden_order):
            raise ValueError("initially public and hidden constraints must be disjoint")
        option_ids = [option.option_id for option in self.ambiguity_options]
        _unique(option_ids, field_name="ambiguity option IDs")
        if self.selected_ambiguity_option_id is not None:
            if self.selected_ambiguity_option_id not in set(option_ids):
                raise ValueError("selected ambiguity option is not registered")
        elif self.ambiguity_options:
            raise ValueError("ambiguity options require one private selected option")
        return self


class TransitionEvent(StrictModel):
    event_id: str = Field(min_length=1)
    trigger: TransitionTrigger
    trigger_ref: str | None = None
    from_goal_version: int = Field(ge=0)
    to_goal_version: int = Field(ge=0)
    reveal_constraint_ids: list[str] = Field(default_factory=list)
    target_phase: DialoguePhaseV1 | None = None
    resulting_stability: IntentStability | None = None
    response_act: BenchmarkUserAct | None = None
    once: bool = True

    @field_validator("reveal_constraint_ids")
    @classmethod
    def validate_reveal_constraint_ids(cls, value: list[str]) -> list[str]:
        return _unique(value, field_name="event reveal constraint IDs")

    @model_validator(mode="after")
    def validate_trigger_ref(self) -> TransitionEvent:
        requires_ref = self.trigger in {
            TransitionTrigger.ASK_CONSTRAINT,
            TransitionTrigger.ASK_INTERPRETATION,
            TransitionTrigger.ALLOWED_RELAXATION,
            TransitionTrigger.TURN_REACHED,
        }
        if requires_ref and not self.trigger_ref:
            raise ValueError(f"{self.trigger.value} requires trigger_ref")
        if not requires_ref and self.trigger_ref:
            raise ValueError(f"{self.trigger.value} does not accept trigger_ref")
        return self


class EvolutionPlan(StrictModel):
    events: list[TransitionEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_events(self) -> EvolutionPlan:
        _unique([event.event_id for event in self.events], field_name="transition event IDs")
        signatures = [
            (
                event.trigger,
                event.trigger_ref,
                event.from_goal_version,
            )
            for event in self.events
        ]
        if len(signatures) != len(set(signatures)):
            raise ValueError("transition events must have unique trigger/ref/from-version signatures")
        return self


class UserBehaviorSpec(StrictModel):
    patience_budget: int = Field(ge=1, le=20)
    question_cost: float = Field(default=0.04, ge=0.0, le=1.0)
    invalid_action_cost: float = Field(default=0.1, ge=0.0, le=1.0)
    cooperation_mode: CooperationMode
    feedback_specificity: FeedbackSpecificity
    language_style_id: str = Field(min_length=1)


class RuntimeUserState(StrictModel):
    turn_index: int = Field(default=0, ge=0)
    goal_version: int = Field(default=0, ge=0)
    phase: DialoguePhaseV1
    completeness: IntentCompleteness
    stability: IntentStability = IntentStability.STABLE
    feasibility: CandidateFeasibility
    disclosed_constraint_ids: list[str] = Field(default_factory=list)
    dismissed_question_targets: list[str] = Field(default_factory=list)
    asked_question_targets: list[str] = Field(default_factory=list)
    rejected_item_ids: list[str] = Field(default_factory=list)
    last_recommended_item_ids: list[str] = Field(default_factory=list)
    pending_acceptable_item_id: str | None = None
    fired_event_ids: list[str] = Field(default_factory=list)
    patience_remaining: int = Field(ge=0, le=20)
    satisfaction: float = Field(default=0.5, ge=0.0, le=1.0)
    last_user_act: BenchmarkUserAct | None = None
    terminal: bool = False
    termination_kind: TerminationKind = TerminationKind.ONGOING

    @field_validator(
        "disclosed_constraint_ids",
        "dismissed_question_targets",
        "asked_question_targets",
        "rejected_item_ids",
        "last_recommended_item_ids",
        "fired_event_ids",
    )
    @classmethod
    def validate_unique_runtime_values(cls, value: list[str]) -> list[str]:
        return _unique(value, field_name="runtime list")

    @model_validator(mode="after")
    def validate_terminal_state(self) -> RuntimeUserState:
        if self.terminal != (self.phase == DialoguePhaseV1.TERMINAL):
            raise ValueError("terminal flag must agree with terminal phase")
        if self.terminal != (self.termination_kind != TerminationKind.ONGOING):
            raise ValueError("termination kind must agree with terminal flag")
        if self.pending_acceptable_item_id is not None:
            if self.pending_acceptable_item_id not in self.last_recommended_item_ids:
                raise ValueError("pending acceptable item must belong to the latest recommendation slate")
        return self


class OracleSpecV1(StrictModel):
    utility_version: str = Field(min_length=1)
    candidate_pool_id: str = Field(min_length=1)


class PrivateInitialUserState(StrictModel):
    simulator_version: str = SIMULATOR_VERSION
    scenario_family: ScenarioFamily
    latent_goal: LatentGoalV1
    disclosure_plan: DisclosurePlan
    evolution_plan: EvolutionPlan
    behavior: UserBehaviorSpec
    runtime_init: RuntimeUserState
    oracle_spec: OracleSpecV1

    @model_validator(mode="after")
    def validate_initial_state(self) -> PrivateInitialUserState:
        if self.simulator_version not in {
            LEGACY_SIMULATOR_VERSION,
            EXPLANATION_GUARD_SIMULATOR_VERSION,
            SIMULATOR_VERSION,
        }:
            raise ValueError(f"unsupported simulator version: {self.simulator_version}")
        initial_goal = self.latent_goal.version(self.latent_goal.initial_version)
        registered = initial_goal.constraint_ids()
        disclosed = set(self.disclosure_plan.initially_public)
        hidden = set(self.disclosure_plan.hidden_order)
        if disclosed.union(hidden) != registered:
            raise ValueError("initial disclosure plan must cover every initial goal constraint exactly once")
        if set(self.runtime_init.disclosed_constraint_ids) != disclosed:
            raise ValueError("runtime initial disclosures must equal disclosure_plan.initially_public")
        if self.runtime_init.goal_version != self.latent_goal.initial_version:
            raise ValueError("runtime must begin at the initial goal version")
        if self.runtime_init.turn_index != 0:
            raise ValueError("runtime initial turn index must be zero")
        if self.runtime_init.patience_remaining != self.behavior.patience_budget:
            raise ValueError("runtime patience must equal the fixed behavior budget")

        all_constraint_ids = self.latent_goal.all_constraint_ids()
        option_constraints = {
            constraint_id
            for option in self.disclosure_plan.ambiguity_options
            for constraint_id in option.constraint_ids
        }
        if not option_constraints.issubset(all_constraint_ids):
            raise ValueError("ambiguity option references an unknown constraint")

        goal_versions = {version.version for version in self.latent_goal.versions}
        event_ids = set()
        for event in self.evolution_plan.events:
            event_ids.add(event.event_id)
            if event.from_goal_version not in goal_versions or event.to_goal_version not in goal_versions:
                raise ValueError("transition event references an unknown goal version")
            if not set(event.reveal_constraint_ids).issubset(all_constraint_ids):
                raise ValueError("transition event reveals an unknown constraint")
        if not set(self.runtime_init.fired_event_ids).issubset(event_ids):
            raise ValueError("runtime contains an unregistered fired event")

        if self.scenario_family == ScenarioFamily.DIRECT and hidden:
            raise ValueError("direct control cases cannot begin with hidden constraints")
        if self.scenario_family == ScenarioFamily.AMBIGUOUS and not self.disclosure_plan.ambiguity_options:
            raise ValueError("ambiguous cases require registered ambiguity options")
        return self


class SourceEvidence(StrictModel):
    pseudonymous_user_id: str = Field(min_length=1)
    source_sequence_sha256: str = Field(min_length=64, max_length=64)
    anchor_item_id: str = Field(min_length=1)
    joined_history_item_ids: list[str] = Field(min_length=1)
    missing_history_item_ids: list[str] = Field(default_factory=list)
    metadata_source_sha256: str = Field(min_length=64, max_length=64)
    user_source_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("joined_history_item_ids", "missing_history_item_ids")
    @classmethod
    def validate_history_ids(cls, value: list[str]) -> list[str]:
        return _unique(value, field_name="source history item IDs")

    @model_validator(mode="after")
    def validate_source_evidence(self) -> SourceEvidence:
        if self.anchor_item_id in set(self.joined_history_item_ids + self.missing_history_item_ids):
            raise ValueError("held-out anchor cannot also appear in source history")
        if set(self.joined_history_item_ids).intersection(self.missing_history_item_ids):
            raise ValueError("joined and missing history IDs must be disjoint")
        return self


class InitialQueryFrame(StrictModel):
    intent_phrase: str = Field(min_length=1)
    public_constraint_ids: list[str] = Field(default_factory=list)
    style_id: str = Field(min_length=1)
    renderer_version: str = Field(min_length=1)

    @field_validator("public_constraint_ids")
    @classmethod
    def validate_public_constraint_ids(cls, value: list[str]) -> list[str]:
        return _unique(value, field_name="query-frame public constraint IDs")


class PublicBenchmarkCase(StrictModel):
    schema_version: str = BENCHMARK_CASE_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    initial_query: str = Field(min_length=1)
    candidate_pool_id: str = Field(min_length=1)
    action_schema_version: str = BENCHMARK_ACTION_SCHEMA_VERSION
    split_group_id: str | None = None

    @model_validator(mode="after")
    def validate_versions(self) -> PublicBenchmarkCase:
        if self.schema_version != BENCHMARK_CASE_SCHEMA_VERSION:
            raise ValueError(f"unsupported case schema: {self.schema_version}")
        if self.action_schema_version != BENCHMARK_ACTION_SCHEMA_VERSION:
            raise ValueError(f"unsupported action schema: {self.action_schema_version}")
        return self


class PrivateBenchmarkCase(StrictModel):
    schema_version: str = BENCHMARK_CASE_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    public_case_sha256: str = Field(min_length=64, max_length=64)
    source_evidence: SourceEvidence
    query_frame: InitialQueryFrame
    initial_state: PrivateInitialUserState

    @model_validator(mode="after")
    def validate_private_case(self) -> PrivateBenchmarkCase:
        if self.schema_version != BENCHMARK_CASE_SCHEMA_VERSION:
            raise ValueError(f"unsupported case schema: {self.schema_version}")
        if set(self.query_frame.public_constraint_ids) != set(
            self.initial_state.disclosure_plan.initially_public
        ):
            raise ValueError("query frame facts must equal initially public constraints")
        return self

    def verify_public_case(self, public_case: PublicBenchmarkCase) -> None:
        if public_case.case_id != self.case_id:
            raise ValueError("public/private case IDs differ")
        if public_case.candidate_pool_id != self.initial_state.oracle_spec.candidate_pool_id:
            raise ValueError("public/private candidate pool IDs differ")
        if benchmark_content_sha256(public_case) != self.public_case_sha256:
            raise ValueError("public case hash does not match private commitment")


@dataclass(frozen=True)
class BenchmarkCasePair:
    public: PublicBenchmarkCase
    private: PrivateBenchmarkCase

    def __post_init__(self) -> None:
        self.private.verify_public_case(self.public)


class EvidenceReference(StrictModel):
    item_id: str = Field(min_length=1)
    catalog_field: str = Field(min_length=1)


class RelaxationProposal(StrictModel):
    constraint_id: str = Field(min_length=1)
    proposed_predicate: ConstraintPredicate


class BenchmarkSystemAction(StrictModel):
    schema_version: str = BENCHMARK_ACTION_SCHEMA_VERSION
    action_type: BenchmarkActionType
    question_target: str | None = None
    item_ids: list[str] = Field(default_factory=list)
    relaxation_proposal: RelaxationProposal | None = None
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    utterance: str = Field(min_length=1)

    @field_validator("item_ids")
    @classmethod
    def validate_item_ids(cls, value: list[str]) -> list[str]:
        return _unique(value, field_name="action item IDs")

    @model_validator(mode="after")
    def validate_payload(self) -> BenchmarkSystemAction:
        if self.schema_version != BENCHMARK_ACTION_SCHEMA_VERSION:
            raise ValueError(f"unsupported action schema: {self.schema_version}")
        if self.action_type == BenchmarkActionType.ASK:
            if not self.question_target or self.item_ids or self.relaxation_proposal or self.evidence_refs:
                raise ValueError("ask requires only question_target and utterance")
        elif self.action_type == BenchmarkActionType.RECOMMEND:
            if not self.item_ids or self.question_target or self.relaxation_proposal:
                raise ValueError("recommend requires item_ids and no question/relaxation")
        elif self.action_type == BenchmarkActionType.EXPLAIN:
            if self.question_target or self.relaxation_proposal:
                raise ValueError("explain cannot contain a question or relaxation")
            if not self.item_ids and not self.evidence_refs:
                raise ValueError("explain requires an item or catalog evidence reference")
        elif self.action_type == BenchmarkActionType.PROPOSE_RELAX:
            if self.relaxation_proposal is None or self.question_target or self.item_ids:
                raise ValueError("propose_relax requires only a relaxation proposal")
        elif self.action_type == BenchmarkActionType.STOP:
            if self.question_target or self.item_ids or self.relaxation_proposal or self.evidence_refs:
                raise ValueError("stop cannot contain action payloads")
        return self


class SemanticResponseFrame(StrictModel):
    user_act: BenchmarkUserAct
    communicative_goal: str = Field(min_length=1)
    authorized_constraint_ids: list[str] = Field(default_factory=list)
    referenced_item_ids: list[str] = Field(default_factory=list)
    style_id: str = Field(min_length=1)
    canonical_text: str = Field(min_length=1)
    surface_text: str | None = Field(default=None, min_length=1)

    @field_validator("authorized_constraint_ids", "referenced_item_ids")
    @classmethod
    def validate_unique_references(cls, value: list[str]) -> list[str]:
        return _unique(value, field_name="response-frame references")

    @property
    def public_text(self) -> str:
        """Return a frozen natural realization when configured, else canonical text."""

        return self.surface_text or self.canonical_text


class PublicEnvironmentResponse(StrictModel):
    utterance: str = Field(min_length=1)
    terminal: bool
    turn_index: int = Field(ge=1)


class RewardBreakdown(StrictModel):
    recommendation_utility: float = Field(default=0.0, ge=0.0, le=1.0)
    acceptance_bonus: float = 0.0
    question_cost: float = Field(default=0.0, ge=0.0)
    invalid_action_cost: float = Field(default=0.0, ge=0.0)
    dialogue_contract_cost: float = Field(default=0.0, ge=0.0)
    hard_violation_cost: float = Field(default=0.0, ge=0.0)
    abandonment_cost: float = Field(default=0.0, ge=0.0)
    total: float

    @model_validator(mode="after")
    def validate_total(self) -> RewardBreakdown:
        expected = (
            self.recommendation_utility
            + self.acceptance_bonus
            - self.question_cost
            - self.invalid_action_cost
            - self.dialogue_contract_cost
            - self.hard_violation_cost
            - self.abandonment_cost
        )
        if abs(self.total - expected) > 1e-9:
            raise ValueError("reward total does not equal its components")
        return self


class BenchmarkTurnTrace(StrictModel):
    schema_version: str = BENCHMARK_TRACE_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    state_before: RuntimeUserState
    system_action: BenchmarkSystemAction
    response_frame: SemanticResponseFrame
    public_user_utterance: str = Field(min_length=1)
    state_after: RuntimeUserState
    fired_event_ids: list[str] = Field(default_factory=list)
    oracle_assessments: list[OracleAssessment] = Field(default_factory=list)
    reward: RewardBreakdown

    @field_validator("fired_event_ids")
    @classmethod
    def validate_fired_event_ids(cls, value: list[str]) -> list[str]:
        return _unique(value, field_name="turn fired event IDs")

    @model_validator(mode="after")
    def validate_turn(self) -> BenchmarkTurnTrace:
        if self.schema_version not in {
            LEGACY_BENCHMARK_TRACE_SCHEMA_VERSION,
            BENCHMARK_TRACE_SCHEMA_VERSION,
        }:
            raise ValueError(f"unsupported trace schema: {self.schema_version}")
        if self.state_before.turn_index != self.turn_index:
            raise ValueError("state_before turn index differs from trace")
        if self.state_after.turn_index != self.turn_index + 1:
            raise ValueError("state_after must advance exactly one turn")
        expected_public_text = (
            self.response_frame.canonical_text
            if self.schema_version == LEGACY_BENCHMARK_TRACE_SCHEMA_VERSION
            else self.response_frame.public_text
        )
        if self.public_user_utterance != expected_public_text:
            raise ValueError("symbolic trace utterance must equal the response-frame public text")
        return self


class BenchmarkEpisodeTrace(StrictModel):
    schema_version: str = BENCHMARK_TRACE_SCHEMA_VERSION
    episode_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    public_case_sha256: str = Field(min_length=64, max_length=64)
    initial_state_sha256: str = Field(min_length=64, max_length=64)
    initial_user_utterance: str | None = Field(default=None, min_length=1)
    language_renderer_id: str | None = Field(default=None, min_length=1)
    runtime_external_calls: bool = False
    turns: list[BenchmarkTurnTrace]
    termination_kind: TerminationKind
    success: bool

    @model_validator(mode="after")
    def validate_episode(self) -> BenchmarkEpisodeTrace:
        if self.schema_version not in {
            LEGACY_BENCHMARK_TRACE_SCHEMA_VERSION,
            BENCHMARK_TRACE_SCHEMA_VERSION,
        }:
            raise ValueError(f"unsupported trace schema: {self.schema_version}")
        if self.schema_version == BENCHMARK_TRACE_SCHEMA_VERSION:
            if self.initial_user_utterance is None:
                raise ValueError("v2 episode trace requires the public initial user utterance")
            if any(turn.schema_version != self.schema_version for turn in self.turns):
                raise ValueError("v2 episode and turn trace versions must agree")
        elif self.initial_user_utterance is not None or self.language_renderer_id is not None:
            raise ValueError("legacy trace cannot claim a v2 language surface")
        if self.runtime_external_calls:
            raise ValueError("benchmark user runtime cannot make external calls")
        if [turn.turn_index for turn in self.turns] != list(range(len(self.turns))):
            raise ValueError("episode turn indices must be contiguous from zero")
        if any(turn.case_id != self.case_id for turn in self.turns):
            raise ValueError("episode contains a turn from another case")
        if self.success != (self.termination_kind == TerminationKind.ACCEPTED):
            raise ValueError("success must agree with accepted termination")
        if self.turns:
            final = self.turns[-1].state_after
            if final.termination_kind != self.termination_kind:
                raise ValueError("episode termination differs from final runtime state")
        return self
