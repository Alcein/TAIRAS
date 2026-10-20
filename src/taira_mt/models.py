"""Typed contracts shared by the simulator, recommender, and dialogue policy."""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Reject unknown fields so dataset/schema drift fails loudly."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def normalize_slot_name(value: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", value.strip().casefold())
    if not normalized:
        raise ValueError("slot name cannot be empty")
    return normalized


class ConstraintImportance(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class ConstraintEvidenceStatus(str, Enum):
    """Three-valued catalog evidence for one constraint."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


class ConstraintOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    ANY_OF = "any_of"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    GTE = "gte"
    LTE = "lte"
    BETWEEN = "between"


class IntentRelation(str, Enum):
    AND = "and"
    OR = "or"
    MATCH = "match"
    CONFLICT = "conflict"


class DialoguePhase(str, Enum):
    ELICITATION = "elicitation"
    RECOMMENDATION = "recommendation"
    FEEDBACK = "feedback"
    REPAIR = "repair"
    TERMINAL = "terminal"


class SystemActionType(str, Enum):
    ASK = "ask"
    RECOMMEND = "recommend"
    EXPLAIN = "explain"
    REPAIR = "repair"
    STOP = "stop"


class UserAct(str, Enum):
    INFORM = "inform"
    ACCEPT = "accept"
    REJECT = "reject"
    REQUEST_EXPLANATION = "request_explanation"
    ABANDON = "abandon"
    OTHER = "other"


class TerminationReason(str, Enum):
    ACCEPTED = "accepted"
    ABANDONED = "abandoned"
    NO_CANDIDATES = "no_candidates"
    MAX_TURNS = "max_turns"


class NumericRange(StrictModel):
    minimum: float | None = None
    maximum: float | None = None
    include_minimum: bool = True
    include_maximum: bool = True

    @model_validator(mode="after")
    def validate_range(self) -> NumericRange:
        if self.minimum is None and self.maximum is None:
            raise ValueError("numeric range requires at least one bound")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("numeric range minimum cannot exceed maximum")
        return self


ScalarValue = str | int | float | bool
AttributeValue = ScalarValue | list[ScalarValue]
ConstraintValue = AttributeValue | NumericRange


class ConstraintPredicate(StrictModel):
    """A public or private constraint whose operator semantics are explicit.

    Catalog attributes remain plain values.  Dialogue state stores predicates so
    that, for example, ``price <= 30`` cannot silently degrade into
    ``price == 30``.
    """

    operator: ConstraintOperator = ConstraintOperator.EQ
    value: ConstraintValue
    importance: ConstraintImportance = ConstraintImportance.HARD
    unit: str | None = None

    @model_validator(mode="after")
    def validate_value_for_operator(self) -> ConstraintPredicate:
        if isinstance(self.value, str) and not self.value.strip():
            raise ValueError("constraint string value cannot be empty")
        if isinstance(self.value, list) and not self.value:
            raise ValueError("constraint value list cannot be empty")
        if isinstance(self.value, NumericRange) and self.operator != ConstraintOperator.BETWEEN:
            raise ValueError("NumericRange is only valid with the between operator")
        if self.operator == ConstraintOperator.BETWEEN and not isinstance(self.value, NumericRange):
            raise ValueError("between constraints require NumericRange")
        if self.operator in {ConstraintOperator.GTE, ConstraintOperator.LTE} and (
            isinstance(self.value, bool) or not isinstance(self.value, (int, float))
        ):
            raise ValueError("gte/lte constraints require a numeric scalar")
        if self.operator in {ConstraintOperator.ANY_OF, ConstraintOperator.NOT_IN} and not isinstance(
            self.value, list
        ):
            raise ValueError("any_of/not_in constraints require a value list")
        return self


def _coerce_constraint_predicate(value: Any) -> ConstraintPredicate | dict[str, Any]:
    """Upgrade legacy scalar dialogue constraints to explicit equality predicates."""

    if isinstance(value, ConstraintPredicate):
        return value
    if isinstance(value, NumericRange):
        return {
            "operator": ConstraintOperator.BETWEEN,
            "value": value,
            "importance": ConstraintImportance.HARD,
        }
    if isinstance(value, dict) and "value" in value:
        return value
    return {
        "operator": ConstraintOperator.EQ,
        "value": value,
        "importance": ConstraintImportance.HARD,
    }


PublicConstraint = Annotated[ConstraintPredicate, BeforeValidator(_coerce_constraint_predicate)]


class Constraint(ConstraintPredicate):
    slot: str = Field(min_length=1)
    initially_revealed: bool = False
    target_node_id: str = "root"

    @field_validator("slot", mode="before")
    @classmethod
    def normalize_slot(cls, value: str) -> str:
        return normalize_slot_name(value)

    def as_predicate(self) -> ConstraintPredicate:
        return ConstraintPredicate(
            operator=self.operator,
            value=self.value,
            importance=self.importance,
            unit=self.unit,
        )


class GoalNode(StrictModel):
    node_id: str
    item_type: str
    quantity: int = Field(default=1, ge=1)


class GoalEdge(StrictModel):
    source_node_id: str
    target_node_id: str
    relation: IntentRelation
    attribute: str | None = None


class IntentGraph(StrictModel):
    nodes: list[GoalNode]
    edges: list[GoalEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> IntentGraph:
        node_ids = [node.node_id for node in self.nodes]
        if not node_ids or len(node_ids) != len(set(node_ids)):
            raise ValueError("intent graph requires unique non-empty nodes")
        known = set(node_ids)
        for edge in self.edges:
            if edge.source_node_id not in known or edge.target_node_id not in known:
                raise ValueError("intent edge references an unknown node")
        return self


class HiddenGoal(StrictModel):
    goal_id: str
    user_id: str
    domain: str
    initial_request: str
    constraints: list[Constraint]
    target_item_ids: list[str] = Field(default_factory=list)
    intent_graph: IntentGraph | None = None

    @model_validator(mode="after")
    def validate_constraints(self) -> HiddenGoal:
        keys = [(constraint.target_node_id, constraint.slot) for constraint in self.constraints]
        if len(keys) != len(set(keys)):
            raise ValueError("hidden-goal constraint node/slot pairs must be unique")
        if self.intent_graph is not None:
            node_ids = {node.node_id for node in self.intent_graph.nodes}
            for constraint in self.constraints:
                if constraint.target_node_id not in node_ids:
                    raise ValueError("constraint references an unknown intent node")
        return self

    def constraint_map(self) -> dict[str, Constraint]:
        return {constraint.slot: constraint for constraint in self.constraints}


class UserPersona(StrictModel):
    persona_id: str
    patience_turns: int = Field(ge=1, le=20)
    cooperativeness: float = Field(ge=0.0, le=1.0)
    acceptance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    question_cost: float = Field(default=0.04, ge=0.0, le=1.0)
    misstatement_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    self_correction_probability: float = Field(default=1.0, ge=0.0, le=1.0)


class CatalogItem(StrictModel):
    item_id: str
    title: str
    attributes: dict[str, AttributeValue]

    @field_validator("attributes", mode="before")
    @classmethod
    def normalize_attribute_slots(cls, value: dict[str, AttributeValue]) -> dict[str, AttributeValue]:
        normalized: dict[str, AttributeValue] = {}
        for slot, attribute_value in value.items():
            key = normalize_slot_name(slot)
            if key in normalized:
                raise ValueError(f"duplicate normalized catalog attribute: {key}")
            normalized[key] = attribute_value
        return normalized


class BeliefState(StrictModel):
    turn_index: int = Field(default=0, ge=0)
    phase: DialoguePhase = DialoguePhase.ELICITATION
    known_constraints: dict[str, PublicConstraint] = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list)
    dismissed_slots: list[str] = Field(default_factory=list)
    uncertainty: float = Field(default=1.0, ge=0.0, le=1.0)
    readiness: float = Field(default=0.0, ge=0.0, le=1.0)
    satisfaction_estimate: float = Field(default=0.5, ge=0.0, le=1.0)
    patience_estimate: int = Field(default=5, ge=0)
    available_candidate_count: int = Field(default=0, ge=0)
    last_feedback: UserAct | None = None
    rejected_item_ids: list[str] = Field(default_factory=list)
    legal_actions: list[SystemActionType] = Field(
        default_factory=lambda: [SystemActionType.ASK, SystemActionType.RECOMMEND]
    )

    @field_validator("known_constraints", mode="before")
    @classmethod
    def normalize_known_slots(cls, value: dict[str, Any]) -> dict[str, Any]:
        return {normalize_slot_name(slot): constraint_value for slot, constraint_value in value.items()}

    @field_validator("missing_slots", "dismissed_slots", mode="before")
    @classmethod
    def normalize_missing_slots(cls, value: list[str]) -> list[str]:
        return [normalize_slot_name(slot) for slot in value]

    @model_validator(mode="after")
    def validate_belief(self) -> BeliefState:
        known = {slot.casefold() for slot in self.known_constraints}
        missing = [slot.casefold() for slot in self.missing_slots]
        dismissed = [slot.casefold() for slot in self.dismissed_slots]
        if len(missing) != len(set(missing)):
            raise ValueError("missing_slots must be unique")
        if len(dismissed) != len(set(dismissed)):
            raise ValueError("dismissed_slots must be unique")
        if known.intersection(missing):
            raise ValueError("known_constraints and missing_slots must be disjoint")
        if known.intersection(dismissed) or set(missing).intersection(dismissed):
            raise ValueError("known, missing, and dismissed slots must be mutually disjoint")
        if len(self.legal_actions) != len(set(self.legal_actions)):
            raise ValueError("legal_actions must be unique")
        return self


class SystemAction(StrictModel):
    action_type: SystemActionType
    thought_pattern_id: str
    question_slot: str | None = None
    item_ids: list[str] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("question_slot", mode="before")
    @classmethod
    def normalize_question_slot(cls, value: str | None) -> str | None:
        return None if value is None else normalize_slot_name(value)

    @model_validator(mode="after")
    def validate_payload(self) -> SystemAction:
        if len(self.item_ids) != len(set(self.item_ids)):
            raise ValueError("item_ids must be unique")
        if self.action_type == SystemActionType.ASK:
            if not self.question_slot or self.item_ids:
                raise ValueError("ask actions require only question_slot")
        elif self.action_type in {SystemActionType.RECOMMEND, SystemActionType.REPAIR}:
            if not self.item_ids or self.question_slot:
                raise ValueError("recommend/repair actions require only item_ids")
        elif self.action_type == SystemActionType.STOP:
            if self.question_slot or self.item_ids:
                raise ValueError("stop actions cannot carry question or item payloads")
        return self


class OracleAssessment(StrictModel):
    item_id: str
    normalized_utility: float = Field(ge=0.0, le=1.0)
    matched_slots: list[str] = Field(default_factory=list)
    violated_hard_slots: list[str] = Field(default_factory=list)
    unknown_hard_slots: list[str] = Field(default_factory=list)
    missed_soft_slots: list[str] = Field(default_factory=list)
    unknown_soft_slots: list[str] = Field(default_factory=list)
    constraint_scores: dict[str, float] = Field(default_factory=dict)
    constraint_statuses: dict[str, ConstraintEvidenceStatus] = Field(default_factory=dict)
    hard_satisfied_fraction: float = Field(default=0.0, ge=0.0, le=1.0)


class UserResponse(StrictModel):
    """Simulator output plus evaluator-only latent annotations.

    The policy must consume ``UserObservation`` rather than this model directly.
    """

    act: UserAct
    text: str
    revealed_constraints: dict[str, PublicConstraint] = Field(default_factory=dict)
    dismissed_slots: list[str] = Field(default_factory=list)
    accepted_item_id: str | None = None
    latent_satisfaction: float = Field(ge=0.0, le=1.0)
    terminal: bool = False
    reason: str = ""

    @field_validator("revealed_constraints", mode="before")
    @classmethod
    def normalize_revealed_slots(cls, value: dict[str, Any]) -> dict[str, Any]:
        return {normalize_slot_name(slot): constraint_value for slot, constraint_value in value.items()}

    @field_validator("dismissed_slots", mode="before")
    @classmethod
    def normalize_response_dismissed_slots(cls, value: list[str]) -> list[str]:
        return [normalize_slot_name(slot) for slot in value]

    @model_validator(mode="after")
    def validate_response(self) -> UserResponse:
        if self.act == UserAct.ACCEPT and (not self.accepted_item_id or not self.terminal):
            raise ValueError("accept responses require an accepted item and terminal=true")
        if self.act == UserAct.ABANDON and not self.terminal:
            raise ValueError("abandon responses must be terminal")
        if self.act != UserAct.ACCEPT and self.accepted_item_id:
            raise ValueError("only accept responses may carry accepted_item_id")
        return self


class UserObservation(StrictModel):
    """Information made available to the evaluated policy/state tracker."""

    text: str
    predicted_act: UserAct
    extracted_constraints: dict[str, PublicConstraint] = Field(default_factory=dict)
    dismissed_slots: list[str] = Field(default_factory=list)

    @field_validator("extracted_constraints", mode="before")
    @classmethod
    def normalize_extracted_slots(cls, value: dict[str, Any]) -> dict[str, Any]:
        return {normalize_slot_name(slot): constraint_value for slot, constraint_value in value.items()}

    @field_validator("dismissed_slots", mode="before")
    @classmethod
    def normalize_observation_dismissed_slots(cls, value: list[str]) -> list[str]:
        return [normalize_slot_name(slot) for slot in value]


class EvaluationTrace(StrictModel):
    """Evaluator-only labels that must never enter policy/state-model input."""

    true_user_act: UserAct
    true_revealed_constraints: dict[str, PublicConstraint] = Field(default_factory=dict)
    true_dismissed_slots: list[str] = Field(default_factory=list)
    accepted_item_id: str | None = None
    latent_satisfaction: float = Field(ge=0.0, le=1.0)
    terminal: bool
    reason: str
    oracle_assessments: list[OracleAssessment] = Field(default_factory=list)


class TurnRecord(StrictModel):
    turn_index: int
    belief_before: BeliefState
    matched_thought_pattern_id: str
    system_action: SystemAction
    public_user_text: str
    policy_observation: UserObservation
    belief_after: BeliefState
    reward: float
    evaluation_trace: EvaluationTrace


class DialogueEpisode(StrictModel):
    schema_version: str = "0.1.0"
    episode_id: str
    seed: int
    domain: str
    goal_id: str
    persona_id: str
    public_initial_request: str
    candidate_pool_id: str
    turns: list[TurnRecord]
    termination_reason: TerminationReason
    success: bool
    metrics: dict[str, Any]

    @model_validator(mode="after")
    def validate_episode(self) -> DialogueEpisode:
        indices = [turn.turn_index for turn in self.turns]
        if indices != list(range(len(self.turns))):
            raise ValueError("turn indices must be contiguous from zero")
        if self.success != (self.termination_reason == TerminationReason.ACCEPTED):
            raise ValueError("success must agree with termination_reason")
        if self.metrics.get("turn_count") != len(self.turns):
            raise ValueError("metrics.turn_count must equal the number of turns")
        if self.metrics.get("success") != self.success:
            raise ValueError("metrics.success must agree with success")
        return self
