"""Public DSUS v2 state and typed delta contracts.

The schema deliberately contains only information that can be grounded in the
public dialogue or in a public catalog observation.  It contains no action
label, Thought Pattern identifier, simulator latent variable, or reward.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, JsonValue, field_validator, model_validator

from taira_mt.models import ConstraintImportance, StrictModel, normalize_slot_name
from taira_mt.provenance import canonical_json

STATE_SCHEMA_VERSION = "dsus_public_state_v2"
PATCH_SCHEMA_VERSION = "dsus_state_patch_v2"


def _non_empty_identifier(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("identifier cannot be empty")
    return value


def _non_empty_text(value: str) -> str:
    if not value.strip():
        raise ValueError("text cannot be empty")
    return value


def _unique_sorted(values: list[str], *, field_name: str) -> list[str]:
    normalized = [_non_empty_identifier(value) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must contain unique values")
    return sorted(normalized)


def _canonicalize_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _canonicalize_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_json(item) for item in value]
    return value


def _validate_open_value(value: JsonValue) -> JsonValue:
    if value is None:
        raise ValueError("open semantic value cannot be null")
    if isinstance(value, str) and not value.strip():
        raise ValueError("open semantic string cannot be empty")
    if isinstance(value, (list, dict)) and not value:
        raise ValueError("open semantic collection cannot be empty")
    return _canonicalize_json(value)


class InteractionPhase(str, Enum):
    OPENING = "opening"
    ELICITATION = "elicitation"
    GROUNDING = "grounding"
    RECOMMENDATION = "recommendation"
    FEEDBACK = "feedback"
    REPAIR = "repair"
    TERMINAL = "terminal"


class InformationStatus(str, Enum):
    MISSING = "missing"
    SUFFICIENT = "sufficient"
    AMBIGUOUS = "ambiguous"
    CONFLICTING = "conflicting"


class IntentStability(str, Enum):
    STABLE = "stable"
    STRENGTHENED = "strengthened"
    RELAXED = "relaxed"
    REVISED = "revised"


class FeedbackType(str, Enum):
    NONE = "none"
    ANSWER = "answer"
    ACCEPTANCE = "acceptance"
    REJECTION = "rejection"
    CORRECTION = "correction"
    EXPLANATION_REQUEST = "explanation_request"


class ExplanationStatus(str, Enum):
    NONE = "none"
    REQUESTED = "requested"
    PENDING = "pending"
    RESOLVED = "resolved"


class CommitmentStatus(str, Enum):
    """Evidence-gated auxiliary state; UNKNOWN is the unobserved default."""

    UNKNOWN = "unknown"
    EXPLORING = "exploring"
    COMPARING = "comparing"
    READY = "ready"
    ACCEPTED = "accepted"
    ABANDONED = "abandoned"


class PatienceRisk(str, Enum):
    """Evidence-gated auxiliary state; hidden simulator patience is forbidden."""

    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXHAUSTED = "exhausted"


class ControlField(str, Enum):
    INTERACTION_PHASE = "interaction_phase"
    INFORMATION_STATUS = "information_status"
    INTENT_STABILITY = "intent_stability"
    FEEDBACK_TYPE = "feedback_type"
    EXPLANATION_STATUS = "explanation_status"
    COMMITMENT_STATUS = "commitment_status"
    PATIENCE_RISK = "patience_risk"


MAIN_CONTROL_FIELDS = frozenset(
    {
        ControlField.INTERACTION_PHASE,
        ControlField.INFORMATION_STATUS,
        ControlField.INTENT_STABILITY,
        ControlField.FEEDBACK_TYPE,
        ControlField.EXPLANATION_STATUS,
    }
)
AUXILIARY_CONTROL_FIELDS = frozenset(
    {ControlField.COMMITMENT_STATUS, ControlField.PATIENCE_RISK}
)

CONTROL_ENUM_BY_FIELD: dict[ControlField, type[Enum]] = {
    ControlField.INTERACTION_PHASE: InteractionPhase,
    ControlField.INFORMATION_STATUS: InformationStatus,
    ControlField.INTENT_STABILITY: IntentStability,
    ControlField.FEEDBACK_TYPE: FeedbackType,
    ControlField.EXPLANATION_STATUS: ExplanationStatus,
    ControlField.COMMITMENT_STATUS: CommitmentStatus,
    ControlField.PATIENCE_RISK: PatienceRisk,
}


class EvidenceSource(str, Enum):
    USER = "user"
    SYSTEM = "system"
    CATALOG = "catalog"


class ConstraintPolarity(str, Enum):
    MUST = "must"
    PREFER = "prefer"
    AVOID = "avoid"


class ConstraintStatus(str, Enum):
    ACTIVE = "active"
    UNCERTAIN = "uncertain"
    SUPERSEDED = "superseded"


class TerminationReason(str, Enum):
    ACCEPTED = "accepted"
    ABANDONED = "abandoned"
    NO_SOLUTION = "no_solution"
    USER_ENDED = "user_ended"


class EvidenceReference(StrictModel):
    evidence_id: str = Field(min_length=1)
    turn_id: int = Field(ge=0)
    source: EvidenceSource
    quote: str = Field(min_length=1)

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        return _non_empty_identifier(value)

    @field_validator("quote")
    @classmethod
    def validate_quote(cls, value: str) -> str:
        return _non_empty_text(value)


class ControlState(StrictModel):
    """Five primary axes plus two explicitly auxiliary, evidence-gated axes."""

    interaction_phase: InteractionPhase = InteractionPhase.OPENING
    information_status: InformationStatus = InformationStatus.MISSING
    intent_stability: IntentStability = IntentStability.STABLE
    feedback_type: FeedbackType = FeedbackType.NONE
    explanation_status: ExplanationStatus = ExplanationStatus.NONE
    commitment_status: CommitmentStatus = CommitmentStatus.UNKNOWN
    patience_risk: PatienceRisk = PatienceRisk.UNKNOWN


class SemanticConstraint(StrictModel):
    constraint_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    value: JsonValue
    polarity: ConstraintPolarity = ConstraintPolarity.MUST
    importance: ConstraintImportance = ConstraintImportance.HARD
    status: ConstraintStatus = ConstraintStatus.ACTIVE
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("constraint_id")
    @classmethod
    def validate_constraint_id(cls, value: str) -> str:
        return _non_empty_identifier(value)

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return normalize_slot_name(value)

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: JsonValue) -> JsonValue:
        return _validate_open_value(value)

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        return _unique_sorted(value, field_name="constraint evidence_ids")


class UnresolvedAxis(StrictModel):
    axis_id: str = Field(min_length=1)
    axis_type: str = Field(min_length=1)
    description: str = Field(min_length=1)
    options: list[JsonValue] = Field(default_factory=list)
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("axis_id")
    @classmethod
    def validate_axis_id(cls, value: str) -> str:
        return _non_empty_identifier(value)

    @field_validator("axis_type", mode="before")
    @classmethod
    def normalize_axis_type(cls, value: str) -> str:
        return normalize_slot_name(value)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return _non_empty_text(value)

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: list[JsonValue]) -> list[JsonValue]:
        canonical = [_validate_open_value(item) for item in value]
        serialized = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in canonical]
        if len(serialized) != len(set(serialized)):
            raise ValueError("unresolved-axis options must be unique")
        return canonical

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        return _unique_sorted(value, field_name="unresolved-axis evidence_ids")


class RejectedItem(StrictModel):
    item_id: str = Field(min_length=1)
    reason: str | None = None
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("item_id")
    @classmethod
    def validate_item_id(cls, value: str) -> str:
        return _non_empty_identifier(value)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        return None if value is None else _non_empty_text(value)

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        return _unique_sorted(value, field_name="rejected-item evidence_ids")


class PublicDSUSState(StrictModel):
    schema_version: Literal[STATE_SCHEMA_VERSION] = STATE_SCHEMA_VERSION
    revision: int = Field(default=0, ge=0)
    initialized: bool = False
    control: ControlState = Field(default_factory=ControlState)
    constraints: list[SemanticConstraint] = Field(default_factory=list)
    unresolved_axes: list[UnresolvedAxis] = Field(default_factory=list)
    rejected_items: list[RejectedItem] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    control_evidence: dict[ControlField, list[str]] = Field(default_factory=dict)
    termination_reason: TerminationReason | None = None
    termination_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("termination_evidence_ids")
    @classmethod
    def validate_termination_evidence_ids(cls, value: list[str]) -> list[str]:
        return _unique_sorted(value, field_name="termination evidence_ids")

    @model_validator(mode="after")
    def validate_and_canonicalize(self) -> PublicDSUSState:
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("state evidence IDs must be unique")
        registered_evidence = set(evidence_ids)

        def require_registered(refs: list[str], owner: str) -> None:
            missing = set(refs).difference(registered_evidence)
            if missing:
                raise ValueError(f"{owner} references unknown evidence IDs: {sorted(missing)}")

        constraint_ids = [item.constraint_id for item in self.constraints]
        axis_ids = [item.axis_id for item in self.unresolved_axes]
        rejected_ids = [item.item_id for item in self.rejected_items]
        for values, owner in (
            (constraint_ids, "constraint IDs"),
            (axis_ids, "unresolved-axis IDs"),
            (rejected_ids, "rejected item IDs"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{owner} must be unique")
        for constraint in self.constraints:
            require_registered(constraint.evidence_ids, f"constraint {constraint.constraint_id}")
        for axis in self.unresolved_axes:
            require_registered(axis.evidence_ids, f"unresolved axis {axis.axis_id}")
        for item in self.rejected_items:
            require_registered(item.evidence_ids, f"rejected item {item.item_id}")
        for field, refs in self.control_evidence.items():
            canonical_refs = _unique_sorted(refs, field_name=f"{field.value} evidence_ids")
            self.control_evidence[field] = canonical_refs
            require_registered(canonical_refs, f"control field {field.value}")
        require_registered(self.termination_evidence_ids, "termination")

        if not self.initialized:
            if self.revision != 0:
                raise ValueError("an uninitialized state must have revision zero")
            if self.control != ControlState():
                raise ValueError("an uninitialized state must use default controls")
            if any(
                (
                    self.constraints,
                    self.unresolved_axes,
                    self.rejected_items,
                    self.evidence,
                    self.control_evidence,
                    self.termination_evidence_ids,
                )
            ) or self.termination_reason is not None:
                raise ValueError("an uninitialized state cannot contain public memory")

        is_terminal = self.control.interaction_phase == InteractionPhase.TERMINAL
        if is_terminal != (self.termination_reason is not None):
            raise ValueError("terminal phase and termination_reason must be set together")
        if self.termination_reason is not None and not self.termination_evidence_ids:
            raise ValueError("termination must carry public evidence")

        for field, value in (
            (ControlField.COMMITMENT_STATUS, self.control.commitment_status),
            (ControlField.PATIENCE_RISK, self.control.patience_risk),
        ):
            if value.value != "unknown" and not self.control_evidence.get(field):
                raise ValueError(f"auxiliary control {field.value} requires public evidence")

        self.evidence.sort(key=lambda item: (item.turn_id, item.evidence_id))
        self.constraints.sort(key=lambda item: item.constraint_id)
        self.unresolved_axes.sort(key=lambda item: item.axis_id)
        self.rejected_items.sort(key=lambda item: item.item_id)
        self.control_evidence = {
            field: self.control_evidence[field]
            for field in ControlField
            if field in self.control_evidence
        }
        return self

    @classmethod
    def empty(cls) -> PublicDSUSState:
        return cls()

    def canonical_payload(self) -> dict[str, JsonValue]:
        """Return a JSON-safe payload with canonical collection ordering."""

        return self.model_dump(mode="json")

    def canonical_json(self) -> str:
        return canonical_json(self.canonical_payload())


class EvidenceBackedOperation(StrictModel):
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, value: list[str]) -> list[str]:
        return _unique_sorted(value, field_name="operation evidence_ids")


class InitializeStateOperation(EvidenceBackedOperation):
    op: Literal["initialize_state"] = "initialize_state"


class SetControlOperation(EvidenceBackedOperation):
    op: Literal["set_control"] = "set_control"
    field: ControlField
    value: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_control_value(self) -> SetControlOperation:
        enum_type = CONTROL_ENUM_BY_FIELD[self.field]
        try:
            enum_type(self.value)
        except ValueError as exc:
            allowed = sorted(item.value for item in enum_type)
            raise ValueError(f"invalid {self.field.value} value; expected one of {allowed}") from exc
        return self


class UpsertConstraintOperation(EvidenceBackedOperation):
    op: Literal["upsert_constraint"] = "upsert_constraint"
    constraint_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    value: JsonValue
    polarity: ConstraintPolarity = ConstraintPolarity.MUST
    importance: ConstraintImportance = ConstraintImportance.HARD
    status: ConstraintStatus = ConstraintStatus.ACTIVE

    @field_validator("constraint_id")
    @classmethod
    def validate_constraint_id(cls, value: str) -> str:
        return _non_empty_identifier(value)

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return normalize_slot_name(value)

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: JsonValue) -> JsonValue:
        return _validate_open_value(value)

    @model_validator(mode="after")
    def reject_direct_superseded_insert(self) -> UpsertConstraintOperation:
        if self.status == ConstraintStatus.SUPERSEDED:
            raise ValueError("upsert_constraint cannot create a superseded constraint")
        return self


class SupersedeConstraintOperation(EvidenceBackedOperation):
    op: Literal["supersede_constraint"] = "supersede_constraint"
    constraint_id: str = Field(min_length=1)

    @field_validator("constraint_id")
    @classmethod
    def validate_constraint_id(cls, value: str) -> str:
        return _non_empty_identifier(value)


class RemoveConstraintOperation(EvidenceBackedOperation):
    op: Literal["remove_constraint"] = "remove_constraint"
    constraint_id: str = Field(min_length=1)

    @field_validator("constraint_id")
    @classmethod
    def validate_constraint_id(cls, value: str) -> str:
        return _non_empty_identifier(value)


class AddUnresolvedAxisOperation(EvidenceBackedOperation):
    op: Literal["add_unresolved_axis"] = "add_unresolved_axis"
    axis_id: str = Field(min_length=1)
    axis_type: str = Field(min_length=1)
    description: str = Field(min_length=1)
    options: list[JsonValue] = Field(default_factory=list)

    @field_validator("axis_id")
    @classmethod
    def validate_axis_id(cls, value: str) -> str:
        return _non_empty_identifier(value)

    @field_validator("axis_type", mode="before")
    @classmethod
    def normalize_axis_type(cls, value: str) -> str:
        return normalize_slot_name(value)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return _non_empty_text(value)

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: list[JsonValue]) -> list[JsonValue]:
        return UnresolvedAxis(
            axis_id="validation",
            axis_type="validation",
            description="validation",
            options=value,
            evidence_ids=["validation"],
        ).options


class ResolveAxisOperation(EvidenceBackedOperation):
    op: Literal["resolve_axis"] = "resolve_axis"
    axis_id: str = Field(min_length=1)

    @field_validator("axis_id")
    @classmethod
    def validate_axis_id(cls, value: str) -> str:
        return _non_empty_identifier(value)


class RecordFeedbackOperation(EvidenceBackedOperation):
    op: Literal["record_feedback"] = "record_feedback"
    feedback_type: FeedbackType
    item_id: str | None = None
    reason: str | None = None

    @field_validator("item_id")
    @classmethod
    def validate_item_id(cls, value: str | None) -> str | None:
        return None if value is None else _non_empty_identifier(value)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        return None if value is None else _non_empty_text(value)

    @model_validator(mode="after")
    def validate_feedback_payload(self) -> RecordFeedbackOperation:
        if self.feedback_type == FeedbackType.NONE:
            raise ValueError("record_feedback requires an observed feedback type")
        if self.reason is not None and (
            self.feedback_type != FeedbackType.REJECTION or self.item_id is None
        ):
            raise ValueError("feedback reason is only stored for a rejected item")
        return self


class TerminateOperation(EvidenceBackedOperation):
    op: Literal["terminate"] = "terminate"
    reason: TerminationReason


class NoopOperation(EvidenceBackedOperation):
    op: Literal["noop"] = "noop"
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return _non_empty_text(value)


StateOperation: TypeAlias = Annotated[
    InitializeStateOperation
    | SetControlOperation
    | UpsertConstraintOperation
    | SupersedeConstraintOperation
    | RemoveConstraintOperation
    | AddUnresolvedAxisOperation
    | ResolveAxisOperation
    | RecordFeedbackOperation
    | TerminateOperation
    | NoopOperation,
    Field(discriminator="op"),
]


class StatePatch(StrictModel):
    schema_version: Literal[PATCH_SCHEMA_VERSION] = PATCH_SCHEMA_VERSION
    base_revision: int = Field(ge=0)
    evidence: list[EvidenceReference] = Field(min_length=1)
    operations: list[StateOperation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_program(self) -> StatePatch:
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("patch evidence IDs must be unique")
        registered = set(evidence_ids)
        referenced: set[str] = set()
        for operation in self.operations:
            missing = set(operation.evidence_ids).difference(registered)
            if missing:
                raise ValueError(
                    f"operation {operation.op} references unknown patch evidence IDs: {sorted(missing)}"
                )
            referenced.update(operation.evidence_ids)
        dangling = registered.difference(referenced)
        if dangling:
            raise ValueError(f"patch contains unreferenced evidence IDs: {sorted(dangling)}")

        noop_indices = [index for index, op in enumerate(self.operations) if op.op == "noop"]
        if noop_indices and len(self.operations) != 1:
            raise ValueError("noop must be the only operation in a patch")
        initialize_indices = [
            index for index, op in enumerate(self.operations) if op.op == "initialize_state"
        ]
        if len(initialize_indices) > 1 or (initialize_indices and initialize_indices[0] != 0):
            raise ValueError("initialize_state may appear once and must be the first operation")
        terminate_indices = [
            index for index, op in enumerate(self.operations) if op.op == "terminate"
        ]
        if len(terminate_indices) > 1 or (
            terminate_indices and terminate_indices[0] != len(self.operations) - 1
        ):
            raise ValueError("terminate may appear once and must be the final operation")

        touched_controls: set[ControlField] = set()
        touched_entities: set[tuple[str, str]] = set()
        for operation in self.operations:
            control_field: ControlField | None = None
            entity: tuple[str, str] | None = None
            if isinstance(operation, SetControlOperation):
                control_field = operation.field
            elif isinstance(operation, RecordFeedbackOperation):
                control_field = ControlField.FEEDBACK_TYPE
                if operation.feedback_type == FeedbackType.REJECTION and operation.item_id:
                    entity = ("rejected_item", operation.item_id)
            elif isinstance(operation, TerminateOperation):
                control_field = ControlField.INTERACTION_PHASE
            elif isinstance(
                operation,
                (
                    UpsertConstraintOperation,
                    SupersedeConstraintOperation,
                    RemoveConstraintOperation,
                ),
            ):
                entity = ("constraint", operation.constraint_id)
            elif isinstance(operation, (AddUnresolvedAxisOperation, ResolveAxisOperation)):
                entity = ("unresolved_axis", operation.axis_id)

            if control_field is not None:
                if control_field in touched_controls:
                    raise ValueError(
                        f"control field {control_field.value} is mutated more than once in one patch"
                    )
                touched_controls.add(control_field)
            if entity is not None:
                if entity in touched_entities:
                    raise ValueError(f"entity {entity[0]}:{entity[1]} is mutated more than once")
                touched_entities.add(entity)

        self.evidence.sort(key=lambda item: (item.turn_id, item.evidence_id))
        return self
