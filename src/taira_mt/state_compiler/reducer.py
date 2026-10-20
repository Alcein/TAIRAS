"""Deterministic reducer for evidence-grounded DSUS v2 state patches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .schema import (
    CONTROL_ENUM_BY_FIELD,
    AddUnresolvedAxisOperation,
    CommitmentStatus,
    ConstraintStatus,
    ControlField,
    FeedbackType,
    InitializeStateOperation,
    InteractionPhase,
    NoopOperation,
    PublicDSUSState,
    RecordFeedbackOperation,
    RejectedItem,
    RemoveConstraintOperation,
    ResolveAxisOperation,
    SemanticConstraint,
    SetControlOperation,
    StatePatch,
    SupersedeConstraintOperation,
    TerminateOperation,
    TerminationReason,
    UnresolvedAxis,
    UpsertConstraintOperation,
)


class TransitionError(ValueError):
    """Raised when a well-formed patch is illegal for the current state."""


def _merge_evidence(current: list[str], added: list[str]) -> list[str]:
    return sorted(set(current).union(added))


def _constraint_by_id(state: PublicDSUSState, constraint_id: str) -> SemanticConstraint | None:
    return next(
        (item for item in state.constraints if item.constraint_id == constraint_id),
        None,
    )


def _axis_by_id(state: PublicDSUSState, axis_id: str) -> UnresolvedAxis | None:
    return next((item for item in state.unresolved_axes if item.axis_id == axis_id), None)


def _rejected_by_id(state: PublicDSUSState, item_id: str) -> RejectedItem | None:
    return next((item for item in state.rejected_items if item.item_id == item_id), None)


def _require_initialized(state: PublicDSUSState, op: str) -> None:
    if not state.initialized:
        raise TransitionError(f"{op} requires an initialized state")


def _set_control(
    state: PublicDSUSState,
    *,
    field: ControlField,
    value: str,
    evidence_ids: list[str],
) -> None:
    if field == ControlField.INTERACTION_PHASE and value == InteractionPhase.TERMINAL.value:
        raise TransitionError("terminal phase can only be entered through terminate")
    enum_value = CONTROL_ENUM_BY_FIELD[field](value)
    setattr(state.control, field.value, enum_value)
    state.control_evidence[field] = list(evidence_ids)


def _apply_operation(state: PublicDSUSState, operation: Any) -> None:
    if isinstance(operation, InitializeStateOperation):
        if state.initialized:
            raise TransitionError("initialize_state cannot be replayed on an initialized state")
        state.initialized = True
        return

    _require_initialized(state, operation.op)

    if isinstance(operation, SetControlOperation):
        _set_control(
            state,
            field=operation.field,
            value=operation.value,
            evidence_ids=operation.evidence_ids,
        )
        return

    if isinstance(operation, UpsertConstraintOperation):
        existing = _constraint_by_id(state, operation.constraint_id)
        if existing is None:
            state.constraints.append(
                SemanticConstraint(
                    constraint_id=operation.constraint_id,
                    name=operation.name,
                    value=operation.value,
                    polarity=operation.polarity,
                    importance=operation.importance,
                    status=operation.status,
                    evidence_ids=operation.evidence_ids,
                )
            )
            return
        if existing.status == ConstraintStatus.SUPERSEDED:
            raise TransitionError(
                f"constraint {operation.constraint_id} is superseded and cannot be reactivated"
            )
        if (
            existing.name != operation.name
            or existing.value != operation.value
            or existing.polarity != operation.polarity
            or existing.importance != operation.importance
        ):
            raise TransitionError(
                "upsert_constraint cannot silently overwrite identity or meaning; "
                "supersede the old constraint and use a new constraint_id"
            )
        existing.status = operation.status
        existing.evidence_ids = _merge_evidence(existing.evidence_ids, operation.evidence_ids)
        return

    if isinstance(operation, SupersedeConstraintOperation):
        existing = _constraint_by_id(state, operation.constraint_id)
        if existing is None:
            raise TransitionError(f"cannot supersede unknown constraint {operation.constraint_id}")
        if existing.status == ConstraintStatus.SUPERSEDED:
            raise TransitionError(f"constraint {operation.constraint_id} is already superseded")
        existing.status = ConstraintStatus.SUPERSEDED
        existing.evidence_ids = _merge_evidence(existing.evidence_ids, operation.evidence_ids)
        return

    if isinstance(operation, RemoveConstraintOperation):
        existing = _constraint_by_id(state, operation.constraint_id)
        if existing is None:
            raise TransitionError(f"cannot remove unknown constraint {operation.constraint_id}")
        state.constraints.remove(existing)
        return

    if isinstance(operation, AddUnresolvedAxisOperation):
        if _axis_by_id(state, operation.axis_id) is not None:
            raise TransitionError(f"unresolved axis {operation.axis_id} already exists")
        state.unresolved_axes.append(
            UnresolvedAxis(
                axis_id=operation.axis_id,
                axis_type=operation.axis_type,
                description=operation.description,
                options=operation.options,
                evidence_ids=operation.evidence_ids,
            )
        )
        return

    if isinstance(operation, ResolveAxisOperation):
        existing = _axis_by_id(state, operation.axis_id)
        if existing is None:
            raise TransitionError(f"cannot resolve unknown axis {operation.axis_id}")
        state.unresolved_axes.remove(existing)
        return

    if isinstance(operation, RecordFeedbackOperation):
        state.control.feedback_type = operation.feedback_type
        state.control_evidence[ControlField.FEEDBACK_TYPE] = list(operation.evidence_ids)
        if operation.feedback_type == FeedbackType.REJECTION and operation.item_id is not None:
            existing = _rejected_by_id(state, operation.item_id)
            if existing is None:
                state.rejected_items.append(
                    RejectedItem(
                        item_id=operation.item_id,
                        reason=operation.reason,
                        evidence_ids=operation.evidence_ids,
                    )
                )
            else:
                existing.evidence_ids = _merge_evidence(
                    existing.evidence_ids, operation.evidence_ids
                )
                if operation.reason is not None:
                    existing.reason = operation.reason
        return

    if isinstance(operation, TerminateOperation):
        state.control.interaction_phase = InteractionPhase.TERMINAL
        state.control_evidence[ControlField.INTERACTION_PHASE] = list(operation.evidence_ids)
        state.termination_reason = operation.reason
        state.termination_evidence_ids = list(operation.evidence_ids)
        if operation.reason == TerminationReason.ACCEPTED:
            state.control.commitment_status = CommitmentStatus.ACCEPTED
            state.control_evidence[ControlField.COMMITMENT_STATUS] = list(operation.evidence_ids)
        elif operation.reason == TerminationReason.ABANDONED:
            state.control.commitment_status = CommitmentStatus.ABANDONED
            state.control_evidence[ControlField.COMMITMENT_STATUS] = list(operation.evidence_ids)
        return

    if isinstance(operation, NoopOperation):
        return

    raise TransitionError(f"unsupported operation type: {type(operation).__name__}")


def reduce_state(
    state: PublicDSUSState | Mapping[str, Any],
    patch: StatePatch | Mapping[str, Any],
) -> PublicDSUSState:
    """Apply one patch without mutating ``state`` and return canonical output.

    The reducer checks optimistic-concurrency revision, global evidence IDs,
    evidence chronology, initialization/terminal boundaries, and entity-level
    transition preconditions.  Pydantic validation handles the closed operation
    vocabulary and operation-local evidence references.
    """

    current = (
        state.model_copy(deep=True)
        if isinstance(state, PublicDSUSState)
        else PublicDSUSState.model_validate(state)
    )
    program = patch if isinstance(patch, StatePatch) else StatePatch.model_validate(patch)

    if program.base_revision != current.revision:
        raise TransitionError(
            f"patch base_revision {program.base_revision} does not match state revision "
            f"{current.revision}"
        )
    if current.control.interaction_phase == InteractionPhase.TERMINAL:
        if not (len(program.operations) == 1 and isinstance(program.operations[0], NoopOperation)):
            raise TransitionError("a terminal state is immutable except for an evidenced noop")
    if not current.initialized and not isinstance(program.operations[0], InitializeStateOperation):
        raise TransitionError("the first patch must begin with initialize_state")
    if current.initialized and any(
        isinstance(operation, InitializeStateOperation) for operation in program.operations
    ):
        raise TransitionError("initialize_state cannot be applied after initialization")

    existing_evidence_ids = {item.evidence_id for item in current.evidence}
    incoming_evidence_ids = {item.evidence_id for item in program.evidence}
    collisions = existing_evidence_ids.intersection(incoming_evidence_ids)
    if collisions:
        raise TransitionError(f"patch reuses existing evidence IDs: {sorted(collisions)}")
    if current.evidence:
        latest_turn = max(item.turn_id for item in current.evidence)
        earliest_incoming = min(item.turn_id for item in program.evidence)
        if earliest_incoming < latest_turn:
            raise TransitionError(
                f"patch evidence turn {earliest_incoming} precedes current turn {latest_turn}"
            )

    current.evidence.extend(item.model_copy(deep=True) for item in program.evidence)
    for index, operation in enumerate(program.operations):
        try:
            _apply_operation(current, operation)
        except TransitionError as exc:
            raise TransitionError(f"operation[{index}] {operation.op}: {exc}") from exc

    current.revision += 1
    return PublicDSUSState.model_validate(current.model_dump(mode="json"))
