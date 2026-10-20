from __future__ import annotations

import pytest
from pydantic import ValidationError

from taira_mt.state_compiler import (
    CommitmentStatus,
    ConstraintImportance,
    ConstraintPolarity,
    ConstraintStatus,
    ControlField,
    EvidenceReference,
    EvidenceSource,
    FeedbackType,
    InformationStatus,
    IntentStability,
    InteractionPhase,
    PublicDSUSState,
    StatePatch,
    TerminationReason,
    TransitionError,
    reduce_state,
)


def _evidence(evidence_id: str, turn_id: int, quote: str) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=evidence_id,
        turn_id=turn_id,
        source=EvidenceSource.USER,
        quote=quote,
    )


def _initial_patch(*, reversed_constraints: bool = False) -> StatePatch:
    constraints = [
        {
            "op": "upsert_constraint",
            "constraint_id": "c-finish-old",
            "name": "finish",
            "value": "glittery",
            "polarity": "prefer",
            "evidence_ids": ["e-query"],
        },
        {
            "op": "upsert_constraint",
            "constraint_id": "c-occasion",
            "name": "occasion",
            "value": "an outdoor summer wedding",
            "polarity": "must",
            "evidence_ids": ["e-query"],
        },
    ]
    if reversed_constraints:
        constraints.reverse()
    return StatePatch.model_validate(
        {
            "base_revision": 0,
            "evidence": [
                {
                    "evidence_id": "e-query",
                    "turn_id": 0,
                    "source": "user",
                    "quote": "I need something glittery for an outdoor summer wedding.",
                }
            ],
            "operations": [
                {"op": "initialize_state", "evidence_ids": ["e-query"]},
                *constraints,
                {
                    "op": "add_unresolved_axis",
                    "axis_id": "a-formality",
                    "axis_type": "style formality",
                    "description": "How formal should the item be?",
                    "options": ["semi-formal", "formal"],
                    "evidence_ids": ["e-query"],
                },
                {
                    "op": "set_control",
                    "field": "information_status",
                    "value": "ambiguous",
                    "evidence_ids": ["e-query"],
                },
            ],
        }
    )


def test_empty_state_exposes_only_five_main_and_unknown_auxiliary_controls() -> None:
    state = PublicDSUSState.empty()
    assert state.initialized is False
    assert state.control.interaction_phase == InteractionPhase.OPENING
    assert state.control.information_status == InformationStatus.MISSING
    assert state.control.intent_stability == IntentStability.STABLE
    assert state.control.feedback_type == FeedbackType.NONE
    assert state.control.explanation_status.value == "none"
    assert state.control.commitment_status == CommitmentStatus.UNKNOWN
    assert state.control.patience_risk.value == "unknown"
    payload = state.canonical_payload()
    assert "optimal_action" not in payload
    assert "pattern_id" not in payload
    assert "satisfaction" not in payload


def test_initial_patch_creates_canonical_evidence_grounded_state() -> None:
    state = reduce_state(PublicDSUSState.empty(), _initial_patch())
    assert state.initialized is True
    assert state.revision == 1
    assert state.control.information_status == InformationStatus.AMBIGUOUS
    assert [item.constraint_id for item in state.constraints] == [
        "c-finish-old",
        "c-occasion",
    ]
    assert state.constraints[0].polarity == ConstraintPolarity.PREFER
    assert state.constraints[0].importance == ConstraintImportance.HARD
    assert [axis.axis_id for axis in state.unresolved_axes] == ["a-formality"]
    assert state.control_evidence[ControlField.INFORMATION_STATUS] == ["e-query"]


def test_canonical_state_does_not_depend_on_independent_operation_order() -> None:
    forward = reduce_state(PublicDSUSState.empty(), _initial_patch())
    reversed_state = reduce_state(
        PublicDSUSState.empty(), _initial_patch(reversed_constraints=True)
    )
    assert forward.canonical_json() == reversed_state.canonical_json()


def test_revision_resolve_supersede_feedback_remove_and_terminate() -> None:
    state = reduce_state(PublicDSUSState.empty(), _initial_patch())
    state = reduce_state(
        state,
        {
            "base_revision": 1,
            "evidence": [
                {
                    "evidence_id": "e-revision",
                    "turn_id": 2,
                    "source": "user",
                    "quote": "Actually, subtle is better; the glitter was too loud.",
                }
            ],
            "operations": [
                {
                    "op": "resolve_axis",
                    "axis_id": "a-formality",
                    "evidence_ids": ["e-revision"],
                },
                {
                    "op": "supersede_constraint",
                    "constraint_id": "c-finish-old",
                    "evidence_ids": ["e-revision"],
                },
                {
                    "op": "upsert_constraint",
                    "constraint_id": "c-finish-new",
                    "name": "finish",
                    "value": "subtle",
                    "polarity": "prefer",
                    "evidence_ids": ["e-revision"],
                },
                {
                    "op": "set_control",
                    "field": "intent_stability",
                    "value": "revised",
                    "evidence_ids": ["e-revision"],
                },
                {
                    "op": "record_feedback",
                    "feedback_type": "correction",
                    "evidence_ids": ["e-revision"],
                },
            ],
        },
    )
    assert state.revision == 2
    assert state.unresolved_axes == []
    assert state.control.intent_stability == IntentStability.REVISED
    assert state.control.feedback_type == FeedbackType.CORRECTION
    assert _constraint(state, "c-finish-old").status == ConstraintStatus.SUPERSEDED
    assert _constraint(state, "c-finish-new").value == "subtle"

    state = reduce_state(
        state,
        {
            "base_revision": 2,
            "evidence": [
                {
                    "evidence_id": "e-reject",
                    "turn_id": 4,
                    "source": "user",
                    "quote": "No, ITEM-9 is still too flashy.",
                }
            ],
            "operations": [
                {
                    "op": "record_feedback",
                    "feedback_type": "rejection",
                    "item_id": "ITEM-9",
                    "reason": "too flashy",
                    "evidence_ids": ["e-reject"],
                },
                {
                    "op": "remove_constraint",
                    "constraint_id": "c-finish-new",
                    "evidence_ids": ["e-reject"],
                },
            ],
        },
    )
    assert [item.item_id for item in state.rejected_items] == ["ITEM-9"]
    assert _constraint_or_none(state, "c-finish-new") is None

    state = reduce_state(
        state,
        {
            "base_revision": 3,
            "evidence": [
                {
                    "evidence_id": "e-accept",
                    "turn_id": 6,
                    "source": "user",
                    "quote": "That option works for me; I'll take it.",
                }
            ],
            "operations": [
                {
                    "op": "record_feedback",
                    "feedback_type": "acceptance",
                    "item_id": "ITEM-10",
                    "evidence_ids": ["e-accept"],
                },
                {
                    "op": "terminate",
                    "reason": "accepted",
                    "evidence_ids": ["e-accept"],
                },
            ],
        },
    )
    assert state.control.interaction_phase == InteractionPhase.TERMINAL
    assert state.control.commitment_status == CommitmentStatus.ACCEPTED
    assert state.termination_reason == TerminationReason.ACCEPTED
    assert state.termination_evidence_ids == ["e-accept"]


def _constraint(state: PublicDSUSState, constraint_id: str):
    value = _constraint_or_none(state, constraint_id)
    assert value is not None
    return value


def _constraint_or_none(state: PublicDSUSState, constraint_id: str):
    return next(
        (item for item in state.constraints if item.constraint_id == constraint_id),
        None,
    )


def test_patch_rejects_unknown_and_dangling_evidence_references() -> None:
    with pytest.raises(ValidationError, match="unknown patch evidence"):
        StatePatch.model_validate(
            {
                "base_revision": 0,
                "evidence": [
                    {
                        "evidence_id": "e-query",
                        "turn_id": 0,
                        "source": "user",
                        "quote": "I need a summer dress.",
                    }
                ],
                "operations": [
                    {"op": "initialize_state", "evidence_ids": ["not-registered"]}
                ],
            }
        )
    with pytest.raises(ValidationError, match="unreferenced evidence"):
        StatePatch.model_validate(
            {
                "base_revision": 0,
                "evidence": [
                    {
                        "evidence_id": "e-query",
                        "turn_id": 0,
                        "source": "user",
                        "quote": "I need a summer dress.",
                    },
                    {
                        "evidence_id": "e-unused",
                        "turn_id": 0,
                        "source": "user",
                        "quote": "This is intentionally unused.",
                    },
                ],
                "operations": [
                    {"op": "initialize_state", "evidence_ids": ["e-query"]}
                ],
            }
        )


def test_auxiliary_control_requires_registered_public_evidence() -> None:
    with pytest.raises(ValidationError, match="auxiliary control commitment_status"):
        PublicDSUSState(
            initialized=True,
            revision=1,
            control={"commitment_status": "ready"},
        )


def test_reducer_enforces_revision_initialization_chronology_and_entity_preconditions() -> None:
    initial = _initial_patch()
    with pytest.raises(TransitionError, match="base_revision"):
        reduce_state(PublicDSUSState.empty(), initial.model_copy(update={"base_revision": 1}))

    with pytest.raises(TransitionError, match="first patch"):
        reduce_state(
            PublicDSUSState.empty(),
            {
                "base_revision": 0,
                "evidence": [_evidence("e1", 0, "Maybe blue.").model_dump(mode="json")],
                "operations": [
                    {
                        "op": "upsert_constraint",
                        "constraint_id": "c1",
                        "name": "color",
                        "value": "blue",
                        "evidence_ids": ["e1"],
                    }
                ],
            },
        )

    state = reduce_state(PublicDSUSState.empty(), initial)
    state = reduce_state(
        state,
        {
            "base_revision": 1,
            "evidence": [_evidence("e-now", 2, "Let me think.").model_dump(mode="json")],
            "operations": [
                {"op": "noop", "reason": "no state change", "evidence_ids": ["e-now"]}
            ],
        },
    )
    with pytest.raises(TransitionError, match="precedes current turn"):
        reduce_state(
            state,
            {
                "base_revision": 2,
                "evidence": [_evidence("e-old", 1, "Old evidence.").model_dump(mode="json")],
                "operations": [{"op": "noop", "reason": "old", "evidence_ids": ["e-old"]}],
            },
        )
    with pytest.raises(TransitionError, match="unknown constraint"):
        reduce_state(
            state,
            {
                "base_revision": 2,
                "evidence": [_evidence("e2", 3, "Drop the red one.").model_dump(mode="json")],
                "operations": [
                    {
                        "op": "remove_constraint",
                        "constraint_id": "missing",
                        "evidence_ids": ["e2"],
                    }
                ],
            },
        )


def test_upsert_cannot_silently_rewrite_constraint_meaning() -> None:
    state = reduce_state(PublicDSUSState.empty(), _initial_patch())
    with pytest.raises(TransitionError, match="cannot silently overwrite"):
        reduce_state(
            state,
            {
                "base_revision": 1,
                "evidence": [_evidence("e2", 2, "Now I want matte.").model_dump(mode="json")],
                "operations": [
                    {
                        "op": "upsert_constraint",
                        "constraint_id": "c-finish-old",
                        "name": "finish",
                        "value": "matte",
                        "polarity": "prefer",
                        "evidence_ids": ["e2"],
                    }
                ],
            },
        )

    with pytest.raises(TransitionError, match="cannot silently overwrite"):
        reduce_state(
            state,
            {
                "base_revision": 1,
                "evidence": [
                    _evidence("e3", 2, "That is only a preference.").model_dump(mode="json")
                ],
                "operations": [
                    {
                        "op": "upsert_constraint",
                        "constraint_id": "c-finish-old",
                        "name": "finish",
                        "value": "glittery",
                        "polarity": "prefer",
                        "importance": "soft",
                        "evidence_ids": ["e3"],
                    }
                ],
            },
        )


def test_terminal_state_rejects_mutation_but_accepts_evidenced_noop() -> None:
    state = reduce_state(PublicDSUSState.empty(), _initial_patch())
    terminal = reduce_state(
        state,
        {
            "base_revision": 1,
            "evidence": [_evidence("e-end", 2, "I am done.").model_dump(mode="json")],
            "operations": [
                {"op": "terminate", "reason": "user_ended", "evidence_ids": ["e-end"]}
            ],
        },
    )
    with pytest.raises(TransitionError, match="terminal state"):
        reduce_state(
            terminal,
            {
                "base_revision": 2,
                "evidence": [_evidence("e-late", 3, "Maybe blue.").model_dump(mode="json")],
                "operations": [
                    {
                        "op": "upsert_constraint",
                        "constraint_id": "c-blue",
                        "name": "color",
                        "value": "blue",
                        "evidence_ids": ["e-late"],
                    }
                ],
            },
        )
    replayed = reduce_state(
        terminal,
        {
            "base_revision": 2,
            "evidence": [_evidence("e-ack", 3, "Okay.").model_dump(mode="json")],
            "operations": [
                {"op": "noop", "reason": "post-terminal acknowledgement", "evidence_ids": ["e-ack"]}
            ],
        },
    )
    assert replayed.revision == 3
    assert replayed.termination_reason == TerminationReason.USER_ENDED


def test_patch_rejects_ambiguous_programs() -> None:
    evidence = [_evidence("e1", 0, "I need something subtle.").model_dump(mode="json")]
    with pytest.raises(ValidationError, match="noop must be the only"):
        StatePatch.model_validate(
            {
                "base_revision": 0,
                "evidence": evidence,
                "operations": [
                    {"op": "initialize_state", "evidence_ids": ["e1"]},
                    {"op": "noop", "reason": "invalid mixed patch", "evidence_ids": ["e1"]},
                ],
            }
        )
    with pytest.raises(ValidationError, match="mutated more than once"):
        StatePatch.model_validate(
            {
                "base_revision": 0,
                "evidence": evidence,
                "operations": [
                    {"op": "initialize_state", "evidence_ids": ["e1"]},
                    {
                        "op": "set_control",
                        "field": "information_status",
                        "value": "missing",
                        "evidence_ids": ["e1"],
                    },
                    {
                        "op": "set_control",
                        "field": "information_status",
                        "value": "sufficient",
                        "evidence_ids": ["e1"],
                    },
                ],
            }
        )
