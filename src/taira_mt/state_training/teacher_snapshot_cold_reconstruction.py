"""Compile full-history teacher snapshots into reducer-verified recovery patches.

This bridge is deliberately separate from :mod:`teacher_snapshot_differ`.
The adjacent differ asks whether one newly observed turn can explain a legal
incremental delta.  Here every sample is a *cold reconstruction*: the model is
given the complete causal public history and ``PublicDSUSState.empty()`` and
must rebuild the current state in one patch.  Consequently, these records are
runtime-compatible recovery supervision, not adjacent dialogue transitions.

The production ``StatePatch`` language can represent most current states from
EMPTY.  It cannot represent multiple historical rejected items while mutating
the singleton ``feedback_type`` control only once, nor can it both recreate a
historical rejection and set a different latest feedback value in one patch.
Such snapshots are retained in a representation-limit ledger rather than
silently simplified.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json
from taira_mt.state_compiler import (
    AddUnresolvedAxisOperation,
    ControlField,
    ControlState,
    FeedbackType,
    InitializeStateOperation,
    PublicDSUSState,
    RecordFeedbackOperation,
    SetControlOperation,
    StateOperation,
    StatePatch,
    TerminateOperation,
    UpsertConstraintOperation,
    reduce_state,
)
from taira_mt.state_compiler.runtime import CompilerPatchOutput, validate_public_evidence
from taira_mt.state_training.dataset_assembly import SFTTokenizerRecord
from taira_mt.state_training.teacher_snapshot_differ import snapshot_decision_sha256
from taira_mt.state_training.teacher_snapshot_freeze import FullSnapshotSFTRecord
from taira_mt.state_training.training_views import DSUSPublicPrompt

COLD_RECONSTRUCTION_VERSION = "dsus_teacher_snapshot_cold_reconstruction_v1"
COLD_RECONSTRUCTION_LIMIT_VERSION = "dsus_teacher_snapshot_cold_limit_v1"
COLD_RECONSTRUCTION_SUMMARY_VERSION = "dsus_teacher_snapshot_cold_summary_v1"

RepresentationLimitCode = Literal[
    "multiple_rejected_items_not_representable_in_one_patch",
    "historical_rejection_with_non_rejection_latest_feedback",
]


class ColdReconstructionTransitionRecord(StrictModel):
    """One full-history recovery target verified with the production reducer."""

    schema_version: Literal["dsus_teacher_snapshot_cold_reconstruction_v1"] = (
        COLD_RECONSTRUCTION_VERSION
    )
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    order_index: int = Field(ge=0)
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trajectory_id: str = Field(min_length=1)
    source_turn_index: int = Field(ge=0)
    split: Literal["train", "dev", "test"]
    previous_state: PublicDSUSState
    public_prompt: DSUSPublicPrompt
    gold_patch: StatePatch
    replayed_state: PublicDSUSState
    teacher_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_verified: Literal[True] = True
    representation_normalizations: list[str] = Field(default_factory=list)
    reconstruction_kind: Literal["empty_to_full_history_current_state"] = (
        "empty_to_full_history_current_state"
    )

    @model_validator(mode="after")
    def validate_replay(self) -> ColdReconstructionTransitionRecord:
        if self.previous_state != PublicDSUSState.empty():
            raise ValueError("cold reconstruction must start from EMPTY")
        if self.target_decision_sha256 != self.replay_decision_sha256:
            raise ValueError("cold reconstruction replay is not decision-equivalent")
        replayed = reduce_state(self.previous_state, self.gold_patch)
        if replayed.canonical_json() != self.replayed_state.canonical_json():
            raise ValueError("stored replayed state differs from production reducer output")
        return self


class ColdReconstructionRepresentationLimit(StrictModel):
    """A valid full snapshot that cannot be encoded as one recovery patch."""

    schema_version: Literal["dsus_teacher_snapshot_cold_limit_v1"] = (
        COLD_RECONSTRUCTION_LIMIT_VERSION
    )
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    order_index: int = Field(ge=0)
    trajectory_id: str = Field(min_length=1)
    source_turn_index: int = Field(ge=0)
    split: Literal["train", "dev", "test"]
    reason: RepresentationLimitCode
    rejected_item_count: int = Field(ge=0)
    latest_feedback_type: str = Field(min_length=1)
    full_snapshot_retained: Literal[True] = True
    adjacent_transition_status_unchanged: Literal[True] = True


class ColdReconstructionSummary(StrictModel):
    schema_version: Literal["dsus_teacher_snapshot_cold_summary_v1"] = (
        COLD_RECONSTRUCTION_SUMMARY_VERSION
    )
    input_full_snapshot_count: int = Field(ge=1)
    compiled_cold_reconstruction_count: int = Field(ge=0)
    representation_limited_count: int = Field(ge=0)
    replay_verified_count: int = Field(ge=0)
    compiled_by_split: dict[str, int]
    representation_limited_by_split: dict[str, int]
    compiled_by_source_turn: dict[str, int]
    representation_limit_reason_counts: dict[str, int]
    operation_counts: dict[str, int]
    representation_normalization_counts: dict[str, int]
    tokenizer_record_counts: dict[str, int]
    max_seq_length_contract: Literal[4096] = 4096
    tokenizer_audited: bool = False
    token_length_reports: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    over_limit_count: int = Field(default=0, ge=0)
    tokenizer_overlength_handling: Literal[
        "fail_build",
        "exclude_from_tokenizer_sft_retain_transition",
    ] = "fail_build"
    tokenizer_overlength_excluded_by_split: dict[str, int] = Field(default_factory=dict)
    tokenizer_overlength_exclusion_ledger_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    full_transitions_retained_after_token_filter: Literal[True] = True
    maximum_observed_tokens: int | None = Field(default=None, ge=0)
    adjacent_compiled_count: int | None = Field(default=None, ge=0)
    adjacent_non_opening_compiled_count: int | None = Field(default=None, ge=0)
    adjacent_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    adjacent_snapshot_differ_is_separate: Literal[True] = True
    adjacent_artifact_modified: Literal[False] = False
    external_model_calls: Literal[False] = False
    gpu_used: Literal[False] = False
    fabricated_count: Literal[0] = 0

    @model_validator(mode="after")
    def validate_counts(self) -> ColdReconstructionSummary:
        if self.input_full_snapshot_count != (
            self.compiled_cold_reconstruction_count + self.representation_limited_count
        ):
            raise ValueError("compiled and limited rows do not cover full snapshot input")
        if self.replay_verified_count != self.compiled_cold_reconstruction_count:
            raise ValueError("every released cold reconstruction must be replay verified")
        if sum(self.compiled_by_split.values()) != self.compiled_cold_reconstruction_count:
            raise ValueError("compiled split counts disagree")
        if sum(self.representation_limited_by_split.values()) != self.representation_limited_count:
            raise ValueError("limited split counts disagree")
        if sum(self.tokenizer_overlength_excluded_by_split.values()) != self.over_limit_count:
            raise ValueError("overlength exclusion split counts disagree")
        expected_tokenizer_count = self.compiled_cold_reconstruction_count - self.over_limit_count
        if expected_tokenizer_count < 0:
            raise ValueError("overlength exclusion count exceeds compiled transitions")
        if sum(self.tokenizer_record_counts.values()) != expected_tokenizer_count:
            raise ValueError("tokenizer split counts disagree")
        if self.tokenizer_overlength_handling == "fail_build":
            if self.over_limit_count:
                raise ValueError("fail-build summary cannot release overlength samples")
            if self.tokenizer_overlength_exclusion_ledger_sha256 is not None:
                raise ValueError("fail-build summary cannot bind an exclusion ledger")
        elif self.tokenizer_overlength_exclusion_ledger_sha256 is None:
            raise ValueError("overlength exclusion policy must bind its private ledger")
        if self.tokenizer_audited:
            if set(self.token_length_reports) != {"train", "dev", "test"}:
                raise ValueError("token audit must cover train/dev/test")
            if self.maximum_observed_tokens is None:
                raise ValueError("token audit must report a maximum")
        return self


@dataclass(frozen=True)
class ColdReconstructionBuildResult:
    transitions: tuple[ColdReconstructionTransitionRecord, ...]
    representation_limits: tuple[ColdReconstructionRepresentationLimit, ...]
    tokenizer_records_by_split: dict[str, tuple[SFTTokenizerRecord, ...]]
    summary: ColdReconstructionSummary


class ColdReconstructionRepresentationError(ValueError):
    def __init__(self, code: RepresentationLimitCode) -> None:
        super().__init__(code)
        self.code = code


def _normalized_target_state(record: FullSnapshotSFTRecord) -> PublicDSUSState:
    """Canonicalize evidence storage implied deterministically by reducer operations."""

    state = record.target.to_public_state(revision=1)
    payload = state.model_dump(mode="json")
    control_evidence = {
        field.value if isinstance(field, ControlField) else str(field): refs
        for field, refs in state.control_evidence.items()
        if refs
    }
    if state.termination_reason is not None:
        control_evidence[ControlField.INTERACTION_PHASE.value] = list(
            state.termination_evidence_ids
        )
        if state.termination_reason.value in {"accepted", "abandoned"}:
            control_evidence[ControlField.COMMITMENT_STATUS.value] = list(
                state.termination_evidence_ids
            )
    if len(state.rejected_items) == 1 and state.control.feedback_type == FeedbackType.REJECTION:
        control_evidence[ControlField.FEEDBACK_TYPE.value] = list(
            state.rejected_items[0].evidence_ids
        )
    payload["control_evidence"] = control_evidence
    return PublicDSUSState.model_validate(payload)


def _normalization_codes(record: FullSnapshotSFTRecord) -> list[str]:
    codes: list[str] = []
    if any(not refs for refs in record.target.control_evidence.values()):
        codes.append("drop_empty_control_evidence_entries")
    if record.target.termination_reason is not None:
        phase_refs = record.target.control_evidence.get(ControlField.INTERACTION_PHASE, [])
        if phase_refs != record.target.termination_evidence_ids:
            codes.append("canonicalize_terminal_phase_evidence_to_termination")
        if record.target.termination_reason.value in {"accepted", "abandoned"}:
            commitment_refs = record.target.control_evidence.get(
                ControlField.COMMITMENT_STATUS,
                [],
            )
            if commitment_refs != record.target.termination_evidence_ids:
                codes.append("canonicalize_terminal_commitment_evidence_to_termination")
    if (
        len(record.target.rejected_items) == 1
        and record.target.control.feedback_type == FeedbackType.REJECTION
        and record.target.control_evidence.get(ControlField.FEEDBACK_TYPE, [])
        != record.target.rejected_items[0].evidence_ids
    ):
        codes.append("canonicalize_rejection_control_evidence_to_rejected_item")
    return codes


def compile_full_snapshot_cold_patch(
    record: FullSnapshotSFTRecord,
) -> tuple[StatePatch, PublicDSUSState]:
    """Compile one full current state from EMPTY or raise a stable limit code."""

    if record.input.previous_public_state != PublicDSUSState.empty():
        raise ValueError("cold full-snapshot supervision must expose EMPTY previous state")
    validate_public_evidence(record.target.evidence, prompt=record.input)
    target = _normalized_target_state(record)
    if len(target.rejected_items) > 1:
        raise ColdReconstructionRepresentationError(
            "multiple_rejected_items_not_representable_in_one_patch"
        )
    if target.rejected_items and target.control.feedback_type != FeedbackType.REJECTION:
        raise ColdReconstructionRepresentationError(
            "historical_rejection_with_non_rejection_latest_feedback"
        )

    operations: list[StateOperation] = [
        InitializeStateOperation(
            evidence_ids=[reference.evidence_id for reference in target.evidence]
        )
    ]
    operations.extend(
        UpsertConstraintOperation(
            constraint_id=item.constraint_id,
            name=item.name,
            value=item.value,
            polarity=item.polarity,
            importance=item.importance,
            status=item.status,
            evidence_ids=item.evidence_ids,
        )
        for item in target.constraints
    )
    operations.extend(
        AddUnresolvedAxisOperation(
            axis_id=item.axis_id,
            axis_type=item.axis_type,
            description=item.description,
            options=item.options,
            evidence_ids=item.evidence_ids,
        )
        for item in target.unresolved_axes
    )

    if target.rejected_items:
        item = target.rejected_items[0]
        operations.append(
            RecordFeedbackOperation(
                feedback_type=FeedbackType.REJECTION,
                item_id=item.item_id,
                reason=item.reason,
                evidence_ids=item.evidence_ids,
            )
        )
    elif target.control.feedback_type != FeedbackType.NONE:
        operations.append(
            RecordFeedbackOperation(
                feedback_type=target.control.feedback_type,
                evidence_ids=target.control_evidence[ControlField.FEEDBACK_TYPE],
            )
        )

    defaults = ControlState()
    terminal = target.termination_reason is not None
    terminal_sets_commitment = terminal and target.termination_reason.value in {
        "accepted",
        "abandoned",
    }
    for field in ControlField:
        if field == ControlField.FEEDBACK_TYPE:
            continue
        if terminal and field == ControlField.INTERACTION_PHASE:
            continue
        if terminal_sets_commitment and field == ControlField.COMMITMENT_STATUS:
            continue
        value = getattr(target.control, field.value)
        evidence_ids = target.control_evidence.get(field, [])
        if value == getattr(defaults, field.value) and not evidence_ids:
            continue
        if not evidence_ids:
            raise ValueError(f"non-default control lacks evidence: {field.value}")
        operations.append(
            SetControlOperation(
                field=field,
                value=value.value,
                evidence_ids=evidence_ids,
            )
        )

    if terminal:
        if not target.termination_evidence_ids:
            raise ValueError("terminal full snapshot lacks evidence")
        operations.append(
            TerminateOperation(
                reason=target.termination_reason,
                evidence_ids=target.termination_evidence_ids,
            )
        )

    patch = StatePatch(
        base_revision=0,
        evidence=[item.model_copy(deep=True) for item in target.evidence],
        operations=operations,
    )
    replayed = reduce_state(PublicDSUSState.empty(), patch)
    if replayed.canonical_json() != target.canonical_json():
        raise ValueError("production reducer replay differs from normalized full snapshot")
    return patch, replayed


def build_cold_reconstruction_views(
    records: Sequence[FullSnapshotSFTRecord],
) -> ColdReconstructionBuildResult:
    """Build paired runtime-compatible recovery views without any external call."""

    if not records:
        raise ValueError("full snapshot SFT input is empty")
    if len({item.sample_id for item in records}) != len(records):
        raise ValueError("full snapshot SFT input repeats sample IDs")

    transitions: list[ColdReconstructionTransitionRecord] = []
    limited: list[ColdReconstructionRepresentationLimit] = []
    tokenizer: dict[str, list[SFTTokenizerRecord]] = {
        "train": [],
        "dev": [],
        "test": [],
    }
    operation_counts: Counter[str] = Counter()
    normalization_counts: Counter[str] = Counter()
    for record in sorted(records, key=lambda item: item.order_index):
        try:
            patch, replayed = compile_full_snapshot_cold_patch(record)
        except ColdReconstructionRepresentationError as exc:
            limited.append(
                ColdReconstructionRepresentationLimit(
                    sample_id=record.sample_id,
                    order_index=record.order_index,
                    trajectory_id=record.trajectory_id,
                    source_turn_index=record.source_turn_index,
                    split=record.split,
                    reason=exc.code,
                    rejected_item_count=len(record.target.rejected_items),
                    latest_feedback_type=record.target.control.feedback_type.value,
                )
            )
            continue

        target_signature = snapshot_decision_sha256(record.target)
        replay_signature = snapshot_decision_sha256(replayed)
        transition = ColdReconstructionTransitionRecord(
            sample_id=record.sample_id,
            order_index=record.order_index,
            public_prompt_sha256=record.public_prompt_sha256,
            trajectory_id=record.trajectory_id,
            source_turn_index=record.source_turn_index,
            split=record.split,
            previous_state=PublicDSUSState.empty(),
            public_prompt=record.input.model_copy(deep=True),
            gold_patch=patch,
            replayed_state=replayed,
            teacher_snapshot_sha256=record.target_sha256,
            target_decision_sha256=target_signature,
            replay_decision_sha256=replay_signature,
            representation_normalizations=_normalization_codes(record),
        )
        transitions.append(transition)
        operation_counts.update(operation.op for operation in patch.operations)
        normalization_counts.update(transition.representation_normalizations)
        tokenizer[record.split].append(
            SFTTokenizerRecord(
                sample_id=record.sample_id,
                public_prompt_sha256=record.public_prompt_sha256,
                model_input=record.input.canonical_json(),
                assistant_target=_runtime_patch_json(patch),
            )
        )

    compiled_by_split = Counter(item.split for item in transitions)
    limited_by_split = Counter(item.split for item in limited)
    compiled_by_turn = Counter(str(item.source_turn_index) for item in transitions)
    reason_counts = Counter(item.reason for item in limited)
    summary = ColdReconstructionSummary(
        input_full_snapshot_count=len(records),
        compiled_cold_reconstruction_count=len(transitions),
        representation_limited_count=len(limited),
        replay_verified_count=len(transitions),
        compiled_by_split=dict(sorted(compiled_by_split.items())),
        representation_limited_by_split=dict(sorted(limited_by_split.items())),
        compiled_by_source_turn=dict(sorted(compiled_by_turn.items())),
        representation_limit_reason_counts=dict(sorted(reason_counts.items())),
        operation_counts=dict(sorted(operation_counts.items())),
        representation_normalization_counts=dict(sorted(normalization_counts.items())),
        tokenizer_record_counts={split: len(rows) for split, rows in tokenizer.items()},
    )
    return ColdReconstructionBuildResult(
        transitions=tuple(transitions),
        representation_limits=tuple(limited),
        tokenizer_records_by_split={
            split: tuple(rows) for split, rows in tokenizer.items()
        },
        summary=summary,
    )


def _runtime_patch_json(patch: StatePatch) -> str:
    output = CompilerPatchOutput(
        schema_version=patch.schema_version,
        base_revision=patch.base_revision,
        evidence=[item.model_copy(deep=True) for item in patch.evidence],
        delta=[item.model_copy(deep=True) for item in patch.operations],
    )
    if output.as_patch() != patch:
        raise ValueError("runtime CompilerPatchOutput round-trip changed the cold patch")
    return canonical_json(output.model_dump(mode="json"))
__all__ = [
    "COLD_RECONSTRUCTION_LIMIT_VERSION",
    "COLD_RECONSTRUCTION_SUMMARY_VERSION",
    "COLD_RECONSTRUCTION_VERSION",
    "ColdReconstructionBuildResult",
    "ColdReconstructionRepresentationError",
    "ColdReconstructionRepresentationLimit",
    "ColdReconstructionSummary",
    "ColdReconstructionTransitionRecord",
    "build_cold_reconstruction_views",
    "compile_full_snapshot_cold_patch",
]
