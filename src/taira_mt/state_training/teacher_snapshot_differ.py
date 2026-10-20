"""Deterministic offline differ from accepted teacher snapshots to DSUS patches.

Strong teachers cold-project a complete *current* state for every public prefix.
Their local evidence/entity IDs are intentionally not assumed to be stable across
prefixes.  This module gives the offline data pipeline a fail-closed bridge to
the unchanged production :class:`~taira_mt.state_compiler.StatePatch` contract:

* exact semantic identities receive deterministic, teacher-independent IDs;
* every emitted operation cites evidence newly observed at the current source
  turn (never an earlier quotation recycled by a later snapshot);
* an adjacent pair is rejected when the patch language cannot express it in one
  legal step or when semantic alignment is not unique; and
* every candidate patch is replayed with the production reducer and compared to
  the target on decision-relevant fields before it is released.

The module is offline-only and contains no provider/API dependency.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, JsonValue, ValidationError, model_validator

from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.state_compiler import (
    AddUnresolvedAxisOperation,
    CommitmentStatus,
    ConstraintStatus,
    ControlField,
    EvidenceReference,
    FeedbackType,
    InitializeStateOperation,
    InteractionPhase,
    NoopOperation,
    PublicDSUSState,
    RecordFeedbackOperation,
    RemoveConstraintOperation,
    ResolveAxisOperation,
    SemanticConstraint,
    SetControlOperation,
    StateOperation,
    StatePatch,
    SupersedeConstraintOperation,
    TerminateOperation,
    TerminationReason,
    TransitionError,
    UnresolvedAxis,
    UpsertConstraintOperation,
    reduce_state,
)
from taira_mt.state_training.teacher_projection import TeacherCurrentStateSnapshot
from taira_mt.state_training.teacher_projection_consensus import TeacherConsensusRecord
from taira_mt.state_training.teacher_snapshot_freeze import FrozenTeacherSnapshotRecord

TEACHER_SNAPSHOT_DIFFER_VERSION = "dsus_teacher_snapshot_transition_v1"
TEACHER_SNAPSHOT_DIFFER_SUMMARY_VERSION = "dsus_teacher_snapshot_transition_summary_v1"

TeacherSnapshotSourceRecord = TeacherConsensusRecord | FrozenTeacherSnapshotRecord


class SnapshotDiffReason(StrictModel):
    code: str = Field(min_length=1)
    detail: dict[str, JsonValue] = Field(default_factory=dict)


class TeacherSnapshotTransitionRecord(StrictModel):
    """One accepted prefix converted to a replay-verified patch or a reason."""

    schema_version: Literal[TEACHER_SNAPSHOT_DIFFER_VERSION] = TEACHER_SNAPSHOT_DIFFER_VERSION
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    trajectory_id: str = Field(min_length=1)
    source_turn_index: int = Field(ge=0)
    predecessor_sample_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    status: Literal["compiled", "uncompilable"]
    patch: StatePatch | None = None
    replayed_state: PublicDSUSState | None = None
    target_decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_decision_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reasons: list[SnapshotDiffReason] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_outcome(self) -> TeacherSnapshotTransitionRecord:
        released = self.patch is not None and self.replayed_state is not None
        if (self.status == "compiled") != released:
            raise ValueError("only a compiled transition may expose patch and replayed_state")
        if (self.status == "compiled") != (self.replay_decision_sha256 is not None):
            raise ValueError("compiled transition requires a replay decision signature")
        if self.status == "compiled" and self.target_decision_sha256 != self.replay_decision_sha256:
            raise ValueError("a compiled transition must be decision-equivalent after replay")
        return self


class TeacherSnapshotDifferSummary(StrictModel):
    schema_version: Literal[TEACHER_SNAPSHOT_DIFFER_SUMMARY_VERSION] = (
        TEACHER_SNAPSHOT_DIFFER_SUMMARY_VERSION
    )
    input_record_count: int = Field(ge=1)
    trajectory_count: int = Field(ge=1)
    compiled_count: int = Field(ge=0)
    uncompilable_count: int = Field(ge=0)
    status_counts: dict[str, int]
    reason_counts: dict[str, int]
    operation_counts: dict[str, int]
    replay_verified_count: int = Field(ge=0)
    external_model_calls: Literal[False] = False


class SnapshotDiffUncompilable(ValueError):
    """Expected fail-closed outcome, carrying a stable machine-readable code."""

    def __init__(self, code: str, **detail: JsonValue) -> None:
        super().__init__(code)
        self.reason = SnapshotDiffReason(code=code, detail=detail)


def _evidence_key(reference: EvidenceReference) -> str:
    return canonical_json(
        {
            "turn_id": reference.turn_id,
            "source": reference.source.value,
            "quote": reference.quote,
        }
    )


def _constraint_identity(item: SemanticConstraint) -> str:
    return canonical_json(
        {
            "name": item.name,
            "value": item.value,
            "polarity": item.polarity.value,
            "importance": item.importance.value,
        }
    )


def _axis_identity(item: UnresolvedAxis) -> str:
    return canonical_json(
        {
            "axis_type": item.axis_type,
            "description": item.description,
            "options": item.options,
        }
    )


def _stable_id(kind: str, trajectory_id: str, identity: str) -> str:
    digest = content_sha256({"kind": kind, "trajectory_id": trajectory_id, "identity": identity})
    prefix = {"evidence": "evd", "constraint": "ctr", "axis": "axis"}[kind]
    return f"{prefix}_{digest[:24]}"


def _unique_by_identity(
    items: Sequence[SemanticConstraint] | Sequence[UnresolvedAxis],
    *,
    family: Literal["constraint", "axis"],
) -> dict[str, SemanticConstraint] | dict[str, UnresolvedAxis]:
    identity = _constraint_identity if family == "constraint" else _axis_identity
    grouped: dict[str, list[SemanticConstraint | UnresolvedAxis]] = defaultdict(list)
    for item in items:
        grouped[identity(item)].append(item)  # type: ignore[arg-type]
    duplicates = sorted(key for key, values in grouped.items() if len(values) > 1)
    if duplicates:
        raise SnapshotDiffUncompilable(
            f"ambiguous_duplicate_{family}_semantics",
            duplicate_count=len(duplicates),
        )
    return {key: values[0] for key, values in grouped.items()}  # type: ignore[return-value]


def snapshot_decision_payload(
    state: TeacherCurrentStateSnapshot | PublicDSUSState,
) -> dict[str, JsonValue]:
    """Canonical decision view; provenance IDs and superseded memory are excluded."""

    constraints = [
        {
            "name": item.name,
            "value": item.value,
            "polarity": item.polarity.value,
            "importance": item.importance.value,
            "status": item.status.value,
        }
        for item in state.constraints
        if item.status != ConstraintStatus.SUPERSEDED
    ]
    axes = [
        {
            "axis_type": item.axis_type,
            "description": item.description,
            "options": item.options,
        }
        for item in state.unresolved_axes
    ]
    rejected = [
        {"item_id": item.item_id, "reason": item.reason}
        for item in state.rejected_items
    ]
    return {
        "control": state.control.model_dump(mode="json"),
        "constraints": sorted(constraints, key=canonical_json),
        "unresolved_axes": sorted(axes, key=canonical_json),
        "rejected_items": sorted(rejected, key=canonical_json),
        "termination_reason": (
            None if state.termination_reason is None else state.termination_reason.value
        ),
    }


def snapshot_decision_sha256(state: TeacherCurrentStateSnapshot | PublicDSUSState) -> str:
    return content_sha256(snapshot_decision_payload(state))


def _normalized_evidence(
    snapshot: TeacherCurrentStateSnapshot,
    *,
    trajectory_id: str,
) -> tuple[dict[str, EvidenceReference], dict[str, EvidenceReference]]:
    """Return teacher-ID -> stable reference and semantic-key -> stable reference."""

    by_teacher_id: dict[str, EvidenceReference] = {}
    by_key: dict[str, EvidenceReference] = {}
    stable_ids: dict[str, str] = {}
    for reference in snapshot.evidence:
        key = _evidence_key(reference)
        if key in by_key:
            raise SnapshotDiffUncompilable(
                "ambiguous_duplicate_evidence_semantics",
                turn_id=reference.turn_id,
                source=reference.source.value,
            )
        stable_id = _stable_id("evidence", trajectory_id, key)
        other = stable_ids.get(stable_id)
        if other is not None and other != key:  # defensive against truncated-hash collision
            raise SnapshotDiffUncompilable("deterministic_evidence_id_collision")
        stable_ids[stable_id] = key
        normalized = reference.model_copy(update={"evidence_id": stable_id})
        by_teacher_id[reference.evidence_id] = normalized
        by_key[key] = normalized
    return by_teacher_id, by_key


def _normalize_snapshot_as_previous_state(
    snapshot: TeacherCurrentStateSnapshot,
    *,
    trajectory_id: str,
    revision: int,
) -> PublicDSUSState:
    """Materialize an adjacent predecessor with teacher-independent local IDs."""

    evidence_by_id, _ = _normalized_evidence(snapshot, trajectory_id=trajectory_id)

    def refs(values: Sequence[str]) -> list[str]:
        try:
            return sorted(evidence_by_id[value].evidence_id for value in values)
        except KeyError as exc:  # snapshot validation normally makes this unreachable
            raise SnapshotDiffUncompilable("predecessor_unknown_evidence_id") from exc

    active_constraints = [
        item for item in snapshot.constraints if item.status != ConstraintStatus.SUPERSEDED
    ]
    constraint_by_identity = _unique_by_identity(active_constraints, family="constraint")
    constraints: list[SemanticConstraint] = []
    used_constraint_ids: dict[str, str] = {}
    for identity, item in constraint_by_identity.items():
        stable_id = _stable_id("constraint", trajectory_id, identity)
        if stable_id in used_constraint_ids and used_constraint_ids[stable_id] != identity:
            raise SnapshotDiffUncompilable("deterministic_constraint_id_collision")
        used_constraint_ids[stable_id] = identity
        constraints.append(
            item.model_copy(
                update={"constraint_id": stable_id, "evidence_ids": refs(item.evidence_ids)}
            )
        )

    axis_by_identity = _unique_by_identity(snapshot.unresolved_axes, family="axis")
    axes: list[UnresolvedAxis] = []
    used_axis_ids: dict[str, str] = {}
    for identity, item in axis_by_identity.items():
        stable_id = _stable_id("axis", trajectory_id, identity)
        if stable_id in used_axis_ids and used_axis_ids[stable_id] != identity:
            raise SnapshotDiffUncompilable("deterministic_axis_id_collision")
        used_axis_ids[stable_id] = identity
        axes.append(
            item.model_copy(update={"axis_id": stable_id, "evidence_ids": refs(item.evidence_ids)})
        )

    return PublicDSUSState(
        revision=revision,
        initialized=True,
        control=snapshot.control.model_copy(deep=True),
        constraints=constraints,
        unresolved_axes=axes,
        rejected_items=[
            item.model_copy(update={"evidence_ids": refs(item.evidence_ids)})
            for item in snapshot.rejected_items
        ],
        evidence=sorted(
            (item.model_copy(deep=True) for item in evidence_by_id.values()),
            key=lambda item: (item.turn_id, item.evidence_id),
        ),
        control_evidence={field: refs(values) for field, values in snapshot.control_evidence.items()},
        termination_reason=snapshot.termination_reason,
        termination_evidence_ids=refs(snapshot.termination_evidence_ids),
    )


class _Differ:
    def __init__(
        self,
        *,
        trajectory_id: str,
        source_turn_index: int,
        previous_state: PublicDSUSState,
        target: TeacherCurrentStateSnapshot,
    ) -> None:
        self.trajectory_id = trajectory_id
        self.turn = source_turn_index
        self.previous = previous_state
        self.target = target
        self.target_evidence, target_by_key = _normalized_evidence(
            target, trajectory_id=trajectory_id
        )
        previous_keys = {_evidence_key(item) for item in previous_state.evidence}
        future = sorted(
            {item.turn_id for item in target.evidence if item.turn_id > source_turn_index}
        )
        if future:
            raise SnapshotDiffUncompilable(
                "future_evidence_in_current_snapshot",
                source_turn_index=source_turn_index,
                future_turn_ids=future,
            )
        self.fresh_by_teacher_id = {
            teacher_id: reference
            for teacher_id, reference in self.target_evidence.items()
            if reference.turn_id == source_turn_index
            and _evidence_key(reference) not in previous_keys
        }
        self.fresh_by_stable_id = {
            item.evidence_id: item for item in self.fresh_by_teacher_id.values()
        }
        # A repeated semantic evidence key at a later cold projection is not a
        # new observation and must never be reissued in a patch.
        del target_by_key
        self.operations: list[StateOperation] = []

    def refs(self, teacher_ids: Sequence[str], *, owner: str) -> list[str]:
        values = sorted(
            {
                self.fresh_by_teacher_id[value].evidence_id
                for value in teacher_ids
                if value in self.fresh_by_teacher_id
            }
        )
        if not values:
            raise SnapshotDiffUncompilable(
                "no_current_turn_evidence",
                owner=owner,
                source_turn_index=self.turn,
            )
        return values

    def transition_refs(self, *, owner: str) -> list[str]:
        """Evidence explicitly attached to revision/correction control axes."""

        candidates: list[str] = []
        for field in (ControlField.INTENT_STABILITY, ControlField.FEEDBACK_TYPE):
            candidates.extend(self.target.control_evidence.get(field, []))
        return self.refs(candidates, owner=owner)

    def compile_controls(self, *, terminating: bool) -> None:
        previous = self.previous.control
        target = self.target.control
        for field in ControlField:
            old_value = getattr(previous, field.value)
            new_value = getattr(target, field.value)
            if old_value == new_value:
                continue
            if field == ControlField.FEEDBACK_TYPE:
                continue
            if terminating and field == ControlField.INTERACTION_PHASE:
                continue
            if (
                terminating
                and field == ControlField.COMMITMENT_STATUS
                and self.target.termination_reason
                in (TerminationReason.ACCEPTED, TerminationReason.ABANDONED)
            ):
                expected = (
                    CommitmentStatus.ACCEPTED
                    if self.target.termination_reason == TerminationReason.ACCEPTED
                    else CommitmentStatus.ABANDONED
                )
                if new_value != expected:
                    raise SnapshotDiffUncompilable(
                        "termination_commitment_mismatch",
                        termination_reason=self.target.termination_reason.value,
                        commitment_status=new_value.value,
                    )
                continue
            self.operations.append(
                SetControlOperation(
                    field=field,
                    value=new_value.value,
                    evidence_ids=self.refs(
                        self.target.control_evidence.get(field, []),
                        owner=f"control:{field.value}",
                    ),
                )
            )

    def compile_constraints(self) -> dict[str, list[str]]:
        previous_items = [
            item for item in self.previous.constraints if item.status != ConstraintStatus.SUPERSEDED
        ]
        previous = _unique_by_identity(previous_items, family="constraint")
        target = _unique_by_identity(self.target.constraints, family="constraint")
        exact = set(previous).intersection(target)
        for identity in sorted(exact):
            before = previous[identity]
            after = target[identity]
            if before.status != after.status:
                self.operations.append(
                    UpsertConstraintOperation(
                        constraint_id=before.constraint_id,
                        name=after.name,
                        value=after.value,
                        polarity=after.polarity,
                        importance=after.importance,
                        status=after.status,
                        evidence_ids=self.refs(
                            after.evidence_ids,
                            owner=f"constraint_status:{after.name}",
                        ),
                    )
                )

        removed = {key: previous[key] for key in set(previous).difference(target)}
        added = {key: target[key] for key in set(target).difference(previous)}
        removed_by_name: dict[str, list[tuple[str, SemanticConstraint]]] = defaultdict(list)
        added_by_name: dict[str, list[tuple[str, SemanticConstraint]]] = defaultdict(list)
        for identity, item in removed.items():
            removed_by_name[item.name].append((identity, item))
        for identity, item in added.items():
            added_by_name[item.name].append((identity, item))

        added_evidence_by_name: dict[str, list[str]] = defaultdict(list)
        for name in sorted(set(removed_by_name).union(added_by_name)):
            old_values = sorted(removed_by_name.get(name, []), key=lambda pair: pair[0])
            new_values = sorted(added_by_name.get(name, []), key=lambda pair: pair[0])
            if old_values and new_values and (len(old_values) != 1 or len(new_values) != 1):
                raise SnapshotDiffUncompilable(
                    "ambiguous_constraint_revision",
                    constraint_name=name,
                    removed_count=len(old_values),
                    added_count=len(new_values),
                )
            if old_values and new_values:
                _, old_item = old_values[0]
                new_identity, new_item = new_values[0]
                evidence_ids = self.refs(
                    new_item.evidence_ids,
                    owner=f"constraint_revision:{name}",
                )
                self.operations.append(
                    SupersedeConstraintOperation(
                        constraint_id=old_item.constraint_id,
                        evidence_ids=evidence_ids,
                    )
                )
                new_id = _stable_id("constraint", self.trajectory_id, new_identity)
                self.operations.append(
                    UpsertConstraintOperation(
                        constraint_id=new_id,
                        name=new_item.name,
                        value=new_item.value,
                        polarity=new_item.polarity,
                        importance=new_item.importance,
                        status=new_item.status,
                        evidence_ids=evidence_ids,
                    )
                )
                added_evidence_by_name[name].extend(evidence_ids)
                continue
            if old_values:
                evidence_ids = self.transition_refs(owner=f"constraint_removal:{name}")
                for _, old_item in old_values:
                    self.operations.append(
                        RemoveConstraintOperation(
                            constraint_id=old_item.constraint_id,
                            evidence_ids=evidence_ids,
                        )
                    )
                continue
            for new_identity, new_item in new_values:
                evidence_ids = self.refs(
                    new_item.evidence_ids,
                    owner=f"constraint_addition:{name}",
                )
                self.operations.append(
                    UpsertConstraintOperation(
                        constraint_id=_stable_id(
                            "constraint", self.trajectory_id, new_identity
                        ),
                        name=new_item.name,
                        value=new_item.value,
                        polarity=new_item.polarity,
                        importance=new_item.importance,
                        status=new_item.status,
                        evidence_ids=evidence_ids,
                    )
                )
                added_evidence_by_name[name].extend(evidence_ids)
        return {key: sorted(set(value)) for key, value in added_evidence_by_name.items()}

    def compile_axes(self, added_constraint_evidence: Mapping[str, list[str]]) -> None:
        previous = _unique_by_identity(self.previous.unresolved_axes, family="axis")
        target = _unique_by_identity(self.target.unresolved_axes, family="axis")
        removed = {key: previous[key] for key in set(previous).difference(target)}
        added = {key: target[key] for key in set(target).difference(previous)}
        removed_by_type: dict[str, list[tuple[str, UnresolvedAxis]]] = defaultdict(list)
        added_by_type: dict[str, list[tuple[str, UnresolvedAxis]]] = defaultdict(list)
        for identity, item in removed.items():
            removed_by_type[item.axis_type].append((identity, item))
        for identity, item in added.items():
            added_by_type[item.axis_type].append((identity, item))

        for axis_type in sorted(set(removed_by_type).union(added_by_type)):
            old_values = sorted(removed_by_type.get(axis_type, []), key=lambda pair: pair[0])
            new_values = sorted(added_by_type.get(axis_type, []), key=lambda pair: pair[0])
            if old_values and new_values and (len(old_values) != 1 or len(new_values) != 1):
                raise SnapshotDiffUncompilable(
                    "ambiguous_axis_revision",
                    axis_type=axis_type,
                    removed_count=len(old_values),
                    added_count=len(new_values),
                )
            replacement_refs: list[str] | None = None
            if new_values:
                replacement_refs = self.refs(
                    new_values[0][1].evidence_ids,
                    owner=f"axis_addition:{axis_type}",
                ) if len(new_values) == 1 else None
            for _, old_item in old_values:
                evidence_ids = replacement_refs
                if evidence_ids is None:
                    evidence_ids = added_constraint_evidence.get(axis_type)
                if not evidence_ids:
                    evidence_ids = self.transition_refs(owner=f"axis_resolution:{axis_type}")
                self.operations.append(
                    ResolveAxisOperation(axis_id=old_item.axis_id, evidence_ids=evidence_ids)
                )
            for identity, new_item in new_values:
                evidence_ids = self.refs(
                    new_item.evidence_ids,
                    owner=f"axis_addition:{axis_type}",
                )
                self.operations.append(
                    AddUnresolvedAxisOperation(
                        axis_id=_stable_id("axis", self.trajectory_id, identity),
                        axis_type=new_item.axis_type,
                        description=new_item.description,
                        options=new_item.options,
                        evidence_ids=evidence_ids,
                    )
                )

    def compile_feedback(self) -> None:
        previous = {item.item_id: item for item in self.previous.rejected_items}
        target = {item.item_id: item for item in self.target.rejected_items}
        removed = sorted(set(previous).difference(target))
        if removed:
            raise SnapshotDiffUncompilable(
                "rejected_item_removal_not_expressible",
                removed_item_ids=removed,
            )
        mutations = []
        for item_id, item in sorted(target.items()):
            old = previous.get(item_id)
            if old is None or old.reason != item.reason:
                if old is not None and old.reason is not None and item.reason is None:
                    raise SnapshotDiffUncompilable(
                        "rejected_item_reason_clear_not_expressible",
                        item_id=item_id,
                    )
                mutations.append(item)
        if len(mutations) > 1:
            raise SnapshotDiffUncompilable(
                "multiple_rejected_item_mutations_not_expressible",
                mutation_count=len(mutations),
                item_ids=[item.item_id for item in mutations],
            )

        old_feedback = self.previous.control.feedback_type
        new_feedback = self.target.control.feedback_type
        if mutations:
            if new_feedback != FeedbackType.REJECTION:
                raise SnapshotDiffUncompilable(
                    "rejected_item_without_rejection_control",
                    feedback_type=new_feedback.value,
                )
            item = mutations[0]
            self.operations.append(
                RecordFeedbackOperation(
                    feedback_type=FeedbackType.REJECTION,
                    item_id=item.item_id,
                    reason=item.reason,
                    evidence_ids=self.refs(
                        item.evidence_ids,
                        owner=f"rejected_item:{item.item_id}",
                    ),
                )
            )
            return
        if old_feedback == new_feedback:
            return
        evidence_ids = self.refs(
            self.target.control_evidence.get(ControlField.FEEDBACK_TYPE, []),
            owner="control:feedback_type",
        )
        if new_feedback == FeedbackType.NONE:
            self.operations.append(
                SetControlOperation(
                    field=ControlField.FEEDBACK_TYPE,
                    value=new_feedback.value,
                    evidence_ids=evidence_ids,
                )
            )
        else:
            self.operations.append(
                RecordFeedbackOperation(
                    feedback_type=new_feedback,
                    evidence_ids=evidence_ids,
                )
            )

    def compile_termination(self) -> None:
        if self.target.termination_reason is None:
            if self.previous.termination_reason is not None:
                raise SnapshotDiffUncompilable("terminal_state_cannot_be_reopened")
            return
        if self.previous.termination_reason is not None:
            if snapshot_decision_sha256(self.previous) != snapshot_decision_sha256(self.target):
                raise SnapshotDiffUncompilable("terminal_state_cannot_be_mutated")
            return
        self.operations.append(
            TerminateOperation(
                reason=self.target.termination_reason,
                evidence_ids=self.refs(
                    self.target.termination_evidence_ids,
                    owner="termination",
                ),
            )
        )

    def build(self) -> tuple[StatePatch, PublicDSUSState]:
        if self.previous.control.interaction_phase == InteractionPhase.TERMINAL:
            if snapshot_decision_sha256(self.previous) != snapshot_decision_sha256(self.target):
                raise SnapshotDiffUncompilable("terminal_state_cannot_be_mutated")

        terminating = self.target.termination_reason is not None and self.previous.termination_reason is None
        if not self.previous.initialized:
            if not self.fresh_by_stable_id:
                raise SnapshotDiffUncompilable(
                    "opening_has_no_current_turn_evidence",
                    source_turn_index=self.turn,
                )
            first = sorted(self.fresh_by_stable_id)[0]
            self.operations.append(InitializeStateOperation(evidence_ids=[first]))

        self.compile_controls(terminating=terminating)
        added_constraint_evidence = self.compile_constraints()
        self.compile_axes(added_constraint_evidence)
        self.compile_feedback()
        self.compile_termination()

        if not self.operations:
            if not self.fresh_by_stable_id:
                raise SnapshotDiffUncompilable(
                    "noop_has_no_current_turn_evidence",
                    source_turn_index=self.turn,
                )
            self.operations = [
                NoopOperation(
                    reason="current public observation does not change decision state",
                    evidence_ids=sorted(self.fresh_by_stable_id),
                )
            ]

        used_evidence_ids = sorted(
            {
                evidence_id
                for operation in self.operations
                for evidence_id in operation.evidence_ids
            }
        )
        missing = sorted(set(used_evidence_ids).difference(self.fresh_by_stable_id))
        if missing:
            raise SnapshotDiffUncompilable(
                "operation_references_nonfresh_evidence",
                evidence_ids=missing,
            )
        try:
            patch = StatePatch(
                base_revision=self.previous.revision,
                evidence=[self.fresh_by_stable_id[value] for value in used_evidence_ids],
                operations=self.operations,
            )
        except ValidationError as exc:
            raise SnapshotDiffUncompilable(
                "generated_patch_schema_rejected",
                error_count=exc.error_count(),
            ) from exc
        try:
            replayed = reduce_state(self.previous, patch)
        except TransitionError as exc:
            raise SnapshotDiffUncompilable(
                "production_reducer_rejected_candidate",
                transition_error=str(exc),
            ) from exc
        expected = snapshot_decision_sha256(self.target)
        actual = snapshot_decision_sha256(replayed)
        if expected != actual:
            raise SnapshotDiffUncompilable(
                "replay_decision_mismatch",
                target_decision_sha256=expected,
                replay_decision_sha256=actual,
            )
        return patch, replayed


def compile_snapshot_transition(
    *,
    current: TeacherSnapshotSourceRecord,
    previous: TeacherSnapshotSourceRecord | None,
) -> TeacherSnapshotTransitionRecord:
    """Compile one opening or adjacent accepted prefix, returning reasons not guesses."""

    if isinstance(current, TeacherConsensusRecord):
        if current.status != "accepted" or current.consensus_snapshot is None:
            raise ValueError("current consensus record must be accepted")
        current_snapshot = current.consensus_snapshot
    else:
        current_snapshot = current.snapshot
    target_sha = snapshot_decision_sha256(current_snapshot)
    predecessor_sample_id = None if previous is None else previous.sample_id
    try:
        if previous is None:
            if current.source_turn_index != 0:
                raise SnapshotDiffUncompilable(
                    "missing_opening_predecessor",
                    source_turn_index=current.source_turn_index,
                )
            previous_state = PublicDSUSState.empty()
        else:
            if isinstance(previous, TeacherConsensusRecord):
                if previous.status != "accepted" or previous.consensus_snapshot is None:
                    raise SnapshotDiffUncompilable("predecessor_not_accepted")
                previous_snapshot = previous.consensus_snapshot
            else:
                previous_snapshot = previous.snapshot
            if previous.trajectory_id != current.trajectory_id:
                raise SnapshotDiffUncompilable("predecessor_trajectory_mismatch")
            if current.source_turn_index != previous.source_turn_index + 1:
                raise SnapshotDiffUncompilable(
                    "non_adjacent_snapshot_pair",
                    predecessor_turn_index=previous.source_turn_index,
                    source_turn_index=current.source_turn_index,
                )
            previous_state = _normalize_snapshot_as_previous_state(
                previous_snapshot,
                trajectory_id=current.trajectory_id,
                revision=current.source_turn_index,
            )
        patch, replayed = _Differ(
            trajectory_id=current.trajectory_id,
            source_turn_index=current.source_turn_index,
            previous_state=previous_state,
            target=current_snapshot,
        ).build()
        replay_sha = snapshot_decision_sha256(replayed)
        return TeacherSnapshotTransitionRecord(
            sample_id=current.sample_id,
            trajectory_id=current.trajectory_id,
            source_turn_index=current.source_turn_index,
            predecessor_sample_id=predecessor_sample_id,
            status="compiled",
            patch=patch,
            replayed_state=replayed,
            target_decision_sha256=target_sha,
            replay_decision_sha256=replay_sha,
            reasons=[SnapshotDiffReason(code="production_reducer_replay_verified")],
        )
    except SnapshotDiffUncompilable as exc:
        return TeacherSnapshotTransitionRecord(
            sample_id=current.sample_id,
            trajectory_id=current.trajectory_id,
            source_turn_index=current.source_turn_index,
            predecessor_sample_id=predecessor_sample_id,
            status="uncompilable",
            target_decision_sha256=target_sha,
            reasons=[exc.reason],
        )


def compile_accepted_snapshot_trajectories(
    records: Sequence[TeacherSnapshotSourceRecord],
) -> tuple[list[TeacherSnapshotTransitionRecord], TeacherSnapshotDifferSummary]:
    """Compile sorted trajectories; a failed pair does not poison a later adjacent pair."""

    if not records:
        raise ValueError("accepted consensus input cannot be empty")
    by_trajectory: dict[str, list[TeacherSnapshotSourceRecord]] = defaultdict(list)
    seen_samples: set[str] = set()
    for record in records:
        if isinstance(record, TeacherConsensusRecord) and (
            record.status != "accepted" or record.consensus_snapshot is None
        ):
            raise ValueError("input contains a non-accepted consensus record")
        if record.sample_id in seen_samples:
            raise ValueError(f"duplicate consensus sample: {record.sample_id}")
        seen_samples.add(record.sample_id)
        by_trajectory[record.trajectory_id].append(record)

    outputs: list[TeacherSnapshotTransitionRecord] = []
    for trajectory_id in sorted(by_trajectory):
        trajectory = sorted(
            by_trajectory[trajectory_id],
            key=lambda item: (item.source_turn_index, item.sample_id),
        )
        turns = [item.source_turn_index for item in trajectory]
        if len(turns) != len(set(turns)):
            raise ValueError(f"trajectory contains duplicate source turn: {trajectory_id}")
        for index, current in enumerate(trajectory):
            previous = trajectory[index - 1] if index else None
            outputs.append(compile_snapshot_transition(current=current, previous=previous))

    outputs.sort(key=lambda item: (item.trajectory_id, item.source_turn_index, item.sample_id))
    status_counts = Counter(item.status for item in outputs)
    reason_counts = Counter(reason.code for item in outputs for reason in item.reasons)
    operation_counts = Counter(
        operation.op
        for item in outputs
        if item.patch is not None
        for operation in item.patch.operations
    )
    summary = TeacherSnapshotDifferSummary(
        input_record_count=len(records),
        trajectory_count=len(by_trajectory),
        compiled_count=status_counts["compiled"],
        uncompilable_count=status_counts["uncompilable"],
        status_counts=dict(sorted(status_counts.items())),
        reason_counts=dict(sorted(reason_counts.items())),
        operation_counts=dict(sorted(operation_counts.items())),
        replay_verified_count=status_counts["compiled"],
    )
    return outputs, summary


__all__ = [
    "TEACHER_SNAPSHOT_DIFFER_SUMMARY_VERSION",
    "TEACHER_SNAPSHOT_DIFFER_VERSION",
    "SnapshotDiffReason",
    "TeacherSnapshotDifferSummary",
    "TeacherSnapshotSourceRecord",
    "TeacherSnapshotTransitionRecord",
    "compile_accepted_snapshot_trajectories",
    "compile_snapshot_transition",
    "snapshot_decision_payload",
    "snapshot_decision_sha256",
]
