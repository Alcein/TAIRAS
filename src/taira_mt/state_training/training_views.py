"""Leakage-safe SFT and RLVR views of compiled DSUS transitions.

One :class:`CompiledStateTransition` is the single source of truth.  This
module projects it into two views with a byte-identical public prompt:

* SFT receives an evidence-grounded typed delta as its private target;
* RLVR receives the same public prompt, while the next state and verifier
  material remain trainer-private.

Dataset loaders must render prompts with :func:`render_sft_public_prompt` or
:func:`render_rlvr_public_prompt`.  Dumping an entire RLVR record into a chat
template is a contract violation because it would expose ``trainer_private``.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, field_validator, model_validator

from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.state_compiler import (
    PATCH_SCHEMA_VERSION,
    STATE_SCHEMA_VERSION,
    ControlField,
    EvidenceReference,
    PublicDSUSState,
    RecordFeedbackOperation,
    SetControlOperation,
    StateOperation,
    StatePatch,
    TerminateOperation,
    reduce_state,
)
from taira_mt.state_training.logical_compiler import (
    CompiledLogicalTrajectory,
    CompiledStateTransition,
)

TRAINING_VIEW_SCHEMA_VERSION = "dsus_training_views_v1"


class OperationName(str, Enum):
    INITIALIZE_STATE = "initialize_state"
    SET_CONTROL = "set_control"
    UPSERT_CONSTRAINT = "upsert_constraint"
    SUPERSEDE_CONSTRAINT = "supersede_constraint"
    REMOVE_CONSTRAINT = "remove_constraint"
    ADD_UNRESOLVED_AXIS = "add_unresolved_axis"
    RESOLVE_AXIS = "resolve_axis"
    RECORD_FEEDBACK = "record_feedback"
    TERMINATE = "terminate"
    NOOP = "noop"


_CANONICAL_OPERATIONS = list(OperationName)
_CANONICAL_CONTROLS = list(ControlField)


class PublicCompilerSchema(StrictModel):
    """Compact, stable schema descriptor included in every model prompt."""

    state_schema_version: Literal[STATE_SCHEMA_VERSION] = STATE_SCHEMA_VERSION
    patch_schema_version: Literal[PATCH_SCHEMA_VERSION] = PATCH_SCHEMA_VERSION
    task: Literal["predict_evidence_grounded_state_delta"] = "predict_evidence_grounded_state_delta"
    evidence_contract: Literal["exact_quote_from_public_observation"] = "exact_quote_from_public_observation"
    output_contract: Literal["base_revision_evidence_delta"] = "base_revision_evidence_delta"
    allowed_delta_operations: list[OperationName] = Field(default_factory=lambda: list(_CANONICAL_OPERATIONS))
    control_fields: list[ControlField] = Field(default_factory=lambda: list(_CANONICAL_CONTROLS))

    @model_validator(mode="after")
    def validate_complete_schema(self) -> PublicCompilerSchema:
        if self.allowed_delta_operations != _CANONICAL_OPERATIONS:
            raise ValueError("allowed_delta_operations must use the complete canonical order")
        if self.control_fields != _CANONICAL_CONTROLS:
            raise ValueError("control_fields must use the complete canonical order")
        return self


class PublicTextObservation(StrictModel):
    """The only transcript shape a state-compiler policy may observe."""

    role: Literal["user", "system", "tool", "catalog"]
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("public observation text cannot be blank")
        return value


class DSUSPublicPrompt(StrictModel):
    """Allow-listed public input shared by SFT and RLVR."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    compiler_schema: PublicCompilerSchema = Field(
        default_factory=PublicCompilerSchema,
        alias="schema",
    )
    previous_public_state: PublicDSUSState
    public_history: list[PublicTextObservation] | None = None
    latest_observations: list[PublicTextObservation] | None = None

    @model_validator(mode="after")
    def validate_observation_mode(self) -> DSUSPublicPrompt:
        populated = [field for field in (self.public_history, self.latest_observations) if field is not None]
        if len(populated) != 1:
            raise ValueError("exactly one of public_history or latest_observations is required")
        if not populated[0]:
            raise ValueError("the selected public observation view cannot be empty")
        return self

    def canonical_payload(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json", by_alias=True)

    def canonical_json(self) -> str:
        return canonical_json(self.canonical_payload())

    def content_hash(self) -> str:
        return content_sha256(self.canonical_payload())


class EvidenceGroundedDeltaTarget(StrictModel):
    """SFT assistant target; gold labels never enter the public prompt."""

    patch_schema_version: Literal[PATCH_SCHEMA_VERSION] = PATCH_SCHEMA_VERSION
    base_revision: int = Field(ge=0)
    gold_evidence: list[EvidenceReference] = Field(min_length=1)
    gold_delta: list[StateOperation] = Field(min_length=1)

    def as_state_patch(self) -> StatePatch:
        return StatePatch(
            schema_version=self.patch_schema_version,
            base_revision=self.base_revision,
            evidence=self.gold_evidence,
            operations=self.gold_delta,
        )

    def assistant_payload(self) -> dict[str, JsonValue]:
        """Return only the JSON object on which assistant-token loss is applied."""

        return {
            "schema_version": self.patch_schema_version,
            "base_revision": self.base_revision,
            "evidence": [item.model_dump(mode="json") for item in self.gold_evidence],
            "delta": [item.model_dump(mode="json") for item in self.gold_delta],
        }


class TransitionInvariantType(str, Enum):
    BASE_REVISION = "base_revision"
    PATCH_REPLAY = "patch_replay"
    EVIDENCE_ALLOWED = "evidence_allowed"
    PRESERVE_CONTROL = "preserve_control"
    PRESERVE_CONSTRAINT = "preserve_constraint"
    PRESERVE_UNRESOLVED_AXIS = "preserve_unresolved_axis"
    PRESERVE_REJECTED_ITEM = "preserve_rejected_item"
    PRESERVE_TERMINATION = "preserve_termination"


class TransitionInvariant(StrictModel):
    """A typed, programmatically checkable trainer-private condition."""

    invariant_type: TransitionInvariantType
    subject: str = Field(min_length=1)
    expected: JsonValue

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("invariant subject cannot be blank")
        return value


class DifficultySlice(str, Enum):
    DIRECT = "direct"
    PROGRESSIVE = "progressive"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"
    REVISION = "revision"


class SFTTrainingView(StrictModel):
    schema_version: Literal[TRAINING_VIEW_SCHEMA_VERSION] = TRAINING_VIEW_SCHEMA_VERSION
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt: DSUSPublicPrompt
    private_target: EvidenceGroundedDeltaTarget

    @model_validator(mode="after")
    def validate_prompt_hash_and_revision(self) -> SFTTrainingView:
        if self.public_prompt_sha256 != self.public_prompt.content_hash():
            raise ValueError("public_prompt_sha256 does not match the canonical public prompt")
        patch = self.private_target.as_state_patch()
        if patch.base_revision != self.public_prompt.previous_public_state.revision:
            raise ValueError("SFT delta base revision does not match previous public state")
        return self


class RLVRTrainerPrivate(StrictModel):
    """Verifier data that a rollout policy and tokenizer must never observe."""

    gold_next_state: PublicDSUSState
    gold_base_revision: int = Field(ge=0)
    gold_evidence: list[EvidenceReference] = Field(min_length=1)
    gold_delta: list[StateOperation] = Field(min_length=1)
    allowed_evidence: list[EvidenceReference] = Field(min_length=1)
    transition_invariants: list[TransitionInvariant] = Field(min_length=1)
    difficulty_slice: DifficultySlice

    def gold_patch(self) -> StatePatch:
        return StatePatch(
            base_revision=self.gold_base_revision,
            evidence=self.gold_evidence,
            operations=self.gold_delta,
        )


class RLVRTrainingView(StrictModel):
    schema_version: Literal[TRAINING_VIEW_SCHEMA_VERSION] = TRAINING_VIEW_SCHEMA_VERSION
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt: DSUSPublicPrompt
    trainer_private: RLVRTrainerPrivate

    @model_validator(mode="after")
    def validate_private_verifier_contract(self) -> RLVRTrainingView:
        if self.public_prompt_sha256 != self.public_prompt.content_hash():
            raise ValueError("public_prompt_sha256 does not match the canonical public prompt")
        patch = self.trainer_private.gold_patch()
        previous = self.public_prompt.previous_public_state
        if patch.base_revision != previous.revision:
            raise ValueError("RLVR delta base revision does not match previous public state")
        replayed = reduce_state(previous, patch)
        if replayed.canonical_json() != self.trainer_private.gold_next_state.canonical_json():
            raise ValueError("RLVR gold delta does not replay to gold_next_state")
        allowed = [item.model_dump(mode="json") for item in self.trainer_private.allowed_evidence]
        gold = [item.model_dump(mode="json") for item in self.trainer_private.gold_evidence]
        if allowed != gold:
            raise ValueError("allowed_evidence must exactly match the frozen gold evidence set")
        return self


class TransitionTrainingViews(StrictModel):
    """Paired views whose public input must remain exactly identical."""

    sft: SFTTrainingView
    rlvr: RLVRTrainingView

    @model_validator(mode="after")
    def validate_shared_prompt(self) -> TransitionTrainingViews:
        if self.sft.public_prompt_sha256 != self.rlvr.public_prompt_sha256:
            raise ValueError("SFT and RLVR public prompt hashes differ")
        if self.sft.public_prompt.canonical_json() != self.rlvr.public_prompt.canonical_json():
            raise ValueError("SFT and RLVR public prompt payloads differ")
        return self


class TrainingViewCollection(StrictModel):
    """JSON-safe paired batch returned by the bulk projector."""

    sft_records: list[SFTTrainingView]
    rlvr_records: list[RLVRTrainingView]

    @model_validator(mode="after")
    def validate_alignment(self) -> TrainingViewCollection:
        if len(self.sft_records) != len(self.rlvr_records):
            raise ValueError("SFT and RLVR record counts differ")
        for sft, rlvr in zip(self.sft_records, self.rlvr_records, strict=True):
            if sft.public_prompt_sha256 != rlvr.public_prompt_sha256:
                raise ValueError("SFT/RLVR batch alignment differs")
        return self


_FORBIDDEN_PUBLIC_KEYS = {
    "acceptable_item_ids",
    "action_type",
    "assigned_score",
    "assignment",
    "assignment_regret",
    "authorized_constraint_ids",
    "behavior_policy",
    "best_score",
    "candidate_feasibility",
    "decision_obligation",
    "difficulty_slice",
    "fired_event_ids",
    "future_reply",
    "gold_delta",
    "gold_evidence",
    "gold_next_state",
    "gold_patch",
    "goal_version",
    "hidden_goal",
    "hidden_state",
    "latent_goal",
    "latent_patience",
    "latent_satisfaction",
    "mechanism",
    "optimal_action",
    "optimal_action_type",
    "oracle",
    "oracle_assessments",
    "oracle_score",
    "pattern_id",
    "private_target",
    "public_constraints",
    "recommendation_readiness",
    "response_frame",
    "reward",
    "reward_components",
    "root_id",
    "scenario_family",
    "scenario_mechanism",
    "source_group",
    "split",
    "state_after",
    "state_before",
    "step_id",
    "thought_pattern",
    "trainer_private",
    "variant_id",
}


def forbidden_public_key_paths(payload: object) -> list[str]:
    """Recursively report forbidden keys in a would-be public model input."""

    paths: list[str] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                child = f"{path}.{key}" if path else str(key)
                normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key))
                normalized = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
                if normalized in _FORBIDDEN_PUBLIC_KEYS:
                    paths.append(child)
                visit(value[key], child)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(payload, "")
    return sorted(paths)


def project_public_prompt(
    transition: CompiledStateTransition,
    *,
    latest_observation_count: int | None = None,
) -> DSUSPublicPrompt:
    """Project only allow-listed state and role/text transcript fields.

    ``latest_observation_count=None`` selects full history.  A positive value
    selects the latest N public messages.  In either mode, upstream identities,
    mechanism labels and typed action annotations are deliberately ignored.
    """

    if latest_observation_count is not None and latest_observation_count < 1:
        raise ValueError("latest_observation_count must be positive")
    observations = [
        PublicTextObservation(role=message.role, text=message.text) for message in transition.public_history
    ]
    if latest_observation_count is None:
        prompt = DSUSPublicPrompt(
            previous_public_state=transition.previous_state,
            public_history=observations,
        )
    else:
        prompt = DSUSPublicPrompt(
            previous_public_state=transition.previous_state,
            latest_observations=observations[-latest_observation_count:],
        )
    forbidden = forbidden_public_key_paths(prompt.model_dump(mode="json", by_alias=True))
    if forbidden:
        raise ValueError(f"forbidden trainer-private fields reached public prompt: {forbidden}")
    return prompt


def _transition_invariants(transition: CompiledStateTransition) -> list[TransitionInvariant]:
    patch = transition.gold_patch
    previous = transition.previous_state
    next_state = transition.gold_next_state
    touched_constraints: set[str] = set()
    touched_axes: set[str] = set()
    touched_rejections: set[str] = set()
    touched_controls: set[ControlField] = set()
    terminates = False

    for operation in patch.operations:
        if operation.op in {
            "upsert_constraint",
            "supersede_constraint",
            "remove_constraint",
        }:
            touched_constraints.add(operation.constraint_id)
        elif operation.op in {"add_unresolved_axis", "resolve_axis"}:
            touched_axes.add(operation.axis_id)
        elif isinstance(operation, SetControlOperation):
            touched_controls.add(operation.field)
        elif isinstance(operation, RecordFeedbackOperation):
            touched_controls.add(ControlField.FEEDBACK_TYPE)
            if operation.item_id is not None:
                touched_rejections.add(operation.item_id)
        elif isinstance(operation, TerminateOperation):
            touched_controls.add(ControlField.INTERACTION_PHASE)
            touched_controls.add(ControlField.COMMITMENT_STATUS)
            terminates = True

    invariants = [
        TransitionInvariant(
            invariant_type=TransitionInvariantType.BASE_REVISION,
            subject="previous_public_state.revision",
            expected=previous.revision,
        ),
        TransitionInvariant(
            invariant_type=TransitionInvariantType.PATCH_REPLAY,
            subject="gold_next_state.sha256",
            expected=content_sha256(next_state.canonical_payload()),
        ),
    ]
    invariants.extend(
        TransitionInvariant(
            invariant_type=TransitionInvariantType.EVIDENCE_ALLOWED,
            subject=evidence.evidence_id,
            expected=evidence.model_dump(mode="json"),
        )
        for evidence in patch.evidence
    )
    invariants.extend(
        TransitionInvariant(
            invariant_type=TransitionInvariantType.PRESERVE_CONTROL,
            subject=f"control.{field.value}",
            expected=getattr(previous.control, field.value).value,
        )
        for field in ControlField
        if field not in touched_controls
    )
    invariants.extend(
        TransitionInvariant(
            invariant_type=TransitionInvariantType.PRESERVE_CONSTRAINT,
            subject=f"constraint.{item.constraint_id}",
            expected=item.model_dump(mode="json"),
        )
        for item in previous.constraints
        if item.constraint_id not in touched_constraints
    )
    invariants.extend(
        TransitionInvariant(
            invariant_type=TransitionInvariantType.PRESERVE_UNRESOLVED_AXIS,
            subject=f"unresolved_axis.{item.axis_id}",
            expected=item.model_dump(mode="json"),
        )
        for item in previous.unresolved_axes
        if item.axis_id not in touched_axes
    )
    invariants.extend(
        TransitionInvariant(
            invariant_type=TransitionInvariantType.PRESERVE_REJECTED_ITEM,
            subject=f"rejected_item.{item.item_id}",
            expected=item.model_dump(mode="json"),
        )
        for item in previous.rejected_items
        if item.item_id not in touched_rejections
    )
    if not terminates:
        invariants.append(
            TransitionInvariant(
                invariant_type=TransitionInvariantType.PRESERVE_TERMINATION,
                subject="termination_reason",
                expected=(previous.termination_reason.value if previous.termination_reason is not None else None),
            )
        )
    return invariants


def project_transition_training_views(
    transition: CompiledStateTransition,
    *,
    difficulty_slice: DifficultySlice,
    latest_observation_count: int | None = None,
) -> TransitionTrainingViews:
    """Create paired SFT/RLVR records from one frozen logical transition."""

    prompt = project_public_prompt(
        transition,
        latest_observation_count=latest_observation_count,
    )
    prompt_hash = prompt.content_hash()
    target = EvidenceGroundedDeltaTarget(
        base_revision=transition.gold_patch.base_revision,
        gold_evidence=transition.gold_patch.evidence,
        gold_delta=transition.gold_patch.operations,
    )
    sft = SFTTrainingView(
        public_prompt_sha256=prompt_hash,
        public_prompt=prompt,
        private_target=target,
    )
    rlvr = RLVRTrainingView(
        public_prompt_sha256=prompt_hash,
        public_prompt=prompt,
        trainer_private=RLVRTrainerPrivate(
            gold_next_state=transition.gold_next_state,
            gold_base_revision=transition.gold_patch.base_revision,
            gold_evidence=transition.gold_patch.evidence,
            gold_delta=transition.gold_patch.operations,
            allowed_evidence=transition.gold_patch.evidence,
            transition_invariants=_transition_invariants(transition),
            difficulty_slice=difficulty_slice,
        ),
    )
    return TransitionTrainingViews(sft=sft, rlvr=rlvr)


def project_compiled_trajectories(
    trajectories: list[CompiledLogicalTrajectory],
    *,
    latest_observation_count: int | None = None,
) -> TrainingViewCollection:
    """Bulk-project trajectories without printing or exposing lineage metadata."""

    sft_records: list[SFTTrainingView] = []
    rlvr_records: list[RLVRTrainingView] = []
    for trajectory in trajectories:
        difficulty_slice = DifficultySlice(trajectory.mechanism.value)
        for transition in trajectory.transitions:
            pair = project_transition_training_views(
                transition,
                difficulty_slice=difficulty_slice,
                latest_observation_count=latest_observation_count,
            )
            sft_records.append(pair.sft)
            rlvr_records.append(pair.rlvr)
    return TrainingViewCollection(sft_records=sft_records, rlvr_records=rlvr_records)


def render_sft_public_prompt(record: SFTTrainingView) -> str:
    """Tokenizer-safe SFT input serialization; excludes ``private_target``."""

    return record.public_prompt.canonical_json()


def render_sft_private_target(record: SFTTrainingView) -> str:
    """Serialize only the assistant target, never as part of the user prompt."""

    return canonical_json(record.private_target.assistant_payload())


def render_rlvr_public_prompt(record: RLVRTrainingView) -> str:
    """Tokenizer-safe RLVR serialization; never renders ``trainer_private``."""

    return record.public_prompt.canonical_json()
