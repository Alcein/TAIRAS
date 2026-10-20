"""Trainer-side executable verifier chain for DSUS TV-RLVR.

The rollout policy receives only :class:`PublicRLPromptRecord`.  This module is
invoked *after* generation and joins the separately stored trainer-private
record by ``sample_id``.  It is deliberately independent of any particular
policy-optimization algorithm: it turns a predicted evidence-grounded patch
into deterministic, auditable verification components and one configurable
reward.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import Field, JsonValue, model_validator

from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.state_compiler import (
    PATCH_SCHEMA_VERSION,
    AddUnresolvedAxisOperation,
    ConstraintStatus,
    ControlField,
    EvidenceReference,
    EvidenceSource,
    InitializeStateOperation,
    InteractionPhase,
    NoopOperation,
    PublicDSUSState,
    RecordFeedbackOperation,
    RemoveConstraintOperation,
    ResolveAxisOperation,
    SetControlOperation,
    StateOperation,
    StatePatch,
    SupersedeConstraintOperation,
    TerminateOperation,
    UpsertConstraintOperation,
    reduce_state,
)
from taira_mt.state_training.dataset_assembly import (
    PrivateRLVerifierRecord,
    PublicRLPromptRecord,
)
from taira_mt.state_training.training_views import (
    RLVRTrainingView,
    TransitionInvariant,
    TransitionInvariantType,
    forbidden_public_key_paths,
)

VERIFIER_CHAIN_VERSION = "dsus_tv_rlvr_verifier_chain_v1"
REWARD_CONFIG_VERSION = "dsus_tv_rlvr_reward_v1"

RecordT = TypeVar("RecordT")


class VerifierComponent(str, Enum):
    SCHEMA = "schema"
    EVIDENCE = "evidence"
    OPERATION_LEGALITY = "operation_legality"
    REVISION = "revision"
    REPLAY = "replay"
    CONTROL_MATCH = "control_match"
    CONSTRAINT_MATCH = "constraint_match"
    STATE_EQUIVALENCE = "state_equivalence"
    PERSISTENCE = "persistence"
    MINIMALITY = "minimality"
    LENGTH = "length"


DEFAULT_REWARD_WEIGHTS: dict[VerifierComponent, float] = {
    VerifierComponent.SCHEMA: 0.04,
    VerifierComponent.EVIDENCE: 0.08,
    VerifierComponent.OPERATION_LEGALITY: 0.08,
    VerifierComponent.REVISION: 0.05,
    VerifierComponent.REPLAY: 0.08,
    VerifierComponent.CONTROL_MATCH: 0.08,
    VerifierComponent.CONSTRAINT_MATCH: 0.12,
    VerifierComponent.STATE_EQUIVALENCE: 0.25,
    VerifierComponent.PERSISTENCE: 0.10,
    VerifierComponent.MINIMALITY: 0.07,
    VerifierComponent.LENGTH: 0.05,
}


class TVRLVRRewardConfig(StrictModel):
    """Algorithm-agnostic reward configuration for the verifier chain."""

    schema_version: Literal[REWARD_CONFIG_VERSION] = REWARD_CONFIG_VERSION
    weights: dict[VerifierComponent, float] = Field(
        default_factory=lambda: dict(DEFAULT_REWARD_WEIGHTS)
    )
    hard_gate_components: list[VerifierComponent] = Field(
        default_factory=lambda: [
            VerifierComponent.SCHEMA,
            VerifierComponent.EVIDENCE,
            VerifierComponent.OPERATION_LEGALITY,
            VerifierComponent.REVISION,
            VerifierComponent.REPLAY,
        ]
    )
    max_output_chars: int = Field(default=4096, ge=128, le=131072)
    soft_gold_length_ratio: float = Field(default=1.35, ge=1.0, le=10.0)

    @model_validator(mode="after")
    def validate_reward_contract(self) -> TVRLVRRewardConfig:
        if set(self.weights) != set(VerifierComponent):
            raise ValueError("reward weights must cover every verifier component exactly once")
        if any(weight < 0 for weight in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("reward weights must be non-negative with a positive total")
        if len(self.hard_gate_components) != len(set(self.hard_gate_components)):
            raise ValueError("hard-gate components must be unique")
        return self


class PredictedPatchPayload(StrictModel):
    """The exact assistant JSON contract exposed in public compiler prompts."""

    schema_version: Literal[PATCH_SCHEMA_VERSION] = PATCH_SCHEMA_VERSION
    base_revision: int = Field(ge=0)
    evidence: list[EvidenceReference] = Field(min_length=1)
    delta: list[StateOperation] = Field(min_length=1)

    def as_patch(self) -> StatePatch:
        return StatePatch(
            schema_version=self.schema_version,
            base_revision=self.base_revision,
            evidence=self.evidence,
            operations=self.delta,
        )

    def canonical_payload(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json")


class RolloutOutputRecord(StrictModel):
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_output: str | dict[str, JsonValue]


class PatchLengthAudit(StrictModel):
    raw_output_chars: int = Field(ge=0)
    canonical_output_chars: int | None = Field(default=None, ge=0)
    gold_output_chars: int = Field(ge=1)
    predicted_operation_count: int | None = Field(default=None, ge=0)
    gold_operation_count: int = Field(ge=1)
    predicted_evidence_count: int | None = Field(default=None, ge=0)
    gold_evidence_count: int = Field(ge=1)
    extra_operation_count: int | None = None
    extra_evidence_count: int | None = None
    within_absolute_limit: bool
    within_soft_gold_limit: bool


class VerifierScoreRecord(StrictModel):
    schema_version: Literal[VERIFIER_CHAIN_VERSION] = VERIFIER_CHAIN_VERSION
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted: bool
    total_reward: float = Field(ge=0.0, le=1.0)
    components: dict[str, float]
    hard_gate_passed: bool
    length_audit: PatchLengthAudit
    persistence_checked: int = Field(ge=0)
    persistence_passed: int = Field(ge=0)
    error_codes: list[str]

    @model_validator(mode="after")
    def validate_score(self) -> VerifierScoreRecord:
        required = {component.value for component in VerifierComponent} | {"concision"}
        if set(self.components) != required:
            raise ValueError("score components do not match the verifier contract")
        if any(value < 0 or value > 1 for value in self.components.values()):
            raise ValueError("verifier component scores must lie in [0,1]")
        if self.persistence_passed > self.persistence_checked:
            raise ValueError("persistence passed count exceeds checked count")
        if self.error_codes != sorted(set(self.error_codes)):
            raise ValueError("error codes must be unique and sorted")
        return self


class VerifierBatchSummary(StrictModel):
    schema_version: Literal[VERIFIER_CHAIN_VERSION] = VERIFIER_CHAIN_VERSION
    sample_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    acceptance_rate: float = Field(ge=0.0, le=1.0)
    mean_total_reward: float = Field(ge=0.0, le=1.0)
    component_means: dict[str, float]
    error_code_counts: dict[str, int]
    trainer_private_join_stage: Literal["post_generation_verifier_only"] = (
        "post_generation_verifier_only"
    )
    public_prompt_private_leakage_count: int = Field(ge=0)
    external_calls: Literal[False] = False


class VerifierBatchResult(StrictModel):
    scores: list[VerifierScoreRecord]
    summary: VerifierBatchSummary
    run_dir: str | None = None


def parse_model_patch(model_output: str | Mapping[str, JsonValue]) -> tuple[PredictedPatchPayload, int]:
    """Parse strict JSON without accepting markdown fences or schema aliases."""

    if isinstance(model_output, str):
        raw_chars = len(model_output)
        value = json.loads(model_output)
    else:
        value = dict(model_output)
        raw_chars = len(canonical_json(value))
    if not isinstance(value, dict):
        raise ValueError("model output must be one JSON object")
    payload = PredictedPatchPayload.model_validate(value)
    # StatePatch performs cross-evidence, noop, order, and duplicate-touch validation.
    payload.as_patch()
    return payload, raw_chars


def _observations(public: PublicRLPromptRecord) -> list[tuple[str, str]]:
    prompt = public.public_prompt
    values = prompt.public_history if prompt.public_history is not None else prompt.latest_observations
    assert values is not None
    return [(item.role, item.text) for item in values]


def _evidence_grounded(
    evidence: Sequence[EvidenceReference],
    *,
    public: PublicRLPromptRecord,
    private: PrivateRLVerifierRecord,
) -> tuple[bool, list[str]]:
    observations = _observations(public)
    allowed = Counter(
        (item.turn_id, item.source.value, item.quote)
        for item in private.trainer_private.allowed_evidence
    )
    predicted = Counter((item.turn_id, item.source.value, item.quote) for item in evidence)
    errors: list[str] = []
    role_by_source = {
        EvidenceSource.USER: {"user"},
        EvidenceSource.SYSTEM: {"system"},
        EvidenceSource.CATALOG: {"catalog", "tool"},
    }
    for item in evidence:
        if not any(
            role in role_by_source[item.source] and item.quote in text
            for role, text in observations
        ):
            errors.append("evidence_not_exact_public_span")
        if (item.turn_id, item.source.value, item.quote) not in allowed:
            errors.append("evidence_not_trainer_allowed")
    if predicted != allowed:
        errors.append("evidence_set_not_complete")
    return not errors, errors


def _operation_legality(previous: PublicDSUSState, patch: StatePatch) -> bool:
    """Check state preconditions without conflating them with revision/evidence replay."""

    initialized = previous.initialized
    terminal = previous.control.interaction_phase == InteractionPhase.TERMINAL
    constraints = {item.constraint_id: item.model_copy(deep=True) for item in previous.constraints}
    axes = {item.axis_id for item in previous.unresolved_axes}
    if terminal and not (
        len(patch.operations) == 1 and isinstance(patch.operations[0], NoopOperation)
    ):
        return False
    for operation in patch.operations:
        if isinstance(operation, InitializeStateOperation):
            if initialized:
                return False
            initialized = True
            continue
        if not initialized:
            return False
        if isinstance(operation, SetControlOperation):
            if (
                operation.field == ControlField.INTERACTION_PHASE
                and operation.value == InteractionPhase.TERMINAL.value
            ):
                return False
        elif isinstance(operation, UpsertConstraintOperation):
            existing = constraints.get(operation.constraint_id)
            if existing is None:
                constraints[operation.constraint_id] = operation
            elif (
                existing.status == ConstraintStatus.SUPERSEDED
                or existing.name != operation.name
                or existing.value != operation.value
                or existing.polarity != operation.polarity
                or existing.importance != operation.importance
            ):
                return False
        elif isinstance(operation, SupersedeConstraintOperation):
            existing = constraints.get(operation.constraint_id)
            if existing is None or existing.status == ConstraintStatus.SUPERSEDED:
                return False
            existing.status = ConstraintStatus.SUPERSEDED
        elif isinstance(operation, RemoveConstraintOperation):
            if operation.constraint_id not in constraints:
                return False
            del constraints[operation.constraint_id]
        elif isinstance(operation, AddUnresolvedAxisOperation):
            if operation.axis_id in axes:
                return False
            axes.add(operation.axis_id)
        elif isinstance(operation, ResolveAxisOperation):
            if operation.axis_id not in axes:
                return False
            axes.remove(operation.axis_id)
        elif isinstance(operation, (RecordFeedbackOperation, TerminateOperation, NoopOperation)):
            continue
        else:  # pragma: no cover - closed StateOperation union guards this branch.
            return False
    return True


def _control_payload(state: PublicDSUSState) -> dict[str, JsonValue]:
    return state.control.model_dump(mode="json")


def _constraint_payload(state: PublicDSUSState) -> dict[str, JsonValue]:
    return {
        "constraints": [
            item.model_dump(mode="json", exclude={"evidence_ids"}) for item in state.constraints
        ],
        "unresolved_axes": [
            item.model_dump(mode="json", exclude={"evidence_ids"})
            for item in state.unresolved_axes
        ],
        "rejected_items": [
            item.model_dump(mode="json", exclude={"evidence_ids"}) for item in state.rejected_items
        ],
    }


def _alpha_normalized_state_payload(state: PublicDSUSState) -> dict[str, JsonValue]:
    """Canonicalize evidence IDs by their public content before joint comparison.

    Evidence identifiers are local variable names chosen by the model.  Full-state
    equivalence must therefore be alpha-invariant while still checking every
    evidence-to-state binding, control provenance, and persistence fact.
    """

    payload = state.model_dump(mode="json")
    id_to_token = {
        item.evidence_id: "evidence-"
        + content_sha256(
            {
                "turn_id": item.turn_id,
                "source": item.source.value,
                "quote": item.quote,
            }
        )
        for item in state.evidence
    }

    evidence_rows = payload["evidence"]
    assert isinstance(evidence_rows, list)
    for row in evidence_rows:
        assert isinstance(row, dict)
        row["evidence_id"] = id_to_token[str(row["evidence_id"])]
    evidence_rows.sort(key=canonical_json)

    for collection_name in ("constraints", "unresolved_axes", "rejected_items"):
        rows = payload[collection_name]
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, dict)
            references = row["evidence_ids"]
            assert isinstance(references, list)
            row["evidence_ids"] = sorted(id_to_token[str(value)] for value in references)

    control_evidence = payload["control_evidence"]
    assert isinstance(control_evidence, dict)
    for field, references in control_evidence.items():
        assert isinstance(references, list)
        control_evidence[field] = sorted(id_to_token[str(value)] for value in references)

    termination_evidence = payload["termination_evidence_ids"]
    assert isinstance(termination_evidence, list)
    payload["termination_evidence_ids"] = sorted(
        id_to_token[str(value)] for value in termination_evidence
    )
    return payload


def _persistence_value(state: PublicDSUSState, invariant: TransitionInvariant) -> JsonValue:
    subject = invariant.subject
    if invariant.invariant_type == TransitionInvariantType.PRESERVE_CONTROL:
        field = subject.removeprefix("control.")
        return getattr(state.control, field).value
    if invariant.invariant_type == TransitionInvariantType.PRESERVE_CONSTRAINT:
        identifier = subject.removeprefix("constraint.")
        item = next((value for value in state.constraints if value.constraint_id == identifier), None)
        return item.model_dump(mode="json") if item is not None else None
    if invariant.invariant_type == TransitionInvariantType.PRESERVE_UNRESOLVED_AXIS:
        identifier = subject.removeprefix("unresolved_axis.")
        item = next((value for value in state.unresolved_axes if value.axis_id == identifier), None)
        return item.model_dump(mode="json") if item is not None else None
    if invariant.invariant_type == TransitionInvariantType.PRESERVE_REJECTED_ITEM:
        identifier = subject.removeprefix("rejected_item.")
        item = next((value for value in state.rejected_items if value.item_id == identifier), None)
        return item.model_dump(mode="json") if item is not None else None
    if invariant.invariant_type == TransitionInvariantType.PRESERVE_TERMINATION:
        return state.termination_reason.value if state.termination_reason is not None else None
    raise ValueError("not a persistence invariant")


def _persistence_score(
    next_state: PublicDSUSState | None,
    invariants: Sequence[TransitionInvariant],
) -> tuple[float, int, int]:
    relevant = [
        item
        for item in invariants
        if item.invariant_type
        in {
            TransitionInvariantType.PRESERVE_CONTROL,
            TransitionInvariantType.PRESERVE_CONSTRAINT,
            TransitionInvariantType.PRESERVE_UNRESOLVED_AXIS,
            TransitionInvariantType.PRESERVE_REJECTED_ITEM,
            TransitionInvariantType.PRESERVE_TERMINATION,
        }
    ]
    if next_state is None:
        return 0.0, len(relevant), 0
    passed = sum(_persistence_value(next_state, item) == item.expected for item in relevant)
    return (passed / len(relevant) if relevant else 1.0), len(relevant), passed


def _gold_output(private: PrivateRLVerifierRecord) -> dict[str, JsonValue]:
    trainer = private.trainer_private
    return {
        "schema_version": PATCH_SCHEMA_VERSION,
        "base_revision": trainer.gold_base_revision,
        "evidence": [item.model_dump(mode="json") for item in trainer.gold_evidence],
        "delta": [item.model_dump(mode="json") for item in trainer.gold_delta],
    }


def _length_and_minimality(
    predicted: PredictedPatchPayload | None,
    *,
    raw_output_chars: int,
    private: PrivateRLVerifierRecord,
    config: TVRLVRRewardConfig,
) -> tuple[PatchLengthAudit, float, float]:
    gold = _gold_output(private)
    gold_chars = len(canonical_json(gold))
    gold_operations = len(private.trainer_private.gold_delta)
    gold_evidence = len(private.trainer_private.gold_evidence)
    if predicted is None:
        audit = PatchLengthAudit(
            raw_output_chars=raw_output_chars,
            canonical_output_chars=None,
            gold_output_chars=gold_chars,
            predicted_operation_count=None,
            gold_operation_count=gold_operations,
            predicted_evidence_count=None,
            gold_evidence_count=gold_evidence,
            extra_operation_count=None,
            extra_evidence_count=None,
            within_absolute_limit=raw_output_chars <= config.max_output_chars,
            within_soft_gold_limit=False,
        )
        return audit, 0.0, min(1.0, config.max_output_chars / max(raw_output_chars, 1))

    canonical_chars = len(canonical_json(predicted.canonical_payload()))
    predicted_operations = len(predicted.delta)
    predicted_evidence = len(predicted.evidence)
    soft_limit = max(gold_chars, int(gold_chars * config.soft_gold_length_ratio))
    operation_efficiency = min(1.0, gold_operations / predicted_operations)
    evidence_efficiency = min(1.0, gold_evidence / predicted_evidence)
    minimality = (operation_efficiency + evidence_efficiency) / 2
    absolute_score = min(1.0, config.max_output_chars / max(raw_output_chars, 1))
    relative_score = min(1.0, soft_limit / max(canonical_chars, 1))
    length_score = min(absolute_score, relative_score)
    audit = PatchLengthAudit(
        raw_output_chars=raw_output_chars,
        canonical_output_chars=canonical_chars,
        gold_output_chars=gold_chars,
        predicted_operation_count=predicted_operations,
        gold_operation_count=gold_operations,
        predicted_evidence_count=predicted_evidence,
        gold_evidence_count=gold_evidence,
        extra_operation_count=predicted_operations - gold_operations,
        extra_evidence_count=predicted_evidence - gold_evidence,
        within_absolute_limit=raw_output_chars <= config.max_output_chars,
        within_soft_gold_limit=canonical_chars <= soft_limit,
    )
    return audit, minimality, length_score


def score_verifier_sample(
    public_record: PublicRLPromptRecord,
    private_record: PrivateRLVerifierRecord,
    rollout: RolloutOutputRecord,
    *,
    config: TVRLVRRewardConfig | None = None,
) -> VerifierScoreRecord:
    """Score one already-generated rollout; trainer-private data is never rendered."""

    config = config or TVRLVRRewardConfig()
    if not (
        public_record.sample_id == private_record.sample_id == rollout.sample_id
        and public_record.public_prompt_sha256
        == private_record.public_prompt_sha256
        == rollout.public_prompt_sha256
    ):
        raise ValueError("sample_id/public_prompt hash join mismatch")
    if forbidden_public_key_paths(public_record.public_prompt.model_dump(mode="json", by_alias=True)):
        raise ValueError("public RL prompt contains trainer-private keys")
    # Revalidates the frozen private gold program without serializing it into model input.
    RLVRTrainingView(
        public_prompt_sha256=public_record.public_prompt_sha256,
        public_prompt=public_record.public_prompt,
        trainer_private=private_record.trainer_private,
    )

    errors: list[str] = []
    predicted: PredictedPatchPayload | None = None
    patch: StatePatch | None = None
    raw_chars = (
        len(rollout.model_output)
        if isinstance(rollout.model_output, str)
        else len(canonical_json(rollout.model_output))
    )
    try:
        predicted, raw_chars = parse_model_patch(rollout.model_output)
        patch = predicted.as_patch()
        schema_score = 1.0
    except (json.JSONDecodeError, TypeError, ValueError):
        schema_score = 0.0
        errors.append("schema_invalid")

    evidence_score = 0.0
    operation_score = 0.0
    revision_score = 0.0
    replay_score = 0.0
    candidate_next: PublicDSUSState | None = None
    previous = public_record.public_prompt.previous_public_state
    if patch is not None:
        grounded, evidence_errors = _evidence_grounded(
            patch.evidence,
            public=public_record,
            private=private_record,
        )
        evidence_score = float(grounded)
        errors.extend(evidence_errors)
        operation_score = float(_operation_legality(previous, patch))
        if not operation_score:
            errors.append("operation_illegal")
        revision_score = float(patch.base_revision == previous.revision)
        if not revision_score:
            errors.append("base_revision_mismatch")
        try:
            candidate_next = reduce_state(previous, patch)
            replay_score = 1.0
        except (TypeError, ValueError):
            errors.append("reducer_not_executable")

    gold_next = private_record.trainer_private.gold_next_state
    control_score = float(
        candidate_next is not None
        and canonical_json(_control_payload(candidate_next))
        == canonical_json(_control_payload(gold_next))
    )
    constraint_score = float(
        candidate_next is not None
        and canonical_json(_constraint_payload(candidate_next))
        == canonical_json(_constraint_payload(gold_next))
    )
    equivalence_score = float(
        candidate_next is not None
        and canonical_json(_alpha_normalized_state_payload(candidate_next))
        == canonical_json(_alpha_normalized_state_payload(gold_next))
    )
    if candidate_next is not None and not control_score:
        errors.append("next_control_mismatch")
    if candidate_next is not None and not constraint_score:
        errors.append("next_constraint_mismatch")
    if candidate_next is not None and not equivalence_score:
        errors.append("next_state_not_equivalent")

    persistence_score, persistence_checked, persistence_passed = _persistence_score(
        candidate_next,
        private_record.trainer_private.transition_invariants,
    )
    if candidate_next is not None and persistence_score < 1.0:
        errors.append("persistence_violation")
    length_audit, minimality_score, length_score = _length_and_minimality(
        predicted,
        raw_output_chars=raw_chars,
        private=private_record,
        config=config,
    )
    if minimality_score < 1.0:
        errors.append("nonminimal_patch")
    if length_score < 1.0:
        errors.append("length_audit_failed")

    canonical_components = {
        VerifierComponent.SCHEMA.value: schema_score,
        VerifierComponent.EVIDENCE.value: evidence_score,
        VerifierComponent.OPERATION_LEGALITY.value: operation_score,
        VerifierComponent.REVISION.value: revision_score,
        VerifierComponent.REPLAY.value: replay_score,
        VerifierComponent.CONTROL_MATCH.value: control_score,
        VerifierComponent.CONSTRAINT_MATCH.value: constraint_score,
        VerifierComponent.STATE_EQUIVALENCE.value: equivalence_score,
        VerifierComponent.PERSISTENCE.value: persistence_score,
        VerifierComponent.MINIMALITY.value: minimality_score,
        VerifierComponent.LENGTH.value: length_score,
    }
    hard_gate_passed = all(
        canonical_components[component.value] == 1.0
        for component in config.hard_gate_components
    )
    weight_total = sum(config.weights.values())
    dense_reward = sum(
        config.weights[component] * canonical_components[component.value]
        for component in VerifierComponent
    ) / weight_total
    total_reward = dense_reward if hard_gate_passed else 0.0
    accepted = bool(
        hard_gate_passed
        and equivalence_score == 1.0
        and persistence_score == 1.0
    )
    components = {
        **canonical_components,
        "concision": (minimality_score + length_score) / 2,
    }
    return VerifierScoreRecord(
        sample_id=rollout.sample_id,
        public_prompt_sha256=rollout.public_prompt_sha256,
        accepted=accepted,
        total_reward=round(total_reward, 12),
        components={key: round(value, 12) for key, value in components.items()},
        hard_gate_passed=hard_gate_passed,
        length_audit=length_audit,
        persistence_checked=persistence_checked,
        persistence_passed=persistence_passed,
        error_codes=sorted(set(errors)),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _unique_index(values: Iterable[RecordT], *, label: str) -> dict[str, RecordT]:
    output: dict[str, RecordT] = {}
    for value in values:
        sample_id = str(value.sample_id)
        if sample_id in output:
            raise ValueError(f"duplicate {label} sample_id: {sample_id}")
        output[sample_id] = value
    return output


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl_text(values: Iterable[StrictModel]) -> str:
    return "".join(canonical_json(value.model_dump(mode="json")) + "\n" for value in values)


def _batch_summary(scores: Sequence[VerifierScoreRecord]) -> VerifierBatchSummary:
    count = len(scores)
    component_names = sorted(scores[0].components) if scores else []
    component_means = {
        name: sum(score.components[name] for score in scores) / count
        for name in component_names
    }
    errors = Counter(code for score in scores for code in score.error_codes)
    accepted = sum(score.accepted for score in scores)
    return VerifierBatchSummary(
        sample_count=count,
        accepted_count=accepted,
        acceptance_rate=accepted / count if count else 0.0,
        mean_total_reward=sum(score.total_reward for score in scores) / count if count else 0.0,
        component_means=component_means,
        error_code_counts=dict(sorted(errors.items())),
        public_prompt_private_leakage_count=0,
    )


def score_verifier_files(
    public_path: Path,
    private_path: Path,
    rollout_path: Path,
    *,
    config: TVRLVRRewardConfig | None = None,
    require_complete: bool = True,
    output_root: Path | None = None,
    progress: Callable[[str], None] = print,
) -> VerifierBatchResult:
    """Join three separate files by sample_id and score in deterministic order."""

    config = config or TVRLVRRewardConfig()
    progress("步骤 1/4：读取 public RL prompts 与已经生成完毕的模型输出。")
    public = _unique_index(
        (PublicRLPromptRecord.model_validate(row) for row in _read_jsonl(public_path)),
        label="public prompt",
    )
    rollouts = _unique_index(
        (RolloutOutputRecord.model_validate(row) for row in _read_jsonl(rollout_path)),
        label="rollout",
    )
    leakage = sum(
        bool(forbidden_public_key_paths(record.public_prompt.model_dump(mode="json", by_alias=True)))
        for record in public.values()
    )
    if leakage:
        raise ValueError(f"public prompt pool contains {leakage} trainer-private leakage records")
    progress(f"步骤 1/4 完成：读取 {len(public)} 个 public prompts / {len(rollouts)} 个 outputs。")

    progress("步骤 2/4：生成结束后才读取 trainer-private verifier，并按 sample_id 严格连接。")
    private = _unique_index(
        (PrivateRLVerifierRecord.model_validate(row) for row in _read_jsonl(private_path)),
        label="trainer-private verifier",
    )
    if set(public) != set(private):
        raise ValueError("public/private verifier pools have different sample_id sets")
    if require_complete and set(rollouts) != set(public):
        raise ValueError("rollout sample_id set is incomplete or contains extras")
    if not set(rollouts) <= set(public):
        raise ValueError("rollout file contains unknown sample_ids")
    for sample_id in public:
        if public[sample_id].public_prompt_sha256 != private[sample_id].public_prompt_sha256:
            raise ValueError(f"public/private prompt hash mismatch: {sample_id}")
    progress("步骤 2/4 完成：private 数据仅进入 post-generation verifier join。")

    progress("步骤 3/4：依次执行 schema/evidence/legality/replay/state/persistence/concision 验证链。")
    scores = [
        score_verifier_sample(
            public[sample_id],
            private[sample_id],
            rollouts[sample_id],
            config=config,
        )
        for sample_id in sorted(rollouts)
    ]
    summary = _batch_summary(scores)
    progress(
        f"步骤 3/4 完成：accepted={summary.accepted_count}/{summary.sample_count}，"
        f"mean reward={summary.mean_total_reward:.4f}。"
    )

    run_dir: Path | None = None
    if output_root is not None:
        progress("步骤 4/4：写入内容寻址的 trainer-private score 工件与审计 manifest。")
        input_ledger = {
            "public_prompt_pool": _file_sha256(public_path),
            "trainer_private_verifier": _file_sha256(private_path),
            "rollout_outputs": _file_sha256(rollout_path),
        }
        identity = {
            "verifier_chain_version": VERIFIER_CHAIN_VERSION,
            "reward_config": config.model_dump(mode="json"),
            "require_complete": require_complete,
            "inputs": input_ledger,
        }
        run_id = f"tv-rlvr-score-{content_sha256(identity)[:16]}"
        run_dir = output_root.resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        score_path = run_dir / "verifier_scores.private.jsonl"
        summary_path = run_dir / "summary.json"
        score_path.write_text(_jsonl_text(scores), encoding="utf-8")
        summary_path.write_text(canonical_json(summary.model_dump(mode="json")) + "\n", encoding="utf-8")
        outputs = [
            {
                "relative_path": path.name,
                "sha256": _file_sha256(path),
                "byte_size": path.stat().st_size,
                "row_count": (
                    sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
                    if path.suffix == ".jsonl"
                    else None
                ),
            }
            for path in (score_path, summary_path)
        ]
        manifest = {
            "schema_version": VERIFIER_CHAIN_VERSION,
            "artifact_role": "dsus_trainer_side_tv_rlvr_scores",
            "run_id": run_id,
            "identity": identity,
            "trainer_private_join_stage": "post_generation_verifier_only",
            "model_input_source": "public_prompt_pool_only",
            "external_calls": False,
            "outputs": outputs,
        }
        (run_dir / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        progress(f"步骤 4/4 完成：离线 verifier 工件写入 {run_dir}。")
    else:
        progress("步骤 4/4：未指定 output_root，仅返回内存评分；未写文件。")
    return VerifierBatchResult(
        scores=scores,
        summary=summary,
        run_dir=str(run_dir) if run_dir is not None else None,
    )


__all__ = [
    "PatchLengthAudit",
    "PredictedPatchPayload",
    "REWARD_CONFIG_VERSION",
    "RolloutOutputRecord",
    "TVRLVRRewardConfig",
    "VERIFIER_CHAIN_VERSION",
    "VerifierBatchResult",
    "VerifierBatchSummary",
    "VerifierComponent",
    "VerifierScoreRecord",
    "parse_model_patch",
    "score_verifier_files",
    "score_verifier_sample",
]
