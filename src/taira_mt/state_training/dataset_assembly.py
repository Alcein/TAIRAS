"""Assemble compiled DSUS transitions into leakage-safe training datasets.

This layer is intentionally offline.  It consumes frozen compiled-transition
JSONL plus private root-lineage JSONL, audits source isolation, de-duplicates
logical samples, and writes paired SFT/RLVR views.  It neither generates data
with an LLM nor launches model training.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.state_compiler import (
    AddUnresolvedAxisOperation,
    ConstraintStatus,
    ControlField,
    ControlState,
    EvidenceReference,
    EvidenceSource,
    InitializeStateOperation,
    InteractionPhase,
    NoopOperation,
    PublicDSUSState,
    SetControlOperation,
    StatePatch,
    TerminateOperation,
    UpsertConstraintOperation,
    reduce_state,
)
from taira_mt.state_training.authoring import InteractionMechanism
from taira_mt.state_training.logical_compiler import (
    CompiledLogicalTrajectory,
    CompiledStateTransition,
    PublicDialogueMessage,
)
from taira_mt.state_training.training_views import (
    DifficultySlice,
    DSUSPublicPrompt,
    RLVRTrainerPrivate,
    SFTTrainingView,
    project_transition_training_views,
    render_rlvr_public_prompt,
    render_sft_private_target,
    render_sft_public_prompt,
)

DATASET_ASSEMBLY_SCHEMA_VERSION = "dsus_state_compiler_dataset_v1"
ASSEMBLY_SPEC_SCHEMA_VERSION = "dsus_dataset_assembly_spec_v1"

ProgressCallback = Callable[[str], None]


class DatasetSplit(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class SourceFamily(str, Enum):
    LEGACY_RICH = "legacy_rich"
    COMPLEX_MULTITURN = "complex_multiturn"
    MULTIFAMILY_WARMUP = "multifamily_warmup"


class CurriculumStage(str, Enum):
    WARMUP = "warmup"
    SEMANTIC_GROUNDING = "semantic_grounding"
    MULTITURN_TRANSITION = "multiturn_transition"
    HARD_STATE_REPAIR = "hard_state_repair"


class IdentityAxis(str, Enum):
    QUERY = "query"
    ITEM = "item"
    USER = "user"


class TransitionDifficulty(str, Enum):
    STANDARD = "standard"
    NOOP = "noop"
    REVISION = "revision"
    CONFLICT = "conflict"
    LONG_HISTORY = "long_history"
    FEEDBACK_REPAIR = "feedback_repair"
    MULTI_OPERATION = "multi_operation"
    TERMINAL = "terminal"


class TrainingViewKind(str, Enum):
    """Paper-facing SFT curriculum role of one public-input/target pair."""

    INCREMENTAL = "incremental"
    COLD_RECONSTRUCTION = "cold_reconstruction"
    NOOP = "noop"
    SELF_RECOVERY = "self_recovery"


class Stage0TrainingViewMix(StrictModel):
    """Deterministic SFT-v0 mix; self-recovery remains reserved for SFT-v1."""

    enabled: bool = False
    incremental_percent: int = Field(default=65, ge=1, le=100)
    cold_reconstruction_percent: int = Field(default=15, ge=0, le=100)
    noop_percent: int = Field(default=10, ge=0, le=100)
    reserved_self_recovery_percent: int = Field(default=10, ge=0, le=100)
    derivation_version: Literal["deterministic_public_curriculum_v1"] = (
        "deterministic_public_curriculum_v1"
    )

    @model_validator(mode="after")
    def validate_paper_mix(self) -> Stage0TrainingViewMix:
        total = (
            self.incremental_percent
            + self.cold_reconstruction_percent
            + self.noop_percent
            + self.reserved_self_recovery_percent
        )
        if total != 100:
            raise ValueError("stage-0/future-recovery curriculum percentages must sum to 100")
        return self


class ConflictingPromptPolicy(str, Enum):
    """Policy for public inputs with more than one incompatible gold state."""

    ERROR = "error"
    DROP_ALL = "drop_all"


class CompiledSourceSpec(StrictModel):
    """One future rich/complex/warmup source and its private lineage table."""

    source_family: SourceFamily
    curriculum_stage: CurriculumStage
    compiled_transitions_path: str = Field(min_length=1)
    lineage_path: str = Field(min_length=1)
    allow_missing_identity_axes: list[IdentityAxis] = Field(default_factory=list)
    excluded_root_ids: list[str] = Field(default_factory=list)

    @field_validator("compiled_transitions_path", "lineage_path")
    @classmethod
    def validate_path_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source path cannot be blank")
        return value.strip()

    @field_validator("allow_missing_identity_axes")
    @classmethod
    def canonicalize_missing_axes(cls, value: list[IdentityAxis]) -> list[IdentityAxis]:
        if len(value) != len(set(value)):
            raise ValueError("allow_missing_identity_axes cannot contain duplicates")
        return sorted(value, key=lambda item: item.value)

    @field_validator("excluded_root_ids")
    @classmethod
    def canonicalize_excluded_roots(cls, value: list[str]) -> list[str]:
        normalized = [root_id.strip() for root_id in value]
        if any(not root_id for root_id in normalized):
            raise ValueError("excluded_root_ids cannot contain blank identifiers")
        if len(normalized) != len(set(normalized)):
            raise ValueError("excluded_root_ids cannot contain duplicates")
        return sorted(normalized)


class DatasetAssemblySpec(StrictModel):
    schema_version: Literal[ASSEMBLY_SPEC_SCHEMA_VERSION] = ASSEMBLY_SPEC_SCHEMA_VERSION
    sources: list[CompiledSourceSpec] = Field(min_length=1)
    latest_observation_count: int | None = Field(default=None, ge=1)
    conflicting_prompt_policy: ConflictingPromptPolicy = ConflictingPromptPolicy.ERROR
    stage0_training_view_mix: Stage0TrainingViewMix = Field(
        default_factory=Stage0TrainingViewMix
    )

    @model_validator(mode="after")
    def validate_source_descriptors(self) -> DatasetAssemblySpec:
        identities = [
            (
                source.source_family,
                source.curriculum_stage,
                source.compiled_transitions_path,
                source.lineage_path,
            )
            for source in self.sources
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("assembly spec contains duplicate source descriptors")
        return self


class CompiledTransitionInput(CompiledStateTransition):
    """Strict flattened row emitted by the deterministic logical compiler."""

    trajectory_id: str = Field(min_length=1)
    transition_index: int = Field(ge=0)
    mechanism: InteractionMechanism


class RootLineage(StrictModel):
    """Canonical private split identity; raw query/user/item values are not exported."""

    source_input_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_family: SourceFamily
    curriculum_stage: CurriculumStage
    root_id: str = Field(min_length=1)
    split: DatasetSplit
    domain: str = Field(min_length=1)
    normalized_query: str | None = None
    item_id: str | None = None
    user_id: str | None = None
    unavailable_identity_axes: list[IdentityAxis] = Field(default_factory=list)

    @field_validator("root_id")
    @classmethod
    def validate_root_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("lineage identifiers cannot be blank")
        return value.strip()

    @field_validator("domain")
    @classmethod
    def normalize_domain(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("lineage domain cannot be blank")
        return value.strip().casefold()

    @field_validator("normalized_query", "item_id", "user_id")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_unavailable_axes(self) -> RootLineage:
        expected = sorted(
            (
                axis
                for axis, value in (
                    (IdentityAxis.QUERY, self.normalized_query),
                    (IdentityAxis.ITEM, self.item_id),
                    (IdentityAxis.USER, self.user_id),
                )
                if value is None
            ),
            key=lambda item: item.value,
        )
        if self.unavailable_identity_axes != expected:
            raise ValueError("unavailable_identity_axes disagrees with canonical lineage fields")
        return self


class LeakageAxisAudit(StrictModel):
    axis: Literal["root", "query", "item", "user"]
    available_root_count: int = Field(ge=0)
    unavailable_root_count: int = Field(ge=0)
    unique_identity_count: int = Field(ge=0)
    cross_split_overlap_count: int = Field(ge=0)
    overlapping_identity_sha256: list[str]
    passed: bool

    @model_validator(mode="after")
    def validate_pass_flag(self) -> LeakageAxisAudit:
        if self.passed != (self.cross_split_overlap_count == 0):
            raise ValueError("axis pass flag disagrees with overlap count")
        if self.cross_split_overlap_count != len(self.overlapping_identity_sha256):
            raise ValueError("overlap count disagrees with overlap hash list")
        return self


class LeakageAuditReport(StrictModel):
    passed: bool
    root_count: int = Field(ge=0)
    split_root_counts: dict[str, int]
    axes: list[LeakageAxisAudit] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_pass_flag(self) -> LeakageAuditReport:
        if self.passed != all(axis.passed for axis in self.axes):
            raise ValueError("leakage report pass flag disagrees with axis reports")
        return self


class SampleRegistryRecord(StrictModel):
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: DatasetSplit
    source_family: SourceFamily
    curriculum_stage: CurriculumStage
    domain: str
    mechanism: InteractionMechanism
    difficulty: TransitionDifficulty
    root_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_identity_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    item_identity_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    user_identity_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class PublicRLPromptRecord(StrictModel):
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt: DSUSPublicPrompt

    @model_validator(mode="after")
    def validate_hash(self) -> PublicRLPromptRecord:
        if self.public_prompt_sha256 != self.public_prompt.content_hash():
            raise ValueError("RL public prompt hash mismatch")
        return self


class PrivateRLVerifierRecord(StrictModel):
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trainer_private: RLVRTrainerPrivate


class TrainingViewRegistryRecord(StrictModel):
    """Private provenance for deterministic curriculum views."""

    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: DatasetSplit
    view_kind: TrainingViewKind
    derived: bool
    parent_public_prompt_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class SFTTokenizerRecord(StrictModel):
    """Loader must tokenize only model_input and apply loss only to assistant_target."""

    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_input: str = Field(min_length=1)
    assistant_target: str = Field(min_length=1)


class RLVRTokenizerRecord(StrictModel):
    """Public rollout input; verifier material is intentionally absent."""

    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_input: str = Field(min_length=1)


class DatasetAssemblyResult(StrictModel):
    build_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_dir: str
    input_transition_count: int = Field(ge=0)
    output_sample_count: int = Field(ge=0)
    deduplicated_count: int = Field(ge=0)
    ambiguous_prompt_dropped_count: int = Field(default=0, ge=0)
    excluded_root_count: int = Field(default=0, ge=0)
    excluded_transition_count: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class _LoadedSource:
    spec: CompiledSourceSpec
    source_input_id: str
    compiled_sha256: str
    lineage_sha256: str
    compiled_rows: tuple[CompiledTransitionInput, ...]
    lineage_by_root: Mapping[str, RootLineage]
    raw_duplicate_count: int
    excluded_root_count: int
    excluded_transition_count: int


@dataclass(frozen=True)
class _Candidate:
    source: _LoadedSource
    row: CompiledTransitionInput
    lineage: RootLineage
    difficulty: TransitionDifficulty
    sft: SFTTrainingView
    view_kind: TrainingViewKind = TrainingViewKind.INCREMENTAL
    derived: bool = False
    parent_public_prompt_sha256: str | None = None


def _normalize_query(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _split(value: object) -> DatasetSplit:
    normalized = str(value).strip().casefold()
    aliases = {
        "train": DatasetSplit.TRAIN,
        "dev": DatasetSplit.DEV,
        "validation": DatasetSplit.DEV,
        "val": DatasetSplit.DEV,
        "test": DatasetSplit.TEST,
        "compiler_test": DatasetSplit.TEST,
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported DSUS source split: {value!r}")
    return aliases[normalized]


def _first_text(payload: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _read_jsonl_bytes(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows, digest


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def load_assembly_spec(path: Path) -> tuple[DatasetAssemblySpec, Path]:
    """Load a strict spec and return the directory used for relative paths."""

    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DatasetAssemblySpec.model_validate(payload), path.parent


def _lineage_record(
    payload: Mapping[str, Any],
    *,
    source: CompiledSourceSpec,
    source_input_id: str,
) -> RootLineage:
    root_id = _first_text(payload, ("root_id",))
    domain = _first_text(payload, ("domain",))
    if root_id is None or domain is None:
        raise ValueError("lineage row requires non-empty root_id and domain")
    normalized_query = _first_text(payload, ("normalized_query",))
    if normalized_query is None:
        query = _first_text(payload, ("query", "initial_query"))
        normalized_query = (_normalize_query(query) or None) if query is not None else None
    else:
        normalized_query = _normalize_query(normalized_query) or None
    item_id = _first_text(payload, ("anchor_item_id", "item_id", "source_item_id"))
    user_id = _first_text(payload, ("pseudonymous_user_id", "user_id"))
    values = {
        IdentityAxis.QUERY: normalized_query,
        IdentityAxis.ITEM: item_id,
        IdentityAxis.USER: user_id,
    }
    unavailable = sorted(
        (axis for axis, value in values.items() if value is None),
        key=lambda item: item.value,
    )
    forbidden_missing = sorted(
        set(unavailable).difference(source.allow_missing_identity_axes),
        key=lambda item: item.value,
    )
    if forbidden_missing:
        raise ValueError(f"root {root_id} lacks required lineage axes: {[axis.value for axis in forbidden_missing]}")
    return RootLineage(
        source_input_id=source_input_id,
        source_family=source.source_family,
        curriculum_stage=source.curriculum_stage,
        root_id=root_id,
        split=_split(payload.get("split")),
        domain=domain,
        normalized_query=normalized_query,
        item_id=item_id,
        user_id=user_id,
        unavailable_identity_axes=unavailable,
    )


def _load_source(source: CompiledSourceSpec, *, base_dir: Path) -> _LoadedSource:
    compiled_path = _resolve_path(base_dir, source.compiled_transitions_path)
    lineage_path = _resolve_path(base_dir, source.lineage_path)
    compiled_payloads, compiled_sha = _read_jsonl_bytes(compiled_path)
    lineage_payloads, lineage_sha = _read_jsonl_bytes(lineage_path)
    source_identity = {
        "source_family": source.source_family.value,
        "curriculum_stage": source.curriculum_stage.value,
        "allow_missing_identity_axes": [axis.value for axis in source.allow_missing_identity_axes],
        "compiled_sha256": compiled_sha,
        "lineage_sha256": lineage_sha,
    }
    if source.excluded_root_ids:
        source_identity["excluded_root_ids"] = source.excluded_root_ids
    source_input_id = content_sha256(source_identity)

    rows_by_identity: dict[tuple[str, int], CompiledTransitionInput] = {}
    raw_duplicate_count = 0
    for payload in compiled_payloads:
        row = CompiledTransitionInput.model_validate(payload)
        identity = (row.trajectory_id, row.transition_index)
        prior = rows_by_identity.get(identity)
        if prior is None:
            rows_by_identity[identity] = row
        elif prior.model_dump(mode="json") == row.model_dump(mode="json"):
            raw_duplicate_count += 1
        else:
            raise ValueError(f"conflicting compiled row identity: {identity}")
    all_rows = tuple(rows_by_identity.values())
    compiled_root_ids = {row.root_id for row in all_rows}
    unknown_exclusions = sorted(set(source.excluded_root_ids).difference(compiled_root_ids))
    if unknown_exclusions:
        raise ValueError(
            f"excluded roots are absent from compiled input: {unknown_exclusions[:10]}"
        )
    excluded_root_set = set(source.excluded_root_ids)
    excluded_transition_count = sum(
        row.root_id in excluded_root_set for row in all_rows
    )
    rows = sorted(
        (row for row in all_rows if row.root_id not in excluded_root_set),
        key=lambda row: (row.trajectory_id, row.transition_index),
    )
    if not rows:
        raise ValueError(f"compiled source is empty: {compiled_path}")
    required_roots = {row.root_id for row in rows}
    lineage_by_root: dict[str, RootLineage] = {}
    for payload in lineage_payloads:
        root_id = _first_text(payload, ("root_id",))
        if root_id not in required_roots:
            continue
        lineage = _lineage_record(payload, source=source, source_input_id=source_input_id)
        if lineage.root_id in lineage_by_root:
            raise ValueError(f"duplicate lineage row for root {lineage.root_id}")
        lineage_by_root[lineage.root_id] = lineage
    missing = sorted(required_roots.difference(lineage_by_root))
    if missing:
        raise ValueError(f"compiled roots are absent from lineage input: {missing[:10]}")

    grouped: dict[str, list[CompiledTransitionInput]] = defaultdict(list)
    for row in rows:
        grouped[row.trajectory_id].append(row)
    for trajectory_id, trajectory_rows in sorted(grouped.items()):
        indices = [row.transition_index for row in trajectory_rows]
        if indices != list(range(len(indices))):
            raise ValueError(f"trajectory {trajectory_id} has non-contiguous transition indices")
        first = trajectory_rows[0]
        if any(
            row.root_id != first.root_id or row.variant_id != first.variant_id or row.mechanism != first.mechanism
            for row in trajectory_rows
        ):
            raise ValueError(f"trajectory {trajectory_id} mixes root, variant, or mechanism")
        CompiledLogicalTrajectory(
            root_id=first.root_id,
            variant_id=first.variant_id,
            mechanism=first.mechanism,
            transitions=[_as_transition(row) for row in trajectory_rows],
        )
    return _LoadedSource(
        spec=source,
        source_input_id=source_input_id,
        compiled_sha256=compiled_sha,
        lineage_sha256=lineage_sha,
        compiled_rows=tuple(rows),
        lineage_by_root=lineage_by_root,
        raw_duplicate_count=raw_duplicate_count,
        excluded_root_count=len(excluded_root_set),
        excluded_transition_count=excluded_transition_count,
    )


def _as_transition(row: CompiledTransitionInput) -> CompiledStateTransition:
    """Drop flattened-row wrapper metadata through an explicit allow-list."""

    return CompiledStateTransition(
        root_id=row.root_id,
        variant_id=row.variant_id,
        step_id=row.step_id,
        public_history=row.public_history,
        previous_state=row.previous_state,
        gold_patch=row.gold_patch,
        gold_next_state=row.gold_next_state,
    )


def _identity_value(lineage: RootLineage, axis: str) -> str | None:
    if axis == "root":
        return lineage.root_id
    if axis == "query":
        return lineage.normalized_query
    if axis == "item":
        return None if lineage.item_id is None else f"{lineage.domain}\0{lineage.item_id}"
    if axis == "user":
        return None if lineage.user_id is None else f"{lineage.domain}\0{lineage.user_id}"
    raise AssertionError(axis)


def _identity_hash(axis: str, value: str) -> str:
    return content_sha256({"axis": axis, "value": value})


def audit_split_leakage(lineages: Sequence[RootLineage]) -> LeakageAuditReport:
    """Audit root/query/(domain,item)/(domain,user) equality across splits."""

    axes: list[LeakageAxisAudit] = []
    for axis in ("root", "query", "item", "user"):
        split_by_value: dict[str, set[DatasetSplit]] = defaultdict(set)
        unavailable = 0
        for lineage in lineages:
            value = _identity_value(lineage, axis)
            if value is None:
                unavailable += 1
                continue
            split_by_value[value].add(lineage.split)
        overlaps = sorted(_identity_hash(axis, value) for value, splits in split_by_value.items() if len(splits) > 1)
        axes.append(
            LeakageAxisAudit(
                axis=axis,
                available_root_count=len(lineages) - unavailable,
                unavailable_root_count=unavailable,
                unique_identity_count=len(split_by_value),
                cross_split_overlap_count=len(overlaps),
                overlapping_identity_sha256=overlaps,
                passed=not overlaps,
            )
        )
    split_counts = Counter(lineage.split.value for lineage in lineages)
    return LeakageAuditReport(
        passed=all(axis.passed for axis in axes),
        root_count=len(lineages),
        split_root_counts=dict(sorted(split_counts.items())),
        axes=axes,
    )


def _difficulty(row: CompiledTransitionInput) -> TransitionDifficulty:
    operations = row.gold_patch.operations
    operation_names = {operation.op for operation in operations}
    if operation_names == {"noop"}:
        return TransitionDifficulty.NOOP
    if operation_names.intersection({"supersede_constraint", "remove_constraint"}):
        return TransitionDifficulty.REVISION
    if row.mechanism == InteractionMechanism.CONFLICT:
        return TransitionDifficulty.CONFLICT
    if len(row.public_history) >= 9:
        return TransitionDifficulty.LONG_HISTORY
    if any(
        operation.op == "record_feedback" and operation.feedback_type.value in {"rejection", "correction"}
        for operation in operations
    ):
        return TransitionDifficulty.FEEDBACK_REPAIR
    if len(operations) >= 5:
        return TransitionDifficulty.MULTI_OPERATION
    if "terminate" in operation_names:
        return TransitionDifficulty.TERMINAL
    return TransitionDifficulty.STANDARD


def _cold_reconstruction_transition(
    row: CompiledTransitionInput,
) -> CompiledTransitionInput | None:
    """Rebuild a current DSUS snapshot from EMPTY using only its public history.

    Cold restart establishes a fresh optimistic-concurrency epoch, so the
    reconstructed state has revision 1 while preserving the complete semantic
    state and public evidence memory.  States requiring two mutations of the
    same entity (for example, a retained superseded constraint) are skipped
    because DSUS deliberately forbids that ambiguity inside one patch.
    """

    state = row.gold_next_state
    if not row.previous_state.initialized or len(row.public_history) < 3:
        return None
    if not state.initialized or not state.evidence or state.rejected_items:
        return None
    if any(item.status == ConstraintStatus.SUPERSEDED for item in state.constraints):
        return None

    all_evidence_ids = [item.evidence_id for item in state.evidence]
    operations: list[Any] = [
        InitializeStateOperation(evidence_ids=all_evidence_ids)
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
        for item in state.constraints
    )
    operations.extend(
        AddUnresolvedAxisOperation(
            axis_id=item.axis_id,
            axis_type=item.axis_type,
            description=item.description,
            options=item.options,
            evidence_ids=item.evidence_ids,
        )
        for item in state.unresolved_axes
    )

    defaults = ControlState()
    terminal = state.control.interaction_phase == InteractionPhase.TERMINAL
    terminal_sets_commitment = (
        state.termination_reason is not None
        and state.termination_reason.value in {"accepted", "abandoned"}
    )
    for field in ControlField:
        if terminal and field == ControlField.INTERACTION_PHASE:
            continue
        if terminal_sets_commitment and field == ControlField.COMMITMENT_STATUS:
            continue
        value = getattr(state.control, field.value)
        evidence_ids = state.control_evidence.get(field, [])
        if value == getattr(defaults, field.value) and not evidence_ids:
            continue
        if not evidence_ids:
            return None
        operations.append(
            SetControlOperation(
                field=field,
                value=value.value,
                evidence_ids=evidence_ids,
            )
        )

    if terminal:
        if state.termination_reason is None or not state.termination_evidence_ids:
            return None
        if (
            state.control_evidence.get(ControlField.INTERACTION_PHASE, [])
            != state.termination_evidence_ids
        ):
            return None
        if terminal_sets_commitment and (
            state.control_evidence.get(ControlField.COMMITMENT_STATUS, [])
            != state.termination_evidence_ids
        ):
            return None
        operations.append(
            TerminateOperation(
                reason=state.termination_reason,
                evidence_ids=state.termination_evidence_ids,
            )
        )

    try:
        patch = StatePatch(
            base_revision=0,
            evidence=[item.model_copy(deep=True) for item in state.evidence],
            operations=operations,
        )
        reconstructed = reduce_state(PublicDSUSState.empty(), patch)
    except (ValueError, TypeError):
        return None
    expected = state.model_copy(update={"revision": 1}, deep=True)
    if reconstructed.canonical_json() != expected.canonical_json():
        return None

    suffix = content_sha256(
        {
            "kind": TrainingViewKind.COLD_RECONSTRUCTION.value,
            "trajectory_id": row.trajectory_id,
            "transition_index": row.transition_index,
            "gold_next_state": state.canonical_payload(),
        }
    )[:16]
    return CompiledTransitionInput(
        trajectory_id=f"{row.trajectory_id}:cold:{suffix}",
        transition_index=row.transition_index,
        mechanism=row.mechanism,
        root_id=row.root_id,
        variant_id=f"{row.variant_id}:cold:{suffix}",
        step_id=f"{row.step_id}:cold:{suffix}",
        public_history=[item.model_copy(deep=True) for item in row.public_history],
        previous_state=PublicDSUSState.empty(),
        gold_patch=patch,
        gold_next_state=reconstructed,
    )


_NOOP_USER_UTTERANCES = (
    "Okay, thanks.",
    "Got it, thank you.",
    "Understood.",
    "That makes sense, thanks.",
)
_NOOP_SYSTEM_UTTERANCE = (
    "I have noted the current preferences. Let me know if you want to change anything."
)


def _noop_supplement_transition(row: CompiledTransitionInput) -> CompiledTransitionInput:
    """Append a natural acknowledgement whose grounded state operation is noop."""

    seed = content_sha256(
        {
            "kind": TrainingViewKind.NOOP.value,
            "trajectory_id": row.trajectory_id,
            "transition_index": row.transition_index,
            "public_history": [item.model_dump(mode="json") for item in row.public_history],
        }
    )
    user_text = _NOOP_USER_UTTERANCES[int(seed[:8], 16) % len(_NOOP_USER_UTTERANCES)]
    next_turn = max(item.turn_id for item in row.public_history) + 1
    evidence_id = f"e_noop_{seed[:20]}"
    evidence = EvidenceReference(
        evidence_id=evidence_id,
        turn_id=next_turn,
        source=EvidenceSource.USER,
        quote=user_text,
    )
    patch = StatePatch(
        base_revision=row.gold_next_state.revision,
        evidence=[evidence],
        operations=[
            NoopOperation(
                reason="acknowledgement adds no decision-relevant user information",
                evidence_ids=[evidence_id],
            )
        ],
    )
    history = [item.model_copy(deep=True) for item in row.public_history]
    history.extend(
        [
            PublicDialogueMessage(
                turn_id=next_turn,
                role="system",
                text=_NOOP_SYSTEM_UTTERANCE,
                action_type="derived_noop_context",
            ),
            PublicDialogueMessage(
                turn_id=next_turn,
                role="user",
                text=user_text,
            ),
        ]
    )
    next_state = reduce_state(row.gold_next_state, patch)
    return CompiledTransitionInput(
        trajectory_id=f"{row.trajectory_id}:noop:{seed[:16]}",
        transition_index=row.transition_index,
        mechanism=row.mechanism,
        root_id=row.root_id,
        variant_id=f"{row.variant_id}:noop:{seed[:16]}",
        step_id=f"{row.step_id}:noop:{seed[:16]}",
        public_history=history,
        previous_state=row.gold_next_state.model_copy(deep=True),
        gold_patch=patch,
        gold_next_state=next_state,
    )


def _ratio_count(base_count: int, numerator: int, denominator: int) -> int:
    return (base_count * numerator + denominator // 2) // denominator


def _derived_rank(candidate: _Candidate, kind: TrainingViewKind) -> str:
    return content_sha256(
        {
            "kind": kind.value,
            "prompt": candidate.sft.public_prompt_sha256,
            "target": candidate.sft.private_target.assistant_payload(),
        }
    )


def _stage0_curriculum_supplements(
    candidates: Sequence[_Candidate],
    *,
    mix: Stage0TrainingViewMix,
) -> tuple[list[_Candidate], dict[str, dict[str, int]]]:
    """Add deterministic cold/noop views in the pre-recovery 65:15:10 ratio."""

    if not mix.enabled:
        return list(candidates), {}

    combined = list(candidates)
    summary: dict[str, dict[str, int]] = {}
    signatures = {_sample_signature(candidate) for candidate in combined}
    prompt_targets: dict[str, str] = {
        prompt_hash: target_hash for prompt_hash, target_hash in signatures
    }

    def accept(parent: _Candidate, row: CompiledTransitionInput, kind: TrainingViewKind) -> bool:
        pair = project_transition_training_views(
            _as_transition(row),
            difficulty_slice=DifficultySlice(row.mechanism.value),
            latest_observation_count=None,
        )
        derived = _Candidate(
            source=parent.source,
            row=row,
            lineage=parent.lineage,
            difficulty=(
                TransitionDifficulty.NOOP if kind == TrainingViewKind.NOOP else _difficulty(row)
            ),
            sft=pair.sft,
            view_kind=kind,
            derived=True,
            parent_public_prompt_sha256=parent.sft.public_prompt_sha256,
        )
        prompt_hash, target_hash = _sample_signature(derived)
        if (prompt_hash, target_hash) in signatures:
            return False
        existing_target = prompt_targets.get(prompt_hash)
        if existing_target is not None and existing_target != target_hash:
            return False
        signatures.add((prompt_hash, target_hash))
        prompt_targets[prompt_hash] = target_hash
        combined.append(derived)
        return True

    for split in DatasetSplit:
        split_candidates = [item for item in candidates if item.lineage.split == split]
        observed_noops = [
            item for item in split_candidates if item.view_kind == TrainingViewKind.NOOP
        ]
        incremental = [
            item for item in split_candidates if item.view_kind == TrainingViewKind.INCREMENTAL
        ]
        cold_target = _ratio_count(
            len(incremental),
            mix.cold_reconstruction_percent,
            mix.incremental_percent,
        )
        noop_target = _ratio_count(
            len(incremental),
            mix.noop_percent,
            mix.incremental_percent,
        )

        cold_added = 0
        if cold_target:
            for parent in sorted(
                incremental,
                key=lambda item: _derived_rank(item, TrainingViewKind.COLD_RECONSTRUCTION),
            ):
                row = _cold_reconstruction_transition(parent.row)
                if row is not None and accept(
                    parent,
                    row,
                    TrainingViewKind.COLD_RECONSTRUCTION,
                ):
                    cold_added += 1
                    if cold_added == cold_target:
                        break
        if cold_added != cold_target:
            raise ValueError(
                f"split {split.value} has only {cold_added}/{cold_target} unique "
                "cold-reconstruction views"
            )

        noop_needed = max(0, noop_target - len(observed_noops))
        noop_added = 0
        if noop_needed:
            for parent in sorted(
                incremental,
                key=lambda item: _derived_rank(item, TrainingViewKind.NOOP),
            ):
                if accept(parent, _noop_supplement_transition(parent.row), TrainingViewKind.NOOP):
                    noop_added += 1
                    if noop_added == noop_needed:
                        break
        if noop_added != noop_needed:
            raise ValueError(
                f"split {split.value} has only {noop_added}/{noop_needed} unique noop views"
            )

        stage0_count = len(incremental) + len(observed_noops) + cold_added + noop_added
        summary[split.value] = {
            "incremental": len(incremental),
            "cold_reconstruction": cold_added,
            "noop": len(observed_noops) + noop_added,
            "derived_noop": noop_added,
            "reserved_self_recovery": _ratio_count(
                stage0_count,
                mix.reserved_self_recovery_percent,
                100 - mix.reserved_self_recovery_percent,
            ),
        }
    return sorted(combined, key=_candidate_sort_key), summary


def _candidate_sort_key(candidate: _Candidate) -> tuple[str, ...]:
    lineage = candidate.lineage
    return (
        lineage.split.value,
        lineage.source_family.value,
        lineage.curriculum_stage.value,
        lineage.domain,
        _identity_hash("root", lineage.root_id),
        candidate.row.variant_id,
        f"{candidate.row.transition_index:08d}",
    )


def _sample_signature(candidate: _Candidate) -> tuple[str, str]:
    prompt_hash = candidate.sft.public_prompt_sha256
    target_hash = content_sha256(
        {
            "sft_target": candidate.sft.private_target.assistant_payload(),
            "gold_next_state": candidate.row.gold_next_state.canonical_payload(),
        }
    )
    return prompt_hash, target_hash


def _registry_record(candidate: _Candidate, *, sample_id: str) -> SampleRegistryRecord:
    lineage = candidate.lineage

    def digest(axis: str, value: str | None) -> str | None:
        return None if value is None else _identity_hash(axis, value)

    return SampleRegistryRecord(
        sample_id=sample_id,
        public_prompt_sha256=candidate.sft.public_prompt_sha256,
        split=lineage.split,
        source_family=lineage.source_family,
        curriculum_stage=lineage.curriculum_stage,
        domain=lineage.domain,
        mechanism=candidate.row.mechanism,
        difficulty=candidate.difficulty,
        root_identity_sha256=digest("root", lineage.root_id),
        query_identity_sha256=digest("query", lineage.normalized_query),
        item_identity_sha256=digest(
            "item",
            None if lineage.item_id is None else f"{lineage.domain}\0{lineage.item_id}",
        ),
        user_identity_sha256=digest(
            "user",
            None if lineage.user_id is None else f"{lineage.domain}\0{lineage.user_id}",
        ),
    )


def _jsonl_text(records: Iterable[Any]) -> str:
    rows = []
    for record in records:
        payload = record.model_dump(mode="json") if hasattr(record, "model_dump") else record
        rows.append(canonical_json(payload))
    return "".join(f"{row}\n" for row in rows)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_idempotently(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite different dataset content: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _strata(registry: Sequence[SampleRegistryRecord]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in DatasetSplit:
        rows = [row for row in registry if row.split == split]
        output[split.value] = {
            "sample_count": len(rows),
            "domain_counts": dict(sorted(Counter(row.domain for row in rows).items())),
            "mechanism_counts": dict(sorted(Counter(row.mechanism.value for row in rows).items())),
            "difficulty_counts": dict(sorted(Counter(row.difficulty.value for row in rows).items())),
            "source_family_counts": dict(sorted(Counter(row.source_family.value for row in rows).items())),
            "curriculum_stage_counts": dict(sorted(Counter(row.curriculum_stage.value for row in rows).items())),
        }
    return output


def _orthogonal_view_coverage(candidates: Sequence[_Candidate]) -> dict[str, Any]:
    """Count overlapping paper slices without forcing one difficulty label."""

    output: dict[str, Any] = {}
    for split in DatasetSplit:
        rows = [item for item in candidates if item.lineage.split == split]
        output[split.value] = {
            "sample_count": len(rows),
            "view_kind_counts": dict(
                sorted(Counter(item.view_kind.value for item in rows).items())
            ),
            "incremental_initialization_count": sum(
                item.view_kind == TrainingViewKind.INCREMENTAL
                and not item.row.previous_state.initialized
                for item in rows
            ),
            "incremental_noninitial_count": sum(
                item.view_kind == TrainingViewKind.INCREMENTAL
                and item.row.previous_state.initialized
                for item in rows
            ),
            "revision_operation_count": sum(
                any(
                    operation.op in {"supersede_constraint", "remove_constraint"}
                    for operation in item.row.gold_patch.operations
                )
                for item in rows
            ),
            "conflict_mechanism_count": sum(
                item.row.mechanism == InteractionMechanism.CONFLICT for item in rows
            ),
            "long_history_count": sum(len(item.row.public_history) >= 9 for item in rows),
        }
    return output


def assemble_training_dataset(
    *,
    spec: DatasetAssemblySpec,
    spec_base_dir: Path,
    output_root: Path,
    progress: ProgressCallback = print,
) -> DatasetAssemblyResult:
    """Build a content-addressed SFT/RLVR dataset without external calls."""

    if spec.stage0_training_view_mix.enabled and spec.latest_observation_count is not None:
        raise ValueError(
            "cold reconstruction requires full public_history; latest_observation_count must be null"
        )

    progress("步骤 1/5：读取 compiled transition 与私有 lineage，并验证逐轨迹回放链。")
    loaded = [_load_source(source, base_dir=spec_base_dir) for source in spec.sources]
    all_lineages: list[RootLineage] = []
    owners_by_root: dict[str, set[str]] = defaultdict(set)
    for source in loaded:
        for root_id, lineage in source.lineage_by_root.items():
            owners_by_root[root_id].add(source.source_input_id)
            all_lineages.append(lineage)
    unique_input_transition_count = sum(len(source.compiled_rows) for source in loaded)
    raw_duplicate_count = sum(source.raw_duplicate_count for source in loaded)
    excluded_root_count = sum(source.excluded_root_count for source in loaded)
    excluded_transition_count = sum(source.excluded_transition_count for source in loaded)
    input_transition_count = (
        unique_input_transition_count
        + raw_duplicate_count
        + excluded_transition_count
    )
    progress(
        f"步骤 1/5 完成：加载 {len(all_lineages)} 个 roots、"
        f"{input_transition_count} 个输入 records（唯一 transition="
        f"{unique_input_transition_count}，质量过滤 root={excluded_root_count} / "
        f"transition={excluded_transition_count}）。"
    )

    progress("步骤 2/5：审计 root/query/item/user 跨 split 泄漏，并冻结 train/dev/test。")
    leakage = audit_split_leakage(all_lineages)
    if not leakage.passed:
        failed = [axis.axis for axis in leakage.axes if not axis.passed]
        raise ValueError(f"cross-split source leakage detected on axes: {failed}")
    multiply_owned = sorted(root_id for root_id, owners in owners_by_root.items() if len(owners) > 1)
    if multiply_owned:
        raise ValueError(f"roots appear in multiple source inputs: {multiply_owned[:10]}")
    progress(f"步骤 2/5 完成：四类 lineage overlap 均为 0；root 分布={leakage.split_root_counts}。")

    progress("步骤 3/5：投影共享 SFT/RLVR public prompt，检查冲突标签并执行内容去重。")
    candidates: list[_Candidate] = []
    for source in loaded:
        for row in source.compiled_rows:
            difficulty = _difficulty(row)
            pair = project_transition_training_views(
                _as_transition(row),
                difficulty_slice=DifficultySlice(row.mechanism.value),
                latest_observation_count=spec.latest_observation_count,
            )
            candidates.append(
                _Candidate(
                    source=source,
                    row=row,
                    lineage=source.lineage_by_root[row.root_id],
                    difficulty=difficulty,
                    sft=pair.sft,
                    view_kind=(
                        TrainingViewKind.NOOP
                        if difficulty == TransitionDifficulty.NOOP
                        else TrainingViewKind.INCREMENTAL
                    ),
                )
            )
    candidates_by_prompt_and_target: dict[str, dict[str, list[_Candidate]]] = defaultdict(lambda: defaultdict(list))
    for candidate in sorted(candidates, key=_candidate_sort_key):
        prompt_hash, target_hash = _sample_signature(candidate)
        candidates_by_prompt_and_target[prompt_hash][target_hash].append(candidate)
    conflicting_prompt_hashes = sorted(
        prompt_hash for prompt_hash, by_target in candidates_by_prompt_and_target.items() if len(by_target) > 1
    )
    if conflicting_prompt_hashes and spec.conflicting_prompt_policy == ConflictingPromptPolicy.ERROR:
        raise ValueError(f"the same public prompt maps to conflicting private targets: {conflicting_prompt_hashes[0]}")
    conflicting_prompt_set = set(conflicting_prompt_hashes)
    ambiguous_prompt_dropped_count = sum(
        len(candidates_for_target)
        for prompt_hash in conflicting_prompt_hashes
        for candidates_for_target in candidates_by_prompt_and_target[prompt_hash].values()
    )
    eligible_candidates = [
        candidate
        for prompt_hash, by_target in sorted(candidates_by_prompt_and_target.items())
        if prompt_hash not in conflicting_prompt_set
        for candidates_for_target in by_target.values()
        for candidate in candidates_for_target
    ]
    unique_by_signature: dict[tuple[str, str], _Candidate] = {}
    semantic_duplicate_count = 0
    for candidate in sorted(eligible_candidates, key=_candidate_sort_key):
        prompt_hash, target_hash = _sample_signature(candidate)
        signature = (prompt_hash, target_hash)
        prior = unique_by_signature.get(signature)
        if prior is not None:
            if prior.lineage.split != candidate.lineage.split:
                raise ValueError("an exact training sample occurs across dataset splits")
            semantic_duplicate_count += 1
            continue
        unique_by_signature[signature] = candidate
    unique_candidates = sorted(unique_by_signature.values(), key=_candidate_sort_key)
    deduplicated_count = raw_duplicate_count + semantic_duplicate_count
    base_unique_count = len(unique_candidates)
    unique_candidates, stage0_mix_by_split = _stage0_curriculum_supplements(
        unique_candidates,
        mix=spec.stage0_training_view_mix,
    )
    derived_view_count = len(unique_candidates) - base_unique_count
    progress(
        f"步骤 3/5 完成：保留 {len(unique_candidates)} 个唯一 paired samples，"
        f"移除 {deduplicated_count} 个重复记录，"
        f"丢弃 {ambiguous_prompt_dropped_count} 个公开输入不足以唯一判定标签的记录，"
        f"确定性补充 {derived_view_count} 个 cold/noop 训练视图。"
    )

    source_ledger = []
    for source in sorted(loaded, key=lambda item: item.source_input_id):
        ledger_row: dict[str, Any] = {
            "source_input_id": source.source_input_id,
            "source_family": source.spec.source_family.value,
            "curriculum_stage": source.spec.curriculum_stage.value,
            "allow_missing_identity_axes": [axis.value for axis in source.spec.allow_missing_identity_axes],
            "compiled_sha256": source.compiled_sha256,
            "lineage_sha256": source.lineage_sha256,
            "compiled_transition_count": len(source.compiled_rows),
        }
        if source.spec.excluded_root_ids:
            ledger_row.update(
                {
                    "excluded_root_count": source.excluded_root_count,
                    "excluded_transition_count": source.excluded_transition_count,
                    "excluded_root_ids_sha256": content_sha256(
                        source.spec.excluded_root_ids
                    ),
                }
            )
        source_ledger.append(ledger_row)
    identity = {
        "schema_version": DATASET_ASSEMBLY_SCHEMA_VERSION,
        "latest_observation_count": spec.latest_observation_count,
        "source_ledger": source_ledger,
        "deduplication": "canonical_public_prompt_plus_gold_target_v1",
        "split_protocol": "root_query_domain-item_domain-user_disjoint_v1",
        "external_calls": False,
    }
    if spec.conflicting_prompt_policy != ConflictingPromptPolicy.ERROR:
        identity["conflicting_prompt_policy"] = spec.conflicting_prompt_policy.value
    if spec.stage0_training_view_mix.enabled:
        identity["stage0_training_view_mix"] = spec.stage0_training_view_mix.model_dump(
            mode="json"
        )
    build_id = content_sha256(identity)
    run_dir = output_root.resolve() / f"dsus-training-data-{build_id[:16]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    progress("步骤 4/5：按 split 写入 paired views、私有 verifier 与安全 tokenizer exports。")
    registry: list[SampleRegistryRecord] = []
    view_registry: list[TrainingViewRegistryRecord] = []
    sft_by_split: dict[DatasetSplit, list[SFTTrainingView]] = defaultdict(list)
    rl_public_by_split: dict[DatasetSplit, list[PublicRLPromptRecord]] = defaultdict(list)
    rl_private_by_split: dict[DatasetSplit, list[PrivateRLVerifierRecord]] = defaultdict(list)
    sft_tokenizer_by_split: dict[DatasetSplit, list[SFTTokenizerRecord]] = defaultdict(list)
    rl_tokenizer_by_split: dict[DatasetSplit, list[RLVRTokenizerRecord]] = defaultdict(list)
    for candidate in unique_candidates:
        prompt_hash, target_hash = _sample_signature(candidate)
        sample_id = content_sha256(
            {
                "public_prompt_sha256": prompt_hash,
                "target_sha256": target_hash,
                "source_family": candidate.lineage.source_family.value,
                "curriculum_stage": candidate.lineage.curriculum_stage.value,
            }
        )
        split = candidate.lineage.split
        registry.append(_registry_record(candidate, sample_id=sample_id))
        view_registry.append(
            TrainingViewRegistryRecord(
                sample_id=sample_id,
                split=split,
                view_kind=candidate.view_kind,
                derived=candidate.derived,
                parent_public_prompt_sha256=candidate.parent_public_prompt_sha256,
            )
        )
        pair = project_transition_training_views(
            _as_transition(candidate.row),
            difficulty_slice=DifficultySlice(candidate.row.mechanism.value),
            latest_observation_count=spec.latest_observation_count,
        )
        sft_by_split[split].append(pair.sft)
        rl_public_by_split[split].append(
            PublicRLPromptRecord(
                sample_id=sample_id,
                public_prompt_sha256=pair.rlvr.public_prompt_sha256,
                public_prompt=pair.rlvr.public_prompt,
            )
        )
        rl_private_by_split[split].append(
            PrivateRLVerifierRecord(
                sample_id=sample_id,
                public_prompt_sha256=pair.rlvr.public_prompt_sha256,
                trainer_private=pair.rlvr.trainer_private,
            )
        )
        sft_tokenizer_by_split[split].append(
            SFTTokenizerRecord(
                sample_id=sample_id,
                public_prompt_sha256=pair.sft.public_prompt_sha256,
                model_input=render_sft_public_prompt(pair.sft),
                assistant_target=render_sft_private_target(pair.sft),
            )
        )
        rl_tokenizer_by_split[split].append(
            RLVRTokenizerRecord(
                sample_id=sample_id,
                public_prompt_sha256=pair.rlvr.public_prompt_sha256,
                model_input=render_rlvr_public_prompt(pair.rlvr),
            )
        )

    files: dict[str, str] = {"sample_registry.private.jsonl": _jsonl_text(registry)}
    if spec.stage0_training_view_mix.enabled:
        files["training_view_registry.private.jsonl"] = _jsonl_text(view_registry)
    for split in DatasetSplit:
        files[f"sft_{split.value}.jsonl"] = _jsonl_text(sft_by_split[split])
        files[f"rl_prompt_pool.{split.value}.public.jsonl"] = _jsonl_text(rl_public_by_split[split])
        files[f"rl_verifier.{split.value}.private.jsonl"] = _jsonl_text(rl_private_by_split[split])
        files[f"tokenizer_sft.{split.value}.private.jsonl"] = _jsonl_text(sft_tokenizer_by_split[split])
        files[f"tokenizer_rlvr.{split.value}.public.jsonl"] = _jsonl_text(rl_tokenizer_by_split[split])
    for name, text in sorted(files.items()):
        _write_idempotently(run_dir / name, text)
    if spec.conflicting_prompt_policy != ConflictingPromptPolicy.ERROR:
        ambiguity_audit = {
            "policy": spec.conflicting_prompt_policy.value,
            "conflicting_prompt_group_count": len(conflicting_prompt_hashes),
            "dropped_transition_count": ambiguous_prompt_dropped_count,
            "groups": [
                {
                    "public_prompt_sha256": prompt_hash,
                    "target_count": len(candidates_by_prompt_and_target[prompt_hash]),
                    "transition_count": sum(
                        len(values) for values in candidates_by_prompt_and_target[prompt_hash].values()
                    ),
                }
                for prompt_hash in conflicting_prompt_hashes
            ],
        }
        _write_idempotently(
            run_dir / "ambiguous_prompt_audit.private.json",
            canonical_json(ambiguity_audit) + "\n",
        )
    progress(f"步骤 4/5 完成：训练视图已写入 {run_dir}。")

    progress("步骤 5/5：汇总 domain/mechanism/difficulty/curriculum 分层并冻结 manifest。")
    leakage_payload = leakage.model_dump(mode="json")
    _write_idempotently(run_dir / "leakage_audit.json", canonical_json(leakage_payload) + "\n")
    summary = {
        "schema_version": DATASET_ASSEMBLY_SCHEMA_VERSION,
        "build_id": build_id,
        "input_transition_count": input_transition_count,
        "output_sample_count": len(registry),
        "raw_duplicate_count": raw_duplicate_count,
        "semantic_duplicate_count": semantic_duplicate_count,
        "deduplicated_count": deduplicated_count,
        "excluded_root_count": excluded_root_count,
        "excluded_transition_count": excluded_transition_count,
        "root_count": len(all_lineages),
        "strata_by_split": _strata(registry),
        "leakage_audit_passed": leakage.passed,
        "external_calls": False,
    }
    if spec.stage0_training_view_mix.enabled:
        summary.update(
            {
                "stage0_training_view_mix": spec.stage0_training_view_mix.model_dump(
                    mode="json"
                ),
                "stage0_mix_by_split": stage0_mix_by_split,
                "orthogonal_view_coverage_by_split": _orthogonal_view_coverage(
                    unique_candidates
                ),
                "derived_training_view_count": derived_view_count,
                "self_recovery_status": "reserved_for_sft_v1_not_fabricated",
            }
        )
    if spec.conflicting_prompt_policy != ConflictingPromptPolicy.ERROR:
        summary.update(
            {
                "conflicting_prompt_policy": spec.conflicting_prompt_policy.value,
                "ambiguous_prompt_group_count": len(conflicting_prompt_hashes),
                "ambiguous_prompt_dropped_count": ambiguous_prompt_dropped_count,
            }
        )
    _write_idempotently(run_dir / "summary.json", canonical_json(summary) + "\n")
    outputs = []
    for path in sorted(item for item in run_dir.iterdir() if item.is_file()):
        if path.name == "manifest.json":
            continue
        outputs.append(
            {
                "relative_path": path.name,
                "sha256": _file_sha256(path),
                "byte_size": path.stat().st_size,
                "row_count": (
                    sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
                    if path.suffix == ".jsonl"
                    else None
                ),
            }
        )
    manifest = {
        "schema_version": DATASET_ASSEMBLY_SCHEMA_VERSION,
        "artifact_role": "dsus_state_compiler_training_dataset",
        "build_id": build_id,
        "status": "complete",
        "identity": identity,
        "outputs": outputs,
    }
    _write_idempotently(run_dir / "manifest.json", canonical_json(manifest) + "\n")
    progress(f"步骤 5/5 完成：sample={len(registry)}，deduplicated={deduplicated_count}，external_calls=False。")
    return DatasetAssemblyResult(
        build_id=build_id,
        run_dir=str(run_dir),
        input_transition_count=input_transition_count,
        output_sample_count=len(registry),
        deduplicated_count=deduplicated_count,
        ambiguous_prompt_dropped_count=ambiguous_prompt_dropped_count,
        excluded_root_count=excluded_root_count,
        excluded_transition_count=excluded_transition_count,
    )
