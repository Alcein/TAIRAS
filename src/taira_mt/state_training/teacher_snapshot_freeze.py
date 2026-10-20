"""Minimal, offline final freeze for accepted full-state teacher snapshots."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator

from taira_mt.models import StrictModel
from taira_mt.provenance import content_sha256
from taira_mt.state_compiler.runtime import validate_public_evidence
from taira_mt.state_training.dataset_assembly import PublicRLPromptRecord
from taira_mt.state_training.teacher_projection import (
    TeacherCurrentStateSnapshot,
    TeacherProjectionValidatedRecord,
)
from taira_mt.state_training.teacher_projection_consensus import TeacherConsensusRecord
from taira_mt.state_training.training_views import DSUSPublicPrompt

TEACHER_SNAPSHOT_FREEZE_VERSION = "dsus_teacher_snapshot_freeze_v1"
TEACHER_SNAPSHOT_SFT_VERSION = "dsus_teacher_full_snapshot_sft_v1"
TEACHER_SNAPSHOT_PROVENANCE_VERSION = "dsus_teacher_snapshot_provenance_v1"
TEACHER_TRAJECTORY_SPLIT_VERSION = "dsus_teacher_trajectory_split_v1"
DEFAULT_TEACHER_SNAPSHOT_SPLIT_SEED = "dsus-real-full-snapshot-sft-v1"

FreezeSplit = Literal["train", "dev", "test"]
SnapshotSource = Literal["base", "repair"]
SelectionReason = Literal[
    "repair_pass_selected",
    "repair_selected_not_passed",
    "base_pass_selected",
    "no_passed_snapshot",
]


class FrozenTeacherSnapshotRecord(StrictModel):
    schema_version: Literal[TEACHER_SNAPSHOT_FREEZE_VERSION] = TEACHER_SNAPSHOT_FREEZE_VERSION
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    order_index: int = Field(ge=0)
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trajectory_id: str = Field(min_length=1)
    source_turn_index: int = Field(ge=0)
    split: FreezeSplit
    source: SnapshotSource
    repair_layer_name: str | None = Field(default=None, min_length=1)
    source_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot: TeacherCurrentStateSnapshot

    @model_validator(mode="after")
    def validate_snapshot_hash(self) -> FrozenTeacherSnapshotRecord:
        if self.snapshot_sha256 != content_sha256(self.snapshot.model_dump(mode="json")):
            raise ValueError("frozen snapshot hash mismatch")
        if (self.source == "repair") != (self.repair_layer_name is not None):
            raise ValueError("repair source must name exactly one repair layer")
        return self


class FullSnapshotSFTRecord(StrictModel):
    schema_version: Literal[TEACHER_SNAPSHOT_SFT_VERSION] = TEACHER_SNAPSHOT_SFT_VERSION
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    order_index: int = Field(ge=0)
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trajectory_id: str = Field(min_length=1)
    source_turn_index: int = Field(ge=0)
    split: FreezeSplit
    input: DSUSPublicPrompt
    target: TeacherCurrentStateSnapshot
    target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_content_hashes(self) -> FullSnapshotSFTRecord:
        if self.public_prompt_sha256 != self.input.content_hash():
            raise ValueError("SFT public prompt hash mismatch")
        if self.target_sha256 != content_sha256(self.target.model_dump(mode="json")):
            raise ValueError("SFT target hash mismatch")
        return self


class RepairLayerDecision(StrictModel):
    layer_name: str = Field(min_length=1)
    selected: bool
    validated_record_present: bool
    passed: bool
    snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class SnapshotFreezeProvenanceRecord(StrictModel):
    schema_version: Literal[TEACHER_SNAPSHOT_PROVENANCE_VERSION] = (
        TEACHER_SNAPSHOT_PROVENANCE_VERSION
    )
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    order_index: int = Field(ge=0)
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trajectory_id: str = Field(min_length=1)
    source_turn_index: int = Field(ge=0)
    split: FreezeSplit
    base_record_present: bool
    base_pass: bool
    repair_selected: bool
    repair_pass: bool
    repair_layer_trace: list[RepairLayerDecision]
    final_source: SnapshotSource | None = None
    final_repair_layer_name: str | None = Field(default=None, min_length=1)
    selection_reason: SelectionReason
    final_snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class SnapshotDuplicateOverrideRecord(StrictModel):
    schema_version: Literal["dsus_teacher_snapshot_duplicate_override_v1"] = (
        "dsus_teacher_snapshot_duplicate_override_v1"
    )
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_sources: list[str] = Field(min_length=2)
    candidate_snapshot_sha256: dict[str, str]
    selected_source: str | None = None
    resolution: Literal["later_layer_overrides", "later_layer_selected_but_not_passed"]


class TeacherTrajectorySplitRecord(StrictModel):
    schema_version: Literal[TEACHER_TRAJECTORY_SPLIT_VERSION] = TEACHER_TRAJECTORY_SPLIT_VERSION
    trajectory_id: str = Field(min_length=1)
    split: FreezeSplit
    split_rank: int = Field(ge=0)
    split_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_seed: str = Field(min_length=1)


class TeacherSnapshotFreezeSummary(StrictModel):
    schema_version: Literal["dsus_teacher_snapshot_freeze_summary_v1"] = (
        "dsus_teacher_snapshot_freeze_summary_v1"
    )
    split_seed: str = Field(min_length=1)
    requested_count: int = Field(ge=1)
    base_record_count: int = Field(ge=0)
    base_pass_count: int = Field(ge=0)
    repair_selected_count: int = Field(ge=0)
    repair_pass_count: int = Field(ge=0)
    repair_layer_counts: dict[str, dict[str, int]]
    final_base_count: int = Field(ge=0)
    final_repair_count: int = Field(ge=0)
    final_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    duplicate_override_count: int = Field(ge=0)
    trajectory_count: int = Field(ge=1)
    trajectory_split_counts: dict[str, int]
    final_sample_split_counts: dict[str, int]
    fabricated_count: Literal[0] = 0
    trajectory_split_leakage_count: Literal[0] = 0
    external_model_calls: Literal[False] = False
    upstream_artifacts_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> TeacherSnapshotFreezeSummary:
        if self.final_count != self.final_base_count + self.final_repair_count:
            raise ValueError("final source counts do not sum to final_count")
        if self.requested_count != self.final_count + self.missing_count:
            raise ValueError("final + missing does not cover requested universe")
        if self.trajectory_count != sum(self.trajectory_split_counts.values()):
            raise ValueError("trajectory split counts do not cover trajectories")
        if self.final_count != sum(self.final_sample_split_counts.values()):
            raise ValueError("sample split counts do not cover final snapshots")
        return self


@dataclass(frozen=True)
class TeacherSnapshotFreezeResult:
    final_snapshots: tuple[FrozenTeacherSnapshotRecord, ...]
    sft_records: tuple[FullSnapshotSFTRecord, ...]
    missing_ids: list[str]
    duplicate_overrides: tuple[SnapshotDuplicateOverrideRecord, ...]
    provenance: tuple[SnapshotFreezeProvenanceRecord, ...]
    trajectory_splits: tuple[TeacherTrajectorySplitRecord, ...]
    summary: TeacherSnapshotFreezeSummary


@dataclass(frozen=True)
class RepairSnapshotLayer:
    name: str
    records: Sequence[TeacherProjectionValidatedRecord]
    selected_ids: Sequence[str]
    pass_ids: Sequence[str]


def _unique_ids(values: Sequence[str], *, owner: str) -> list[str]:
    output = [value.strip() for value in values]
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in output):
        raise ValueError(f"{owner} contains a non-SHA256 sample ID")
    if len(output) != len(set(output)):
        raise ValueError(f"{owner} contains duplicate sample IDs")
    return output


def _index_by_sample(records: Sequence[object], *, owner: str) -> dict[str, object]:
    output: dict[str, object] = {}
    for record in records:
        sample_id = str(record.sample_id)  # type: ignore[attr-defined]
        if sample_id in output:
            raise ValueError(f"{owner} contains duplicate sample ID: {sample_id}")
        output[sample_id] = record
    return output


def _lineage_index(
    records: Sequence[Mapping[str, object]],
    *,
    requested: set[str],
    public_by_id: Mapping[str, PublicRLPromptRecord],
) -> dict[str, tuple[str, int]]:
    output: dict[str, tuple[str, int]] = {}
    seen_all: set[str] = set()
    for row in records:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("lineage contains a row without sample_id")
        if sample_id in seen_all:
            raise ValueError(f"lineage contains duplicate sample ID: {sample_id}")
        seen_all.add(sample_id)
        if sample_id not in public_by_id:
            raise ValueError(f"lineage sample is absent from merged public prompts: {sample_id}")
        trajectory_id = row.get("trajectory_id", row.get("source_conversation_id"))
        turn = row.get("source_turn_index", row.get("trajectory_turn_index"))
        if not isinstance(trajectory_id, str) or not trajectory_id.strip():
            raise ValueError(f"lineage lacks trajectory ID for {sample_id}")
        if not isinstance(turn, int) or turn < 0:
            raise ValueError(f"lineage lacks nonnegative source turn for {sample_id}")
        prompt_hash = row.get("public_prompt_sha256")
        if prompt_hash is not None and prompt_hash != public_by_id[sample_id].public_prompt_sha256:
            raise ValueError(f"lineage/public prompt hash mismatch for {sample_id}")
        if sample_id not in requested:
            continue
        output[sample_id] = (trajectory_id.strip(), turn)
    missing = sorted(requested.difference(output))
    if missing:
        raise ValueError(f"lineage misses requested sample IDs: {missing[:3]}")
    return output


def deterministic_trajectory_splits(
    trajectory_ids: Sequence[str],
    *,
    split_seed: str,
) -> tuple[TeacherTrajectorySplitRecord, ...]:
    if not split_seed.strip():
        raise ValueError("split seed cannot be blank")
    unique = sorted(set(trajectory_ids))
    if not unique:
        raise ValueError("trajectory set cannot be empty")
    ranked = sorted(
        (
            hashlib.sha256(f"{split_seed}\0{trajectory_id}".encode()).hexdigest(),
            trajectory_id,
        )
        for trajectory_id in unique
    )
    train_count = int(len(ranked) * 0.8)
    dev_count = int(len(ranked) * 0.1)
    output: list[TeacherTrajectorySplitRecord] = []
    for rank, (split_hash, trajectory_id) in enumerate(ranked):
        split: FreezeSplit
        if rank < train_count:
            split = "train"
        elif rank < train_count + dev_count:
            split = "dev"
        else:
            split = "test"
        output.append(
            TeacherTrajectorySplitRecord(
                trajectory_id=trajectory_id,
                split=split,
                split_rank=rank,
                split_hash=split_hash,
                split_seed=split_seed,
            )
        )
    return tuple(sorted(output, key=lambda item: item.trajectory_id))


def freeze_teacher_snapshot_sft(
    *,
    public_records: Sequence[PublicRLPromptRecord],
    lineage_records: Sequence[Mapping[str, object]],
    base_records: Sequence[TeacherConsensusRecord],
    base_pass_ids: Sequence[str],
    repair_records: Sequence[TeacherProjectionValidatedRecord] = (),
    repair_pass_ids: Sequence[str] = (),
    repair_selected_ids: Sequence[str] | None = None,
    repair_layers: Sequence[RepairSnapshotLayer] | None = None,
    all_sample_ids: Sequence[str],
    split_seed: str = DEFAULT_TEACHER_SNAPSHOT_SPLIT_SEED,
) -> TeacherSnapshotFreezeResult:
    """Resolve pass-only base/repair snapshots without synthesis or fallback guessing."""

    ordered_ids = _unique_ids(all_sample_ids, owner="all sample IDs")
    requested = set(ordered_ids)
    public_untyped = _index_by_sample(public_records, owner="public prompts")
    public_by_id = {
        key: value for key, value in public_untyped.items() if isinstance(value, PublicRLPromptRecord)
    }
    missing_public = sorted(requested.difference(public_by_id))
    if missing_public:
        raise ValueError(f"public prompts miss requested sample IDs: {missing_public[:3]}")
    lineage = _lineage_index(
        lineage_records,
        requested=requested,
        public_by_id=public_by_id,
    )

    base_untyped = _index_by_sample(base_records, owner="base consensus")
    base_all = {
        key: value for key, value in base_untyped.items() if isinstance(value, TeacherConsensusRecord)
    }
    if any(record.status != "accepted" or record.consensus_snapshot is None for record in base_all.values()):
        raise ValueError("base consensus input must contain accepted snapshots only")

    base_pass_all = set(_unique_ids(base_pass_ids, owner="base pass IDs"))
    missing_base_pass = sorted(base_pass_all.difference(base_all))
    if missing_base_pass:
        raise ValueError(f"base pass IDs absent from base consensus: {missing_base_pass[:3]}")
    base_by_id = {key: value for key, value in base_all.items() if key in requested}
    base_pass = base_pass_all.intersection(requested)

    if repair_layers is not None and (
        repair_records or repair_pass_ids or repair_selected_ids is not None
    ):
        raise ValueError("named repair_layers cannot be mixed with legacy single-layer inputs")
    effective_layers: list[RepairSnapshotLayer]
    if repair_layers is None:
        inferred_selected = (
            [record.sample_id for record in repair_records]
            if repair_selected_ids is None
            else list(repair_selected_ids)
        )
        effective_layers = [
            RepairSnapshotLayer(
                name="repair",
                records=repair_records,
                selected_ids=inferred_selected,
                pass_ids=repair_pass_ids,
            )
        ]
    else:
        effective_layers = list(repair_layers)
    names = [layer.name.strip() for layer in effective_layers]
    if any(not name for name in names) or "base" in names or len(names) != len(set(names)):
        raise ValueError("repair layer names must be unique and nonempty")

    normalized_layers: list[
        tuple[
            str,
            dict[str, TeacherProjectionValidatedRecord],
            set[str],
            set[str],
        ]
    ] = []
    repair_layer_counts: dict[str, dict[str, int]] = {}
    for layer, name in zip(effective_layers, names, strict=True):
        untyped = _index_by_sample(layer.records, owner=f"repair layer {name} validated")
        records_by_id = {
            key: value
            for key, value in untyped.items()
            if isinstance(value, TeacherProjectionValidatedRecord)
        }
        selected = set(_unique_ids(layer.selected_ids, owner=f"repair layer {name} selected IDs"))
        passed = set(_unique_ids(layer.pass_ids, owner=f"repair layer {name} pass IDs"))
        stray_records = sorted(set(records_by_id).difference(selected))
        if stray_records:
            raise ValueError(
                f"repair layer {name} validated records outside its selected IDs: "
                f"{stray_records[:3]}"
            )
        pass_not_validated = sorted(passed.difference(records_by_id))
        if pass_not_validated:
            label = (
                "repair pass IDs absent from repair validated"
                if name == "repair" and len(effective_layers) == 1
                else f"repair layer {name} pass IDs absent from validated records"
            )
            raise ValueError(f"{label}: {pass_not_validated[:3]}")
        pass_not_selected = sorted(passed.difference(selected))
        if pass_not_selected:
            raise ValueError(
                f"repair layer {name} pass IDs absent from selected IDs: {pass_not_selected[:3]}"
            )
        records_by_id = {
            key: value for key, value in records_by_id.items() if key in requested
        }
        selected.intersection_update(requested)
        passed.intersection_update(requested)
        normalized_layers.append((name, records_by_id, selected, passed))
        repair_layer_counts[name] = {
            "selected": len(selected),
            "validated": len(records_by_id),
            "pass": len(passed),
        }

    trajectory_splits = deterministic_trajectory_splits(
        [trajectory_id for trajectory_id, _ in lineage.values()],
        split_seed=split_seed,
    )
    split_by_trajectory = {item.trajectory_id: item.split for item in trajectory_splits}

    final: list[FrozenTeacherSnapshotRecord] = []
    sft: list[FullSnapshotSFTRecord] = []
    missing_ids: list[str] = []
    provenance: list[SnapshotFreezeProvenanceRecord] = []
    duplicates: list[SnapshotDuplicateOverrideRecord] = []
    for order_index, sample_id in enumerate(ordered_ids):
        public = public_by_id[sample_id]
        trajectory_id, source_turn = lineage[sample_id]
        split = split_by_trajectory[trajectory_id]
        base_record = base_by_id.get(sample_id)
        base_ok = sample_id in base_pass

        selected_snapshot: TeacherCurrentStateSnapshot | None = None
        selected_source: SnapshotSource | None = None
        selected_repair_layer: str | None = None
        selected_repair_record: TeacherProjectionValidatedRecord | None = None
        source_record_hash: str | None = None
        reason: SelectionReason
        if base_ok:
            assert base_record is not None and base_record.consensus_snapshot is not None
            selected_snapshot = base_record.consensus_snapshot
            selected_source = "base"
            source_record_hash = content_sha256(base_record.model_dump(mode="json"))
            reason = "base_pass_selected"
        else:
            reason = "no_passed_snapshot"

        trace: list[RepairLayerDecision] = []
        candidate_hashes: dict[str, str] = {}
        if base_record is not None:
            assert base_record.consensus_snapshot is not None
            candidate_hashes["base"] = content_sha256(
                base_record.consensus_snapshot.model_dump(mode="json")
            )
        any_repair_selected = False
        for layer_name, layer_records, layer_selected, layer_pass in normalized_layers:
            layer_record = layer_records.get(sample_id)
            is_selected = sample_id in layer_selected
            is_passed = sample_id in layer_pass
            if layer_record is not None:
                candidate_hashes[layer_name] = content_sha256(
                    layer_record.snapshot.model_dump(mode="json")
                )
            trace.append(
                RepairLayerDecision(
                    layer_name=layer_name,
                    selected=is_selected,
                    validated_record_present=layer_record is not None,
                    passed=is_passed,
                    snapshot_sha256=(
                        None
                        if layer_record is None
                        else content_sha256(layer_record.snapshot.model_dump(mode="json"))
                    ),
                )
            )
            if not is_selected:
                continue
            any_repair_selected = True
            # Selection is itself an override.  A failed or invalid later layer
            # clears every earlier candidate rather than silently falling back.
            selected_snapshot = None
            selected_source = None
            selected_repair_layer = None
            selected_repair_record = None
            source_record_hash = None
            reason = "repair_selected_not_passed"
            if is_passed:
                assert layer_record is not None
                selected_snapshot = layer_record.snapshot
                selected_source = "repair"
                selected_repair_layer = layer_name
                selected_repair_record = layer_record
                source_record_hash = content_sha256(layer_record.model_dump(mode="json"))
                reason = "repair_pass_selected"

        if len(candidate_hashes) >= 2:
            selected_source_label = (
                selected_repair_layer if selected_source == "repair" else selected_source
            )
            duplicates.append(
                SnapshotDuplicateOverrideRecord(
                    sample_id=sample_id,
                    candidate_sources=list(candidate_hashes),
                    candidate_snapshot_sha256=candidate_hashes,
                    selected_source=selected_source_label,
                    resolution=(
                        "later_layer_overrides"
                        if selected_snapshot is not None
                        else "later_layer_selected_but_not_passed"
                    ),
                )
            )

        snapshot_hash = (
            None
            if selected_snapshot is None
            else content_sha256(selected_snapshot.model_dump(mode="json"))
        )
        provenance.append(
            SnapshotFreezeProvenanceRecord(
                sample_id=sample_id,
                order_index=order_index,
                public_prompt_sha256=public.public_prompt_sha256,
                trajectory_id=trajectory_id,
                source_turn_index=source_turn,
                split=split,
                base_record_present=base_record is not None,
                base_pass=base_ok,
                repair_selected=any_repair_selected,
                repair_pass=selected_source == "repair",
                repair_layer_trace=trace,
                final_source=selected_source,
                final_repair_layer_name=selected_repair_layer,
                selection_reason=reason,
                final_snapshot_sha256=snapshot_hash,
            )
        )
        if selected_snapshot is None:
            missing_ids.append(sample_id)
            continue

        if selected_source == "base":
            assert base_record is not None
            if base_record.public_prompt_sha256 != public.public_prompt_sha256:
                raise ValueError(f"base/public prompt hash mismatch for {sample_id}")
            if base_record.trajectory_id != trajectory_id or base_record.source_turn_index != source_turn:
                raise ValueError(f"base/lineage mismatch for {sample_id}")
        else:
            assert selected_repair_record is not None
            if selected_repair_record.public_prompt_sha256 != public.public_prompt_sha256:
                raise ValueError(f"repair/public prompt hash mismatch for {sample_id}")
        validate_public_evidence(selected_snapshot.evidence, prompt=public.public_prompt)
        assert source_record_hash is not None and selected_source is not None and snapshot_hash is not None
        final.append(
            FrozenTeacherSnapshotRecord(
                sample_id=sample_id,
                order_index=order_index,
                public_prompt_sha256=public.public_prompt_sha256,
                trajectory_id=trajectory_id,
                source_turn_index=source_turn,
                split=split,
                source=selected_source,
                repair_layer_name=selected_repair_layer,
                source_record_sha256=source_record_hash,
                snapshot_sha256=snapshot_hash,
                snapshot=selected_snapshot,
            )
        )
        sft.append(
            FullSnapshotSFTRecord(
                sample_id=sample_id,
                order_index=order_index,
                public_prompt_sha256=public.public_prompt_sha256,
                trajectory_id=trajectory_id,
                source_turn_index=source_turn,
                split=split,
                input=public.public_prompt,
                target=selected_snapshot,
                target_sha256=snapshot_hash,
            )
        )

    source_counts = Counter(item.source for item in final)
    trajectory_split_counts = Counter(item.split for item in trajectory_splits)
    sample_split_counts = Counter(item.split for item in final)
    summary = TeacherSnapshotFreezeSummary(
        split_seed=split_seed,
        requested_count=len(ordered_ids),
        base_record_count=len(base_by_id),
        base_pass_count=len(base_pass),
        repair_selected_count=len(
            set().union(*(selected for _, _, selected, _ in normalized_layers))
        ),
        repair_pass_count=len(set().union(*(passed for _, _, _, passed in normalized_layers))),
        repair_layer_counts=repair_layer_counts,
        final_base_count=source_counts["base"],
        final_repair_count=source_counts["repair"],
        final_count=len(final),
        missing_count=len(missing_ids),
        duplicate_override_count=len(duplicates),
        trajectory_count=len(trajectory_splits),
        trajectory_split_counts=dict(sorted(trajectory_split_counts.items())),
        final_sample_split_counts=dict(sorted(sample_split_counts.items())),
    )
    return TeacherSnapshotFreezeResult(
        final_snapshots=tuple(final),
        sft_records=tuple(sft),
        missing_ids=missing_ids,
        duplicate_overrides=tuple(duplicates),
        provenance=tuple(provenance),
        trajectory_splits=trajectory_splits,
        summary=summary,
    )


__all__ = [
    "DEFAULT_TEACHER_SNAPSHOT_SPLIT_SEED",
    "FrozenTeacherSnapshotRecord",
    "FullSnapshotSFTRecord",
    "RepairLayerDecision",
    "RepairSnapshotLayer",
    "SnapshotDuplicateOverrideRecord",
    "SnapshotFreezeProvenanceRecord",
    "TeacherSnapshotFreezeResult",
    "TeacherSnapshotFreezeSummary",
    "TeacherTrajectorySplitRecord",
    "deterministic_trajectory_splits",
    "freeze_teacher_snapshot_sft",
]
