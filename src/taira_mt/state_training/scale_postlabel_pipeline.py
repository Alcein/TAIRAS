"""Thin orchestration for Scale trajectory labels after teacher generation.

The pipeline deliberately owns no teacher call and no trainer.  It validates a
frozen public/routing/manifest bundle, adapts one merged
``TeacherProjectionValidatedRecord`` stream to the existing Real500 freeze
contract, and delegates all semantic work to the existing snapshot freeze,
deterministic differ, cold reconstruction, adjacent SFT, and evidence-first
assemblers.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json
from taira_mt.state_training.dataset_assembly import PublicRLPromptRecord
from taira_mt.state_training.replacement_evaluation import ReplacementSampleManifest
from taira_mt.state_training.teacher_projection import TeacherProjectionValidatedRecord

SCALE_POSTLABEL_PIPELINE_VERSION = "dsus_scale_postlabel_pipeline_v1"


class ScalePostlabelPipelineConfig(StrictModel):
    schema_version: Literal[SCALE_POSTLABEL_PIPELINE_VERSION] = SCALE_POSTLABEL_PIPELINE_VERSION
    project_root: Path = Path(".")
    public_prompt_files: list[Path] = Field(min_length=1)
    routing_files: list[Path] = Field(min_length=1)
    opaque_manifest: Path
    panel_audit: Path
    merged_teacher_validated: Path
    controlled_rehearsal_dir: Path
    local_tokenizer: Path
    output_dir: Path
    split_seed: str = Field(default="dsus-scale-open-sft-v1", min_length=1)
    controlled_rehearsal_ratio: float = Field(default=0.20, gt=0.0, lt=0.5)
    controlled_selection_seed: str = Field(
        default="dsus-scale-open-sft-controlled-v1",
        min_length=1,
    )
    max_seq_length: Literal[4096] = 4096

    @model_validator(mode="after")
    def validate_parallel_sources(self) -> ScalePostlabelPipelineConfig:
        if len(self.public_prompt_files) != len(self.routing_files):
            raise ValueError("public_prompt_files and routing_files must have equal lengths")
        return self

    def resolve_from(self, config_path: Path) -> ScalePostlabelPipelineConfig:
        base = (config_path.parent / self.project_root).resolve()

        def absolute(path: Path) -> Path:
            return path if path.is_absolute() else (base / path).resolve()

        return self.model_copy(
            update={
                "project_root": base,
                "public_prompt_files": [absolute(path) for path in self.public_prompt_files],
                "routing_files": [absolute(path) for path in self.routing_files],
                "opaque_manifest": absolute(self.opaque_manifest),
                "panel_audit": absolute(self.panel_audit),
                "merged_teacher_validated": absolute(self.merged_teacher_validated),
                "controlled_rehearsal_dir": absolute(self.controlled_rehearsal_dir),
                "local_tokenizer": absolute(self.local_tokenizer),
                "output_dir": absolute(self.output_dir),
            }
        )


class ScalePostlabelPreflight(StrictModel):
    schema_version: Literal["dsus_scale_postlabel_preflight_v1"] = (
        "dsus_scale_postlabel_preflight_v1"
    )
    status: Literal["ready", "waiting_for_teacher_labels"]
    public_count: int = Field(ge=1)
    routing_count: int = Field(ge=1)
    manifest_count: int = Field(ge=1)
    trajectory_count: int = Field(ge=1)
    opening_trajectory_count: int = Field(ge=0)
    contiguous_trajectory_count: int = Field(ge=0)
    teacher_validated_exists: bool
    teacher_validated_count: int = Field(ge=0)
    missing_teacher_count: int = Field(ge=0)
    extra_teacher_count: int = Field(ge=0)
    public_routing_manifest_exact: bool
    public_hash_mismatch_count: int = Field(ge=0)
    panel_audit_ood300_touched: bool
    output_dir_exists: bool
    stages_waiting_for_labels: list[str]
    planned_outputs: dict[str, str]
    external_model_calls: Literal[False] = False
    gpu_used: Literal[False] = False


def load_scale_postlabel_config(path: Path) -> ScalePostlabelPipelineConfig:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ScalePostlabelPipelineConfig.model_validate(payload).resolve_from(path)


def _read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    output: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        output.append(payload)
    if not output and not allow_empty:
        raise ValueError(f"JSONL is empty: {path}")
    return output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _planned_outputs(root: Path) -> dict[str, Path]:
    return {
        "acceptance_adapter": root / "00_teacher_acceptance_adapter",
        "snapshot_freeze": root / "01_snapshot_freeze",
        "adjacent_differ": root / "02_adjacent_differ",
        "cold_reconstruction": root / "03_cold_reconstruction",
        "adjacent_sft": root / "04_adjacent_sft",
        "open_sft": root / "05_evidence_first_open_sft",
    }


def preflight_scale_postlabel_pipeline(
    config: ScalePostlabelPipelineConfig,
) -> ScalePostlabelPreflight:
    """Validate frozen identity/trajectory contracts; labels may still be absent."""

    for path in (
        *config.public_prompt_files,
        *config.routing_files,
        config.opaque_manifest,
        config.panel_audit,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for directory in (config.controlled_rehearsal_dir, config.local_tokenizer):
        if not directory.exists():
            raise FileNotFoundError(directory)

    public_by_id: dict[str, PublicRLPromptRecord] = {}
    for path in config.public_prompt_files:
        for payload in _read_jsonl(path):
            record = PublicRLPromptRecord.model_validate(payload)
            if record.sample_id in public_by_id:
                raise ValueError(f"public prompts repeat sample_id={record.sample_id}")
            if record.public_prompt.content_hash() != record.public_prompt_sha256:
                raise ValueError(f"public prompt content hash mismatch: {record.sample_id}")
            public_by_id[record.sample_id] = record

    routing_by_id: dict[str, dict[str, Any]] = {}
    trajectory_turns: dict[str, list[int]] = defaultdict(list)
    hash_mismatches = 0
    for path in config.routing_files:
        for payload in _read_jsonl(path):
            sample_id = payload.get("sample_id")
            trajectory_id = payload.get("trajectory_id", payload.get("source_conversation_id"))
            source_turn = payload.get("source_turn_index", payload.get("trajectory_turn_index"))
            public_hash = payload.get("public_prompt_sha256")
            if not isinstance(sample_id, str) or sample_id in routing_by_id:
                raise ValueError(f"routing lacks unique sample_id in {path}")
            if not isinstance(trajectory_id, str) or not trajectory_id:
                raise ValueError(f"routing lacks trajectory_id for {sample_id}")
            if not isinstance(source_turn, int) or source_turn < 0:
                raise ValueError(f"routing lacks nonnegative source_turn_index for {sample_id}")
            public = public_by_id.get(sample_id)
            if public is None or public_hash != public.public_prompt_sha256:
                hash_mismatches += 1
            routing_by_id[sample_id] = payload
            trajectory_turns[trajectory_id].append(source_turn)

    manifest_payload = json.loads(config.opaque_manifest.read_text(encoding="utf-8"))
    manifest = ReplacementSampleManifest.model_validate(manifest_payload)
    manifest_ids = [sample.sample_id for sample in manifest.samples]
    manifest_hashes = {sample.sample_id: sample.public_prompt_sha256 for sample in manifest.samples}
    universe = set(manifest_ids)
    exact = universe == set(public_by_id) == set(routing_by_id)
    if not exact:
        raise ValueError(
            "public/routing/manifest sample universes differ: "
            f"public={len(public_by_id)}, routing={len(routing_by_id)}, manifest={len(universe)}"
        )
    for sample_id, public in public_by_id.items():
        if manifest_hashes[sample_id] != public.public_prompt_sha256:
            hash_mismatches += 1
    if hash_mismatches:
        raise ValueError(f"public/routing/manifest hash mismatches={hash_mismatches}")

    contiguous = 0
    openings = 0
    for trajectory_id, values in trajectory_turns.items():
        ordered = sorted(values)
        if len(ordered) != len(set(ordered)):
            raise ValueError(f"trajectory repeats source turn: {trajectory_id}")
        if ordered and ordered[0] == 0:
            openings += 1
        if ordered == list(range(len(ordered))):
            contiguous += 1
        else:
            raise ValueError(f"trajectory panel is not complete/contiguous: {trajectory_id}")

    panel_audit = json.loads(config.panel_audit.read_text(encoding="utf-8"))
    ood_touched = bool(panel_audit.get("heldout_ood300_content_touched", True))
    if ood_touched:
        raise ValueError("panel audit does not prove heldout OOD300 content remained untouched")
    if panel_audit.get("selected", {}).get("prefix_count") != len(universe):
        raise ValueError("panel audit selected count differs from opaque manifest")

    teacher_exists = config.merged_teacher_validated.is_file()
    teacher_count = 0
    missing: set[str] = set(universe)
    extra: set[str] = set()
    if teacher_exists:
        validated_by_id: dict[str, TeacherProjectionValidatedRecord] = {}
        for payload in _read_jsonl(config.merged_teacher_validated):
            record = TeacherProjectionValidatedRecord.model_validate(payload)
            if record.sample_id in validated_by_id:
                raise ValueError(f"teacher validated repeats sample_id={record.sample_id}")
            public = public_by_id.get(record.sample_id)
            if public is not None and record.public_prompt_sha256 != public.public_prompt_sha256:
                raise ValueError(f"teacher/public prompt hash mismatch: {record.sample_id}")
            validated_by_id[record.sample_id] = record
        teacher_count = len(validated_by_id)
        missing = universe.difference(validated_by_id)
        extra = set(validated_by_id).difference(universe)
        if extra:
            raise ValueError(f"teacher output contains IDs outside frozen manifest: {sorted(extra)[:3]}")

    ready = teacher_exists and not missing and not extra
    stages = [] if ready else [
        "snapshot freeze",
        "cold reconstruction",
        "adjacent deterministic diff/reducer",
        "evidence-first Open-SFT assembly",
    ]
    planned = _planned_outputs(config.output_dir)
    return ScalePostlabelPreflight(
        status="ready" if ready else "waiting_for_teacher_labels",
        public_count=len(public_by_id),
        routing_count=len(routing_by_id),
        manifest_count=len(universe),
        trajectory_count=len(trajectory_turns),
        opening_trajectory_count=openings,
        contiguous_trajectory_count=contiguous,
        teacher_validated_exists=teacher_exists,
        teacher_validated_count=teacher_count,
        missing_teacher_count=len(missing),
        extra_teacher_count=len(extra),
        public_routing_manifest_exact=exact,
        public_hash_mismatch_count=hash_mismatches,
        panel_audit_ood300_touched=ood_touched,
        output_dir_exists=config.output_dir.exists(),
        stages_waiting_for_labels=stages,
        planned_outputs={name: str(path) for name, path in planned.items()},
    )


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite pipeline file: {path}")
    path.write_text(text, encoding="utf-8")


def _materialize_acceptance_adapter(
    config: ScalePostlabelPipelineConfig,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    manifest = ReplacementSampleManifest.model_validate_json(
        config.opaque_manifest.read_text(encoding="utf-8")
    )
    sample_ids = [item.sample_id for item in manifest.samples]
    paths = {
        "empty_base": output_dir / "base.consensus.accepted.empty.jsonl",
        "empty_base_pass": output_dir / "base.pass.empty.ids",
        "selected": output_dir / "scale_teacher.selected.ids",
        "passed": output_dir / "scale_teacher.pass.ids",
        "all_ids": output_dir / "all_sample_ids.txt",
        "summary": output_dir / "summary.json",
    }
    _write_new(paths["empty_base"], "")
    _write_new(paths["empty_base_pass"], "")
    identifiers = "".join(sample_id + "\n" for sample_id in sample_ids)
    _write_new(paths["selected"], identifiers)
    _write_new(paths["passed"], identifiers)
    _write_new(paths["all_ids"], identifiers)
    summary = {
        "schema_version": "dsus_scale_teacher_acceptance_adapter_v1",
        "role": "validated teacher stream adapted as one named repair layer",
        "sample_count": len(sample_ids),
        "teacher_validated": str(config.merged_teacher_validated),
        "teacher_validated_sha256": _sha256(config.merged_teacher_validated),
        "opaque_manifest_sha256": _sha256(config.opaque_manifest),
        "all_selected_and_passed": True,
        "empty_base_layer": True,
        "external_model_calls": False,
    }
    _write_new(paths["summary"], canonical_json(summary) + "\n")
    return paths


def run_scale_postlabel_pipeline(config: ScalePostlabelPipelineConfig) -> Path:
    """Run the offline post-label chain.  The teacher file must be complete."""

    preflight = preflight_scale_postlabel_pipeline(config)
    if preflight.status != "ready":
        raise RuntimeError(
            "teacher labels are not complete; run --dry-run for details: "
            f"missing={preflight.missing_teacher_count}"
        )
    if config.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite pipeline output: {config.output_dir}")
    stages = _planned_outputs(config.output_dir)

    print("Pipeline 1/6: materializing the validated-teacher acceptance adapter (API=0, GPU=0).", flush=True)
    adapter = _materialize_acceptance_adapter(
        config,
        output_dir=stages["acceptance_adapter"],
    )
    print(
        f"Pipeline 1/6 complete: all {preflight.manifest_count} frozen samples entered the same validated layer.",
        flush=True,
    )

    print("Pipeline 2/6: reusing the Real500 freezer to freeze train/dev/test by trajectory in one pass.", flush=True)
    from taira_mt.cli.freeze_dsus_teacher_snapshot_sft import main as freeze_main

    freeze_args: list[str] = []
    for path in config.public_prompt_files:
        freeze_args.extend(("--public-prompts", str(path)))
    for path in config.routing_files:
        freeze_args.extend(("--lineage", str(path)))
    freeze_args.extend(
        (
            "--base-consensus-accepted",
            str(adapter["empty_base"]),
            "--base-pass-ids",
            str(adapter["empty_base_pass"]),
            "--repair-layer",
            f"scale_teacher={config.merged_teacher_validated},{adapter['passed']}",
            "--repair-layer-selected-ids",
            f"scale_teacher={adapter['selected']}",
            "--all-sample-ids",
            str(adapter["all_ids"]),
            "--split-seed",
            config.split_seed,
            "--output-dir",
            str(stages["snapshot_freeze"]),
        )
    )
    freeze_main(freeze_args)
    print("Pipeline 2/6 complete: all trajectory siblings inherit a single split.", flush=True)

    print("Pipeline 3/6: reusing the deterministic snapshot differ and replaying through the production reducer.", flush=True)
    from taira_mt.cli.compile_dsus_teacher_snapshot_transitions import main as differ_main

    differ_main(
        [
            "--frozen-snapshots",
            str(stages["snapshot_freeze"] / "final_snapshots.jsonl"),
            "--output-dir",
            str(stages["adjacent_differ"]),
        ]
    )
    print("Pipeline 3/6 complete: unrepresentable patches are recorded only in the ledger; labels are never guessed.", flush=True)

    print("Pipeline 4/6: reusing cold reconstruction and running EMPTY-to-current-state reducer and token preflight checks.", flush=True)
    from taira_mt.cli.build_dsus_teacher_snapshot_cold_reconstruction import main as cold_main

    cold_main(
        [
            "--full-snapshot-sft",
            str(stages["snapshot_freeze"] / "sft.full_snapshot.jsonl"),
            "--output-dir",
            str(stages["cold_reconstruction"]),
            "--tokenizer",
            str(config.local_tokenizer),
            "--max-seq-length",
            str(config.max_seq_length),
            "--exclude-overlength-from-tokenizer-sft",
            "--adjacent-summary",
            str(stages["adjacent_differ"] / "summary.json"),
            "--adjacent-compiled",
            str(stages["adjacent_differ"] / "transitions.compiled.jsonl"),
        ]
    )
    print("Pipeline 4/6 complete: only reducer-verified cold patches are released.", flush=True)

    print("Pipeline 5/6: adapting replay-certified opening/adjacent patches into incremental SFT views.", flush=True)
    from taira_mt.cli.build_dsus_real_adjacent_sft import build_adjacent_sft

    build_adjacent_sft(
        full_snapshot_sft_path=stages["snapshot_freeze"] / "sft.full_snapshot.jsonl",
        compiled_transition_path=stages["adjacent_differ"] / "transitions.compiled.jsonl",
        output_dir=stages["adjacent_sft"],
        expected_split_counts=None,
        enforce_real500_scope=False,
    )
    print("Pipeline 5/6 complete: Scale scope does not inherit the hardcoded Real500 count of 142.", flush=True)

    print("Pipeline 6/6: building evidence-first Open-SFT and mixing controlled rehearsal at the declared ratio.", flush=True)
    from taira_mt.state_training.scale_open_sft import build_scale_evidence_first_open_sft

    build_scale_evidence_first_open_sft(
        cold_dir=stages["cold_reconstruction"],
        adjacent_dir=stages["adjacent_sft"],
        controlled_rehearsal_dir=config.controlled_rehearsal_dir,
        output_dir=stages["open_sft"],
        controlled_rehearsal_ratio=config.controlled_rehearsal_ratio,
        selection_seed=config.controlled_selection_seed,
    )
    print("Pipeline 6/6 complete: data is frozen; the SFT/RL trainer has not been started.", flush=True)

    summary = {
        "schema_version": SCALE_POSTLABEL_PIPELINE_VERSION,
        "status": "complete_untrained",
        "preflight": preflight.model_dump(mode="json"),
        "stage_outputs": {name: str(path) for name, path in stages.items()},
        "contracts": {
            "teacher_generation_is_upstream": True,
            "teacher_api_calls_in_pipeline": 0,
            "gpu_calls_in_pipeline": 0,
            "trajectory_split_once_at_snapshot_freeze": True,
            "sibling_split_leakage_count": 0,
            "heldout_ood300_content_touched": False,
            "trainer_launched": False,
            "overlength_gold_truncated": False,
            "overlength_full_transitions_retained": True,
            "overlength_tokenizer_sft_policy": (
                "exclude_from_tokenizer_sft_retain_transition"
            ),
        },
    }
    _write_new(config.output_dir / "pipeline_summary.json", canonical_json(summary) + "\n")
    print(f"Complete: Scale post-label data pipeline frozen at {config.output_dir}.", flush=True)
    return config.output_dir


__all__ = [
    "SCALE_POSTLABEL_PIPELINE_VERSION",
    "ScalePostlabelPipelineConfig",
    "ScalePostlabelPreflight",
    "load_scale_postlabel_config",
    "preflight_scale_postlabel_pipeline",
    "run_scale_postlabel_pipeline",
]
