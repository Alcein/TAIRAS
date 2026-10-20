"""Assemble evidence-first Open-SFT data from a frozen Scale trajectory panel.

This is intentionally a narrow adapter over the existing cold-reconstruction,
adjacent-patch, and evidence-first contracts.  It keeps every real dialogue
view once per epoch, then deterministically samples/repeats an independently
audited controlled rehearsal source to a declared train-time fraction.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.state_training.dataset_assembly import SFTTokenizerRecord
from taira_mt.state_training.evidence_first_sft import evidence_first_target
from taira_mt.state_training.training_views import DSUSPublicPrompt

SCALE_OPEN_SFT_VERSION = "dsus_scale_evidence_first_open_sft_v1"
SPLITS = ("train", "dev", "test")
REAL_VIEWS = ("cold", "adjacent")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise ValueError(f"JSONL input is empty: {path}")
    return output


def _load_tokenizer_rows(
    directory: Path,
    *,
    splits: Sequence[str],
    allow_empty: bool = False,
) -> dict[str, list[SFTTokenizerRecord]]:
    output: dict[str, list[SFTTokenizerRecord]] = {}
    seen: set[str] = set()
    for split in splits:
        path = directory / f"tokenizer_sft.{split}.private.jsonl"
        rows: list[SFTTokenizerRecord] = []
        for payload in _read_jsonl(path, allow_empty=allow_empty):
            row = SFTTokenizerRecord.model_validate(payload)
            prompt = DSUSPublicPrompt.model_validate_json(row.model_input)
            if prompt.content_hash() != row.public_prompt_sha256:
                raise ValueError(f"public prompt hash mismatch for {row.sample_id}")
            if row.sample_id in seen:
                raise ValueError(f"sample repeats across source splits: {row.sample_id}")
            seen.add(row.sample_id)
            rows.append(row)
        output[split] = rows
    return output


def _lineage_index(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in output:
            raise ValueError(f"missing or duplicate lineage sample_id in {path}")
        output[sample_id] = row
    return output


def _derived_row(
    source: SFTTokenizerRecord,
    *,
    source_view: str,
    exposure_index: int,
) -> SFTTokenizerRecord:
    target, _ = evidence_first_target(source.assistant_target)
    sample_id = content_sha256(
        {
            "schema_version": SCALE_OPEN_SFT_VERSION,
            "source_view": source_view,
            "source_sample_id": source.sample_id,
            "public_prompt_sha256": source.public_prompt_sha256,
            "exposure_index": exposure_index,
        }
    )
    return source.model_copy(update={"sample_id": sample_id, "assistant_target": target})


def _controlled_schedule(
    rows: Sequence[SFTTokenizerRecord],
    *,
    lineage: Mapping[str, Mapping[str, Any]],
    target_count: int,
    seed: str,
) -> list[tuple[SFTTokenizerRecord, int, str]]:
    """Round-robin mechanism/role cells, then repeat only when necessary."""

    if target_count < 1:
        return []
    grouped: dict[tuple[str, str], list[SFTTokenizerRecord]] = defaultdict(list)
    for row in rows:
        metadata = lineage.get(row.sample_id)
        if metadata is None:
            raise ValueError(f"controlled rehearsal lacks lineage for {row.sample_id}")
        mechanism = str(metadata.get("mechanism", "unknown"))
        role = str(metadata.get("role", "unknown"))
        grouped[(mechanism, role)].append(row)
    for key, values in grouped.items():
        values.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{key[0]}:{key[1]}:{row.sample_id}".encode()
            ).hexdigest()
        )
    if not grouped:
        raise ValueError("controlled rehearsal train split is empty")

    keys = sorted(grouped)
    cursors = {key: 0 for key in keys}
    exposures: Counter[str] = Counter()
    output: list[tuple[SFTTokenizerRecord, int, str]] = []
    while len(output) < target_count:
        for key in keys:
            if len(output) >= target_count:
                break
            values = grouped[key]
            source = values[cursors[key] % len(values)]
            cursors[key] += 1
            exposure_index = exposures[source.sample_id]
            exposures[source.sample_id] += 1
            output.append((source, exposure_index, f"{key[0]}/{key[1]}"))
    return output


def _real_trajectory_index(cold_dir: Path, adjacent_dir: Path) -> dict[str, tuple[str, str]]:
    """Map original real-view sample IDs to (trajectory, inherited split)."""

    output: dict[str, tuple[str, str]] = {}
    cold_path = cold_dir / "cold_reconstruction.transitions.private.jsonl"
    for row in _read_jsonl(cold_path):
        sample_id = str(row["sample_id"])
        output[sample_id] = (str(row["trajectory_id"]), str(row["split"]))
    adjacent_path = adjacent_dir / "sample_lineage.private.jsonl"
    for row in _read_jsonl(adjacent_path):
        sample_id = str(row["sample_id"])
        identity = (str(row["trajectory_id"]), str(row["split"]))
        previous = output.get(sample_id)
        if previous is not None and previous != identity:
            raise ValueError(f"cold/adjacent lineage disagreement for {sample_id}")
        output[sample_id] = identity
    return output


def build_scale_evidence_first_open_sft(
    *,
    cold_dir: Path,
    adjacent_dir: Path,
    controlled_rehearsal_dir: Path,
    output_dir: Path,
    controlled_rehearsal_ratio: float = 0.20,
    selection_seed: str = "dsus-scale-open-sft-controlled-v1",
) -> Path:
    """Build the final train view and real-only checkpoint-selection splits."""

    cold_dir = cold_dir.expanduser().resolve()
    adjacent_dir = adjacent_dir.expanduser().resolve()
    controlled_rehearsal_dir = controlled_rehearsal_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not 0.0 < controlled_rehearsal_ratio < 0.5:
        raise ValueError("controlled_rehearsal_ratio must lie in (0, 0.5)")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen Open-SFT artifact: {output_dir}")

    print("Open-SFT 步骤 1/5：读取 cold、adjacent 与 controlled rehearsal；不会调用 API/GPU。", flush=True)
    real_sources = {
        "cold": _load_tokenizer_rows(cold_dir, splits=SPLITS, allow_empty=True),
        "adjacent": _load_tokenizer_rows(adjacent_dir, splits=SPLITS, allow_empty=True),
    }
    controlled = _load_tokenizer_rows(
        controlled_rehearsal_dir,
        splits=("train", "dev"),
        allow_empty=False,
    )
    controlled_lineage = _lineage_index(
        controlled_rehearsal_dir / "sample_lineage.private.jsonl"
    )
    real_counts = {
        split: sum(len(real_sources[view][split]) for view in REAL_VIEWS) for split in SPLITS
    }
    if any(real_counts[split] == 0 for split in SPLITS):
        raise ValueError(f"every real split must contain at least one cold/adjacent row: {real_counts}")
    print(
        f"Open-SFT 步骤 1/5 完成：real={real_counts}，controlled train/dev="
        f"{len(controlled['train'])}/{len(controlled['dev'])}。",
        flush=True,
    )

    print("Open-SFT 步骤 2/5：保持每个 real view 每 epoch 暴露一次，并按目标比例平衡 rehearsal。", flush=True)
    target_controlled = round(
        real_counts["train"] * controlled_rehearsal_ratio / (1.0 - controlled_rehearsal_ratio)
    )
    scheduled = _controlled_schedule(
        controlled["train"],
        lineage=controlled_lineage,
        target_count=target_controlled,
        seed=selection_seed,
    )
    derived: dict[str, list[SFTTokenizerRecord]] = {split: [] for split in SPLITS}
    lineage_rows: list[dict[str, Any]] = []
    trajectory_index = _real_trajectory_index(cold_dir, adjacent_dir)
    trajectory_splits: dict[str, set[str]] = defaultdict(set)
    for split in SPLITS:
        for view in REAL_VIEWS:
            for source in real_sources[view][split]:
                row = _derived_row(source, source_view=view, exposure_index=0)
                derived[split].append(row)
                trajectory_id, inherited_split = trajectory_index[source.sample_id]
                if inherited_split != split:
                    raise ValueError(f"real lineage split mismatch for {source.sample_id}")
                trajectory_splits[trajectory_id].add(split)
                lineage_rows.append(
                    {
                        "schema_version": SCALE_OPEN_SFT_VERSION,
                        "sample_id": row.sample_id,
                        "split": split,
                        "source_view": view,
                        "source_sample_id": source.sample_id,
                        "source_trajectory_id": trajectory_id,
                        "exposure_index": 0,
                    }
                )
    for source, exposure_index, cell in scheduled:
        row = _derived_row(
            source,
            source_view="controlled_rehearsal",
            exposure_index=exposure_index,
        )
        derived["train"].append(row)
        lineage_rows.append(
            {
                "schema_version": SCALE_OPEN_SFT_VERSION,
                "sample_id": row.sample_id,
                "split": "train",
                "source_view": "controlled_rehearsal",
                "source_sample_id": source.sample_id,
                "source_trajectory_id": None,
                "exposure_index": exposure_index,
                "controlled_cell": cell,
            }
        )
    leaked = {key: splits for key, splits in trajectory_splits.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"real trajectory siblings leak across splits: {list(leaked)[:3]}")
    actual_ratio = len(scheduled) / len(derived["train"])
    tolerance = 1.0 / len(derived["train"])
    if abs(actual_ratio - controlled_rehearsal_ratio) > tolerance:
        raise AssertionError("controlled rehearsal ratio materialization drifted")
    print(
        f"Open-SFT 步骤 2/5 完成：train={len(derived['train'])}，"
        f"controlled={len(scheduled)}（{actual_ratio:.2%}），trajectory leakage=0。",
        flush=True,
    )

    print("Open-SFT 步骤 3/5：确定性打散 train，dev/test 保持 real-only。", flush=True)
    derived["train"].sort(
        key=lambda row: hashlib.sha256(f"{selection_seed}:{row.sample_id}".encode()).hexdigest()
    )
    seen = [row.sample_id for split in SPLITS for row in derived[split]]
    if len(seen) != len(set(seen)):
        raise ValueError("derived Open-SFT sample IDs collide")
    print("Open-SFT 步骤 3/5 完成：所有派生 ID 唯一，split membership 未重切。", flush=True)

    print("Open-SFT 步骤 4/5：写出 evidence-first train、real-only dev/test 与 retention monitor。", flush=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_paths: dict[str, Path] = {}
    for split in SPLITS:
        path = output_dir / f"tokenizer_sft.{split}.private.jsonl"
        path.write_text(
            "".join(canonical_json(row.model_dump(mode="json")) + "\n" for row in derived[split]),
            encoding="utf-8",
        )
        output_paths[split] = path
    controlled_dev = [
        _derived_row(row, source_view="controlled_rehearsal_dev", exposure_index=0)
        for row in controlled["dev"]
    ]
    monitor_path = output_dir / "tokenizer_sft.dev.controlled_retention.private.jsonl"
    monitor_path.write_text(
        "".join(canonical_json(row.model_dump(mode="json")) + "\n" for row in controlled_dev),
        encoding="utf-8",
    )
    lineage_path = output_dir / "sample_lineage.private.jsonl"
    lineage_path.write_text(
        "".join(canonical_json(row) + "\n" for row in lineage_rows),
        encoding="utf-8",
    )
    print("Open-SFT 步骤 4/5 完成：训练与双 checkpoint-monitor 数据已写出。", flush=True)

    print("Open-SFT 步骤 5/5：冻结比例、暴露次数、防泄漏和输入输出哈希。", flush=True)
    exposure_counts = Counter(source.sample_id for source, _, _ in scheduled)
    controlled_cells = Counter(cell for _, _, cell in scheduled)
    summary = {
        "schema_version": SCALE_OPEN_SFT_VERSION,
        "status": "frozen_untrained",
        "train_source_counts": {
            "cold": len(real_sources["cold"]["train"]),
            "adjacent": len(real_sources["adjacent"]["train"]),
            "controlled_rehearsal": len(scheduled),
        },
        "output_split_counts": {split: len(derived[split]) for split in SPLITS},
        "controlled_rehearsal": {
            "target_ratio": controlled_rehearsal_ratio,
            "actual_ratio": actual_ratio,
            "selection_seed": selection_seed,
            "source_unique_count": len(exposure_counts),
            "maximum_source_exposure": max(exposure_counts.values(), default=0),
            "cell_counts": dict(sorted(controlled_cells.items())),
        },
        "contracts": {
            "target_serialization": "schema_version/base_revision/evidence/delta",
            "real_view_exposure_per_epoch": 1,
            "controlled_rehearsal_ratio_is_train_only": True,
            "real_dev_test_are_checkpoint_selection_only": True,
            "controlled_dev_is_separate_retention_monitor": True,
            "trajectory_split_leakage_count": 0,
            "external_model_calls": False,
            "gpu_used": False,
            "training_launched": False,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    manifest = {
        **summary,
        "inputs": {
            "cold_dir": str(cold_dir),
            "adjacent_dir": str(adjacent_dir),
            "controlled_rehearsal_dir": str(controlled_rehearsal_dir),
        },
        "input_hashes": {
            "cold_manifest": _sha256(cold_dir / "manifest.json"),
            "adjacent_manifest": _sha256(adjacent_dir / "manifest.json"),
            "controlled_manifest": _sha256(controlled_rehearsal_dir / "manifest.json"),
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in (*output_paths.values(), monitor_path, lineage_path, summary_path)
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(f"Open-SFT 步骤 5/5 完成：最终数据已冻结到 {output_dir}。", flush=True)
    return output_dir


__all__ = ["SCALE_OPEN_SFT_VERSION", "build_scale_evidence_first_open_sft"]
