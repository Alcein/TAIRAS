"""Freeze pass-only base/repair full snapshots into an offline SFT artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json
from taira_mt.state_training.dataset_assembly import PublicRLPromptRecord
from taira_mt.state_training.teacher_projection import TeacherProjectionValidatedRecord
from taira_mt.state_training.teacher_projection_consensus import TeacherConsensusRecord
from taira_mt.state_training.teacher_snapshot_freeze import (
    DEFAULT_TEACHER_SNAPSHOT_SPLIT_SEED,
    RepairSnapshotLayer,
    freeze_teacher_snapshot_sft,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline final freeze of pass-only base and repair full-state snapshots; "
            "repair validated IDs form the override domain and no missing label is synthesized."
        )
    )
    parser.add_argument(
        "--public-prompts",
        action="append",
        type=Path,
        required=True,
        help="Repeat to merge disjoint main/reserve public-prompt JSONL files.",
    )
    parser.add_argument(
        "--lineage",
        action="append",
        type=Path,
        required=True,
        help="Repeat to merge disjoint main/reserve lineage JSONL files.",
    )
    parser.add_argument("--base-consensus-accepted", type=Path, required=True)
    parser.add_argument("--base-pass-ids", type=Path, required=True)
    parser.add_argument("--repair-validated", type=Path)
    parser.add_argument("--repair-pass-ids", type=Path)
    parser.add_argument(
        "--repair-selected-ids",
        type=Path,
        help="Legacy single-layer selection domain; omitted only for backward compatibility.",
    )
    parser.add_argument(
        "--repair-layer",
        action="append",
        default=[],
        metavar="NAME=VALIDATED_JSONL,PASS_IDS",
        help="Repeat in override order; later layers have higher precedence.",
    )
    parser.add_argument(
        "--repair-layer-selected-ids",
        action="append",
        default=[],
        metavar="NAME=SELECTED_IDS",
        help="Required exactly once for every named repair layer.",
    )
    parser.add_argument("--all-sample-ids", type=Path, required=True)
    parser.add_argument("--split-seed", default=DEFAULT_TEACHER_SNAPSHOT_SPLIT_SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _read_jsonl(path: Path, model: type[StrictModel]) -> list[StrictModel]:
    output: list[StrictModel] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            output.append(model.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid JSONL record at {path}:{line_number}") from exc
    return output


def _read_lineage(path: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise ValueError(f"invalid lineage JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"lineage row is not an object at {path}:{line_number}")
        output.append(row)
    return output


def _read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _jsonl(rows: Sequence[Any]) -> str:
    return "".join(
        canonical_json(row.model_dump(mode="json") if hasattr(row, "model_dump") else row) + "\n"
        for row in rows
    )


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen snapshot artifact: {path}")
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _layer_name(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
        raise ValueError(f"invalid repair layer name: {value!r}")
    return value


def _parse_repair_layers(
    layer_values: Sequence[str],
    selection_values: Sequence[str],
) -> list[tuple[str, Path, Path, Path]]:
    layers: list[tuple[str, Path, Path]] = []
    seen: set[str] = set()
    for value in layer_values:
        name_raw, separator, paths_raw = value.partition("=")
        components = paths_raw.split(",") if separator else []
        if len(components) != 2 or any(not component.strip() for component in components):
            raise ValueError("--repair-layer must use NAME=VALIDATED_JSONL,PASS_IDS")
        name = _layer_name(name_raw)
        if name in seen:
            raise ValueError(f"duplicate repair layer name: {name}")
        seen.add(name)
        layers.append((name, Path(components[0].strip()), Path(components[1].strip())))

    selections: dict[str, Path] = {}
    for value in selection_values:
        name_raw, separator, path_raw = value.partition("=")
        if not separator or not path_raw.strip():
            raise ValueError("--repair-layer-selected-ids must use NAME=SELECTED_IDS")
        name = _layer_name(name_raw)
        if name in selections:
            raise ValueError(f"duplicate repair layer selected-ID mapping: {name}")
        selections[name] = Path(path_raw.strip())
    missing = [name for name, _, _ in layers if name not in selections]
    extra = sorted(set(selections).difference(name for name, _, _ in layers))
    if missing or extra:
        raise ValueError(
            f"named repair selection domains mismatch; missing={missing}, extra={extra}"
        )
    return [(name, validated, passed, selections[name]) for name, validated, passed in layers]


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    named_layer_paths = _parse_repair_layers(
        args.repair_layer,
        args.repair_layer_selected_ids,
    )
    legacy_values = (args.repair_validated, args.repair_pass_ids, args.repair_selected_ids)
    if named_layer_paths and any(value is not None for value in legacy_values):
        raise ValueError("named --repair-layer cannot be mixed with legacy repair arguments")
    if not named_layer_paths and (
        args.repair_validated is None or args.repair_pass_ids is None
    ):
        raise ValueError(
            "provide either named --repair-layer entries or both legacy "
            "--repair-validated/--repair-pass-ids"
        )
    public_prompt_paths = [path.expanduser().resolve() for path in args.public_prompts]
    lineage_paths = [path.expanduser().resolve() for path in args.lineage]
    inputs = {
        "base_consensus_accepted": args.base_consensus_accepted.expanduser().resolve(),
        "base_pass_ids": args.base_pass_ids.expanduser().resolve(),
        "all_sample_ids": args.all_sample_ids.expanduser().resolve(),
    }
    inputs.update(
        {f"public_prompts_{index:03d}": path for index, path in enumerate(public_prompt_paths)}
    )
    inputs.update({f"lineage_{index:03d}": path for index, path in enumerate(lineage_paths)})
    if named_layer_paths:
        for name, validated, passed, selected in named_layer_paths:
            inputs[f"repair_{name}_validated"] = validated.expanduser().resolve()
            inputs[f"repair_{name}_pass_ids"] = passed.expanduser().resolve()
            inputs[f"repair_{name}_selected_ids"] = selected.expanduser().resolve()
    else:
        assert args.repair_validated is not None and args.repair_pass_ids is not None
        inputs["repair_validated"] = args.repair_validated.expanduser().resolve()
        inputs["repair_pass_ids"] = args.repair_pass_ids.expanduser().resolve()
        if args.repair_selected_ids is not None:
            inputs["repair_selected_ids"] = args.repair_selected_ids.expanduser().resolve()
    missing_inputs = [str(path) for path in inputs.values() if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"freeze inputs missing: {missing_inputs}")
    input_hashes = {str(path): _sha256(path) for path in inputs.values()}
    output_dir = args.output_dir.expanduser().resolve()
    outputs = {
        "final_snapshots": output_dir / "final_snapshots.jsonl",
        "sft": output_dir / "sft.full_snapshot.jsonl",
        "missing": output_dir / "missing.ids",
        "duplicates": output_dir / "duplicate_overrides.jsonl",
        "provenance": output_dir / "provenance.jsonl",
        "trajectory_splits": output_dir / "trajectory_splits.jsonl",
        "summary": output_dir / "summary.json",
        "hashes": output_dir / "hashes.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite an existing final snapshot freeze")

    print("步骤 1/4：只读加载 full sample 顺序、公开 prompt 与 trajectory lineage。")
    all_ids = _read_ids(inputs["all_sample_ids"])
    public = [
        row
        for path in public_prompt_paths
        for row in _read_jsonl(path, PublicRLPromptRecord)
    ]
    lineage = [row for path in lineage_paths for row in _read_lineage(path)]
    print(
        f"步骤 1/4 完成：requested={len(all_ids)}，public rows={len(public)}，"
        f"lineage rows={len(lineage)}，public/lineage files="
        f"{len(public_prompt_paths)}/{len(lineage_paths)}。"
    )

    print("步骤 2/4：加载 base/repair snapshot 与各自 acceptance pass；repair selected IDs 优先。")
    base = _read_jsonl(inputs["base_consensus_accepted"], TeacherConsensusRecord)
    base_pass = _read_ids(inputs["base_pass_ids"])
    repair_layers: list[RepairSnapshotLayer] | None = None
    legacy_repair: list[StrictModel] = []
    legacy_pass: list[str] = []
    legacy_selected: list[str] | None = None
    if named_layer_paths:
        repair_layers = []
        layer_progress: list[str] = []
        for name, _, _, _ in named_layer_paths:
            records = _read_jsonl(
                inputs[f"repair_{name}_validated"],
                TeacherProjectionValidatedRecord,
            )
            passed = _read_ids(inputs[f"repair_{name}_pass_ids"])
            selected = _read_ids(inputs[f"repair_{name}_selected_ids"])
            repair_layers.append(
                RepairSnapshotLayer(
                    name=name,
                    records=records,  # type: ignore[arg-type]
                    selected_ids=selected,
                    pass_ids=passed,
                )
            )
            layer_progress.append(
                f"{name}:selected/validated/pass={len(selected)}/{len(records)}/{len(passed)}"
            )
        print(
            f"步骤 2/4 完成：base records/pass={len(base)}/{len(base_pass)}；"
            + "；".join(layer_progress)
            + "。"
        )
    else:
        legacy_repair = _read_jsonl(inputs["repair_validated"], TeacherProjectionValidatedRecord)
        legacy_pass = _read_ids(inputs["repair_pass_ids"])
        legacy_selected = (
            None
            if "repair_selected_ids" not in inputs
            else _read_ids(inputs["repair_selected_ids"])
        )
        selected_count = len(legacy_repair) if legacy_selected is None else len(legacy_selected)
        print(
            f"步骤 2/4 完成：base records/pass={len(base)}/{len(base_pass)}，"
            f"repair selected/validated/pass={selected_count}/{len(legacy_repair)}/"
            f"{len(legacy_pass)}。"
        )

    print("步骤 3/4：按原 sample 顺序冻结 snapshot，并按 trajectory 确定性切 train/dev/test。")
    result = freeze_teacher_snapshot_sft(
        public_records=public,  # type: ignore[arg-type]
        lineage_records=lineage,
        base_records=base,  # type: ignore[arg-type]
        base_pass_ids=base_pass,
        repair_records=legacy_repair,  # type: ignore[arg-type]
        repair_pass_ids=legacy_pass,
        repair_selected_ids=legacy_selected,
        repair_layers=repair_layers,
        all_sample_ids=all_ids,
        split_seed=args.split_seed,
    )
    print(
        f"步骤 3/4 完成：final={result.summary.final_count}，"
        f"missing={result.summary.missing_count}，fabricated=0，trajectory leakage=0。"
    )

    print("步骤 4/4：写出 final/SFT/missing/duplicate/provenance/split，并冻结输入输出 hash。")
    _write_new(outputs["final_snapshots"], _jsonl(result.final_snapshots))
    _write_new(outputs["sft"], _jsonl(result.sft_records))
    _write_new(outputs["missing"], "".join(sample_id + "\n" for sample_id in result.missing_ids))
    _write_new(outputs["duplicates"], _jsonl(result.duplicate_overrides))
    _write_new(outputs["provenance"], _jsonl(result.provenance))
    _write_new(outputs["trajectory_splits"], _jsonl(result.trajectory_splits))
    summary_payload = result.summary.model_dump(mode="json")
    summary_payload.update(
        {
            "input_sha256": input_hashes,
            "output_files": {key: str(path) for key, path in outputs.items()},
        }
    )
    _write_new(outputs["summary"], canonical_json(summary_payload) + "\n")
    output_hashes = {
        path.name: _sha256(path)
        for key, path in outputs.items()
        if key != "hashes"
    }
    _write_new(
        outputs["hashes"],
        canonical_json(
            {
                "schema_version": "dsus_teacher_snapshot_freeze_hashes_v1",
                "input_sha256": input_hashes,
                "output_sha256": dict(sorted(output_hashes.items())),
                "hashes_json_self_excluded": True,
            }
        )
        + "\n",
    )
    print(
        f"步骤 4/4 完成：final={len(result.final_snapshots)}，"
        f"missing={len(result.missing_ids)}，目录={output_dir}。"
    )
    return output_dir


if __name__ == "__main__":
    main()
