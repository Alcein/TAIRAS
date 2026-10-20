"""Compile accepted full-state teacher snapshots into replay-verified patches offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from taira_mt.provenance import canonical_json
from taira_mt.state_training.teacher_projection_consensus import TeacherConsensusRecord
from taira_mt.state_training.teacher_snapshot_differ import (
    TEACHER_SNAPSHOT_DIFFER_VERSION,
    TeacherSnapshotSourceRecord,
    TeacherSnapshotTransitionRecord,
    compile_accepted_snapshot_trajectories,
)
from taira_mt.state_training.teacher_snapshot_freeze import FrozenTeacherSnapshotRecord


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline deterministic adjacent-snapshot differ. It never calls an API and "
            "releases only StatePatch programs verified by the production reducer."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--accepted-consensus",
        type=Path,
        help="Legacy accepted TeacherConsensusRecord JSONL input.",
    )
    source.add_argument(
        "--frozen-snapshots",
        type=Path,
        help="Final FrozenTeacherSnapshotRecord JSONL input.",
    )
    parser.add_argument(
        "--accepted-id-file",
        type=Path,
        help=(
            "Optional newline-delimited pass IDs. Every requested ID must occur in the "
            "accepted consensus; filtering is read-only."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _read_accepted(path: Path) -> list[TeacherConsensusRecord]:
    records: list[TeacherConsensusRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = TeacherConsensusRecord.model_validate(json.loads(line))
        except Exception as exc:
            raise ValueError(f"invalid consensus record at {path}:{line_number}") from exc
        if record.status != "accepted":
            raise ValueError(f"non-accepted record at {path}:{line_number}")
        records.append(record)
    if not records:
        raise ValueError("accepted consensus file is empty")
    return records


def _read_frozen(path: Path) -> list[FrozenTeacherSnapshotRecord]:
    records: list[FrozenTeacherSnapshotRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(FrozenTeacherSnapshotRecord.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid frozen snapshot at {path}:{line_number}") from exc
    if not records:
        raise ValueError("frozen snapshot file is empty")
    return records


def _read_accepted_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError("accepted ID file is empty")
    if len(values) != len(set(values)):
        raise ValueError("accepted ID file contains duplicate IDs")
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in values):
        raise ValueError("accepted ID file contains a non-SHA256 sample ID")
    return values


def _filter_accepted(
    records: Sequence[TeacherSnapshotSourceRecord],
    *,
    accepted_ids: Sequence[str],
    owner: str = "consensus",
) -> list[TeacherSnapshotSourceRecord]:
    requested = set(accepted_ids)
    selected = [item for item in records if item.sample_id in requested]
    missing = sorted(requested.difference(item.sample_id for item in selected))
    if missing:
        raise ValueError(f"accepted ID file references IDs absent from {owner}: {missing[:3]}")
    return selected


def _jsonl(records: Sequence[TeacherSnapshotTransitionRecord]) -> str:
    return "".join(canonical_json(item.model_dump(mode="json")) + "\n" for item in records)


def _reason_ledger(records: Sequence[TeacherSnapshotTransitionRecord]) -> str:
    rows = []
    for item in records:
        rows.append(
            {
                "schema_version": TEACHER_SNAPSHOT_DIFFER_VERSION,
                "sample_id": item.sample_id,
                "trajectory_id": item.trajectory_id,
                "source_turn_index": item.source_turn_index,
                "predecessor_sample_id": item.predecessor_sample_id,
                "status": item.status,
                "operation_types": (
                    [] if item.patch is None else [operation.op for operation in item.patch.operations]
                ),
                "target_decision_sha256": item.target_decision_sha256,
                "replay_decision_sha256": item.replay_decision_sha256,
                "reasons": [reason.model_dump(mode="json") for reason in item.reasons],
            }
        )
    return "".join(canonical_json(row) + "\n" for row in rows)


def _contiguous_ranges(values: Sequence[int]) -> list[list[int]]:
    unique = sorted(set(values))
    if not unique:
        return []
    output: list[list[int]] = []
    start = previous = unique[0]
    for value in unique[1:]:
        if value != previous + 1:
            output.append([start, previous])
            start = value
        previous = value
    output.append([start, previous])
    return output


def _trajectory_coverage(
    accepted: Sequence[TeacherSnapshotSourceRecord],
    transitions: Sequence[TeacherSnapshotTransitionRecord],
) -> list[dict[str, object]]:
    accepted_by_trajectory: dict[str, list[int]] = defaultdict(list)
    transitions_by_trajectory: dict[str, list[TeacherSnapshotTransitionRecord]] = defaultdict(list)
    for item in accepted:
        accepted_by_trajectory[item.trajectory_id].append(item.source_turn_index)
    for item in transitions:
        transitions_by_trajectory[item.trajectory_id].append(item)

    rows: list[dict[str, object]] = []
    for trajectory_id in sorted(accepted_by_trajectory):
        selected = sorted(accepted_by_trajectory[trajectory_id])
        trajectory_transitions = transitions_by_trajectory[trajectory_id]
        compiled = sorted(
            item.source_turn_index for item in trajectory_transitions if item.status == "compiled"
        )
        uncompilable = sorted(
            item.source_turn_index
            for item in trajectory_transitions
            if item.status == "uncompilable"
        )
        opening_through: int | None = None
        for value in compiled:
            expected = 0 if opening_through is None else opening_through + 1
            if value != expected:
                break
            opening_through = value
        selected_ranges = _contiguous_ranges(selected)
        rows.append(
            {
                "schema_version": "dsus_teacher_snapshot_trajectory_coverage_v1",
                "trajectory_id": trajectory_id,
                "selected_turn_indices": selected,
                "selected_turn_ranges": selected_ranges,
                "compiled_turn_indices": compiled,
                "compiled_turn_ranges": _contiguous_ranges(compiled),
                "uncompilable_turn_indices": uncompilable,
                "opening_contiguous_compiled_through": opening_through,
                "all_selected_compiled": len(compiled) == len(selected),
                "selected_has_turn_gap": len(selected_ranges) > 1 or selected[0] != 0,
            }
        )
    return rows


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite snapshot-differ artifact: {path}")
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    frozen_input = args.frozen_snapshots is not None
    source_value = args.frozen_snapshots if frozen_input else args.accepted_consensus
    assert source_value is not None
    input_path = source_value.expanduser().resolve()
    source_type = "frozen_snapshots" if frozen_input else "accepted_consensus"
    source_label = "frozen snapshots" if frozen_input else "consensus"
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    outputs = {
        "all": output_dir / "transitions.all.jsonl",
        "compiled": output_dir / "transitions.compiled.jsonl",
        "uncompilable": output_dir / "transitions.uncompilable.jsonl",
        "ledger": output_dir / "reason_ledger.jsonl",
        "coverage": output_dir / "trajectory_coverage.jsonl",
        "summary": output_dir / "summary.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite an existing snapshot-differ run")

    print("步骤 1/4：读取最终冻结或三教师已接受的完整状态快照；本程序不会调用任何 API。")
    all_records: list[TeacherSnapshotSourceRecord]
    if frozen_input:
        all_records = _read_frozen(input_path)
    else:
        all_records = _read_accepted(input_path)
    accepted_id_path = (
        None if args.accepted_id_file is None else args.accepted_id_file.expanduser().resolve()
    )
    if accepted_id_path is None:
        records = all_records
    else:
        accepted_ids = _read_accepted_ids(accepted_id_path)
        records = _filter_accepted(
            all_records,
            accepted_ids=accepted_ids,
            owner=source_label,
        )
    trajectory_count = len({item.trajectory_id for item in records})
    print(
        f"步骤 1/4 完成：上游 accepted={len(all_records)}，本次选择={len(records)}，"
        f"轨迹={trajectory_count}。"
    )

    print("步骤 2/4：按轨迹和 turn 排序，仅对 opening 或严格相邻快照做确定性差分。")
    transitions, summary = compile_accepted_snapshot_trajectories(records)
    print(
        f"步骤 2/4 完成：可编译={summary.compiled_count}，"
        f"不可编译={summary.uncompilable_count}；失败不会猜测补丁。"
    )

    print("步骤 3/4：核对所有可发布补丁均已由 production reduce_state 回放且决策等价。")
    if summary.replay_verified_count != summary.compiled_count:
        raise RuntimeError("replay verification count does not cover every compiled transition")
    print(
        f"步骤 3/4 完成：回放验证={summary.replay_verified_count}/"
        f"{summary.compiled_count}。"
    )

    print("步骤 4/4：写出 compiled/uncompilable、逐条原因账本和汇总。")
    compiled = [item for item in transitions if item.status == "compiled"]
    uncompilable = [item for item in transitions if item.status == "uncompilable"]
    _write_new(outputs["all"], _jsonl(transitions))
    _write_new(outputs["compiled"], _jsonl(compiled))
    _write_new(outputs["uncompilable"], _jsonl(uncompilable))
    _write_new(outputs["ledger"], _reason_ledger(transitions))
    coverage = _trajectory_coverage(records, transitions)
    _write_new(
        outputs["coverage"],
        "".join(canonical_json(row) + "\n" for row in coverage),
    )
    summary_payload = summary.model_dump(mode="json")
    summary_payload.update(
        {
            "source_type": source_type,
            "source_consensus": None if frozen_input else str(input_path),
            "source_consensus_sha256": (
                None
                if frozen_input
                else hashlib.sha256(input_path.read_bytes()).hexdigest()
            ),
            "source_frozen_snapshots": str(input_path) if frozen_input else None,
            "source_frozen_snapshots_sha256": (
                hashlib.sha256(input_path.read_bytes()).hexdigest()
                if frozen_input
                else None
            ),
            "accepted_id_file": None if accepted_id_path is None else str(accepted_id_path),
            "accepted_id_file_sha256": (
                None
                if accepted_id_path is None
                else hashlib.sha256(accepted_id_path.read_bytes()).hexdigest()
            ),
            "upstream_accepted_count": len(all_records),
            "selected_accepted_count": len(records),
            "trajectory_coverage_counts": dict(
                sorted(
                    Counter(
                        "all_selected_compiled" if row["all_selected_compiled"] else "partial"
                        for row in coverage
                    ).items()
                )
            ),
            "output_files": {key: str(path) for key, path in outputs.items()},
        }
    )
    _write_new(outputs["summary"], canonical_json(summary_payload) + "\n")
    print(
        f"步骤 4/4 完成：compiled={len(compiled)}，uncompilable={len(uncompilable)}，"
        f"目录={output_dir}。"
    )
    return output_dir


if __name__ == "__main__":
    main()
