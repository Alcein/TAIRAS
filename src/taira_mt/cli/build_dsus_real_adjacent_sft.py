"""Build the Real500 replay-certified adjacent-patch SFT view offline.

This command deliberately exports only the 142 transitions already released by
the snapshot differ.  It does not reinterpret the other 358 full snapshots as
transitions.  For a non-opening sample, the model input is reconstructed from
the frozen predecessor snapshot plus only the newly appended public messages.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.state_compiler import PublicDSUSState, StatePatch, reduce_state
from taira_mt.state_compiler.runtime import validate_public_evidence
from taira_mt.state_training.dataset_assembly import SFTTokenizerRecord
from taira_mt.state_training.teacher_snapshot_differ import (
    TeacherSnapshotTransitionRecord,
    _normalize_snapshot_as_previous_state,
    snapshot_decision_sha256,
)
from taira_mt.state_training.teacher_snapshot_freeze import FullSnapshotSFTRecord
from taira_mt.state_training.training_views import (
    DSUSPublicPrompt,
    EvidenceGroundedDeltaTarget,
)

SCHEMA_VERSION = "dsus_real_adjacent_patch_sft_v1"
EXPECTED_SPLIT_COUNTS = {"train": 111, "dev": 15, "test": 16}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    output = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            output.append(model.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid record at {path}:{line_number}") from exc
    if not output:
        raise ValueError(f"input JSONL is empty: {path}")
    return output


def _unique_by_sample(records: Sequence[Any], *, owner: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for record in records:
        if record.sample_id in output:
            raise ValueError(f"{owner} contains duplicate sample_id: {record.sample_id}")
        output[record.sample_id] = record
    return output


def _new_observations(
    *,
    current: FullSnapshotSFTRecord,
    previous: FullSnapshotSFTRecord | None,
) -> list[Any]:
    current_history = current.input.public_history
    if current_history is None:
        raise ValueError(f"final500 sample does not contain full public_history: {current.sample_id}")
    if previous is None:
        if current.source_turn_index != 0:
            raise ValueError(f"non-opening sample has no predecessor: {current.sample_id}")
        return list(current_history)
    previous_history = previous.input.public_history
    if previous_history is None:
        raise ValueError(f"predecessor does not contain full public_history: {previous.sample_id}")
    prefix_size = len(previous_history)
    if current_history[:prefix_size] != previous_history:
        raise ValueError(f"predecessor public_history is not an exact prefix: {current.sample_id}")
    suffix = list(current_history[prefix_size:])
    if not suffix:
        raise ValueError(f"adjacent transition has no new public observation: {current.sample_id}")
    return suffix


def _assistant_target(patch: StatePatch) -> str:
    target = EvidenceGroundedDeltaTarget(
        base_revision=patch.base_revision,
        gold_evidence=patch.evidence,
        gold_delta=patch.operations,
    )
    return canonical_json(target.assistant_payload())


def build_adjacent_sft(
    *,
    full_snapshot_sft_path: Path,
    compiled_transition_path: Path,
    output_dir: Path,
    expected_split_counts: dict[str, int] | None = None,
    enforce_real500_scope: bool = True,
) -> Path:
    full_snapshot_sft_path = full_snapshot_sft_path.expanduser().resolve()
    compiled_transition_path = compiled_transition_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    for path in (full_snapshot_sft_path, compiled_transition_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected = (
        EXPECTED_SPLIT_COUNTS
        if expected_split_counts is None and enforce_real500_scope
        else expected_split_counts
    )
    output_paths = {
        split: output_dir / f"tokenizer_sft.{split}.private.jsonl"
        for split in ("train", "dev", "test")
    }
    output_paths.update(
        {
            "lineage": output_dir / "sample_lineage.private.jsonl",
            "summary": output_dir / "summary.json",
            "manifest": output_dir / "manifest.json",
        }
    )
    if any(path.exists() for path in output_paths.values()):
        raise FileExistsError(f"refusing to overwrite an existing adjacent-SFT run: {output_dir}")

    print("步骤 1/4：读取冻结 full-snapshot SFT 与 differ 已回放通过的 compiled transitions；不会调用 API。")
    frozen = _read_jsonl(full_snapshot_sft_path, FullSnapshotSFTRecord)
    compiled = _read_jsonl(compiled_transition_path, TeacherSnapshotTransitionRecord)
    frozen_by_id = _unique_by_sample(frozen, owner="final500 SFT")
    compiled_by_id = _unique_by_sample(compiled, owner="compiled differ")
    if any(item.status != "compiled" for item in compiled):
        raise ValueError("compiled transition input contains a non-compiled row")
    missing = sorted(set(compiled_by_id).difference(frozen_by_id))
    if missing:
        raise ValueError(f"compiled differ IDs absent from final500 split: {missing[:3]}")
    print(f"步骤 1/4 完成：frozen snapshots={len(frozen)}，replay-certified={len(compiled)}。")

    print("步骤 2/4：按 sample_id 严格连接，并为非 opening 构造 predecessor state + latest observations。")
    tokenizer_rows: dict[str, list[SFTTokenizerRecord]] = {
        "train": [],
        "dev": [],
        "test": [],
    }
    lineage_rows: list[dict[str, object]] = []
    observation_mode_counts: Counter[str] = Counter()
    opening_count = 0
    nonopening_count = 0
    for transition in sorted(
        compiled,
        key=lambda item: (
            frozen_by_id[item.sample_id].split,
            frozen_by_id[item.sample_id].order_index,
            item.sample_id,
        ),
    ):
        current = frozen_by_id[transition.sample_id]
        if transition.patch is None or transition.replayed_state is None:
            raise ValueError(f"compiled row has no released patch: {transition.sample_id}")
        if (
            current.trajectory_id != transition.trajectory_id
            or current.source_turn_index != transition.source_turn_index
        ):
            raise ValueError(f"sample_id join metadata mismatch: {transition.sample_id}")

        predecessor: FullSnapshotSFTRecord | None = None
        if transition.predecessor_sample_id is None:
            previous_state = PublicDSUSState.empty()
            opening_count += 1
        else:
            predecessor = frozen_by_id.get(transition.predecessor_sample_id)
            if predecessor is None:
                raise ValueError(
                    f"predecessor ID absent from final500 split: {transition.predecessor_sample_id}"
                )
            if (
                predecessor.trajectory_id != current.trajectory_id
                or predecessor.source_turn_index + 1 != current.source_turn_index
                or predecessor.split != current.split
            ):
                raise ValueError(f"invalid adjacent predecessor join: {transition.sample_id}")
            previous_state = _normalize_snapshot_as_previous_state(
                predecessor.target,
                trajectory_id=current.trajectory_id,
                revision=current.source_turn_index,
            )
            nonopening_count += 1

        latest = _new_observations(current=current, previous=predecessor)
        prompt = DSUSPublicPrompt(
            previous_public_state=previous_state,
            latest_observations=latest,
        )
        patch = transition.patch
        if patch.base_revision != previous_state.revision:
            raise ValueError(f"patch base_revision mismatch: {transition.sample_id}")
        validate_public_evidence(patch.evidence, prompt=prompt)
        replayed = reduce_state(previous_state, patch)
        if replayed != transition.replayed_state:
            raise ValueError(f"replayed state differs from frozen differ output: {transition.sample_id}")
        if snapshot_decision_sha256(replayed) != transition.target_decision_sha256:
            raise ValueError(f"replayed decision differs from frozen target: {transition.sample_id}")

        row = SFTTokenizerRecord(
            sample_id=transition.sample_id,
            public_prompt_sha256=prompt.content_hash(),
            model_input=prompt.canonical_json(),
            assistant_target=_assistant_target(patch),
        )
        tokenizer_rows[current.split].append(row)
        observation_mode_counts["latest_observations"] += 1
        lineage_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": transition.sample_id,
                "split": current.split,
                "trajectory_id": current.trajectory_id,
                "source_turn_index": current.source_turn_index,
                "predecessor_sample_id": transition.predecessor_sample_id,
                "source_public_prompt_sha256": current.public_prompt_sha256,
                "adjacent_public_prompt_sha256": prompt.content_hash(),
                "patch_sha256": content_sha256(patch.model_dump(mode="json")),
                "production_reducer_replay_verified": True,
            }
        )

    actual_counts = {split: len(rows) for split, rows in tokenizer_rows.items()}
    if expected is not None and actual_counts != expected:
        raise ValueError(f"unexpected adjacent split counts: expected={expected}, actual={actual_counts}")
    if enforce_real500_scope and (
        len(compiled) != 142 or opening_count != 101 or nonopening_count != 41
    ):
        raise ValueError(
            "frozen adjacent scope changed: "
            f"total={len(compiled)}, opening={opening_count}, nonopening={nonopening_count}"
        )
    print(
        f"步骤 2/4 完成：{len(compiled)} 条均重放一致；"
        f"opening={opening_count}，真正相邻多轮={nonopening_count}，split={actual_counts}。"
    )

    print("步骤 3/4：写出与现有 SFT harness 兼容的 tokenizer 文件和逐条私有 lineage。")
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, rows in tokenizer_rows.items():
        output_paths[split].write_text(
            "".join(canonical_json(row.model_dump(mode="json")) + "\n" for row in rows),
            encoding="utf-8",
        )
    output_paths["lineage"].write_text(
        "".join(canonical_json(row) + "\n" for row in lineage_rows),
        encoding="utf-8",
    )
    print("步骤 3/4 完成：train/dev/test tokenizer SFT 与 lineage 已写出。")

    print("步骤 4/4：冻结 summary/manifest，明确它是 replay-certified adjacent 辅助视图。")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "view_role": "adjacent_patch_auxiliary_view",
        "full_snapshot_universe_count": len(frozen),
        "replay_certified_transition_count": len(compiled),
        "not_a_500_transition_dataset": True,
        "full_snapshot_transition_equivalence_claimed": False,
        "scope_contract": (
            "frozen_real500_replay142" if enforce_real500_scope else "generic_frozen_trajectory_panel"
        ),
        "opening_transition_count": opening_count,
        "nonopening_adjacent_transition_count": nonopening_count,
        "expected_split_counts": expected,
        "actual_split_counts": actual_counts,
        "observation_mode_counts": dict(sorted(observation_mode_counts.items())),
        "external_model_calls": False,
        "api_call_count": 0,
        "production_reducer_replay_verified_count": len(compiled),
    }
    output_paths["summary"].write_text(canonical_json(summary) + "\n", encoding="utf-8")
    output_hashes = {
        name: _sha256_file(path)
        for name, path in output_paths.items()
        if name not in {"manifest"} and path.is_file()
    }
    manifest = {
        **summary,
        "inputs": {
            "final500_full_snapshot_sft": {
                "path": str(full_snapshot_sft_path),
                "sha256": _sha256_file(full_snapshot_sft_path),
            },
            "snapshot_differ_compiled": {
                "path": str(compiled_transition_path),
                "sha256": _sha256_file(compiled_transition_path),
            },
        },
        "outputs": {
            name: {"path": str(path), "sha256": output_hashes[name]}
            for name, path in output_paths.items()
            if name in output_hashes
        },
        "contracts": {
            "join_key": "sample_id",
            "assistant_target": "dsus_state_patch_v2 as evidence + delta",
            "opening_input": "empty previous state + latest opening observation",
            "nonopening_input": (
                "normalized predecessor frozen snapshot + exact public_history suffix "
                "as latest_observations"
            ),
            "training_loader": "SFTTokenizerRecord/model_input/assistant_target",
        },
    }
    output_paths["manifest"].write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(f"步骤 4/4 完成：离线辅助视图已冻结到 {output_dir}。")
    return output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-snapshot-sft", type=Path, required=True)
    parser.add_argument("--compiled-transitions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    return build_adjacent_sft(
        full_snapshot_sft_path=args.full_snapshot_sft,
        compiled_transition_path=args.compiled_transitions,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
