"""Build reducer-verified cold-reconstruction SFT views from frozen snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from taira_mt.provenance import canonical_json
from taira_mt.state_training.dataset_assembly import SFTTokenizerRecord
from taira_mt.state_training.sft_harness import (
    _assistant_boundary_and_full_ids,
    build_token_length_report,
)
from taira_mt.state_training.teacher_snapshot_cold_reconstruction import (
    ColdReconstructionSummary,
    ColdReconstructionTransitionRecord,
    build_cold_reconstruction_views,
)
from taira_mt.state_training.teacher_snapshot_freeze import FullSnapshotSFTRecord


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-only: convert full-current-state teacher labels into EMPTY + full-history "
            "StatePatch recovery supervision and verify every released patch with the production reducer."
        )
    )
    parser.add_argument("--full-snapshot-sft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Local Qwen tokenizer or adapter directory; this command never downloads a model.",
    )
    parser.add_argument("--max-seq-length", type=int, default=4096, choices=(4096,))
    parser.add_argument(
        "--exclude-overlength-from-tokenizer-sft",
        action="store_true",
        help=(
            "Keep the complete reducer-verified transition, but exclude any chat-template "
            "sequence above max_seq_length from tokenizer_sft and bind a private ledger. "
            "Gold is never truncated."
        ),
    )
    parser.add_argument(
        "--adjacent-summary",
        type=Path,
        help="Optional existing adjacent-differ summary; it is read only and remains independent.",
    )
    parser.add_argument(
        "--adjacent-compiled",
        type=Path,
        help="Optional existing adjacent compiled JSONL used only to count non-opening rows.",
    )
    return parser.parse_args(argv)


def _read_full_snapshot_sft(path: Path) -> list[FullSnapshotSFTRecord]:
    records: list[FullSnapshotSFTRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(FullSnapshotSFTRecord.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid full snapshot SFT row at {path}:{line_number}") from exc
    if not records:
        raise ValueError("full snapshot SFT file is empty")
    return records


def _jsonl(rows: Sequence[Any]) -> str:
    return "".join(canonical_json(row.model_dump(mode="json")) + "\n" for row in rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite cold-reconstruction artifact: {path}")
    path.write_text(text, encoding="utf-8")


def _adjacent_counts(summary_path: Path, compiled_path: Path) -> tuple[int, int, str]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    compiled_count = int(summary.get("compiled_count", -1))
    if compiled_count < 0:
        raise ValueError("adjacent summary lacks compiled_count")
    non_opening = 0
    rows = 0
    for line_number, line in enumerate(compiled_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("source_turn_index"), int):
            raise ValueError(f"invalid adjacent compiled row at line {line_number}")
        rows += 1
        non_opening += int(row["source_turn_index"] > 0)
    if rows != compiled_count:
        raise ValueError("adjacent summary and compiled JSONL counts disagree")
    return compiled_count, non_opening, _sha256(summary_path)


def _partition_tokenizer_rows_by_length(
    *,
    rows_by_split: Mapping[str, Sequence[SFTTokenizerRecord]],
    transitions: Sequence[ColdReconstructionTransitionRecord],
    tokenizer: Any,
    max_seq_length: int,
) -> tuple[dict[str, list[SFTTokenizerRecord]], list[dict[str, Any]], int]:
    """Exclude overlength tokenizer rows while retaining their complete transitions."""

    transition_by_id = {row.sample_id: row for row in transitions}
    if len(transition_by_id) != len(transitions):
        raise ValueError("cold transitions repeat sample IDs")
    included: dict[str, list[SFTTokenizerRecord]] = {
        split: [] for split in ("train", "dev", "test")
    }
    excluded: list[dict[str, Any]] = []
    maximum_observed = 0
    for split in ("train", "dev", "test"):
        for row in rows_by_split[split]:
            _, full_ids = _assistant_boundary_and_full_ids(tokenizer, row)
            observed = len(full_ids)
            maximum_observed = max(maximum_observed, observed)
            if observed <= max_seq_length:
                included[split].append(row)
                continue
            transition = transition_by_id[row.sample_id]
            excluded.append(
                {
                    "schema_version": "dsus_teacher_snapshot_cold_overlength_exclusion_v1",
                    "sample_id": row.sample_id,
                    "public_prompt_sha256": row.public_prompt_sha256,
                    "trajectory_id": transition.trajectory_id,
                    "source_turn_index": transition.source_turn_index,
                    "split": split,
                    "observed_total_tokens": observed,
                    "max_seq_length": max_seq_length,
                    "reason": "chat_template_total_tokens_exceed_max_seq_length",
                    "full_transition_retained": True,
                    "tokenizer_sft_included": False,
                    "gold_truncated": False,
                }
            )
    return included, excluded, maximum_observed


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    source = args.full_snapshot_sft.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)
    if (args.adjacent_summary is None) != (args.adjacent_compiled is None):
        raise ValueError("provide --adjacent-summary and --adjacent-compiled together")

    paths = {
        "transitions": output_dir / "cold_reconstruction.transitions.private.jsonl",
        "limits": output_dir / "representation_limited.private.jsonl",
        "overlength_exclusions": output_dir / "overlength_excluded.private.jsonl",
        "train": output_dir / "tokenizer_sft.train.private.jsonl",
        "dev": output_dir / "tokenizer_sft.dev.private.jsonl",
        "test": output_dir / "tokenizer_sft.test.private.jsonl",
        "token_train": output_dir / "token_length_report.train.json",
        "token_dev": output_dir / "token_length_report.dev.json",
        "token_test": output_dir / "token_length_report.test.json",
        "summary": output_dir / "summary.json",
        "manifest": output_dir / "manifest.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite an existing cold-reconstruction build")

    print("Step 1/5: strictly reading frozen full-snapshot SFT; no API or GPU use.", flush=True)
    records = _read_full_snapshot_sft(source)
    print(f"Step 1/5 complete: read {len(records)} frozen snapshots.", flush=True)

    print("Step 2/5: compiling full-history recovery patches from EMPTY and replaying through the production reducer.", flush=True)
    result = build_cold_reconstruction_views(records)
    print(
        "Step 2/5 complete: "
        f"replay-verified={len(result.transitions)}, "
        f"representation-limited={len(result.representation_limits)}.",
        flush=True,
    )

    print("Step 3/5: writing complete transitions and the representation-limit ledger.", flush=True)
    _write_new(paths["transitions"], _jsonl(result.transitions))
    _write_new(paths["limits"], _jsonl(result.representation_limits))
    print(
        "Step 3/5 complete: complete cold transitions retained; the original adjacent artifact is unchanged.",
        flush=True,
    )

    print("Step 4/5: running an actual chat-template 4096-token preflight check with the local tokenizer.", flush=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer_rows: Mapping[str, Sequence[SFTTokenizerRecord]] = (
        result.tokenizer_records_by_split
    )
    excluded_rows: list[dict[str, Any]] = []
    maximum_observed = 0
    if args.exclude_overlength_from_tokenizer_sft:
        tokenizer_rows, excluded_rows, maximum_observed = _partition_tokenizer_rows_by_length(
            rows_by_split=result.tokenizer_records_by_split,
            transitions=result.transitions,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
        _write_new(
            paths["overlength_exclusions"],
            "".join(canonical_json(row) + "\n" for row in excluded_rows),
        )
    for split in ("train", "dev", "test"):
        _write_new(paths[split], _jsonl(tokenizer_rows[split]))
    token_reports = {}
    for split in ("train", "dev", "test"):
        report = build_token_length_report(
            dataset_path=paths[split],
            tokenizer=tokenizer,
            tokenizer_name_or_path=str(tokenizer_path),
            max_seq_length=args.max_seq_length,
            progress=lambda message, split=split: print(f"[{split}] {message}", flush=True),
        )
        token_reports[split] = report
        _write_new(paths[f"token_{split}"], canonical_json(report.model_dump(mode="json")) + "\n")
    released_over_limit = sum(report.over_limit_count for report in token_reports.values())
    maximum = max(
        maximum_observed,
        max(report.total_tokens.maximum for report in token_reports.values()),
    )
    if released_over_limit:
        raise ValueError(
            f"4096-token contract failed for {released_over_limit} samples; "
            "outputs retained for diagnosis"
        )
    excluded_by_split = Counter(str(row["split"]) for row in excluded_rows)
    over_limit = len(excluded_rows)
    print(
        f"Step 4/5 complete: maximum observed length={maximum}, tokenizer_sft over limit=0, "
        f"complete transitions retained but excluded from training={over_limit}.",
        flush=True,
    )

    print("Step 5/5: registering the separate adjacent artifact and freezing the summary and hash manifest.", flush=True)
    updates: dict[str, Any] = {
        "tokenizer_audited": True,
        "token_length_reports": {
            split: report.model_dump(mode="json") for split, report in token_reports.items()
        },
        "over_limit_count": over_limit,
        "tokenizer_record_counts": {
            split: len(tokenizer_rows[split]) for split in ("train", "dev", "test")
        },
        "tokenizer_overlength_handling": (
            "exclude_from_tokenizer_sft_retain_transition"
            if args.exclude_overlength_from_tokenizer_sft
            else "fail_build"
        ),
        "tokenizer_overlength_excluded_by_split": dict(sorted(excluded_by_split.items())),
        "tokenizer_overlength_exclusion_ledger_sha256": (
            _sha256(paths["overlength_exclusions"])
            if args.exclude_overlength_from_tokenizer_sft
            else None
        ),
        "maximum_observed_tokens": maximum,
    }
    adjacent_sources: dict[str, str] = {}
    if args.adjacent_summary is not None and args.adjacent_compiled is not None:
        adjacent_summary = args.adjacent_summary.expanduser().resolve()
        adjacent_compiled = args.adjacent_compiled.expanduser().resolve()
        compiled_count, non_opening_count, adjacent_sha = _adjacent_counts(
            adjacent_summary,
            adjacent_compiled,
        )
        updates.update(
            {
                "adjacent_compiled_count": compiled_count,
                "adjacent_non_opening_compiled_count": non_opening_count,
                "adjacent_artifact_sha256": adjacent_sha,
            }
        )
        adjacent_sources = {
            "adjacent_summary": str(adjacent_summary),
            "adjacent_compiled": str(adjacent_compiled),
        }
    summary = ColdReconstructionSummary.model_validate(
        result.summary.model_copy(update=updates).model_dump(mode="json")
    )
    _write_new(paths["summary"], canonical_json(summary.model_dump(mode="json")) + "\n")
    output_hashes = {
        name: _sha256(path)
        for name, path in paths.items()
        if name not in {"manifest"} and path.exists()
    }
    manifest = {
        "schema_version": "dsus_teacher_snapshot_cold_manifest_v1",
        "source_full_snapshot_sft": str(source),
        "source_full_snapshot_sft_sha256": _sha256(source),
        "tokenizer": str(tokenizer_path),
        "max_seq_length": args.max_seq_length,
        "adjacent_sources_read_only": adjacent_sources,
        "overlength_policy": {
            "handling": summary.tokenizer_overlength_handling,
            "excluded_count": summary.over_limit_count,
            "excluded_by_split": summary.tokenizer_overlength_excluded_by_split,
            "private_ledger_sha256": summary.tokenizer_overlength_exclusion_ledger_sha256,
            "full_transition_count_retained": len(result.transitions),
            "gold_truncated": False,
        },
        "output_files": {
            name: str(path)
            for name, path in paths.items()
            if name != "overlength_exclusions" or path.exists()
        },
        "output_sha256": output_hashes,
        "external_model_calls": False,
        "gpu_used": False,
    }
    _write_new(paths["manifest"], canonical_json(manifest) + "\n")
    print(
        f"Step 5/5 complete: artifact={output_dir}; cold={summary.compiled_cold_reconstruction_count}, "
        f"limited={summary.representation_limited_count}, "
        f"overlength-excluded={summary.over_limit_count}.",
        flush=True,
    )
    return output_dir


if __name__ == "__main__":
    main()
