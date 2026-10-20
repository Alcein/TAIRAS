"""Freeze evidence-first SFT targets without changing their DSUS semantics.

The runtime compiler asks the model to emit a patch in this top-level order::

    schema_version -> base_revision -> evidence -> delta

Historically, tokenizer SFT targets were written with globally sorted JSON keys,
which placed ``delta`` before ``evidence``.  This module performs a deliberately
narrow intervention: parse every target, validate it, and serialize only the
four top-level keys in the runtime order.  Nested values, model inputs, sample
identities, split membership, and parsed JSON meaning are preserved.

It also materializes one predeclared Real500-focused continuation view.  The
view uses every unique cold and adjacent example once per epoch plus a balanced
subset of a separately audited clean-retention set.  The complete mechanically
transformed source mix is retained beside it, so serialization and sampling
interventions remain independently auditable.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.state_compiler import StatePatch
from taira_mt.state_training.dataset_assembly import SFTTokenizerRecord
from taira_mt.state_training.training_views import DSUSPublicPrompt

SCHEMA_VERSION = "dsus_evidence_first_real_sft_v1"
TARGET_KEY_ORDER = ("schema_version", "base_revision", "evidence", "delta")
SPLITS = ("train", "dev", "test")
REAL_VIEWS = ("cold", "adjacent")
CLEAN_RETENTION_ROLES = ("opening", "rehearsal")
CLEAN_RETENTION_MECHANISMS = (
    "ambiguous",
    "conflict",
    "direct",
    "progressive",
    "revision",
)
CLEAN_SUBSET_SEED = "dsus-evidence-first-clean-retention-balanced-v1"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(payload)
    return rows


def evidence_first_target(assistant_target: str) -> tuple[str, str]:
    """Return runtime-ordered JSON and its order-insensitive semantic hash."""

    try:
        parsed = json.loads(assistant_target)
    except json.JSONDecodeError as exc:
        raise ValueError("assistant_target is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("assistant_target must be a JSON object")
    if set(parsed) != set(TARGET_KEY_ORDER):
        raise ValueError(
            "assistant_target top-level keys must be exactly "
            f"{TARGET_KEY_ORDER}; got {tuple(parsed)}"
        )

    # StatePatch validation checks operation/evidence structure and the reducer
    # contract without imposing a textual key order.
    StatePatch(
        schema_version=parsed["schema_version"],
        base_revision=parsed["base_revision"],
        evidence=parsed["evidence"],
        operations=parsed["delta"],
    )
    semantic_hash = content_sha256(parsed)
    reordered = {key: parsed[key] for key in TARGET_KEY_ORDER}
    rendered = json.dumps(reordered, ensure_ascii=False, separators=(",", ":"))
    if json.loads(rendered) != parsed or content_sha256(json.loads(rendered)) != semantic_hash:
        raise AssertionError("evidence-first serialization changed target semantics")
    return rendered, semantic_hash


def _validate_and_transform_row(
    payload: Mapping[str, Any],
) -> tuple[SFTTokenizerRecord, dict[str, Any]]:
    source = SFTTokenizerRecord.model_validate(payload)
    prompt = DSUSPublicPrompt.model_validate_json(source.model_input)
    if prompt.content_hash() != source.public_prompt_sha256:
        raise ValueError(f"public prompt hash mismatch for {source.sample_id}")

    transformed_target, semantic_hash = evidence_first_target(source.assistant_target)
    transformed = source.model_copy(update={"assistant_target": transformed_target})
    audit = {
        "source_assistant_target_sha256": _sha256_text(source.assistant_target),
        "evidence_first_assistant_target_sha256": _sha256_text(transformed_target),
        "semantic_target_sha256": semantic_hash,
        "model_input_sha256": _sha256_text(source.model_input),
    }
    return transformed, audit


def _write_tokenizer_rows(path: Path, rows: list[SFTTokenizerRecord]) -> None:
    path.write_text(
        "".join(canonical_json(row.model_dump(mode="json")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_source_mix(
    source_dir: Path,
) -> tuple[
    dict[str, list[SFTTokenizerRecord]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    lineage_rows = _read_jsonl(source_dir / "sample_lineage.private.jsonl")
    lineage_by_id: dict[str, dict[str, Any]] = {}
    for row in lineage_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in lineage_by_id:
            raise ValueError(f"missing or duplicate source lineage sample_id: {sample_id}")
        lineage_by_id[sample_id] = row

    transformed: dict[str, list[SFTTokenizerRecord]] = {}
    audits: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for split in SPLITS:
        rows: list[SFTTokenizerRecord] = []
        source_path = source_dir / f"tokenizer_sft.{split}.private.jsonl"
        for payload in _read_jsonl(source_path):
            row, audit = _validate_and_transform_row(payload)
            if row.sample_id in seen_ids:
                raise ValueError(f"sample_id repeats across source splits: {row.sample_id}")
            seen_ids.add(row.sample_id)
            lineage = lineage_by_id.get(row.sample_id)
            if lineage is None or lineage.get("split") != split:
                raise ValueError(f"source lineage split mismatch for {row.sample_id}")
            rows.append(row)
            audits[row.sample_id] = audit
        transformed[split] = rows
    if seen_ids != set(lineage_by_id):
        raise ValueError("source lineage and tokenizer rows do not have identical sample IDs")
    return transformed, audits, lineage_rows


def _load_clean_retention(
    clean_retention_dir: Path,
) -> tuple[
    dict[str, list[SFTTokenizerRecord]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    lineage_rows = _read_jsonl(clean_retention_dir / "sample_lineage.private.jsonl")
    lineage_by_id = {str(row.get("sample_id", "")): row for row in lineage_rows}
    if "" in lineage_by_id or len(lineage_by_id) != len(lineage_rows):
        raise ValueError("clean-retention lineage has missing or duplicate sample IDs")
    transformed: dict[str, list[SFTTokenizerRecord]] = {}
    audits: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for split in ("train", "dev"):
        rows: list[SFTTokenizerRecord] = []
        path = clean_retention_dir / f"tokenizer_sft.{split}.private.jsonl"
        for payload in _read_jsonl(path):
            row, audit = _validate_and_transform_row(payload)
            if row.sample_id in seen_ids:
                raise ValueError(f"clean-retention sample_id repeats: {row.sample_id}")
            lineage = lineage_by_id.get(row.sample_id)
            if lineage is None or lineage.get("calibration_split") != split:
                raise ValueError(f"clean-retention lineage split mismatch: {row.sample_id}")
            seen_ids.add(row.sample_id)
            rows.append(row)
            audits[row.sample_id] = audit
        transformed[split] = rows
    if seen_ids != set(lineage_by_id):
        raise ValueError("clean-retention lineage and tokenizer rows disagree")
    return transformed, audits, lineage_by_id


def _balanced_clean_retention_subset(
    rows: list[SFTTokenizerRecord],
    *,
    lineage_by_id: Mapping[str, Mapping[str, Any]],
    per_mechanism_role: int,
) -> list[SFTTokenizerRecord]:
    if per_mechanism_role < 1:
        raise ValueError("clean_per_mechanism_role must be positive")
    groups: dict[tuple[str, str], list[SFTTokenizerRecord]] = defaultdict(list)
    for row in rows:
        lineage = lineage_by_id[row.sample_id]
        groups[(str(lineage.get("mechanism")), str(lineage.get("role")))].append(row)

    selected_ids: set[str] = set()
    for mechanism in CLEAN_RETENTION_MECHANISMS:
        for role in CLEAN_RETENTION_ROLES:
            candidates = groups[(mechanism, role)]
            if len(candidates) < per_mechanism_role:
                raise ValueError(
                    f"clean-retention capacity is insufficient for {mechanism}/{role}: "
                    f"{len(candidates)} < {per_mechanism_role}"
                )
            ranked = sorted(
                candidates,
                key=lambda row: hashlib.sha256(
                    f"{CLEAN_SUBSET_SEED}|{row.sample_id}".encode()
                ).hexdigest(),
            )
            selected_ids.update(row.sample_id for row in ranked[:per_mechanism_role])
    # Preserve the frozen source order after deterministic membership selection.
    return [row for row in rows if row.sample_id in selected_ids]


def build_evidence_first_real_sft(
    *,
    source_mix_dir: Path,
    clean_retention_dir: Path,
    output_dir: Path,
    cold_repeat_limit: int = 1,
    clean_per_mechanism_role: int = 12,
) -> Path:
    """Build a frozen semantic-equivalent source view and focused train view."""

    source_mix_dir = source_mix_dir.expanduser().resolve()
    clean_retention_dir = clean_retention_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if cold_repeat_limit < 1:
        raise ValueError("cold_repeat_limit must be positive")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {output_dir}")

    print("Step 1/5: reading the legacy Real500 three-view mixture and validating each DSUS structure; no API calls.", flush=True)
    transformed, source_audits, source_lineage = _load_source_mix(source_mix_dir)
    source_counts = {split: len(rows) for split, rows in transformed.items()}
    print(f"Step 1/5 complete: validated and transformed {sum(source_counts.values())} records, split={source_counts}.", flush=True)

    print("Step 2/5: validating clean-retention data and applying the same evidence-first serialization.", flush=True)
    clean, clean_audits, clean_lineage = _load_clean_retention(clean_retention_dir)
    print(
        f"Step 2/5 complete: clean-retention train/dev={len(clean['train'])}/{len(clean['dev'])}.",
        flush=True,
    )

    print("Step 3/5: constructing Real-focused continuation-training views and Real-only checkpoint selection slices.", flush=True)
    lineage_by_id = {str(row["sample_id"]): row for row in source_lineage}
    focused_real_train = [
        row
        for row in transformed["train"]
        if (
            lineage_by_id[row.sample_id]["source_view"] == "adjacent"
            or (
                lineage_by_id[row.sample_id]["source_view"] == "cold"
                and int(lineage_by_id[row.sample_id]["repeat_index"]) < cold_repeat_limit
            )
        )
    ]
    real_dev = [
        row
        for row in transformed["dev"]
        if lineage_by_id[row.sample_id]["source_view"] in REAL_VIEWS
    ]
    real_test = [
        row
        for row in transformed["test"]
        if lineage_by_id[row.sample_id]["source_view"] in REAL_VIEWS
    ]
    selected_clean_train = _balanced_clean_retention_subset(
        clean["train"],
        lineage_by_id=clean_lineage,
        per_mechanism_role=clean_per_mechanism_role,
    )
    focused_train = [*focused_real_train, *selected_clean_train]
    focused_ids = [row.sample_id for row in focused_train]
    if len(focused_ids) != len(set(focused_ids)):
        raise ValueError("focused train contains colliding sample IDs")
    clean_ratio = len(selected_clean_train) / len(focused_train)
    if not 0.20 <= clean_ratio <= 0.30:
        raise ValueError(f"clean-retention ratio is outside [20%, 30%]: {clean_ratio:.4f}")
    focused_counts = Counter(
        (
            "clean_retention"
            if row.sample_id in clean_audits
            else str(lineage_by_id[row.sample_id]["source_view"])
        )
        for row in focused_train
    )
    clean_balance = Counter(
        (
            str(clean_lineage[row.sample_id]["mechanism"]),
            str(clean_lineage[row.sample_id]["role"]),
        )
        for row in selected_clean_train
    )
    print(
        "Step 3/5 complete: focused train="
        f"{len(focused_train)} ({dict(sorted(focused_counts.items()))}, clean={clean_ratio:.2%}), "
        f"Real-only dev/test={len(real_dev)}/{len(real_test)}.",
        flush=True,
    )

    print("Step 4/5: writing complete semantically equivalent views, focused training views, and per-sample audit ledgers.", flush=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_paths: dict[str, Path] = {}
    for split, rows in transformed.items():
        name = f"tokenizer_sft.{split}.private.jsonl"
        output_paths[name] = output_dir / name
        _write_tokenizer_rows(output_paths[name], rows)
    focused_name = "tokenizer_sft.train.real_focused_clean20.private.jsonl"
    output_paths[focused_name] = output_dir / focused_name
    _write_tokenizer_rows(output_paths[focused_name], focused_train)
    real_dev_name = "tokenizer_sft.dev.real_only.private.jsonl"
    output_paths[real_dev_name] = output_dir / real_dev_name
    _write_tokenizer_rows(output_paths[real_dev_name], real_dev)
    real_test_name = "tokenizer_sft.test.real_only.private.jsonl"
    output_paths[real_test_name] = output_dir / real_test_name
    _write_tokenizer_rows(output_paths[real_test_name], real_test)
    clean_dev_name = "tokenizer_sft.dev.clean_retention.private.jsonl"
    output_paths[clean_dev_name] = output_dir / clean_dev_name
    _write_tokenizer_rows(output_paths[clean_dev_name], clean["dev"])

    transformed_lineage: list[dict[str, Any]] = []
    for source in source_lineage:
        sample_id = str(source["sample_id"])
        audit = source_audits[sample_id]
        transformed_lineage.append(
            {
                **source,
                "schema_version": SCHEMA_VERSION,
                "source_lineage_schema_version": source.get("schema_version"),
                **audit,
                "target_top_level_order": list(TARGET_KEY_ORDER),
            }
        )
    lineage_name = "sample_lineage.private.jsonl"
    output_paths[lineage_name] = output_dir / lineage_name
    output_paths[lineage_name].write_text(
        "".join(canonical_json(row) + "\n" for row in transformed_lineage),
        encoding="utf-8",
    )
    print("Step 4/5 complete: all data files written; the legacy directory remains read-only and unchanged.", flush=True)

    print("Step 5/5: freezing summary, provenance, and file hashes.", flush=True)
    source_view_counts = {
        split: dict(
            sorted(Counter(str(lineage_by_id[row.sample_id]["source_view"]) for row in rows).items())
        )
        for split, rows in transformed.items()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_untrained",
        "serialization_intervention": {
            "source_top_level_order": ["base_revision", "delta", "evidence", "schema_version"],
            "target_top_level_order": list(TARGET_KEY_ORDER),
            "parsed_json_semantics_changed_count": 0,
            "model_input_changed_count": 0,
            "sample_id_changed_count": 0,
        },
        "complete_semantic_equivalent_view": {
            "counts": source_counts,
            "source_view_counts": source_view_counts,
            "total_count": sum(source_counts.values()),
        },
        "real_focused_continuation_view": {
            "train_count": len(focused_train),
            "train_source_counts": dict(sorted(focused_counts.items())),
            "clean_retention_ratio": clean_ratio,
            "cold_repeat_limit": cold_repeat_limit,
            "real_examples_unique_per_epoch": True,
            "clean_retention_selection": {
                "seed": CLEAN_SUBSET_SEED,
                "per_mechanism_role": clean_per_mechanism_role,
                "selected_count": len(selected_clean_train),
                "balance_counts": {
                    f"{mechanism}/{role}": clean_balance[(mechanism, role)]
                    for mechanism in CLEAN_RETENTION_MECHANISMS
                    for role in CLEAN_RETENTION_ROLES
                },
                "selected_sample_id_set_sha256": content_sha256(
                    sorted(row.sample_id for row in selected_clean_train)
                ),
            },
            "real_only_dev_count": len(real_dev),
            "real_only_test_count": len(real_test),
            "clean_retention_dev_count": len(clean["dev"]),
            "checkpoint_selection_primary_slice": real_dev_name,
            "retention_monitor_slice": clean_dev_name,
        },
        "external_model_calls": False,
        "api_call_count": 0,
        "training_launched": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    output_paths["summary.json"] = summary_path
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "source_mix": {
            "path": str(source_mix_dir),
            "manifest_sha256": _sha256_file(source_mix_dir / "manifest.json"),
            "summary_sha256": _sha256_file(source_mix_dir / "summary.json"),
            "tokenizer_sha256": {
                split: _sha256_file(source_mix_dir / f"tokenizer_sft.{split}.private.jsonl")
                for split in SPLITS
            },
            "lineage_sha256": _sha256_file(source_mix_dir / "sample_lineage.private.jsonl"),
        },
        "clean_retention": {
            "path": str(clean_retention_dir),
            "manifest_sha256": _sha256_file(clean_retention_dir / "manifest.json"),
            "tokenizer_sha256": {
                split: _sha256_file(clean_retention_dir / f"tokenizer_sft.{split}.private.jsonl")
                for split in ("train", "dev")
            },
        },
        "outputs": {
            name: {
                "sha256": _sha256_file(path),
                "row_count": len(_read_jsonl(path)) if path.suffix == ".jsonl" else None,
            }
            for name, path in sorted(output_paths.items())
        },
        "contracts": {
            "assistant_target_parsed_json_semantics_preserved": True,
            "assistant_target_semantic_hash_algorithm": "sha256(canonical_json(parsed_target))",
            "model_input_bytes_preserved": True,
            "source_sample_ids_and_splits_preserved": True,
            "complete_source_mix_counts_preserved": True,
            "focused_sampling_is_a_separate_named_view": True,
            "old_rehearsal_excluded_from_focused_train": True,
            "clean_retention_source_is_train_only_and_independently_audited": True,
            "clean_retention_subset_is_balanced_and_deterministic": True,
            "failed_delta_first_checkpoint_is_not_the_formal_warm_start": True,
            "training_launched": False,
        },
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(canonical_json(provenance) + "\n", encoding="utf-8")
    print(f"Step 5/5 complete: evidence-first SFT frozen at {output_dir}.", flush=True)
    return output_dir
