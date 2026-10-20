"""Read-only Amazon source audit for one fixed benchmark case per user."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json, content_sha256

BENCHMARK_SOURCE_AUDIT_VERSION = "benchmark_source_audit_v1"
PSEUDONYM_VERSION = "amazon_user_pseudonym_v1"
ANCHOR_HISTORY_CONTRACT = "sequence_first_is_held_out_anchor_remaining_items_are_history_v1"
BENCHMARK_DOMAIN_DIRECTORIES = (
    "amazon_beauty",
    "amazon_clothing",
    "amazon_music",
)


class SourceRejectionReason(str, Enum):
    MISSING_USER_ID = "missing_user_id"
    DUPLICATE_USER_ID = "duplicate_user_id"
    SHORT_SEQUENCE = "short_sequence"
    DUPLICATE_ANCHOR_IN_HISTORY = "duplicate_anchor_in_history"
    MISSING_ANCHOR_METADATA = "missing_anchor_metadata"
    INSUFFICIENT_JOINED_HISTORY = "insufficient_joined_history"


class SourceAuditConfig(StrictModel):
    schema_version: str = BENCHMARK_SOURCE_AUDIT_VERSION
    anchor_history_contract: str = ANCHOR_HISTORY_CONTRACT
    minimum_sequence_length: int = Field(default=4, ge=2)
    minimum_joined_history_items: int = Field(default=3, ge=1)
    pseudonym_namespace: str = "taira-mt-public-v1"

    @model_validator(mode="after")
    def validate_versions(self) -> SourceAuditConfig:
        if self.schema_version != BENCHMARK_SOURCE_AUDIT_VERSION:
            raise ValueError(f"unsupported source audit version: {self.schema_version}")
        if self.anchor_history_contract != ANCHOR_HISTORY_CONTRACT:
            raise ValueError("unsupported anchor/history contract")
        return self


class SourceFileFingerprint(StrictModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    byte_size: int = Field(ge=0)
    row_count: int = Field(ge=0)
    columns: list[str]


class EligibleUserSource(StrictModel):
    domain: str = Field(min_length=1)
    pseudonymous_user_id: str = Field(min_length=1)
    source_sequence_sha256: str = Field(min_length=64, max_length=64)
    anchor_item_id: str = Field(min_length=1)
    joined_history_item_ids: list[str] = Field(min_length=1)
    missing_history_item_ids: list[str] = Field(default_factory=list)
    original_sequence_length: int = Field(ge=2)

    @field_validator("joined_history_item_ids", "missing_history_item_ids")
    @classmethod
    def validate_unique_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("eligible source history IDs must be unique")
        return value

    @model_validator(mode="after")
    def validate_source(self) -> EligibleUserSource:
        if self.anchor_item_id in set(self.joined_history_item_ids + self.missing_history_item_ids):
            raise ValueError("held-out anchor appears in source history")
        if set(self.joined_history_item_ids).intersection(self.missing_history_item_ids):
            raise ValueError("joined and missing history IDs overlap")
        return self


class DomainSourceAudit(StrictModel):
    domain: str = Field(min_length=1)
    metadata_source: SourceFileFingerprint
    user_source: SourceFileFingerprint
    metadata_item_count: int = Field(ge=0)
    duplicate_metadata_item_count: int = Field(ge=0)
    user_row_count: int = Field(ge=0)
    unique_user_count: int = Field(ge=0)
    sequence_length_min: int = Field(ge=0)
    sequence_length_p25: float = Field(ge=0.0)
    sequence_length_median: float = Field(ge=0.0)
    sequence_length_p75: float = Field(ge=0.0)
    sequence_length_max: int = Field(ge=0)
    metadata_join_rate: float = Field(ge=0.0, le=1.0)
    anchor_join_rate: float = Field(ge=0.0, le=1.0)
    eligible_user_count: int = Field(ge=0)
    rejection_counts: dict[str, int]


class BenchmarkSourceAudit(StrictModel):
    schema_version: str = BENCHMARK_SOURCE_AUDIT_VERSION
    audit_id: str = Field(min_length=64, max_length=64)
    config: SourceAuditConfig
    domains: list[DomainSourceAudit]
    total_users: int = Field(ge=0)
    total_eligible_users: int = Field(ge=0)
    eligible_records_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_totals(self) -> BenchmarkSourceAudit:
        if self.schema_version != BENCHMARK_SOURCE_AUDIT_VERSION:
            raise ValueError(f"unsupported source audit version: {self.schema_version}")
        if self.total_users != sum(domain.user_row_count for domain in self.domains):
            raise ValueError("total user count differs from domain audits")
        if self.total_eligible_users != sum(domain.eligible_user_count for domain in self.domains):
            raise ValueError("total eligible user count differs from domain audits")
        return self


class ArtifactFileRecord(StrictModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    byte_size: int = Field(ge=0)
    row_count: int | None = Field(default=None, ge=0)


class SourceAuditArtifactManifest(StrictModel):
    schema_version: str = BENCHMARK_SOURCE_AUDIT_VERSION
    artifact_role: str = "benchmark_source_audit"
    audit_id: str = Field(min_length=64, max_length=64)
    files: list[ArtifactFileRecord]


@dataclass(frozen=True)
class SourceAuditResult:
    audit: BenchmarkSourceAudit
    eligible_users: tuple[EligibleUserSource, ...]


@dataclass(frozen=True)
class _DomainAuditResult:
    audit: DomainSourceAudit
    eligible_users: tuple[EligibleUserSource, ...]


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_sequence(raw: str | None) -> list[str]:
    return [item.strip() for item in (raw or "").split("|") if item.strip()]


def _quantile(values: list[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _pseudonym(domain: str, raw_user_id: str, config: SourceAuditConfig) -> str:
    digest = content_sha256(
        {
            "version": PSEUDONYM_VERSION,
            "namespace": config.pseudonym_namespace,
            "domain": domain,
            "raw_user_id": raw_user_id,
        }
    )
    return f"user-{digest[:24]}"


def _source_fingerprint(path: Path, *, relative_to: Path, row_count: int, columns: list[str]) -> SourceFileFingerprint:
    return SourceFileFingerprint(
        relative_path=str(path.relative_to(relative_to)),
        sha256=_file_sha256(path),
        byte_size=path.stat().st_size,
        row_count=row_count,
        columns=columns,
    )


def _read_metadata_ids(path: Path) -> tuple[set[str], int, int, list[str]]:
    item_ids: set[str] = set()
    duplicate_count = 0
    row_count = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        missing = {"id", "title", "category"}.difference(columns)
        if missing:
            raise ValueError(f"metadata is missing required columns: {sorted(missing)}")
        for row in reader:
            row_count += 1
            item_id = (row.get("id") or "").strip()
            if not item_id:
                continue
            if item_id in item_ids:
                duplicate_count += 1
            item_ids.add(item_id)
    return item_ids, duplicate_count, row_count, columns


def _read_user_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        missing = {"user_id", "interaction_string"}.difference(columns)
        if missing:
            raise ValueError(f"user source is missing required columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    return rows, columns


def audit_domain_source(
    source_root: Path,
    domain: str,
    config: SourceAuditConfig | None = None,
) -> _DomainAuditResult:
    """Audit one domain and emit one eligible source record per unique user."""

    config = config or SourceAuditConfig()
    source_root = source_root.resolve()
    domain_root = source_root / domain
    metadata_path = domain_root / "metadata.csv"
    users_path = domain_root / "user_profile.csv"
    for path in (metadata_path, users_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata_ids, duplicate_metadata, metadata_rows, metadata_columns = _read_metadata_ids(metadata_path)
    user_rows, user_columns = _read_user_rows(users_path)
    user_id_counts = Counter(
        (row.get("user_id") or "").strip()
        for row in user_rows
        if (row.get("user_id") or "").strip()
    )

    sequence_lengths: list[int] = []
    total_interactions = 0
    joined_interactions = 0
    anchor_rows = 0
    joined_anchors = 0
    rejections: Counter[str] = Counter()
    eligible: list[EligibleUserSource] = []

    for row in user_rows:
        raw_user_id = (row.get("user_id") or "").strip()
        sequence = _parse_sequence(row.get("interaction_string"))
        sequence_lengths.append(len(sequence))
        total_interactions += len(sequence)
        joined_interactions += sum(item_id in metadata_ids for item_id in sequence)
        if sequence:
            anchor_rows += 1
            joined_anchors += int(sequence[0] in metadata_ids)

        reason: SourceRejectionReason | None = None
        if not raw_user_id:
            reason = SourceRejectionReason.MISSING_USER_ID
        elif user_id_counts[raw_user_id] != 1:
            reason = SourceRejectionReason.DUPLICATE_USER_ID
        elif len(sequence) < config.minimum_sequence_length:
            reason = SourceRejectionReason.SHORT_SEQUENCE
        else:
            anchor = sequence[0]
            history = sequence[1:]
            if anchor in history:
                reason = SourceRejectionReason.DUPLICATE_ANCHOR_IN_HISTORY
            elif anchor not in metadata_ids:
                reason = SourceRejectionReason.MISSING_ANCHOR_METADATA
            else:
                joined_history = list(dict.fromkeys(item_id for item_id in history if item_id in metadata_ids))
                missing_history = list(dict.fromkeys(item_id for item_id in history if item_id not in metadata_ids))
                if len(joined_history) < config.minimum_joined_history_items:
                    reason = SourceRejectionReason.INSUFFICIENT_JOINED_HISTORY
                else:
                    eligible.append(
                        EligibleUserSource(
                            domain=domain,
                            pseudonymous_user_id=_pseudonym(domain, raw_user_id, config),
                            source_sequence_sha256=content_sha256(
                                {
                                    "contract": config.anchor_history_contract,
                                    "domain": domain,
                                    "sequence": sequence,
                                }
                            ),
                            anchor_item_id=anchor,
                            joined_history_item_ids=joined_history,
                            missing_history_item_ids=missing_history,
                            original_sequence_length=len(sequence),
                        )
                    )
        if reason is not None:
            rejections[reason.value] += 1

    eligible.sort(key=lambda record: (record.pseudonymous_user_id, record.anchor_item_id))
    metadata_source = _source_fingerprint(
        metadata_path,
        relative_to=source_root,
        row_count=metadata_rows,
        columns=metadata_columns,
    )
    user_source = _source_fingerprint(
        users_path,
        relative_to=source_root,
        row_count=len(user_rows),
        columns=user_columns,
    )
    audit = DomainSourceAudit(
        domain=domain,
        metadata_source=metadata_source,
        user_source=user_source,
        metadata_item_count=len(metadata_ids),
        duplicate_metadata_item_count=duplicate_metadata,
        user_row_count=len(user_rows),
        unique_user_count=len(user_id_counts),
        sequence_length_min=min(sequence_lengths, default=0),
        sequence_length_p25=_quantile(sequence_lengths, 0.25),
        sequence_length_median=_quantile(sequence_lengths, 0.5),
        sequence_length_p75=_quantile(sequence_lengths, 0.75),
        sequence_length_max=max(sequence_lengths, default=0),
        metadata_join_rate=joined_interactions / total_interactions if total_interactions else 0.0,
        anchor_join_rate=joined_anchors / anchor_rows if anchor_rows else 0.0,
        eligible_user_count=len(eligible),
        rejection_counts=dict(sorted(rejections.items())),
    )
    return _DomainAuditResult(audit=audit, eligible_users=tuple(eligible))


def audit_benchmark_sources(
    source_root: Path,
    config: SourceAuditConfig | None = None,
    domains: tuple[str, ...] = BENCHMARK_DOMAIN_DIRECTORIES,
) -> SourceAuditResult:
    config = config or SourceAuditConfig()
    domain_results = [audit_domain_source(source_root, domain, config) for domain in domains]
    eligible = tuple(
        sorted(
            (record for result in domain_results for record in result.eligible_users),
            key=lambda record: (record.domain, record.pseudonymous_user_id, record.anchor_item_id),
        )
    )
    eligible_payload = [record.model_dump(mode="json") for record in eligible]
    eligible_sha256 = content_sha256(eligible_payload)
    audit_payload: dict[str, Any] = {
        "schema_version": BENCHMARK_SOURCE_AUDIT_VERSION,
        "config": config.model_dump(mode="json"),
        "domains": [result.audit.model_dump(mode="json") for result in domain_results],
        "total_users": sum(result.audit.user_row_count for result in domain_results),
        "total_eligible_users": len(eligible),
        "eligible_records_sha256": eligible_sha256,
    }
    audit = BenchmarkSourceAudit(
        audit_id=content_sha256(audit_payload),
        **audit_payload,
    )
    return SourceAuditResult(audit=audit, eligible_users=eligible)


def _write_text_idempotently(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different artifact file: {path}")
        return
    path.write_text(text, encoding="utf-8")


def write_source_audit_artifact(result: SourceAuditResult, output_root: Path) -> Path:
    """Write a deterministic content-addressed source audit and private user ledger."""

    run_dir = output_root / f"source-audit-{result.audit.audit_id[:16]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_path = run_dir / "source_audit.json"
    eligible_path = run_dir / "eligible_users.private.jsonl"
    audit_text = json.dumps(result.audit.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    eligible_text = "".join(
        canonical_json(record.model_dump(mode="json")) + "\n" for record in result.eligible_users
    )
    _write_text_idempotently(audit_path, audit_text)
    _write_text_idempotently(eligible_path, eligible_text)

    files = [
        ArtifactFileRecord(
            relative_path=audit_path.name,
            sha256=_file_sha256(audit_path),
            byte_size=audit_path.stat().st_size,
        ),
        ArtifactFileRecord(
            relative_path=eligible_path.name,
            sha256=_file_sha256(eligible_path),
            byte_size=eligible_path.stat().st_size,
            row_count=len(result.eligible_users),
        ),
    ]
    manifest = SourceAuditArtifactManifest(audit_id=result.audit.audit_id, files=files)
    manifest_path = run_dir / "manifest.json"
    manifest_text = json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    _write_text_idempotently(manifest_path, manifest_text)
    return run_dir
