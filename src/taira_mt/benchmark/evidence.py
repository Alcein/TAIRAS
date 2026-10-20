"""Deterministic, auditable evidence profiles derived from Amazon item sequences."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from taira_mt.benchmark.source import (
    BenchmarkSourceAudit,
    EligibleUserSource,
    SourceAuditConfig,
    audit_benchmark_sources,
)
from taira_mt.data.goal_construction import NormalizedCatalogRow, load_normalized_catalog
from taira_mt.data.numeric_goal_specs import POPULARITY_MAPPING_VERSION, popularity_percentiles
from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json, content_sha256

SEQUENCE_EVIDENCE_VERSION = "sequence_evidence_v1"
SEQUENCE_EVIDENCE_INTERPRETATION = (
    "descriptive_history_evidence_not_automatically_a_hard_user_preference"
)


class EvidenceStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class SequenceEvidenceConfig(StrictModel):
    schema_version: str = SEQUENCE_EVIDENCE_VERSION
    minimum_category_support: int = Field(default=2, ge=1)
    maximum_category_evidence: int = Field(default=12, ge=1)
    minimum_numeric_observations: int = Field(default=2, ge=1)
    moderate_category_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    strong_category_fraction: float = Field(default=0.50, gt=0.0, le=1.0)
    moderate_numeric_coverage: float = Field(default=0.50, gt=0.0, le=1.0)
    strong_numeric_coverage: float = Field(default=0.75, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_config(self) -> SequenceEvidenceConfig:
        if self.schema_version != SEQUENCE_EVIDENCE_VERSION:
            raise ValueError(f"unsupported sequence evidence version: {self.schema_version}")
        if self.moderate_category_fraction > self.strong_category_fraction:
            raise ValueError("moderate category threshold cannot exceed strong threshold")
        if self.moderate_numeric_coverage > self.strong_numeric_coverage:
            raise ValueError("moderate numeric threshold cannot exceed strong threshold")
        return self


class HistoryCategoryEvidence(StrictModel):
    tag: str = Field(min_length=1)
    support_count: int = Field(ge=1)
    history_item_count: int = Field(ge=1)
    support_fraction: float = Field(gt=0.0, le=1.0)
    also_on_anchor: bool
    strength: EvidenceStrength
    supporting_item_ids_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_support(self) -> HistoryCategoryEvidence:
        if self.support_count > self.history_item_count:
            raise ValueError("category support exceeds joined history size")
        expected = self.support_count / self.history_item_count
        if abs(expected - self.support_fraction) > 1e-9:
            raise ValueError("category support fraction is inconsistent")
        return self


class HistoryNumericEvidence(StrictModel):
    slot: str = Field(min_length=1)
    observed_count: int = Field(ge=1)
    history_item_count: int = Field(ge=1)
    coverage: float = Field(gt=0.0, le=1.0)
    minimum: float
    lower_quartile: float
    median: float
    upper_quartile: float
    maximum: float
    strength: EvidenceStrength
    supporting_item_ids_sha256: str = Field(min_length=64, max_length=64)
    transformation_version: str | None = None

    @model_validator(mode="after")
    def validate_numeric_summary(self) -> HistoryNumericEvidence:
        if self.observed_count > self.history_item_count:
            raise ValueError("numeric observations exceed joined history size")
        expected = self.observed_count / self.history_item_count
        if abs(expected - self.coverage) > 1e-9:
            raise ValueError("numeric coverage is inconsistent")
        ordered = [
            self.minimum,
            self.lower_quartile,
            self.median,
            self.upper_quartile,
            self.maximum,
        ]
        if ordered != sorted(ordered):
            raise ValueError("numeric summary must be monotonic")
        if self.slot == "popularity_percentile":
            if self.transformation_version != POPULARITY_MAPPING_VERSION:
                raise ValueError("popularity evidence requires its transformation version")
            if self.minimum < 0.0 or self.maximum > 1.0:
                raise ValueError("popularity percentile must lie in [0, 1]")
        elif self.transformation_version is not None:
            raise ValueError("only transformed slots may set transformation_version")
        return self


class SequenceEvidenceProfile(StrictModel):
    schema_version: str = SEQUENCE_EVIDENCE_VERSION
    profile_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    pseudonymous_user_id: str = Field(min_length=1)
    source_sequence_sha256: str = Field(min_length=64, max_length=64)
    anchor_item_id: str = Field(min_length=1)
    anchor_category_tags: list[str]
    joined_history_item_count: int = Field(ge=1)
    missing_history_item_count: int = Field(ge=0)
    category_evidence: list[HistoryCategoryEvidence]
    numeric_evidence: list[HistoryNumericEvidence]
    anchor_reinforced_category_tags: list[str]
    interpretation: str = SEQUENCE_EVIDENCE_INTERPRETATION

    @field_validator("anchor_category_tags", "anchor_reinforced_category_tags")
    @classmethod
    def validate_unique_tags(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("profile tag lists must be sorted and unique")
        return value

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"profile_id"})

    @model_validator(mode="after")
    def validate_profile(self) -> SequenceEvidenceProfile:
        if self.schema_version != SEQUENCE_EVIDENCE_VERSION:
            raise ValueError(f"unsupported sequence evidence version: {self.schema_version}")
        if self.interpretation != SEQUENCE_EVIDENCE_INTERPRETATION:
            raise ValueError("sequence evidence interpretation cannot be changed")
        category_tags = [item.tag for item in self.category_evidence]
        if len(category_tags) != len(set(category_tags)):
            raise ValueError("category evidence cannot repeat a tag")
        numeric_slots = [item.slot for item in self.numeric_evidence]
        if len(numeric_slots) != len(set(numeric_slots)):
            raise ValueError("numeric evidence cannot repeat a slot")
        expected_reinforced = sorted(
            set(self.anchor_category_tags).intersection(category_tags)
        )
        if self.anchor_reinforced_category_tags != expected_reinforced:
            raise ValueError("anchor-reinforced category tags are inconsistent")
        expected_id = f"evidence-{content_sha256(self.identity_payload())[:24]}"
        if self.profile_id != expected_id:
            raise ValueError("sequence evidence profile ID does not match its content")
        return self


class SequenceEvidenceSummary(StrictModel):
    build_id: str
    profile_count: int = Field(ge=0)
    domain_profile_counts: dict[str, int]
    profiles_with_recurrent_category: int = Field(ge=0)
    profiles_with_anchor_reinforcement: int = Field(ge=0)
    profiles_with_moderate_or_strong_anchor_reinforcement: int = Field(ge=0)
    numeric_profile_counts: dict[str, int]
    numeric_moderate_or_strong_counts: dict[str, int]


class SequenceEvidenceArtifactFile(StrictModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    byte_size: int = Field(ge=0)
    row_count: int | None = Field(default=None, ge=0)


class SequenceEvidenceManifest(StrictModel):
    schema_version: str = SEQUENCE_EVIDENCE_VERSION
    artifact_role: str = "private_sequence_evidence_profiles"
    build_id: str
    source_audit_id: str = Field(min_length=64, max_length=64)
    config: SequenceEvidenceConfig
    profiles_sha256: str = Field(min_length=64, max_length=64)
    files: list[SequenceEvidenceArtifactFile]


@dataclass(frozen=True)
class SequenceEvidenceBuildResult:
    build_id: str
    source_audit: BenchmarkSourceAudit
    config: SequenceEvidenceConfig
    profiles: tuple[SequenceEvidenceProfile, ...]
    summary: SequenceEvidenceSummary


def load_sequence_evidence_artifact(
    run_dir: Path,
) -> tuple[SequenceEvidenceManifest, tuple[SequenceEvidenceProfile, ...]]:
    """Fail closed when loading a private sequence-evidence artifact."""

    run_dir = run_dir.resolve()
    manifest = SequenceEvidenceManifest.model_validate(
        json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    )
    for item in manifest.files:
        path = (run_dir / item.relative_path).resolve()
        if run_dir not in path.parents:
            raise ValueError("sequence evidence manifest path escapes its artifact")
        if not path.is_file() or _file_sha256(path) != item.sha256:
            raise ValueError(f"sequence evidence file hash mismatch: {item.relative_path}")
        if path.stat().st_size != item.byte_size:
            raise ValueError(f"sequence evidence file size mismatch: {item.relative_path}")
    profile_path = run_dir / "profiles.private.jsonl"
    raw_profiles = [
        json.loads(line)
        for line in profile_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if content_sha256(raw_profiles) != manifest.profiles_sha256:
        raise ValueError("sequence evidence profile collection differs from manifest")
    profiles = tuple(SequenceEvidenceProfile.model_validate(item) for item in raw_profiles)
    if len({item.profile_id for item in profiles}) != len(profiles):
        raise ValueError("sequence evidence artifact contains duplicate profile IDs")
    if len({item.pseudonymous_user_id for item in profiles}) != len(profiles):
        raise ValueError("sequence evidence artifact contains duplicate users")
    return manifest, profiles


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires observations")
    if len(ordered) == 1:
        return round(ordered[0], 8)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction, 8)


def _category_strength(
    support_count: int,
    support_fraction: float,
    config: SequenceEvidenceConfig,
) -> EvidenceStrength:
    if support_count >= 3 and support_fraction >= config.strong_category_fraction:
        return EvidenceStrength.STRONG
    if support_count >= 2 and support_fraction >= config.moderate_category_fraction:
        return EvidenceStrength.MODERATE
    return EvidenceStrength.WEAK


def _numeric_strength(
    observed_count: int,
    coverage: float,
    config: SequenceEvidenceConfig,
) -> EvidenceStrength:
    if observed_count >= 4 and coverage >= config.strong_numeric_coverage:
        return EvidenceStrength.STRONG
    if observed_count >= 3 and coverage >= config.moderate_numeric_coverage:
        return EvidenceStrength.MODERATE
    return EvidenceStrength.WEAK


def build_sequence_evidence_profile(
    source: EligibleUserSource,
    row_by_id: dict[str, NormalizedCatalogRow],
    popularity: dict[str, float],
    config: SequenceEvidenceConfig | None = None,
) -> SequenceEvidenceProfile:
    """Build one descriptive evidence profile without interpreting it as hard intent."""

    config = config or SequenceEvidenceConfig()
    try:
        anchor = row_by_id[source.anchor_item_id]
        history = [row_by_id[item_id] for item_id in source.joined_history_item_ids]
    except KeyError as exc:
        raise ValueError(f"source audit references missing normalized item: {exc.args[0]}") from exc
    history_count = len(history)

    tag_items: dict[str, list[str]] = defaultdict(list)
    for row in history:
        for tag in set(row.category_tags):
            tag_items[tag].append(row.item_id)
    category_evidence = [
        HistoryCategoryEvidence(
            tag=tag,
            support_count=len(item_ids),
            history_item_count=history_count,
            support_fraction=len(item_ids) / history_count,
            also_on_anchor=tag in set(anchor.category_tags),
            strength=_category_strength(len(item_ids), len(item_ids) / history_count, config),
            supporting_item_ids_sha256=content_sha256(sorted(item_ids)),
        )
        for tag, item_ids in tag_items.items()
        if len(item_ids) >= config.minimum_category_support
    ]
    category_evidence.sort(key=lambda item: (-item.support_count, item.tag))
    category_evidence = category_evidence[: config.maximum_category_evidence]

    numeric_sources: dict[str, list[tuple[str, float]]] = {
        "price": [
            (row.item_id, row.price)
            for row in history
            if row.price is not None
        ],
        "rating": [
            (row.item_id, row.rating)
            for row in history
            if row.rating is not None and row.rating <= 5.0
        ],
        "popularity_percentile": [
            (row.item_id, popularity[row.item_id])
            for row in history
            if row.item_id in popularity
        ],
    }
    numeric_evidence: list[HistoryNumericEvidence] = []
    for slot in ("price", "rating", "popularity_percentile"):
        observations = numeric_sources[slot]
        if len(observations) < config.minimum_numeric_observations:
            continue
        values = [value for _, value in observations]
        coverage = len(values) / history_count
        numeric_evidence.append(
            HistoryNumericEvidence(
                slot=slot,
                observed_count=len(values),
                history_item_count=history_count,
                coverage=coverage,
                minimum=_quantile(values, 0.0),
                lower_quartile=_quantile(values, 0.25),
                median=_quantile(values, 0.5),
                upper_quartile=_quantile(values, 0.75),
                maximum=_quantile(values, 1.0),
                strength=_numeric_strength(len(values), coverage, config),
                supporting_item_ids_sha256=content_sha256(
                    sorted(item_id for item_id, _ in observations)
                ),
                transformation_version=(
                    POPULARITY_MAPPING_VERSION
                    if slot == "popularity_percentile"
                    else None
                ),
            )
        )

    payload: dict[str, Any] = {
        "schema_version": SEQUENCE_EVIDENCE_VERSION,
        "domain": source.domain,
        "pseudonymous_user_id": source.pseudonymous_user_id,
        "source_sequence_sha256": source.source_sequence_sha256,
        "anchor_item_id": source.anchor_item_id,
        "anchor_category_tags": sorted(set(anchor.category_tags)),
        "joined_history_item_count": history_count,
        "missing_history_item_count": len(source.missing_history_item_ids),
        "category_evidence": [item.model_dump(mode="json") for item in category_evidence],
        "numeric_evidence": [item.model_dump(mode="json") for item in numeric_evidence],
        "anchor_reinforced_category_tags": sorted(
            set(anchor.category_tags).intersection(item.tag for item in category_evidence)
        ),
        "interpretation": SEQUENCE_EVIDENCE_INTERPRETATION,
    }
    return SequenceEvidenceProfile(
        profile_id=f"evidence-{content_sha256(payload)[:24]}",
        **payload,
    )


def build_sequence_evidence_profiles(
    source_root: Path,
    config: SequenceEvidenceConfig | None = None,
) -> SequenceEvidenceBuildResult:
    config = config or SequenceEvidenceConfig()
    source_root = source_root.resolve()
    source_result = audit_benchmark_sources(source_root, SourceAuditConfig())
    by_domain: dict[str, list[EligibleUserSource]] = defaultdict(list)
    for source in source_result.eligible_users:
        by_domain[source.domain].append(source)

    profiles: list[SequenceEvidenceProfile] = []
    for domain in sorted(by_domain):
        rows, _ = load_normalized_catalog(source_root / domain / "metadata.csv")
        row_by_id = {row.item_id: row for row in rows}
        popularity = popularity_percentiles(rows)
        profiles.extend(
            build_sequence_evidence_profile(source, row_by_id, popularity, config)
            for source in by_domain[domain]
        )
    profiles.sort(key=lambda item: (item.domain, item.pseudonymous_user_id))
    profile_payload = [item.model_dump(mode="json") for item in profiles]
    identity = {
        "schema_version": SEQUENCE_EVIDENCE_VERSION,
        "source_audit_id": source_result.audit.audit_id,
        "config": config.model_dump(mode="json"),
        "profiles_sha256": content_sha256(profile_payload),
    }
    build_id = f"sequence-evidence-{content_sha256(identity)[:16]}"

    numeric_counts: Counter[str] = Counter()
    numeric_reliable: Counter[str] = Counter()
    for profile in profiles:
        for item in profile.numeric_evidence:
            numeric_counts[item.slot] += 1
            if item.strength in {EvidenceStrength.MODERATE, EvidenceStrength.STRONG}:
                numeric_reliable[item.slot] += 1
    summary = SequenceEvidenceSummary(
        build_id=build_id,
        profile_count=len(profiles),
        domain_profile_counts=dict(sorted(Counter(item.domain for item in profiles).items())),
        profiles_with_recurrent_category=sum(bool(item.category_evidence) for item in profiles),
        profiles_with_anchor_reinforcement=sum(
            bool(item.anchor_reinforced_category_tags) for item in profiles
        ),
        profiles_with_moderate_or_strong_anchor_reinforcement=sum(
            any(
                category.also_on_anchor
                and category.strength in {EvidenceStrength.MODERATE, EvidenceStrength.STRONG}
                for category in item.category_evidence
            )
            for item in profiles
        ),
        numeric_profile_counts=dict(sorted(numeric_counts.items())),
        numeric_moderate_or_strong_counts=dict(sorted(numeric_reliable.items())),
    )
    return SequenceEvidenceBuildResult(
        build_id=build_id,
        source_audit=source_result.audit,
        config=config,
        profiles=tuple(profiles),
        summary=summary,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_idempotently(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different artifact file: {path}")
        return
    path.write_text(text, encoding="utf-8")


def write_sequence_evidence_artifact(
    result: SequenceEvidenceBuildResult,
    output_root: Path,
) -> Path:
    run_dir = output_root / result.build_id
    run_dir.mkdir(parents=True, exist_ok=True)
    profiles = [item.model_dump(mode="json") for item in result.profiles]
    outputs = {
        "profiles.private.jsonl": "".join(canonical_json(item) + "\n" for item in profiles),
        "summary.json": json.dumps(
            result.summary.model_dump(mode="json"), ensure_ascii=False, indent=2
        )
        + "\n",
        "source_audit.json": json.dumps(
            result.source_audit.model_dump(mode="json"), ensure_ascii=False, indent=2
        )
        + "\n",
    }
    for filename, text in outputs.items():
        _write_idempotently(run_dir / filename, text)
    files = [
        SequenceEvidenceArtifactFile(
            relative_path=filename,
            sha256=_file_sha256(run_dir / filename),
            byte_size=(run_dir / filename).stat().st_size,
            row_count=(text.count("\n") if filename.endswith(".jsonl") else None),
        )
        for filename, text in sorted(outputs.items())
    ]
    manifest = SequenceEvidenceManifest(
        build_id=result.build_id,
        source_audit_id=result.source_audit.audit_id,
        config=result.config,
        profiles_sha256=content_sha256(profiles),
        files=files,
    )
    _write_idempotently(
        run_dir / "manifest.json",
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
    )
    return run_dir
