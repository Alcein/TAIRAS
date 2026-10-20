"""Read legacy metadata without importing legacy runtime code or mutating data."""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from taira_mt.models import CatalogItem, StrictModel
from taira_mt.recommender.catalog import CatalogIndex


class LegacySourceManifest(StrictModel):
    domain: str
    relative_path: str
    sha256: str
    byte_size: int
    row_count: int
    columns: list[str]
    missing_counts: dict[str, int]


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_missing(value: str | None) -> bool:
    return value is None or not value.strip() or value.strip().casefold() in {"nan", "none", "null"}


def inspect_legacy_source(path: Path, domain: str, relative_to: Path | None = None) -> LegacySourceManifest:
    missing_counts: dict[str, int] = {}
    row_count = 0
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        missing_counts = {column: 0 for column in columns}
        for row in reader:
            row_count += 1
            for column in columns:
                if _is_missing(row.get(column)):
                    missing_counts[column] += 1
    relative_path = str(path.relative_to(relative_to)) if relative_to is not None else str(path)
    return LegacySourceManifest(
        domain=domain,
        relative_path=relative_path,
        sha256=_sha256(path),
        byte_size=path.stat().st_size,
        row_count=row_count,
        columns=columns,
        missing_counts=missing_counts,
    )


class LegacyCatalogAdapter:
    """Stream a domain catalog and preserve raw text alongside parsed scalars."""

    _number_pattern = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")

    def __init__(self, metadata_path: Path, domain: str) -> None:
        self.metadata_path = metadata_path
        self.domain = domain
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)

    @classmethod
    def parse_number(cls, raw_value: str | None) -> float | None:
        if _is_missing(raw_value):
            return None
        assert raw_value is not None
        match = cls._number_pattern.search(raw_value)
        if match is None:
            return None
        return float(match.group(0).replace(",", ""))

    def iter_items(self, candidate_ids: Iterable[str] | None = None) -> Iterator[CatalogItem]:
        requested = None if candidate_ids is None else set(candidate_ids)
        seen: set[str] = set()
        with self.metadata_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"id", "title", "category"}
            missing_columns = required.difference(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(f"metadata is missing required columns: {sorted(missing_columns)}")
            for row in reader:
                item_id = (row.get("id") or "").strip()
                if not item_id or (requested is not None and item_id not in requested):
                    continue
                if item_id in seen:
                    raise ValueError(f"duplicate item ID in legacy metadata: {item_id}")
                seen.add(item_id)
                attributes: dict[str, str | float] = {}
                for column in ("category", "description"):
                    value = row.get(column)
                    if not _is_missing(value):
                        attributes[column] = value.strip()
                for column in ("price", "ranking", "rating"):
                    raw_value = row.get(column)
                    parsed = self.parse_number(raw_value)
                    if parsed is not None:
                        attributes[column] = parsed
                    elif not _is_missing(raw_value):
                        attributes[f"{column}_raw"] = raw_value.strip()
                yield CatalogItem(
                    item_id=item_id,
                    title=(row.get("title") or "").strip(),
                    attributes=attributes,
                )
        if requested is not None:
            missing_ids = requested.difference(seen)
            if missing_ids:
                preview = sorted(missing_ids)[:5]
                raise KeyError(f"{len(missing_ids)} requested items missing from metadata; first IDs: {preview}")

    def load_index(self, candidate_ids: Iterable[str] | None = None) -> CatalogIndex:
        return CatalogIndex(self.iter_items(candidate_ids), domain=self.domain)
