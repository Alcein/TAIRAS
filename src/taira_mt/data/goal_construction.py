"""Catalog-statistical discovery for target-free multi-turn goals."""

from __future__ import annotations

import csv
import html
import itertools
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from taira_mt.data.legacy_catalog import _sha256
from taira_mt.models import StrictModel
from taira_mt.provenance import content_sha256

DOMAIN_HIDDEN_TAG_DENYLIST = {
    "amazon_beauty": {"beauty", "skin care"},
    "amazon_clothing": {"clothing"},
    "amazon_music": {"music"},
}


class RelationDiscoveryConfig(StrictModel):
    schema_version: str = "relation_discovery_v1"
    minimum_visible_count: int = Field(default=200, ge=1)
    maximum_visible_fraction: float = Field(default=0.50, gt=0.0, le=1.0)
    minimum_hidden_count: int = Field(default=30, ge=1)
    maximum_hidden_fraction: float = Field(default=0.20, gt=0.0, le=1.0)
    minimum_pair_count: int = Field(default=30, ge=1)
    minimum_visible_given_hidden: float = Field(default=0.70, ge=0.0, le=1.0)
    minimum_hidden_given_visible: float = Field(default=0.06, ge=0.0, le=1.0)
    maximum_hidden_given_visible: float = Field(default=0.40, ge=0.0, le=1.0)
    maximum_token_jaccard: float = Field(default=0.80, ge=0.0, le=1.0)


class CatalogCoverage(StrictModel):
    domain: str
    row_count: int
    usable_category_count: int
    unique_tag_count: int
    price_count: int
    rating_count: int
    ranking_count: int


class CategoryTagStatistic(StrictModel):
    domain: str
    tag: str
    item_count: int
    item_fraction: float
    eligible_visible: bool
    eligible_hidden: bool


class CategoryRelation(StrictModel):
    relation_id: str
    domain: str
    visible_tag: str
    hidden_tag: str
    visible_count: int
    hidden_count: int
    pair_count: int
    visible_given_hidden: float
    hidden_given_visible: float
    token_jaccard: float
    human_review_status: str = "pending"


class RelationDiscoveryResult(StrictModel):
    domain: str
    metadata_relative_path: str
    metadata_sha256: str
    config_hash: str
    coverage: CatalogCoverage
    tag_statistics: list[CategoryTagStatistic]
    relations: list[CategoryRelation]
    rejection_counts: dict[str, int]


@dataclass(frozen=True)
class NormalizedCatalogRow:
    item_id: str
    title: str
    category_tags: tuple[str, ...]
    price: float | None
    rating: float | None
    ranking: float | None


_strict_number_pattern = re.compile(r"^\s*\$?\s*(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)\s*$")


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", html.unescape(value))
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_category_tag(value: str) -> str:
    return normalize_text(value).casefold()


def parse_category_tags(value: str) -> tuple[str, ...]:
    normalized = {normalize_category_tag(part) for part in value.split("|")}
    normalized.discard("")
    return tuple(sorted(normalized))


def parse_strict_positive_number(value: str | None) -> float | None:
    if value is None:
        return None
    match = _strict_number_pattern.fullmatch(value)
    if match is None:
        return None
    number = float(match.group("number").replace(",", ""))
    return number if math.isfinite(number) and number > 0.0 else None


def _valid_tag(tag: str) -> bool:
    if len(tag) < 3 or tag.isnumeric():
        return False
    if any(unicodedata.category(character).startswith("C") for character in tag):
        return False
    return any(character.isalpha() for character in tag)


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"\w+", left))
    right_tokens = set(re.findall(r"\w+", right))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def load_normalized_catalog(path: Path) -> tuple[list[NormalizedCatalogRow], int]:
    rows: list[NormalizedCatalogRow] = []
    raw_row_count = 0
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "title", "category"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"metadata is missing required columns: {sorted(missing)}")
        for raw in reader:
            raw_row_count += 1
            item_id = (raw.get("id") or "").strip()
            if not item_id:
                continue
            if item_id in seen:
                raise ValueError(f"duplicate catalog ID: {item_id}")
            seen.add(item_id)
            tags = tuple(tag for tag in parse_category_tags(raw.get("category") or "") if _valid_tag(tag))
            rows.append(
                NormalizedCatalogRow(
                    item_id=item_id,
                    title=normalize_text(raw.get("title") or ""),
                    category_tags=tags,
                    price=parse_strict_positive_number(raw.get("price")),
                    rating=parse_strict_positive_number(raw.get("rating")),
                    ranking=parse_strict_positive_number(raw.get("ranking")),
                )
            )
    return rows, raw_row_count


def discover_category_relations(
    rows: list[NormalizedCatalogRow],
    domain: str,
    metadata_path: Path,
    relative_to: Path,
    config: RelationDiscoveryConfig | None = None,
) -> RelationDiscoveryResult:
    config = config or RelationDiscoveryConfig()
    domain_size = len(rows)
    if domain_size == 0:
        raise ValueError("normalized catalog cannot be empty")
    tag_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        tags = tuple(sorted(set(row.category_tags)))
        tag_counts.update(tags)
        pair_counts.update(itertools.combinations(tags, 2))

    denylist = DOMAIN_HIDDEN_TAG_DENYLIST.get(domain, set())
    tag_statistics: list[CategoryTagStatistic] = []
    visible_eligible: set[str] = set()
    hidden_eligible: set[str] = set()
    for tag, count in sorted(tag_counts.items()):
        fraction = count / domain_size
        visible = count >= config.minimum_visible_count and fraction <= config.maximum_visible_fraction
        hidden = (
            count >= config.minimum_hidden_count
            and fraction <= config.maximum_hidden_fraction
            and tag not in denylist
        )
        if visible:
            visible_eligible.add(tag)
        if hidden:
            hidden_eligible.add(tag)
        tag_statistics.append(
            CategoryTagStatistic(
                domain=domain,
                tag=tag,
                item_count=count,
                item_fraction=fraction,
                eligible_visible=visible,
                eligible_hidden=hidden,
            )
        )

    rejection_counts: Counter[str] = Counter()
    relations: list[CategoryRelation] = []
    for unordered_pair, pair_count in sorted(pair_counts.items()):
        for visible_tag, hidden_tag in (unordered_pair, tuple(reversed(unordered_pair))):
            if visible_tag not in visible_eligible:
                rejection_counts["VISIBLE_TAG_INELIGIBLE"] += 1
                continue
            if hidden_tag not in hidden_eligible:
                rejection_counts["HIDDEN_TAG_INELIGIBLE"] += 1
                continue
            if pair_count < config.minimum_pair_count:
                rejection_counts["PAIR_SUPPORT_TOO_LOW"] += 1
                continue
            visible_count = tag_counts[visible_tag]
            hidden_count = tag_counts[hidden_tag]
            visible_given_hidden = pair_count / hidden_count
            hidden_given_visible = pair_count / visible_count
            if visible_given_hidden < config.minimum_visible_given_hidden:
                rejection_counts["PAIR_NOT_DIRECTIONAL"] += 1
                continue
            if not config.minimum_hidden_given_visible <= hidden_given_visible <= config.maximum_hidden_given_visible:
                rejection_counts["PAIR_PREVALENCE_OUT_OF_RANGE"] += 1
                continue
            token_jaccard = _token_jaccard(visible_tag, hidden_tag)
            if token_jaccard >= config.maximum_token_jaccard:
                rejection_counts["PAIR_TOKEN_OVERLAP"] += 1
                continue
            signature = {
                "domain": domain,
                "visible_tag": visible_tag,
                "hidden_tag": hidden_tag,
                "config_hash": content_sha256(config.model_dump(mode="json")),
            }
            relations.append(
                CategoryRelation(
                    relation_id=content_sha256(signature)[:24],
                    domain=domain,
                    visible_tag=visible_tag,
                    hidden_tag=hidden_tag,
                    visible_count=visible_count,
                    hidden_count=hidden_count,
                    pair_count=pair_count,
                    visible_given_hidden=visible_given_hidden,
                    hidden_given_visible=hidden_given_visible,
                    token_jaccard=token_jaccard,
                )
            )

    coverage = CatalogCoverage(
        domain=domain,
        row_count=domain_size,
        usable_category_count=sum(bool(row.category_tags) for row in rows),
        unique_tag_count=len(tag_counts),
        price_count=sum(row.price is not None for row in rows),
        rating_count=sum(row.rating is not None and row.rating <= 5.0 for row in rows),
        ranking_count=sum(row.ranking is not None for row in rows),
    )
    return RelationDiscoveryResult(
        domain=domain,
        metadata_relative_path=str(metadata_path.relative_to(relative_to)),
        metadata_sha256=_sha256(metadata_path),
        config_hash=content_sha256(config.model_dump(mode="json")),
        coverage=coverage,
        tag_statistics=tag_statistics,
        relations=relations,
        rejection_counts=dict(sorted(rejection_counts.items())),
    )
