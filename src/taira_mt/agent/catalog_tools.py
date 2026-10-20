"""A public catalog-tool envelope shared by all interactive recommendation agents.

The benchmark evaluator owns latent goals and acceptance labels.  This module
exposes only the candidate-pool items and their observable attributes, so a
state tracker, a direct LLM baseline, and TAIRAS use the same retrieval
surface.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field, field_validator

from taira_mt.constraints.predicates import predicate_evidence_status
from taira_mt.dsus.schemas import PublicConstraintObservation
from taira_mt.models import (
    AttributeValue,
    ConstraintEvidenceStatus,
    ConstraintImportance,
    StrictModel,
)
from taira_mt.recommender.catalog import CandidatePool


class PublicCatalogItem(StrictModel):
    """A public copy of one candidate item; no benchmark-case metadata is attached."""

    item_id: str
    title: str
    attributes: dict[str, AttributeValue]


class PublicCatalogSearchResult(StrictModel):
    """Catalog evidence partitioned without an oracle utility or hidden target."""

    pool_id: str = Field(min_length=64, max_length=64)
    verified_item_ids: list[str] = Field(default_factory=list)
    unknown_evidence_item_ids: list[str] = Field(default_factory=list)
    excluded_item_ids: list[str] = Field(default_factory=list)

    @field_validator(
        "verified_item_ids",
        "unknown_evidence_item_ids",
        "excluded_item_ids",
    )
    @classmethod
    def unique_item_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("public catalog result item IDs must be unique")
        return value

    def model_inputs(self) -> dict[str, object]:
        """The Agent-facing retrieval summary, intentionally excluding ranking internals."""

        return self.model_dump(mode="json")


class PublicCatalogTool:
    """Deterministic candidate retrieval using only public predicates and attributes."""

    def __init__(self, candidate_pool: CandidatePool) -> None:
        self._candidate_pool = candidate_pool

    @property
    def pool_id(self) -> str:
        return self._candidate_pool.pool_id

    @property
    def domain(self) -> str:
        return self._candidate_pool.index.domain

    @property
    def item_ids(self) -> tuple[str, ...]:
        return self._candidate_pool.item_ids

    @property
    def attribute_slots(self) -> tuple[str, ...]:
        """Observed catalog schema available to a text interpreter or Agent prompt."""

        return tuple(
            sorted(
                {
                    slot
                    for item_id in self._candidate_pool.item_ids
                    for slot in self._candidate_pool.index.get(item_id).attributes
                }
            )
        )

    def public_lexical_value_lexicon(
        self,
        *,
        max_values_per_slot: int = 16,
        max_total_values: int = 64,
        max_value_characters: int = 80,
    ) -> dict[str, list[str]]:
        """Return a compact, deterministic lexical map derived from public attributes.

        A language parser needs category/brand words to map an utterance to a
        catalog slot, but it does not need item IDs, titles, ranks, or an
        evaluator-owned target.  This view therefore retains only non-empty
        string attribute values, deduplicated case-insensitively and sorted
        independently of candidate-pool traversal order.  Numeric attributes
        remain represented by their public slot names and natural-language
        predicate evidence rather than a large enumeration of values.
        """

        if max_values_per_slot < 1:
            raise ValueError("max_values_per_slot must be positive")
        if max_total_values < 1:
            raise ValueError("max_total_values must be positive")
        if max_value_characters < 1:
            raise ValueError("max_value_characters must be positive")

        raw_values_by_slot: dict[str, dict[str, str]] = {}
        for item in self.public_items():
            for slot, raw_value in item.attributes.items():
                values = raw_value if isinstance(raw_value, list) else [raw_value]
                for value in values:
                    if not isinstance(value, str):
                        continue
                    normalized = " ".join(value.split())
                    if not normalized or len(normalized) > max_value_characters:
                        continue
                    values_by_normalized_form = raw_values_by_slot.setdefault(slot, {})
                    folded = normalized.casefold()
                    existing = values_by_normalized_form.get(folded)
                    if existing is None or normalized < existing:
                        values_by_normalized_form[folded] = normalized

        preferred_slots = {
            "specific_category": 0,
            "broad_category": 1,
            "category_tags": 2,
        }
        ordered_slots = sorted(
            raw_values_by_slot,
            key=lambda slot: (preferred_slots.get(slot, len(preferred_slots)), slot),
        )
        remaining = max_total_values
        lexicon: dict[str, list[str]] = {}
        for slot in ordered_slots:
            if remaining <= 0:
                break
            values = sorted(
                raw_values_by_slot[slot].values(),
                key=lambda value: (value.casefold(), value),
            )
            selected = values[: min(max_values_per_slot, remaining)]
            if selected:
                lexicon[slot] = selected
                remaining -= len(selected)
        return lexicon

    def public_items(self) -> tuple[PublicCatalogItem, ...]:
        """Return the candidate pool as public item records in its fixed order."""

        return tuple(self.inspect(item_id) for item_id in self._candidate_pool.item_ids)

    def inspect(self, item_id: str) -> PublicCatalogItem:
        """Return a deep public representation of a candidate item."""

        if item_id not in set(self._candidate_pool.item_ids):
            raise KeyError(f"item does not belong to this public candidate pool: {item_id}")
        item = self._candidate_pool.index.get(item_id)
        return PublicCatalogItem(
            item_id=item.item_id,
            title=item.title,
            attributes=item.attributes,
        )

    @staticmethod
    def _public_score(item: PublicCatalogItem) -> tuple[float, str]:
        """Use observable quality fields only for deterministic tie breaking."""

        rating = item.attributes.get("rating")
        popularity = item.attributes.get("popularity_percentile")
        price = item.attributes.get("price")
        score = (
            float(rating) if isinstance(rating, (int, float)) and not isinstance(rating, bool) else 0.0
        )
        score += (
            float(popularity)
            if isinstance(popularity, (int, float)) and not isinstance(popularity, bool)
            else 0.0
        )
        score -= (
            0.001 * float(price)
            if isinstance(price, (int, float)) and not isinstance(price, bool)
            else 0.0
        )
        return score, item.item_id

    def search(
        self,
        constraints: Iterable[PublicConstraintObservation],
        *,
        excluded_item_ids: Iterable[str] = (),
    ) -> PublicCatalogSearchResult:
        """Partition remaining items into fully verified and UNKNOWN-evidence candidates.

        Hard public constraints filter items.  Soft constraints do not silently
        eliminate candidates; they remain available for a later learned/value
        ranking stage.  Any item with a missing hard attribute is explicit in
        ``unknown_evidence_item_ids`` and may not be automatically recommended.
        """

        materialized_constraints = tuple(constraints)
        hard_constraints = tuple(
            item
            for item in materialized_constraints
            if item.predicate.importance == ConstraintImportance.HARD
        )
        excluded = tuple(sorted(set(excluded_item_ids)))
        unknown_excluded = set(excluded).difference(self._candidate_pool.item_ids)
        if unknown_excluded:
            raise KeyError(f"excluded IDs are outside candidate pool: {sorted(unknown_excluded)}")
        verified: list[tuple[float, str]] = []
        unknown: list[tuple[float, str]] = []
        for raw_item in self._candidate_pool.remaining_items(set(excluded)):
            public_item = PublicCatalogItem(
                item_id=raw_item.item_id,
                title=raw_item.title,
                attributes=raw_item.attributes,
            )
            statuses = [
                predicate_evidence_status(
                    public_item.attributes.get(constraint.slot),
                    constraint.predicate,
                )
                for constraint in hard_constraints
            ]
            if ConstraintEvidenceStatus.VIOLATED in statuses:
                continue
            destination = (
                unknown if ConstraintEvidenceStatus.UNKNOWN in statuses else verified
            )
            destination.append(self._public_score(public_item))

        def ordered(values: list[tuple[float, str]]) -> list[str]:
            return [item_id for _, item_id in sorted(values, key=lambda item: (-item[0], item[1]))]

        return PublicCatalogSearchResult(
            pool_id=self.pool_id,
            verified_item_ids=ordered(verified),
            unknown_evidence_item_ids=ordered(unknown),
            excluded_item_ids=list(excluded),
        )
