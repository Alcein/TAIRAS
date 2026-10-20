"""Shared immutable catalog index and content-addressed candidate pools."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from taira_mt.models import CatalogItem
from taira_mt.provenance import canonical_json


class CatalogIndex:
    def __init__(
        self,
        items: Iterable[CatalogItem],
        *,
        domain: str = "in_memory",
        source_sha256: str | None = None,
    ) -> None:
        materialized = tuple(items)
        if not materialized:
            raise ValueError("catalog cannot be empty")
        item_ids = [item.item_id for item in materialized]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("catalog item IDs must be unique")
        self._items = materialized
        self._by_id = {item.item_id: item for item in materialized}
        self.domain = domain
        if source_sha256 is None:
            source_payload = [item.model_dump(mode="json") for item in materialized]
            source_sha256 = hashlib.sha256(canonical_json(source_payload).encode("utf-8")).hexdigest()
        self.source_sha256 = source_sha256

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(item.item_id for item in self._items)

    def get(self, item_id: str) -> CatalogItem:
        try:
            return self._by_id[item_id]
        except KeyError as exc:
            raise KeyError(f"unknown catalog item: {item_id}") from exc

    def __contains__(self, item_id: object) -> bool:
        return item_id in self._by_id

    def pool(
        self,
        candidate_ids: Iterable[str] | None = None,
        *,
        recipe_id: str = "explicit_order_v1",
        recipe_config: dict[str, Any] | None = None,
    ) -> CandidatePool:
        return CandidatePool(
            self,
            self.item_ids if candidate_ids is None else candidate_ids,
            recipe_id=recipe_id,
            recipe_config=recipe_config,
        )


class CandidatePool:
    def __init__(
        self,
        index: CatalogIndex,
        candidate_ids: Iterable[str],
        *,
        recipe_id: str = "explicit_order_v1",
        recipe_config: dict[str, Any] | None = None,
    ) -> None:
        ids = tuple(candidate_ids)
        if not ids:
            raise ValueError("candidate pool cannot be empty")
        if len(ids) != len(set(ids)):
            raise ValueError("candidate pool IDs must be unique")
        for item_id in ids:
            index.get(item_id)
        self.index = index
        self.item_ids = ids
        self.recipe_id = recipe_id
        self.recipe_config = dict(recipe_config or {})
        self.identity_payload = {
            "domain": index.domain,
            "catalog_source_sha256": index.source_sha256,
            "retrieval_recipe_id": self.recipe_id,
            "retrieval_recipe_config": self.recipe_config,
            "ordered_item_ids": list(ids),
        }
        self.pool_id = hashlib.sha256(canonical_json(self.identity_payload).encode("utf-8")).hexdigest()

    def remaining_ids(self, excluded_item_ids: set[str] | None = None) -> tuple[str, ...]:
        excluded = excluded_item_ids or set()
        return tuple(item_id for item_id in self.item_ids if item_id not in excluded)

    def remaining_items(self, excluded_item_ids: set[str] | None = None) -> tuple[CatalogItem, ...]:
        return tuple(self.index.get(item_id) for item_id in self.remaining_ids(excluded_item_ids))

    def remaining_count(self, excluded_item_ids: set[str] | None = None) -> int:
        return len(self.remaining_ids(excluded_item_ids))
