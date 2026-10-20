"""Strict, auditable allowlist for publishable category relations."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from taira_mt.data.goal_construction import CategoryRelation
from taira_mt.models import StrictModel
from taira_mt.provenance import content_sha256

RELATION_ID_PATTERN = r"^[0-9a-f]{24}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
SCOPE_HASH_METHOD = "sha256_canonical_json_sorted_relation_ids_v1"


class RelationReviewSource(StrictModel):
    """Immutable artifacts from which the review scope was selected."""

    relation_discovery_build_id: str = Field(min_length=1)
    category_data_build_id: str = Field(min_length=1)
    selected_relations_file: str = Field(min_length=1)
    selected_relations_file_sha256: str = Field(pattern=SHA256_PATTERN)
    relation_table_sha256: str = Field(pattern=SHA256_PATTERN)


class RelationReviewScope(StrictModel):
    """Counts and digest of the exact relation IDs presented for review."""

    relation_count: int = Field(ge=1)
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    hash_method: Literal["sha256_canonical_json_sorted_relation_ids_v1"] = SCOPE_HASH_METHOD
    sorted_relation_ids_sha256: str = Field(pattern=SHA256_PATTERN)


class RejectedRelation(StrictModel):
    relation_id: str = Field(pattern=RELATION_ID_PATTERN)
    case_id: str = Field(min_length=1)
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]+$")
    reason: str = Field(min_length=1, max_length=240)


class RelationDisplayAlias(StrictModel):
    """Surface-only wording; aliases never change review validity or truth."""

    relation_id: str = Field(pattern=RELATION_ID_PATTERN)
    field: Literal["visible_tag", "hidden_tag"]
    source_value: str = Field(min_length=1)
    display_value: str = Field(min_length=1)


class CategoryRelationReview(StrictModel):
    schema_version: Literal["category_relation_review_v1"] = "category_relation_review_v1"
    review_id: str = Field(min_length=1)
    review_date: date
    reviewer_type: Literal["research_agent_manual_semantic_screen"]
    source: RelationReviewSource
    scope: RelationReviewScope
    accepted_relation_ids: list[str]
    rejected_relations: list[RejectedRelation]
    display_aliases: list[RelationDisplayAlias] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_partition_and_scope(self) -> CategoryRelationReview:
        accepted = self.accepted_relation_ids
        if any(not _is_relation_id(relation_id) for relation_id in accepted):
            raise ValueError("accepted_relation_ids contains an invalid relation ID")
        if len(accepted) != len(set(accepted)):
            raise ValueError("accepted_relation_ids contains duplicates")
        if accepted != sorted(accepted):
            raise ValueError("accepted_relation_ids must be sorted")

        rejected = [entry.relation_id for entry in self.rejected_relations]
        if len(rejected) != len(set(rejected)):
            raise ValueError("rejected_relations contains duplicate relation IDs")
        overlap = set(accepted).intersection(rejected)
        if overlap:
            raise ValueError(f"accepted/rejected relation IDs overlap: {sorted(overlap)}")

        all_relation_ids = [*accepted, *rejected]
        if len(all_relation_ids) != self.scope.relation_count:
            raise ValueError("scope relation_count does not match the accepted/rejected partition")
        if len(accepted) != self.scope.accepted_count:
            raise ValueError("scope accepted_count does not match accepted_relation_ids")
        if len(rejected) != self.scope.rejected_count:
            raise ValueError("scope rejected_count does not match rejected_relations")
        actual_scope_hash = relation_scope_sha256(all_relation_ids)
        if actual_scope_hash != self.scope.sorted_relation_ids_sha256:
            raise ValueError("scope relation-ID hash does not match the accepted/rejected partition")

        alias_keys = [(alias.relation_id, alias.field) for alias in self.display_aliases]
        if len(alias_keys) != len(set(alias_keys)):
            raise ValueError("display_aliases contains duplicate relation/field entries")
        unknown_aliases = {alias.relation_id for alias in self.display_aliases}.difference(all_relation_ids)
        if unknown_aliases:
            raise ValueError(f"display aliases reference relations outside the review scope: {sorted(unknown_aliases)}")
        return self

    def is_publishable(self, relation_id: str) -> bool:
        """Unknown and explicitly rejected relations fail closed."""

        return relation_id in set(self.accepted_relation_ids)


class CategoryRelationReviewExtension(StrictModel):
    """A compact, hash-bound manual extension of an existing frozen review."""

    schema_version: Literal["category_relation_review_extension_v1"]
    review_id: str = Field(min_length=1)
    review_date: date
    reviewer_type: Literal["research_agent_manual_semantic_screen"]
    parent_review_path: str = Field(min_length=1)
    parent_review_sha256: str = Field(pattern=SHA256_PATTERN)
    added_accepted_relation_ids: list[str]
    added_rejected_relations: list[RejectedRelation] = Field(default_factory=list)
    added_display_aliases: list[RelationDisplayAlias] = Field(default_factory=list)
    scope: RelationReviewScope

    @model_validator(mode="after")
    def validate_extension(self) -> CategoryRelationReviewExtension:
        if self.added_accepted_relation_ids != sorted(set(self.added_accepted_relation_ids)):
            raise ValueError("added accepted relation IDs must be sorted and unique")
        if any(not _is_relation_id(item) for item in self.added_accepted_relation_ids):
            raise ValueError("extension contains an invalid accepted relation ID")
        rejected_ids = [item.relation_id for item in self.added_rejected_relations]
        if len(rejected_ids) != len(set(rejected_ids)):
            raise ValueError("extension contains duplicate rejected relation IDs")
        if set(self.added_accepted_relation_ids).intersection(rejected_ids):
            raise ValueError("extension accepted and rejected relation IDs overlap")
        alias_keys = [(item.relation_id, item.field) for item in self.added_display_aliases]
        if len(alias_keys) != len(set(alias_keys)):
            raise ValueError("extension contains duplicate display aliases")
        return self


def _is_relation_id(value: str) -> bool:
    return len(value) == 24 and all(character in "0123456789abcdef" for character in value)


def relation_scope_sha256(relation_ids: list[str]) -> str:
    """Digest the canonical JSON representation of the sorted review scope."""

    return content_sha256(sorted(relation_ids))


def relation_review_sha256(review: CategoryRelationReview) -> str:
    """Content digest independent of JSON whitespace and key order."""

    return content_sha256(review.model_dump(mode="json"))


def load_relation_review(path: Path) -> CategoryRelationReview:
    """Load a full review or materialize a hash-bound compact review extension."""

    return _load_relation_review(path.resolve(), seen=set())


def _load_relation_review(path: Path, *, seen: set[Path]) -> CategoryRelationReview:
    if path in seen:
        raise ValueError("cyclic category relation review extension")
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == "category_relation_review_v1":
        return CategoryRelationReview.model_validate(payload)
    extension = CategoryRelationReviewExtension.model_validate(payload)
    parent_path = (path.parent / extension.parent_review_path).resolve()
    parent = _load_relation_review(parent_path, seen={*seen, path})
    if relation_review_sha256(parent) != extension.parent_review_sha256:
        raise ValueError("parent category relation review hash differs from extension commitment")

    parent_accepted = set(parent.accepted_relation_ids)
    parent_rejected = {item.relation_id for item in parent.rejected_relations}
    additions = set(extension.added_accepted_relation_ids)
    new_rejections = {item.relation_id for item in extension.added_rejected_relations}
    if additions.intersection(parent_accepted | parent_rejected):
        raise ValueError("extension accepted IDs already occur in the parent scope")
    if new_rejections.intersection(parent_accepted | parent_rejected):
        raise ValueError("extension rejected IDs already occur in the parent scope")

    accepted = sorted(parent_accepted | additions)
    rejected = [*parent.rejected_relations, *extension.added_rejected_relations]
    all_ids = [*accepted, *(item.relation_id for item in rejected)]
    expected_scope = RelationReviewScope(
        relation_count=len(all_ids),
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        sorted_relation_ids_sha256=relation_scope_sha256(all_ids),
    )
    if extension.scope != expected_scope:
        raise ValueError("extension scope does not match the materialized review")
    return CategoryRelationReview(
        review_id=extension.review_id,
        review_date=extension.review_date,
        reviewer_type=extension.reviewer_type,
        source=parent.source,
        scope=extension.scope,
        accepted_relation_ids=accepted,
        rejected_relations=rejected,
        display_aliases=[*parent.display_aliases, *extension.added_display_aliases],
    )


def filter_publishable_relations(
    relations: list[CategoryRelation],
    review: CategoryRelationReview,
) -> list[CategoryRelation]:
    """Return only allowlisted relations; unseen IDs are rejected by default."""

    relation_ids = [relation.relation_id for relation in relations]
    if len(relation_ids) != len(set(relation_ids)):
        raise ValueError("input relations contain duplicate relation IDs")
    accepted = set(review.accepted_relation_ids)
    return [
        relation.model_copy(update={"human_review_status": "accepted"})
        for relation in relations
        if relation.relation_id in accepted
    ]
