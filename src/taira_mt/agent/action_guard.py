"""Shared public structural guard for interactive recommendation actions.

The guard deliberately operates on a tiny public ledger and the public catalog
only.  It is shared by the Pattern policy and the equal-tool Direct-Agent
baseline, while task-specific policies remain responsible for choosing an
action.  In particular, it never receives a benchmark case, user simulator,
state-estimator output, or evaluator signal.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field, field_validator

from taira_mt.agent.catalog_tools import PublicCatalogTool
from taira_mt.benchmark.schemas import BenchmarkActionType, BenchmarkSystemAction
from taira_mt.models import StrictModel, normalize_slot_name

PUBLIC_ACTION_GUARD_VERSION = "public_structural_action_guard_v1"

# This is intentionally the same finite action vocabulary documented for the
# frozen capability gate.  It is not a latent-state-dependent policy rule.
PUBLIC_QUESTION_TARGETS: frozenset[str] = frozenset(
    {
        "occasion_interpretation",
        "specific_category",
        "price",
        "rating",
        "popularity_percentile",
    }
)


class PublicActionGuardError(ValueError):
    """Raised when a typed action violates a public structural contract."""


class PublicActionLedger(StrictModel):
    """Policy-neutral public facts needed for structural action validation.

    ``rejected_item_ids`` and ``explainable_item_ids`` are records of public
    dialogue outcomes, rather than evaluator labels.  A Direct-Agent baseline
    derives them conservatively from the observable text/action history;
    TAIRAS obtains the same public facts from its interaction memory.
    """

    asked_question_targets: list[str] = Field(default_factory=list)
    rejected_item_ids: list[str] = Field(default_factory=list)
    explainable_item_ids: list[str] = Field(default_factory=list)
    completed_turn_count: int = Field(default=0, ge=0)

    @field_validator(
        "asked_question_targets",
        "rejected_item_ids",
        "explainable_item_ids",
    )
    @classmethod
    def unique_values(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("public action ledger values must be unique")
        return values

    @classmethod
    def from_public_memory(
        cls,
        *,
        asked_question_targets: Iterable[str],
        rejected_item_ids: Iterable[str],
        explainable_item_ids: Iterable[str],
        completed_turn_count: int,
    ) -> PublicActionLedger:
        """Adapt a policy's public memory without importing its implementation."""

        return cls(
            asked_question_targets=[normalize_slot_name(item) for item in asked_question_targets],
            rejected_item_ids=list(rejected_item_ids),
            explainable_item_ids=list(explainable_item_ids),
            completed_turn_count=completed_turn_count,
        )


def _validate_catalog_references(
    action: BenchmarkSystemAction,
    catalog: PublicCatalogTool,
) -> None:
    catalog_item_ids = set(catalog.item_ids)
    action_item_ids = set(action.item_ids)
    unknown_action_items = action_item_ids.difference(catalog_item_ids)
    if unknown_action_items:
        raise PublicActionGuardError(
            f"action references item IDs outside the public catalog: {sorted(unknown_action_items)}"
        )
    for reference in action.evidence_refs:
        if reference.item_id not in catalog_item_ids:
            raise PublicActionGuardError(
                f"evidence references an item outside the public catalog: {reference.item_id}"
            )
        item = catalog.inspect(reference.item_id)
        if reference.catalog_field != "title" and reference.catalog_field not in item.attributes:
            raise PublicActionGuardError(
                "evidence references a field absent from the public catalog item: "
                f"{reference.item_id}.{reference.catalog_field}"
            )


def validate_public_action(
    action: BenchmarkSystemAction,
    *,
    catalog: PublicCatalogTool,
    ledger: PublicActionLedger,
    maximum_recommendation_items: int = 1,
    allow_propose_relax: bool = False,
) -> None:
    """Fail closed on action errors provable from public state alone.

    This function is intentionally *not* an oracle filter: it does not decide
    whether a recommendation satisfies latent preferences, whether a question
    is informative, or whether it is premature.  Those are either policy
    choices or evaluator-only metrics.
    """

    if maximum_recommendation_items < 1:
        raise ValueError("maximum_recommendation_items must be positive")
    _validate_catalog_references(action, catalog)
    if action.action_type == BenchmarkActionType.ASK:
        assert action.question_target is not None
        target = normalize_slot_name(action.question_target)
        if action.question_target != target or target not in PUBLIC_QUESTION_TARGETS:
            raise PublicActionGuardError(f"unsupported public question target: {target}")
        if target in set(ledger.asked_question_targets):
            raise PublicActionGuardError(f"question target was already answered publicly: {target}")
        return
    if action.action_type == BenchmarkActionType.RECOMMEND:
        if len(action.item_ids) > maximum_recommendation_items:
            raise PublicActionGuardError(
                "recommendation exceeds the shared public top-k contract: "
                f"{len(action.item_ids)} > {maximum_recommendation_items}"
            )
        rejected = set(ledger.rejected_item_ids)
        repeated = rejected.intersection(action.item_ids)
        if repeated:
            raise PublicActionGuardError(
                f"recommendation repeats publicly rejected items: {sorted(repeated)}"
            )
        if any(reference.item_id not in set(action.item_ids) for reference in action.evidence_refs):
            raise PublicActionGuardError(
                "recommendation evidence must refer to the recommended public item"
            )
        return
    if action.action_type == BenchmarkActionType.EXPLAIN:
        explained = set(action.item_ids)
        current = set(ledger.explainable_item_ids)
        if not explained or not explained.issubset(current):
            raise PublicActionGuardError(
                "explanation must concern a currently public recommendation of the same policy"
            )
        if any(reference.item_id not in explained for reference in action.evidence_refs):
            raise PublicActionGuardError(
                "explanation evidence must refer to the explained public recommendation"
            )
        return
    if action.action_type == BenchmarkActionType.PROPOSE_RELAX and not allow_propose_relax:
        raise PublicActionGuardError("propose_relax is disabled by the shared capability-gate contract")
