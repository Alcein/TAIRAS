"""Executable public/private predicate semantics.

This module is deliberately independent of the simulator.  The online ranker
receives public predicates only, while the evaluator may apply the same
operators to private constraints.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from taira_mt.models import (
    AttributeValue,
    CatalogItem,
    ConstraintEvidenceStatus,
    ConstraintImportance,
    ConstraintOperator,
    ConstraintPredicate,
    NumericRange,
    ScalarValue,
)


def _as_values(value: AttributeValue | None) -> list[ScalarValue]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _normalized_scalar(value: ScalarValue) -> str | float | bool:
    if isinstance(value, str):
        return value.strip().casefold()
    if isinstance(value, bool):
        return value
    return float(value)


def _same_scalar(left: ScalarValue, right: ScalarValue) -> bool:
    return _normalized_scalar(left) == _normalized_scalar(right)


def _numeric(value: ScalarValue) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    try:
        number = float(value.strip())
        return number if math.isfinite(number) else None
    except (AttributeError, ValueError):
        return None


def _in_range(value: float, bounds: NumericRange) -> bool:
    if bounds.minimum is not None:
        if value < bounds.minimum or (value == bounds.minimum and not bounds.include_minimum):
            return False
    if bounds.maximum is not None:
        if value > bounds.maximum or (value == bounds.maximum and not bounds.include_maximum):
            return False
    return True


def predicate_evidence_status(
    attribute: AttributeValue | None,
    predicate: ConstraintPredicate,
) -> ConstraintEvidenceStatus:
    """Distinguish observed violation from absent or unusable catalog evidence."""

    item_values = [
        value
        for value in _as_values(attribute)
        if not isinstance(value, str) or bool(value.strip())
    ]
    if not item_values:
        return ConstraintEvidenceStatus.UNKNOWN
    if isinstance(predicate.value, NumericRange):
        numeric_values = [
            number for item_value in item_values if (number := _numeric(item_value)) is not None
        ]
        if not numeric_values:
            return ConstraintEvidenceStatus.UNKNOWN
        return (
            ConstraintEvidenceStatus.SATISFIED
            if any(_in_range(number, predicate.value) for number in numeric_values)
            else ConstraintEvidenceStatus.VIOLATED
        )

    expected_values = _as_values(predicate.value)
    equality = any(_same_scalar(item, expected) for item in item_values for expected in expected_values)
    if predicate.operator in {ConstraintOperator.EQ, ConstraintOperator.ANY_OF}:
        matched = equality
    elif predicate.operator in {ConstraintOperator.NE, ConstraintOperator.NOT_IN}:
        matched = not equality
    elif predicate.operator == ConstraintOperator.CONTAINS:
        matched = any(
            str(_normalized_scalar(expected)) in str(_normalized_scalar(item))
            for item in item_values
            for expected in expected_values
        )
    else:
        threshold = _numeric(expected_values[0])
        numeric_values = [number for value in item_values if (number := _numeric(value)) is not None]
        if threshold is None or not numeric_values:
            return ConstraintEvidenceStatus.UNKNOWN
        if predicate.operator == ConstraintOperator.GTE:
            matched = any(number >= threshold for number in numeric_values)
        elif predicate.operator == ConstraintOperator.LTE:
            matched = any(number <= threshold for number in numeric_values)
        elif predicate.operator == ConstraintOperator.BETWEEN:
            raise AssertionError("between predicate must carry NumericRange")
        else:  # pragma: no cover - guarded by the enum/model contract
            raise ValueError(f"unsupported constraint operator: {predicate.operator}")
    return (
        ConstraintEvidenceStatus.SATISFIED
        if matched
        else ConstraintEvidenceStatus.VIOLATED
    )


def satisfies_predicate(attribute: AttributeValue | None, predicate: ConstraintPredicate) -> bool:
    """Return whether observed catalog evidence satisfies the predicate."""

    return predicate_evidence_status(attribute, predicate) == ConstraintEvidenceStatus.SATISFIED


def soft_predicate_score(
    attribute: AttributeValue | None,
    predicate: ConstraintPredicate,
    *,
    slot: str | None = None,
) -> float:
    """Grade a soft preference on [0, 1], with missing values scoring zero."""

    if satisfies_predicate(attribute, predicate):
        return 1.0
    values = [number for raw in _as_values(attribute) if (number := _numeric(raw)) is not None]
    if not values or isinstance(predicate.value, NumericRange):
        return 0.0
    threshold = _numeric(predicate.value) if not isinstance(predicate.value, list) else None
    if threshold is None or threshold <= 0.0:
        return 0.0
    if predicate.operator == ConstraintOperator.LTE:
        number = min(values)
        return max(0.0, min(1.0, 1.0 - (number - threshold) / threshold))
    if predicate.operator == ConstraintOperator.GTE:
        number = max(values)
        if slot == "rating":
            denominator = max(threshold - 1.0, 1e-12)
            return max(0.0, min(1.0, 1.0 - (threshold - number) / denominator))
        return max(0.0, min(1.0, number / threshold))
    return 0.0


def public_posterior_items(
    known_constraints: Mapping[str, ConstraintPredicate],
    items: tuple[CatalogItem, ...] | list[CatalogItem],
) -> list[CatalogItem]:
    """Filter hard public predicates and rank by observed soft preferences."""

    hard = {
        slot: predicate
        for slot, predicate in known_constraints.items()
        if predicate.importance == ConstraintImportance.HARD
    }
    soft = {
        slot: predicate
        for slot, predicate in known_constraints.items()
        if predicate.importance == ConstraintImportance.SOFT
    }
    posterior = [
        item
        for item in items
        if all(satisfies_predicate(item.attributes.get(slot), predicate) for slot, predicate in hard.items())
    ]
    if not soft:
        return posterior

    def soft_score(item: CatalogItem) -> float:
        return sum(
            soft_predicate_score(item.attributes.get(slot), predicate, slot=slot)
            for slot, predicate in soft.items()
        ) / len(soft)

    return sorted(posterior, key=lambda item: -soft_score(item))
