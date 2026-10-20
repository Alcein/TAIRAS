"""Catalog-equivalent natural-language surfaces for numeric benchmark goals.

The hidden predicate remains unchanged.  A displayed boundary may be simplified
only when it induces exactly the same pass/fail partition over the frozen
candidate pool.  This prevents awkward values such as 4.675 stars without
quietly changing which recommendations the simulator accepts.
"""

from __future__ import annotations

import math

from taira_mt.constraints.rendering import render_constraint
from taira_mt.models import ConstraintOperator, ConstraintPredicate
from taira_mt.recommender.catalog import CandidatePool

LEGACY_CONSTRAINT_SURFACE_POLICY_VERSION = "literal_numeric_surface_v0"
CATALOG_EQUIVALENT_CONSTRAINT_SURFACE_VERSION = (
    "catalog_equivalent_natural_surface_v1"
)


def _catalog_values(pool: CandidatePool, slot: str) -> tuple[float, ...]:
    return tuple(
        float(value)
        for item in pool.remaining_items()
        if isinstance((value := item.attributes.get(slot)), (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _passes(value: float, operator: ConstraintOperator, boundary: float) -> bool:
    if operator == ConstraintOperator.LTE:
        return value <= boundary
    if operator == ConstraintOperator.GTE:
        return value >= boundary
    raise ValueError(f"unsupported numeric surface operator: {operator}")


def _same_catalog_partition(
    values: tuple[float, ...],
    operator: ConstraintOperator,
    original: float,
    candidate: float,
) -> bool:
    return all(
        _passes(value, operator, original) == _passes(value, operator, candidate)
        for value in values
    )


def _grid_neighbors(value: float, digits: int) -> tuple[float, ...]:
    scale = 10**digits
    scaled = value * scale
    lower = math.floor(scaled + 1e-10)
    upper = math.ceil(scaled - 1e-10)
    grid = {index / scale for index in range(lower - 2, upper + 3)}
    return tuple(sorted(grid, key=lambda item: (abs(item - value), item)))


def catalog_equivalent_surface_predicate(
    pool: CandidatePool,
    slot: str,
    predicate: ConstraintPredicate,
) -> ConstraintPredicate:
    """Return the simplest safe display predicate for a frozen catalog.

    Prices retain cent precision, star ratings target one decimal place, and
    popularity is expressed as a whole or one-decimal top-percent value.  If no
    such boundary preserves the catalog partition, progressively more precise
    candidates are tried before falling back to the exact predicate.
    """

    if predicate.operator not in {ConstraintOperator.LTE, ConstraintOperator.GTE}:
        return predicate
    if isinstance(predicate.value, bool) or not isinstance(
        predicate.value, (int, float)
    ):
        return predicate
    original = float(predicate.value)
    if not math.isfinite(original):
        return predicate
    values = _catalog_values(pool, slot)
    if not values:
        return predicate

    if slot == "popularity_percentile":
        display_value = (1.0 - original) * 100.0
        precisions = (0, 1, 2, 3)
    elif slot == "rating":
        display_value = original
        precisions = (1, 2, 3)
    elif slot == "price":
        display_value = original
        precisions = (2, 3, 4)
    else:
        return predicate

    for digits in precisions:
        for displayed in _grid_neighbors(display_value, digits):
            if slot == "popularity_percentile":
                valid_display = 0.0 <= displayed <= 100.0
                candidate = 1.0 - displayed / 100.0
            elif slot == "rating":
                valid_display = 0.0 <= displayed <= 5.0
                candidate = displayed
            else:
                valid_display = displayed >= 0.0
                candidate = displayed
            if not valid_display:
                continue
            if _same_catalog_partition(
                values,
                predicate.operator,
                original,
                candidate,
            ):
                return predicate.model_copy(update={"value": candidate}, deep=True)
    return predicate


def catalog_equivalent_constraint_surface(
    pool: CandidatePool,
    slot: str,
    predicate: ConstraintPredicate,
) -> str:
    """Render a natural boundary with a verified unchanged catalog partition."""

    surface_predicate = catalog_equivalent_surface_predicate(pool, slot, predicate)
    return render_constraint(slot, surface_predicate)
