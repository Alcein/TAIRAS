"""Canonical, human-readable rendering for constraint predicates."""

from __future__ import annotations

from taira_mt.models import (
    ConstraintImportance,
    ConstraintOperator,
    ConstraintPredicate,
    NumericRange,
)


def _number(value: float | int) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:.2f}".rstrip("0").rstrip(".")


def render_constraint_value(predicate: ConstraintPredicate) -> str:
    value = predicate.value
    if isinstance(value, NumericRange):
        assert value.minimum is not None and value.maximum is not None
        return f"between {_number(value.minimum)} and {_number(value.maximum)}"
    if isinstance(value, list):
        return ", ".join(str(part) for part in value)
    return str(value)


def render_constraint(slot: str, predicate: ConstraintPredicate) -> str:
    """Render a predicate using templates that the rule tracker can invert."""

    value = predicate.value
    soft_prefix = "Ideally, " if predicate.importance == ConstraintImportance.SOFT else ""
    if slot == "price" and predicate.operator == ConstraintOperator.LTE:
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        if predicate.importance == ConstraintImportance.SOFT:
            return f"Ideally, I'd prefer to stay under ${_number(value)}."
        return f"My budget is at most ${_number(value)}."
    if slot == "price" and predicate.operator == ConstraintOperator.BETWEEN:
        assert isinstance(value, NumericRange)
        assert value.minimum is not None and value.maximum is not None
        return (
            f"{soft_prefix}I'd like to spend between ${_number(value.minimum)} "
            f"and ${_number(value.maximum)}."
        )
    if slot == "rating" and predicate.operator == ConstraintOperator.GTE:
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        modal = "I'd prefer" if predicate.importance == ConstraintImportance.SOFT else "I need"
        return f"{soft_prefix}{modal} a rating of at least {_number(value)} stars."
    if slot == "popularity_percentile" and predicate.operator == ConstraintOperator.GTE:
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        top_percent = max(0.0, min(100.0, (1.0 - float(value)) * 100.0))
        return f"{soft_prefix}I'd prefer something roughly in the top {_number(top_percent)}% by popularity."
    if predicate.operator == ConstraintOperator.EQ:
        modal = "prefer" if predicate.importance == ConstraintImportance.SOFT else "need"
        return f"For {slot}, I {modal} {render_constraint_value(predicate)}."
    operator = predicate.operator.value.upper()
    return f"For {slot}, I {'prefer' if soft_prefix else 'need'} {operator} {render_constraint_value(predicate)}."
