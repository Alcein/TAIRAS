"""Enumerate auditable numeric goal specifications before any pool is built."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum

from pydantic import Field, model_validator

from taira_mt.constraints import render_constraint, satisfies_predicate
from taira_mt.data.goal_construction import CategoryRelation, NormalizedCatalogRow
from taira_mt.models import (
    Constraint,
    ConstraintImportance,
    ConstraintOperator,
    NumericRange,
    StrictModel,
)
from taira_mt.provenance import content_sha256

NUMERIC_SPEC_VERSION = "numeric_goal_spec_v1"
POPULARITY_MAPPING_VERSION = "inverse_midrank_percentile_v1"
QUANTILE_VERSION = "nearest_rank_floor_v1"


class GoalFamily(str, Enum):
    C = "C"
    CP_MAX = "CP_MAX"
    CP_BAND = "CP_BAND"
    CR = "CR"
    CPR = "CPR"
    CS = "CS"


class NumericGoalConfig(StrictModel):
    price_max_quantiles: list[float] = Field(default_factory=lambda: [0.35, 0.50, 0.65])
    price_band_quantiles: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0.25, 0.60), (0.35, 0.75)]
    )
    rating_min_quantiles: list[float] = Field(default_factory=lambda: [0.50, 0.70])
    soft_price_quantile: float = 0.35
    soft_rating_quantile: float = 0.70
    soft_popularity_quantile: float = 0.70
    minimum_numeric_observations: int = Field(default=50, ge=2)
    minimum_numeric_coverage: float = Field(default=0.25, gt=0.0, le=1.0)
    minimum_hidden_hard_feasible: int = Field(default=20, ge=1)
    minimum_acceptable_count: int = Field(default=10, ge=1)
    minimum_boundary_negative_count: int = Field(default=5, ge=1)
    minimum_acceptable_prevalence: float = Field(default=0.05, gt=0.0, le=1.0)
    maximum_acceptable_prevalence: float = Field(default=0.40, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_quantiles(self) -> NumericGoalConfig:
        quantiles = [
            *self.price_max_quantiles,
            *self.rating_min_quantiles,
            self.soft_price_quantile,
            self.soft_rating_quantile,
            self.soft_popularity_quantile,
            *(value for pair in self.price_band_quantiles for value in pair),
        ]
        if any(not 0.0 < value < 1.0 for value in quantiles):
            raise ValueError("all numeric goal quantiles must be strictly between zero and one")
        if any(lower >= upper for lower, upper in self.price_band_quantiles):
            raise ValueError("price band quantiles must be increasing")
        if self.minimum_acceptable_prevalence >= self.maximum_acceptable_prevalence:
            raise ValueError("acceptable prevalence bounds must be increasing")
        return self


class NumericThresholdProvenance(StrictModel):
    slot: str
    quantiles: list[float]
    raw_values: list[float]
    normalized_values: list[str]
    observed_count: int = Field(ge=1)
    visible_universe_count: int = Field(ge=1)
    source_universe: str = "full_visible_universe"
    quantile_version: str = QUANTILE_VERSION
    transformation_version: str | None = None


class GoalSpecStatistics(StrictModel):
    visible_universe_count: int = Field(ge=1)
    hidden_category_count: int = Field(ge=1)
    hidden_hard_feasible_count: int = Field(ge=0)
    acceptable_count: int = Field(ge=0)
    acceptable_prevalence: float = Field(ge=0.0, le=1.0)
    observed_count_by_slot: dict[str, int]
    missing_count_by_slot: dict[str, int]
    boundary_negative_count_by_slot: dict[str, int]
    soft_miss_count_by_slot: dict[str, int]


class NumericGoalSpec(StrictModel):
    spec_id: str
    spec_version: str = NUMERIC_SPEC_VERSION
    family: GoalFamily
    relation_id: str
    domain: str
    visible_tag: str
    hidden_tag: str
    numeric_constraints: list[Constraint]
    humanized_constraints: list[str]
    threshold_provenance: list[NumericThresholdProvenance]
    statistics: GoalSpecStatistics

    def identity_payload(self) -> dict[str, object]:
        return {
            "spec_version": self.spec_version,
            "family": self.family.value,
            "relation_id": self.relation_id,
            "domain": self.domain,
            "visible_tag": self.visible_tag,
            "hidden_tag": self.hidden_tag,
            "numeric_constraints": [_canonical_constraint(item) for item in self.numeric_constraints],
            "threshold_provenance": [
                item.model_dump(mode="json") for item in self.threshold_provenance
            ],
        }

    @model_validator(mode="after")
    def validate_identity_and_family(self) -> NumericGoalSpec:
        expected = f"numeric-{content_sha256(self.identity_payload())[:24]}"
        if self.spec_id != expected:
            raise ValueError("numeric goal spec ID does not match its canonical definition")
        if self.family == GoalFamily.C and self.numeric_constraints:
            raise ValueError("category-only goal cannot contain numeric constraints")
        if self.family != GoalFamily.C and not self.numeric_constraints:
            raise ValueError("numeric goal family requires at least one numeric constraint")
        slots = [constraint.slot for constraint in self.numeric_constraints]
        if len(slots) != len(set(slots)):
            raise ValueError("numeric goal spec cannot repeat a slot")
        return self


class NumericSpecRejection(StrictModel):
    family: GoalFamily
    candidate_key: str
    reason: str


class NumericGoalEnumerationResult(StrictModel):
    relation_id: str
    specs: list[NumericGoalSpec]
    rejections: list[NumericSpecRejection]

    @property
    def rejection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rejection in self.rejections:
            counts[rejection.reason] = counts.get(rejection.reason, 0) + 1
        return dict(sorted(counts.items()))


@dataclass(frozen=True)
class _CandidateSpec:
    family: GoalFamily
    constraints: tuple[Constraint, ...]
    provenance: tuple[NumericThresholdProvenance, ...]


def _canonical_decimal(value: float | int) -> str:
    decimal = Decimal(str(value)).normalize()
    rendered = format(decimal, "f")
    return "0" if rendered in {"-0", ""} else rendered


def _canonical_constraint(constraint: Constraint) -> dict[str, object]:
    if isinstance(constraint.value, NumericRange):
        value: object = {
            "minimum": None
            if constraint.value.minimum is None
            else _canonical_decimal(constraint.value.minimum),
            "maximum": None
            if constraint.value.maximum is None
            else _canonical_decimal(constraint.value.maximum),
            "include_minimum": constraint.value.include_minimum,
            "include_maximum": constraint.value.include_maximum,
        }
    elif isinstance(constraint.value, (int, float)) and not isinstance(constraint.value, bool):
        value = _canonical_decimal(constraint.value)
    else:
        value = constraint.value
    return {
        "slot": constraint.slot,
        "operator": constraint.operator.value,
        "value": value,
        "importance": constraint.importance.value,
        "unit": constraint.unit,
    }


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile from no values")
    ordered = sorted(values)
    index = math.floor(quantile * (len(ordered) - 1))
    return ordered[index]


def _human_price(value: float) -> float:
    step = Decimal("1")
    if value >= 500:
        step = Decimal("25")
    elif value >= 100:
        step = Decimal("10")
    elif value >= 20:
        step = Decimal("5")
    rounded = (Decimal(str(value)) / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * step
    return float(max(rounded, step))


def _human_rating(value: float) -> float:
    rounded = (Decimal(str(value)) * 2).quantize(Decimal("1"), rounding=ROUND_HALF_UP) / 2
    return float(max(Decimal("1"), min(Decimal("5"), rounded)))


def popularity_percentiles(rows: list[NormalizedCatalogRow]) -> dict[str, float]:
    """Map lower Amazon rank to higher popularity with stable tie midranks."""

    ranked = sorted(
        ((row.ranking, row.item_id) for row in rows if row.ranking is not None),
        key=lambda pair: (pair[0], pair[1]),
    )
    if not ranked:
        return {}
    if len(ranked) == 1:
        return {ranked[0][1]: 1.0}
    result: dict[str, float] = {}
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        midrank = (start + end - 1) / 2.0
        percentile = 1.0 - midrank / (len(ranked) - 1)
        for _, item_id in ranked[start:end]:
            result[item_id] = percentile
        start = end
    return result


def _slot_value(
    row: NormalizedCatalogRow,
    slot: str,
    popularity: dict[str, float],
) -> float | None:
    if slot == "price":
        return row.price
    if slot == "rating":
        return row.rating if row.rating is not None and row.rating <= 5.0 else None
    if slot == "popularity_percentile":
        return popularity.get(row.item_id)
    raise KeyError(f"unsupported numeric slot: {slot}")


def _provenance(
    *,
    slot: str,
    quantiles: list[float],
    raw_values: list[float],
    normalized_values: list[float],
    observed_count: int,
    visible_count: int,
) -> NumericThresholdProvenance:
    return NumericThresholdProvenance(
        slot=slot,
        quantiles=quantiles,
        raw_values=raw_values,
        normalized_values=[_canonical_decimal(value) for value in normalized_values],
        observed_count=observed_count,
        visible_universe_count=visible_count,
        transformation_version=(
            POPULARITY_MAPPING_VERSION if slot == "popularity_percentile" else None
        ),
    )


def _predicate_matches(
    row: NormalizedCatalogRow,
    constraint: Constraint,
    popularity: dict[str, float],
) -> bool:
    return satisfies_predicate(_slot_value(row, constraint.slot, popularity), constraint)


def _candidate_specs(
    relation: CategoryRelation,
    visible_rows: list[NormalizedCatalogRow],
    popularity: dict[str, float],
    config: NumericGoalConfig,
) -> tuple[list[_CandidateSpec], list[NumericSpecRejection]]:
    values = {
        slot: [
            value
            for row in visible_rows
            if (value := _slot_value(row, slot, popularity)) is not None
        ]
        for slot in ("price", "rating", "popularity_percentile")
    }
    candidates = [_CandidateSpec(GoalFamily.C, (), ())]
    rejections: list[NumericSpecRejection] = []

    def covered(slot: str) -> bool:
        count = len(values[slot])
        return (
            count >= config.minimum_numeric_observations
            and count / len(visible_rows) >= config.minimum_numeric_coverage
        )

    if covered("price"):
        for quantile in config.price_max_quantiles:
            raw = _quantile(values["price"], quantile)
            threshold = _human_price(raw)
            constraint = Constraint(
                slot="price",
                operator=ConstraintOperator.LTE,
                value=threshold,
                importance=ConstraintImportance.HARD,
                unit="USD",
            )
            provenance = _provenance(
                slot="price",
                quantiles=[quantile],
                raw_values=[raw],
                normalized_values=[threshold],
                observed_count=len(values["price"]),
                visible_count=len(visible_rows),
            )
            candidates.append(_CandidateSpec(GoalFamily.CP_MAX, (constraint,), (provenance,)))
        for lower_q, upper_q in config.price_band_quantiles:
            raw_lower = _quantile(values["price"], lower_q)
            raw_upper = _quantile(values["price"], upper_q)
            lower = _human_price(raw_lower)
            upper = _human_price(raw_upper)
            if lower >= upper:
                rejections.append(
                    NumericSpecRejection(
                        family=GoalFamily.CP_BAND,
                        candidate_key=f"price_band:{lower_q}:{upper_q}",
                        reason="ROUNDED_BAND_NOT_INCREASING",
                    )
                )
                continue
            constraint = Constraint(
                slot="price",
                operator=ConstraintOperator.BETWEEN,
                value=NumericRange(minimum=lower, maximum=upper),
                importance=ConstraintImportance.HARD,
                unit="USD",
            )
            provenance = _provenance(
                slot="price",
                quantiles=[lower_q, upper_q],
                raw_values=[raw_lower, raw_upper],
                normalized_values=[lower, upper],
                observed_count=len(values["price"]),
                visible_count=len(visible_rows),
            )
            candidates.append(_CandidateSpec(GoalFamily.CP_BAND, (constraint,), (provenance,)))
        soft_q = config.soft_price_quantile
        raw = _quantile(values["price"], soft_q)
        threshold = _human_price(raw)
        candidates.append(
            _CandidateSpec(
                GoalFamily.CS,
                (
                    Constraint(
                        slot="price",
                        operator=ConstraintOperator.LTE,
                        value=threshold,
                        importance=ConstraintImportance.SOFT,
                        unit="USD",
                    ),
                ),
                (
                    _provenance(
                        slot="price",
                        quantiles=[soft_q],
                        raw_values=[raw],
                        normalized_values=[threshold],
                        observed_count=len(values["price"]),
                        visible_count=len(visible_rows),
                    ),
                ),
            )
        )
    else:
        rejections.append(
            NumericSpecRejection(
                family=GoalFamily.CP_MAX,
                candidate_key="price",
                reason="PRICE_COVERAGE_TOO_LOW",
            )
        )

    rating_constraints: list[tuple[Constraint, NumericThresholdProvenance]] = []
    if relation.domain == "amazon_music" and covered("rating"):
        for quantile in config.rating_min_quantiles:
            raw = _quantile(values["rating"], quantile)
            threshold = _human_rating(raw)
            constraint = Constraint(
                slot="rating",
                operator=ConstraintOperator.GTE,
                value=threshold,
                importance=ConstraintImportance.HARD,
                unit="stars",
            )
            provenance = _provenance(
                slot="rating",
                quantiles=[quantile],
                raw_values=[raw],
                normalized_values=[threshold],
                observed_count=len(values["rating"]),
                visible_count=len(visible_rows),
            )
            rating_constraints.append((constraint, provenance))
            candidates.append(_CandidateSpec(GoalFamily.CR, (constraint,), (provenance,)))
        for price_candidate in [item for item in candidates if item.family == GoalFamily.CP_MAX]:
            for rating_constraint, rating_provenance in rating_constraints:
                candidates.append(
                    _CandidateSpec(
                        GoalFamily.CPR,
                        (*price_candidate.constraints, rating_constraint),
                        (*price_candidate.provenance, rating_provenance),
                    )
                )
        soft_q = config.soft_rating_quantile
        raw = _quantile(values["rating"], soft_q)
        threshold = _human_rating(raw)
        candidates.append(
            _CandidateSpec(
                GoalFamily.CS,
                (
                    Constraint(
                        slot="rating",
                        operator=ConstraintOperator.GTE,
                        value=threshold,
                        importance=ConstraintImportance.SOFT,
                        unit="stars",
                    ),
                ),
                (
                    _provenance(
                        slot="rating",
                        quantiles=[soft_q],
                        raw_values=[raw],
                        normalized_values=[threshold],
                        observed_count=len(values["rating"]),
                        visible_count=len(visible_rows),
                    ),
                ),
            )
        )

    if covered("popularity_percentile"):
        soft_q = config.soft_popularity_quantile
        raw = _quantile(values["popularity_percentile"], soft_q)
        threshold = float(_canonical_decimal(raw))
        candidates.append(
            _CandidateSpec(
                GoalFamily.CS,
                (
                    Constraint(
                        slot="popularity_percentile",
                        operator=ConstraintOperator.GTE,
                        value=threshold,
                        importance=ConstraintImportance.SOFT,
                        unit="percentile",
                    ),
                ),
                (
                    _provenance(
                        slot="popularity_percentile",
                        quantiles=[soft_q],
                        raw_values=[raw],
                        normalized_values=[threshold],
                        observed_count=len(values["popularity_percentile"]),
                        visible_count=len(visible_rows),
                    ),
                ),
            )
        )
    return candidates, rejections


def _validate_candidate(
    relation: CategoryRelation,
    visible_rows: list[NormalizedCatalogRow],
    popularity: dict[str, float],
    candidate: _CandidateSpec,
    config: NumericGoalConfig,
) -> GoalSpecStatistics | str:
    hidden_rows = [row for row in visible_rows if relation.hidden_tag in row.category_tags]
    if not hidden_rows:
        return "NO_HIDDEN_CATEGORY_ROWS"
    hard = [
        constraint
        for constraint in candidate.constraints
        if constraint.importance == ConstraintImportance.HARD
    ]
    soft = [
        constraint
        for constraint in candidate.constraints
        if constraint.importance == ConstraintImportance.SOFT
    ]
    hard_feasible = [
        row
        for row in hidden_rows
        if all(_predicate_matches(row, constraint, popularity) for constraint in hard)
    ]
    acceptable = [
        row
        for row in hard_feasible
        if all(_predicate_matches(row, constraint, popularity) for constraint in soft)
    ]
    if len(hard_feasible) < config.minimum_hidden_hard_feasible:
        return "HIDDEN_HARD_FEASIBLE_TOO_LOW"
    if len(acceptable) < config.minimum_acceptable_count:
        return "ACCEPTABLE_COUNT_TOO_LOW"
    prevalence = len(acceptable) / len(visible_rows)
    if not config.minimum_acceptable_prevalence <= prevalence <= config.maximum_acceptable_prevalence:
        return "ACCEPTABLE_PREVALENCE_OUT_OF_RANGE"

    observed: dict[str, int] = {}
    missing: dict[str, int] = {}
    boundary: dict[str, int] = {}
    soft_miss: dict[str, int] = {}
    for constraint in candidate.constraints:
        slot = constraint.slot
        observed[slot] = sum(_slot_value(row, slot, popularity) is not None for row in visible_rows)
        missing[slot] = len(visible_rows) - observed[slot]
        if constraint.importance == ConstraintImportance.HARD:
            others = [item for item in hard if item.slot != slot]
            boundary[slot] = sum(
                _slot_value(row, slot, popularity) is not None
                and all(_predicate_matches(row, other, popularity) for other in others)
                and not _predicate_matches(row, constraint, popularity)
                for row in hidden_rows
            )
            if boundary[slot] < config.minimum_boundary_negative_count:
                return f"BOUNDARY_NEGATIVES_TOO_LOW:{slot}"
        else:
            soft_miss[slot] = sum(
                _slot_value(row, slot, popularity) is not None
                and not _predicate_matches(row, constraint, popularity)
                for row in hard_feasible
            )
            if soft_miss[slot] < config.minimum_boundary_negative_count:
                return f"SOFT_MISSES_TOO_LOW:{slot}"
    return GoalSpecStatistics(
        visible_universe_count=len(visible_rows),
        hidden_category_count=len(hidden_rows),
        hidden_hard_feasible_count=len(hard_feasible),
        acceptable_count=len(acceptable),
        acceptable_prevalence=prevalence,
        observed_count_by_slot=dict(sorted(observed.items())),
        missing_count_by_slot=dict(sorted(missing.items())),
        boundary_negative_count_by_slot=dict(sorted(boundary.items())),
        soft_miss_count_by_slot=dict(sorted(soft_miss.items())),
    )


def enumerate_numeric_goal_specs(
    relation: CategoryRelation,
    rows: list[NormalizedCatalogRow],
    config: NumericGoalConfig | None = None,
) -> NumericGoalEnumerationResult:
    """Enumerate specs from the full visible universe without a pool argument."""

    config = config or NumericGoalConfig()
    visible_rows = [row for row in rows if relation.visible_tag in row.category_tags]
    if not visible_rows:
        raise ValueError("relation visible tag has no catalog rows")
    ids = [row.item_id for row in visible_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("visible universe item IDs must be unique")
    popularity = popularity_percentiles(visible_rows)
    candidates, rejections = _candidate_specs(relation, visible_rows, popularity, config)
    specs: list[NumericGoalSpec] = []
    seen_payloads: set[str] = set()
    for index, candidate in enumerate(candidates):
        statistics = _validate_candidate(relation, visible_rows, popularity, candidate, config)
        key = f"{candidate.family.value}:{index}"
        if isinstance(statistics, str):
            rejections.append(
                NumericSpecRejection(
                    family=candidate.family,
                    candidate_key=key,
                    reason=statistics,
                )
            )
            continue
        payload = {
            "spec_version": NUMERIC_SPEC_VERSION,
            "family": candidate.family.value,
            "relation_id": relation.relation_id,
            "domain": relation.domain,
            "visible_tag": relation.visible_tag,
            "hidden_tag": relation.hidden_tag,
            "numeric_constraints": [_canonical_constraint(item) for item in candidate.constraints],
            "threshold_provenance": [
                item.model_dump(mode="json") for item in candidate.provenance
            ],
        }
        signature = content_sha256(payload)
        if signature in seen_payloads:
            rejections.append(
                NumericSpecRejection(
                    family=candidate.family,
                    candidate_key=key,
                    reason="DUPLICATE_NORMALIZED_SPEC",
                )
            )
            continue
        seen_payloads.add(signature)
        specs.append(
            NumericGoalSpec(
                spec_id=f"numeric-{signature[:24]}",
                family=candidate.family,
                relation_id=relation.relation_id,
                domain=relation.domain,
                visible_tag=relation.visible_tag,
                hidden_tag=relation.hidden_tag,
                numeric_constraints=list(candidate.constraints),
                humanized_constraints=[
                    render_constraint(constraint.slot, constraint)
                    for constraint in candidate.constraints
                ],
                threshold_provenance=list(candidate.provenance),
                statistics=statistics,
            )
        )
    return NumericGoalEnumerationResult(
        relation_id=relation.relation_id,
        specs=sorted(specs, key=lambda item: item.spec_id),
        rejections=sorted(
            rejections,
            key=lambda item: (item.family.value, item.candidate_key, item.reason),
        ),
    )
