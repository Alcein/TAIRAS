"""Independent set-valued catalog oracle for benchmark goal versions."""

from __future__ import annotations

from dataclasses import dataclass

from taira_mt.benchmark.schemas import GoalVersion
from taira_mt.constraints.predicates import predicate_evidence_status, soft_predicate_score
from taira_mt.models import (
    CatalogItem,
    ConstraintEvidenceStatus,
    ConstraintImportance,
    OracleAssessment,
)
from taira_mt.recommender.catalog import CandidatePool

LEGACY_BENCHMARK_UTILITY_VERSION = "benchmark_set_utility_v1"
BENCHMARK_UTILITY_VERSION = "benchmark_set_utility_v2_missing_aware"
SUPPORTED_BENCHMARK_UTILITY_VERSIONS = {
    LEGACY_BENCHMARK_UTILITY_VERSION,
    BENCHMARK_UTILITY_VERSION,
}


@dataclass(frozen=True)
class BenchmarkSlateAssessment:
    assessments: tuple[OracleAssessment, ...]
    best_item_id: str | None
    best_utility: float
    accepted_item_id: str | None
    all_items_hard_violating: bool


@dataclass(frozen=True)
class RecomputedGoalTruth:
    feasible_item_ids: tuple[str, ...]
    acceptable_item_ids: tuple[str, ...]
    unverifiable_item_ids: tuple[str, ...] = ()


class IndependentBenchmarkOracle:
    """Judge items from frozen predicates without consulting dialogue policy state."""

    def __init__(self, utility_version: str = BENCHMARK_UTILITY_VERSION) -> None:
        if utility_version not in SUPPORTED_BENCHMARK_UTILITY_VERSIONS:
            raise ValueError(f"unsupported benchmark utility version: {utility_version}")
        self.utility_version = utility_version

    def assess(self, goal: GoalVersion, item: CatalogItem) -> OracleAssessment:
        matched: list[str] = []
        violated_hard: list[str] = []
        unknown_hard: list[str] = []
        missed_soft: list[str] = []
        unknown_soft: list[str] = []
        scores: dict[str, float] = {}
        statuses: dict[str, ConstraintEvidenceStatus] = {}
        hard_scores: list[float] = []
        soft_scores: list[float] = []

        for constraint in goal.constraints:
            predicate = constraint.predicate
            status = predicate_evidence_status(item.attributes.get(constraint.slot), predicate)
            statuses[constraint.constraint_id] = status
            if predicate.importance == ConstraintImportance.HARD:
                score = float(status == ConstraintEvidenceStatus.SATISFIED)
                hard_scores.append(score)
                if status == ConstraintEvidenceStatus.UNKNOWN:
                    if self.utility_version == LEGACY_BENCHMARK_UTILITY_VERSION:
                        violated_hard.append(constraint.slot)
                    else:
                        unknown_hard.append(constraint.slot)
                elif status == ConstraintEvidenceStatus.VIOLATED:
                    violated_hard.append(constraint.slot)
                else:
                    matched.append(constraint.slot)
            else:
                score = soft_predicate_score(
                    item.attributes.get(constraint.slot),
                    predicate,
                    slot=constraint.slot,
                )
                soft_scores.append(score)
                if status == ConstraintEvidenceStatus.UNKNOWN:
                    if self.utility_version == LEGACY_BENCHMARK_UTILITY_VERSION:
                        missed_soft.append(constraint.slot)
                    else:
                        unknown_soft.append(constraint.slot)
                elif score >= 1.0:
                    matched.append(constraint.slot)
                else:
                    missed_soft.append(constraint.slot)
            scores[constraint.constraint_id] = score

        hard_fraction = sum(hard_scores) / len(hard_scores) if hard_scores else 1.0
        if not goal.constraints:
            utility = 0.0
        elif violated_hard or unknown_hard:
            utility = 0.2 * hard_fraction
        elif not soft_scores:
            utility = 1.0
        else:
            utility = 0.65 + 0.35 * (sum(soft_scores) / len(soft_scores))
        return OracleAssessment(
            item_id=item.item_id,
            normalized_utility=max(0.0, min(1.0, utility)),
            matched_slots=matched,
            violated_hard_slots=violated_hard,
            unknown_hard_slots=unknown_hard,
            missed_soft_slots=missed_soft,
            unknown_soft_slots=unknown_soft,
            constraint_scores=scores,
            constraint_statuses=statuses,
            hard_satisfied_fraction=hard_fraction,
        )

    def assess_slate(
        self,
        goal: GoalVersion,
        candidate_pool: CandidatePool,
        item_ids: list[str] | tuple[str, ...],
    ) -> BenchmarkSlateAssessment:
        assessments = tuple(
            self.assess(goal, candidate_pool.index.get(item_id))
            for item_id in item_ids
            if item_id in set(candidate_pool.item_ids)
        )
        if not assessments:
            return BenchmarkSlateAssessment(
                assessments=(),
                best_item_id=None,
                best_utility=0.0,
                accepted_item_id=None,
                all_items_hard_violating=False,
            )
        best = max(assessments, key=lambda value: (value.normalized_utility, value.item_id))
        accepted = (
            best.item_id
            if not best.violated_hard_slots
            and not best.unknown_hard_slots
            and best.normalized_utility >= goal.acceptance_threshold
            else None
        )
        return BenchmarkSlateAssessment(
            assessments=assessments,
            best_item_id=best.item_id,
            best_utility=best.normalized_utility,
            accepted_item_id=accepted,
            all_items_hard_violating=all(bool(value.violated_hard_slots) for value in assessments),
        )

    def recompute_goal_truth(
        self,
        goal: GoalVersion,
        candidate_pool: CandidatePool,
    ) -> RecomputedGoalTruth:
        feasible: list[str] = []
        acceptable: list[str] = []
        unverifiable: list[str] = []
        for item_id in candidate_pool.item_ids:
            assessment = self.assess(goal, candidate_pool.index.get(item_id))
            if assessment.violated_hard_slots:
                continue
            if assessment.unknown_hard_slots:
                unverifiable.append(item_id)
                continue
            feasible.append(item_id)
            if assessment.normalized_utility >= goal.acceptance_threshold:
                acceptable.append(item_id)
        return RecomputedGoalTruth(
            feasible_item_ids=tuple(feasible),
            acceptable_item_ids=tuple(acceptable),
            unverifiable_item_ids=tuple(unverifiable),
        )

    def validate_registered_truth(self, goal: GoalVersion, candidate_pool: CandidatePool) -> None:
        recomputed = self.recompute_goal_truth(goal, candidate_pool)
        if goal.feasible_item_count != len(recomputed.feasible_item_ids):
            raise ValueError("registered feasible item count differs from independent oracle")
        if set(goal.acceptable_item_ids) != set(recomputed.acceptable_item_ids):
            raise ValueError("registered acceptable set differs from independent oracle")
