"""Keep public ranking separate from hidden-goal evaluation."""

from __future__ import annotations

from taira_mt.constraints.predicates import (
    predicate_evidence_status,
    public_posterior_items,
    satisfies_predicate,
    soft_predicate_score,
)
from taira_mt.models import (
    AttributeValue,
    BeliefState,
    CatalogItem,
    Constraint,
    ConstraintEvidenceStatus,
    ConstraintImportance,
    HiddenGoal,
    OracleAssessment,
)
from taira_mt.recommender.catalog import CandidatePool


def satisfies_constraint(attribute: AttributeValue | None, constraint: Constraint) -> bool:
    """Backward-compatible name for the shared predicate evaluator."""

    return satisfies_predicate(attribute, constraint)


def posterior_catalog_from_belief(
    state: BeliefState,
    candidate_pool: CandidatePool,
    excluded_item_ids: set[str] | None = None,
) -> list[CatalogItem]:
    """Return the public hard-filtered posterior, soft-ranked stably."""

    remaining = candidate_pool.remaining_items(excluded_item_ids or set())
    return public_posterior_items(state.known_constraints, remaining)


def rank_catalog_from_belief(
    state: BeliefState,
    candidate_pool: CandidatePool,
    excluded_item_ids: set[str] | None = None,
) -> list[CatalogItem]:
    """Rank using observable predicates only; hidden constraints are unavailable."""

    return posterior_catalog_from_belief(state, candidate_pool, excluded_item_ids)


class IndependentCompatibilityOracle:
    """Evaluate catalog attributes against the hidden goal independently."""

    def assess(self, goal: HiddenGoal, item: CatalogItem) -> OracleAssessment:
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
            key = (
                constraint.slot
                if constraint.target_node_id == "root"
                else f"{constraint.target_node_id}:{constraint.slot}"
            )
            status = predicate_evidence_status(item.attributes.get(constraint.slot), constraint)
            statuses[key] = status
            if constraint.importance == ConstraintImportance.HARD:
                score = float(status == ConstraintEvidenceStatus.SATISFIED)
                hard_scores.append(score)
                if status == ConstraintEvidenceStatus.UNKNOWN:
                    unknown_hard.append(constraint.slot)
                elif status == ConstraintEvidenceStatus.VIOLATED:
                    violated_hard.append(constraint.slot)
                else:
                    matched.append(constraint.slot)
            else:
                score = soft_predicate_score(
                    item.attributes.get(constraint.slot),
                    constraint,
                    slot=constraint.slot,
                )
                soft_scores.append(score)
                if status == ConstraintEvidenceStatus.UNKNOWN:
                    unknown_soft.append(constraint.slot)
                elif score >= 1.0:
                    matched.append(constraint.slot)
                else:
                    missed_soft.append(constraint.slot)
            scores[key] = score

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
