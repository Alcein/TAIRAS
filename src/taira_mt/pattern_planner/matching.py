"""DSUS-conditioned rank-all matching for a compact Pattern Skill library."""

from __future__ import annotations

import math
from dataclasses import dataclass

from taira_mt.dsus.schemas import DSUSPosterior, DSUSStateLabel
from taira_mt.models import StrictModel
from taira_mt.pattern_planner.schemas import PatternSkill, StateConditionedPatternLibrary


class PatternMatch(StrictModel):
    pattern_id: str
    structural_probability: float
    evidence_terms: dict[str, float]
    rank: int


@dataclass(frozen=True)
class PatternMatchResult:
    """A deterministic rank-all result; Top-K is an execution budget, not a claim."""

    matches: tuple[PatternMatch, ...]

    @property
    def best(self) -> PatternMatch:
        if not self.matches:
            raise ValueError("Pattern match result is empty")
        return self.matches[0]

    def selected(self, top_k: int | None = None) -> tuple[PatternMatch, ...]:
        if top_k is None:
            return self.matches
        if top_k < 1:
            raise ValueError("Pattern top_k must be positive")
        return self.matches[:top_k]


def _allowed_probability(
    distribution: dict[str, float], allowed: list[object]
) -> float:
    if not allowed:
        return 1.0
    return sum(distribution.get(getattr(item, "value", str(item)), 0.0) for item in allowed)


def pattern_structural_probability(
    pattern: PatternSkill,
    posterior: DSUSPosterior,
) -> tuple[float, dict[str, float]]:
    """Compute expected public applicability under the factorized DSUS posterior."""

    applicability = pattern.applicability
    evidence = {
        "phase": _allowed_probability(posterior.dialogue_phase, applicability.allowed_phases),
        "completeness": _allowed_probability(
            posterior.intent_completeness, applicability.allowed_completeness
        ),
        "stability": _allowed_probability(posterior.intent_stability, applicability.allowed_stability),
        "feasibility": _allowed_probability(
            posterior.candidate_feasibility, applicability.allowed_feasibility
        ),
        "last_user_act": _allowed_probability(posterior.last_user_act, applicability.allowed_last_user_acts),
        "patience_risk": _allowed_probability(posterior.patience_risk, applicability.allowed_patience_risks),
    }
    if applicability.requires_explanation_pending is None:
        evidence["explanation_pending"] = 1.0
    else:
        expected = str(applicability.requires_explanation_pending).lower()
        evidence["explanation_pending"] = posterior.explanation_pending.get(expected, 0.0)
    active_terms = [
        value
        for name, value in evidence.items()
        if (
            name == "phase" and applicability.allowed_phases
            or name == "completeness" and applicability.allowed_completeness
            or name == "stability" and applicability.allowed_stability
            or name == "feasibility" and applicability.allowed_feasibility
            or name == "last_user_act" and applicability.allowed_last_user_acts
            or name == "patience_risk" and applicability.allowed_patience_risks
            or name == "explanation_pending" and applicability.requires_explanation_pending is not None
        )
    ]
    if not active_terms:
        return 1.0, evidence
    # Geometric aggregation prevents a broad Pattern from winning merely by
    # having fewer predicates while retaining uncertainty rather than a hard
    # one-label gate.
    score = math.prod(max(value, 1e-12) for value in active_terms) ** (1.0 / len(active_terms))
    return score, evidence


def match_patterns(
    library: StateConditionedPatternLibrary,
    posterior: DSUSPosterior,
) -> PatternMatchResult:
    """Rank every compact-library Skill deterministically by DSUS compatibility."""

    scored = []
    for pattern in library.patterns:
        probability, evidence = pattern_structural_probability(pattern, posterior)
        scored.append((pattern.pattern_id, probability, evidence))
    ordered = sorted(scored, key=lambda item: (-item[1], item[0]))
    return PatternMatchResult(
        matches=tuple(
            PatternMatch(
                pattern_id=pattern_id,
                structural_probability=probability,
                evidence_terms=evidence,
                rank=index,
            )
            for index, (pattern_id, probability, evidence) in enumerate(ordered, start=1)
        )
    )


def applicable_patterns_from_label(
    library: StateConditionedPatternLibrary,
    label: DSUSStateLabel,
) -> tuple[PatternSkill, ...]:
    """Evaluator-only coverage helper used in development diagnostics."""

    return tuple(pattern for pattern in library.patterns if pattern.applicability.matches(label))


def primary_action_step(pattern: PatternSkill):
    """Return the first executable action in a Skill's SOP, if it has one."""

    return next((step for step in pattern.sop_steps if step.system_action_type is not None), None)
