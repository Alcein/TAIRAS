"""Deterministic matching baseline for State-aware Thought Patterns."""

from __future__ import annotations

import hashlib
import json

from taira_mt.models import BeliefState, DialoguePhase, SystemActionType, UserAct
from taira_mt.thought_patterns.models import PatternOutcome, StateAwareThoughtPattern


class ThoughtPatternLibrary:
    def __init__(self, patterns: list[StateAwareThoughtPattern]) -> None:
        if not patterns:
            raise ValueError("at least one Thought Pattern is required")
        ids = [pattern.pattern_id for pattern in patterns]
        if len(ids) != len(set(ids)):
            raise ValueError("Thought Pattern IDs must be unique")
        self._patterns = patterns

    @property
    def patterns(self) -> tuple[StateAwareThoughtPattern, ...]:
        return tuple(self._patterns)

    @property
    def content_hash(self) -> str:
        payload = [
            pattern.model_dump(mode="json")
            for pattern in sorted(self._patterns, key=lambda item: item.pattern_id)
        ]
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def applicable(self, state: BeliefState) -> tuple[StateAwareThoughtPattern, ...]:
        candidates: list[tuple[float, StateAwareThoughtPattern]] = []
        for pattern in self._patterns:
            if state.phase not in pattern.applicable_phases:
                continue
            if pattern.requires_missing_slot and not state.missing_slots:
                continue
            if pattern.requires_no_candidates and state.available_candidate_count != 0:
                continue
            if pattern.required_feedback and state.last_feedback not in pattern.required_feedback:
                continue
            if not pattern.min_readiness <= state.readiness <= pattern.max_readiness:
                continue
            if pattern.preferred_action not in state.legal_actions:
                continue
            score = float(pattern.priority) + pattern.expected_utility
            score += 1.0
            candidates.append((score, pattern))
        if not candidates:
            raise LookupError(f"no Thought Pattern applies to state at turn {state.turn_index}")
        candidates.sort(key=lambda pair: (pair[0], pair[1].pattern_id), reverse=True)
        return tuple(pattern for _, pattern in candidates)

    def match(self, state: BeliefState) -> StateAwareThoughtPattern:
        return self.applicable(state)[0]

    def get(self, pattern_id: str) -> StateAwareThoughtPattern:
        for pattern in self._patterns:
            if pattern.pattern_id == pattern_id:
                return pattern
        raise KeyError(f"unknown Thought Pattern: {pattern_id}")


def default_pattern_library() -> ThoughtPatternLibrary:
    patterns = [
        StateAwareThoughtPattern(
            pattern_id="clarify_missing_constraint",
            name="Clarify the highest-value missing constraint",
            task_description="Resolve material uncertainty before committing to a recommendation.",
            state_description="At least one public-ontology slot is uncertain and asking remains legal.",
            solution_description="Ask only the highest-priority missing slot, then update the belief.",
            thought_template=[
                "inspect_belief_uncertainty",
                "select_highest_value_missing_slot",
                "ask_one_focused_question",
                "apply_observed_answer",
                "rematch_thought_pattern",
            ],
            applicable_phases=[DialoguePhase.ELICITATION, DialoguePhase.FEEDBACK, DialoguePhase.REPAIR],
            preferred_action=SystemActionType.ASK,
            information_targets=["highest_value_missing_slot"],
            requires_missing_slot=True,
            max_readiness=0.99,
            priority=30,
            action_template="Ask one concise question about {slot}.",
            outcome_rules=[
                PatternOutcome(user_act=UserAct.INFORM),
                PatternOutcome(user_act=UserAct.ABANDON, next_phase=DialoguePhase.TERMINAL, terminal=True),
            ],
            fallback_outcome=PatternOutcome(user_act=UserAct.OTHER, next_phase=DialoguePhase.ELICITATION),
            expected_utility=0.7,
        ),
        StateAwareThoughtPattern(
            pattern_id="recommend_under_uncertainty",
            name="Recommend under unresolved uncertainty",
            task_description=(
                "Test whether acting now is preferable to paying for another clarification turn."
            ),
            state_description=(
                "Recommendation remains legal, but the public belief is not yet decision-ready."
            ),
            solution_description=(
                "Rank a compact public-evidence slate while retaining unresolved constraints."
            ),
            thought_template=[
                "inspect_current_public_evidence",
                "retain_unresolved_constraint_risk",
                "rank_candidates_without_private_information",
                "recommend_compact_probe_slate",
                "observe_acceptance_or_corrective_feedback",
            ],
            applicable_phases=[DialoguePhase.ELICITATION],
            preferred_action=SystemActionType.RECOMMEND,
            requires_missing_slot=True,
            max_readiness=0.5,
            priority=10,
            action_template=(
                "Recommend a compact probe slate using only current public evidence."
            ),
            outcome_rules=[
                PatternOutcome(user_act=UserAct.INFORM, next_phase=DialoguePhase.ELICITATION),
                PatternOutcome(
                    user_act=UserAct.ACCEPT,
                    next_phase=DialoguePhase.TERMINAL,
                    terminal=True,
                ),
                PatternOutcome(user_act=UserAct.REJECT, next_phase=DialoguePhase.REPAIR),
                PatternOutcome(
                    user_act=UserAct.REQUEST_EXPLANATION,
                    next_phase=DialoguePhase.ELICITATION,
                ),
                PatternOutcome(
                    user_act=UserAct.ABANDON,
                    next_phase=DialoguePhase.TERMINAL,
                    terminal=True,
                ),
            ],
            fallback_outcome=PatternOutcome(
                user_act=UserAct.OTHER,
                next_phase=DialoguePhase.ELICITATION,
            ),
            expected_utility=0.2,
            provenance="expert_counterfactual_probe_v1",
        ),
        StateAwareThoughtPattern(
            pattern_id="repair_inconsistent_public_belief",
            name="Repair after a posterior-collapsing user statement",
            task_description=(
                "Keep the dialogue recoverable when one unconfirmed statement removes every candidate."
            ),
            state_description=(
                "The public tracker rolled back a posterior-collapsing INFORM and entered repair."
            ),
            solution_description=(
                "Retain a public-evidence fallback slate while the suspect constraint is confirmed."
            ),
            thought_template=[
                "detect_posterior_collapse",
                "rollback_unconfirmed_constraint",
                "retain_suspect_slot_for_confirmation",
                "rank_candidates_from_last_consistent_belief",
                "compare_confirmation_with_repair_action",
            ],
            applicable_phases=[DialoguePhase.REPAIR],
            preferred_action=SystemActionType.REPAIR,
            required_feedback=[UserAct.INFORM],
            requires_missing_slot=True,
            priority=25,
            action_template=(
                "Use the last consistent public belief while confirming the conflicting statement."
            ),
            outcome_rules=[
                PatternOutcome(
                    user_act=UserAct.ACCEPT,
                    next_phase=DialoguePhase.TERMINAL,
                    terminal=True,
                ),
                PatternOutcome(user_act=UserAct.REJECT, next_phase=DialoguePhase.REPAIR),
                PatternOutcome(user_act=UserAct.INFORM, next_phase=DialoguePhase.ELICITATION),
                PatternOutcome(
                    user_act=UserAct.ABANDON,
                    next_phase=DialoguePhase.TERMINAL,
                    terminal=True,
                ),
            ],
            fallback_outcome=PatternOutcome(
                user_act=UserAct.OTHER,
                next_phase=DialoguePhase.ELICITATION,
            ),
            expected_utility=0.4,
            provenance="expert_posterior_recovery_v1",
        ),
        StateAwareThoughtPattern(
            pattern_id="repair_after_rejection",
            name="Repair after negative recommendation feedback",
            task_description="Recover from a rejected recommendation without repeating failed options.",
            state_description="The latest observed act is rejection and repair is legal.",
            solution_description="Exclude rejected evidence, apply new constraints, and produce a revised slate.",
            thought_template=[
                "attribute_negative_feedback",
                "update_constraint_belief",
                "exclude_rejected_items",
                "rank_unseen_candidates",
                "recommend_repaired_slate",
            ],
            applicable_phases=[DialoguePhase.FEEDBACK, DialoguePhase.REPAIR],
            preferred_action=SystemActionType.REPAIR,
            required_feedback=[UserAct.REJECT],
            priority=40,
            action_template="Remove rejected evidence, resolve the conflict, and retry.",
            outcome_rules=[
                PatternOutcome(user_act=UserAct.ACCEPT, next_phase=DialoguePhase.TERMINAL, terminal=True),
                PatternOutcome(user_act=UserAct.REJECT, next_phase=DialoguePhase.REPAIR),
                PatternOutcome(user_act=UserAct.ABANDON, next_phase=DialoguePhase.TERMINAL, terminal=True),
            ],
            fallback_outcome=PatternOutcome(user_act=UserAct.OTHER, next_phase=DialoguePhase.REPAIR),
            expected_utility=0.6,
        ),
        StateAwareThoughtPattern(
            pattern_id="recommend_when_ready",
            name="Recommend when the belief is decision-ready",
            task_description="Recommend when further clarification has insufficient net value.",
            state_description="Recommendation is legal and the belief meets the readiness threshold.",
            solution_description="Rank unseen catalog items using only observable belief evidence.",
            thought_template=[
                "estimate_recommendation_readiness",
                "compare_clarification_and_recommendation_value",
                "rank_unseen_candidates",
                "recommend_compact_slate",
                "observe_user_feedback",
            ],
            applicable_phases=[
                DialoguePhase.ELICITATION,
                DialoguePhase.RECOMMENDATION,
                DialoguePhase.FEEDBACK,
                DialoguePhase.REPAIR,
            ],
            preferred_action=SystemActionType.RECOMMEND,
            min_readiness=0.5,
            priority=20,
            action_template="Recommend a compact slate with evidence grounded in known constraints.",
            outcome_rules=[
                PatternOutcome(user_act=UserAct.ACCEPT, next_phase=DialoguePhase.TERMINAL, terminal=True),
                PatternOutcome(user_act=UserAct.REJECT, next_phase=DialoguePhase.REPAIR),
                PatternOutcome(user_act=UserAct.REQUEST_EXPLANATION, next_phase=DialoguePhase.FEEDBACK),
                PatternOutcome(user_act=UserAct.ABANDON, next_phase=DialoguePhase.TERMINAL, terminal=True),
            ],
            fallback_outcome=PatternOutcome(user_act=UserAct.OTHER, next_phase=DialoguePhase.FEEDBACK),
            expected_utility=0.5,
        ),
        StateAwareThoughtPattern(
            pattern_id="stop_when_candidates_exhausted",
            name="Stop when no unseen catalog candidates remain",
            task_description="Terminate safely when every catalog candidate has been exhausted.",
            state_description="No unseen candidate remains and stop is the only legal action.",
            solution_description="Explain candidate exhaustion and stop instead of repeating a rejection.",
            thought_template=[
                "verify_candidate_exhaustion",
                "avoid_repeating_rejected_items",
                "report_no_unseen_candidates",
                "terminate_dialogue",
            ],
            applicable_phases=[DialoguePhase.FEEDBACK, DialoguePhase.REPAIR, DialoguePhase.RECOMMENDATION],
            preferred_action=SystemActionType.STOP,
            requires_no_candidates=True,
            priority=100,
            action_template="Explain that no unseen compatible candidate remains and stop.",
            outcome_rules=[
                PatternOutcome(user_act=UserAct.ABANDON, next_phase=DialoguePhase.TERMINAL, terminal=True),
            ],
            fallback_outcome=PatternOutcome(user_act=UserAct.OTHER, next_phase=DialoguePhase.TERMINAL, terminal=True),
            expected_utility=0.0,
        ),
    ]
    return ThoughtPatternLibrary(patterns)
