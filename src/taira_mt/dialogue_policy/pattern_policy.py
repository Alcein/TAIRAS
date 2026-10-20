"""A transparent baseline that matches a Thought Pattern every turn."""

from __future__ import annotations

from taira_mt.constraints.predicates import public_posterior_items
from taira_mt.models import (
    BeliefState,
    DialoguePhase,
    SystemAction,
    SystemActionType,
    UserAct,
    UserObservation,
)
from taira_mt.recommender.catalog import CandidatePool
from taira_mt.recommender.oracle import rank_catalog_from_belief
from taira_mt.thought_patterns.library import ThoughtPatternLibrary


class PatternConditionedPolicy:
    def __init__(self, patterns: ThoughtPatternLibrary, candidate_pool: CandidatePool, slate_size: int = 2) -> None:
        if slate_size <= 0:
            raise ValueError("slate_size must be positive")
        self.patterns = patterns
        self.candidate_pool = candidate_pool
        self.slate_size = slate_size

    def choose_action(self, state: BeliefState) -> SystemAction:
        actual_count = len(
            public_posterior_items(
                state.known_constraints,
                self.candidate_pool.remaining_items(set(state.rejected_item_ids)),
            )
        )
        if state.available_candidate_count != actual_count:
            raise RuntimeError(
                f"belief candidate count {state.available_candidate_count} does not match pool count {actual_count}"
            )
        pattern = self.patterns.match(state)
        if pattern.preferred_action not in state.legal_actions:
            raise RuntimeError(f"matched Pattern proposes illegal action: {pattern.preferred_action}")
        if pattern.preferred_action == SystemActionType.ASK:
            explicit_targets = [target for target in pattern.information_targets if target in state.missing_slots]
            slot = explicit_targets[0] if explicit_targets else state.missing_slots[0]
            return SystemAction(
                action_type=SystemActionType.ASK,
                thought_pattern_id=pattern.pattern_id,
                question_slot=slot,
                rationale=pattern.action_template.format(slot=slot),
            )

        if pattern.preferred_action == SystemActionType.STOP:
            return SystemAction(
                action_type=SystemActionType.STOP,
                thought_pattern_id=pattern.pattern_id,
                rationale=pattern.action_template,
            )

        ranked = rank_catalog_from_belief(state, self.candidate_pool, set(state.rejected_item_ids))
        if not ranked:
            raise RuntimeError("Pattern requested recommendation/repair with no unseen candidates")
        action_type = pattern.preferred_action
        if action_type not in {SystemActionType.RECOMMEND, SystemActionType.REPAIR}:
            raise NotImplementedError(f"Pattern action is not implemented: {action_type}")
        return SystemAction(
            action_type=action_type,
            thought_pattern_id=pattern.pattern_id,
            item_ids=[item.item_id for item in ranked[: self.slate_size]],
            rationale=pattern.action_template,
        )

    def update_belief(self, state: BeliefState, action: SystemAction, observation: UserObservation) -> BeliefState:
        known = dict(state.known_constraints)
        known.update(observation.extracted_constraints)
        observed_slot_order = list(observation.extracted_constraints)
        observed_slots = set(observed_slot_order)
        dismissed = [
            slot
            for slot in state.dismissed_slots
            if slot not in observed_slots and slot not in observation.dismissed_slots
        ]
        for slot in observation.dismissed_slots:
            if slot not in observed_slots and slot not in dismissed:
                dismissed.append(slot)
            known.pop(slot, None)
        resolved = observed_slots | set(observation.dismissed_slots)
        missing = [slot for slot in state.missing_slots if slot not in resolved]
        rejected = list(state.rejected_item_ids)
        if observation.predicted_act == UserAct.REJECT:
            for item_id in action.item_ids:
                if item_id not in rejected:
                    rejected.append(item_id)
        available_candidate_count = len(
            public_posterior_items(known, self.candidate_pool.remaining_items(set(rejected)))
        )
        recovered_from_posterior_collapse = False
        if (
            available_candidate_count == 0
            and observation.predicted_act == UserAct.INFORM
            and observed_slot_order
        ):
            for slot in observed_slot_order:
                known.pop(slot, None)
            missing = list(dict.fromkeys([*observed_slot_order, *missing]))
            available_candidate_count = len(
                public_posterior_items(known, self.candidate_pool.remaining_items(set(rejected)))
            )
            recovered_from_posterior_collapse = available_candidate_count > 0

        denominator = len(known) + len(missing)
        readiness = len(known) / denominator if denominator else 1.0
        pattern = self.patterns.get(action.thought_pattern_id)
        matching_outcome = next(
            (rule for rule in pattern.outcome_rules if rule.user_act == observation.predicted_act),
            pattern.fallback_outcome,
        )

        if matching_outcome.terminal:
            phase = DialoguePhase.TERMINAL
            legal_actions: list[SystemActionType] = []
        elif recovered_from_posterior_collapse:
            phase = DialoguePhase.REPAIR
            legal_actions = self._legal_actions(phase, missing)
        elif available_candidate_count == 0:
            phase = DialoguePhase.REPAIR
            legal_actions = [SystemActionType.STOP]
        elif matching_outcome.next_phase is not None:
            phase = matching_outcome.next_phase
            legal_actions = self._legal_actions(phase, missing)
        elif missing:
            phase = DialoguePhase.ELICITATION
            legal_actions = self._legal_actions(phase, missing)
        else:
            phase = DialoguePhase.RECOMMENDATION
            legal_actions = self._legal_actions(phase, missing)

        satisfaction_delta = {
            UserAct.INFORM: -0.03,
            UserAct.ACCEPT: 0.25,
            UserAct.REJECT: -0.15,
            UserAct.REQUEST_EXPLANATION: -0.05,
            UserAct.ABANDON: -0.3,
            UserAct.OTHER: -0.08,
        }[observation.predicted_act]
        satisfaction_estimate = max(0.0, min(1.0, state.satisfaction_estimate + satisfaction_delta))

        return BeliefState(
            turn_index=state.turn_index + 1,
            phase=phase,
            known_constraints=known,
            missing_slots=missing,
            dismissed_slots=dismissed,
            uncertainty=1.0 - readiness,
            readiness=readiness,
            satisfaction_estimate=satisfaction_estimate,
            patience_estimate=max(0, state.patience_estimate - 1),
            available_candidate_count=available_candidate_count,
            last_feedback=observation.predicted_act,
            rejected_item_ids=rejected,
            legal_actions=legal_actions,
        )

    @staticmethod
    def _legal_actions(phase: DialoguePhase, missing: list[str]) -> list[SystemActionType]:
        if phase == DialoguePhase.REPAIR:
            actions = [SystemActionType.REPAIR]
            if missing:
                actions.insert(0, SystemActionType.ASK)
            return actions
        if phase == DialoguePhase.FEEDBACK:
            return [SystemActionType.RECOMMEND, SystemActionType.REPAIR]
        if phase == DialoguePhase.ELICITATION and missing:
            return [SystemActionType.ASK, SystemActionType.RECOMMEND]
        return [SystemActionType.RECOMMEND]
