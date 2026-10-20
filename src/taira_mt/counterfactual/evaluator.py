"""Generate short-horizon labels from isolated private simulator branches."""

from __future__ import annotations

import hashlib
import math
import statistics

from taira_mt.counterfactual.schemas import (
    BranchRollout,
    BranchStepTrace,
    CounterfactualActionLabel,
    CounterfactualDecisionLabel,
    CounterfactualPublicState,
    CounterfactualRewardConfig,
    PairwisePreference,
)
from taira_mt.dialogue_policy.cost_aware import (
    PATTERN_INSTANTIATOR_VERSION,
    PatternActionInstantiator,
)
from taira_mt.dialogue_policy.pattern_policy import PatternConditionedPolicy
from taira_mt.models import BeliefState, SystemAction, SystemActionType, UserAct
from taira_mt.provenance import content_sha256
from taira_mt.simulator import DeterministicUserSimulator, simulator_checkpoint_digest
from taira_mt.state import ObservationBuilder
from taira_mt.thought_patterns import ThoughtPatternLibrary


def _branch_seed(root_seed: int, replicate_index: int) -> int:
    payload = f"{root_seed}|counterfactual_common_random_number|{replicate_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _stable_sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


class CounterfactualBranchEvaluator:
    """Label public candidates using private state without changing online inputs."""

    def __init__(
        self,
        patterns: ThoughtPatternLibrary,
        instantiator: PatternActionInstantiator,
        observation_builder: ObservationBuilder,
        config: CounterfactualRewardConfig | None = None,
    ) -> None:
        self.patterns = patterns
        self.instantiator = instantiator
        self.observation_builder = observation_builder
        self.config = config or CounterfactualRewardConfig()
        self.continuation_policy = PatternConditionedPolicy(
            patterns,
            instantiator.candidate_pool,
            slate_size=instantiator.slate_size,
        )

    def evaluate(
        self,
        state: BeliefState,
        simulator: DeterministicUserSimulator,
        root_seed: int,
    ) -> tuple[CounterfactualPublicState, CounterfactualDecisionLabel]:
        candidates = self.instantiator.instantiate(state)
        checkpoint = simulator.checkpoint()
        action_labels: list[CounterfactualActionLabel] = []
        for candidate in sorted(candidates, key=lambda item: item.instantiation_id):
            rollouts = [
                self._rollout(
                    initial_state=state,
                    initial_action=candidate.action,
                    simulator=simulator.fork(
                        checkpoint,
                        branch_seed=_branch_seed(root_seed, replicate_index),
                    ),
                    instantiation_id=candidate.instantiation_id,
                    rollout_seed=_branch_seed(root_seed, replicate_index),
                )
                for replicate_index in range(self.config.rollout_replicates)
            ]
            returns = [rollout.discounted_return for rollout in rollouts]
            action_labels.append(
                CounterfactualActionLabel(
                    instantiation_id=candidate.instantiation_id,
                    pattern_id=candidate.pattern_id,
                    action_type=candidate.action.action_type.value,
                    q_mean=statistics.fmean(returns),
                    q_std=statistics.pstdev(returns),
                    accept_probability=statistics.fmean(float(rollout.accepted) for rollout in rollouts),
                    abandon_probability=statistics.fmean(float(rollout.abandoned) for rollout in rollouts),
                    hard_violation_probability=statistics.fmean(
                        float(rollout.hard_violation) for rollout in rollouts
                    ),
                    rollouts=rollouts,
                )
            )
        if simulator.checkpoint() != checkpoint:
            raise AssertionError("counterfactual evaluation mutated the parent simulator")

        ranked = sorted(action_labels, key=lambda label: (-label.q_mean, label.instantiation_id))
        best = ranked[0]
        regret = {label.instantiation_id: best.q_mean - label.q_mean for label in action_labels}
        pairwise = self._pairwise_preferences(action_labels)
        ask_labels = [label for label in action_labels if label.action_type == SystemActionType.ASK.value]
        non_ask_labels = [label for label in action_labels if label.action_type != SystemActionType.ASK.value]
        ask_comparison_available = bool(ask_labels and non_ask_labels)
        if ask_comparison_available:
            best_non_ask = max(label.q_mean for label in non_ask_labels)
            net_voc = {label.instantiation_id: label.q_mean - best_non_ask for label in ask_labels}
        else:
            net_voc = {}
        if ask_comparison_available:
            should_ask = max(net_voc.values()) > self.config.ask_margin
        else:
            should_ask = bool(ask_labels) and not non_ask_labels
        if not ask_labels:
            readiness = 1.0
        elif not non_ask_labels:
            readiness = 0.0
        else:
            best_ask = max(label.q_mean for label in ask_labels)
            readiness = _stable_sigmoid((best_non_ask - best_ask) / self.config.readiness_temperature)

        public = CounterfactualPublicState(
            belief_hash=content_sha256(state.model_dump(mode="json")),
            candidate_pool_id=self.instantiator.candidate_pool.pool_id,
            pattern_library_hash=self.patterns.content_hash,
            tracker_name=self.observation_builder.name,
            instantiator_version=PATTERN_INSTANTIATOR_VERSION,
            belief=state,
            candidates=candidates,
        )
        private = CounterfactualDecisionLabel(
            belief_hash=public.belief_hash,
            candidate_pool_id=public.candidate_pool_id,
            root_seed=root_seed,
            reward_config_hash=content_sha256(self.config.model_dump(mode="json")),
            simulator_checkpoint_digest=simulator_checkpoint_digest(checkpoint),
            action_labels=action_labels,
            best_instantiation_id=best.instantiation_id,
            best_pattern_id=best.pattern_id,
            regret_by_instantiation=regret,
            pairwise_preferences=pairwise,
            net_voc_by_ask_instantiation=net_voc,
            ask_comparison_available=ask_comparison_available,
            should_ask=should_ask,
            readiness_label=readiness,
        )
        return public, private

    def _rollout(
        self,
        initial_state: BeliefState,
        initial_action: SystemAction,
        simulator: DeterministicUserSimulator,
        instantiation_id: str,
        rollout_seed: int,
    ) -> BranchRollout:
        state = initial_state.model_copy(deep=True)
        action = initial_action
        steps: list[BranchStepTrace] = []
        accepted = False
        abandoned = False
        hard_violation_any = False
        total_return = 0.0
        for step_index in range(self.config.horizon):
            satisfaction_before = simulator.runtime.satisfaction
            response, assessments = simulator.respond(action)
            satisfaction_delta = response.latent_satisfaction - satisfaction_before
            all_hard_violations = bool(assessments) and all(
                bool(assessment.violated_hard_slots) for assessment in assessments
            )
            hard_violation_any = hard_violation_any or all_hard_violations
            accepted_now = response.act == UserAct.ACCEPT
            abandoned_now = response.act == UserAct.ABANDON and action.action_type != SystemActionType.STOP
            accepted = accepted or accepted_now
            abandoned = abandoned or abandoned_now
            acceptance_component = self.config.acceptance_bonus if accepted_now else 0.0
            abandonment_component = -self.config.abandonment_penalty if abandoned_now else 0.0
            hard_component = -self.config.hard_violation_penalty if all_hard_violations else 0.0
            turn_component = -self.config.turn_cost
            action_component = -self._action_cost(action.action_type)
            reward = sum(
                [
                    satisfaction_delta,
                    acceptance_component,
                    abandonment_component,
                    hard_component,
                    turn_component,
                    action_component,
                ]
            )
            discounted = (self.config.discount**step_index) * reward
            total_return += discounted
            steps.append(
                BranchStepTrace(
                    step_index=step_index,
                    action=action,
                    true_user_act=response.act,
                    satisfaction_delta=satisfaction_delta,
                    acceptance_component=acceptance_component,
                    abandonment_component=abandonment_component,
                    hard_violation_component=hard_component,
                    turn_cost_component=turn_component,
                    action_cost_component=action_component,
                    undiscounted_reward=reward,
                    discounted_reward=discounted,
                )
            )
            observation = self.observation_builder.build(state, action, response.text)
            state = self.continuation_policy.update_belief(state, action, observation)
            if response.terminal or step_index + 1 >= self.config.horizon:
                break
            action = self.continuation_policy.choose_action(state)
        return BranchRollout(
            instantiation_id=instantiation_id,
            rollout_seed=rollout_seed,
            steps=steps,
            discounted_return=total_return,
            accepted=accepted,
            abandoned=abandoned,
            hard_violation=hard_violation_any,
        )

    def _action_cost(self, action_type: SystemActionType) -> float:
        return {
            SystemActionType.ASK: self.config.ask_cost,
            SystemActionType.REPAIR: self.config.repair_cost,
            SystemActionType.STOP: self.config.stop_cost,
            SystemActionType.RECOMMEND: 0.0,
            SystemActionType.EXPLAIN: 0.0,
        }[action_type]

    @staticmethod
    def _pairwise_preferences(labels: list[CounterfactualActionLabel]) -> list[PairwisePreference]:
        preferences: list[PairwisePreference] = []
        ordered = sorted(labels, key=lambda label: label.instantiation_id)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                if left.q_mean == right.q_mean:
                    continue
                preferred, dispreferred = (left, right) if left.q_mean > right.q_mean else (right, left)
                preferences.append(
                    PairwisePreference(
                        preferred_instantiation_id=preferred.instantiation_id,
                        dispreferred_instantiation_id=dispreferred.instantiation_id,
                        value_margin=preferred.q_mean - dispreferred.q_mean,
                    )
                )
        return preferences
