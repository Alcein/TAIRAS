"""Offline dialogue runner with complete per-turn provenance."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from taira_mt.dialogue_policy.pattern_policy import PatternConditionedPolicy
from taira_mt.models import (
    BeliefState,
    DialogueEpisode,
    EvaluationTrace,
    SystemActionType,
    TerminationReason,
    TurnRecord,
    UserAct,
)
from taira_mt.simulator.deterministic import DeterministicUserSimulator
from taira_mt.state.observation import ObservationBuilder


def _public_value_lifecycle(policy: PatternConditionedPolicy) -> tuple[Callable[..., Any], Callable[..., Any]] | None:
    """Discover an optional public-only lifecycle on a CostAware value model.

    The runner deliberately passes text that has already been rendered for the
    user, rather than episode IDs, hidden goals, simulator traces, or raw item
    identifiers.  This keeps learned online models structurally outside the
    evaluator/private channel while leaving ordinary policies unchanged.
    """

    value_model = getattr(policy, "value_model", None)
    begin_episode = getattr(value_model, "begin_episode", None)
    record_public_turn = getattr(value_model, "record_public_turn", None)
    if begin_episode is None and record_public_turn is None:
        return None
    if not callable(begin_episode) or not callable(record_public_turn):
        raise TypeError("a value-model public lifecycle must define both begin_episode and record_public_turn")
    return begin_episode, record_public_turn


def render_public_system_text(policy: PatternConditionedPolicy, action) -> str:
    """Render the same visible system surface used by offline state features."""

    if action.action_type == SystemActionType.ASK:
        return f"System asks about {action.question_slot}."
    if action.action_type in {SystemActionType.RECOMMEND, SystemActionType.REPAIR}:
        titles = [policy.candidate_pool.index.get(item_id).title for item_id in action.item_ids]
        verb = "recommends" if action.action_type == SystemActionType.RECOMMEND else "revises the recommendation to"
        return f"System {verb}: {'; '.join(titles)}."
    if action.action_type == SystemActionType.STOP:
        return "System reports that no unseen candidate remains."
    return "System provides an explanation."


def run_episode(
    episode_id: str,
    seed: int,
    initial_belief: BeliefState,
    policy: PatternConditionedPolicy,
    simulator: DeterministicUserSimulator,
    observation_builder: ObservationBuilder,
    max_turns: int = 8,
    before_decision_hook: Callable[[BeliefState, DeterministicUserSimulator], None] | None = None,
    initial_public_turns: Sequence[tuple[str, str]] | None = None,
) -> DialogueEpisode:
    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if simulator.seed != seed:
        raise ValueError(f"episode seed {seed} does not match simulator seed {simulator.seed}")
    lifecycle = _public_value_lifecycle(policy)
    if lifecycle is not None:
        begin_episode, record_public_turn = lifecycle
        begin_episode(
            domain=simulator.goal.domain,
            public_initial_request=simulator.goal.initial_request,
        )
        for public_system_text, public_user_text in initial_public_turns or ():
            # A repair prelude was already visible to the user before this
            # measured episode starts.  Preserve that public evidence for an
            # online state-conditioned model, while accepting only rendered
            # text (never IDs, hidden state, or evaluator traces).
            record_public_turn(
                public_system_text=public_system_text,
                public_user_text=public_user_text,
            )
    state = initial_belief
    turns: list[TurnRecord] = []
    termination_reason = TerminationReason.MAX_TURNS

    for _ in range(max_turns):
        if before_decision_hook is not None:
            before_decision_hook(state.model_copy(deep=True), simulator)
        belief_before = state.model_copy(deep=True)
        action = policy.choose_action(state)
        latent_satisfaction_before = simulator.runtime.satisfaction
        response, assessments = simulator.respond(action)
        observation = observation_builder.build(state, action, response.text)
        state = policy.update_belief(state, action, observation)
        if lifecycle is not None:
            _, record_public_turn = lifecycle
            record_public_turn(
                public_system_text=render_public_system_text(policy, action),
                public_user_text=response.text,
            )
        reward = response.latent_satisfaction - latent_satisfaction_before
        turns.append(
            TurnRecord(
                turn_index=belief_before.turn_index,
                belief_before=belief_before,
                matched_thought_pattern_id=action.thought_pattern_id,
                system_action=action,
                public_user_text=response.text,
                policy_observation=observation,
                belief_after=state.model_copy(deep=True),
                reward=reward,
                evaluation_trace=EvaluationTrace(
                    true_user_act=response.act,
                    true_revealed_constraints=response.revealed_constraints,
                    true_dismissed_slots=response.dismissed_slots,
                    accepted_item_id=response.accepted_item_id,
                    latent_satisfaction=response.latent_satisfaction,
                    terminal=response.terminal,
                    reason=response.reason,
                    oracle_assessments=assessments,
                ),
            )
        )
        if response.terminal:
            if response.act == UserAct.ACCEPT:
                termination_reason = TerminationReason.ACCEPTED
            elif action.action_type == SystemActionType.STOP:
                termination_reason = TerminationReason.NO_CANDIDATES
            else:
                termination_reason = TerminationReason.ABANDONED
            break

    success = termination_reason == TerminationReason.ACCEPTED
    ask_count = sum(turn.system_action.action_type.value == "ask" for turn in turns)
    recommend_count = sum(turn.system_action.action_type.value == "recommend" for turn in turns)
    repair_count = sum(turn.system_action.action_type.value == "repair" for turn in turns)
    stop_count = sum(turn.system_action.action_type.value == "stop" for turn in turns)
    return DialogueEpisode(
        episode_id=episode_id,
        seed=seed,
        domain=simulator.goal.domain,
        goal_id=simulator.goal.goal_id,
        persona_id=simulator.persona.persona_id,
        public_initial_request=simulator.goal.initial_request,
        candidate_pool_id=policy.candidate_pool.pool_id,
        turns=turns,
        termination_reason=termination_reason,
        success=success,
        metrics={
            "turn_count": len(turns),
            "ask_count": ask_count,
            "recommend_count": recommend_count,
            "repair_count": repair_count,
            "stop_count": stop_count,
            "final_latent_satisfaction": simulator.runtime.satisfaction,
            "final_satisfaction_estimate": state.satisfaction_estimate,
            "success": success,
        },
    )
