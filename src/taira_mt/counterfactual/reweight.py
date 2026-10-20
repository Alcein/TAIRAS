"""Post-hoc reward sensitivity over persisted counterfactual rollouts.

The branch trajectories are fixed by the simulator and continuation policy.  This
module only changes the linear utility assigned to those already persisted
trajectories; it never changes an action, response, or transition.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from typing import Any

from taira_mt.counterfactual.schemas import CounterfactualRewardConfig


def reweighted_rollout_return(
    rollout: Mapping[str, Any],
    config: CounterfactualRewardConfig,
) -> float:
    """Recompute one fixed rollout's return under ``config``."""

    total = 0.0
    for step in rollout["steps"]:
        action_type = step["action"]["action_type"]
        user_act = step["true_user_act"]
        accepted = user_act == "accept"
        abandoned = user_act == "abandon" and action_type != "stop"
        # v3 traces predate an explicit boolean field. A non-zero persisted
        # component is lossless because the generating config used a positive
        # hard-violation penalty.
        hard_violation = bool(step.get("hard_violation", False)) or float(
            step["hard_violation_component"]
        ) < 0.0
        action_cost = {
            "ask": config.ask_cost,
            "repair": config.repair_cost,
            "stop": config.stop_cost,
            "recommend": 0.0,
            "explain": 0.0,
        }[action_type]
        reward = (
            float(step["satisfaction_delta"])
            + (config.acceptance_bonus if accepted else 0.0)
            - (config.abandonment_penalty if abandoned else 0.0)
            - (config.hard_violation_penalty if hard_violation else 0.0)
            - config.turn_cost
            - action_cost
        )
        total += (config.discount ** int(step["step_index"])) * reward
    return total


def reweighted_action_values(
    action_labels: Iterable[Mapping[str, Any]],
    config: CounterfactualRewardConfig,
) -> list[dict[str, Any]]:
    """Return deterministic action-level Q statistics for a fixed state."""

    values: list[dict[str, Any]] = []
    for label in action_labels:
        returns = [reweighted_rollout_return(rollout, config) for rollout in label["rollouts"]]
        values.append(
            {
                "instantiation_id": label["instantiation_id"],
                "pattern_id": label["pattern_id"],
                "action_type": label["action_type"],
                "q_mean": statistics.fmean(returns),
                "q_std": statistics.pstdev(returns),
            }
        )
    return values


def summarize_reweighted_state(
    action_labels: Iterable[Mapping[str, Any]],
    config: CounterfactualRewardConfig,
) -> dict[str, Any]:
    """Summarize the best action and ask-vs-non-ask comparison for one state."""

    values = reweighted_action_values(action_labels, config)
    if not values:
        raise ValueError("counterfactual state has no action labels")
    best = sorted(values, key=lambda item: (-item["q_mean"], item["instantiation_id"]))[0]
    asks = [item for item in values if item["action_type"] == "ask"]
    non_asks = [item for item in values if item["action_type"] != "ask"]
    comparison_available = bool(asks and non_asks)
    gap = None
    if comparison_available:
        gap = max(item["q_mean"] for item in asks) - max(item["q_mean"] for item in non_asks)
        should_ask = gap > config.ask_margin
    else:
        should_ask = bool(asks) and not non_asks
    return {
        "best_action_type": best["action_type"],
        "best_pattern_id": best["pattern_id"],
        "best_instantiation_id": best["instantiation_id"],
        "ask_comparison_available": comparison_available,
        "ask_q_gap": gap,
        "should_ask": should_ask,
        "action_values": values,
    }
