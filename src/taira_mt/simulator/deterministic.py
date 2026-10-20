"""A deterministic simulator policy; an LLM may later realize text only."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from random import Random
from typing import Any

from taira_mt.constraints.predicates import satisfies_predicate
from taira_mt.constraints.rendering import render_constraint
from taira_mt.models import (
    ConstraintOperator,
    ConstraintPredicate,
    HiddenGoal,
    OracleAssessment,
    SystemAction,
    SystemActionType,
    UserAct,
    UserPersona,
    UserResponse,
)
from taira_mt.provenance import content_sha256
from taira_mt.recommender.catalog import CatalogIndex
from taira_mt.recommender.oracle import IndependentCompatibilityOracle


@dataclass
class SimulatorRuntime:
    patience_remaining: int
    satisfaction: float = 0.5
    revealed_slots: set[str] = field(default_factory=set)
    rejected_item_ids: set[str] = field(default_factory=set)
    pending_corrections: dict[str, ConstraintPredicate] = field(default_factory=dict)
    terminal: bool = False


@dataclass(frozen=True)
class SimulatorCheckpoint:
    """Complete branchable simulator state, including the random stream."""

    runtime: SimulatorRuntime
    rng_state: tuple[Any, ...]
    reveal_order: tuple[str, ...] | None


def simulator_checkpoint_digest(checkpoint: SimulatorCheckpoint) -> str:
    """Hash all private transition state needed to reproduce a branch."""

    runtime = checkpoint.runtime
    payload = {
        "patience_remaining": runtime.patience_remaining,
        "satisfaction": runtime.satisfaction,
        "revealed_slots": sorted(runtime.revealed_slots),
        "rejected_item_ids": sorted(runtime.rejected_item_ids),
        "pending_corrections": {
            slot: predicate.model_dump(mode="json")
            for slot, predicate in sorted(runtime.pending_corrections.items())
        },
        "terminal": runtime.terminal,
        "rng_state": checkpoint.rng_state,
        "reveal_order": checkpoint.reveal_order,
    }
    return content_sha256(payload)


class DeterministicUserSimulator:
    def __init__(
        self,
        goal: HiddenGoal,
        persona: UserPersona,
        catalog_index: CatalogIndex,
        oracle: IndependentCompatibilityOracle,
        seed: int = 0,
        reveal_order: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.goal = goal
        self.persona = persona
        self.catalog_index = catalog_index
        initially_revealed = {constraint.slot for constraint in goal.constraints if constraint.initially_revealed}
        self.runtime = SimulatorRuntime(
            patience_remaining=persona.patience_turns,
            revealed_slots=initially_revealed,
        )
        self.oracle = oracle
        self.seed = seed
        self.rng = Random(seed)
        self.reveal_order = self._validate_reveal_order(reveal_order)

    def _validate_reveal_order(
        self, reveal_order: tuple[str, ...] | list[str] | None
    ) -> tuple[str, ...] | None:
        """Validate a pre-registered private reveal plan when one is supplied."""

        if reveal_order is None:
            return None
        normalized = tuple(reveal_order)
        hidden_unrevealed = {
            constraint.slot
            for constraint in self.goal.constraints
            if not constraint.initially_revealed
        }
        if len(normalized) != len(set(normalized)):
            raise ValueError("reveal_order slots must be unique")
        if set(normalized) != hidden_unrevealed:
            raise ValueError("reveal_order must cover exactly the initially hidden goal slots")
        return normalized

    def checkpoint(self) -> SimulatorCheckpoint:
        return SimulatorCheckpoint(
            runtime=copy.deepcopy(self.runtime),
            rng_state=self.rng.getstate(),
            reveal_order=self.reveal_order,
        )

    def restore(self, checkpoint: SimulatorCheckpoint) -> None:
        if checkpoint.reveal_order != self.reveal_order:
            raise ValueError("cannot restore a checkpoint from a different reveal plan")
        self.runtime = copy.deepcopy(checkpoint.runtime)
        self.rng.setstate(checkpoint.rng_state)

    def fork(
        self,
        checkpoint: SimulatorCheckpoint | None = None,
        *,
        branch_seed: int | None = None,
    ) -> DeterministicUserSimulator:
        """Create an isolated counterfactual branch without mutating this simulator."""

        source = checkpoint or self.checkpoint()
        clone = DeterministicUserSimulator(
            goal=self.goal,
            persona=self.persona,
            catalog_index=self.catalog_index,
            oracle=self.oracle,
            seed=self.seed if branch_seed is None else branch_seed,
            reveal_order=self.reveal_order,
        )
        clone.runtime = copy.deepcopy(source.runtime)
        if branch_seed is None:
            clone.rng.setstate(source.rng_state)
        return clone

    def respond(self, action: SystemAction) -> tuple[UserResponse, list[OracleAssessment]]:
        if self.runtime.terminal:
            raise RuntimeError("cannot respond after the dialogue has terminated")
        self.runtime.patience_remaining -= 1

        if (
            self.runtime.pending_corrections
            and action.action_type != SystemActionType.STOP
            and self.rng.random() < self.persona.self_correction_probability
        ):
            response = self._respond_with_correction()
            assessments = []
        elif action.action_type == SystemActionType.ASK:
            response = self._respond_to_question(action)
            assessments: list[OracleAssessment] = []
        elif action.action_type in {SystemActionType.RECOMMEND, SystemActionType.REPAIR}:
            response, assessments = self._respond_to_recommendation(action)
        elif action.action_type == SystemActionType.STOP:
            response = self._respond_to_stop()
            assessments = []
        else:
            response = self._respond_to_unsupported_action(action)
            assessments = []

        if not response.terminal and self.runtime.patience_remaining <= 0:
            self.runtime.satisfaction = max(0.0, self.runtime.satisfaction - 0.2)
            self.runtime.terminal = True
            response = UserResponse(
                act=UserAct.ABANDON,
                text="This is taking too long, so I will stop here.",
                latent_satisfaction=self.runtime.satisfaction,
                terminal=True,
                reason="patience_exhausted",
            )
        return response, assessments

    def _respond_to_question(self, action: SystemAction) -> UserResponse:
        assert action.question_slot is not None
        if action.question_slot in self.runtime.pending_corrections:
            return self._respond_with_correction(action.question_slot)
        constraint = self.goal.constraint_map().get(action.question_slot)
        if constraint is None:
            self.runtime.satisfaction = max(0.0, self.runtime.satisfaction - 0.12)
            return UserResponse(
                act=UserAct.INFORM,
                text=f"I do not have a particular preference about {action.question_slot}.",
                dismissed_slots=[action.question_slot],
                latent_satisfaction=self.runtime.satisfaction,
                reason="irrelevant_or_unknown_slot",
            )

        if self.rng.random() > self.persona.cooperativeness:
            self.runtime.satisfaction = max(0.0, self.runtime.satisfaction - 0.08)
            return UserResponse(
                act=UserAct.INFORM,
                text="I am not sure how to answer that yet.",
                latent_satisfaction=self.runtime.satisfaction,
                reason="withheld_due_to_low_cooperativeness",
            )

        self.runtime.revealed_slots.add(constraint.slot)
        self.runtime.satisfaction = max(0.0, self.runtime.satisfaction - self.persona.question_cost)
        misstatement = None
        if (
            self.persona.misstatement_probability > 0.0
            and self.rng.random() < self.persona.misstatement_probability
        ):
            misstatement = self._sample_misstatement(constraint.as_predicate(), constraint.slot)
        if misstatement is not None:
            self.runtime.pending_corrections[constraint.slot] = constraint.as_predicate()
            return UserResponse(
                act=UserAct.INFORM,
                text=render_constraint(constraint.slot, misstatement),
                revealed_constraints={constraint.slot: misstatement},
                latent_satisfaction=self.runtime.satisfaction,
                reason="misstated_then_pending_correction",
            )
        return UserResponse(
            act=UserAct.INFORM,
            text=render_constraint(constraint.slot, constraint),
            revealed_constraints={constraint.slot: constraint.as_predicate()},
            latent_satisfaction=self.runtime.satisfaction,
            reason="answered_clarification",
        )

    def _sample_misstatement(
        self,
        predicate: ConstraintPredicate,
        slot: str,
    ) -> ConstraintPredicate | None:
        if predicate.operator != ConstraintOperator.EQ or isinstance(predicate.value, list):
            return None
        alternatives = set()
        for item_id in self.catalog_index.item_ids:
            raw = self.catalog_index.get(item_id).attributes.get(slot)
            values = raw if isinstance(raw, list) else [raw]
            for value in values:
                if value is not None and not satisfies_predicate(value, predicate):
                    alternatives.add(value)
        if not alternatives:
            return None
        ordered = sorted(alternatives, key=lambda value: (str(value).casefold(), str(type(value))))
        alternative = ordered[self.rng.randrange(len(ordered))]
        return ConstraintPredicate(
            operator=ConstraintOperator.EQ,
            value=alternative,
            importance=predicate.importance,
            unit=predicate.unit,
        )

    def _respond_with_correction(self, slot: str | None = None) -> UserResponse:
        selected = slot or sorted(self.runtime.pending_corrections)[0]
        predicate = self.runtime.pending_corrections.pop(selected)
        self.runtime.satisfaction = max(0.0, self.runtime.satisfaction - 0.02)
        return UserResponse(
            act=UserAct.INFORM,
            text=f"I need to correct myself. {render_constraint(selected, predicate)}",
            revealed_constraints={selected: predicate},
            latent_satisfaction=self.runtime.satisfaction,
            reason="self_correction",
        )

    def _respond_to_recommendation(
        self, action: SystemAction
    ) -> tuple[UserResponse, list[OracleAssessment]]:
        assessments = [
            self.oracle.assess(self.goal, self.catalog_index.get(item_id))
            for item_id in action.item_ids
            if item_id in self.catalog_index
        ]
        if not assessments:
            self.runtime.satisfaction = max(0.0, self.runtime.satisfaction - 0.25)
            return (
                UserResponse(
                    act=UserAct.REJECT,
                    text="Those options are not available to me. Please try again.",
                    latent_satisfaction=self.runtime.satisfaction,
                    reason="no_valid_catalog_items",
                ),
                assessments,
            )

        best = max(assessments, key=lambda item: (item.normalized_utility, item.item_id))
        if (
            not best.violated_hard_slots
            and not best.unknown_hard_slots
            and best.normalized_utility >= self.persona.acceptance_threshold
        ):
            self.runtime.satisfaction = min(1.0, self.runtime.satisfaction + 0.35)
            self.runtime.terminal = True
            return (
                UserResponse(
                    act=UserAct.ACCEPT,
                    text=f"{self.catalog_index.get(best.item_id).title} works for me.",
                    accepted_item_id=best.item_id,
                    latent_satisfaction=self.runtime.satisfaction,
                    terminal=True,
                    reason="compatible_item_found",
                ),
                assessments,
            )

        self.runtime.rejected_item_ids.update(action.item_ids)
        self.runtime.satisfaction = max(0.0, self.runtime.satisfaction - 0.18)
        reveal: dict[str, ConstraintPredicate] = {}
        eligible_slots = list(
            dict.fromkeys(
                [
                    *best.violated_hard_slots,
                    *best.unknown_hard_slots,
                    *best.missed_soft_slots,
                    *best.unknown_soft_slots,
                ]
            )
        )
        eligible_slots = [
            slot for slot in eligible_slots if slot not in self.runtime.revealed_slots
        ]
        if self.reveal_order is None:
            selected_slot = eligible_slots[0] if eligible_slots else None
        else:
            selected_slot = next(
                (slot for slot in self.reveal_order if slot in eligible_slots),
                None,
            )
            if eligible_slots and selected_slot is None:
                raise AssertionError("reveal plan omitted an eligible hidden goal slot")
        if selected_slot is not None:
            constraint = self.goal.constraint_map()[selected_slot]
            reveal[selected_slot] = constraint.as_predicate()
            self.runtime.revealed_slots.add(selected_slot)
        suffix = (
            ""
            if not reveal
            else f" {render_constraint(next(iter(reveal)), next(iter(reveal.values())))}"
        )
        return (
            UserResponse(
                act=UserAct.REJECT,
                text=f"These options do not fit what I need.{suffix}",
                revealed_constraints=reveal,
                latent_satisfaction=self.runtime.satisfaction,
                reason="recommendation_conflicts_with_hidden_goal",
            ),
            assessments,
        )

    def _respond_to_unsupported_action(self, action: SystemAction) -> UserResponse:
        self.runtime.satisfaction = max(0.0, self.runtime.satisfaction - 0.1)
        return UserResponse(
            act=UserAct.REQUEST_EXPLANATION,
            text=f"Could you make the {action.action_type.value} more concrete?",
            latent_satisfaction=self.runtime.satisfaction,
            reason="unsupported_offline_action",
        )

    def _respond_to_stop(self) -> UserResponse:
        self.runtime.satisfaction = max(0.0, self.runtime.satisfaction - 0.05)
        self.runtime.terminal = True
        return UserResponse(
            act=UserAct.ABANDON,
            text="Understood. We can stop if there are no suitable unseen options.",
            latent_satisfaction=self.runtime.satisfaction,
            terminal=True,
            reason="system_reported_no_candidates",
        )
