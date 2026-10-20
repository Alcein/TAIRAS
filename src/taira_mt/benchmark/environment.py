"""Deterministic evolving-intent user environment over the frozen action schema."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from taira_mt.benchmark.constraint_surface import (
    catalog_equivalent_constraint_surface,
)
from taira_mt.benchmark.oracle import IndependentBenchmarkOracle
from taira_mt.benchmark.schemas import (
    EXPLANATION_GUARD_SIMULATOR_VERSION,
    SIMULATOR_VERSION,
    BenchmarkActionType,
    BenchmarkCasePair,
    BenchmarkConstraint,
    BenchmarkEpisodeTrace,
    BenchmarkSystemAction,
    BenchmarkTurnTrace,
    BenchmarkUserAct,
    CandidateFeasibility,
    DialoguePhaseV1,
    GoalVersion,
    IntentCompleteness,
    IntentStability,
    PublicEnvironmentResponse,
    RewardBreakdown,
    RuntimeUserState,
    SemanticResponseFrame,
    TerminationKind,
    TransitionEvent,
    TransitionTrigger,
    benchmark_content_sha256,
)
from taira_mt.models import ConstraintPredicate, normalize_slot_name
from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.recommender.catalog import CandidatePool

if TYPE_CHECKING:
    from taira_mt.benchmark.language_renderer import BenchmarkLanguageRenderer


@dataclass(frozen=True)
class EnvironmentCheckpoint:
    runtime: RuntimeUserState
    traces: tuple[BenchmarkTurnTrace, ...]

    @property
    def digest(self) -> str:
        return content_sha256(
            {
                "runtime": self.runtime.model_dump(mode="json"),
                "traces": [trace.model_dump(mode="json") for trace in self.traces],
            }
        )


class DeterministicEvolvingUserEnvironment:
    """Branchable environment whose semantic behavior performs no model call."""

    def __init__(
        self,
        case: BenchmarkCasePair,
        candidate_pool: CandidatePool,
        oracle: IndependentBenchmarkOracle | None = None,
        language_renderer: BenchmarkLanguageRenderer | None = None,
    ) -> None:
        case.private.verify_public_case(case.public)
        if candidate_pool.pool_id != case.public.candidate_pool_id:
            raise ValueError("candidate pool identity differs from the benchmark case")
        self.case = case
        self.candidate_pool = candidate_pool
        self.language_renderer = language_renderer
        requested_utility = self.case.private.initial_state.oracle_spec.utility_version
        self.oracle = oracle or IndependentBenchmarkOracle(requested_utility)
        if self.case.private.initial_state.oracle_spec.utility_version != self.oracle.utility_version:
            raise ValueError("benchmark case requests an unsupported utility version")
        for goal_version in self.case.private.initial_state.latent_goal.versions:
            self.oracle.validate_registered_truth(goal_version, self.candidate_pool)
        self.runtime = self.case.private.initial_state.runtime_init.model_copy(deep=True)
        self._traces: list[BenchmarkTurnTrace] = []
        declared = (
            self.runtime.phase,
            self.runtime.completeness,
            self.runtime.feasibility,
        )
        self._recompute_public_state()
        derived = (
            self.runtime.phase,
            self.runtime.completeness,
            self.runtime.feasibility,
        )
        if declared != derived:
            raise ValueError(
                "registered initial phase/completeness/feasibility differs from the deterministic goal state"
            )

    @property
    def initial_query(self) -> str:
        if self.language_renderer is not None:
            return self.language_renderer.render_initial_query(
                case_id=self.case.public.case_id,
                language_style_id=self.case.private.initial_state.behavior.language_style_id,
            )
        return self.case.public.initial_query

    @property
    def terminal(self) -> bool:
        return self.runtime.terminal

    @property
    def private_traces(self) -> tuple[BenchmarkTurnTrace, ...]:
        return tuple(copy.deepcopy(self._traces))

    def reset(self) -> str:
        self.runtime = self.case.private.initial_state.runtime_init.model_copy(deep=True)
        self._traces = []
        return self.initial_query

    def checkpoint(self) -> EnvironmentCheckpoint:
        return EnvironmentCheckpoint(
            runtime=self.runtime.model_copy(deep=True),
            traces=tuple(copy.deepcopy(self._traces)),
        )

    def restore(self, checkpoint: EnvironmentCheckpoint) -> None:
        self.runtime = checkpoint.runtime.model_copy(deep=True)
        self._traces = list(copy.deepcopy(checkpoint.traces))

    def fork(self, checkpoint: EnvironmentCheckpoint | None = None) -> DeterministicEvolvingUserEnvironment:
        source = checkpoint or self.checkpoint()
        clone = DeterministicEvolvingUserEnvironment(
            self.case,
            self.candidate_pool,
            self.oracle,
            self.language_renderer,
        )
        clone.restore(source)
        return clone

    def _goal(self, version: int | None = None) -> GoalVersion:
        target = self.runtime.goal_version if version is None else version
        return self.case.private.initial_state.latent_goal.version(target)

    def _constraint_by_id(self, constraint_id: str, version: int | None = None):
        goal = self._goal(version)
        return next(
            (constraint for constraint in goal.constraints if constraint.constraint_id == constraint_id),
            None,
        )

    def _constraint_for_question(self, question_target: str):
        direct = self._constraint_by_id(question_target)
        if direct is not None:
            return direct
        normalized = normalize_slot_name(question_target)
        return next((constraint for constraint in self._goal().constraints if constraint.slot == normalized), None)

    def _selected_ambiguity_label(self) -> str | None:
        plan = self.case.private.initial_state.disclosure_plan
        selected = plan.selected_ambiguity_option_id
        if selected is None:
            return None
        option = next(option for option in plan.ambiguity_options if option.option_id == selected)
        return option.label

    def _event(
        self,
        trigger: TransitionTrigger,
        trigger_ref: str | None = None,
    ) -> TransitionEvent | None:
        candidates = []
        fired = set(self.runtime.fired_event_ids)
        for event in self.case.private.initial_state.evolution_plan.events:
            if event.trigger != trigger or event.from_goal_version != self.runtime.goal_version:
                continue
            if event.once and event.event_id in fired:
                continue
            if event.trigger_ref != trigger_ref:
                continue
            candidates.append(event)
        if len(candidates) > 1:
            raise RuntimeError("multiple transition events match the same deterministic trigger")
        return candidates[0] if candidates else None

    def _apply_event(self, event: TransitionEvent) -> list[str]:
        target_goal = self.case.private.initial_state.latent_goal.version(event.to_goal_version)
        valid_target_ids = target_goal.constraint_ids()
        disclosed = [
            constraint_id
            for constraint_id in self.runtime.disclosed_constraint_ids
            if constraint_id in valid_target_ids
        ]
        for constraint_id in event.reveal_constraint_ids:
            if constraint_id not in disclosed:
                disclosed.append(constraint_id)
        fired = list(self.runtime.fired_event_ids)
        if event.event_id not in fired:
            fired.append(event.event_id)
        self.runtime = self.runtime.model_copy(
            update={
                "goal_version": event.to_goal_version,
                "disclosed_constraint_ids": disclosed,
                "fired_event_ids": fired,
                "stability": event.resulting_stability or self.runtime.stability,
                "phase": event.target_phase or self.runtime.phase,
            },
            deep=True,
        )
        return list(event.reveal_constraint_ids)

    def _ambiguity_resolved(self) -> bool:
        plan = self.case.private.initial_state.disclosure_plan
        if not plan.ambiguity_options:
            return True
        selected = next(
            option
            for option in plan.ambiguity_options
            if option.option_id == plan.selected_ambiguity_option_id
        )
        if set(selected.constraint_ids).issubset(self.runtime.disclosed_constraint_ids):
            return True
        event_ids = set(self.runtime.fired_event_ids)
        return any(
            event.event_id in event_ids and event.trigger == TransitionTrigger.ASK_INTERPRETATION
            for event in self.case.private.initial_state.evolution_plan.events
        )

    def _recompute_public_state(self, *, preserve_feedback_phase: bool = False) -> None:
        goal = self._goal()
        disclosed = set(self.runtime.disclosed_constraint_ids)
        hard_ids = {
            constraint.constraint_id
            for constraint in goal.constraints
            if constraint.predicate.importance.value == "hard"
        }
        if not self._ambiguity_resolved():
            completeness = IntentCompleteness.AMBIGUOUS
        elif hard_ids.issubset(disclosed):
            completeness = IntentCompleteness.SUFFICIENT
        else:
            completeness = IntentCompleteness.INCOMPLETE

        if goal.feasible_item_count == 0:
            feasibility = CandidateFeasibility.EMPTY
            phase = DialoguePhaseV1.NEGOTIATE
        elif goal.feasible_item_count <= 10:
            feasibility = CandidateFeasibility.NARROW
            phase = DialoguePhaseV1.READY if completeness == IntentCompleteness.SUFFICIENT else DialoguePhaseV1.ELICIT
        else:
            feasibility = CandidateFeasibility.BROAD
            phase = DialoguePhaseV1.READY if completeness == IntentCompleteness.SUFFICIENT else DialoguePhaseV1.ELICIT
        if preserve_feedback_phase and self.runtime.phase in {
            DialoguePhaseV1.FEEDBACK,
            DialoguePhaseV1.REPAIR,
        }:
            phase = self.runtime.phase
        self.runtime = self.runtime.model_copy(
            update={
                "completeness": completeness,
                "feasibility": feasibility,
                "phase": phase,
            },
            deep=True,
        )

    def _constraint_text(self, constraint_id: str) -> str:
        constraint = self._constraint_by_id(constraint_id)
        if constraint is not None:
            return self._natural_constraint_surface(constraint)
        for goal in self.case.private.initial_state.latent_goal.versions:
            candidate = next(
                (item for item in goal.constraints if item.constraint_id == constraint_id),
                None,
            )
            if candidate is not None:
                return self._natural_constraint_surface(candidate)
        raise KeyError(f"unknown constraint: {constraint_id}")

    def _natural_constraint_surface(self, constraint: BenchmarkConstraint) -> str:
        """Return concise wording without changing the frozen catalog partition."""

        if self.language_renderer is None:
            return constraint.surface_form.rstrip(". ") + "."
        if constraint.slot in {"price", "rating", "popularity_percentile"}:
            return (
                catalog_equivalent_constraint_surface(
                    self.candidate_pool,
                    constraint.slot,
                    constraint.predicate,
                ).rstrip(". ")
                + "."
            )
        if constraint.slot == "specific_category":
            surface = constraint.surface_form.strip().rstrip(". ")
            if surface.startswith("More specifically, "):
                surface = surface.removeprefix("More specifically, ")
            return surface + "."
        return constraint.surface_form.rstrip(". ") + "."

    def _response(
        self,
        user_act: BenchmarkUserAct,
        communicative_goal: str,
        text: str,
        *,
        authorized_constraint_ids: list[str] | None = None,
        referenced_item_ids: list[str] | None = None,
        question_target: str | None = None,
    ) -> SemanticResponseFrame:
        constraint_ids = authorized_constraint_ids or []
        item_ids = referenced_item_ids or []
        frame = SemanticResponseFrame(
            user_act=user_act,
            communicative_goal=communicative_goal,
            authorized_constraint_ids=constraint_ids,
            referenced_item_ids=item_ids,
            style_id=self.case.private.initial_state.behavior.language_style_id,
            canonical_text=text,
        )
        if self.language_renderer is None:
            return frame
        from taira_mt.benchmark.language_renderer import RuntimeResponseLanguageContext

        surface_text = self.language_renderer.render_response(
            RuntimeResponseLanguageContext(
                case_id=self.case.public.case_id,
                turn_index=self.runtime.turn_index,
                language_style_id=frame.style_id,
                user_act=frame.user_act,
                communicative_goal=frame.communicative_goal,
                canonical_text=frame.canonical_text,
                authorized_constraint_surfaces=tuple(
                    self._constraint_text(value) for value in constraint_ids
                ),
                referenced_item_titles=tuple(
                    self.candidate_pool.index.get(value).title for value in item_ids
                ),
                selected_interpretation=(
                    self._selected_ambiguity_label()
                    if communicative_goal == "resolve_ambiguity"
                    else None
                ),
                question_target=question_target,
            )
        )
        return frame.model_copy(update={"surface_text": surface_text}, deep=True)

    @staticmethod
    def _reward(
        *,
        recommendation_utility: float = 0.0,
        acceptance_bonus: float = 0.0,
        question_cost: float = 0.0,
        invalid_action_cost: float = 0.0,
        dialogue_contract_cost: float = 0.0,
        hard_violation_cost: float = 0.0,
        abandonment_cost: float = 0.0,
    ) -> RewardBreakdown:
        total = (
            recommendation_utility
            + acceptance_bonus
            - question_cost
            - invalid_action_cost
            - dialogue_contract_cost
            - hard_violation_cost
            - abandonment_cost
        )
        return RewardBreakdown(
            recommendation_utility=recommendation_utility,
            acceptance_bonus=acceptance_bonus,
            question_cost=question_cost,
            invalid_action_cost=invalid_action_cost,
            dialogue_contract_cost=dialogue_contract_cost,
            hard_violation_cost=hard_violation_cost,
            abandonment_cost=abandonment_cost,
            total=total,
        )

    def _semantic_invalid(self, reason: str) -> tuple[SemanticResponseFrame, RewardBreakdown, list, list[str]]:
        cost = self.case.private.initial_state.behavior.invalid_action_cost
        self.runtime = self.runtime.model_copy(
            update={"satisfaction": max(0.0, self.runtime.satisfaction - cost)},
            deep=True,
        )
        return (
            self._response(
                BenchmarkUserAct.CORRECT_INVALID_ACTION,
                reason,
                "I cannot respond to that as a valid recommendation step. Please try again.",
            ),
            self._reward(invalid_action_cost=cost),
            [],
            [],
        )

    def _handle_ask(self, action: BenchmarkSystemAction):
        assert action.question_target is not None
        target = action.question_target
        question_cost = self.case.private.initial_state.behavior.question_cost
        asked = list(self.runtime.asked_question_targets)
        repeated = target in set(asked)
        if not repeated:
            asked.append(target)
        self.runtime = self.runtime.model_copy(
            update={
                "asked_question_targets": asked,
                "satisfaction": max(0.0, self.runtime.satisfaction - question_cost),
            },
            deep=True,
        )

        interpretation_event = self._event(TransitionTrigger.ASK_INTERPRETATION, target)
        if interpretation_event is not None:
            revealed = self._apply_event(interpretation_event)
            label = self._selected_ambiguity_label() or "the intended interpretation"
            frame = self._response(
                interpretation_event.response_act or BenchmarkUserAct.CLARIFY,
                "resolve_ambiguity",
                f"I mean {label}.",
                authorized_constraint_ids=revealed,
            )
            self._recompute_public_state()
            return frame, self._reward(question_cost=question_cost), [], [interpretation_event.event_id]

        constraint = self._constraint_for_question(target)
        if constraint is None:
            dismissed = list(self.runtime.dismissed_question_targets)
            if target not in dismissed:
                dismissed.append(target)
            self.runtime = self.runtime.model_copy(update={"dismissed_question_targets": dismissed}, deep=True)
            frame = self._response(
                BenchmarkUserAct.NO_PREFERENCE,
                "dismiss_irrelevant_question",
                f"I do not have a particular preference about {target}.",
                question_target=target,
            )
            self._recompute_public_state()
            return frame, self._reward(question_cost=question_cost), [], []

        event = self._event(TransitionTrigger.ASK_CONSTRAINT, constraint.constraint_id)
        if event is not None:
            revealed = self._apply_event(event)
            if constraint.constraint_id not in revealed:
                revealed.append(constraint.constraint_id)
            frame = self._response(
                event.response_act or BenchmarkUserAct.INFORM,
                "answer_registered_tradeoff_question",
                self._constraint_text(constraint.constraint_id),
                authorized_constraint_ids=revealed,
            )
            self._recompute_public_state()
            return frame, self._reward(question_cost=question_cost), [], [event.event_id]

        if constraint.constraint_id in set(self.runtime.disclosed_constraint_ids):
            frame = self._response(
                BenchmarkUserAct.REMIND,
                "remind_already_disclosed_constraint",
                f"I already mentioned that {self._constraint_text(constraint.constraint_id)}",
                authorized_constraint_ids=[constraint.constraint_id],
            )
            self._recompute_public_state()
            return frame, self._reward(question_cost=question_cost), [], []

        disclosed = list(self.runtime.disclosed_constraint_ids)
        disclosed.append(constraint.constraint_id)
        self.runtime = self.runtime.model_copy(update={"disclosed_constraint_ids": disclosed}, deep=True)
        fired: list[str] = []
        frame = self._response(
            BenchmarkUserAct.INFORM,
            "answer_relevant_question",
            self._constraint_text(constraint.constraint_id),
            authorized_constraint_ids=[constraint.constraint_id],
        )
        self._recompute_public_state()
        return frame, self._reward(question_cost=question_cost), [], fired

    def _feedback_constraint(self, assessments) -> str | None:
        disclosed = set(self.runtime.disclosed_constraint_ids)
        hidden_order = self.case.private.initial_state.disclosure_plan.hidden_order
        violated_slots = {
            slot
            for assessment in assessments
            for slot in [
                *assessment.violated_hard_slots,
                *assessment.unknown_hard_slots,
                *assessment.missed_soft_slots,
                *assessment.unknown_soft_slots,
            ]
        }
        goal = self._goal()
        by_slot = {constraint.slot: constraint.constraint_id for constraint in goal.constraints}
        candidates = [by_slot[slot] for slot in violated_slots if slot in by_slot]
        for constraint_id in hidden_order:
            if constraint_id in candidates and constraint_id not in disclosed:
                return constraint_id
        return next((value for value in candidates if value not in disclosed), None)

    def _handle_recommend(self, action: BenchmarkSystemAction):
        unknown = [item_id for item_id in action.item_ids if item_id not in set(self.candidate_pool.item_ids)]
        if unknown:
            return self._semantic_invalid("recommendation_contains_out_of_pool_item")
        pending = self.runtime.pending_acceptable_item_id
        if (
            pending is not None
            and self.case.private.initial_state.simulator_version
            in {EXPLANATION_GUARD_SIMULATOR_VERSION, SIMULATOR_VERSION}
        ):
            cost = self.case.private.initial_state.behavior.invalid_action_cost
            self.runtime = self.runtime.model_copy(
                update={
                    "satisfaction": max(0.0, self.runtime.satisfaction - cost),
                },
                deep=True,
            )
            frame = self._response(
                BenchmarkUserAct.REQUEST_EXPLANATION,
                "request_explanation_before_acceptance",
                "I still need an explanation for that option before I can decide.",
                referenced_item_ids=[pending],
            )
            return frame, self._reward(dialogue_contract_cost=cost), [], []
        slate = self.oracle.assess_slate(self._goal(), self.candidate_pool, action.item_ids)
        self.runtime = self.runtime.model_copy(
            update={"last_recommended_item_ids": list(action.item_ids)},
            deep=True,
        )

        hard_cost = 1.5 if slate.all_items_hard_violating else 0.0
        if slate.accepted_item_id is not None:
            event = self._event(TransitionTrigger.SATISFACTORY_RECOMMENDATION)
            if event is not None:
                revealed = self._apply_event(event)
                is_goal_revision = event.to_goal_version != event.from_goal_version
                if is_goal_revision:
                    self.runtime = self.runtime.model_copy(
                        update={
                            "pending_acceptable_item_id": None,
                            "phase": event.target_phase or DialoguePhaseV1.REPAIR,
                            "satisfaction": max(0.0, self.runtime.satisfaction - 0.05),
                        },
                        deep=True,
                    )
                    details = " ".join(self._constraint_text(value) for value in revealed)
                    frame = self._response(
                        event.response_act or BenchmarkUserAct.CLARIFY,
                        "apply_precommitted_goal_revision",
                        f"Seeing that option helps me clarify my need. {details}".strip(),
                        authorized_constraint_ids=revealed,
                        referenced_item_ids=[slate.accepted_item_id],
                    )
                    self._recompute_public_state(preserve_feedback_phase=True)
                    return (
                        frame,
                        self._reward(recommendation_utility=slate.best_utility),
                        list(slate.assessments),
                        [event.event_id],
                    )
                self.runtime = self.runtime.model_copy(
                    update={
                        "pending_acceptable_item_id": slate.accepted_item_id,
                        "phase": event.target_phase or DialoguePhaseV1.FEEDBACK,
                        "satisfaction": min(1.0, self.runtime.satisfaction + 0.1),
                    },
                    deep=True,
                )
                frame = self._response(
                    event.response_act or BenchmarkUserAct.REQUEST_EXPLANATION,
                    "request_explanation_before_acceptance",
                    "That option seems plausible. Could you explain why it fits my needs?",
                    authorized_constraint_ids=revealed,
                    referenced_item_ids=[slate.accepted_item_id],
                )
                return (
                    frame,
                    self._reward(recommendation_utility=slate.best_utility),
                    list(slate.assessments),
                    [event.event_id],
                )

            title = self.candidate_pool.index.get(slate.accepted_item_id).title
            self.runtime = self.runtime.model_copy(
                update={
                    "phase": DialoguePhaseV1.TERMINAL,
                    "terminal": True,
                    "termination_kind": TerminationKind.ACCEPTED,
                    "last_user_act": BenchmarkUserAct.ACCEPT,
                    "pending_acceptable_item_id": None,
                    "satisfaction": min(1.0, self.runtime.satisfaction + 0.35),
                },
                deep=True,
            )
            frame = self._response(
                BenchmarkUserAct.ACCEPT,
                "accept_satisfactory_recommendation",
                f"{title} works for me.",
                referenced_item_ids=[slate.accepted_item_id],
            )
            return (
                frame,
                self._reward(recommendation_utility=slate.best_utility, acceptance_bonus=1.0),
                list(slate.assessments),
                [],
            )

        event = self._event(TransitionTrigger.UNSATISFACTORY_RECOMMENDATION)
        fired: list[str] = []
        revealed: list[str] = []
        if event is not None:
            revealed = self._apply_event(event)
            fired.append(event.event_id)
        if not revealed:
            feedback_constraint = self._feedback_constraint(slate.assessments)
            if feedback_constraint is not None:
                disclosed = list(self.runtime.disclosed_constraint_ids)
                if feedback_constraint not in disclosed:
                    disclosed.append(feedback_constraint)
                    self.runtime = self.runtime.model_copy(
                        update={"disclosed_constraint_ids": disclosed},
                        deep=True,
                    )
                revealed = [feedback_constraint]
        rejected = list(self.runtime.rejected_item_ids)
        for item_id in action.item_ids:
            if item_id not in rejected:
                rejected.append(item_id)
        self.runtime = self.runtime.model_copy(
            update={
                "rejected_item_ids": rejected,
                "phase": DialoguePhaseV1.REPAIR,
                "satisfaction": max(0.0, self.runtime.satisfaction - 0.18),
                "pending_acceptable_item_id": None,
            },
            deep=True,
        )
        details = " ".join(self._constraint_text(value) for value in revealed)
        only_unverifiable = bool(slate.assessments) and all(
            assessment.unknown_hard_slots and not assessment.violated_hard_slots
            for assessment in slate.assessments
        )
        text = (
            "I cannot verify that those options meet my stated requirements from "
            "the available product information."
            if only_unverifiable
            else "Those options do not fit what I need."
        )
        if details:
            text = f"{text} {details}"
        frame = self._response(
            event.response_act if event and event.response_act else BenchmarkUserAct.REJECT,
            (
                "reject_unverifiable_recommendation"
                if only_unverifiable
                else "reject_unsatisfactory_recommendation"
            ),
            text,
            authorized_constraint_ids=revealed,
            referenced_item_ids=list(action.item_ids),
        )
        self._recompute_public_state(preserve_feedback_phase=True)
        return (
            frame,
            self._reward(
                recommendation_utility=slate.best_utility,
                hard_violation_cost=hard_cost,
            ),
            list(slate.assessments),
            fired,
        )

    def _evidence_is_grounded(self, action: BenchmarkSystemAction) -> bool:
        for reference in action.evidence_refs:
            if reference.item_id not in set(self.candidate_pool.item_ids):
                return False
            item = self.candidate_pool.index.get(reference.item_id)
            if reference.catalog_field != "title" and reference.catalog_field not in item.attributes:
                return False
        return True

    def required_explanation_fields(self, item_id: str) -> tuple[str, ...]:
        """Return evaluator-required public fields for a pending explanation."""

        item = self.candidate_pool.index.get(item_id)
        disclosed = set(self.runtime.disclosed_constraint_ids)
        fields = [
            constraint.slot
            for constraint in self._goal().constraints
            if constraint.constraint_id in disclosed
            and constraint.predicate.importance.value == "hard"
            and constraint.slot != "broad_category"
            and constraint.slot in item.attributes
        ]
        return tuple(dict.fromkeys(fields)) or ("title",)

    def _explanation_covers_preferences(
        self,
        action: BenchmarkSystemAction,
        pending_item_id: str,
    ) -> bool:
        required = set(self.required_explanation_fields(pending_item_id))
        provided = {
            reference.catalog_field
            for reference in action.evidence_refs
            if reference.item_id == pending_item_id
        }
        return required.issubset(provided)

    def _handle_explain(self, action: BenchmarkSystemAction):
        if any(item_id not in set(self.candidate_pool.item_ids) for item_id in action.item_ids):
            return self._semantic_invalid("explanation_contains_out_of_pool_item")
        if not self._evidence_is_grounded(action):
            return self._semantic_invalid("explanation_uses_ungrounded_catalog_evidence")
        pending = self.runtime.pending_acceptable_item_id
        if pending is not None and (not action.item_ids or pending in action.item_ids):
            if (
                self.case.private.initial_state.simulator_version == SIMULATOR_VERSION
                and not self._explanation_covers_preferences(action, pending)
            ):
                cost = self.case.private.initial_state.behavior.invalid_action_cost
                self.runtime = self.runtime.model_copy(
                    update={
                        "satisfaction": max(0.0, self.runtime.satisfaction - cost),
                    },
                    deep=True,
                )
                frame = self._response(
                    BenchmarkUserAct.REQUEST_EXPLANATION,
                    "request_explanation_before_acceptance",
                    "I still need the explanation to connect this option to the "
                    "preferences I shared.",
                    referenced_item_ids=[pending],
                )
                return frame, self._reward(dialogue_contract_cost=cost), [], []
            title = self.candidate_pool.index.get(pending).title
            self.runtime = self.runtime.model_copy(
                update={
                    "phase": DialoguePhaseV1.TERMINAL,
                    "terminal": True,
                    "termination_kind": TerminationKind.ACCEPTED,
                    "pending_acceptable_item_id": None,
                    "satisfaction": min(1.0, self.runtime.satisfaction + 0.25),
                },
                deep=True,
            )
            frame = self._response(
                BenchmarkUserAct.ACCEPT,
                "accept_after_grounded_explanation",
                f"That explanation helps. {title} works for me.",
                referenced_item_ids=[pending],
            )
            return frame, self._reward(acceptance_bonus=1.0), [], []
        frame = self._response(
            BenchmarkUserAct.ACKNOWLEDGE,
            "acknowledge_grounded_explanation",
            "That explanation is helpful. I would still like a suitable recommendation.",
            referenced_item_ids=list(action.item_ids),
        )
        self._recompute_public_state()
        return frame, self._reward(), [], []

    @staticmethod
    def _same_predicate(left: ConstraintPredicate, right: ConstraintPredicate) -> bool:
        return canonical_json(left.model_dump(mode="json")) == canonical_json(right.model_dump(mode="json"))

    def _handle_relaxation(self, action: BenchmarkSystemAction):
        assert action.relaxation_proposal is not None
        proposal = action.relaxation_proposal
        event = self._event(TransitionTrigger.ALLOWED_RELAXATION, proposal.constraint_id)
        if event is None:
            frame = self._response(
                BenchmarkUserAct.REFUSE_RELAXATION,
                "refuse_unregistered_relaxation",
                "I would rather not relax that requirement in this way.",
            )
            return frame, self._reward(), [], []
        target_constraint = next(
            (
                constraint
                for constraint in self.case.private.initial_state.latent_goal.version(
                    event.to_goal_version
                ).constraints
                if constraint.constraint_id == proposal.constraint_id
            ),
            None,
        )
        if target_constraint is None or not self._same_predicate(
            proposal.proposed_predicate,
            target_constraint.predicate,
        ):
            frame = self._response(
                BenchmarkUserAct.REFUSE_RELAXATION,
                "refuse_mismatched_relaxation",
                "That is not the trade-off I am willing to make.",
            )
            return frame, self._reward(), [], []
        revealed = self._apply_event(event)
        if proposal.constraint_id not in revealed:
            revealed.append(proposal.constraint_id)
            disclosed = list(self.runtime.disclosed_constraint_ids)
            if proposal.constraint_id not in disclosed:
                disclosed.append(proposal.constraint_id)
            self.runtime = self.runtime.model_copy(update={"disclosed_constraint_ids": disclosed}, deep=True)
        self.runtime = self.runtime.model_copy(update={"stability": IntentStability.RELAXED}, deep=True)
        self._recompute_public_state()
        frame = self._response(
            event.response_act or BenchmarkUserAct.AGREE_RELAXATION,
            "accept_registered_relaxation",
            f"That trade-off works for me. {self._constraint_text(proposal.constraint_id)}",
            authorized_constraint_ids=revealed,
        )
        return frame, self._reward(), [], [event.event_id]

    def _handle_stop(self):
        if self.runtime.feasibility == CandidateFeasibility.EMPTY:
            termination = TerminationKind.GRACEFUL_NO_SOLUTION
            act = BenchmarkUserAct.GRACEFUL_END
            text = "I understand that there is no suitable option under those requirements."
            cost = 0.0
        else:
            termination = TerminationKind.ABANDONED
            act = BenchmarkUserAct.ABANDON
            text = "I still needed a recommendation, so I will stop here."
            cost = 1.0
        self.runtime = self.runtime.model_copy(
            update={
                "phase": DialoguePhaseV1.TERMINAL,
                "terminal": True,
                "termination_kind": termination,
                "satisfaction": max(0.0, self.runtime.satisfaction - cost),
            },
            deep=True,
        )
        frame = self._response(act, "respond_to_system_stop", text)
        return frame, self._reward(abandonment_cost=cost), [], []

    def _apply_patience_and_advance(
        self,
        frame: SemanticResponseFrame,
        reward: RewardBreakdown,
    ) -> tuple[SemanticResponseFrame, RewardBreakdown]:
        patience = max(0, self.runtime.patience_remaining - 1)
        updates = {
            "turn_index": self.runtime.turn_index + 1,
            "patience_remaining": patience,
            "last_user_act": frame.user_act,
        }
        self.runtime = self.runtime.model_copy(update=updates, deep=True)
        if not self.runtime.terminal and patience == 0:
            frame = self._response(
                BenchmarkUserAct.ABANDON,
                "patience_exhausted",
                "This is taking too long, so I will stop here.",
            )
            self.runtime = self.runtime.model_copy(
                update={
                    "phase": DialoguePhaseV1.TERMINAL,
                    "terminal": True,
                    "termination_kind": TerminationKind.ABANDONED,
                    "last_user_act": BenchmarkUserAct.ABANDON,
                    "satisfaction": max(0.0, self.runtime.satisfaction - 0.2),
                },
                deep=True,
            )
            reward = self._reward(
                recommendation_utility=reward.recommendation_utility,
                acceptance_bonus=reward.acceptance_bonus,
                question_cost=reward.question_cost,
                invalid_action_cost=reward.invalid_action_cost,
                dialogue_contract_cost=reward.dialogue_contract_cost,
                hard_violation_cost=reward.hard_violation_cost,
                abandonment_cost=reward.abandonment_cost + 1.0,
            )
        return frame, reward

    def step(self, action: BenchmarkSystemAction) -> PublicEnvironmentResponse:
        if self.runtime.terminal:
            raise RuntimeError("cannot step a terminal user environment")
        state_before = self.runtime.model_copy(deep=True)

        if action.action_type == BenchmarkActionType.ASK:
            frame, reward, assessments, fired = self._handle_ask(action)
        elif action.action_type == BenchmarkActionType.RECOMMEND:
            frame, reward, assessments, fired = self._handle_recommend(action)
        elif action.action_type == BenchmarkActionType.EXPLAIN:
            frame, reward, assessments, fired = self._handle_explain(action)
        elif action.action_type == BenchmarkActionType.PROPOSE_RELAX:
            frame, reward, assessments, fired = self._handle_relaxation(action)
        elif action.action_type == BenchmarkActionType.STOP:
            frame, reward, assessments, fired = self._handle_stop()
        else:  # pragma: no cover - exhaustive enum guard
            raise AssertionError(f"unsupported action type: {action.action_type}")

        frame, reward = self._apply_patience_and_advance(frame, reward)
        trace = BenchmarkTurnTrace(
            case_id=self.case.public.case_id,
            turn_index=state_before.turn_index,
            state_before=state_before,
            system_action=action,
            response_frame=frame,
            public_user_utterance=frame.public_text,
            state_after=self.runtime.model_copy(deep=True),
            fired_event_ids=fired,
            oracle_assessments=assessments,
            reward=reward,
        )
        self._traces.append(trace)
        return PublicEnvironmentResponse(
            utterance=frame.public_text,
            terminal=self.runtime.terminal,
            turn_index=self.runtime.turn_index,
        )

    def episode_trace(self, episode_id: str) -> BenchmarkEpisodeTrace:
        return BenchmarkEpisodeTrace(
            episode_id=episode_id,
            case_id=self.case.public.case_id,
            public_case_sha256=benchmark_content_sha256(self.case.public),
            initial_state_sha256=benchmark_content_sha256(self.case.private.initial_state),
            initial_user_utterance=self.initial_query,
            language_renderer_id=(
                self.language_renderer.renderer_id
                if self.language_renderer is not None
                else None
            ),
            turns=list(self.private_traces),
            termination_kind=self.runtime.termination_kind,
            success=self.runtime.termination_kind == TerminationKind.ACCEPTED,
        )
