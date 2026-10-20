"""Compile frozen logical plans and rendered dialogue into replayable DSUS data."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, model_validator

from taira_mt.models import StrictModel
from taira_mt.state_compiler import (
    AddUnresolvedAxisOperation,
    ConstraintImportance,
    ConstraintPolarity,
    ControlField,
    EvidenceReference,
    EvidenceSource,
    ExplanationStatus,
    FeedbackType,
    InformationStatus,
    InitializeStateOperation,
    IntentStability,
    InteractionPhase,
    NoopOperation,
    PublicDSUSState,
    RecordFeedbackOperation,
    RemoveConstraintOperation,
    ResolveAxisOperation,
    SetControlOperation,
    StatePatch,
    SupersedeConstraintOperation,
    TerminateOperation,
    TerminationReason,
    UpsertConstraintOperation,
    reduce_state,
)
from taira_mt.state_training.authoring import (
    InteractionMechanism,
    LogicalTurnPlan,
    PlannedSystemIntent,
    PlannedUserEvent,
    RichLogicalTrajectoryPlan,
    RichSourceRoot,
    SemanticFactTemplate,
    SemanticImportance,
    SemanticPolarity,
)
from taira_mt.state_training.rendering import (
    FactSurfaceEvidence,
    RenderedTrajectoryVariant,
    RenderedTurn,
    validate_rendered_variant,
)


class PublicDialogueMessage(StrictModel):
    turn_id: int = Field(ge=0)
    role: Literal["user", "system"]
    text: str = Field(min_length=1)
    action_type: str | None = None


class CompiledStateTransition(StrictModel):
    root_id: str
    variant_id: str
    step_id: str
    public_history: list[PublicDialogueMessage] = Field(min_length=1)
    previous_state: PublicDSUSState
    gold_patch: StatePatch
    gold_next_state: PublicDSUSState

    @model_validator(mode="after")
    def validate_replay(self) -> CompiledStateTransition:
        replayed = reduce_state(self.previous_state, self.gold_patch)
        if replayed.canonical_json() != self.gold_next_state.canonical_json():
            raise ValueError("compiled transition gold patch does not replay to gold next state")
        return self


class CompiledLogicalTrajectory(StrictModel):
    root_id: str
    variant_id: str
    mechanism: InteractionMechanism
    transitions: list[CompiledStateTransition] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_chain(self) -> CompiledLogicalTrajectory:
        if any(
            transition.root_id != self.root_id or transition.variant_id != self.variant_id
            for transition in self.transitions
        ):
            raise ValueError("compiled trajectory transition identity mismatch")
        for previous, current in zip(self.transitions, self.transitions[1:], strict=False):
            if previous.gold_next_state.canonical_json() != current.previous_state.canonical_json():
                raise ValueError("compiled trajectory state chain is discontinuous")
        return self


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return normalized[:64] or "open_axis"


def _polarity(value: SemanticPolarity) -> ConstraintPolarity:
    return {
        SemanticPolarity.REQUIRE: ConstraintPolarity.MUST,
        SemanticPolarity.PREFER: ConstraintPolarity.PREFER,
        SemanticPolarity.AVOID: ConstraintPolarity.AVOID,
    }[value]


def _importance(value: SemanticImportance) -> ConstraintImportance:
    return {
        SemanticImportance.HARD: ConstraintImportance.HARD,
        SemanticImportance.SOFT: ConstraintImportance.SOFT,
    }[value]


def _opening_controls(mechanism: InteractionMechanism) -> tuple[InteractionPhase, InformationStatus]:
    if mechanism == InteractionMechanism.PROGRESSIVE:
        return InteractionPhase.ELICITATION, InformationStatus.MISSING
    if mechanism == InteractionMechanism.AMBIGUOUS:
        return InteractionPhase.ELICITATION, InformationStatus.AMBIGUOUS
    if mechanism == InteractionMechanism.CONFLICT:
        return InteractionPhase.ELICITATION, InformationStatus.CONFLICTING
    return InteractionPhase.RECOMMENDATION, InformationStatus.SUFFICIENT


def _evidence_records(
    *,
    step_id: str,
    turn_id: int,
    user_text: str,
    event_quotes: list[str],
    fact_evidence: list[FactSurfaceEvidence],
) -> tuple[list[EvidenceReference], list[str], dict[str, list[str]]]:
    ordered_quotes: list[str] = []
    for quote in event_quotes:
        normalized = " ".join(quote.split())
        if normalized.casefold() not in user_text.casefold():
            raise ValueError(f"event evidence is absent from user text: {normalized}")
        if normalized.casefold() not in {value.casefold() for value in ordered_quotes}:
            ordered_quotes.append(normalized)
    for evidence in fact_evidence:
        for quote in evidence.exact_quotes:
            if quote.casefold() not in {value.casefold() for value in ordered_quotes}:
                ordered_quotes.append(quote)
    records: list[EvidenceReference] = []
    id_by_quote: dict[str, str] = {}
    for index, quote in enumerate(ordered_quotes):
        evidence_id = f"e_{step_id}_{index}"
        records.append(
            EvidenceReference(
                evidence_id=evidence_id,
                turn_id=turn_id,
                source=EvidenceSource.USER,
                quote=quote,
            )
        )
        id_by_quote[quote.casefold()] = evidence_id
    event_ids = [id_by_quote[" ".join(quote.split()).casefold()] for quote in event_quotes]
    fact_ids = {
        evidence.fact_id: [id_by_quote[quote.casefold()] for quote in evidence.exact_quotes]
        for evidence in fact_evidence
    }
    return records, sorted(set(event_ids)), fact_ids


def _upsert_fact(fact: SemanticFactTemplate, evidence_ids: list[str]) -> UpsertConstraintOperation:
    return UpsertConstraintOperation(
        constraint_id=fact.fact_id,
        name=fact.name,
        value=fact.value,
        polarity=_polarity(fact.polarity),
        importance=_importance(fact.importance),
        evidence_ids=evidence_ids,
    )


def _opening_public_axis(
    plan: RichLogicalTrajectoryPlan,
    variant: RenderedTrajectoryVariant,
) -> tuple[str, str]:
    """Build an axis solely from opening-user evidence, never future facts."""

    facts = {fact.fact_id: fact for fact in plan.semantic_facts}
    first_evidence = variant.initial_fact_evidence[0]
    public_fact = facts[first_evidence.fact_id]
    return (
        f"public_{_slug(public_fact.name)}_clarification",
        f"Clarification of the public {public_fact.name} requirement: {public_fact.value}",
    )


def _opening_patch(
    plan: RichLogicalTrajectoryPlan,
    variant: RenderedTrajectoryVariant,
) -> StatePatch:
    event_quote = variant.initial_user_utterance
    evidence, event_ids, fact_evidence = _evidence_records(
        step_id="opening",
        turn_id=0,
        user_text=variant.initial_user_utterance,
        event_quotes=[event_quote],
        fact_evidence=variant.initial_fact_evidence,
    )
    operations = [InitializeStateOperation(evidence_ids=event_ids)]
    facts = {fact.fact_id: fact for fact in plan.semantic_facts}
    for fact_id in sorted(fact_evidence):
        operations.append(_upsert_fact(facts[fact_id], fact_evidence[fact_id]))

    phase, information = _opening_controls(plan.mechanism)
    if plan.mechanism in {
        InteractionMechanism.PROGRESSIVE,
        InteractionMechanism.AMBIGUOUS,
        InteractionMechanism.CONFLICT,
    }:
        axis_type, description = _opening_public_axis(plan, variant)
        operations.append(
            AddUnresolvedAxisOperation(
                axis_id="axis_public_clarification",
                axis_type=axis_type,
                description=description,
                options=[],
                evidence_ids=event_ids,
            )
        )
    operations.extend(
        (
            SetControlOperation(
                field=ControlField.INTERACTION_PHASE,
                value=phase.value,
                evidence_ids=event_ids,
            ),
            SetControlOperation(
                field=ControlField.INFORMATION_STATUS,
                value=information.value,
                evidence_ids=event_ids,
            ),
        )
    )
    return StatePatch(base_revision=0, evidence=evidence, operations=operations)


def _turn_feedback(event: PlannedUserEvent) -> FeedbackType | None:
    return {
        PlannedUserEvent.DISCLOSE: FeedbackType.ANSWER,
        PlannedUserEvent.CHOOSE: FeedbackType.ANSWER,
        PlannedUserEvent.RELAX: FeedbackType.ANSWER,
        PlannedUserEvent.REJECT: FeedbackType.REJECTION,
        PlannedUserEvent.REVISE: FeedbackType.CORRECTION,
        PlannedUserEvent.REMOVE: FeedbackType.CORRECTION,
        PlannedUserEvent.REQUEST_EXPLANATION: FeedbackType.EXPLANATION_REQUEST,
        PlannedUserEvent.ACCEPT: FeedbackType.ACCEPTANCE,
    }.get(event)


def _turn_stability(event: PlannedUserEvent) -> IntentStability | None:
    return {
        PlannedUserEvent.DISCLOSE: IntentStability.STRENGTHENED,
        PlannedUserEvent.CHOOSE: IntentStability.STABLE,
        PlannedUserEvent.RELAX: IntentStability.RELAXED,
        PlannedUserEvent.REVISE: IntentStability.REVISED,
        PlannedUserEvent.REMOVE: IntentStability.RELAXED,
    }.get(event)


def _turn_phase(logical: LogicalTurnPlan) -> InteractionPhase:
    if logical.user_event in {
        PlannedUserEvent.REJECT,
        PlannedUserEvent.REVISE,
        PlannedUserEvent.REMOVE,
    }:
        return InteractionPhase.REPAIR
    if logical.user_event == PlannedUserEvent.REQUEST_EXPLANATION:
        return InteractionPhase.FEEDBACK
    if logical.system_intent == PlannedSystemIntent.ASK:
        return InteractionPhase.GROUNDING
    if logical.system_intent == PlannedSystemIntent.EXPLAIN:
        return InteractionPhase.FEEDBACK
    return InteractionPhase.RECOMMENDATION


def _information_after_turn(logical: LogicalTurnPlan) -> InformationStatus | None:
    if logical.user_event in {
        PlannedUserEvent.DISCLOSE,
        PlannedUserEvent.CHOOSE,
        PlannedUserEvent.RELAX,
    }:
        return InformationStatus.SUFFICIENT
    if logical.user_event in {
        PlannedUserEvent.REJECT,
        PlannedUserEvent.REVISE,
        PlannedUserEvent.REMOVE,
    }:
        # A bare negative reaction identifies failure but not how to repair it.
        return InformationStatus.SUFFICIENT if logical.reveal_fact_ids else InformationStatus.MISSING
    return None


def _turn_patch(
    *,
    state: PublicDSUSState,
    plan: RichLogicalTrajectoryPlan,
    logical: LogicalTurnPlan,
    rendered: RenderedTurn,
) -> StatePatch:
    step_id = f"turn_{logical.turn_index}"
    evidence, event_ids, fact_evidence = _evidence_records(
        step_id=step_id,
        turn_id=logical.turn_index,
        user_text=rendered.user_utterance,
        event_quotes=rendered.user_event_evidence,
        fact_evidence=rendered.revealed_fact_evidence,
    )
    system_evidence_ids: list[str] = []
    if logical.system_intent == PlannedSystemIntent.EXPLAIN:
        system_evidence_id = f"e_{step_id}_system"
        evidence.append(
            EvidenceReference(
                evidence_id=system_evidence_id,
                turn_id=logical.turn_index,
                source=EvidenceSource.SYSTEM,
                quote=rendered.system_utterance,
            )
        )
        system_evidence_ids.append(system_evidence_id)
    if logical.user_event == PlannedUserEvent.NOOP:
        if logical.system_intent == PlannedSystemIntent.EXPLAIN:
            return StatePatch(
                base_revision=state.revision,
                evidence=evidence,
                operations=[
                    SetControlOperation(
                        field=ControlField.INTERACTION_PHASE,
                        value=InteractionPhase.FEEDBACK.value,
                        evidence_ids=event_ids,
                    ),
                    SetControlOperation(
                        field=ControlField.EXPLANATION_STATUS,
                        value=ExplanationStatus.RESOLVED.value,
                        evidence_ids=system_evidence_ids,
                    ),
                ],
            )
        return StatePatch(
            base_revision=state.revision,
            evidence=evidence,
            operations=[NoopOperation(reason="public observation changes no registered state", evidence_ids=event_ids)],
        )

    facts = {fact.fact_id: fact for fact in plan.semantic_facts}
    operations = []
    for fact_id in logical.supersede_fact_ids:
        operations.append(SupersedeConstraintOperation(constraint_id=fact_id, evidence_ids=event_ids))
    for fact_id in logical.remove_fact_ids:
        operations.append(RemoveConstraintOperation(constraint_id=fact_id, evidence_ids=event_ids))
    for fact_id in logical.reveal_fact_ids:
        operations.append(_upsert_fact(facts[fact_id], fact_evidence[fact_id]))

    if logical.system_intent == PlannedSystemIntent.ASK and logical.user_event in {
        PlannedUserEvent.DISCLOSE,
        PlannedUserEvent.CHOOSE,
        PlannedUserEvent.RELAX,
    }:
        active_axis = state.unresolved_axes[0] if state.unresolved_axes else None
        if active_axis is not None:
            operations.append(ResolveAxisOperation(axis_id=active_axis.axis_id, evidence_ids=event_ids))

    feedback = _turn_feedback(logical.user_event)
    if feedback is not None:
        operations.append(RecordFeedbackOperation(feedback_type=feedback, evidence_ids=event_ids))

    terminal_reason = None
    if logical.user_event == PlannedUserEvent.ACCEPT:
        terminal_reason = TerminationReason.ACCEPTED
    elif logical.user_event == PlannedUserEvent.ABANDON:
        terminal_reason = TerminationReason.ABANDONED

    if terminal_reason is None:
        operations.append(
            SetControlOperation(
                field=ControlField.INTERACTION_PHASE,
                value=_turn_phase(logical).value,
                evidence_ids=event_ids,
            )
        )
        information = _information_after_turn(logical)
        if information is not None:
            operations.append(
                SetControlOperation(
                    field=ControlField.INFORMATION_STATUS,
                    value=information.value,
                    evidence_ids=event_ids,
                )
            )
        stability = _turn_stability(logical.user_event)
        if stability is not None:
            operations.append(
                SetControlOperation(
                    field=ControlField.INTENT_STABILITY,
                    value=stability.value,
                    evidence_ids=event_ids,
                )
            )
        if logical.user_event == PlannedUserEvent.REQUEST_EXPLANATION:
            operations.append(
                SetControlOperation(
                    field=ControlField.EXPLANATION_STATUS,
                    value=ExplanationStatus.REQUESTED.value,
                    evidence_ids=event_ids,
                )
            )
        elif logical.system_intent == PlannedSystemIntent.EXPLAIN:
            operations.append(
                SetControlOperation(
                    field=ControlField.EXPLANATION_STATUS,
                    value=ExplanationStatus.RESOLVED.value,
                    evidence_ids=system_evidence_ids,
                )
            )
    else:
        if logical.system_intent == PlannedSystemIntent.EXPLAIN:
            operations.append(
                SetControlOperation(
                    field=ControlField.EXPLANATION_STATUS,
                    value=ExplanationStatus.RESOLVED.value,
                    evidence_ids=system_evidence_ids,
                )
            )
        operations.append(TerminateOperation(reason=terminal_reason, evidence_ids=event_ids))

    return StatePatch(base_revision=state.revision, evidence=evidence, operations=operations)


def compile_logical_trajectory(
    *,
    source: RichSourceRoot,
    plan: RichLogicalTrajectoryPlan,
    variant: RenderedTrajectoryVariant,
) -> CompiledLogicalTrajectory:
    """Compile one surface variant and replay every patch immediately."""

    validate_rendered_variant(variant, source=source, plan=plan)
    state = PublicDSUSState.empty()
    history = [PublicDialogueMessage(turn_id=0, role="user", text=variant.initial_user_utterance)]
    opening_patch = _opening_patch(plan, variant)
    opening_next = reduce_state(state, opening_patch)
    transitions = [
        CompiledStateTransition(
            root_id=source.root_id,
            variant_id=variant.variant_id,
            step_id="opening",
            public_history=list(history),
            previous_state=state,
            gold_patch=opening_patch,
            gold_next_state=opening_next,
        )
    ]
    state = opening_next

    for logical, rendered in zip(plan.turns, variant.turns, strict=True):
        history.append(
            PublicDialogueMessage(
                turn_id=logical.turn_index,
                role="system",
                action_type=logical.system_intent.value,
                text=rendered.system_utterance,
            )
        )
        history.append(
            PublicDialogueMessage(
                turn_id=logical.turn_index,
                role="user",
                text=rendered.user_utterance,
            )
        )
        patch = _turn_patch(
            state=state,
            plan=plan,
            logical=logical,
            rendered=rendered,
        )
        next_state = reduce_state(state, patch)
        transitions.append(
            CompiledStateTransition(
                root_id=source.root_id,
                variant_id=variant.variant_id,
                step_id=f"turn_{logical.turn_index}",
                public_history=list(history),
                previous_state=state,
                gold_patch=patch,
                gold_next_state=next_state,
            )
        )
        state = next_state
    return CompiledLogicalTrajectory(
        root_id=source.root_id,
        variant_id=variant.variant_id,
        mechanism=plan.mechanism,
        transitions=transitions,
    )
