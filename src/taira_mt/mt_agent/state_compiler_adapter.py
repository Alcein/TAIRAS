"""Drop-in DSUS estimator backed by the trainable v2 state compiler.

The adapter is the only compatibility boundary between the new evidence-
grounded finite state and the already frozen SCTP/CIV/Agent interfaces.  It
does not predict an action.  Instead it deterministically projects a validated
``PublicDSUSState`` into the legacy decision-sufficient view consumed by the
current Thought-Pattern stack.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from taira_mt.benchmark.schemas import BenchmarkActionType
from taira_mt.models import ConstraintImportance
from taira_mt.mt_agent.memory import PublicDialogueEvent, PublicDialogueMemory
from taira_mt.mt_agent.retrieval import LexicalCatalogRetriever
from taira_mt.mt_agent.runner import LLMCompletionAudit
from taira_mt.mt_agent.state import (
    ComplexIntentShape,
    DecisionSufficientUserState,
    DialoguePhase,
    DisclosureStatus,
    DSUSDecisionPosterior,
    DSUSInference,
    FeedbackSignal,
    RecommendationReadiness,
    RetrievalFrame,
    RuleDSUSEstimator,
    explicit_direct_request,
    prefers_initial_probe,
    visible_choice_axis,
)
from taira_mt.provenance import content_sha256
from taira_mt.state_compiler import (
    ConstraintPolarity,
    ConstraintStatus,
    FeedbackType,
    InformationStatus,
    IntentStability,
    InteractionPhase,
    PublicDSUSState,
)
from taira_mt.state_compiler.runtime import (
    CompilerAttemptTelemetry,
    DSUSStateCompilerRuntime,
    StateCompilerFailure,
    StateCompilerResult,
    build_public_compiler_prompt,
)
from taira_mt.state_training.training_views import PublicTextObservation

STATE_COMPILER_ESTIMATOR_VERSION = (
    "trainable_dsus_state_compiler_adapter_v3_public_contract_recovery"
)


def _unique(values: Sequence[str], *, limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.replace("_", " ").split())
        if normalized and normalized.casefold() not in seen:
            seen.add(normalized.casefold())
            output.append(normalized)
        if len(output) >= limit:
            break
    return output


def _semantic_strings(value: object) -> list[str]:
    if isinstance(value, dict):
        if "value" in value:
            values = _semantic_strings(value["value"])
            unit = value.get("unit")
            if unit is not None:
                values = [f"{item} {unit}" for item in values]
            return values
        output: list[str] = []
        for key in sorted(value):
            output.extend(_semantic_strings(value[key]))
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(_semantic_strings(item))
        return output
    if value is None:
        return []
    return [str(value)]


def _legacy_phase(state: PublicDSUSState) -> DialoguePhase:
    mapping = {
        InteractionPhase.OPENING: DialoguePhase.INITIAL,
        InteractionPhase.ELICITATION: DialoguePhase.ELICITATION,
        InteractionPhase.GROUNDING: DialoguePhase.RECOMMENDATION,
        InteractionPhase.RECOMMENDATION: DialoguePhase.RECOMMENDATION,
        InteractionPhase.FEEDBACK: DialoguePhase.FEEDBACK,
        InteractionPhase.REPAIR: DialoguePhase.REPAIR,
        InteractionPhase.TERMINAL: DialoguePhase.FEEDBACK,
    }
    return mapping[state.control.interaction_phase]


def _legacy_disclosure(state: PublicDSUSState) -> DisclosureStatus:
    if state.control.intent_stability == IntentStability.REVISED or state.control.feedback_type in {
        FeedbackType.CORRECTION,
        FeedbackType.REJECTION,
    }:
        return DisclosureStatus.REVISED
    return {
        InformationStatus.MISSING: DisclosureStatus.MISSING_DECISIVE,
        InformationStatus.SUFFICIENT: DisclosureStatus.SUFFICIENT,
        InformationStatus.AMBIGUOUS: DisclosureStatus.AMBIGUOUS,
        InformationStatus.CONFLICTING: DisclosureStatus.CONFLICT,
    }[state.control.information_status]


def _legacy_feedback(state: PublicDSUSState) -> FeedbackSignal:
    return {
        FeedbackType.NONE: FeedbackSignal.NONE,
        FeedbackType.ANSWER: FeedbackSignal.CLARIFICATION,
        FeedbackType.ACCEPTANCE: FeedbackSignal.ACCEPTANCE,
        FeedbackType.REJECTION: FeedbackSignal.REJECTION,
        FeedbackType.CORRECTION: FeedbackSignal.REVISION,
        FeedbackType.EXPLANATION_REQUEST: FeedbackSignal.CLARIFICATION,
    }[state.control.feedback_type]


def _legacy_readiness(state: PublicDSUSState) -> RecommendationReadiness:
    if state.control.interaction_phase == InteractionPhase.TERMINAL or state.termination_reason is not None:
        return RecommendationReadiness.SAFE_STOP
    # The v2 control phase may enter ``repair`` while the user is still
    # explaining or answering.  A repair recommendation is actionable only
    # after explicit public correction/rejection evidence; phase alone is not
    # enough and would violate the legacy decision-state invariant.
    if state.control.feedback_type in {
        FeedbackType.CORRECTION,
        FeedbackType.REJECTION,
    }:
        return RecommendationReadiness.REPAIR_RECOMMEND
    # ``information_status`` is the learned, decision-level control boundary:
    # MISSING/AMBIGUOUS/CONFLICTING means that a public uncertainty blocks a
    # responsible recommendation.  ``unresolved_axes`` is semantic memory and
    # may also retain optional personalisation dimensions after the request is
    # already sufficient.  Treating every retained axis as blocking duplicated
    # the control decision and caused needless opening questions.
    if state.control.information_status in {
        InformationStatus.MISSING,
        InformationStatus.AMBIGUOUS,
        InformationStatus.CONFLICTING,
    }:
        return RecommendationReadiness.ASK_DECISIVE
    return RecommendationReadiness.RECOMMEND_NOW


def _fuse_public_control(
    state: PublicDSUSState,
    *,
    base: DecisionSufficientUserState,
    memory: PublicDialogueMemory,
) -> RecommendationReadiness:
    """Fuse learned control with visible-transition and lexical safeguards.

    The compiler state remains the semantic source of truth.  The safeguards
    only repair the compatibility boundary: an answer after an Agent ASK must
    be consumed before another ASK, and an explicit public uncertainty should
    not be silently collapsed into a recommendation merely because the
    compiler omitted one control bit.  Conversely, a known product request
    should not be blocked by a spurious ``missing`` prediction.  No benchmark
    private mechanism label is consulted here.
    """

    if memory.last_system_action_type == BenchmarkActionType.ASK:
        return RecommendationReadiness.RECOMMEND_NOW
    if memory.last_system_action_type == BenchmarkActionType.RECOMMEND:
        return RecommendationReadiness.REPAIR_RECOMMEND

    learned = _legacy_readiness(state)
    if memory.completed_action_count == 0:
        if visible_choice_axis(memory.initial_user_utterance) is not None:
            return RecommendationReadiness.ASK_DECISIVE
        if prefers_initial_probe(memory.initial_user_utterance):
            return RecommendationReadiness.PROBE_RECOMMEND
        if explicit_direct_request(memory.initial_user_utterance):
            # A literal request to recommend is a stronger public contract
            # than the deliberately conservative rule estimator's generic
            # ``best + no lexicon term`` fallback.  The frozen compiler still
            # supplies all semantic constraints; this guard only prevents a
            # false ASK caused by projecting its finite control state.
            return RecommendationReadiness.RECOMMEND_NOW
        # Do not let the deliberately coarse RuleDSUS fallback veto a valid
        # learned ``sufficient`` state.  Explicit public ASK/PROBE/RECOMMEND
        # contracts were handled above; all remaining semantic cases belong
        # to the trained compiler.  The former veto was especially harmful
        # for product nouns outside RuleDSUS's small hand-written lexicon.
    return learned


def _fuse_public_disclosure(
    state: PublicDSUSState,
    *,
    base: DecisionSufficientUserState,
    memory: PublicDialogueMemory,
    readiness: RecommendationReadiness,
) -> DisclosureStatus:
    """Keep the legacy disclosure view consistent with the fused boundary."""

    if memory.last_system_action_type == BenchmarkActionType.ASK:
        return DisclosureStatus.SUFFICIENT
    if memory.last_system_action_type == BenchmarkActionType.RECOMMEND:
        return DisclosureStatus.REVISED
    if memory.completed_action_count == 0:
        if readiness in {
            RecommendationReadiness.RECOMMEND_NOW,
            RecommendationReadiness.PROBE_RECOMMEND,
        }:
            return DisclosureStatus.SUFFICIENT
        if base.recommendation_readiness == RecommendationReadiness.ASK_DECISIVE:
            if base.disclosure_status == DisclosureStatus.AMBIGUOUS:
                return DisclosureStatus.AMBIGUOUS
            return DisclosureStatus.CONFLICT
    return _legacy_disclosure(state)


def _retrieval_frame(
    state: PublicDSUSState,
    *,
    base: RetrievalFrame,
    memory: PublicDialogueMemory,
) -> RetrievalFrame:
    must: list[str] = []
    preferred: list[str] = []
    avoid: list[str] = []
    target_candidates: list[str] = []
    target_name_cues = (
        "product_type",
        "target_type",
        "item_type",
        "specific_category",
        "category",
    )
    for constraint in state.constraints:
        if constraint.status == ConstraintStatus.SUPERSEDED:
            continue
        values = _semantic_strings(constraint.value)
        if any(cue in constraint.name for cue in target_name_cues):
            target_candidates.extend(values)
        if constraint.status == ConstraintStatus.UNCERTAIN:
            preferred.extend(values)
        elif constraint.polarity == ConstraintPolarity.AVOID:
            avoid.extend(values)
        elif constraint.polarity == ConstraintPolarity.PREFER or constraint.importance == ConstraintImportance.SOFT:
            preferred.extend(values)
        else:
            must.extend(values)
    compiled_target = next(
        (
            value
            for value in target_candidates
            if value.casefold().strip() not in {"product", "item", "goods", "option", "thing"}
        ),
        "",
    )
    rule_target = (
        base.target_type
        if base.target_type.casefold().strip() not in {"product", "item", "goods", "option", "thing"}
        else ""
    )
    public_target = _extract_public_product_head(memory.initial_user_utterance)
    return RetrievalFrame(
        target_type=compiled_target or rule_target or public_target or "product",
        must_terms=_unique([*must, *base.must_terms], limit=12),
        preferred_terms=_unique([*preferred, *base.preferred_terms], limit=12),
        avoid_terms=_unique([*avoid, *base.avoid_terms], limit=12),
    )


_PUBLIC_PRODUCT_HEAD_STARTS = (
    r"\b(?:i(?:'m|’m| am)\s+after|i\s+need|i\s+want|"
    r"i(?:'m|’m| am)\s+(?:looking|searching)\s+for|"
    r"i(?:'d|’d| would)\s+like\s+to\s+(?:find|see|buy)|"
    r"i(?:'m|’m| am)\s+interested\s+in)\s+",
    r"\b(?:can|could)\s+you\s+(?:recommend|suggest)\s+",
    r"\b(?:can|could)\s+you\s+(?:guide|point)\s+me\s+to\s+",
    r"\b(?:can|could)\s+you\s+help\s+me\s+(?:pick|choose)\s+",
    r"\bhelp\s+me\s+find\s+",
    r"\b(?:can|could)\s+you\s+show\s+me\s+",
    r"\bdo\s+you\s+have\s+(?:any\s+suggestions\s+for\s+)?",
    r"\bwhat\s+would\s+you\s+recommend\s+for\s+",
    r"\bwhat(?:'s|’s|\s+is|\s+are)\s+(?:one\s+good|the\s+best)\s+",
    r"\bwhich\s+specific\s+",
)
_PUBLIC_PRODUCT_HEAD_STOP = re.compile(
    r"\s+(?:that|which|who|whose|for|to|with|suitable|suited|focused|featuring|from|"
    r"although|though|but|considering|given|while|where|would|by|recorded|i\s+can|as\s+i)\b|[,?;]",
    flags=re.IGNORECASE,
)


def _extract_public_product_head(text: str) -> str:
    """Recover a conservative product noun phrase from the public request.

    The trained state remains authoritative whenever it emitted a product
    identity.  This parser is used only for the otherwise generic ``product``
    fallback and never turns its phrase into a hard constraint.
    """

    normalized = " ".join(text.split())
    matches = [
        match
        for pattern in _PUBLIC_PRODUCT_HEAD_STARTS
        if (match := re.search(pattern, normalized, flags=re.IGNORECASE)) is not None
    ]
    for match in sorted(matches, key=lambda item: item.start()):
        phrase = _PUBLIC_PRODUCT_HEAD_STOP.split(normalized[match.end() :], maxsplit=1)[0]
        phrase = re.sub(
            r"^(?:(?:a|an|the|some|one|any)\s+|(?:specific|good|best)\s+)+",
            "",
            phrase,
            flags=re.IGNORECASE,
        )
        words = phrase.strip(" \"'.,:;?!").split()[:10]
        while words and words[-1].casefold().strip("s") in {
            "option",
            "suggestion",
            "recommendation",
        }:
            words.pop()
        candidate = " ".join(words).strip(" \"'.,:;?!")
        if candidate.casefold() in {"", "product", "item", "goods", "option", "thing"}:
            continue
        return candidate[:160]
    return ""


def _intent_shapes(
    state: PublicDSUSState,
    *,
    base: Sequence[ComplexIntentShape],
) -> list[ComplexIntentShape]:
    shapes = list(base)
    names = {constraint.name for constraint in state.constraints if constraint.status != ConstraintStatus.SUPERSEDED}
    if state.control.feedback_type in {FeedbackType.CORRECTION, FeedbackType.REJECTION}:
        shapes.insert(0, ComplexIntentShape.FEEDBACK_REPAIR)
    if state.control.information_status == InformationStatus.AMBIGUOUS:
        shapes.append(ComplexIntentShape.TAXONOMY_AMBIGUITY)
    if any("audience" in name or "fit" in name for name in names):
        shapes.append(ComplexIntentShape.FIT_OR_AUDIENCE)
    if any("occasion" in name or "use_case" in name for name in names):
        shapes.append(ComplexIntentShape.OCCASION_TO_ATTRIBUTE)
    if any("bundle" in name for name in names):
        shapes.append(ComplexIntentShape.BUNDLE)
    if any("pair" in name or "match" in name for name in names):
        shapes.append(ComplexIntentShape.PAIRING_OR_MATCHING)
    if len([name for name in names if "target" in name or "category" in name]) > 1:
        shapes.append(ComplexIntentShape.MULTI_TARGET)
    output: list[ComplexIntentShape] = []
    for shape in shapes:
        if shape not in output:
            output.append(shape)
    return output[:4] or [ComplexIntentShape.DIRECT_ATTRIBUTE]


def project_compiler_state_for_sctp(
    state: PublicDSUSState,
    *,
    memory: PublicDialogueMemory,
    retriever: LexicalCatalogRetriever,
    estimator_id: str,
) -> DecisionSufficientUserState:
    """Deterministic compatibility projection; it never sees an evaluator label."""

    base = RuleDSUSEstimator().infer(memory=memory, retriever=retriever).state
    readiness = _fuse_public_control(state, base=base, memory=memory)
    disclosure = _fuse_public_disclosure(state, base=base, memory=memory, readiness=readiness)
    feedback = _legacy_feedback(state)
    if readiness == RecommendationReadiness.REPAIR_RECOMMEND and feedback not in {
        FeedbackSignal.REVISION,
        FeedbackSignal.REJECTION,
    }:
        # A non-terminal user turn after a recommendation is a visible repair
        # boundary in the two-action harness.  Keep the typed legacy contract
        # valid even when the compiler describes the feedback with a weaker
        # synonym such as ``none`` or ``answer``.
        feedback = FeedbackSignal.REVISION
    semantic_axes = _unique([item.description for item in state.unresolved_axes], limit=5)
    if not semantic_axes and base.unresolved_axes:
        semantic_axes = _unique(base.unresolved_axes, limit=5)
    axes = semantic_axes if readiness == RecommendationReadiness.ASK_DECISIVE else []
    if readiness == RecommendationReadiness.ASK_DECISIVE and not axes:
        axes = ["the unresolved product property visible in the current request"]
    active = [
        f"{item.name}={', '.join(_semantic_strings(item.value))}"
        for item in state.constraints
        if item.status == ConstraintStatus.ACTIVE
    ]
    summary_parts = [
        f"DSUS v2 revision {state.revision}",
        f"information={state.control.information_status.value}",
        f"stability={state.control.intent_stability.value}",
    ]
    if active:
        summary_parts.append("active public constraints: " + "; ".join(active[:6]))
    if semantic_axes:
        summary_parts.append("semantic axes: " + "; ".join(semantic_axes))
    return DecisionSufficientUserState(
        estimator_id=estimator_id,
        domain=memory.domain,
        turn_index=memory.completed_action_count,
        dialogue_phase=_legacy_phase(state),
        disclosure_status=disclosure,
        intent_shapes=_intent_shapes(state, base=base.intent_shapes),
        active_subgoal=(
            "Resolve the evidence-backed public uncertainty before retrieval."
            if readiness == RecommendationReadiness.ASK_DECISIVE
            else "Ground the active evidence-backed constraints in the public catalog."
        ),
        unresolved_axes=axes,
        feedback_signal=feedback,
        recommendation_readiness=readiness,
        retrieval_frame=_retrieval_frame(state, base=base.retrieval_frame, memory=memory),
        confidence=0.92 if state.initialized else 0.55,
        public_evidence_summary="; ".join(summary_parts)[:800],
    )


def _audit(attempt: CompilerAttemptTelemetry) -> LLMCompletionAudit:
    return LLMCompletionAudit(
        stage="dsus_state_compiler",
        attempt_index=attempt.attempt_index,
        status=attempt.status,
        requested_model=attempt.requested_model,
        returned_model=attempt.returned_model,
        finish_reason=attempt.finish_reason,
        raw_content_sha256=attempt.raw_content_sha256,
        input_tokens=attempt.input_tokens,
        output_tokens=attempt.output_tokens,
        total_tokens=attempt.total_tokens,
        latency_ms=attempt.latency_ms,
        backend_id=attempt.backend_id,
        failure_kind=attempt.failure_kind,
    )


def _memory_prefix_key(domain: str, events: Sequence[PublicDialogueEvent]) -> str:
    return content_sha256(
        {
            "domain": domain,
            "events": [event.model_dump(mode="json") for event in events],
        }
    )


def _compiler_observations(events: Sequence[PublicDialogueEvent]) -> list[PublicTextObservation]:
    return [
        PublicTextObservation(role=event.speaker, text=event.text)
        for event in events
    ]


class StateCompilerDSUSEstimator:
    """A ``DSUSEstimator`` whose only swappable component is JSON generation."""

    def __init__(
        self,
        *,
        runtime: DSUSStateCompilerRuntime,
        fallback_to_rule: bool = True,
    ) -> None:
        self.runtime = runtime
        self.fallback_to_rule = fallback_to_rule
        self.estimator_id = f"{STATE_COMPILER_ESTIMATOR_VERSION}::{runtime.compiler_id}"
        self._prefix_states: dict[str, PublicDSUSState] = {}
        self.last_compiler_state: PublicDSUSState | None = None
        self.last_compiler_results: list[StateCompilerResult] = []

    def _compile_memory(self, memory: PublicDialogueMemory) -> tuple[PublicDSUSState, list[StateCompilerResult]]:
        # A first-turn call is the unambiguous boundary between sequential
        # benchmark episodes, so cache reuse never suppresses per-episode cost.
        if memory.completed_action_count == 0:
            self._prefix_states.clear()
        state = PublicDSUSState.empty()
        results: list[StateCompilerResult] = []
        for index, event in enumerate(memory.events):
            if event.speaker != "user":
                continue
            events = memory.events[: index + 1]
            key = _memory_prefix_key(memory.domain, events)
            cached = self._prefix_states.get(key)
            if cached is not None:
                state = cached
                continue
            prompt = build_public_compiler_prompt(
                previous_state=state,
                observations=_compiler_observations(events),
            )
            result = self.runtime.compile(prompt=prompt)
            state = result.next_state
            self._prefix_states[key] = state
            results.append(result)
        return state, results

    def infer(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> DSUSInference:
        try:
            compiled_state, results = self._compile_memory(memory)
        except StateCompilerFailure as exc:
            if exc.previous_state.initialized:
                # Incremental compilation is transactional.  A malformed
                # newest patch must not erase the valid public prefix and all
                # previously grounded constraints.  The existing projection
                # consumes the latest visible action/reply to choose the safe
                # deterministic transition (ASK -> recommend, prior slate ->
                # repair) while the executor still sees the full dialogue.
                self.last_compiler_state = exc.previous_state
                self.last_compiler_results = []
                state = project_compiler_state_for_sctp(
                    exc.previous_state,
                    memory=memory,
                    retriever=retriever,
                    estimator_id=f"{self.estimator_id}:last_valid_public_transition",
                )
                return DSUSInference(
                    state=state,
                    llm_audits=[_audit(item) for item in exc.attempts],
                    used_fallback=True,
                    decision_posterior=DSUSDecisionPosterior.from_raw(
                        None,
                        selected=state.recommendation_readiness,
                        selected_probability=state.confidence,
                    ),
                )
            if not self.fallback_to_rule:
                raise
            fallback = RuleDSUSEstimator().infer(memory=memory, retriever=retriever)
            return DSUSInference(
                state=fallback.state.model_copy(
                    update={"estimator_id": f"{self.estimator_id}:rule_fallback"},
                    deep=True,
                ),
                llm_audits=[_audit(item) for item in exc.attempts],
                used_fallback=True,
                decision_posterior=DSUSDecisionPosterior.from_raw(
                    None,
                    selected=fallback.state.recommendation_readiness,
                ),
            )
        self.last_compiler_state = compiled_state
        self.last_compiler_results = results
        state = project_compiler_state_for_sctp(
            compiled_state,
            memory=memory,
            retriever=retriever,
            estimator_id=self.estimator_id,
        )
        return DSUSInference(
            state=state,
            llm_audits=[_audit(attempt) for result in results for attempt in result.attempts],
            decision_posterior=DSUSDecisionPosterior.from_raw(
                None,
                selected=state.recommendation_readiness,
                selected_probability=state.confidence,
            ),
        )


__all__ = [
    "STATE_COMPILER_ESTIMATOR_VERSION",
    "StateCompilerDSUSEstimator",
    "project_compiler_state_for_sctp",
]
