"""Matched controls for the paper's state-local experience claims.

These controls deliberately keep the public dialogue, catalog tool, base LLM,
turn budget, and downstream executor unchanged.  Each class changes one
research variable only:

* ``OpeningFrozenDSUSEstimator`` removes state updating after an observation;
* ``OpeningFrozenSCTPCompiler`` reuses the opening Pattern and its binding;
* ``mismatched_pattern_library`` keeps the Pattern scaffold but permutes its
  directive within the same layer.

They are experiment controls, not alternative production implementations.
"""

from __future__ import annotations

from copy import deepcopy

from taira_mt.mt_agent.memory import PublicDialogueMemory
from taira_mt.mt_agent.retrieval import LexicalCatalogRetriever
from taira_mt.mt_agent.sctp import SCTPActionFrontier, SCTPSkillCompiler
from taira_mt.mt_agent.skills import ThoughtPatternLibrary, ThoughtPatternSkill
from taira_mt.mt_agent.state import (
    ComplexIntentShape,
    DecisionSufficientUserState,
    DisclosureStatus,
    DSUSDecisionPosterior,
    DSUSEstimator,
    DSUSInference,
    FeedbackSignal,
    RecommendationReadiness,
    RetrievalFrame,
)


class OpeningFrozenDSUSEstimator:
    """Reuse the opening public-state interpretation after later observations."""

    def __init__(self, base: DSUSEstimator) -> None:
        self.base = base
        self.estimator_id = f"opening_frozen::{base.estimator_id}"
        self._opening_by_dialogue: dict[tuple[str, str], DSUSInference] = {}

    @staticmethod
    def _dialogue_key(memory: PublicDialogueMemory) -> tuple[str, str]:
        return memory.domain, memory.initial_user_utterance

    def infer(
        self,
        *,
        memory: PublicDialogueMemory,
        retriever: LexicalCatalogRetriever,
    ) -> DSUSInference:
        key = self._dialogue_key(memory)
        if memory.completed_action_count == 0 or key not in self._opening_by_dialogue:
            inference = self.base.infer(memory=memory, retriever=retriever)
            frozen_state = inference.state.model_copy(
                update={"estimator_id": self.estimator_id},
                deep=True,
            )
            frozen = inference.model_copy(update={"state": frozen_state}, deep=True)
            self._opening_by_dialogue[key] = frozen
            return frozen

        opening = self._opening_by_dialogue[key]
        # The cached completion was paid for and audited on the opening turn.
        # Returning its audits again would double count calls and tokens.
        return opening.model_copy(update={"llm_audits": []}, deep=True)


class CoarsePublicStateEstimator:
    """Keep only phase/feedback/readiness as a simple public-history index.

    The full dialogue remains available to Pattern lexical matching and the
    executor.  This control therefore tests the fine-grained semantic address
    supplied by DSUS, not whether the Agent is allowed to read its history.
    """

    def __init__(self, base: DSUSEstimator) -> None:
        self.base = base
        self.estimator_id = f"coarse_public_index::{base.estimator_id}"

    def infer(
        self,
        *,
        memory: PublicDialogueMemory,
        retriever: LexicalCatalogRetriever,
    ) -> DSUSInference:
        inference = self.base.infer(memory=memory, retriever=retriever)
        state = inference.state
        if state.recommendation_readiness == RecommendationReadiness.ASK_DECISIVE:
            disclosure = DisclosureStatus.MISSING_DECISIVE
            unresolved = ["the visible ranking distinction that would materially change the recommendation"]
        elif state.recommendation_readiness == RecommendationReadiness.REPAIR_RECOMMEND:
            disclosure = DisclosureStatus.REVISED
            unresolved = []
        else:
            disclosure = DisclosureStatus.SUFFICIENT
            unresolved = []
        shapes = (
            [ComplexIntentShape.FEEDBACK_REPAIR]
            if state.recommendation_readiness == RecommendationReadiness.REPAIR_RECOMMEND
            else [ComplexIntentShape.DIRECT_ATTRIBUTE]
        )
        coarse = state.model_copy(
            update={
                "estimator_id": self.estimator_id,
                "disclosure_status": disclosure,
                "intent_shapes": shapes,
                "active_subgoal": "Act from a coarse phase/feedback/readiness index over the complete public history.",
                "unresolved_axes": unresolved,
                "retrieval_frame": RetrievalFrame(target_type=state.retrieval_frame.target_type),
                "public_evidence_summary": (
                    "Coarse public index only: "
                    f"phase={state.dialogue_phase.value}; "
                    f"feedback={state.feedback_signal.value}; "
                    f"readiness={state.recommendation_readiness.value}. "
                    "The complete public dialogue remains authoritative."
                ),
            },
            deep=True,
        )
        return inference.model_copy(update={"state": coarse}, deep=True)


class FieldMaskedDSUSEstimator:
    """Remove one decision-state channel while retaining complete public history.

    The diagnostic deliberately masks both a field and its immediately derived
    signal when needed.  Otherwise, for example, ``repair_recommend`` in the
    readiness posterior would trivially reveal the feedback channel that was
    meant to be removed.  It is a small appendix diagnostic rather than a
    replacement state model.
    """

    SUPPORTED_FACTORS = frozenset({"feedback", "unresolved-axis", "readiness"})

    def __init__(self, base: DSUSEstimator, *, factor: str) -> None:
        if factor not in self.SUPPORTED_FACTORS:
            raise ValueError(
                f"unsupported DSUS factor mask {factor!r}; "
                f"expected one of {sorted(self.SUPPORTED_FACTORS)}"
            )
        self.base = base
        self.factor = factor
        self.estimator_id = f"mask_{factor}::{base.estimator_id}"

    @staticmethod
    def _without_feedback_posterior(
        posterior: DSUSDecisionPosterior | None,
    ) -> DSUSDecisionPosterior | None:
        if posterior is None:
            return None
        values = posterior.model_dump()
        repair_mass = float(values["repair_recommend"])
        values["repair_recommend"] = 0.0
        values["probe_recommend"] = float(values["probe_recommend"]) + repair_mass
        return DSUSDecisionPosterior(**values)

    @staticmethod
    def _neutral_readiness_posterior() -> DSUSDecisionPosterior:
        return DSUSDecisionPosterior(
            recommend_now=0.25,
            ask_decisive=0.25,
            probe_recommend=0.25,
            repair_recommend=0.25,
            safe_stop=0.0,
        )

    def infer(
        self,
        *,
        memory: PublicDialogueMemory,
        retriever: LexicalCatalogRetriever,
    ) -> DSUSInference:
        inference = self.base.infer(memory=memory, retriever=retriever)
        state = inference.state
        update: dict[str, object] = {
            "estimator_id": self.estimator_id,
            "public_evidence_summary": (
                f"{state.public_evidence_summary} "
                f"[Appendix diagnostic: DSUS {self.factor} channel masked.]"
            ),
        }
        posterior = inference.decision_posterior

        if self.factor == "feedback":
            update["feedback_signal"] = FeedbackSignal.NONE
            update["intent_shapes"] = [
                shape
                for shape in state.intent_shapes
                if shape != ComplexIntentShape.FEEDBACK_REPAIR
            ] or [ComplexIntentShape.DIRECT_ATTRIBUTE]
            if state.recommendation_readiness == RecommendationReadiness.REPAIR_RECOMMEND:
                update["recommendation_readiness"] = RecommendationReadiness.PROBE_RECOMMEND
            posterior = self._without_feedback_posterior(posterior)
        elif self.factor == "unresolved-axis":
            update["unresolved_axes"] = (
                ["an unspecified unresolved ranking distinction"]
                if state.unresolved_axes
                else []
            )
        else:
            update["recommendation_readiness"] = RecommendationReadiness.PROBE_RECOMMEND
            posterior = self._neutral_readiness_posterior()

        masked = state.model_copy(update=update, deep=True)
        return inference.model_copy(
            update={"state": masked, "decision_posterior": posterior},
            deep=True,
        )


class OpeningFrozenSCTPCompiler:
    """Freeze the opening Pattern frontier and binding across observations.

    This is the matched whole-task/query-level experience control.  The
    downstream CIV selector and LLM executor still run at every decision, but
    they receive the opening-boundary Pattern directive rather than a Pattern
    rematched to the new public observation.
    """

    def __init__(self, base: SCTPSkillCompiler) -> None:
        self.base = base
        self.library = base.library
        self.compiler_id = f"opening_frozen::{base.compiler_id}"
        self._opening_by_dialogue: dict[tuple[str, str], SCTPActionFrontier] = {}

    @staticmethod
    def _dialogue_key(memory: PublicDialogueMemory) -> tuple[str, str]:
        return memory.domain, memory.initial_user_utterance

    def compile(
        self,
        *,
        memory: PublicDialogueMemory,
        state: DecisionSufficientUserState,
        retriever: LexicalCatalogRetriever,
        decision_posterior=None,
    ) -> SCTPActionFrontier:
        key = self._dialogue_key(memory)
        if memory.completed_action_count == 0 or key not in self._opening_by_dialogue:
            frontier = self.base.compile(
                memory=memory,
                state=state,
                retriever=retriever,
                decision_posterior=decision_posterior,
            )
            frozen = frontier.model_copy(
                update={"compiler_id": self.compiler_id},
                deep=True,
            )
            self._opening_by_dialogue[key] = frozen
            return frozen
        return self._opening_by_dialogue[key].model_copy(deep=True)


def mismatched_pattern_library(library: ThoughtPatternLibrary) -> ThoughtPatternLibrary:
    """Return a deterministic length/scaffold-matched negative control.

    Applicability and retrieval text remain attached to each Pattern ID, while
    the procedural directive (solution, SOP, insights, query policy, and legal
    action type) is circularly permuted among Patterns in the same layer.  The
    control therefore supplies equally structured Pattern text without the
    correct scope-to-directive association.
    """

    skills = list(library.skills)
    by_layer: dict[str, list[ThoughtPatternSkill]] = {}
    for skill in skills:
        by_layer.setdefault(skill.layer.value, []).append(skill)

    donor_by_id: dict[str, ThoughtPatternSkill] = {}
    for layer_skills in by_layer.values():
        ordered = sorted(layer_skills, key=lambda item: item.pattern_id)
        if len(ordered) < 2:
            raise ValueError("mismatched Pattern control needs at least two Patterns per layer")
        for index, skill in enumerate(ordered):
            donor_by_id[skill.pattern_id] = ordered[(index + 1) % len(ordered)]

    mismatched: list[ThoughtPatternSkill] = []
    for skill in skills:
        donor = donor_by_id[skill.pattern_id]
        mismatched.append(
            skill.model_copy(
                update={
                    "version": f"{skill.version}-mismatch-control",
                    "solution_description": donor.solution_description,
                    "thought_template": deepcopy(list(donor.thought_template)),
                    "insights": deepcopy(list(donor.insights)),
                    "query_policy": donor.query_policy,
                    "action_types": deepcopy(list(donor.action_types)),
                    "provenance": (
                        "Negative control: retrieval scope retained but directive "
                        f"deterministically permuted from {donor.pattern_id}."
                    ),
                },
                deep=True,
            )
        )
    return ThoughtPatternLibrary(
        mismatched,
        source=f"mismatched-control::{library.source}",
    )
