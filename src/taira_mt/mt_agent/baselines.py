"""Transparent full-catalog baselines used before proposed TAIRAS modules.

They intentionally share :class:`LexicalCatalogRetriever` with the proposed
Agent.  Differences in a later experiment therefore come from interaction
reasoning and state/skill selection rather than a hidden candidate-pool or
retrieval advantage.
"""

from __future__ import annotations

from taira_mt.benchmark.schemas import BenchmarkActionType, BenchmarkSystemAction
from taira_mt.mt_agent.executor import LLMThoughtPatternExecutor
from taira_mt.mt_agent.memory import PublicDialogueMemory
from taira_mt.mt_agent.retrieval import CatalogRetrievalResult, LexicalCatalogRetriever, SearchSpec
from taira_mt.mt_agent.runner import MTAgentDecision
from taira_mt.mt_agent.sctp import SCTPSkillCompiler
from taira_mt.mt_agent.skills import ThoughtPatternLibrary
from taira_mt.mt_agent.state import (
    DialoguePhase,
    DisclosureStatus,
    FeedbackSignal,
    RecommendationReadiness,
    RuleDSUSEstimator,
)


def _recommendation_decision(
    *,
    result: CatalogRetrievalResult,
    decision_source: str,
    rationale: str,
) -> MTAgentDecision:
    if not result.results:
        return MTAgentDecision(
            action=BenchmarkSystemAction(
                action_type=BenchmarkActionType.STOP,
                utterance="I could not find a catalog-grounded option to recommend.",
            ),
            search_spec=result.search_spec,
            retrieval=result,
            decision_source=decision_source,
            rationale=rationale,
        )
    first = result.results[0]
    item_ids = [item.item_id for item in result.results]
    return MTAgentDecision(
        action=BenchmarkSystemAction(
            action_type=BenchmarkActionType.RECOMMEND,
            item_ids=item_ids,
            utterance=(f"I found {len(item_ids)} catalog-grounded options. My leading option is {first.title}."),
        ),
        search_spec=result.search_spec,
        retrieval=result,
        decision_source=decision_source,
        rationale=rationale,
    )


class OneShotLexicalPolicy:
    """Non-interactive full-catalog retrieval control using only the first query."""

    policy_id = "baseline_one_shot_lexical_v1"

    def __init__(self, *, ranking_k: int = 10) -> None:
        self.ranking_k = ranking_k

    def decide(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> MTAgentDecision:
        if memory.completed_action_count:
            return MTAgentDecision(
                action=BenchmarkSystemAction(
                    action_type=BenchmarkActionType.STOP,
                    utterance="This one-shot baseline does not conduct a follow-up dialogue.",
                ),
                decision_source=self.policy_id,
                rationale="The baseline is intentionally restricted to one initial recommendation.",
            )
        spec = SearchSpec.from_public_query(memory.initial_user_utterance, ranking_k=self.ranking_k)
        return _recommendation_decision(
            result=retriever.search(spec),
            decision_source=self.policy_id,
            rationale="Rank the full catalog directly from the initial public request.",
        )


class ReactiveLexicalPolicy:
    """A no-state, no-pattern repair control that concatenates visible user text."""

    policy_id = "baseline_reactive_lexical_v1"

    def __init__(self, *, ranking_k: int = 10) -> None:
        self.ranking_k = ranking_k

    def decide(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> MTAgentDecision:
        spec = SearchSpec(
            query=memory.retrieval_text,
            excluded_item_ids=memory.previously_recommended_item_ids,
            ranking_k=self.ranking_k,
        )
        return _recommendation_decision(
            result=retriever.search(spec),
            decision_source=self.policy_id,
            rationale="Concatenate public user utterances and avoid previously shown item IDs; no state model or SOP.",
        )


class AskFirstLexicalPolicy:
    """A deliberately simple clarification-first control with no state assessment."""

    policy_id = "baseline_ask_first_lexical_v1"

    def __init__(self, *, ranking_k: int = 10) -> None:
        self.ranking_k = ranking_k

    def decide(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> MTAgentDecision:
        if memory.completed_action_count == 0:
            return MTAgentDecision(
                action=BenchmarkSystemAction(
                    action_type=BenchmarkActionType.ASK,
                    question_target="deciding preference",
                    utterance="Which single product property matters most for this choice?",
                ),
                decision_source=self.policy_id,
                rationale="Always ask once, without estimating whether the request is already sufficient.",
            )
        spec = SearchSpec(
            query=memory.retrieval_text,
            ranking_k=self.ranking_k,
        )
        return _recommendation_decision(
            result=retriever.search(spec),
            decision_source=self.policy_id,
            rationale="Recommend after a fixed generic question, without state-conditioned skill selection.",
        )


class StaticTAIRAPolicy:
    """Query-only Thought Pattern adaptation of the original single-turn TAIRA.

    This baseline deliberately freezes the state at the initial query, retrieves
    one semantic/interaction Pattern composition, and recommends once.  It
    therefore preserves TAIRA's scenario-to-catalog reasoning advantage while
    excluding the proposed multi-turn DSUS update and learned CIV selection.
    """

    policy_id = "baseline_static_taira_query_pattern_v1"

    def __init__(self, *, library: ThoughtPatternLibrary | None = None) -> None:
        self.library = library or ThoughtPatternLibrary.default()
        self.compiler = SCTPSkillCompiler(
            library=self.library,
            include_initial_counterfactuals=False,
        )
        self.estimator = RuleDSUSEstimator()

    def decide(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> MTAgentDecision:
        if memory.completed_action_count:
            return MTAgentDecision(
                action=BenchmarkSystemAction(
                    action_type=BenchmarkActionType.STOP,
                    utterance="This static TAIRA adaptation has completed its single query-conditioned plan.",
                ),
                decision_source=self.policy_id,
                rationale="The legacy-style Pattern is frozen at the initial query and cannot update a user state.",
            )

        query_only_memory = PublicDialogueMemory.begin(
            domain=memory.domain,
            initial_user_utterance=memory.initial_user_utterance,
        )
        initial_state = self.estimator.infer(
            memory=query_only_memory,
            retriever=retriever,
        ).state
        static_state = initial_state.model_copy(
            update={
                "dialogue_phase": DialoguePhase.RECOMMENDATION,
                "disclosure_status": DisclosureStatus.SUFFICIENT,
                "unresolved_axes": [],
                "recommendation_readiness": RecommendationReadiness.RECOMMEND_NOW,
                "public_evidence_summary": (
                    "Static query-only TAIRA adaptation: compile one recommendation plan "
                    "without multi-turn state estimation."
                ),
            },
            deep=True,
        )
        frontier = self.compiler.compile(
            memory=query_only_memory,
            state=static_state,
            retriever=retriever,
        )
        recommendations = [
            proposal for proposal in frontier.proposals if proposal.action.action_type == BenchmarkActionType.RECOMMEND
        ]
        if not recommendations:
            spec = SearchSpec.from_public_query(memory.initial_user_utterance, ranking_k=10)
            return _recommendation_decision(
                result=retriever.search(spec),
                decision_source=f"{self.policy_id}:lexical_fallback",
                rationale="No executable static Pattern recommendation was available.",
            )
        selected = max(
            recommendations,
            key=lambda proposal: (
                proposal.semantic_match_score + proposal.interaction_match_score,
                proposal.proposal_id,
            ),
        )
        return MTAgentDecision(
            action=selected.action,
            search_spec=selected.search_spec,
            retrieval=selected.retrieval,
            decision_source=self.policy_id,
            selected_skill_id=f"{selected.semantic_skill_id}+{selected.interaction_skill_id}",
            rationale=(
                "Retrieve and execute a single query-conditioned Thought Pattern composition; "
                "no dialogue-state update or CIV reranking is available."
            ),
            method_trace={
                "baseline_contract": "query_only_static_thought_pattern",
                "thought_pattern_library_id": self.library.library_id,
                "selected_proposal_id": selected.proposal_id,
                "skill_retrieval": frontier.skill_retrieval.model_dump(mode="json"),
            },
        )


class LegacyTAIRALLMPolicy:
    """Faithful recommend-first TAIRA continuation with visible-feedback replan.

    The original TAIRA does not choose between ASK and RECOMMEND.  This
    adaptation therefore always executes a query-conditioned Thought Pattern
    recommendation.  When the public environment supplies feedback after an
    initial slate, it reruns the same single-turn planning contract over the
    now-expanded public query.  It has no DSUS uncertainty, counterfactual
    action frontier, or CIV timing decision.
    """

    policy_version = "baseline_legacy_taira_recommend_first_llm_v1"

    def __init__(
        self,
        *,
        executor: LLMThoughtPatternExecutor,
        library: ThoughtPatternLibrary | None = None,
    ) -> None:
        self.library = library or ThoughtPatternLibrary.default()
        self.compiler = SCTPSkillCompiler(
            library=self.library,
            include_initial_counterfactuals=False,
        )
        self.estimator = RuleDSUSEstimator()
        self.executor = executor
        self.policy_id = f"{self.policy_version}:{self.library.library_id}:{executor.executor_id}"

    def decide(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> MTAgentDecision:
        inferred = self.estimator.infer(memory=memory, retriever=retriever).state
        is_replan = memory.completed_action_count > 0
        static_state = inferred.model_copy(
            update={
                "dialogue_phase": DialoguePhase.REPAIR if is_replan else DialoguePhase.RECOMMENDATION,
                "disclosure_status": DisclosureStatus.REVISED if is_replan else DisclosureStatus.SUFFICIENT,
                "feedback_signal": FeedbackSignal.REVISION if is_replan else FeedbackSignal.NONE,
                "unresolved_axes": [],
                "recommendation_readiness": (
                    RecommendationReadiness.REPAIR_RECOMMEND
                    if is_replan
                    else RecommendationReadiness.RECOMMEND_NOW
                ),
                "public_evidence_summary": (
                    "Legacy TAIRA visible-feedback replan: rerun one recommendation Thought Pattern."
                    if is_replan
                    else "Legacy TAIRA query-only plan: execute one recommendation Thought Pattern."
                ),
            },
            deep=True,
        )
        frontier = self.compiler.compile(
            memory=memory,
            state=static_state,
            retriever=retriever,
        )
        recommendations = [
            proposal for proposal in frontier.proposals if proposal.action.action_type == BenchmarkActionType.RECOMMEND
        ]
        if not recommendations:
            spec = SearchSpec(
                query=memory.retrieval_text,
                excluded_item_ids=memory.previously_recommended_item_ids,
                ranking_k=10,
            )
            return _recommendation_decision(
                result=retriever.search(spec),
                decision_source=f"{self.policy_id}:lexical_fallback",
                rationale="No executable legacy Thought Pattern recommendation was available.",
            )
        selected = max(
            recommendations,
            key=lambda proposal: (
                proposal.semantic_match_score + proposal.interaction_match_score,
                proposal.retrieval.results[0].score if proposal.retrieval and proposal.retrieval.results else -1.0,
                proposal.proposal_id,
            ),
        )
        execution = self.executor.execute(
            proposal=selected,
            memory=memory,
            retriever=retriever,
        )
        return MTAgentDecision(
            action=execution.action,
            search_spec=execution.search_spec,
            retrieval=execution.retrieval,
            decision_source=self.policy_version,
            selected_skill_id=f"{selected.semantic_skill_id}+{selected.interaction_skill_id}",
            rationale=(
                "Execute a single recommend-first TAIRA Thought Pattern; visible feedback may trigger "
                "a fresh replan, but no Ask/Recommend policy is learned."
            ),
            llm_audits=execution.llm_audits,
            method_trace={
                "baseline_contract": "legacy_taira_recommend_first_with_visible_feedback_replan",
                "thought_pattern_library_id": self.library.library_id,
                "selected_proposal_id": selected.proposal_id,
                "feedback_replan": is_replan,
                "skill_execution": execution.model_dump(mode="json", exclude={"action", "search_spec", "retrieval"}),
            },
        )
