"""LLM Agent baselines adapted to the frozen TAIRAS tool surface.

The module includes a fixed-timing LLM Ask-then-Recommend control plus
paper-backed adaptations that preserve one central mechanism from each cited
method. All share the same public dialogue, typed actions, complete-catalog
search, candidate selector, and LLM transport as Direct and TAIRAS.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from taira_mt.benchmark.schemas import BenchmarkActionType, BenchmarkSystemAction
from taira_mt.models import StrictModel
from taira_mt.mt_agent.direct_llm import (
    DIRECT_TOOL_PLAN_PROMPT,
    DIRECT_TOOL_SELECTION_PROMPT,
    DirectCandidateSelection,
    DirectLLMToolPolicy,
    DirectToolPlan,
)
from taira_mt.mt_agent.memory import PublicDialogueMemory
from taira_mt.mt_agent.retrieval import LexicalCatalogRetriever, SearchSpec
from taira_mt.mt_agent.runner import LLMCompletionAudit, MTAgentDecision
from taira_mt.provenance import content_sha256
from taira_mt.providers.openai_compatible import JsonCompletionTransport

CHATCRS_SOURCE = "Li et al., ChatCRS, Findings of NAACL 2025"
ITEM_CENTERED_SOURCE = "Lucas de Carvalho et al., A Flash in the Pan, COLING 2025"
RECMIND_SOURCE = "Wang et al., RecMind, Findings of NAACL 2024"

ASK_FIRST_QUESTION_PROMPT = """You are the elicitation step of a fixed Ask-then-Recommend shopping Agent.
Use only the initial public shopping request. This baseline must ask exactly one question before any catalog
recommendation, even when the request may already be actionable.

Ask one concise and concrete question about the single product property, use-case trade-off, style, format,
or constraint whose answer would most change catalog ranking. Ground it in the user's scenario. Do not ask
broad questions such as "what matters most?", "can you provide more details?", or a multi-part questionnaire.

Return only one JSON object with exactly these keys: question_target, utterance.
question_target must be a short descriptive label for the one property being asked.
"""

ASK_THEN_RECOMMEND_PROMPT = """You are the recommendation step of a fixed Ask-then-Recommend shopping Agent.
The Agent has already asked its one allowed question. Use only the complete public dialogue, including the
user's answer, and now recommend through the shared catalog tool. Do not ask another question and do not
assume any hidden user preference.

Translate the visible scenario and answer into catalog-facing product type, required properties, preferred
properties, and negative properties. Required terms must be genuinely necessary. The response is a search
plan, not an invented catalog result.

Return only one JSON object with exactly these keys:
action_type, utterance, target_type, must_terms, preferred_terms, avoid_terms.
action_type must be "recommend"; use [] for irrelevant term lists.
"""

CHATCRS_GOAL_PROMPT = """You are the goal-planning component of an adapted ChatCRS shopping Agent.
Use only the public dialogue and prior public actions. Predict one next dialogue goal:
clarify, recommend, revise, or stop.

Choose clarify only when one concrete, decision-critical preference is missing. Ask one targeted question,
not a broad request for more information. Choose recommend when the visible request is actionable. Choose
revise only after visible negative feedback on a prior recommendation. For recommend or revise, translate
the user's scenario into catalog-facing product type, required properties, preferred properties, and
negative properties. The catalog itself is the external knowledge tool and must only be invoked after a
recommend/revise goal is selected.

Return only one JSON object with exactly these keys:
dialogue_goal, question_target, utterance, target_type, must_terms, preferred_terms, avoid_terms.
Use null for question_target unless the goal is clarify, and [] for irrelevant term lists.
"""

CHATCRS_FUSION_PROMPT = """You are the conservative fusion controller of an adapted ChatCRS shopping Agent.
The DIRECT plan is a complete, independently generated base-Agent proposal. The GOAL plan is an additional
proposal from dialogue-goal planning. Both use only public dialogue, and any catalog previews are produced by
the same public retrieval tool.

Keep DIRECT by default. Choose GOAL only when it clearly fixes an observable weakness in DIRECT: a
decision-critical missing preference, premature or unnecessarily delayed recommendation, failure to react to
visible feedback, an illegal repeated question, or materially better public retrieval evidence. Do not choose
GOAL merely because its wording is nicer. Never assume a hidden user state or evaluator label.

Return only one JSON object with exactly these keys:
selected_plan_id, alternative_improves_over_direct, confidence, selection_reason.
selected_plan_id must be "direct" or "goal"; confidence must be between 0 and 1.
"""

ITEM_CENTERED_ELICITATION_PROMPT = """You are the preference-elicitation component of an item-centered
conversational recommender adapted from 'A Flash in the Pan'. Use only the initial public shopping request.

Form a short current preference hypothesis, then ask exactly one concrete, decision-critical question.
Anchor the question in a recognizable product style, feature example, or feature trade-off. Do not ask
generic questions such as 'what matters most?' or 'can you tell me more?'. The answer must be capable of
changing which catalog products should rank highest.

Return only one JSON object with exactly these keys:
preference_hypothesis, anchor_item_or_style, question_target, utterance.
"""

ITEM_CENTERED_INCREMENTAL_PROMPT = """You are the incremental recommender component of an item-centered
conversational recommender adapted from 'A Flash in the Pan'. Use only the public dialogue and the supplied
previous preference hypothesis.

First identify the new evidence in the user's latest answer. Then state how that evidence updates the
previous preference estimate. Translate the updated estimate into a catalog search specification. The
normal action is recommend; use stop only when the public user explicitly ends the task. Do not ask another
question in this bounded adapted baseline.

Return only one JSON object with exactly these keys:
action_type, new_evidence_summary, updated_preference_hypothesis, utterance,
target_type, must_terms, preferred_terms, avoid_terms.
"""

RECMIND_BRANCH_PROMPT = """You are the alternative planning branch of an adapted RecMind Self-Inspiring
recommendation Agent. Use only the public dialogue, prior public actions, and the supplied DIRECT branch.

Propose one useful next plan: ask, recommend, or stop. Ask only one decision-critical question. For recommend,
translate scenario language into catalog-facing product terms. Retain the DIRECT branch's useful insight while
exploring a materially different legal plan only when there is a public reason to do so. Otherwise provide a
careful refinement of its search terms. Do not call or invent catalog results; public previews are computed
afterward by the shared catalog tool.

Return only one JSON object with exactly these keys:
branch_id, thought_summary, action_type, question_target, utterance,
target_type, must_terms, preferred_terms, avoid_terms.
Use branch_id "B", null for question_target unless asking, and [] for irrelevant lists.
"""

RECMIND_SELECTOR_PROMPT = """You are the branch selector of an adapted RecMind Self-Inspiring Agent.
Branch A is the complete DIRECT base-Agent proposal. Branch B is a self-inspired alternative. Given the
public dialogue, both branches, and their same-tool public catalog previews, keep A by default. Choose B only
when it clearly fixes an observable weakness in A: a decision-critical missing preference, premature or
unnecessarily delayed recommendation, failure to react to visible feedback, an illegal repeated question, or
materially better public retrieval evidence. Never assume hidden user state or evaluator labels. Do not invent
a third plan or catalog result.

Return only one JSON object with exactly these keys:
selected_branch_id, branch_b_improves_over_direct, confidence, selection_reason.
selected_branch_id must be "A" or "B"; confidence must be between 0 and 1.
"""


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _clean_terms(values: object) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError("catalog term fields must be lists")
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return output


class _CatalogTermPlan(StrictModel):
    target_type: str = Field(default="", max_length=160)
    must_terms: list[str] = Field(default_factory=list, max_length=12)
    preferred_terms: list[str] = Field(default_factory=list, max_length=12)
    avoid_terms: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("target_type", mode="before")
    @classmethod
    def clean_target_type(cls, value: object) -> str:
        return _clean_text(value)

    @field_validator("must_terms", "preferred_terms", "avoid_terms", mode="before")
    @classmethod
    def clean_term_lists(cls, value: object) -> list[str]:
        return _clean_terms(value)


class AskFirstQuestion(StrictModel):
    question_target: str = Field(min_length=1, max_length=240)
    utterance: str = Field(min_length=1, max_length=800)

    @field_validator("question_target", "utterance", mode="before")
    @classmethod
    def clean_fields(cls, value: object) -> str:
        return _clean_text(value)


class AskThenRecommendPlan(_CatalogTermPlan):
    action_type: Literal["recommend"]
    utterance: str = Field(min_length=1, max_length=800)

    @field_validator("utterance", mode="before")
    @classmethod
    def clean_utterance(cls, value: object) -> str:
        return _clean_text(value)


class ChatCRSGoalPlan(_CatalogTermPlan):
    dialogue_goal: Literal["clarify", "recommend", "revise", "stop"]
    question_target: str | None = Field(default=None, max_length=240)
    utterance: str = Field(min_length=1, max_length=800)

    @field_validator("question_target", "utterance", mode="before")
    @classmethod
    def clean_text_fields(cls, value: object) -> str | None:
        if value is None:
            return None
        return _clean_text(value)

    @model_validator(mode="after")
    def validate_goal(self) -> ChatCRSGoalPlan:
        if self.dialogue_goal == "clarify" and not self.question_target:
            raise ValueError("a clarify goal requires question_target")
        if self.dialogue_goal != "clarify" and self.question_target is not None:
            raise ValueError("only clarify may carry question_target")
        return self


class ItemCenteredElicitation(StrictModel):
    preference_hypothesis: str = Field(min_length=1, max_length=800)
    anchor_item_or_style: str = Field(min_length=1, max_length=400)
    question_target: str = Field(min_length=1, max_length=240)
    utterance: str = Field(min_length=1, max_length=800)

    @field_validator(
        "preference_hypothesis",
        "anchor_item_or_style",
        "question_target",
        "utterance",
        mode="before",
    )
    @classmethod
    def clean_fields(cls, value: object) -> str:
        return _clean_text(value)


class IncrementalPreferencePlan(_CatalogTermPlan):
    action_type: Literal["recommend", "stop"]
    new_evidence_summary: str = Field(min_length=1, max_length=800)
    updated_preference_hypothesis: str = Field(min_length=1, max_length=1200)
    utterance: str = Field(min_length=1, max_length=800)

    @field_validator(
        "new_evidence_summary",
        "updated_preference_hypothesis",
        "utterance",
        mode="before",
    )
    @classmethod
    def clean_fields(cls, value: object) -> str:
        return _clean_text(value)


class RecMindBranchPlan(_CatalogTermPlan):
    branch_id: Literal["A", "B"]
    thought_summary: str = Field(min_length=1, max_length=1000)
    action_type: Literal["ask", "recommend", "stop"]
    question_target: str | None = Field(default=None, max_length=240)
    utterance: str = Field(min_length=1, max_length=800)

    @field_validator("thought_summary", "question_target", "utterance", mode="before")
    @classmethod
    def clean_text_fields(cls, value: object) -> str | None:
        if value is None:
            return None
        return _clean_text(value)

    @model_validator(mode="after")
    def validate_action(self) -> RecMindBranchPlan:
        if self.action_type == "ask" and not self.question_target:
            raise ValueError("an ask branch requires question_target")
        if self.action_type != "ask" and self.question_target is not None:
            raise ValueError("only an ask branch may carry question_target")
        return self


class RecMindBranchSelection(StrictModel):
    selected_branch_id: Literal["A", "B"]
    branch_b_improves_over_direct: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    selection_reason: str = Field(min_length=1, max_length=800)

    @field_validator("selection_reason", mode="before")
    @classmethod
    def clean_reason(cls, value: object) -> str:
        return _clean_text(value)


class ChatCRSPlanSelection(StrictModel):
    selected_plan_id: Literal["direct", "goal"]
    alternative_improves_over_direct: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    selection_reason: str = Field(min_length=1, max_length=800)

    @field_validator("selection_reason", mode="before")
    @classmethod
    def clean_reason(cls, value: object) -> str:
        return _clean_text(value)


def _public_plan_preview(
    *,
    policy: DirectLLMToolPolicy,
    memory: PublicDialogueMemory,
    retriever: LexicalCatalogRetriever,
    plan: DirectToolPlan,
) -> dict[str, object]:
    """Expose only action legality and same-tool public retrieval evidence."""

    preview: dict[str, object] = {
        "action_type": plan.action_type,
        "question_target": plan.question_target,
        "question_already_asked": (
            plan.question_target in memory.asked_question_targets
            if plan.question_target is not None
            else False
        ),
    }
    if plan.action_type != "recommend":
        return preview

    spec = SearchSpec(
        query=memory.retrieval_text,
        target_type=plan.target_type,
        must_terms=plan.must_terms,
        preferred_terms=plan.preferred_terms,
        avoid_terms=plan.avoid_terms,
        excluded_item_ids=memory.previously_recommended_item_ids,
        ranking_k=policy.shortlist_k,
    )
    result = retriever.search(spec)
    preview.update(
        {
            "search_spec": spec.model_dump(mode="json"),
            "result_count": len(result.results),
            "top_public_candidates": [
                {
                    "item_id": item.item_id,
                    "title": item.title,
                    "category": item.category,
                    "score": round(item.score, 6),
                    "matched_must_terms": item.matched_must_terms,
                    "matched_preferred_terms": item.matched_preferred_terms,
                    "matched_avoid_terms": item.matched_avoid_terms,
                }
                for item in result.results[:5]
            ],
        }
    )
    return preview


def _goal_to_direct_plan(
    *,
    goal: ChatCRSGoalPlan,
    has_previous_recommendation: bool,
) -> tuple[DirectToolPlan, str]:
    effective_goal = (
        "recommend"
        if goal.dialogue_goal == "revise" and not has_previous_recommendation
        else goal.dialogue_goal
    )
    action_type = {
        "clarify": "ask",
        "recommend": "recommend",
        "revise": "recommend",
        "stop": "stop",
    }[effective_goal]
    return (
        DirectToolPlan(
            action_type=action_type,
            question_target=goal.question_target,
            utterance=goal.utterance,
            target_type=goal.target_type,
            must_terms=goal.must_terms,
            preferred_terms=goal.preferred_terms,
            avoid_terms=goal.avoid_terms,
        ),
        effective_goal,
    )


def _direct_to_recmind_branch(plan: DirectToolPlan) -> RecMindBranchPlan:
    return RecMindBranchPlan(
        branch_id="A",
        thought_summary="Retain the complete independently generated Direct Agent proposal.",
        action_type=plan.action_type,
        question_target=plan.question_target,
        utterance=plan.utterance,
        target_type=plan.target_type,
        must_terms=plan.must_terms,
        preferred_terms=plan.preferred_terms,
        avoid_terms=plan.avoid_terms,
    )


def _guard_stop(
    *,
    policy: DirectLLMToolPolicy,
    rationale: str,
    audits: list[LLMCompletionAudit],
    method_trace: dict[str, object],
) -> MTAgentDecision:
    return MTAgentDecision(
        action=BenchmarkSystemAction(
            action_type=BenchmarkActionType.STOP,
            utterance="I cannot continue because a valid public recommendation action was not produced.",
        ),
        decision_source=f"{policy.policy_id}:guard_stop",
        rationale=rationale,
        llm_audits=audits,
        method_trace=method_trace,
    )


def _execute_selected_plan(
    *,
    policy: DirectLLMToolPolicy,
    memory: PublicDialogueMemory,
    retriever: LexicalCatalogRetriever,
    plan: DirectToolPlan,
    plan_audits: list[LLMCompletionAudit],
    method_trace: dict[str, object],
    rationale: str,
    query_context: str = "",
) -> MTAgentDecision:
    """Execute one already-selected plan; catalog search is lazy and necessary."""

    if plan.action_type == "stop":
        return MTAgentDecision(
            action=BenchmarkSystemAction(action_type=BenchmarkActionType.STOP, utterance=plan.utterance),
            decision_source=policy.policy_id,
            rationale=rationale,
            llm_audits=plan_audits,
            method_trace=method_trace,
        )
    if plan.action_type == "ask":
        assert plan.question_target is not None
        if plan.question_target in memory.asked_question_targets:
            return _guard_stop(
                policy=policy,
                rationale="The adapted baseline attempted to repeat an already asked question target.",
                audits=plan_audits,
                method_trace=method_trace,
            )
        return MTAgentDecision(
            action=BenchmarkSystemAction(
                action_type=BenchmarkActionType.ASK,
                question_target=plan.question_target,
                utterance=plan.utterance,
            ),
            decision_source=policy.policy_id,
            rationale=rationale,
            llm_audits=plan_audits,
            method_trace=method_trace,
        )

    query = _clean_text(f"{memory.retrieval_text} {query_context}")
    spec = SearchSpec(
        query=query,
        target_type=plan.target_type,
        must_terms=plan.must_terms,
        preferred_terms=plan.preferred_terms,
        avoid_terms=plan.avoid_terms,
        excluded_item_ids=memory.previously_recommended_item_ids,
        ranking_k=policy.shortlist_k,
    )
    result = retriever.search(spec)
    if not result.results:
        return _guard_stop(
            policy=policy,
            rationale="The selected paper-backed plan returned no available public catalog candidate.",
            audits=plan_audits,
            method_trace={**method_trace, "catalog_result_count": 0},
        )

    selection_value, selection_audits = policy._request_model(
        stage="catalog_candidate_selection",
        system_prompt=DIRECT_TOOL_SELECTION_PROMPT,
        user_payload=policy._selection_payload(memory=memory, result=result),
        parser=DirectCandidateSelection,
    )
    audits = [*plan_audits, *selection_audits]
    trace = {**method_trace, "catalog_result_count": len(result.results)}
    if not isinstance(selection_value, DirectCandidateSelection):
        fallback_ids = [item.item_id for item in result.results[: policy.final_ranking_k]]
        return MTAgentDecision(
            action=BenchmarkSystemAction(
                action_type=BenchmarkActionType.RECOMMEND,
                item_ids=fallback_ids,
                utterance=f"I found a catalog-grounded option: {result.results[0].title}.",
            ),
            search_spec=spec,
            retrieval=result,
            decision_source=f"{policy.policy_id}:selection_fallback",
            rationale=f"{rationale} Candidate selection used the shared retrieval-order fallback.",
            llm_audits=audits,
            method_trace=trace,
        )

    retrieved_ids = {item.item_id for item in result.results}
    selected_ids = [item_id for item_id in selection_value.item_ids if item_id in retrieved_ids]
    if not selected_ids:
        return _guard_stop(
            policy=policy,
            rationale="The candidate selector returned no ID from the public catalog shortlist.",
            audits=audits,
            method_trace=trace,
        )
    return MTAgentDecision(
        action=BenchmarkSystemAction(
            action_type=BenchmarkActionType.RECOMMEND,
            item_ids=selected_ids,
            utterance=selection_value.utterance,
        ),
        search_spec=spec,
        retrieval=result,
        decision_source=policy.policy_id,
        rationale=rationale,
        llm_audits=audits,
        method_trace=trace,
    )


class LLMAskThenRecommendPolicy(DirectLLMToolPolicy):
    """Fixed one-question timing with LLM elicitation and catalog grounding."""

    def __init__(
        self,
        *,
        transport: JsonCompletionTransport,
        planner_id: str,
        max_attempts: int = 2,
        shortlist_k: int = 40,
        final_ranking_k: int = 10,
    ) -> None:
        super().__init__(
            transport=transport,
            planner_id=planner_id,
            max_attempts=max_attempts,
            shortlist_k=shortlist_k,
            final_ranking_k=final_ranking_k,
            policy_version="baseline_llm_ask_then_recommend_v1",
        )

    def decide(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> MTAgentDecision:
        base_trace: dict[str, object] = {
            "baseline_contract": "fixed_one_llm_question_then_llm_grounded_recommendation",
            "timing_policy": "always_ask_once_then_recommend",
        }
        if memory.completed_action_count == 0:
            question_value, audits = self._request_model(
                stage="ask_first_targeted_question",
                system_prompt=ASK_FIRST_QUESTION_PROMPT,
                user_payload={
                    "task": "ask_exactly_one_decision_critical_question",
                    "domain": memory.domain,
                    "initial_user_request": memory.initial_user_utterance,
                    "instruction": "Return the strict JSON question object requested by the system message.",
                },
                parser=AskFirstQuestion,
            )
            if not isinstance(question_value, AskFirstQuestion):
                return _guard_stop(
                    policy=self,
                    rationale="The LLM targeted question could not be schema-validated.",
                    audits=audits,
                    method_trace=base_trace,
                )
            return MTAgentDecision(
                action=BenchmarkSystemAction(
                    action_type=BenchmarkActionType.ASK,
                    question_target=question_value.question_target,
                    utterance=question_value.utterance,
                ),
                decision_source=self.policy_id,
                rationale=(
                    "The fixed-timing baseline used the LLM to ask one concrete question before any "
                    "catalog recommendation."
                ),
                llm_audits=audits,
                method_trace={
                    **base_trace,
                    "elicited_question_target": question_value.question_target,
                    "catalog_tool_invoked": False,
                },
            )

        plan_value, audits = self._request_model(
            stage="ask_then_recommend_plan",
            system_prompt=ASK_THEN_RECOMMEND_PROMPT,
            user_payload={
                **self._plan_payload(memory),
                "task": "recommend_after_the_single_completed_question",
                "latest_user_answer": memory.latest_user_utterance,
                "instruction": "Do not ask again; return the strict catalog search plan.",
            },
            parser=AskThenRecommendPlan,
        )
        if not isinstance(plan_value, AskThenRecommendPlan):
            return _guard_stop(
                policy=self,
                rationale="The post-question recommendation plan could not be schema-validated.",
                audits=audits,
                method_trace=base_trace,
            )
        plan = DirectToolPlan(
            action_type="recommend",
            question_target=None,
            utterance=plan_value.utterance,
            target_type=plan_value.target_type,
            must_terms=plan_value.must_terms,
            preferred_terms=plan_value.preferred_terms,
            avoid_terms=plan_value.avoid_terms,
        )
        return _execute_selected_plan(
            policy=self,
            memory=memory,
            retriever=retriever,
            plan=plan,
            plan_audits=audits,
            method_trace={
                **base_trace,
                "asked_question_targets": memory.asked_question_targets,
                "catalog_tool_invoked": True,
            },
            rationale=(
                "After its single LLM-generated question, the fixed-timing baseline converted the public "
                "answer into a catalog search and ranked only retrieved candidates."
            ),
        )


class ChatCRSGoalGuidedPolicy(DirectLLMToolPolicy):
    """Direct Agent augmented by ChatCRS-style dialogue-goal planning."""

    def __init__(
        self,
        *,
        transport: JsonCompletionTransport,
        planner_id: str,
        max_attempts: int = 2,
        shortlist_k: int = 40,
        final_ranking_k: int = 10,
    ) -> None:
        super().__init__(
            transport=transport,
            planner_id=planner_id,
            max_attempts=max_attempts,
            shortlist_k=shortlist_k,
            final_ranking_k=final_ranking_k,
            policy_version="baseline_chatcrs_goal_guided_additive_v2",
        )

    def decide(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> MTAgentDecision:
        direct_value, direct_audits = self._request_model(
            stage="chatcrs_direct_base_plan",
            system_prompt=DIRECT_TOOL_PLAN_PROMPT,
            user_payload=self._plan_payload(memory),
            parser=DirectToolPlan,
        )
        goal_value, goal_audits = self._request_model(
            stage="chatcrs_dialogue_goal_planning",
            system_prompt=CHATCRS_GOAL_PROMPT,
            user_payload={
                **self._plan_payload(memory),
                "task": "predict_next_dialogue_goal_then_prepare_external_catalog_knowledge_query",
                "allowed_dialogue_goals": ["clarify", "recommend", "revise", "stop"],
            },
            parser=ChatCRSGoalPlan,
        )
        proposal_audits = [*direct_audits, *goal_audits]
        base_trace: dict[str, object] = {
            "baseline_contract": "direct_agent_plus_chatcrs_goal_planning_with_conservative_fusion",
            "source_paper": CHATCRS_SOURCE,
            "direct_plan_preserved": isinstance(direct_value, DirectToolPlan),
            "direct_plan": (
                direct_value.model_dump(mode="json")
                if isinstance(direct_value, DirectToolPlan)
                else None
            ),
        }
        if not isinstance(direct_value, DirectToolPlan) and not isinstance(goal_value, ChatCRSGoalPlan):
            return _guard_stop(
                policy=self,
                rationale="Neither the Direct proposal nor the ChatCRS goal proposal could be validated.",
                audits=proposal_audits,
                method_trace=base_trace,
            )

        if isinstance(goal_value, ChatCRSGoalPlan):
            goal_plan, effective_goal = _goal_to_direct_plan(
                goal=goal_value,
                has_previous_recommendation=bool(memory.previously_recommended_item_ids),
            )
        else:
            goal_plan = None
            effective_goal = None

        trace = {
            **base_trace,
            "goal_plan": None if goal_plan is None else goal_plan.model_dump(mode="json"),
            "predicted_dialogue_goal": (
                goal_value.dialogue_goal if isinstance(goal_value, ChatCRSGoalPlan) else None
            ),
            "effective_dialogue_goal": effective_goal,
        }
        if not isinstance(direct_value, DirectToolPlan):
            assert goal_plan is not None
            selected_plan = goal_plan
            selected_plan_id = "goal"
            selection_reason = "The Direct proposal failed validation; the valid goal proposal was used."
            planning_audits = proposal_audits
            trace["direct_preview"] = None
            trace["goal_preview"] = _public_plan_preview(
                policy=self,
                memory=memory,
                retriever=retriever,
                plan=goal_plan,
            )
        elif goal_plan is None:
            selected_plan = direct_value
            selected_plan_id = "direct"
            selection_reason = "The goal proposal failed validation; the complete Direct proposal was retained."
            planning_audits = proposal_audits
            trace["direct_preview"] = _public_plan_preview(
                policy=self,
                memory=memory,
                retriever=retriever,
                plan=direct_value,
            )
            trace["goal_preview"] = None
        else:
            direct_preview = _public_plan_preview(
                policy=self,
                memory=memory,
                retriever=retriever,
                plan=direct_value,
            )
            goal_preview = _public_plan_preview(
                policy=self,
                memory=memory,
                retriever=retriever,
                plan=goal_plan,
            )
            selection_value, selection_audits = self._request_model(
                stage="chatcrs_conservative_fusion",
                system_prompt=CHATCRS_FUSION_PROMPT,
                user_payload={
                    "task": "select_direct_or_goal_plan",
                    "domain": memory.domain,
                    "public_dialogue": memory.serialized_dialogue,
                    "direct_plan": direct_value.model_dump(mode="json"),
                    "direct_public_preview": direct_preview,
                    "goal_plan": goal_plan.model_dump(mode="json"),
                    "goal_public_preview": goal_preview,
                },
                parser=ChatCRSPlanSelection,
            )
            planning_audits = [*proposal_audits, *selection_audits]
            use_goal = (
                isinstance(selection_value, ChatCRSPlanSelection)
                and selection_value.selected_plan_id == "goal"
                and selection_value.alternative_improves_over_direct
                and selection_value.confidence >= 0.65
            )
            selected_plan = goal_plan if use_goal else direct_value
            selected_plan_id = "goal" if use_goal else "direct"
            selection_reason = (
                selection_value.selection_reason
                if isinstance(selection_value, ChatCRSPlanSelection)
                else "The fusion response failed validation; the Direct proposal was retained."
            )
            trace.update(
                {
                    "direct_preview": direct_preview,
                    "goal_preview": goal_preview,
                    "fusion_response": (
                        selection_value.model_dump(mode="json")
                        if isinstance(selection_value, ChatCRSPlanSelection)
                        else None
                    ),
                }
            )

        trace.update(
            {
                "selected_plan_id": selected_plan_id,
                "selection_reason": selection_reason,
                "catalog_tool_invoked_after_fusion": selected_plan.action_type == "recommend",
            }
        )
        return _execute_selected_plan(
            policy=self,
            memory=memory,
            retriever=retriever,
            plan=selected_plan,
            plan_audits=planning_audits,
            method_trace=trace,
            rationale=(
                "The adapted ChatCRS module proposed an additional dialogue-goal plan; conservative public-"
                "evidence fusion retained either it or the complete Direct Agent proposal."
            ),
        )


class ItemCenteredIncrementalPolicy(DirectLLMToolPolicy):
    """Concrete preference elicitation followed by incremental preference update."""

    def __init__(
        self,
        *,
        transport: JsonCompletionTransport,
        planner_id: str,
        max_attempts: int = 2,
        shortlist_k: int = 40,
        final_ranking_k: int = 10,
    ) -> None:
        super().__init__(
            transport=transport,
            planner_id=planner_id,
            max_attempts=max_attempts,
            shortlist_k=shortlist_k,
            final_ranking_k=final_ranking_k,
            policy_version="baseline_item_centered_incremental_crs_adapted_v1",
        )
        self._preference_hypotheses: dict[str, str] = {}

    @staticmethod
    def _conversation_key(memory: PublicDialogueMemory) -> str:
        return content_sha256(
            {
                "domain": memory.domain,
                "initial_user_utterance": memory.initial_user_utterance,
            }
        )

    def decide(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> MTAgentDecision:
        key = self._conversation_key(memory)
        base_trace: dict[str, object] = {
            "baseline_contract": "adapted_item_centered_elicitation_plus_incremental_prediction",
            "source_paper": ITEM_CENTERED_SOURCE,
        }
        if memory.completed_action_count == 0:
            elicitation_value, audits = self._request_model(
                stage="item_centered_preference_elicitation",
                system_prompt=ITEM_CENTERED_ELICITATION_PROMPT,
                user_payload={
                    "task": "form_initial_preference_hypothesis_and_ask_one_concrete_question",
                    "domain": memory.domain,
                    "initial_user_request": memory.initial_user_utterance,
                },
                parser=ItemCenteredElicitation,
            )
            if not isinstance(elicitation_value, ItemCenteredElicitation):
                return _guard_stop(
                    policy=self,
                    rationale="The item-centered elicitation could not be schema-validated.",
                    audits=audits,
                    method_trace=base_trace,
                )
            self._preference_hypotheses[key] = elicitation_value.preference_hypothesis
            return MTAgentDecision(
                action=BenchmarkSystemAction(
                    action_type=BenchmarkActionType.ASK,
                    question_target=elicitation_value.question_target,
                    utterance=elicitation_value.utterance,
                ),
                decision_source=self.policy_id,
                rationale=(
                    "The adapted item-centered policy asks one concrete product/attribute question before "
                    "incrementally updating its preference estimate."
                ),
                llm_audits=audits,
                method_trace={
                    **base_trace,
                    "preference_hypothesis": elicitation_value.preference_hypothesis,
                    "anchor_item_or_style": elicitation_value.anchor_item_or_style,
                },
            )

        previous_hypothesis = self._preference_hypotheses.get(key, memory.initial_user_utterance)
        update_value, audits = self._request_model(
            stage="item_centered_incremental_preference_update",
            system_prompt=ITEM_CENTERED_INCREMENTAL_PROMPT,
            user_payload={
                "task": "incrementally_update_preference_then_prepare_catalog_query",
                "domain": memory.domain,
                "public_dialogue": memory.serialized_dialogue,
                "previous_preference_hypothesis": previous_hypothesis,
                "latest_user_evidence": memory.latest_user_utterance,
                "previously_recommended_item_ids": memory.previously_recommended_item_ids,
            },
            parser=IncrementalPreferencePlan,
        )
        if not isinstance(update_value, IncrementalPreferencePlan):
            return _guard_stop(
                policy=self,
                rationale="The incremental preference update could not be schema-validated.",
                audits=audits,
                method_trace={**base_trace, "previous_preference_hypothesis": previous_hypothesis},
            )
        self._preference_hypotheses[key] = update_value.updated_preference_hypothesis
        plan = DirectToolPlan(
            action_type=update_value.action_type,
            question_target=None,
            utterance=update_value.utterance,
            target_type=update_value.target_type,
            must_terms=update_value.must_terms,
            preferred_terms=update_value.preferred_terms,
            avoid_terms=update_value.avoid_terms,
        )
        return _execute_selected_plan(
            policy=self,
            memory=memory,
            retriever=retriever,
            plan=plan,
            plan_audits=audits,
            method_trace={
                **base_trace,
                "previous_preference_hypothesis": previous_hypothesis,
                "new_evidence_summary": update_value.new_evidence_summary,
                "updated_preference_hypothesis": update_value.updated_preference_hypothesis,
            },
            rationale=(
                "The adapted item-centered policy explicitly revised its previous preference hypothesis "
                "before calling the shared catalog tool."
            ),
            query_context=update_value.updated_preference_hypothesis,
        )


class RecMindSelfInspiringPolicy(DirectLLMToolPolicy):
    """Direct Agent augmented by one bounded RecMind-style alternative branch."""

    def __init__(
        self,
        *,
        transport: JsonCompletionTransport,
        planner_id: str,
        max_attempts: int = 2,
        shortlist_k: int = 40,
        final_ranking_k: int = 10,
    ) -> None:
        super().__init__(
            transport=transport,
            planner_id=planner_id,
            max_attempts=max_attempts,
            shortlist_k=shortlist_k,
            final_ranking_k=final_ranking_k,
            policy_version="baseline_recmind_self_inspiring_additive_v2",
        )

    def _branch_payload(
        self,
        *,
        memory: PublicDialogueMemory,
        direct_branch: RecMindBranchPlan,
        direct_preview: dict[str, object],
    ) -> dict[str, object]:
        return {
            **self._plan_payload(memory),
            "task": "generate_one_self_inspiring_alternative_to_direct",
            "required_branch_id": "B",
            "direct_branch": direct_branch.model_dump(mode="json"),
            "direct_public_preview": direct_preview,
        }

    def decide(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> MTAgentDecision:
        base_trace: dict[str, object] = {
            "baseline_contract": "direct_agent_plus_recmind_self_inspiring_alternative",
            "source_paper": RECMIND_SOURCE,
        }
        direct_value, direct_audits = self._request_model(
            stage="recmind_direct_base_plan",
            system_prompt=DIRECT_TOOL_PLAN_PROMPT,
            user_payload=self._plan_payload(memory),
            parser=DirectToolPlan,
        )
        if not isinstance(direct_value, DirectToolPlan):
            return _guard_stop(
                policy=self,
                rationale="The complete Direct base proposal could not be schema-validated.",
                audits=direct_audits,
                method_trace=base_trace,
            )
        branch_a = _direct_to_recmind_branch(direct_value)
        direct_preview = _public_plan_preview(
            policy=self,
            memory=memory,
            retriever=retriever,
            plan=direct_value,
        )
        branch_b, audits_b = self._request_model(
            stage="recmind_branch_b",
            system_prompt=RECMIND_BRANCH_PROMPT,
            user_payload=self._branch_payload(
                memory=memory,
                direct_branch=branch_a,
                direct_preview=direct_preview,
            ),
            parser=RecMindBranchPlan,
        )
        branch_audits = [*direct_audits, *audits_b]
        trace = {
            **base_trace,
            "direct_plan_preserved": True,
            "branch_a_source": "complete_direct_agent_proposal",
            "branch_a": branch_a.model_dump(mode="json"),
            "direct_preview": direct_preview,
        }
        if not isinstance(branch_b, RecMindBranchPlan) or branch_b.branch_id != "B":
            return _execute_selected_plan(
                policy=self,
                memory=memory,
                retriever=retriever,
                plan=direct_value,
                plan_audits=branch_audits,
                method_trace={
                    **trace,
                    "branch_b": None,
                    "selected_branch_id": "A",
                    "selection_reason": (
                        "The self-inspired alternative failed validation; the complete Direct proposal "
                        "was retained."
                    ),
                },
                rationale="RecMind alternative generation failed, so the complete Direct proposal was retained.",
            )
        branch_b_plan = DirectToolPlan(
            action_type=branch_b.action_type,
            question_target=branch_b.question_target,
            utterance=branch_b.utterance,
            target_type=branch_b.target_type,
            must_terms=branch_b.must_terms,
            preferred_terms=branch_b.preferred_terms,
            avoid_terms=branch_b.avoid_terms,
        )
        branch_b_preview = _public_plan_preview(
            policy=self,
            memory=memory,
            retriever=retriever,
            plan=branch_b_plan,
        )
        selection_value, selection_audits = self._request_model(
            stage="recmind_branch_selection",
            system_prompt=RECMIND_SELECTOR_PROMPT,
            user_payload={
                "task": "select_retained_self_inspiring_branch",
                "domain": memory.domain,
                "public_dialogue": memory.serialized_dialogue,
                "branch_a": branch_a.model_dump(mode="json"),
                "branch_a_public_preview": direct_preview,
                "branch_b": branch_b.model_dump(mode="json"),
                "branch_b_public_preview": branch_b_preview,
            },
            parser=RecMindBranchSelection,
        )
        planning_audits = [*branch_audits, *selection_audits]
        trace.update(
            {
                "branch_b": branch_b.model_dump(mode="json"),
                "branch_b_preview": branch_b_preview,
                "selector_response": (
                    selection_value.model_dump(mode="json")
                    if isinstance(selection_value, RecMindBranchSelection)
                    else None
                ),
            }
        )
        use_branch_b = (
            isinstance(selection_value, RecMindBranchSelection)
            and selection_value.selected_branch_id == "B"
            and selection_value.branch_b_improves_over_direct
            and selection_value.confidence >= 0.65
        )
        selected_plan = branch_b_plan if use_branch_b else direct_value
        selected_branch_id = "B" if use_branch_b else "A"
        selection_reason = (
            selection_value.selection_reason
            if isinstance(selection_value, RecMindBranchSelection)
            else "The branch selector failed validation; the complete Direct proposal was retained."
        )
        return _execute_selected_plan(
            policy=self,
            memory=memory,
            retriever=retriever,
            plan=selected_plan,
            plan_audits=planning_audits,
            method_trace={
                **trace,
                "selected_branch_id": selected_branch_id,
                "selection_reason": selection_reason,
                "catalog_tool_invoked_after_selection": selected_plan.action_type == "recommend",
            },
            rationale=(
                "The adapted RecMind module generated one self-inspired alternative to the complete Direct "
                "proposal; conservative public-evidence selection retained one branch."
            ),
        )


__all__ = [
    "ChatCRSGoalGuidedPolicy",
    "ItemCenteredIncrementalPolicy",
    "LLMAskThenRecommendPolicy",
    "RecMindSelfInspiringPolicy",
]
