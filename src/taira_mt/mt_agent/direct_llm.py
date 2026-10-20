"""Retrieval-grounded unstructured LLM Tool-Agent baseline for TAIRAS.

Unlike the retired direct baseline, this implementation never serializes an
entire Amazon catalog into a prompt.  The model first chooses an observable
next action and a public search specification; only if it recommends does it
receive a bounded Top-k result list from the same shared retriever used by
TAIRAS.  It has no DSUS, Thought Pattern, CIV, mechanism label, or evaluator
state.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from taira_mt.benchmark.schemas import BenchmarkActionType, BenchmarkSystemAction
from taira_mt.models import StrictModel
from taira_mt.mt_agent.memory import PublicDialogueMemory
from taira_mt.mt_agent.retrieval import CatalogRetrievalResult, LexicalCatalogRetriever, SearchSpec
from taira_mt.mt_agent.runner import LLMCompletionAudit, MTAgentDecision
from taira_mt.provenance import content_sha256
from taira_mt.providers.openai_compatible import JsonCompletion, JsonCompletionTransport, JsonResponseError

DIRECT_TOOL_POLICY_VERSION = "unstructured_llm_retrieval_tool_policy_v2"

DIRECT_TOOL_PLAN_PROMPT = """You are a direct multi-turn shopping recommendation Agent.
Use only the public dialogue and the Agent's own public action history. Do not assume a hidden user goal.
Choose one next action: ask, recommend, or stop.

Ask exactly one concise question only when one decision-critical property is genuinely missing or ambiguous.
Recommend when the current public request is sufficiently actionable. After visible feedback on a prior
recommendation, incorporate that feedback into a repair query rather than asking a generic repeat question.
For a recommendation, translate scenario language into concise product/catalog terms. Required terms must be
necessary; preferred terms are desirable; avoid terms should exclude visibly unsuitable products.

Return only one JSON object with these keys:
action_type, question_target, utterance, target_type, must_terms, preferred_terms, avoid_terms.
Use null for question_target if the action is not ask, and [] for irrelevant term lists.
"""

DIRECT_TOOL_SELECTION_PROMPT = """You are selecting a recommendation from a public retrieved catalog shortlist.
Use only the supplied dialogue, search specification, and candidate fields. Rank up to ten candidate item IDs
that best satisfy the currently visible request. Do not invent or output IDs outside the candidate list.
Return only one JSON object with exactly these keys: item_ids, utterance.
The utterance should name the leading item and briefly ground it in visible catalog evidence.
"""

REACT_TOOL_PLAN_PROMPT = """You are a ReAct-style multi-turn shopping recommendation Agent.
At every turn, first observe only the public dialogue and your own prior public actions, then decide whether
the next useful operation is to ask, recommend through the catalog-search tool, or stop. Do not assume a
hidden user goal and do not use benchmark mechanism labels.

Ask one concise question only when its answer could materially change the catalog ranking. Recommend when
the visible request is actionable. After visible rejection or revision, update the tool query and avoid
repeating the prior slate. Translate scenario language into product/catalog terms before using the tool.
Required terms must be necessary, preferred terms are desirable, and avoid terms represent visible negatives.

Do the observation and planning internally. Return only one flat JSON object with these keys:
action_type, question_target, utterance, target_type, must_terms, preferred_terms, avoid_terms.
Use null for question_target if the action is not ask, and [] for irrelevant term lists.
Never wrap the object under action, plan, tool_call, or reasoning.

Valid ask shape:
{"action_type":"ask","question_target":"the one preference to clarify","utterance":"Which do you prefer?",
 "target_type":"","must_terms":[],"preferred_terms":[],"avoid_terms":[]}
Valid catalog-search shape:
{"action_type":"recommend","question_target":null,"utterance":"I will search the catalog now.",
 "target_type":"product type","must_terms":["required property"],"preferred_terms":[],"avoid_terms":[]}
"""


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _clean_terms(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            output.append(text)
    return output


class DirectToolPlan(StrictModel):
    """First constrained LLM response: action timing plus public query grounding."""

    action_type: Literal["ask", "recommend", "stop"]
    question_target: str | None = None
    utterance: str = Field(min_length=1, max_length=800)
    target_type: str = Field(default="", max_length=160)
    must_terms: list[str] = Field(default_factory=list, max_length=12)
    preferred_terms: list[str] = Field(default_factory=list, max_length=12)
    avoid_terms: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="before")
    @classmethod
    def normalize_common_provider_shapes(cls, value: object) -> object:
        """Accept common JSON-only ReAct wrappers without relaxing action semantics.

        Some otherwise capable models return ``{"reasoning": ..., "action":
        {...}}`` even when asked for a flat object.  The reasoning is ignored
        and only the same seven public action/query fields are retained.
        """

        if not isinstance(value, Mapping):
            return value
        source = dict(value)
        nested = source.get("action")
        if isinstance(nested, Mapping):
            source = {**source, **dict(nested)}
        query = source.get("search_spec") or source.get("query")
        if isinstance(query, Mapping):
            source = {**source, **dict(query)}

        action_type = source.get("action_type") or source.get("operation") or source.get("next_action")
        if isinstance(action_type, str):
            normalized_action = action_type.strip().casefold().replace("-", "_")
            if normalized_action in {"search", "search_catalog", "tool", "use_tool", "retrieve"}:
                action_type = "recommend"
            elif normalized_action in {"clarify", "question"}:
                action_type = "ask"
            elif normalized_action in {"finish", "terminate"}:
                action_type = "stop"

        question_target = source.get("question_target") or source.get("clarification_target")
        utterance = source.get("utterance") or source.get("response") or source.get("question")
        if action_type == "ask" and question_target is None and isinstance(utterance, str):
            question_target = utterance[:160]
        return {
            "action_type": action_type,
            "question_target": question_target,
            "utterance": utterance,
            "target_type": source.get("target_type") or source.get("product_type") or "",
            "must_terms": source.get("must_terms") or source.get("required_terms") or [],
            "preferred_terms": source.get("preferred_terms") or source.get("soft_terms") or [],
            "avoid_terms": source.get("avoid_terms") or source.get("negative_terms") or [],
        }

    @field_validator("question_target", "target_type", "utterance", mode="before")
    @classmethod
    def clean_scalar(cls, value: object) -> str | None:
        if value is None:
            return None
        return _clean_text(value)

    @field_validator("must_terms", "preferred_terms", "avoid_terms", mode="before")
    @classmethod
    def clean_term_lists(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("term fields must be lists")
        return _clean_terms([str(item) for item in value])

    @model_validator(mode="after")
    def validate_action_shape(self) -> DirectToolPlan:
        if self.action_type == "ask" and not self.question_target:
            raise ValueError("ask requires a question_target")
        if self.action_type != "ask" and self.question_target is not None:
            raise ValueError("only ask may carry question_target")
        return self


class DirectCandidateSelection(StrictModel):
    """Second constrained LLM response: rank IDs from a retrieved public shortlist."""

    item_ids: list[str] = Field(min_length=1, max_length=10)
    utterance: str = Field(min_length=1, max_length=800)

    @model_validator(mode="before")
    @classmethod
    def normalize_common_provider_shapes(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        source = dict(value)
        for wrapper in ("selection", "action", "result"):
            nested = source.get(wrapper)
            if isinstance(nested, Mapping):
                source = {**source, **dict(nested)}
        item_ids = (
            source.get("item_ids")
            or source.get("selected_item_ids")
            or source.get("recommended_item_ids")
            or source.get("item_id")
        )
        if isinstance(item_ids, str):
            item_ids = [item_ids]
        utterance = source.get("utterance") or source.get("response") or source.get("answer")
        if not utterance and item_ids:
            utterance = "I recommend the selected catalog-grounded options."
        return {"item_ids": item_ids, "utterance": utterance}

    @field_validator("item_ids")
    @classmethod
    def unique_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("selected candidate IDs must be unique")
        return value


def _strict_json_object(content: str) -> dict[str, Any]:
    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"LLM JSON repeats key: {key}")
            payload[key] = value
        return payload

    try:
        payload = json.loads(content, object_pairs_hook=no_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM completion is not a JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM completion must be a JSON object")
    return payload


def _audit(
    *,
    stage: str,
    attempt_index: int,
    status: str,
    completion: JsonCompletion | None,
    failure: Exception | None = None,
) -> LLMCompletionAudit:
    if completion is None and isinstance(failure, JsonResponseError):
        return LLMCompletionAudit(stage=stage, attempt_index=attempt_index,
            status=status, failure_kind=type(failure).__name__, **failure.audit_metadata)
    return LLMCompletionAudit(
        stage=stage,
        attempt_index=attempt_index,
        status=status,
        requested_model=None if completion is None else completion.requested_model,
        returned_model=None if completion is None else completion.returned_model,
        finish_reason=None if completion is None else completion.finish_reason,
        raw_content_sha256=None if completion is None else content_sha256(completion.content),
        input_tokens=None if completion is None else completion.usage.input_tokens,
        output_tokens=None if completion is None else completion.usage.output_tokens,
        total_tokens=None if completion is None else completion.usage.total_tokens,
        reasoning_tokens=None if completion is None else completion.usage.reasoning_tokens,
        failure_kind=None if failure is None else type(failure).__name__,
    )


class DirectLLMToolPolicy:
    """Equal-tool LLM Agent without an explicit state, Skill, or value module."""

    def __init__(
        self,
        *,
        transport: JsonCompletionTransport,
        planner_id: str,
        max_attempts: int = 2,
        shortlist_k: int = 40,
        final_ranking_k: int = 10,
        plan_prompt: str = DIRECT_TOOL_PLAN_PROMPT,
        policy_version: str = DIRECT_TOOL_POLICY_VERSION,
    ) -> None:
        if not planner_id.strip():
            raise ValueError("planner_id cannot be blank")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if not 1 <= final_ranking_k <= shortlist_k <= 50:
            raise ValueError("require 1 <= final_ranking_k <= shortlist_k <= 50")
        self.transport = transport
        self.planner_id = planner_id
        self.max_attempts = max_attempts
        self.shortlist_k = shortlist_k
        self.final_ranking_k = final_ranking_k
        self.plan_prompt = plan_prompt
        self.policy_id = f"{policy_version}:{planner_id}"

    def _request_model(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_payload: Mapping[str, object],
        parser: type[DirectToolPlan] | type[DirectCandidateSelection],
    ) -> tuple[DirectToolPlan | DirectCandidateSelection | None, list[LLMCompletionAudit]]:
        audits: list[LLMCompletionAudit] = []
        retry_feedback: str | None = None
        for attempt_index in range(1, self.max_attempts + 1):
            completion: JsonCompletion | None = None
            try:
                completion = self.transport.complete_json(
                    system_prompt=system_prompt,
                    user_payload=user_payload,
                    retry_feedback=retry_feedback,
                )
                parsed = parser.model_validate(_strict_json_object(completion.content))
                audits.append(
                    _audit(
                        stage=stage,
                        attempt_index=attempt_index,
                        status="committed",
                        completion=completion,
                    )
                )
                return parsed, audits
            except Exception as exc:  # Provider, JSON, and schema errors use the same bounded retry path.
                audits.append(
                    _audit(
                        stage=stage,
                        attempt_index=attempt_index,
                        status="rejected",
                        completion=completion,
                        failure=exc,
                    )
                )
                retry_feedback = (
                    "Your previous response could not be validated. Return only one flat JSON object that strictly "
                    f"follows the requested schema; do not add prose or markdown. Error: {type(exc).__name__}: "
                    f"{str(exc)[:400]}"
                )
        return None, audits

    @staticmethod
    def _stop(*, rationale: str, audits: list[LLMCompletionAudit]) -> MTAgentDecision:
        return MTAgentDecision(
            action=BenchmarkSystemAction(
                action_type=BenchmarkActionType.STOP,
                utterance="I cannot continue because a valid public recommendation action was not produced.",
            ),
            decision_source="baseline_direct_llm_tool_guard_stop",
            rationale=rationale,
            llm_audits=audits,
        )

    @staticmethod
    def _plan_payload(memory: PublicDialogueMemory) -> dict[str, object]:
        return {
            "task": "choose_next_action_and_public_catalog_query",
            "domain": memory.domain,
            "public_dialogue": memory.serialized_dialogue,
            "asked_question_targets": memory.asked_question_targets,
            "previously_recommended_item_ids": memory.previously_recommended_item_ids,
            "instruction": "Return the strict JSON action/query object requested by the system message.",
        }

    @staticmethod
    def _selection_payload(
        *,
        memory: PublicDialogueMemory,
        result: CatalogRetrievalResult,
    ) -> dict[str, object]:
        return {
            "task": "select_ranked_catalog_items",
            "domain": memory.domain,
            "public_dialogue": memory.serialized_dialogue,
            "search_spec": result.search_spec.model_dump(mode="json"),
            "candidates": [
                {
                    "item_id": item.item_id,
                    "title": item.title,
                    "category": item.category,
                    "matched_must_terms": item.matched_must_terms,
                    "matched_preferred_terms": item.matched_preferred_terms,
                }
                for item in result.results
            ],
            "instruction": "Return the strict JSON selection object requested by the system message.",
        }

    def decide(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> MTAgentDecision:
        plan_value, plan_audits = self._request_model(
            stage="plan",
            system_prompt=self.plan_prompt,
            user_payload=self._plan_payload(memory),
            parser=DirectToolPlan,
        )
        if not isinstance(plan_value, DirectToolPlan):
            return self._stop(rationale="The unstructured Tool Agent plan failed validation.", audits=plan_audits)
        if plan_value.action_type == "stop":
            return MTAgentDecision(
                action=BenchmarkSystemAction(action_type=BenchmarkActionType.STOP, utterance=plan_value.utterance),
                decision_source=self.policy_id,
                rationale="The unstructured Tool Agent chose a public STOP action.",
                llm_audits=plan_audits,
            )
        if plan_value.action_type == "ask":
            assert plan_value.question_target is not None
            if plan_value.question_target in memory.asked_question_targets:
                return self._stop(
                    rationale="The unstructured Tool Agent repeated a public question target.",
                    audits=plan_audits,
                )
            return MTAgentDecision(
                action=BenchmarkSystemAction(
                    action_type=BenchmarkActionType.ASK,
                    question_target=plan_value.question_target,
                    utterance=plan_value.utterance,
                ),
                decision_source=self.policy_id,
                rationale="The unstructured Tool Agent selected a clarification without DSUS, Skills, or CIV.",
                llm_audits=plan_audits,
            )

        spec = SearchSpec(
            query=memory.retrieval_text,
            target_type=plan_value.target_type,
            must_terms=plan_value.must_terms,
            preferred_terms=plan_value.preferred_terms,
            avoid_terms=plan_value.avoid_terms,
            excluded_item_ids=memory.previously_recommended_item_ids,
            ranking_k=self.shortlist_k,
        )
        result = retriever.search(spec)
        if not result.results:
            return self._stop(
                rationale="The unstructured Tool Agent's search returned no public catalog candidate.",
                audits=plan_audits,
            )
        selection_value, selection_audits = self._request_model(
            stage="select",
            system_prompt=DIRECT_TOOL_SELECTION_PROMPT,
            user_payload=self._selection_payload(memory=memory, result=result),
            parser=DirectCandidateSelection,
        )
        audits = [*plan_audits, *selection_audits]
        if not isinstance(selection_value, DirectCandidateSelection):
            # The fallback remains an openly logged public retrieval rule; it
            # does not introduce an evaluator label or a TAIRAS component.
            fallback_ids = [item.item_id for item in result.results[: self.final_ranking_k]]
            return MTAgentDecision(
                action=BenchmarkSystemAction(
                    action_type=BenchmarkActionType.RECOMMEND,
                    item_ids=fallback_ids,
                    utterance=f"I found a catalog-grounded option: {result.results[0].title}.",
                ),
                search_spec=spec,
                retrieval=result,
                decision_source=f"{self.policy_id}:selection_fallback",
                rationale="The Direct LLM selection schema failed, so the shared public retrieval order was used.",
                llm_audits=audits,
            )
        retrieved_ids = {item.item_id for item in result.results}
        selected_ids = [item_id for item_id in selection_value.item_ids if item_id in retrieved_ids]
        if not selected_ids:
            return self._stop(
                rationale="The Direct LLM selected no valid ID from the public retrieved shortlist.",
                audits=audits,
            )
        return MTAgentDecision(
            action=BenchmarkSystemAction(
                action_type=BenchmarkActionType.RECOMMEND,
                item_ids=selected_ids,
                utterance=selection_value.utterance,
            ),
            search_spec=spec,
            retrieval=result,
            decision_source=self.policy_id,
            rationale="The Direct LLM planned and ranked only public retrieved candidates.",
            llm_audits=audits,
        )


class ReActLLMToolPolicy(DirectLLMToolPolicy):
    """Observe-plan-act LLM baseline with the same bounded catalog tool surface."""

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
            plan_prompt=REACT_TOOL_PLAN_PROMPT,
            policy_version="baseline_react_llm_catalog_tool_v1",
        )
