"""Equal-tool Direct-Agent baseline over the public session facade.

This module is intentionally separate from the historical difficulty-
calibration baseline.  It gives one injectable planner the complete public
dialogue and the exact public catalog, then compiles its response into the
canonical :class:`BenchmarkSystemAction`.  It never imports the raw benchmark
environment, loaded case, DSUS, Pattern library, or CIV.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from taira_mt.agent.action_guard import (
    PUBLIC_ACTION_GUARD_VERSION,
    PUBLIC_QUESTION_TARGETS,
    PublicActionGuardError,
    PublicActionLedger,
    validate_public_action,
)
from taira_mt.agent.catalog_tools import PublicCatalogItem, PublicCatalogTool
from taira_mt.agent.public_language import is_explicit_public_rejection
from taira_mt.agent.session import PublicAgentSession
from taira_mt.benchmark.schemas import (
    BENCHMARK_ACTION_SCHEMA_VERSION,
    BenchmarkActionType,
    BenchmarkSystemAction,
    EvidenceReference,
    PublicEnvironmentResponse,
    RelaxationProposal,
)
from taira_mt.models import StrictModel
from taira_mt.provenance import content_sha256
from taira_mt.providers.openai_compatible import JsonCompletion, JsonCompletionTransport, JsonTokenUsage

DIRECT_AGENT_RUNNER_VERSION = "equal_tool_direct_agent_runner_v1"
DIRECT_AGENT_REQUEST_SCHEMA_VERSION = "equal_tool_direct_agent_request_v1"

# A context key is forbidden when it would expose evaluator-only state or a
# TAIRAS internal intermediate.  The audit is deliberately key-based: public
# user language may naturally contain words such as "reward" or "pattern".
_FORBIDDEN_CONTEXT_KEYS = frozenset(
    {
        "case_id",
        "source_user_id",
        "split_group_id",
        "scenario",
        "scenario_family",
        "private",
        "latent_goal",
        "goal_version",
        "hidden_constraint",
        "acceptable_item_id",
        "acceptable_item_ids",
        "runtime_user_state",
        "response_frame",
        "oracle",
        "assessment",
        "reward",
        "patience",
        "satisfaction",
        "evaluator",
        "dsus",
        "pattern",
        "civ",
        "checkpoint",
        "coverage_policy",
    }
)

_CANONICAL_ACTION_KEYS = frozenset(
    {
        "schema_version",
        "action_type",
        "question_target",
        "item_ids",
        "relaxation_proposal",
        "evidence_refs",
        "utterance",
    }
)


DIRECT_AGENT_SYSTEM_PROMPT = """You are a direct multi-turn shopping recommender.
Choose the next system action using only the supplied public dialogue, public action ledger, and public catalog.
Balance useful clarification with timely recommendation. Do not ask a question target that appears in the ledger.
Recommend exactly one catalog item ID when recommending. Explain only an item in the ledger's explainable IDs,
using only fields that are present for that item in the catalog. The relaxation action is unavailable in this run.
Return one strict JSON object and no markdown, rationale, or extra keys. It must contain exactly these keys:
schema_version, action_type, question_target, item_ids, relaxation_proposal, evidence_refs, utterance.
Use null or [] for irrelevant fields. schema_version must be user_benchmark_action_v1.
action_type must be ask, recommend, explain, or stop. For ask, question_target must be exactly one of:
occasion_interpretation, specific_category, price, rating, popularity_percentile.
"""


class DirectPlannerCallable(Protocol):
    """Callable alternative to the shared JSON transport interface."""

    def __call__(
        self,
        *,
        system_prompt: str,
        user_payload: Mapping[str, object],
        retry_feedback: str | None = None,
    ) -> JsonCompletion: ...


DirectPlanner = JsonCompletionTransport | DirectPlannerCallable


class DirectPublicDialogueEvent(StrictModel):
    """One public dialogue event shown to the direct planner."""

    speaker: Literal["user", "system"]
    text: str = Field(min_length=1)
    system_action: BenchmarkSystemAction | None = None

    @model_validator(mode="after")
    def validate_event_shape(self) -> DirectPublicDialogueEvent:
        if self.speaker == "system" and self.system_action is None:
            raise ValueError("a direct system dialogue event needs its typed action")
        if self.speaker == "user" and self.system_action is not None:
            raise ValueError("a direct user dialogue event cannot carry a system action")
        return self


class DirectPublicActionLedger(StrictModel):
    """Direct policy's public action-history summary, without a text parser."""

    asked_question_targets: list[str] = Field(default_factory=list)
    recommended_item_ids: list[str] = Field(default_factory=list)
    rejected_item_ids: list[str] = Field(default_factory=list)
    explainable_item_ids: list[str] = Field(default_factory=list)
    completed_turn_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_unique_values(self) -> DirectPublicActionLedger:
        for field_name in (
            "asked_question_targets",
            "recommended_item_ids",
            "rejected_item_ids",
            "explainable_item_ids",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must contain unique public IDs")
        return self

    def structural_ledger(self) -> PublicActionLedger:
        """Return exactly the public facts required by the shared guard."""

        return PublicActionLedger.from_public_memory(
            asked_question_targets=self.asked_question_targets,
            rejected_item_ids=self.rejected_item_ids,
            explainable_item_ids=self.explainable_item_ids,
            completed_turn_count=self.completed_turn_count,
        )

    def record_system_action(self, action: BenchmarkSystemAction) -> DirectPublicActionLedger:
        asked = list(self.asked_question_targets)
        recommended = list(self.recommended_item_ids)
        explainable = list(self.explainable_item_ids)
        if action.action_type == BenchmarkActionType.ASK:
            assert action.question_target is not None
            asked.append(action.question_target)
        if action.action_type == BenchmarkActionType.RECOMMEND:
            for item_id in action.item_ids:
                if item_id not in recommended:
                    recommended.append(item_id)
            explainable = list(action.item_ids)
        return self.model_copy(
            update={
                "asked_question_targets": asked,
                "recommended_item_ids": recommended,
                "explainable_item_ids": explainable,
            },
            deep=True,
        )

    def record_user_response(
        self,
        *,
        preceding_action: BenchmarkSystemAction,
        response: PublicEnvironmentResponse,
    ) -> DirectPublicActionLedger:
        """Update only facts unambiguously visible in public text.

        The Direct Agent deliberately does not reuse TAIRAS's structured
        interpreter.  A small lexical detector is sufficient for a structural
        no-repeat guard: it marks an item rejected only when the public user
        text explicitly communicates rejection.  Ambiguous text is left to the
        planner rather than silently converted to an oracle label.
        """

        rejected = list(self.rejected_item_ids)
        explainable = list(self.explainable_item_ids)
        if preceding_action.action_type == BenchmarkActionType.RECOMMEND and _is_public_rejection(
            response.utterance
        ):
            for item_id in preceding_action.item_ids:
                if item_id not in rejected:
                    rejected.append(item_id)
            explainable = [item_id for item_id in explainable if item_id not in set(rejected)]
        return self.model_copy(
            update={
                "rejected_item_ids": rejected,
                "explainable_item_ids": explainable,
                "completed_turn_count": self.completed_turn_count + 1,
            },
            deep=True,
        )


def _is_public_rejection(utterance: str) -> bool:
    """Recognize only explicit rejection language visible to the Direct policy.

    This is deliberately a small fixed public-text guard, not a call into the
    user simulator or TAIRAS's state interpreter.  The frozen language bank
    expresses an unambiguous rejection through several ordinary paraphrases
    (for example, ``none of those work`` and ``cannot verify``), so checking
    only the literal word ``reject`` would make the supposedly shared
    no-repeat contract needlessly weaker for the direct baseline.
    """

    return is_explicit_public_rejection(utterance)


class DirectPlannerContext(StrictModel):
    """The complete, auditable public input to one Direct-Agent decision."""

    request_schema_version: Literal[DIRECT_AGENT_REQUEST_SCHEMA_VERSION]
    task: Literal["choose_next_system_action"]
    domain: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    terminal: bool
    action_schema_version: Literal[BENCHMARK_ACTION_SCHEMA_VERSION]
    action_guard_version: Literal[PUBLIC_ACTION_GUARD_VERSION]
    recommendation_top_k: Literal[1]
    propose_relax_enabled: Literal[False]
    question_targets: list[str] = Field(min_length=1)
    catalog_field_semantics: dict[str, str] = Field(min_length=1)
    dialogue: list[DirectPublicDialogueEvent] = Field(min_length=1)
    action_ledger: DirectPublicActionLedger
    catalog: list[PublicCatalogItem] = Field(min_length=1)
    catalog_snapshot_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_public_payload(self) -> DirectPlannerContext:
        if self.question_targets != sorted(PUBLIC_QUESTION_TARGETS):
            raise ValueError("direct planner context must expose the shared question-target contract")
        expected_catalog_hash = content_sha256(
            [item.model_dump(mode="json") for item in self.catalog]
        )
        if self.catalog_snapshot_sha256 != expected_catalog_hash:
            raise ValueError("direct planner context catalog snapshot hash does not match catalog order/content")
        assert_direct_context_is_public(self.model_dump(mode="json"))
        return self

    def model_inputs(self) -> dict[str, object]:
        payload = self.model_dump(mode="json")
        assert_direct_context_is_public(payload)
        return payload


class DirectActionPayload(StrictModel):
    """Strict top-level JSON contract used by a direct planner completion."""

    schema_version: Literal[BENCHMARK_ACTION_SCHEMA_VERSION]
    action_type: BenchmarkActionType
    question_target: str | None
    item_ids: list[str]
    relaxation_proposal: RelaxationProposal | None
    evidence_refs: list[EvidenceReference]
    utterance: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="after")
    def validate_canonical_action_shape(self) -> DirectActionPayload:
        # Reuse the canonical benchmark action validator, but keep this separate
        # model so every direct completion must explicitly include all keys.
        BenchmarkSystemAction.model_validate(self.model_dump(mode="json"))
        return self

    def compile(self) -> BenchmarkSystemAction:
        return BenchmarkSystemAction.model_validate(self.model_dump(mode="json"))


class DirectAgentAttempt(StrictModel):
    """Secret-free planner attempt telemetry; raw provider text is never saved."""

    attempt_index: int = Field(ge=1)
    status: Literal["committed", "rejected"]
    provider_request_id: str | None = None
    requested_model: str | None = None
    returned_model: str | None = None
    finish_reason: str | None = None
    raw_content_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    usage: JsonTokenUsage | None = None
    failure_kind: str | None = None


class DirectAgentTurn(StrictModel):
    """One executed public decision and its public simulator response."""

    turn_index: int = Field(ge=0)
    planner_context_sha256: str = Field(min_length=64, max_length=64)
    attempts: list[DirectAgentAttempt] = Field(min_length=1)
    generation_or_guard_failure: bool
    parsed_action: BenchmarkSystemAction | None = None
    executed_action: BenchmarkSystemAction
    public_response: PublicEnvironmentResponse
    terminal_after_turn: bool


class DirectAgentEpisode(StrictModel):
    """A public transcript fit for the shared evaluator, with no case metadata."""

    runner_id: str
    planner_id: str
    initial_user_utterance: str = Field(min_length=1)
    turns: list[DirectAgentTurn]
    final_ledger: DirectPublicActionLedger
    terminal: bool
    reached_turn_limit: bool

    def model_inputs(self) -> dict[str, object]:
        """Return only the public rollout record suitable for reproducibility."""

        return self.model_dump(mode="json")


def _normalized_context_key(value: object) -> str:
    raw = str(value).casefold()
    return "".join(character if character.isalnum() else "_" for character in raw).strip("_")


def assert_direct_context_is_public(payload: object) -> None:
    """Reject forbidden context fields recursively before a planner can see them."""

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                normalized = _normalized_context_key(key)
                if normalized in _FORBIDDEN_CONTEXT_KEYS:
                    raise ValueError(f"Direct-Agent context contains forbidden field at {path}.{key}")
                visit(nested, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")

    visit(payload, "context")


def build_direct_planner_context(
    *,
    domain: str,
    terminal: bool,
    dialogue: list[DirectPublicDialogueEvent],
    ledger: DirectPublicActionLedger,
    catalog: PublicCatalogTool,
) -> DirectPlannerContext:
    """Materialize a public, complete-catalog Direct-Agent request deterministically."""

    public_items = list(catalog.public_items())
    known_slots = set(catalog.attribute_slots)
    semantics = {
        "title": "public product title",
        **{
            slot: _catalog_field_semantics(slot)
            for slot in sorted(known_slots)
        },
    }
    return DirectPlannerContext(
        request_schema_version=DIRECT_AGENT_REQUEST_SCHEMA_VERSION,
        task="choose_next_system_action",
        domain=domain,
        turn_index=ledger.completed_turn_count,
        terminal=terminal,
        action_schema_version=BENCHMARK_ACTION_SCHEMA_VERSION,
        action_guard_version=PUBLIC_ACTION_GUARD_VERSION,
        recommendation_top_k=1,
        propose_relax_enabled=False,
        question_targets=sorted(PUBLIC_QUESTION_TARGETS),
        catalog_field_semantics=semantics,
        dialogue=dialogue,
        action_ledger=ledger,
        catalog=public_items,
        catalog_snapshot_sha256=content_sha256(
            [item.model_dump(mode="json") for item in public_items]
        ),
    )


def _catalog_field_semantics(slot: str) -> str:
    known = {
        "price": "public price; lower values are cheaper",
        "rating": "public rating; larger values indicate higher rating",
        "popularity_percentile": "public popularity percentile in [0, 1]; larger values are more popular",
        "specific_category": "public fine-grained product taxonomy",
        "broad_category": "public broad product taxonomy",
        "category_tags": "public product taxonomy tags",
    }
    return known.get(slot, "public catalog attribute")


def parse_direct_action_payload(content: str) -> DirectActionPayload:
    """Parse strict JSON while rejecting duplicate, missing, and extra action keys."""

    def no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"direct planner JSON repeats key: {key}")
            payload[key] = value
        return payload

    try:
        raw: Any = json.loads(content, object_pairs_hook=no_duplicate_keys)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("direct planner completion is not a strict JSON object") from exc
    if not isinstance(raw, dict):
        raise ValueError("direct planner completion must be a JSON object")
    keys = frozenset(raw)
    if keys != _CANONICAL_ACTION_KEYS:
        missing = sorted(_CANONICAL_ACTION_KEYS.difference(keys))
        extra = sorted(keys.difference(_CANONICAL_ACTION_KEYS))
        raise ValueError(f"direct planner JSON keys differ from canonical contract: missing={missing}, extra={extra}")
    return DirectActionPayload.model_validate(raw)


def _fallback_stop_action() -> BenchmarkSystemAction:
    return BenchmarkSystemAction(
        action_type=BenchmarkActionType.STOP,
        utterance="I cannot continue because a valid public action could not be generated.",
    )


class DirectAgentRunner:
    """Facade-only direct policy with a pluggable JSON planner.

    The runner intentionally accepts no case identifier or raw environment.
    A provider is injected as a transport or simple callable, which allows all
    structural and evaluator tests to run offline with deterministic fakes.
    """

    runner_id = DIRECT_AGENT_RUNNER_VERSION

    def __init__(
        self,
        *,
        planner: DirectPlanner,
        planner_id: str,
        max_attempts: int = 2,
        system_prompt: str = DIRECT_AGENT_SYSTEM_PROMPT,
    ) -> None:
        if not planner_id.strip():
            raise ValueError("direct planner ID cannot be blank")
        if max_attempts < 1:
            raise ValueError("direct planner max_attempts must be positive")
        if not system_prompt.strip():
            raise ValueError("direct planner system prompt cannot be blank")
        self.planner = planner
        self.planner_id = planner_id
        self.max_attempts = max_attempts
        self.system_prompt = system_prompt

    def _complete(
        self,
        *,
        context: DirectPlannerContext,
        retry_feedback: str | None,
    ) -> JsonCompletion:
        planner = self.planner
        if hasattr(planner, "complete_json"):
            return planner.complete_json(
                system_prompt=self.system_prompt,
                user_payload=context.model_inputs(),
                retry_feedback=retry_feedback,
            )
        return planner(
            system_prompt=self.system_prompt,
            user_payload=context.model_inputs(),
            retry_feedback=retry_feedback,
        )

    def _choose_action(
        self,
        *,
        context: DirectPlannerContext,
        catalog: PublicCatalogTool,
        ledger: DirectPublicActionLedger,
    ) -> tuple[BenchmarkSystemAction | None, BenchmarkSystemAction, list[DirectAgentAttempt]]:
        attempts: list[DirectAgentAttempt] = []
        retry_feedback: str | None = None
        parsed_action: BenchmarkSystemAction | None = None
        for attempt_index in range(1, self.max_attempts + 1):
            completion: JsonCompletion | None = None
            try:
                completion = self._complete(context=context, retry_feedback=retry_feedback)
                payload = parse_direct_action_payload(completion.content)
                candidate = payload.compile()
                validate_public_action(
                    candidate,
                    catalog=catalog,
                    ledger=ledger.structural_ledger(),
                    maximum_recommendation_items=1,
                    allow_propose_relax=False,
                )
                parsed_action = candidate
                attempts.append(
                    DirectAgentAttempt(
                        attempt_index=attempt_index,
                        status="committed",
                        provider_request_id=completion.provider_request_id,
                        requested_model=completion.requested_model,
                        returned_model=completion.returned_model,
                        finish_reason=completion.finish_reason,
                        raw_content_sha256=content_sha256(completion.content),
                        usage=completion.usage,
                    )
                )
                return parsed_action, parsed_action, attempts
            except Exception as exc:  # Provider and schema/guard failures share the retry budget.
                attempts.append(
                    DirectAgentAttempt(
                        attempt_index=attempt_index,
                        status="rejected",
                        provider_request_id=(
                            completion.provider_request_id if completion is not None else None
                        ),
                        requested_model=(completion.requested_model if completion is not None else None),
                        returned_model=(completion.returned_model if completion is not None else None),
                        finish_reason=(completion.finish_reason if completion is not None else None),
                        raw_content_sha256=(
                            content_sha256(completion.content) if completion is not None else None
                        ),
                        usage=(completion.usage if completion is not None else None),
                        failure_kind=type(exc).__name__,
                    )
                )
                retry_feedback = _retry_feedback(exc)
        return None, _fallback_stop_action(), attempts

    def run(self, session: PublicAgentSession, *, maximum_turns: int = 6) -> DirectAgentEpisode:
        """Run against only a :class:`PublicAgentSession` and public catalog tools."""

        if maximum_turns < 1:
            raise ValueError("direct Agent maximum_turns must be positive")
        initial = session.reset()
        dialogue = [DirectPublicDialogueEvent(speaker="user", text=initial)]
        ledger = DirectPublicActionLedger()
        turns: list[DirectAgentTurn] = []
        while not session.terminal and len(turns) < maximum_turns:
            context = build_direct_planner_context(
                domain=session.domain,
                terminal=session.terminal,
                dialogue=dialogue,
                ledger=ledger,
                catalog=session.catalog,
            )
            parsed_action, action, attempts = self._choose_action(
                context=context,
                catalog=session.catalog,
                ledger=ledger,
            )
            # Fallback STOP is itself checked through the same public contract.
            validate_public_action(
                action,
                catalog=session.catalog,
                ledger=ledger.structural_ledger(),
                maximum_recommendation_items=1,
                allow_propose_relax=False,
            )
            ledger_after_action = ledger.record_system_action(action)
            response = session.step(action)
            dialogue.append(
                DirectPublicDialogueEvent(
                    speaker="system",
                    text=action.utterance,
                    system_action=action,
                )
            )
            dialogue.append(DirectPublicDialogueEvent(speaker="user", text=response.utterance))
            ledger = ledger_after_action.record_user_response(
                preceding_action=action,
                response=response,
            )
            turns.append(
                DirectAgentTurn(
                    turn_index=len(turns),
                    planner_context_sha256=content_sha256(context.model_inputs()),
                    attempts=attempts,
                    generation_or_guard_failure=parsed_action is None,
                    parsed_action=parsed_action,
                    executed_action=action,
                    public_response=response,
                    terminal_after_turn=response.terminal,
                )
            )
        return DirectAgentEpisode(
            runner_id=self.runner_id,
            planner_id=self.planner_id,
            initial_user_utterance=initial,
            turns=turns,
            final_ledger=ledger,
            terminal=session.terminal,
            reached_turn_limit=not session.terminal and len(turns) >= maximum_turns,
        )


def _retry_feedback(error: Exception) -> str:
    """Return public, non-content-bearing retry feedback for one planner retry."""

    if isinstance(error, PublicActionGuardError):
        return (
            "The previous JSON violated the shared public action contract. Choose a different legal action "
            "without repeating an answered question, rejected item, or non-current explanation."
        )
    return (
        "The previous response did not satisfy the strict action JSON contract. Return exactly the required "
        "keys with null or [] for irrelevant fields."
    )
