"""Public-text interpretation interfaces for the TAIRAS Agent.

The deterministic interpreter is a transparent development fallback.  The LLM
interpreter consumes exactly the same public context and must return literal
evidence quotes, so replacing the fallback with GPT/Qwen cannot grant access to
the simulator's semantic response frame or hidden goal.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Literal, Protocol

from pydantic import Field, model_validator

from taira_mt.agent.catalog_tools import PublicCatalogItem, PublicCatalogTool
from taira_mt.agent.memory import (
    GroundedPublicConstraint,
    PublicInteractionMemory,
    PublicUtteranceInterpretation,
)
from taira_mt.agent.public_language import (
    has_direct_public_rejection_marker,
    has_public_unverifiable_rejection,
)
from taira_mt.benchmark.schemas import BenchmarkActionType, BenchmarkSystemAction, BenchmarkUserAct
from taira_mt.dsus.schemas import PublicConstraintObservation
from taira_mt.models import ConstraintImportance, ConstraintOperator, ConstraintPredicate, StrictModel
from taira_mt.provenance import content_sha256
from taira_mt.providers.openai_compatible import (
    JsonCompletion,
    JsonCompletionTransport,
    JsonTokenUsage,
)

PUBLIC_DIALOGUE_INTERPRETER_VERSION = "public_dialogue_interpreter_v3"
LLM_PUBLIC_PARSER_CONTEXT_VERSION = "llm_public_parser_context_v1"

# This is deliberately separate from the Direct-Agent context scan.  The
# parser has a smaller public surface and must not import the baseline merely
# to reuse a list of forbidden evaluator fields.
_PARSER_FORBIDDEN_CONTEXT_KEYS = frozenset(
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
        "state_before",
        "state_after",
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
# Exact keys above protect the fixed parser payload.  Token checks below also
# catch future aliases such as ``oracleScore`` or ``reward_components`` without
# inspecting user-language values, which may legitimately contain those words.
_PARSER_FORBIDDEN_CONTEXT_KEY_TOKENS = frozenset(
    {
        "private",
        "latent",
        "hidden",
        "goal",
        "acceptable",
        "runtime",
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
        "coverage",
    }
)
_INTERPRETATION_JSON_KEYS = frozenset(
    {
        "user_act",
        "grounded_constraints",
        "dismissed_slots",
        "referenced_item_ids",
    }
)


class PublicParserAttempt(StrictModel):
    """Secret-free telemetry for one external public-parser completion.

    Raw provider content is represented only by a digest.  The public parsed
    interpretation itself is persisted in interaction memory when accepted.
    """

    attempt_index: int = Field(ge=1)
    status: Literal["committed", "rejected"]
    provider_request_id: str | None = None
    requested_model: str | None = None
    returned_model: str | None = None
    finish_reason: str | None = None
    raw_content_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    usage: JsonTokenUsage | None = None
    failure_kind: str | None = None

    @model_validator(mode="after")
    def validate_attempt_shape(self) -> PublicParserAttempt:
        if self.status == "committed" and self.failure_kind is not None:
            raise ValueError("a committed parser attempt cannot carry a failure kind")
        if self.status == "rejected" and not self.failure_kind:
            raise ValueError("a rejected parser attempt requires a failure kind")
        return self


class PublicParserExecution(StrictModel):
    """Auditable outcome of one initial-query or user-response parse.

    A deterministic RuleBased interpreter has no external completion record,
    so the runner stores ``None`` for it.  A failed local parser invocation may
    have zero attempts; every LLM completion attempt is nevertheless recorded
    here before the runner executes its fail-closed STOP path.
    """

    parser_id: str = Field(min_length=1)
    parser_context_version: str | None = Field(default=None, min_length=1)
    phase: Literal["initial", "response"]
    public_context_sha256: str = Field(min_length=64, max_length=64)
    attempts: list[PublicParserAttempt] = Field(default_factory=list)
    succeeded: bool
    failure_kind: str | None = None

    @model_validator(mode="after")
    def validate_execution_shape(self) -> PublicParserExecution:
        expected_indices = list(range(1, len(self.attempts) + 1))
        if [item.attempt_index for item in self.attempts] != expected_indices:
            raise ValueError("public parser attempt indices must be contiguous and one-based")
        committed_attempts = [item for item in self.attempts if item.status == "committed"]
        if self.succeeded:
            if not self.attempts:
                raise ValueError("a successful public parser execution requires a completion attempt")
            if self.failure_kind is not None:
                raise ValueError("successful public parser execution cannot have a failure kind")
            if len(committed_attempts) != 1 or self.attempts[-1].status != "committed":
                raise ValueError("successful public parser execution must end in a committed attempt")
        else:
            if not self.failure_kind:
                raise ValueError("failed public parser execution requires a failure kind")
            if committed_attempts:
                raise ValueError("failed public parser execution cannot contain a committed attempt")
        return self


class PublicParserFailure(RuntimeError):
    """A parser failure whose public, secret-free telemetry is retained."""

    def __init__(self, execution: PublicParserExecution) -> None:
        if execution.succeeded:
            raise ValueError("cannot raise PublicParserFailure for a successful execution")
        self.execution = execution
        super().__init__("public parser failed under the bounded JSON contract")


class PublicParserContractError(ValueError):
    """Raised when an LLM completion violates the parser's public contract."""


class PublicDialogueInterpreter(Protocol):
    """Parse user text without any evaluator-private input."""

    interpreter_id: str

    def interpret_initial(
        self,
        *,
        domain: str,
        catalog: PublicCatalogTool,
        utterance: str,
    ) -> PublicUtteranceInterpretation: ...

    def interpret_response(
        self,
        *,
        memory: PublicInteractionMemory,
        catalog: PublicCatalogTool,
        utterance: str,
    ) -> PublicUtteranceInterpretation: ...


def _grounded(
    *,
    slot: str,
    predicate: ConstraintPredicate,
    quote: str,
) -> GroundedPublicConstraint:
    return GroundedPublicConstraint(
        observation=PublicConstraintObservation(
            slot=slot,
            predicate=predicate,
            surface_form=quote,
        ),
        evidence_quote=quote,
    )


def _catalog_value_index(catalog: PublicCatalogTool) -> tuple[tuple[str, str], ...]:
    """Index public textual attributes, preferring specific over broad category labels."""

    preferred_slots = ("specific_category", "broad_category", "category_tags")
    values: dict[tuple[str, str], None] = {}
    for item in catalog.public_items():
        for slot in preferred_slots:
            raw = item.attributes.get(slot)
            raw_values = raw if isinstance(raw, list) else [raw]
            for value in raw_values:
                if not isinstance(value, str):
                    continue
                normalized = value.strip().casefold()
                if len(normalized) >= 3:
                    values[(slot, normalized)] = None
    slot_priority = {slot: index for index, slot in enumerate(preferred_slots)}
    return tuple(
        sorted(
            values,
            key=lambda item: (-len(item[1]), slot_priority.get(item[0], len(slot_priority)), item),
        )
    )


def _is_explicit_broad_category_mention(*, value: str, utterance: str) -> bool:
    """Recognize ``<broad value> category`` without a hidden taxonomy lookup.

    Amazon pools can contain the same public token under both a broad and a
    specific catalog field.  In an utterance such as ``skin care category ...
    exact type``, treating the token as a specific subtype would erase the very
    information gap the user has stated.  The linguistic marker is public and
    deterministic, so it is safe to resolve the collision here rather than
    consulting the simulator's latent disclosure plan.
    """

    return re.search(
        rf"(?<!\w){re.escape(value)}\s+category\b",
        utterance.casefold(),
    ) is not None


def _title_references(
    catalog_items: Iterable[PublicCatalogItem],
    utterance: str,
) -> list[str]:
    folded = utterance.casefold()
    return [
        item.item_id
        for item in catalog_items
        if len(item.title.strip()) >= 3 and item.title.casefold() in folded
    ]


class RuleBasedPublicDialogueInterpreter:
    """Transparent, catalog-aware parser for canonical and near-canonical wording.

    It intentionally makes no claim of broad natural-language understanding.
    The same public interface is later exercised by an LLM interpreter on the
    frozen natural-language view.
    """

    interpreter_id = f"rule_based_{PUBLIC_DIALOGUE_INTERPRETER_VERSION}"

    _price_between = re.compile(
        r"(?:spend|budget).*?between\s+\$?(?P<minimum>\d+(?:\.\d+)?)\s+and\s+\$?(?P<maximum>\d+(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )
    _price_max = re.compile(
        r"(?:"
        r"price\s+(?:to\s+be\s+)?no\s+more\s+than"
        r"|(?:for\s+)?no\s+more\s+than"
        r"|budget\s+(?:is\s+)?at\s+most"
        r"|budget\s+of\s+at\s+most"
        r"|(?:my\s+)?maximum\s+(?:budget|price|expenditure)\s+(?:is|of)?"
        r"|(?:can\s+)?spend\s+up\s+to"
        r"|stay\s+under"
        r"|under"
        r")\s+\$?(?P<value>\d+(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )
    _rating_min = re.compile(
        r"(?:rating\s+of\s+at\s+least|minimum\s+rating\s+of)\s+"
        r"(?P<value>\d+(?:\.\d+)?)\s+stars?",
        flags=re.IGNORECASE,
    )
    _popularity_top = re.compile(
        r"(?:top|upper)\s+(?P<value>\d+(?:\.\d+)?)%\s+(?:by|in)\s+popularity",
        flags=re.IGNORECASE,
    )
    _explanation_request_markers = (
        "detail why",
        "why it meets",
        "explain why",
        "clarify how",
        "explanation of",
        "what grounds",
        "why does",
        "why do you think",
        "tell me how",
        "tell me why",
        "explain the reasoning",
        "how exactly",
        "what makes",
        "how does it align",
    )

    def _user_act(
        self,
        utterance: str,
        *,
        last_action: BenchmarkSystemAction | None,
        has_constraints: bool,
    ) -> BenchmarkUserAct | None:
        folded = utterance.casefold()
        if any(marker in folded for marker in self._explanation_request_markers):
            return BenchmarkUserAct.REQUEST_EXPLANATION
        if any(
            marker in folded
            for marker in ("i'll take", "works for me", "is agreeable", "satisfactory", "i'm okay with")
        ):
            return BenchmarkUserAct.ACCEPT
        if has_direct_public_rejection_marker(utterance):
            return BenchmarkUserAct.REJECT
        has_public_recommendation_context = (
            last_action is not None
            and last_action.action_type in {BenchmarkActionType.RECOMMEND, BenchmarkActionType.EXPLAIN}
        )
        if has_public_recommendation_context and (
            has_public_unverifiable_rejection(utterance)
        ):
            return BenchmarkUserAct.REJECT
        if last_action is not None and last_action.action_type == BenchmarkActionType.ASK:
            return (
                BenchmarkUserAct.CLARIFY
                if last_action.question_target == "occasion_interpretation"
                else BenchmarkUserAct.INFORM
            )
        return BenchmarkUserAct.INFORM if has_constraints else None

    def _constraints(
        self,
        *,
        catalog: PublicCatalogTool,
        utterance: str,
    ) -> list[GroundedPublicConstraint]:
        extracted: dict[str, GroundedPublicConstraint] = {}
        price_between = self._price_between.search(utterance)
        if price_between is not None:
            quote = price_between.group(0)
            extracted["price"] = _grounded(
                slot="price",
                predicate=ConstraintPredicate(
                    operator=ConstraintOperator.BETWEEN,
                    value={
                        "minimum": float(price_between.group("minimum")),
                        "maximum": float(price_between.group("maximum")),
                    },
                    importance=ConstraintImportance.HARD,
                    unit="USD",
                ),
                quote=quote,
            )
        else:
            price_max = self._price_max.search(utterance)
            if price_max is not None:
                quote = price_max.group(0)
                extracted["price"] = _grounded(
                    slot="price",
                    predicate=ConstraintPredicate(
                        operator=ConstraintOperator.LTE,
                        value=float(price_max.group("value")),
                        importance=ConstraintImportance.HARD,
                        unit="USD",
                    ),
                    quote=quote,
                )
        rating_min = self._rating_min.search(utterance)
        if rating_min is not None:
            quote = rating_min.group(0)
            extracted["rating"] = _grounded(
                slot="rating",
                predicate=ConstraintPredicate(
                    operator=ConstraintOperator.GTE,
                    value=float(rating_min.group("value")),
                    importance=ConstraintImportance.HARD,
                    unit="stars",
                ),
                quote=quote,
            )
        popularity_top = self._popularity_top.search(utterance)
        if popularity_top is not None:
            quote = popularity_top.group(0)
            extracted["popularity_percentile"] = _grounded(
                slot="popularity_percentile",
                predicate=ConstraintPredicate(
                    operator=ConstraintOperator.GTE,
                    value=max(0.0, min(1.0, 1.0 - float(popularity_top.group("value")) / 100.0)),
                    importance=ConstraintImportance.HARD,
                    unit="percentile",
                ),
                quote=quote,
            )

        folded = utterance.casefold()
        matching_slots_by_value: dict[str, set[str]] = {}
        for slot, value in _catalog_value_index(catalog):
            if re.search(rf"(?<!\w){re.escape(value)}(?!\w)", folded) is not None:
                matching_slots_by_value.setdefault(value, set()).add(slot)

        # Catalog taxonomies can have nested public values such as ``work
        # wear`` and ``work wear & uniforms``.  The user text only states the
        # longer broad label in that case; greedily retain the longest
        # non-overlapping spans so an incidental substring cannot masquerade
        # as a disclosed subtype.
        value_spans: list[tuple[int, int, str]] = []
        for value in matching_slots_by_value:
            for match in re.finditer(rf"(?<!\w){re.escape(value)}(?!\w)", folded):
                value_spans.append((match.start(), match.end(), value))
        selected_values: list[str] = []
        occupied_spans: list[tuple[int, int]] = []
        for start, end, value in sorted(value_spans, key=lambda item: (-(item[1] - item[0]), item)):
            if any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied_spans):
                continue
            selected_values.append(value)
            occupied_spans.append((start, end))
        for value in selected_values:
            slots = matching_slots_by_value[value]
            if _is_explicit_broad_category_mention(value=value, utterance=utterance):
                target_slot = "broad_category"
            elif "specific_category" in slots or "category_tags" in slots:
                target_slot = "specific_category"
            else:
                target_slot = "broad_category"
            if target_slot in extracted:
                continue
            extracted[target_slot] = _grounded(
                slot=target_slot,
                predicate=ConstraintPredicate(
                    operator=ConstraintOperator.EQ,
                    value=value,
                    importance=ConstraintImportance.HARD,
                ),
                quote=value,
            )
        return list(extracted.values())

    def _interpret(
        self,
        *,
        catalog: PublicCatalogTool,
        utterance: str,
        last_action: BenchmarkSystemAction | None,
        reference_item_ids: Iterable[str],
    ) -> PublicUtteranceInterpretation:
        constraints = self._constraints(catalog=catalog, utterance=utterance)
        return PublicUtteranceInterpretation(
            user_act=self._user_act(
                utterance,
                last_action=last_action,
                has_constraints=bool(constraints),
            ),
            grounded_constraints=constraints,
            referenced_item_ids=_title_references(
                (catalog.inspect(item_id) for item_id in reference_item_ids),
                utterance,
            ),
        )

    def interpret_initial(
        self,
        *,
        domain: str,
        catalog: PublicCatalogTool,
        utterance: str,
    ) -> PublicUtteranceInterpretation:
        if domain != catalog.domain:
            raise ValueError("public interpreter domain differs from catalog")
        return self._interpret(
            catalog=catalog,
            utterance=utterance,
            last_action=None,
            reference_item_ids=(),
        )

    def interpret_response(
        self,
        *,
        memory: PublicInteractionMemory,
        catalog: PublicCatalogTool,
        utterance: str,
    ) -> PublicUtteranceInterpretation:
        if memory.domain != catalog.domain:
            raise ValueError("public interaction memory differs from catalog domain")
        last_event = memory.events[-1]
        action = last_event.system_action
        return self._interpret(
            catalog=catalog,
            utterance=utterance,
            last_action=action,
            reference_item_ids=memory.latest_recommendation_item_ids,
        )


class LLMPublicDialogueInterpreter:
    """Bounded, schema- and grounding-validated public LLM parser.

    This class never falls back to :class:`RuleBasedPublicDialogueInterpreter`.
    A completion may be retried once with static, public-contract feedback; a
    second failure is surfaced to the public Agent runner as
    :class:`PublicParserFailure`, which must execute an auditable typed STOP.
    """

    def __init__(
        self,
        transport: JsonCompletionTransport,
        *,
        interpreter_id: str,
        max_attempts: int = 2,
    ) -> None:
        if not interpreter_id.strip():
            raise ValueError("LLM public interpreter requires a non-empty interpreter ID")
        if max_attempts not in {1, 2}:
            raise ValueError("LLM public interpreter allows one completion and at most one retry")
        self._transport = transport
        self.interpreter_id = interpreter_id
        self.max_attempts = max_attempts
        self.parser_context_version = LLM_PUBLIC_PARSER_CONTEXT_VERSION

    @staticmethod
    def _system_prompt() -> str:
        return """You update public preference memory for a recommendation Agent.
Return one JSON object matching this schema:
{
  \"user_act\": one of [inform, clarify, remind, no_preference, accept, reject, request_explanation, abandon] or null,
  \"grounded_constraints\": [{
    \"observation\": {
      \"slot\": string,
      \"predicate\": {\"operator\": string, \"value\": scalar/list/range,
        \"importance\": \"hard\"|\"soft\", \"unit\": string|null},
      \"surface_form\": string},
    \"evidence_quote\": string
  }],
  \"dismissed_slots\": [string],
  \"referenced_item_ids\": [string]
}
Use only the user utterance and public context. Every evidence_quote must be an
exact literal substring of the user utterance, and every observation.surface_form
must also be a literal substring of that utterance. Do not invent preferences,
IDs, hidden goals, satisfaction labels, acceptable items, or relaxation
boundaries. Use only listed catalog slots, and reference only a listed recent
or rejected recommendation ID. The compact catalog_lexicon contains public
attribute words that can help map a literal user mention to its slot; it is not
evidence for a preference unless that word appears in the user utterance.
Return exactly the four schema keys and no markdown, rationale, or extra keys."""

    @staticmethod
    def _context(
        *,
        domain: str,
        catalog: PublicCatalogTool,
        utterance: str,
        memory: PublicInteractionMemory | None,
    ) -> dict[str, object]:
        latest_action = None
        if memory is not None and memory.events[-1].system_action is not None:
            latest_action = memory.events[-1].system_action.model_dump(mode="json")
        context = {
            "parser_context_version": LLM_PUBLIC_PARSER_CONTEXT_VERSION,
            "interpreter_version": PUBLIC_DIALOGUE_INTERPRETER_VERSION,
            "domain": domain,
            "user_utterance": utterance,
            "catalog_slots": list(catalog.attribute_slots),
            "catalog_lexicon": catalog.public_lexical_value_lexicon(),
            "current_public_constraints": (
                []
                if memory is None
                else [item.model_dump(mode="json") for item in memory.public_constraints]
            ),
            "latest_system_action": latest_action,
            "recent_recommendation_item_ids": (
                [] if memory is None else memory.latest_recommendation_item_ids
            ),
            "rejected_item_ids": [] if memory is None else memory.rejected_item_ids,
        }
        assert_llm_parser_context_is_public(context)
        return context

    @staticmethod
    def _strict_json_payload(content: str) -> dict[str, object]:
        """Decode exactly one non-duplicated interpretation object."""

        def no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
            payload: dict[str, object] = {}
            for key, value in pairs:
                if key in payload:
                    raise PublicParserContractError(f"public parser JSON repeats key: {key}")
                payload[key] = value
            return payload

        try:
            payload = json.loads(content, object_pairs_hook=no_duplicate_keys)
        except (TypeError, json.JSONDecodeError) as exc:
            raise PublicParserContractError("LLM public interpreter returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise PublicParserContractError("LLM public interpreter must return a JSON object")
        keys = frozenset(payload)
        if keys != _INTERPRETATION_JSON_KEYS:
            missing = sorted(_INTERPRETATION_JSON_KEYS.difference(keys))
            extra = sorted(keys.difference(_INTERPRETATION_JSON_KEYS))
            raise PublicParserContractError(
                "LLM public interpreter JSON keys differ from the strict contract: "
                f"missing={missing}, extra={extra}"
            )
        return payload

    @staticmethod
    def _validate_grounding(
        *,
        interpretation: PublicUtteranceInterpretation,
        catalog: PublicCatalogTool,
        utterance: str,
        memory: PublicInteractionMemory | None,
    ) -> None:
        """Reject parser facts not provable from its own public input surface."""

        catalog_slots = set(catalog.attribute_slots)
        utterance_folded = utterance.casefold()
        for grounded in interpretation.grounded_constraints:
            if grounded.observation.slot not in catalog_slots:
                raise PublicParserContractError(
                    "LLM public interpreter emitted a constraint slot outside the public catalog"
                )
            if grounded.evidence_quote.casefold() not in utterance_folded:
                raise PublicParserContractError(
                    "LLM public interpreter emitted a non-literal constraint evidence quote"
                )
            if grounded.observation.surface_form.casefold() not in utterance_folded:
                raise PublicParserContractError(
                    "LLM public interpreter emitted a non-literal constraint surface form"
                )
        invalid_dismissals = set(interpretation.dismissed_slots).difference(catalog_slots)
        if invalid_dismissals:
            raise PublicParserContractError(
                "LLM public interpreter dismissed a slot outside the public catalog"
            )
        known_reference_ids: set[str] = set()
        if memory is not None:
            known_reference_ids.update(memory.latest_recommendation_item_ids)
            known_reference_ids.update(memory.rejected_item_ids)
        invalid_references = set(interpretation.referenced_item_ids).difference(known_reference_ids)
        if invalid_references:
            raise PublicParserContractError(
                "LLM public interpreter referenced an item absent from public recommendation history"
            )

    @classmethod
    def _parse_completion(
        cls,
        *,
        completion: JsonCompletion,
        catalog: PublicCatalogTool,
        utterance: str,
        memory: PublicInteractionMemory | None,
    ) -> PublicUtteranceInterpretation:
        payload = cls._strict_json_payload(completion.content)
        try:
            interpretation = PublicUtteranceInterpretation.model_validate(payload)
        except ValueError as exc:
            raise PublicParserContractError("LLM public interpreter violated the output schema") from exc
        cls._validate_grounding(
            interpretation=interpretation,
            catalog=catalog,
            utterance=utterance,
            memory=memory,
        )
        return interpretation

    @staticmethod
    def _retry_feedback(error: Exception) -> str:
        """Return static public feedback without raw content or exception text."""

        if isinstance(error, PublicParserContractError):
            return (
                "The previous response violated the strict public parser JSON or literal-grounding "
                "contract. Return exactly the four required JSON keys; use only literal text from the "
                "current user utterance, listed catalog slots, and listed public recommendation IDs."
            )
        return (
            "The previous parser completion could not be used. Return exactly the required public parser "
            "JSON object with literal evidence and no extra keys."
        )

    def _execute(
        self,
        *,
        phase: Literal["initial", "response"],
        context: dict[str, object],
        catalog: PublicCatalogTool,
        utterance: str,
        memory: PublicInteractionMemory | None,
    ) -> tuple[PublicUtteranceInterpretation, PublicParserExecution]:
        """Run one bounded parser decision; never select a hidden fallback."""

        try:
            assert_llm_parser_context_is_public(context)
        except Exception as exc:
            execution = PublicParserExecution(
                parser_id=self.interpreter_id,
                parser_context_version=self.parser_context_version,
                phase=phase,
                public_context_sha256=content_sha256(context),
                succeeded=False,
                failure_kind=type(exc).__name__,
            )
            raise PublicParserFailure(execution) from exc

        context_sha256 = content_sha256(context)
        attempts: list[PublicParserAttempt] = []
        retry_feedback: str | None = None
        for attempt_index in range(1, self.max_attempts + 1):
            completion: JsonCompletion | None = None
            try:
                completion = self._transport.complete_json(
                    system_prompt=self._system_prompt(),
                    user_payload=context,
                    retry_feedback=retry_feedback,
                )
                interpretation = self._parse_completion(
                    completion=completion,
                    catalog=catalog,
                    utterance=utterance,
                    memory=memory,
                )
                attempts.append(
                    PublicParserAttempt(
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
                return interpretation, PublicParserExecution(
                    parser_id=self.interpreter_id,
                    parser_context_version=self.parser_context_version,
                    phase=phase,
                    public_context_sha256=context_sha256,
                    attempts=attempts,
                    succeeded=True,
                )
            except Exception as exc:
                attempts.append(
                    PublicParserAttempt(
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
                retry_feedback = self._retry_feedback(exc)
        execution = PublicParserExecution(
            parser_id=self.interpreter_id,
            parser_context_version=self.parser_context_version,
            phase=phase,
            public_context_sha256=context_sha256,
            attempts=attempts,
            succeeded=False,
            failure_kind=attempts[-1].failure_kind,
        )
        raise PublicParserFailure(execution)

    def interpret_initial_with_execution(
        self,
        *,
        domain: str,
        catalog: PublicCatalogTool,
        utterance: str,
    ) -> tuple[PublicUtteranceInterpretation, PublicParserExecution]:
        """Parse an initial query and return secret-free attempt telemetry."""

        if domain != catalog.domain:
            raise ValueError("public interpreter domain differs from catalog")
        return self._execute(
            phase="initial",
            context=self._context(domain=domain, catalog=catalog, utterance=utterance, memory=None),
            catalog=catalog,
            utterance=utterance,
            memory=None,
        )

    def interpret_response_with_execution(
        self,
        *,
        memory: PublicInteractionMemory,
        catalog: PublicCatalogTool,
        utterance: str,
    ) -> tuple[PublicUtteranceInterpretation, PublicParserExecution]:
        """Parse one non-terminal public user response with bounded retries."""

        if memory.domain != catalog.domain:
            raise ValueError("public interaction memory differs from catalog domain")
        return self._execute(
            phase="response",
            context=self._context(domain=memory.domain, catalog=catalog, utterance=utterance, memory=memory),
            catalog=catalog,
            utterance=utterance,
            memory=memory,
        )

    def interpret_initial(
        self,
        *,
        domain: str,
        catalog: PublicCatalogTool,
        utterance: str,
    ) -> PublicUtteranceInterpretation:
        interpretation, _ = self.interpret_initial_with_execution(
            domain=domain,
            catalog=catalog,
            utterance=utterance,
        )
        return interpretation

    def interpret_response(
        self,
        *,
        memory: PublicInteractionMemory,
        catalog: PublicCatalogTool,
        utterance: str,
    ) -> PublicUtteranceInterpretation:
        interpretation, _ = self.interpret_response_with_execution(
            memory=memory,
            catalog=catalog,
            utterance=utterance,
        )
        return interpretation


def _normalized_context_key(value: object) -> str:
    raw = str(value)
    raw = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", raw)
    raw = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", raw).casefold()
    return "".join(character if character.isalnum() else "_" for character in raw).strip("_")


def _is_forbidden_parser_context_key(normalized_key: str) -> bool:
    if normalized_key in _PARSER_FORBIDDEN_CONTEXT_KEYS:
        return True
    return any(
        normalized_key == token
        or normalized_key.startswith(f"{token}_")
        or normalized_key.endswith(f"_{token}")
        or f"_{token}_" in normalized_key
        for token in _PARSER_FORBIDDEN_CONTEXT_KEY_TOKENS
    )


def assert_llm_parser_context_is_public(payload: object) -> None:
    """Reject evaluator/private fields recursively before an LLM call is made."""

    def visit(value: object, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                normalized = _normalized_context_key(key)
                if _is_forbidden_parser_context_key(normalized):
                    raise ValueError(f"LLM parser context contains forbidden field at {path}.{key}")
                visit(nested, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")

    visit(payload, "context")
