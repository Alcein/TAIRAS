"""Contracts for source-grounded rich-state trajectory authoring.

The authoring model does not write gold states or natural dialogue.  It chooses
query-specific semantic facts and a compact event plan.  A deterministic
compiler later turns the plan into state transitions, and a separate renderer
realizes only the already-authorized public propositions.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from html import unescape
from typing import Literal

from pydantic import Field, field_validator, model_validator

from taira_mt.models import StrictModel


class InteractionMechanism(str, Enum):
    DIRECT = "direct"
    PROGRESSIVE = "progressive"
    AMBIGUOUS = "ambiguous"
    REVISION = "revision"
    CONFLICT = "conflict"


class AuthoringEvidenceField(str, Enum):
    LEGACY_QUERY = "legacy_query"
    TARGET_HINT = "target_hint"
    ANCHOR_TITLE = "anchor_title"
    ANCHOR_CATEGORY = "anchor_category"
    ANCHOR_DESCRIPTION = "anchor_description"


class SemanticPolarity(str, Enum):
    REQUIRE = "require"
    PREFER = "prefer"
    AVOID = "avoid"


class SemanticImportance(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class PlannedSystemIntent(str, Enum):
    ASK = "ask"
    RECOMMEND = "recommend"
    EXPLAIN = "explain"
    REPAIR = "repair"
    STOP = "stop"


class PlannedUserEvent(str, Enum):
    DISCLOSE = "disclose"
    CHOOSE = "choose"
    REJECT = "reject"
    REVISE = "revise"
    REMOVE = "remove"
    RELAX = "relax"
    REQUEST_EXPLANATION = "request_explanation"
    ACCEPT = "accept"
    ABANDON = "abandon"
    NOOP = "noop"


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SIMPLE_SLOT_NAMES = {
    "broad_category",
    "category",
    "item_type",
    "popularity",
    "popularity_percentile",
    "price",
    "rating",
    "specific_category",
}


class RichSourceRoot(StrictModel):
    """Private authoring input for one source-disjoint legacy TAIRA root."""

    root_id: str = Field(min_length=1)
    split: Literal["train", "dev", "compiler_test"]
    domain: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    target_hint: str = ""
    anchor_title: str = ""
    anchor_category: str = ""
    anchor_description: str = ""

    def evidence_text(self, field: AuthoringEvidenceField) -> str:
        return {
            AuthoringEvidenceField.LEGACY_QUERY: self.query,
            AuthoringEvidenceField.TARGET_HINT: self.target_hint,
            AuthoringEvidenceField.ANCHOR_TITLE: self.anchor_title,
            AuthoringEvidenceField.ANCHOR_CATEGORY: self.anchor_category,
            AuthoringEvidenceField.ANCHOR_DESCRIPTION: self.anchor_description,
        }[field]


class AuthoringEvidence(StrictModel):
    field: AuthoringEvidenceField
    quote: str = Field(min_length=2, max_length=400)

    @field_validator("quote", mode="before")
    @classmethod
    def normalize_quote(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("authoring evidence quote must be text")
        return " ".join(value.split())


class SemanticFactTemplate(StrictModel):
    """One open semantic fact before it receives dialogue-surface evidence."""

    fact_id: str
    name: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=240)
    polarity: SemanticPolarity
    importance: SemanticImportance
    initially_public: bool
    source_grounding: list[AuthoringEvidence] = Field(min_length=1, max_length=4)

    @field_validator("fact_id")
    @classmethod
    def validate_fact_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("fact_id must be a short lowercase identifier")
        return value

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("semantic fact name must be text")
        normalized = re.sub(r"[\s-]+", "_", value.strip().casefold())
        if not normalized:
            raise ValueError("semantic fact name cannot be empty")
        return normalized

    @field_validator("value", mode="before")
    @classmethod
    def normalize_value(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("semantic fact value must be text")
        return " ".join(value.split())


class LogicalTurnPlan(StrictModel):
    turn_index: int = Field(ge=1, le=8)
    system_intent: PlannedSystemIntent
    user_event: PlannedUserEvent
    reveal_fact_ids: list[str] = Field(default_factory=list, max_length=4)
    supersede_fact_ids: list[str] = Field(default_factory=list, max_length=4)
    remove_fact_ids: list[str] = Field(default_factory=list, max_length=4)
    choice_fact_ids: list[str] = Field(default_factory=list, max_length=4)
    unresolved_axis: str | None = Field(default=None, max_length=120)
    renderer_instruction: str = Field(min_length=3, max_length=360)

    @field_validator(
        "reveal_fact_ids",
        "supersede_fact_ids",
        "remove_fact_ids",
        "choice_fact_ids",
    )
    @classmethod
    def validate_fact_references(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("logical turn fact references must be unique")
        if any(_IDENTIFIER.fullmatch(value) is None for value in values):
            raise ValueError("logical turn references an invalid fact identifier")
        return values

    @field_validator("unresolved_axis", mode="before")
    @classmethod
    def normalize_optional_axis(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("unresolved_axis must be text or null")
        normalized = " ".join(value.split())
        return normalized or None


class RichLogicalTrajectoryPlan(StrictModel):
    root_id: str = Field(min_length=1)
    mechanism: InteractionMechanism
    semantic_facts: list[SemanticFactTemplate] = Field(min_length=1, max_length=12)
    turns: list[LogicalTurnPlan] = Field(min_length=1, max_length=8)
    transferable_rationale: str = Field(min_length=8, max_length=360)

    @model_validator(mode="after")
    def validate_internal_graph(self) -> RichLogicalTrajectoryPlan:
        fact_ids = [fact.fact_id for fact in self.semantic_facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("semantic fact identifiers must be unique")
        if not any(fact.initially_public for fact in self.semantic_facts):
            raise ValueError("logical trajectory requires at least one initially public fact")
        expected_turns = list(range(1, len(self.turns) + 1))
        if [turn.turn_index for turn in self.turns] != expected_turns:
            raise ValueError("logical turn indices must be consecutive and ordered")
        known = set(fact_ids)
        initially_public = {fact.fact_id for fact in self.semantic_facts if fact.initially_public}
        active = set(initially_public)
        asserted = set(initially_public)
        surfaced = set(initially_public)
        for turn in self.turns:
            reveal = set(turn.reveal_fact_ids)
            supersede = set(turn.supersede_fact_ids)
            remove = set(turn.remove_fact_ids)
            choices = set(turn.choice_fact_ids)
            referenced = reveal | supersede | remove | choices
            if not referenced <= known:
                raise ValueError("logical turn references an unknown semantic fact")
            lifecycle_sets = (reveal, supersede, remove)
            if any(left & right for index, left in enumerate(lifecycle_sets) for right in lifecycle_sets[index + 1 :]):
                raise ValueError("reveal, supersede, and remove fact sets must be disjoint")
            if reveal & asserted:
                raise ValueError("a semantic fact can be revealed only once")
            if not (supersede | remove) <= active:
                raise ValueError("only an active public fact can be superseded or removed")
            if choices and len(choices) < 2:
                raise ValueError("a choice must present at least two semantic alternatives")
            if turn.user_event == PlannedUserEvent.CHOOSE:
                if len(choices) < 2:
                    raise ValueError("choose requires at least two explicit alternatives")
                selected_new = reveal & choices
                selected_existing = (choices & active) - supersede - remove
                if len(selected_new | selected_existing) != 1:
                    raise ValueError("choose must leave exactly one selected alternative")
            elif choices:
                if not (self.mechanism == InteractionMechanism.CONFLICT and turn.user_event == PlannedUserEvent.RELAX):
                    raise ValueError("choice alternatives are only valid for choose or conflict relax")
            if turn.user_event == PlannedUserEvent.REVISE and not (reveal and supersede):
                raise ValueError("revise must reveal a replacement and supersede an active fact")
            if turn.user_event == PlannedUserEvent.REMOVE and not remove:
                raise ValueError("remove event must remove at least one active fact")
            if turn.user_event in {
                PlannedUserEvent.NOOP,
                PlannedUserEvent.ACCEPT,
                PlannedUserEvent.ABANDON,
                PlannedUserEvent.REQUEST_EXPLANATION,
            } and (reveal or supersede or remove):
                raise ValueError(f"{turn.user_event.value} cannot mutate semantic facts")

            active.difference_update(supersede | remove)
            active.update(reveal)
            asserted.update(reveal)
            surfaced.update(reveal | choices)

        unused = known - surfaced
        if unused:
            raise ValueError(f"semantic facts are never surfaced: {sorted(unused)}")
        return self

    def _validate_mechanism_grammar(self, initially_public: set[str]) -> None:
        """Enforce the causal grammar that makes each mechanism identifiable."""

        intents = [turn.system_intent for turn in self.turns]
        if PlannedSystemIntent.RECOMMEND not in intents:
            raise ValueError("every trajectory requires a recommendation observation")

        if self.mechanism == InteractionMechanism.DIRECT:
            if PlannedSystemIntent.ASK in intents:
                raise ValueError("direct trajectory cannot ask a clarification question")
            forbidden = {
                PlannedUserEvent.CHOOSE,
                PlannedUserEvent.REVISE,
                PlannedUserEvent.REMOVE,
                PlannedUserEvent.RELAX,
            }
            if any(turn.user_event in forbidden for turn in self.turns):
                raise ValueError("direct trajectory cannot contain preference resolution or change")
            return

        if self.mechanism == InteractionMechanism.PROGRESSIVE:
            qualifying = [
                turn
                for turn in self.turns
                if turn.system_intent == PlannedSystemIntent.ASK
                and turn.user_event == PlannedUserEvent.DISCLOSE
                and turn.reveal_fact_ids
            ]
            if not qualifying:
                raise ValueError("progressive trajectory requires ASK followed by disclosure of a new ranking fact")
            if any(set(turn.reveal_fact_ids) & initially_public for turn in qualifying):
                raise ValueError("progressive disclosure must reveal a previously private fact")
            return

        if self.mechanism == InteractionMechanism.AMBIGUOUS:
            qualifying = [
                turn
                for turn in self.turns
                if turn.system_intent == PlannedSystemIntent.ASK
                and turn.user_event == PlannedUserEvent.CHOOSE
                and len(turn.choice_fact_ids) >= 2
                and len(turn.reveal_fact_ids) == 1
                and set(turn.reveal_fact_ids) <= set(turn.choice_fact_ids)
            ]
            if not qualifying:
                raise ValueError("ambiguous trajectory requires ASK with explicit alternatives and one chosen fact")
            return

        if self.mechanism == InteractionMechanism.CONFLICT:
            facts = {fact.fact_id: fact for fact in self.semantic_facts}
            qualifying = [
                turn
                for turn in self.turns
                if turn.system_intent == PlannedSystemIntent.ASK
                and turn.user_event in {PlannedUserEvent.CHOOSE, PlannedUserEvent.RELAX}
                and len(turn.choice_fact_ids) >= 2
                and set(turn.choice_fact_ids) <= initially_public
                and bool(set(turn.supersede_fact_ids) | set(turn.remove_fact_ids))
            ]
            if not qualifying:
                raise ValueError("conflict trajectory requires two public priorities and choose/relax with retirement")
            priority_facts = {fact_id for turn in qualifying for fact_id in turn.choice_fact_ids}
            importance = {facts[fact_id].importance for fact_id in priority_facts}
            if (
                not {
                    SemanticImportance.HARD,
                    SemanticImportance.SOFT,
                }
                <= importance
            ):
                raise ValueError("conflict alternatives must make hard and soft priorities explicit")
            for turn in qualifying:
                if turn.user_event == PlannedUserEvent.RELAX and turn.supersede_fact_ids:
                    if not turn.reveal_fact_ids:
                        raise ValueError("relax via supersession must reveal the relaxed replacement")
            return

        if self.mechanism == InteractionMechanism.REVISION:
            revision_events = {
                PlannedUserEvent.REJECT,
                PlannedUserEvent.REVISE,
                PlannedUserEvent.REMOVE,
            }
            for index, turn in enumerate(self.turns):
                if (
                    turn.system_intent == PlannedSystemIntent.RECOMMEND
                    and turn.user_event in revision_events
                    and any(later.system_intent == PlannedSystemIntent.RECOMMEND for later in self.turns[index + 1 :])
                ):
                    return
            raise ValueError("revision requires RECOMMEND, an observed revision event, then another RECOMMEND")


class RichPlanBatch(StrictModel):
    plans: list[RichLogicalTrajectoryPlan] = Field(min_length=1, max_length=20)


def validate_plan_against_source(plan: RichLogicalTrajectoryPlan, source: RichSourceRoot) -> None:
    """Fail a plan that is ungrounded, shortcut-only, or violates its mechanism."""

    if plan.root_id != source.root_id:
        raise ValueError("logical plan root_id does not match its source")
    for fact in plan.semantic_facts:
        for evidence in fact.source_grounding:
            haystack = " ".join(source.evidence_text(evidence.field).split()).casefold()
            if evidence.quote.casefold() not in haystack:
                raise ValueError(f"source-grounding quote is absent from {evidence.field.value}: {evidence.quote}")

    open_semantic = [fact for fact in plan.semantic_facts if fact.name not in _SIMPLE_SLOT_NAMES]
    if not open_semantic:
        raise ValueError("rich trajectory requires at least one open product-semantic fact")
    initial_query_rich = [
        fact
        for fact in open_semantic
        if fact.initially_public
        and any(evidence.field == AuthoringEvidenceField.LEGACY_QUERY for evidence in fact.source_grounding)
    ]
    if not initial_query_rich:
        raise ValueError("at least one initially public non-simple fact must be grounded directly in legacy_query")

    # Evidence-field labels alone are not enough to establish temporal
    # privacy.  A model can ground a fact in ``target_hint`` while choosing a
    # value that is already stated verbatim in the legacy opening query.  Such
    # a fact cannot later be presented as newly disclosed/revised (for example
    # hiding "diverse rhythms" when those exact words are already in the
    # opening).  Reject that causal contradiction so the full builder can
    # re-author or reassign the interaction mechanism.
    normalized_query = " ".join(
        re.sub(r"[^a-z0-9]+", " ", unescape(source.query).casefold()).split()
    )
    padded_query = f" {normalized_query} "
    for fact in plan.semantic_facts:
        if fact.initially_public:
            continue
        # Progressive disclosure and preference revision claim that a value
        # becomes known only at a later user event.  Ambiguous-choice facts are
        # different: they may deliberately restate two interpretations of an
        # already worded need, so the exact-value test is not applicable there.
        if plan.mechanism not in {
            InteractionMechanism.PROGRESSIVE,
            InteractionMechanism.REVISION,
        }:
            continue
        normalized_value = " ".join(
            re.sub(r"[^a-z0-9]+", " ", unescape(fact.value).casefold()).split()
        )
        if normalized_value and f" {normalized_value} " in padded_query:
            raise ValueError(
                "a later-revealed fact value is already explicit in legacy_query: "
                f"{fact.fact_id}"
            )
    plan._validate_mechanism_grammar({fact.fact_id for fact in plan.semantic_facts if fact.initially_public})


def _mechanism_skeleton(mechanism: InteractionMechanism) -> dict[str, object]:
    """Give the author a mechanism SOP instead of asking it to invent control flow."""

    empty = {
        "reveal_fact_ids": [],
        "supersede_fact_ids": [],
        "remove_fact_ids": [],
        "choice_fact_ids": [],
    }
    accept = {
        "turn_index": 2,
        "system_intent": "recommend",
        "user_event": "accept",
        **empty,
    }
    if mechanism == InteractionMechanism.DIRECT:
        return {
            "fact_roles": "All semantic facts are initially_public=true.",
            "turns": [{**accept, "turn_index": 1}],
        }
    if mechanism == InteractionMechanism.PROGRESSIVE:
        return {
            "fact_roles": (
                "Exactly one ranking-changing answer fact is initially_public=false; "
                "it must be grounded outside legacy_query. Every legacy_query fact is initially public."
            ),
            "turns": [
                {
                    "turn_index": 1,
                    "system_intent": "ask",
                    "user_event": "disclose",
                    "reveal_fact_ids": ["THE_HIDDEN_ANSWER_FACT"],
                    "supersede_fact_ids": [],
                    "remove_fact_ids": [],
                    "choice_fact_ids": [],
                },
                accept,
            ],
        }
    if mechanism == InteractionMechanism.AMBIGUOUS:
        return {
            "fact_roles": (
                "Create exactly two mutually exclusive initially_public=false alternative facts; "
                "both must be grounded outside legacy_query. Every legacy_query fact is initially public."
            ),
            "turns": [
                {
                    "turn_index": 1,
                    "system_intent": "ask",
                    "user_event": "choose",
                    "reveal_fact_ids": ["ONE_SELECTED_ALTERNATIVE"],
                    "supersede_fact_ids": [],
                    "remove_fact_ids": [],
                    "choice_fact_ids": ["ALTERNATIVE_A", "ALTERNATIVE_B"],
                },
                accept,
            ],
        }
    if mechanism == InteractionMechanism.CONFLICT:
        return {
            "fact_roles": (
                "Create two competing initially_public=true priorities: one hard and one soft. "
                "Use only legacy_query-grounded facts for this opening conflict."
            ),
            "turns": [
                {
                    "turn_index": 1,
                    "system_intent": "ask",
                    "user_event": "choose",
                    "reveal_fact_ids": [],
                    "supersede_fact_ids": [],
                    "remove_fact_ids": ["THE_SOFT_PRIORITY_NOT_CHOSEN"],
                    "choice_fact_ids": ["THE_HARD_PRIORITY", "THE_SOFT_PRIORITY"],
                },
                accept,
            ],
        }
    return {
        "fact_roles": (
            "Create one initially_public=true old preference and one initially_public=false replacement "
            "preference, preferably with the same semantic name and different values. The old fact must be "
            "legacy_query-grounded; the replacement must be grounded outside legacy_query."
        ),
        "turns": [
            {
                "turn_index": 1,
                "system_intent": "recommend",
                "user_event": "revise",
                "reveal_fact_ids": ["THE_REPLACEMENT_FACT"],
                "supersede_fact_ids": ["THE_OLD_FACT"],
                "remove_fact_ids": [],
                "choice_fact_ids": [],
            },
            accept,
        ],
    }


def authoring_messages(
    assignments: list[tuple[RichSourceRoot, InteractionMechanism]],
) -> tuple[str, dict[str, object]]:
    """Build a compact logical-authoring request; it deliberately asks for no dialogue prose."""

    system_prompt = (
        "You design source-grounded logical trajectories for a multi-turn shopping-state benchmark. "
        "Return semantic fact templates and event plans, not natural dialogue, gold states, action scores, "
        "Thought Pattern IDs, rewards, or evaluator labels. Every semantic fact must quote exact evidence "
        "from the supplied query/target/item fields. Preserve implicit usage-to-attribute reasoning. Include "
        "at least one initially-public open product attribute beyond category, price, rating, or popularity "
        "whose exact grounding comes directly from legacy_query. The assigned mechanism is immutable. "
        "Each case contains a required_causal_skeleton. Copy that skeleton exactly: do not add turns, "
        "change system_intent/user_event, or move mutations to another turn. Replace skeleton placeholders "
        "only with IDs of semantic facts you author. Facts can be revealed once and "
        "only active public facts can be superseded or removed; these three ID sets must never overlap. "
        "Every fact grounded in legacy_query is already public in the opening. A fact first revealed later "
        "must instead be grounded in target/item fields and must not restate the legacy query. "
        "unresolved_axis is only a high-level question topic for rendering: never copy an undisclosed fact "
        "value into it. Reply with exactly one JSON object containing plans."
    )
    cases = [
        {
            "root_id": source.root_id,
            "assigned_mechanism": mechanism.value,
            "domain": source.domain,
            "legacy_query": source.query,
            "target_hint": source.target_hint,
            "anchor_title": source.anchor_title,
            "anchor_category": source.anchor_category,
            "anchor_description": source.anchor_description[:800],
            "required_causal_skeleton": _mechanism_skeleton(mechanism),
        }
        for source, mechanism in assignments
    ]
    schema = {
        "plans": [
            {
                "root_id": "copy exactly",
                "mechanism": "copy assigned mechanism",
                "semantic_facts": [
                    {
                        "fact_id": "short lowercase id",
                        "name": "open semantic slot",
                        "value": "grounded value",
                        "polarity": "require | prefer | avoid",
                        "importance": "hard | soft",
                        "initially_public": "boolean",
                        "source_grounding": [
                            {
                                "field": (
                                    "legacy_query | target_hint | anchor_title | anchor_category | anchor_description"
                                ),
                                "quote": "exact substring",
                            }
                        ],
                    }
                ],
                "turns": [
                    {
                        "turn_index": "consecutive integer from 1",
                        "system_intent": "ask | recommend | explain | repair | stop",
                        "user_event": (
                            "disclose | choose | reject | revise | remove | relax | "
                            "request_explanation | accept | abandon | noop"
                        ),
                        "reveal_fact_ids": [],
                        "supersede_fact_ids": [],
                        "remove_fact_ids": [],
                        "choice_fact_ids": [],
                        "unresolved_axis": "high-level non-leaking question topic or null",
                        "renderer_instruction": "semantic instruction only; no final dialogue prose",
                    }
                ],
                "transferable_rationale": "why this event sequence is necessary for this shopping need",
            }
        ]
    }
    return system_prompt, {"cases": cases, "required_output_schema": schema}


def _normalize_model_fact_identifiers(value: object) -> object:
    """Repair harmless punctuation in model-authored local IDs and their references."""

    if not isinstance(value, dict) or not isinstance(value.get("plans"), list):
        return value
    for plan in value["plans"]:
        if not isinstance(plan, dict) or not isinstance(plan.get("semantic_facts"), list):
            continue
        # An ungrounded optional fact is unusable as supervision.  Dropping it
        # is deterministic and safer than asking the model to rewrite every
        # otherwise valid fact in the batch.
        plan["semantic_facts"] = [
            fact
            for fact in plan["semantic_facts"]
            if isinstance(fact, dict)
            and isinstance(fact.get("source_grounding"), list)
            and bool(fact["source_grounding"])
        ]
        mapping: dict[str, str] = {}
        for fact in plan["semantic_facts"]:
            if not isinstance(fact, dict) or not isinstance(fact.get("fact_id"), str):
                continue
            original = fact["fact_id"]
            normalized = re.sub(r"[^a-z0-9_-]+", "_", original.casefold()).strip("_-")
            if not normalized or not normalized[0].isalpha():
                normalized = f"fact_{normalized or 'value'}"
            normalized = normalized[:64]
            if normalized in mapping.values() and mapping.get(original) != normalized:
                raise ValueError("fact identifier normalization creates a duplicate")
            mapping[original] = normalized
            fact["fact_id"] = normalized
        for turn in plan.get("turns", []):
            if not isinstance(turn, dict):
                continue
            for field in (
                "reveal_fact_ids",
                "supersede_fact_ids",
                "remove_fact_ids",
                "choice_fact_ids",
            ):
                if isinstance(turn.get(field), list):
                    turn[field] = [mapping.get(item, item) for item in turn[field]]
    return value


def _canonicalize_model_control_flow(value: object) -> object:
    """Compile semantic facts into the assigned finite mechanism SOP.

    The language model authors open semantic content; deterministic code owns
    the enumerable control flow.  This prevents harmless intent-label drift
    (for example ``choose`` used as a system intent) from consuming repeated
    calls or changing the benchmark transition semantics.
    """

    if not isinstance(value, dict) or not isinstance(value.get("plans"), list):
        return value

    def fact_ids(plan: dict[str, object]) -> list[str]:
        facts = plan.get("semantic_facts")
        if not isinstance(facts, list):
            return []
        return [
            fact["fact_id"]
            for fact in facts
            if isinstance(fact, dict) and isinstance(fact.get("fact_id"), str)
        ]

    def referenced(plan: dict[str, object], field: str) -> list[str]:
        turns = plan.get("turns")
        if not isinstance(turns, list):
            return []
        output: list[str] = []
        for turn in turns:
            values = turn.get(field) if isinstance(turn, dict) else None
            if isinstance(values, list):
                output.extend(item for item in values if isinstance(item, str))
        return list(dict.fromkeys(output))

    def instruction(plan: dict[str, object], first: bool, fallback: str) -> str:
        turns = plan.get("turns")
        if not isinstance(turns, list) or not turns:
            return fallback
        candidates = turns if first else list(reversed(turns))
        for turn in candidates:
            text = turn.get("renderer_instruction") if isinstance(turn, dict) else None
            if isinstance(text, str) and len(text.strip()) >= 3:
                return text.strip()[:360]
        return fallback

    def turn(
        *,
        index: int,
        system_intent: str,
        user_event: str,
        reveal: list[str] | None = None,
        supersede: list[str] | None = None,
        remove: list[str] | None = None,
        choices: list[str] | None = None,
        axis: str | None = None,
        renderer_instruction: str,
    ) -> dict[str, object]:
        return {
            "turn_index": index,
            "system_intent": system_intent,
            "user_event": user_event,
            "reveal_fact_ids": reveal or [],
            "supersede_fact_ids": supersede or [],
            "remove_fact_ids": remove or [],
            "choice_fact_ids": choices or [],
            "unresolved_axis": axis,
            "renderer_instruction": renderer_instruction,
        }

    for raw_plan in value["plans"]:
        if not isinstance(raw_plan, dict):
            continue
        ids = fact_ids(raw_plan)
        facts = raw_plan.get("semantic_facts")
        if not ids or not isinstance(facts, list):
            continue
        by_id = {
            fact["fact_id"]: fact
            for fact in facts
            if isinstance(fact, dict) and isinstance(fact.get("fact_id"), str)
        }
        query_grounded = {
            item
            for item, fact in by_id.items()
            if any(
                isinstance(evidence, dict) and evidence.get("field") == "legacy_query"
                for evidence in fact.get("source_grounding", [])
            )
        }
        if not query_grounded:
            raise ValueError(
                "at least one initially public non-simple fact must be grounded directly in legacy_query"
            )
        later_grounded = [item for item in ids if item not in query_grounded]
        mechanism = raw_plan.get("mechanism")
        opening_instruction = instruction(
            raw_plan, True, "Realize the assigned public state transition naturally."
        )
        accept_instruction = instruction(
            raw_plan, False, "Accept the recommendation after the state has been resolved."
        )
        accept = turn(
            index=2,
            system_intent="recommend",
            user_event="accept",
            renderer_instruction=accept_instruction,
        )

        if mechanism == InteractionMechanism.DIRECT.value:
            direct_facts = [by_id[item] for item in ids if item in query_grounded]
            if not direct_facts:
                raise ValueError("direct plan has no legacy-query-grounded public fact")
            raw_plan["semantic_facts"] = direct_facts
            for fact in direct_facts:
                if isinstance(fact, dict):
                    fact["initially_public"] = True
            raw_plan["turns"] = [{**accept, "turn_index": 1}]
            continue

        if mechanism == InteractionMechanism.PROGRESSIVE.value:
            candidates = referenced(raw_plan, "reveal_fact_ids") + [
                item
                for item in ids
                if by_id[item].get("initially_public") is False
            ]
            hidden = next((item for item in candidates if item in later_grounded), None)
            if hidden is None:
                raise ValueError(
                    "progressive plan needs one later fact grounded outside legacy_query"
                )
            raw_plan["semantic_facts"] = [
                by_id[item] for item in ids if item in query_grounded or item == hidden
            ]
            for item in query_grounded:
                by_id[item]["initially_public"] = True
            by_id[hidden]["initially_public"] = False
            # Dialogue order is system -> user.  The reveal operation belongs
            # to the user's reply, so a model-authored instruction that names
            # the hidden value makes the preceding question leak the answer
            # (for example asking about "Scarlet Blossom" before the user has
            # stated a scent preference).  The semantic fact remains available
            # to the renderer for the user turn; the system side is compiled
            # into a value-free elicitation SOP.
            progressive_instruction = (
                "Ask one open-ended question about the unresolved ranking preference without "
                "naming, paraphrasing, implying, or offering its hidden value. The user's "
                "following utterance must introduce that preference in their own words."
            )
            raw_plan["turns"] = [
                turn(
                    index=1,
                    system_intent="ask",
                    user_event="disclose",
                    reveal=[hidden],
                    axis="one unresolved ranking preference",
                    renderer_instruction=progressive_instruction,
                ),
                accept,
            ]
            continue

        if mechanism == InteractionMechanism.AMBIGUOUS.value:
            candidates = referenced(raw_plan, "choice_fact_ids") + referenced(
                raw_plan, "reveal_fact_ids"
            )
            candidates += [
                item
                for item in ids
                if by_id[item].get("initially_public") is False
            ]
            alternatives = [
                item
                for item in dict.fromkeys(candidates)
                if item in later_grounded
            ][:2]
            if len(alternatives) < 2:
                alternatives += [
                    item
                    for item in reversed(later_grounded)
                    if item not in alternatives
                ][: 2 - len(alternatives)]
            if len(alternatives) < 2:
                raise ValueError(
                    "ambiguous plan needs two alternative facts grounded outside legacy_query"
                )
            selected = next(
                (
                    item
                    for item in referenced(raw_plan, "reveal_fact_ids")
                    if item in alternatives
                ),
                alternatives[0] if alternatives else "",
            )
            raw_plan["semantic_facts"] = [
                by_id[item]
                for item in ids
                if item in query_grounded or item in alternatives
            ]
            for item in query_grounded:
                by_id[item]["initially_public"] = True
            for item in alternatives:
                by_id[item]["initially_public"] = False
            raw_plan["turns"] = [
                turn(
                    index=1,
                    system_intent="ask",
                    user_event="choose",
                    reveal=[selected] if selected else [],
                    choices=alternatives,
                    axis="two plausible interpretations of the stated need",
                    renderer_instruction=opening_instruction,
                ),
                accept,
            ]
            continue

        if mechanism == InteractionMechanism.CONFLICT.value:
            conflict_ids = [item for item in ids if item in query_grounded]
            if len(conflict_ids) < 2:
                raise ValueError("conflict plan needs two legacy-query-grounded priorities")
            hard = next(
                (
                    item
                    for item in conflict_ids
                    if by_id[item].get("importance") == "hard"
                ),
                conflict_ids[0],
            )
            soft = next(
                (
                    item
                    for item in conflict_ids
                    if item != hard and by_id[item].get("importance") == "soft"
                ),
                next(item for item in conflict_ids if item != hard),
            )
            raw_plan["semantic_facts"] = [by_id[item] for item in conflict_ids]
            for item in conflict_ids:
                by_id[item]["initially_public"] = True
            by_id[hard]["importance"] = "hard"
            by_id[soft]["importance"] = "soft"
            raw_plan["turns"] = [
                turn(
                    index=1,
                    system_intent="ask",
                    user_event="choose",
                    remove=[soft] if soft != hard else [],
                    choices=list(dict.fromkeys((hard, soft))),
                    axis="trade-off between two stated priorities",
                    renderer_instruction=opening_instruction,
                ),
                accept,
            ]
            continue

        if mechanism == InteractionMechanism.REVISION.value:
            replacement_candidates = referenced(raw_plan, "reveal_fact_ids") + [
                item
                for item in ids
                if by_id[item].get("initially_public") is False
            ]
            replacement = next(
                (
                    item
                    for item in replacement_candidates
                    if item in later_grounded
                ),
                later_grounded[-1] if later_grounded else None,
            )
            if replacement is None:
                raise ValueError(
                    "revision plan needs one replacement grounded outside legacy_query"
                )
            old_candidates = referenced(raw_plan, "supersede_fact_ids") + [
                item for item in ids if item in query_grounded
            ]
            old = next(
                (item for item in old_candidates if item in query_grounded),
                None,
            )
            if old is None:
                raise ValueError("revision plan needs one legacy-query-grounded old fact")
            raw_plan["semantic_facts"] = [
                by_id[item]
                for item in ids
                if item in query_grounded or item == replacement
            ]
            for item in query_grounded:
                by_id[item]["initially_public"] = True
            by_id[replacement]["initially_public"] = False
            # Dialogue order is system -> user for every planned turn.  The
            # replacement fact is therefore introduced by the *user* at the
            # end of turn 1.  Reusing the model-authored instruction here can
            # accidentally tell the preceding system utterance to name that
            # still-private fact.  Keep the temporal boundary deterministic:
            # first respond only to the opening request, then acknowledge the
            # now-public revision in the following system turn.
            pre_revision_instruction = (
                "Respond only to the initially public opening preferences and make a plausible "
                "recommendation without mentioning, implying, or anticipating the replacement "
                "fact. The following user utterance must introduce the replacement and revise "
                "the old fact."
            )
            post_revision_instruction = (
                "After the user has publicly introduced the replacement fact and revised the old "
                "preference in the preceding turn, acknowledge that revision and recommend "
                "accordingly without adding any new preference."
            )
            raw_plan["turns"] = [
                turn(
                    index=1,
                    system_intent="recommend",
                    user_event="revise",
                    reveal=[replacement],
                    supersede=[old],
                    renderer_instruction=pre_revision_instruction,
                ),
                turn(
                    index=2,
                    system_intent="recommend",
                    user_event="accept",
                    renderer_instruction=post_revision_instruction,
                ),
            ]
    return value


def _rewrite_progressive_target_identity(
    value: object,
    sources: dict[str, RichSourceRoot],
) -> object:
    """Replace an exact target-title shortcut with a transferable title-theme fact.

    A progressive question may ask whether title wording itself should reflect
    the requested scenario, but the hidden *user state* must not be the exact
    identity of the anchor item.  This deterministic repair preserves the
    assigned causal mechanism while preventing the compiler from learning to
    emit a target answer key such as a song title.
    """

    if not isinstance(value, dict) or not isinstance(value.get("plans"), list):
        return value

    def normalized(text: object) -> str:
        if not isinstance(text, str):
            return ""
        return " ".join(re.sub(r"[^a-z0-9]+", " ", unescape(text).casefold()).split())

    for plan in value["plans"]:
        if not isinstance(plan, dict) or plan.get("mechanism") != InteractionMechanism.PROGRESSIVE.value:
            continue
        source = sources.get(str(plan.get("root_id", "")))
        facts = plan.get("semantic_facts")
        turns = plan.get("turns")
        if source is None or not isinstance(facts, list) or not isinstance(turns, list):
            continue
        anchor_identity = normalized(source.anchor_title)
        if not anchor_identity:
            continue
        rewritten_ids: set[str] = set()
        for fact in facts:
            if not isinstance(fact, dict) or fact.get("initially_public") is not False:
                continue
            grounding = fact.get("source_grounding")
            grounded_in_title = isinstance(grounding, list) and any(
                isinstance(item, dict) and item.get("field") == AuthoringEvidenceField.ANCHOR_TITLE.value
                for item in grounding
            )
            if not grounded_in_title or normalized(fact.get("value")) != anchor_identity:
                continue
            fact_id = fact.get("fact_id")
            if not isinstance(fact_id, str):
                continue
            fact["name"] = "title_theme_preference"
            fact["value"] = "title wording explicitly aligned with the requested theme"
            rewritten_ids.add(fact_id)
        if not rewritten_ids:
            continue
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            reveal_ids = turn.get("reveal_fact_ids")
            if isinstance(reveal_ids, list) and rewritten_ids.intersection(reveal_ids):
                turn["renderer_instruction"] = (
                    "Ask whether the user wants the item title itself to reflect the requested theme. "
                    "The user confirms only that transferable title-theme preference without naming "
                    "a specific product; do not mention the anchor title before the final recommendation."
                )
    return value


def parse_authored_batch(content: str, sources: list[RichSourceRoot]) -> list[RichLogicalTrajectoryPlan]:
    """Parse and source-check one model response without accepting partial batches."""

    source_by_id = {source.root_id: source for source in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("authoring request contains duplicate source root IDs")
    value = _rewrite_progressive_target_identity(
        _canonicalize_model_control_flow(
            _normalize_model_fact_identifiers(json.loads(content))
        ),
        source_by_id,
    )
    batch = RichPlanBatch.model_validate(value)
    if {plan.root_id for plan in batch.plans} != set(source_by_id):
        raise ValueError("authoring response root IDs do not exactly match the requested batch")
    for plan in batch.plans:
        validate_plan_against_source(plan, source_by_id[plan.root_id])
    return batch.plans
