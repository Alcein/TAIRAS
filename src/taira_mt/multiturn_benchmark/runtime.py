"""Runnable deterministic user environment for the frozen TAIRAS test set.

The construction artifact stores one compact transition per example.  This
module turns that compact contract into the actual public interaction surface
used by recommendation Agents:

* the Agent sees only the initial user query and a full-domain catalog tool;
* the simulator owns the hidden transition, user reply, and evidence labels;
* the expected action schedule distinguishes direct recommendation,
  clarification-first interaction, and recommendation-revision interaction.

No LLM call is made while an episode is rolled out.  The class intentionally
uses the existing ``BenchmarkSystemAction`` schema so it can be supplied to
both the new Agent code and the legacy public Agent runner.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from taira_mt.agent.catalog_tools import PublicCatalogTool
from taira_mt.benchmark.schemas import BenchmarkActionType, BenchmarkSystemAction, PublicEnvironmentResponse
from taira_mt.data.legacy_catalog import LegacyCatalogAdapter
from taira_mt.recommender.catalog import CandidatePool

FROZEN_RUNTIME_VERSION = "frozen_multiturn_dialogue_runtime_v5_information_asymmetry"
INTERACTION_UTILITY_VERSION = "mt_path_utility_v1"
TURN_COST = 0.02
QUESTION_COST = 0.04
REJECTED_RECOMMENDATION_COST = 0.10
UNNECESSARY_ACTION_COST = 0.10
SUPPORTED_MECHANISMS = frozenset({"direct", "progressive", "ambiguous", "revision", "conflict"})

# Two author responses omitted a positive item although the full Amazon catalog
# contains an exact lexical counterpart.  These deterministic repairs preserve
# the source query and state transition; they merely make the final evaluator
# total rather than silently treating an impossible label as a failure.
RUNTIME_EVIDENCE_REPAIRS: dict[str, tuple[str, ...]] = {
    "test-clothing-revision-B00F4UVD9I": ("B001PO6NB0",),
    "test-beauty-revision-B0009953JA": ("B000KEMKM2",),
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9&-]{1,}")
_QUESTION_STOPWORDS = frozenset(
    {
        "about",
        "also",
        "and",
        "are",
        "can",
        "could",
        "does",
        "for",
        "have",
        "i",
        "in",
        "is",
        "it",
        "me",
        "need",
        "of",
        "or",
        "please",
        "prefer",
        "should",
        "the",
        "to",
        "what",
        "with",
        "you",
        "your",
    }
)


class FrozenDialoguePhase(str, Enum):
    """Private state of the compact deterministic dialogue machine."""

    INITIAL = "initial"
    READY_TO_RECOMMEND = "ready_to_recommend"
    READY_TO_REPAIR = "ready_to_repair"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class FrozenRuntimeProtocol:
    """Evaluator-owned interaction schedule derived from one frozen record."""

    episode_id: str
    domain: str
    mechanism: str
    expected_action_sequence: tuple[str, ...]
    agent_turn_budget: int
    positive_item_ids: tuple[str, ...]
    hard_negative_item_ids: tuple[str, ...]
    fallback_catalog_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_version": FROZEN_RUNTIME_VERSION,
            "episode_id": self.episode_id,
            "domain": self.domain,
            "mechanism": self.mechanism,
            "expected_action_sequence": list(self.expected_action_sequence),
            "agent_turn_budget": self.agent_turn_budget,
            "evaluation": {
                "positive_item_ids": list(self.positive_item_ids),
                "hard_negative_item_ids": list(self.hard_negative_item_ids),
                "fallback_catalog_terms": list(self.fallback_catalog_terms),
            },
        }


@dataclass(frozen=True)
class FrozenRecommendationAssessment:
    """Private deterministic evidence assessment for one recommendation slate."""

    recommended_item_ids: tuple[str, ...]
    unknown_item_ids: tuple[str, ...]
    known_positive_rank: int | None
    catalog_term_rank: int | None
    hard_negative_item_ids: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.known_positive_rank is not None or self.catalog_term_rank is not None

    def relevant_within(self, cutoff: int) -> bool:
        """Return rank-aware relevance without changing the public Top-10 slate."""

        return any(
            rank is not None and rank <= cutoff
            for rank in (self.known_positive_rank, self.catalog_term_rank)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_item_ids": list(self.recommended_item_ids),
            "unknown_item_ids": list(self.unknown_item_ids),
            "known_positive_rank": self.known_positive_rank,
            "catalog_term_rank": self.catalog_term_rank,
            "hard_negative_item_ids": list(self.hard_negative_item_ids),
            "accepted": self.accepted,
            "relevant_at_1": self.relevant_within(1),
            "relevant_at_3": self.relevant_within(3),
            "relevant_at_10": self.relevant_within(10),
        }


@dataclass(frozen=True)
class FrozenRuntimeTurn:
    """Private provenance for a public action and deterministic response."""

    turn_index: int
    phase_before: FrozenDialoguePhase
    phase_after: FrozenDialoguePhase
    action_type: str
    response_text: str
    terminal: bool
    event: str
    question_looks_targeted: bool | None = None
    assessment: FrozenRecommendationAssessment | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "phase_before": self.phase_before.value,
            "phase_after": self.phase_after.value,
            "action_type": self.action_type,
            "response_text": self.response_text,
            "terminal": self.terminal,
            "event": self.event,
            "question_looks_targeted": self.question_looks_targeted,
            "assessment": None if self.assessment is None else self.assessment.to_dict(),
        }


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _item_ids(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(
        item_id
        for item_id in (_clean_text(value.get("item_id")) for value in values if isinstance(value, Mapping))
        if item_id
    )


def _string_ids(values: Any) -> tuple[str, ...]:
    """Read an evaluator-owned list of catalog IDs without accepting scalars."""

    if not isinstance(values, list):
        return ()
    return tuple(item_id for item_id in (_clean_text(value) for value in values) if item_id)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _runtime_terms(record: Mapping[str, Any]) -> tuple[str, ...]:
    transition = record.get("transition", {})
    priority = transition.get("resolved_priority", {}) if isinstance(transition, Mapping) else {}
    raw_terms = priority.get("required_catalog_terms", []) if isinstance(priority, Mapping) else []
    terms = _unique(_clean_text(value) for value in raw_terms if _clean_text(value))
    if terms:
        return terms
    label = _clean_text(priority.get("label")) if isinstance(priority, Mapping) else ""
    return (label,) if label else ()


def runtime_protocol_from_record(record: Mapping[str, Any]) -> FrozenRuntimeProtocol:
    """Derive the actual interaction schedule from a frozen trajectory record."""

    episode_id = _clean_text(record.get("episode_id"))
    domain = _clean_text(record.get("domain"))
    mechanism = _clean_text(record.get("mechanism"))
    if not episode_id or not domain or mechanism not in SUPPORTED_MECHANISMS:
        raise ValueError(
            f"invalid frozen Agent record: episode={episode_id!r}, domain={domain!r}, mechanism={mechanism!r}"
        )

    contract = record.get("evaluation_contract", {})
    if not isinstance(contract, Mapping):
        raise ValueError(f"{episode_id} has no evaluation contract")
    hard_negative_ids = _unique(_item_ids(contract.get("hard_negative_evidence")))
    negative_set = set(hard_negative_ids)
    positive_ids = tuple(
        item_id
        for item_id in _unique(
            (
                *_item_ids(contract.get("positive_evidence")),
                *_string_ids(contract.get("resolved_need_relevance_item_ids")),
                *RUNTIME_EVIDENCE_REPAIRS.get(episode_id, ()),
            )
        )
        if item_id not in negative_set
    )
    if mechanism == "direct":
        expected, budget = ("recommend",), 2
    elif mechanism == "revision":
        # Revision means a plausible initial recommendation exposes a stable
        # priority, after which the Agent must repair its result.
        expected, budget = ("recommend", "recommend"), 2
    else:
        expected, budget = ("ask", "recommend"), 2
    return FrozenRuntimeProtocol(
        episode_id=episode_id,
        domain=domain,
        mechanism=mechanism,
        expected_action_sequence=expected,
        agent_turn_budget=budget,
        positive_item_ids=positive_ids,
        hard_negative_item_ids=hard_negative_ids,
        fallback_catalog_terms=_runtime_terms(record),
    )


def load_frozen_agent_records(path: Path) -> list[dict[str, Any]]:
    """Read the frozen Agent-facing JSONL and perform only schema-level checks."""

    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        protocol = runtime_protocol_from_record(record)
        if protocol.episode_id in seen:
            raise ValueError(f"duplicate frozen episode ID: {protocol.episode_id}")
        if not _clean_text(record.get("initial_query")):
            raise ValueError(f"{protocol.episode_id} has no initial query")
        seen.add(protocol.episode_id)
        records.append(record)
    if not records:
        raise ValueError(f"frozen Agent artifact contains no records: {path}")
    return records


class FullCatalogRegistry:
    """Lazy, per-domain full-catalog loader shared by many frozen episodes.

    The evidence menu in a frozen record is an evaluator label set, not the
    retrieval pool.  This registry deliberately loads the entire legacy
    domain catalog once and hands the Agent a normal ``PublicCatalogTool``.
    """

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self._pools: dict[str, CandidatePool] = {}

    def candidate_pool(self, domain: str) -> CandidatePool:
        if domain in self._pools:
            return self._pools[domain]
        metadata_path = self.data_root / domain / "metadata.csv"
        adapter = LegacyCatalogAdapter(metadata_path, domain=domain)
        index = adapter.load_index()
        pool = index.pool(
            recipe_id="full_legacy_amazon_catalog_v1",
            recipe_config={"scope": "all_metadata_items", "domain": domain},
        )
        self._pools[domain] = pool
        return pool


@dataclass
class FrozenMultiTurnSession:
    """Public-session-compatible deterministic user simulator for one record.

    Agent code should use only ``reset``, ``step``, ``terminal``, ``domain``,
    and ``catalog``.  ``evaluation_summary`` is deliberately an evaluator-side
    handoff invoked after a rollout, not an input to the Agent.
    """

    record: Mapping[str, Any]
    candidate_pool: CandidatePool
    protocol: FrozenRuntimeProtocol = field(init=False)
    catalog: PublicCatalogTool = field(init=False)
    domain: str = field(init=False)
    candidate_pool_id: str = field(init=False)
    _phase: FrozenDialoguePhase = field(default=FrozenDialoguePhase.INITIAL, init=False)
    _terminal: bool = field(default=False, init=False)
    _success: bool = field(default=False, init=False)
    _termination_reason: str = field(default="ongoing", init=False)
    _turns: list[FrozenRuntimeTurn] = field(default_factory=list, init=False)
    _candidate_ids: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.protocol = runtime_protocol_from_record(self.record)
        if self.candidate_pool.index.domain != self.protocol.domain:
            raise ValueError(
                f"episode {self.protocol.episode_id} uses {self.protocol.domain}, "
                f"but candidate pool is {self.candidate_pool.index.domain}"
            )
        self.catalog = PublicCatalogTool(self.candidate_pool)
        self.domain = self.protocol.domain
        self.candidate_pool_id = self.candidate_pool.pool_id
        self._candidate_ids = set(self.candidate_pool.item_ids)

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def episode_id(self) -> str:
        return self.protocol.episode_id

    @property
    def private_turns(self) -> tuple[FrozenRuntimeTurn, ...]:
        """Evaluator provenance; never pass this value into an Agent prompt."""

        return tuple(self._turns)

    def reset(self) -> str:
        self._phase = FrozenDialoguePhase.INITIAL
        self._terminal = False
        self._success = False
        self._termination_reason = "ongoing"
        self._turns = []
        return _clean_text(self.record.get("initial_query"))

    def _response(self, text: str, *, terminal: bool) -> PublicEnvironmentResponse:
        return PublicEnvironmentResponse(
            utterance=_clean_text(text),
            terminal=terminal,
            turn_index=len(self._turns) + 1,
        )

    @staticmethod
    def _flatten_public_value(value: Any) -> str:
        if isinstance(value, list):
            return " ".join(FrozenMultiTurnSession._flatten_public_value(item) for item in value)
        return _clean_text(value)

    def _item_text(self, item_id: str) -> str:
        item = self.catalog.inspect(item_id)
        parts = [item.title]
        parts.extend(self._flatten_public_value(value) for value in item.attributes.values())
        return html.unescape(" ".join(parts)).casefold()

    @staticmethod
    def _rank_item_ids(item_ids: Iterable[str], accepted: set[str]) -> int | None:
        return next((position for position, item_id in enumerate(item_ids, start=1) if item_id in accepted), None)

    def _matches_fallback_terms(self, item_id: str) -> bool:
        terms = self.protocol.fallback_catalog_terms
        if not terms:
            return False
        text = self._item_text(item_id)
        return all(html.unescape(term).casefold() in text for term in terms)

    def _assess_recommendation(self, item_ids: Iterable[str]) -> FrozenRecommendationAssessment:
        raw_ids = tuple(item_ids)
        valid_ids = tuple(item_id for item_id in raw_ids if item_id in self._candidate_ids)
        unknown_ids = tuple(item_id for item_id in raw_ids if item_id not in self._candidate_ids)
        known_positive_rank = self._rank_item_ids(valid_ids, set(self.protocol.positive_item_ids))
        negative_set = set(self.protocol.hard_negative_item_ids)
        catalog_term_rank = next(
            (
                position
                for position, item_id in enumerate(valid_ids, start=1)
                if item_id not in negative_set and self._matches_fallback_terms(item_id)
            ),
            None,
        )
        hard_negatives = tuple(item_id for item_id in valid_ids if item_id in negative_set)
        return FrozenRecommendationAssessment(
            recommended_item_ids=valid_ids,
            unknown_item_ids=unknown_ids,
            known_positive_rank=known_positive_rank,
            catalog_term_rank=catalog_term_rank,
            hard_negative_item_ids=hard_negatives,
        )

    def _question_looks_targeted(self, action: BenchmarkSystemAction) -> bool:
        expected = _clean_text(self.record.get("transition", {}).get("agent_action_required", ""))
        private_terms = " ".join((*self.protocol.fallback_catalog_terms, expected))
        expected_tokens = {
            token
            for token in _TOKEN_RE.findall(private_terms.casefold())
            if token not in _QUESTION_STOPWORDS and len(token) > 2
        }
        question_text = f"{action.question_target or ''} {action.utterance}".casefold()
        question_tokens = {
            token
            for token in _TOKEN_RE.findall(question_text)
            if token not in _QUESTION_STOPWORDS and len(token) > 2
        }
        return bool(expected_tokens.intersection(question_tokens))

    def _append_turn(
        self,
        *,
        phase_before: FrozenDialoguePhase,
        action: BenchmarkSystemAction,
        response: PublicEnvironmentResponse,
        event: str,
        question_looks_targeted: bool | None = None,
        assessment: FrozenRecommendationAssessment | None = None,
    ) -> PublicEnvironmentResponse:
        if not response.terminal and len(self._turns) + 1 >= self.protocol.agent_turn_budget:
            self._terminal = True
            self._phase = FrozenDialoguePhase.TERMINAL
            self._termination_reason = "turn_budget_exhausted"
            response = self._response(
                f"{response.utterance} I do not want to extend the conversation further.",
                terminal=True,
            )
            event = f"{event}|turn_budget_exhausted"
        self._turns.append(
            FrozenRuntimeTurn(
                turn_index=len(self._turns),
                phase_before=phase_before,
                phase_after=self._phase,
                action_type=action.action_type.value,
                response_text=response.utterance,
                terminal=response.terminal,
                event=event,
                question_looks_targeted=question_looks_targeted,
                assessment=assessment,
            )
        )
        return response

    def _finish_recommendation(
        self,
        *,
        phase_before: FrozenDialoguePhase,
        action: BenchmarkSystemAction,
        stage: str,
    ) -> PublicEnvironmentResponse:
        assessment = self._assess_recommendation(action.item_ids)
        self._terminal = True
        self._phase = FrozenDialoguePhase.TERMINAL
        self._success = assessment.accepted
        self._termination_reason = "accepted" if assessment.accepted else "recommendation_not_accepted"
        if assessment.accepted:
            text = "This recommendation fits what I need. Thank you."
            event = f"{stage}_accepted"
        elif not assessment.recommended_item_ids:
            text = "Those item identifiers are not available in this catalog."
            event = f"{stage}_unknown_items"
        else:
            text = "These options do not meet my deciding requirement."
            event = f"{stage}_rejected"
        response = self._response(text, terminal=True)
        return self._append_turn(
            phase_before=phase_before,
            action=action,
            response=response,
            event=event,
            assessment=assessment,
        )

    def _transition_reply(self) -> str:
        reply = _clean_text(self.record.get("transition", {}).get("user_reply", ""))
        return reply or "I can clarify the deciding preference now."

    def _revision_feedback(self) -> str:
        transition = self.record.get("transition", {})
        priority = transition.get("resolved_priority", {}) if isinstance(transition, Mapping) else {}
        label = _clean_text(priority.get("label")) if isinstance(priority, Mapping) else "the revised priority"
        reply = self._transition_reply()
        return f"The first option is close, but I need to prioritize {label}. {reply}"

    def _reaction_feedback(self) -> str:
        """Return a precise but local reaction, not the answer to a targeted ASK.

        A user reacting to an unsatisfactory slate can name the practical
        mismatch, but should not emit the benchmark's catalog terms or the
        exact clarification answer simply because any item was shown.
        """

        transition = self.record.get("transition", {})
        priority = transition.get("resolved_priority", {}) if isinstance(transition, Mapping) else {}
        label = _clean_text(priority.get("label")) if isinstance(priority, Mapping) else ""
        if label:
            return f"These options are not quite right; please adjust them toward {label}."
        return "These options are not quite right for how I plan to use the product."

    @staticmethod
    def _untargeted_question_reply() -> str:
        return (
            "I am not sure which concrete distinction you mean. "
            "Please ask about a product attribute or trade-off that would change the ranking."
        )

    def _stop(self, phase_before: FrozenDialoguePhase, action: BenchmarkSystemAction) -> PublicEnvironmentResponse:
        self._terminal = True
        self._phase = FrozenDialoguePhase.TERMINAL
        self._success = False
        self._termination_reason = "agent_stopped"
        return self._append_turn(
            phase_before=phase_before,
            action=action,
            response=self._response("I will stop looking for a recommendation now.", terminal=True),
            event="agent_stop",
        )

    def step(self, action: BenchmarkSystemAction) -> PublicEnvironmentResponse:
        """Apply one typed Agent action and return only a public user utterance."""

        if self._terminal:
            raise RuntimeError("cannot step a frozen episode after it has terminated")
        phase_before = self._phase
        if action.action_type == BenchmarkActionType.STOP:
            return self._stop(phase_before, action)

        if self.protocol.mechanism == "direct":
            if action.action_type == BenchmarkActionType.RECOMMEND:
                return self._finish_recommendation(phase_before=phase_before, action=action, stage="direct")
            self._phase = FrozenDialoguePhase.READY_TO_RECOMMEND
            response = self._response(
                "Everything decision-critical is already in my request; please make a recommendation.",
                terminal=False,
            )
            return self._append_turn(
                phase_before=phase_before,
                action=action,
                response=response,
                event="redundant_nonrecommendation",
            )

        if self.protocol.mechanism == "revision":
            if self._phase == FrozenDialoguePhase.INITIAL:
                if action.action_type == BenchmarkActionType.ASK:
                    targeted = self._question_looks_targeted(action)
                    self._phase = FrozenDialoguePhase.READY_TO_REPAIR
                    return self._append_turn(
                        phase_before=phase_before,
                        action=action,
                        response=self._response(
                            (
                                "I would rather react to an initial option, but I can state the likely adjustment. "
                                f"{self._transition_reply() if targeted else self._untargeted_question_reply()}"
                            ),
                            terminal=False,
                        ),
                        event="question_before_revision_exposure",
                        question_looks_targeted=targeted,
                    )
                if action.action_type != BenchmarkActionType.RECOMMEND:
                    return self._append_turn(
                        phase_before=phase_before,
                        action=action,
                        response=self._response(
                            "I would like to react to an initial option before refining the request.",
                            terminal=False,
                        ),
                        event="nonrecommendation_before_revision_exposure",
                    )
                self._phase = FrozenDialoguePhase.READY_TO_REPAIR
                return self._append_turn(
                    phase_before=phase_before,
                    action=action,
                    response=self._response(self._revision_feedback(), terminal=False),
                    event="revision_feedback_revealed",
                    assessment=self._assess_recommendation(action.item_ids),
                )
            if (
                self._phase == FrozenDialoguePhase.READY_TO_REPAIR
                and action.action_type == BenchmarkActionType.RECOMMEND
            ):
                return self._finish_recommendation(phase_before=phase_before, action=action, stage="repair")
            return self._append_turn(
                phase_before=phase_before,
                action=action,
                response=self._response(
                    "I have stated the revised priority; please use it to repair the recommendation.", terminal=False
                ),
                event="redundant_question_after_revision",
                question_looks_targeted=(
                    self._question_looks_targeted(action) if action.action_type == BenchmarkActionType.ASK else None
                ),
            )

        # Progressive, ambiguous, and conflict examples have two legal paths.
        # Asking first reveals the deciding preference cheaply. Recommending
        # first also reveals the same frozen preference as item-grounded
        # feedback, so a Reactive Agent can recover, but the rejected slate is
        # retained as an interaction cost. This avoids making hidden mechanism
        # classification a structural prerequisite for task success.
        if self._phase == FrozenDialoguePhase.INITIAL:
            if action.action_type == BenchmarkActionType.ASK:
                targeted = self._question_looks_targeted(action)
                self._phase = FrozenDialoguePhase.READY_TO_RECOMMEND
                return self._append_turn(
                    phase_before=phase_before,
                    action=action,
                    response=self._response(
                        self._transition_reply() if targeted else self._untargeted_question_reply(),
                        terminal=False,
                    ),
                    event=(
                        "targeted_clarification_revealed"
                        if targeted
                        else "untargeted_question_no_decisive_information"
                    ),
                    question_looks_targeted=targeted,
                )
            if action.action_type == BenchmarkActionType.RECOMMEND:
                self._phase = FrozenDialoguePhase.READY_TO_RECOMMEND
                return self._append_turn(
                    phase_before=phase_before,
                    action=action,
                    response=self._response(self._reaction_feedback(), terminal=False),
                    event="premature_recommendation_local_feedback",
                    assessment=self._assess_recommendation(action.item_ids),
                )
            return self._append_turn(
                phase_before=phase_before,
                action=action,
                response=self._response(
                    "Before I can judge options, I need one clarification about my preference.", terminal=False
                ),
                event="premature_recommendation",
            )
        if (
            self._phase == FrozenDialoguePhase.READY_TO_RECOMMEND
            and action.action_type == BenchmarkActionType.RECOMMEND
        ):
            return self._finish_recommendation(phase_before=phase_before, action=action, stage="post_clarification")
        return self._append_turn(
            phase_before=phase_before,
            action=action,
            response=self._response(
                "I have already clarified the deciding preference; please use it to recommend.", terminal=False
            ),
            event="redundant_question_after_clarification",
            question_looks_targeted=(
                self._question_looks_targeted(action) if action.action_type == BenchmarkActionType.ASK else None
            ),
        )

    def evaluation_summary(self) -> dict[str, Any]:
        """Return private, reproducible metrics after a rollout has finished."""

        assessments = [turn.assessment for turn in self._turns if turn.assessment is not None]
        final_assessment = assessments[-1] if assessments else None
        events = [turn.event for turn in self._turns]
        ask_turns = [turn for turn in self._turns if turn.action_type == BenchmarkActionType.ASK.value]
        unnecessary = sum(
            marker in event
            for event in events
            for marker in ("premature_recommendation", "redundant_", "question_before_revision_exposure")
        )
        targeted = sum(turn.question_looks_targeted is True for turn in ask_turns)
        rejected_recommendations = sum(
            turn.action_type == BenchmarkActionType.RECOMMEND.value
            and not turn.event.endswith("_accepted")
            for turn in self._turns
        )
        utility_components = {
            "acceptance": 1.0 if self._success else 0.0,
            "turn_cost": -TURN_COST * len(self._turns),
            "question_cost": -QUESTION_COST * len(ask_turns),
            "rejected_recommendation_cost": -REJECTED_RECOMMENDATION_COST * rejected_recommendations,
            "unnecessary_action_cost": -UNNECESSARY_ACTION_COST * unnecessary,
        }
        success_at_1 = bool(final_assessment and final_assessment.relevant_within(1))
        success_at_3 = bool(final_assessment and final_assessment.relevant_within(3))
        success_at_10 = bool(final_assessment and final_assessment.relevant_within(10))
        return {
            "runtime_version": FROZEN_RUNTIME_VERSION,
            "interaction_utility_version": INTERACTION_UTILITY_VERSION,
            "episode_id": self.protocol.episode_id,
            "mechanism": self.protocol.mechanism,
            "success": self._success,
            "success_at_1": success_at_1,
            "success_at_3": success_at_3,
            "success_at_10": success_at_10,
            "terminal": self._terminal,
            "termination_reason": self._termination_reason,
            "agent_turn_count": len(self._turns),
            "agent_turn_budget": self.protocol.agent_turn_budget,
            "ask_count": len(ask_turns),
            "targeted_question_count": targeted,
            "untargeted_question_count": len(ask_turns) - targeted,
            "unnecessary_action_count": unnecessary,
            "rejected_recommendation_count": rejected_recommendations,
            "interaction_utility": sum(utility_components.values()),
            "interaction_utility_components": utility_components,
            "expected_action_sequence": list(self.protocol.expected_action_sequence),
            "final_recommendation": None if final_assessment is None else final_assessment.to_dict(),
            "turns": [turn.to_dict() for turn in self._turns],
        }


class FrozenMultiTurnBenchmark:
    """Lazy full-catalog benchmark facade for a frozen Agent test artifact."""

    def __init__(self, records: Iterable[Mapping[str, Any]], data_root: Path) -> None:
        materialized = [dict(record) for record in records]
        self._records = {runtime_protocol_from_record(record).episode_id: record for record in materialized}
        if len(self._records) != len(materialized):
            raise ValueError("frozen benchmark records have duplicate episode IDs")
        self._catalogs = FullCatalogRegistry(data_root)

    @classmethod
    def from_artifact(cls, artifact_path: Path, data_root: Path) -> FrozenMultiTurnBenchmark:
        return cls(load_frozen_agent_records(artifact_path), data_root)

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    def public_manifest(self) -> list[dict[str, str]]:
        """Return query-level metadata only; no transition/evaluation state leaks."""

        return [
            {
                "episode_id": episode_id,
                "domain": _clean_text(record.get("domain")),
                "mechanism": _clean_text(record.get("mechanism")),
                "initial_query": _clean_text(record.get("initial_query")),
            }
            for episode_id, record in sorted(self._records.items())
        ]

    def session(self, episode_id: str) -> FrozenMultiTurnSession:
        try:
            record = self._records[episode_id]
        except KeyError as exc:
            raise KeyError(f"unknown frozen episode: {episode_id}") from exc
        pool = self._catalogs.candidate_pool(_clean_text(record.get("domain")))
        return FrozenMultiTurnSession(record=record, candidate_pool=pool)


def materialize_runtime_protocols(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Produce a stable evaluator-side protocol ledger without making model calls."""

    return [runtime_protocol_from_record(record).to_dict() for record in records]
