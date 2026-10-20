"""Public interaction memory and evidence guards for a live TAIRAS Agent.

No class in this module accepts a benchmark case, response frame, runtime user
state, hidden goal, oracle result, or evaluator trace.  A language interpreter
may propose an update, but every extracted constraint must point to literal
public text before it affects DSUS or Pattern planning.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, field_validator, model_validator

from taira_mt.agent.catalog_tools import PublicCatalogTool
from taira_mt.benchmark.schemas import (
    BenchmarkActionType,
    BenchmarkSystemAction,
    BenchmarkUserAct,
)
from taira_mt.dsus.schemas import PublicConstraintObservation, PublicDSUSFeature
from taira_mt.models import StrictModel, normalize_slot_name


class PublicDialogueSpeaker(str, Enum):
    USER = "user"
    SYSTEM = "system"


class GroundedPublicConstraint(StrictModel):
    """One parser-proposed preference backed by a literal user-text quote."""

    observation: PublicConstraintObservation
    evidence_quote: str = Field(min_length=1)

    @field_validator("evidence_quote")
    @classmethod
    def nonblank_evidence_quote(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("constraint evidence quote cannot be blank")
        return value.strip()


class PublicUtteranceInterpretation(StrictModel):
    """A public-only parser output, suitable for an LLM or a rule baseline."""

    user_act: BenchmarkUserAct | None = None
    grounded_constraints: list[GroundedPublicConstraint] = Field(default_factory=list)
    dismissed_slots: list[str] = Field(default_factory=list)
    referenced_item_ids: list[str] = Field(default_factory=list)

    @field_validator("dismissed_slots")
    @classmethod
    def normalize_dismissed_slots(cls, value: list[str]) -> list[str]:
        normalized = [normalize_slot_name(item) for item in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("dismissed public slots must be unique")
        return normalized

    @field_validator("referenced_item_ids")
    @classmethod
    def unique_referenced_items(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("referenced public item IDs must be unique")
        return value

    @model_validator(mode="after")
    def unique_constraint_slots(self) -> PublicUtteranceInterpretation:
        slots = [item.observation.slot for item in self.grounded_constraints]
        if len(slots) != len(set(slots)):
            raise ValueError("one user interpretation may update each public slot once")
        return self


class PublicDialogueEvent(StrictModel):
    speaker: PublicDialogueSpeaker
    text: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    system_action: BenchmarkSystemAction | None = None

    @model_validator(mode="after")
    def validate_speaker_payload(self) -> PublicDialogueEvent:
        if self.speaker == PublicDialogueSpeaker.SYSTEM and self.system_action is None:
            raise ValueError("a system dialogue event requires the executed public action")
        if self.speaker == PublicDialogueSpeaker.USER and self.system_action is not None:
            raise ValueError("a user dialogue event cannot contain a system action")
        return self


class PublicInteractionMemory(StrictModel):
    """The complete online state visible to a TAIRAS decision cycle."""

    domain: str = Field(min_length=1)
    events: list[PublicDialogueEvent] = Field(min_length=1)
    public_constraints: list[PublicConstraintObservation] = Field(default_factory=list)
    asked_question_targets: list[str] = Field(default_factory=list)
    rejected_item_ids: list[str] = Field(default_factory=list)
    latest_recommendation_item_ids: list[str] = Field(default_factory=list)
    last_system_action_type: BenchmarkActionType | None = None
    last_user_act: BenchmarkUserAct | None = None
    completed_turn_count: int = Field(default=0, ge=0)

    @field_validator(
        "asked_question_targets",
        "rejected_item_ids",
        "latest_recommendation_item_ids",
    )
    @classmethod
    def unique_memory_lists(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("public interaction memory lists must be unique")
        return value

    @model_validator(mode="after")
    def validate_interaction_shape(self) -> PublicInteractionMemory:
        if self.events[0].speaker != PublicDialogueSpeaker.USER:
            raise ValueError("public interaction must start with the initial user query")
        expected = PublicDialogueSpeaker.USER
        user_turns = 0
        for event in self.events:
            if event.speaker != expected:
                raise ValueError("public dialogue events must alternate user and system")
            if event.speaker == PublicDialogueSpeaker.USER:
                user_turns += 1
                expected = PublicDialogueSpeaker.SYSTEM
            else:
                expected = PublicDialogueSpeaker.USER
        if self.completed_turn_count != user_turns - 1:
            raise ValueError("completed turn count must equal non-initial user events")
        slots = [item.slot for item in self.public_constraints]
        if len(slots) != len(set(slots)):
            raise ValueError("public interaction constraints must have unique slots")
        if set(self.rejected_item_ids).intersection(self.latest_recommendation_item_ids):
            raise ValueError("latest recommendation cannot contain a rejected item")
        return self

    @classmethod
    def begin(
        cls,
        *,
        domain: str,
        initial_utterance: str,
        interpretation: PublicUtteranceInterpretation,
    ) -> PublicInteractionMemory:
        """Initialize memory from a public query and a grounded parser result."""

        memory = cls(
            domain=domain,
            events=[
                PublicDialogueEvent(
                    speaker=PublicDialogueSpeaker.USER,
                    text=initial_utterance,
                    turn_index=0,
                )
            ],
            last_user_act=interpretation.user_act,
        )
        return memory._apply_user_interpretation(initial_utterance, interpretation)

    @property
    def serialized_dialogue(self) -> str:
        return "\n".join(
            f"{event.speaker.value.title()}: {event.text}" for event in self.events
        )

    def _apply_user_interpretation(
        self,
        utterance: str,
        interpretation: PublicUtteranceInterpretation,
    ) -> PublicInteractionMemory:
        folded_utterance = utterance.casefold()
        constraints = {item.slot: item for item in self.public_constraints}
        for grounded in interpretation.grounded_constraints:
            if grounded.evidence_quote.casefold() not in folded_utterance:
                raise ValueError("parser constraint evidence quote is absent from public user text")
            constraints[grounded.observation.slot] = grounded.observation
        for slot in interpretation.dismissed_slots:
            constraints.pop(slot, None)
        rejected = list(self.rejected_item_ids)
        if interpretation.user_act == BenchmarkUserAct.REJECT:
            rejected.extend(
                item_id
                for item_id in self.latest_recommendation_item_ids
                if item_id not in rejected
            )
        referenced = set(interpretation.referenced_item_ids)
        known_recommendations = set(self.latest_recommendation_item_ids).union(rejected)
        if not referenced.issubset(known_recommendations):
            raise ValueError("user text may reference only an item introduced by a public recommendation")
        latest = [item for item in self.latest_recommendation_item_ids if item not in set(rejected)]
        return self.model_copy(
            update={
                "public_constraints": list(constraints.values()),
                "rejected_item_ids": rejected,
                "latest_recommendation_item_ids": latest,
                "last_user_act": interpretation.user_act,
            },
            deep=True,
        )

    def record_system_action(self, action: BenchmarkSystemAction) -> PublicInteractionMemory:
        """Append a typed, public action before the simulator returns a user utterance."""

        if self.events[-1].speaker != PublicDialogueSpeaker.USER:
            raise ValueError("a user response is required before the next system action")
        asked = list(self.asked_question_targets)
        if action.action_type == BenchmarkActionType.ASK:
            assert action.question_target is not None
            target = normalize_slot_name(action.question_target)
            if target in asked:
                raise ValueError("TAIRAS public memory forbids repeated question targets")
            asked.append(target)
        rejected = set(self.rejected_item_ids)
        if action.action_type == BenchmarkActionType.RECOMMEND and rejected.intersection(action.item_ids):
            raise ValueError("TAIRAS cannot re-recommend a publicly rejected item")
        if action.action_type == BenchmarkActionType.EXPLAIN:
            if not set(action.item_ids).issubset(self.latest_recommendation_item_ids):
                raise ValueError("an explanation must ground a currently public recommendation")
            evidence_items = {item.item_id for item in action.evidence_refs}
            if not evidence_items.issubset(self.latest_recommendation_item_ids):
                raise ValueError("explanation evidence must reference a currently public recommendation")
        latest = (
            list(action.item_ids)
            if action.action_type == BenchmarkActionType.RECOMMEND
            else list(self.latest_recommendation_item_ids)
        )
        return self.model_copy(
            update={
                "events": [
                    *self.events,
                    PublicDialogueEvent(
                        speaker=PublicDialogueSpeaker.SYSTEM,
                        text=action.utterance,
                        turn_index=self.completed_turn_count,
                        system_action=action,
                    ),
                ],
                "asked_question_targets": asked,
                "latest_recommendation_item_ids": latest,
                "last_system_action_type": action.action_type,
            },
            deep=True,
        )

    def replay_observed_system_action(self, action: BenchmarkSystemAction) -> PublicInteractionMemory:
        """Append an off-policy historical action without treating it as Agent-legal.

        Parser-aligned DSUS supervision deliberately includes random and
        failure-correction trajectories.  Those histories can contain repeated
        questions or rejected-item recommendations that the online Agent must
        never emit.  Replay records their *public text* while normalizing memory
        summaries so they cannot become an online legality loophole.
        """

        if self.events[-1].speaker != PublicDialogueSpeaker.USER:
            raise ValueError("a user response is required before a replayed system action")
        asked = list(self.asked_question_targets)
        if action.action_type == BenchmarkActionType.ASK:
            assert action.question_target is not None
            target = normalize_slot_name(action.question_target)
            if target not in asked:
                asked.append(target)
        rejected = set(self.rejected_item_ids)
        latest = (
            [item_id for item_id in action.item_ids if item_id not in rejected]
            if action.action_type == BenchmarkActionType.RECOMMEND
            else list(self.latest_recommendation_item_ids)
        )
        return self.model_copy(
            update={
                "events": [
                    *self.events,
                    PublicDialogueEvent(
                        speaker=PublicDialogueSpeaker.SYSTEM,
                        text=action.utterance,
                        turn_index=self.completed_turn_count,
                        system_action=action,
                    ),
                ],
                "asked_question_targets": asked,
                "latest_recommendation_item_ids": latest,
                "last_system_action_type": action.action_type,
            },
            deep=True,
        )

    def record_user_response(
        self,
        utterance: str,
        interpretation: PublicUtteranceInterpretation,
    ) -> PublicInteractionMemory:
        """Append a user utterance and apply only grounded public facts."""

        if self.events[-1].speaker != PublicDialogueSpeaker.SYSTEM:
            raise ValueError("a system action is required before a user response")
        next_memory = self.model_copy(
            update={
                "events": [
                    *self.events,
                    PublicDialogueEvent(
                        speaker=PublicDialogueSpeaker.USER,
                        text=utterance,
                        turn_index=self.completed_turn_count + 1,
                    ),
                ],
                "completed_turn_count": self.completed_turn_count + 1,
            },
            deep=True,
        )
        return next_memory._apply_user_interpretation(utterance, interpretation)

    def to_dsus_feature(self, catalog: PublicCatalogTool) -> PublicDSUSFeature:
        """Materialize the state-estimator input from public memory and tool output."""

        if catalog.domain != self.domain:
            raise ValueError("public catalog domain differs from interaction memory domain")
        result = catalog.search(
            self.public_constraints,
            excluded_item_ids=self.rejected_item_ids,
        )
        return PublicDSUSFeature(
            domain=self.domain,
            turn_index=self.completed_turn_count,
            serialized_dialogue=self.serialized_dialogue,
            public_constraints=self.public_constraints,
            public_verified_candidate_count=len(result.verified_item_ids),
            public_unknown_candidate_count=len(result.unknown_evidence_item_ids),
            asked_question_targets=self.asked_question_targets,
            rejected_item_count=len(self.rejected_item_ids),
            last_system_action_type=self.last_system_action_type,
        )
