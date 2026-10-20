"""Small public dialogue memory for the paper-canonical Agent path.

This is intentionally separate from evaluator state.  It records only text
that the recommendation system has observed, its own actions, and public item
identifiers that it has already shown.  DSUS and Thought Patterns will
consume this object in later phases.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from taira_mt.benchmark.schemas import BenchmarkActionType, BenchmarkSystemAction
from taira_mt.models import StrictModel


class PublicDialogueEvent(StrictModel):
    """One observable user or system turn."""

    speaker: Literal["user", "system"]
    text: str = Field(min_length=1)
    action: BenchmarkSystemAction | None = None

    @model_validator(mode="after")
    def validate_event(self) -> PublicDialogueEvent:
        if self.speaker == "system" and self.action is None:
            raise ValueError("a system event requires its executed action")
        if self.speaker == "user" and self.action is not None:
            raise ValueError("a user event cannot carry a system action")
        return self


class PublicDialogueMemory(StrictModel):
    """Replayable public state available before the next Agent decision."""

    domain: str = Field(min_length=1)
    initial_user_utterance: str = Field(min_length=1)
    events: list[PublicDialogueEvent] = Field(min_length=1)
    completed_action_count: int = Field(default=0, ge=0)
    asked_question_targets: list[str] = Field(default_factory=list)
    latest_recommendation_item_ids: list[str] = Field(default_factory=list)
    previously_recommended_item_ids: list[str] = Field(default_factory=list)
    terminal: bool = False

    @field_validator(
        "asked_question_targets",
        "latest_recommendation_item_ids",
        "previously_recommended_item_ids",
    )
    @classmethod
    def validate_unique_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("public dialogue memory lists must be unique")
        return value

    @model_validator(mode="after")
    def validate_shape(self) -> PublicDialogueMemory:
        if self.events[0].speaker != "user":
            raise ValueError("a public dialogue must begin with the user query")
        expected = "user"
        action_count = 0
        for event in self.events:
            if event.speaker != expected:
                raise ValueError("public dialogue events must alternate user and system")
            expected = "system" if expected == "user" else "user"
            action_count += int(event.speaker == "system")
        if action_count != self.completed_action_count:
            raise ValueError("completed_action_count differs from recorded system actions")
        return self

    @classmethod
    def begin(cls, *, domain: str, initial_user_utterance: str) -> PublicDialogueMemory:
        return cls(
            domain=domain,
            initial_user_utterance=initial_user_utterance,
            events=[PublicDialogueEvent(speaker="user", text=initial_user_utterance)],
        )

    @property
    def serialized_dialogue(self) -> str:
        return "\n".join(f"{event.speaker.title()}: {event.text}" for event in self.events)

    @property
    def user_utterances(self) -> list[str]:
        return [event.text for event in self.events if event.speaker == "user"]

    @property
    def retrieval_text(self) -> str:
        """All observed user language, suitable for a public retrieval query."""

        return " ".join(self.user_utterances)

    @property
    def latest_user_utterance(self) -> str:
        if self.events[-1].speaker != "user":
            raise RuntimeError("a system response is pending")
        return self.events[-1].text

    @property
    def last_system_action_type(self) -> BenchmarkActionType | None:
        for event in reversed(self.events):
            if event.action is not None:
                return event.action.action_type
        return None

    def record_system_action(self, action: BenchmarkSystemAction) -> PublicDialogueMemory:
        if self.terminal:
            raise RuntimeError("cannot append an action after a terminal response")
        if self.events[-1].speaker != "user":
            raise RuntimeError("a user response is required before the next Agent action")
        asked_targets = list(self.asked_question_targets)
        if action.action_type == BenchmarkActionType.ASK:
            assert action.question_target is not None
            if action.question_target not in asked_targets:
                asked_targets.append(action.question_target)
        latest = list(self.latest_recommendation_item_ids)
        previously = list(self.previously_recommended_item_ids)
        if action.action_type == BenchmarkActionType.RECOMMEND:
            latest = list(action.item_ids)
            previously.extend(item_id for item_id in action.item_ids if item_id not in previously)
        return self.model_copy(
            update={
                "events": [*self.events, PublicDialogueEvent(speaker="system", text=action.utterance, action=action)],
                "completed_action_count": self.completed_action_count + 1,
                "asked_question_targets": asked_targets,
                "latest_recommendation_item_ids": latest,
                "previously_recommended_item_ids": previously,
            },
            deep=True,
        )

    def record_user_response(self, utterance: str, *, terminal: bool) -> PublicDialogueMemory:
        if self.events[-1].speaker != "system":
            raise RuntimeError("a system action is required before the user response")
        return self.model_copy(
            update={
                "events": [*self.events, PublicDialogueEvent(speaker="user", text=utterance)],
                "terminal": terminal,
            },
            deep=True,
        )
