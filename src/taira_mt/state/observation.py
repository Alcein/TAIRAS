"""Public-text observation builders for oracle/rule/learned tracker separation."""

from __future__ import annotations

import re
from typing import Protocol

from taira_mt.models import (
    BeliefState,
    ConstraintImportance,
    ConstraintOperator,
    ConstraintPredicate,
    NumericRange,
    SystemAction,
    SystemActionType,
    UserAct,
    UserObservation,
)
from taira_mt.state.initialization import DomainSlotOntology


class ObservationBuilder(Protocol):
    """A tracker sees public text and prior public policy state only."""

    name: str

    def build(self, state: BeliefState, action: SystemAction, public_user_text: str) -> UserObservation: ...


class RuleBasedObservationBuilder:
    """Transparent smoke baseline; a learned tracker will implement the same API."""

    name = "rule_based_public_text"
    _preference_pattern = re.compile(
        r"(?:^|\s)For\s+(?P<slot>[\w\s_-]+),\s+I\s+(?:prefer|need)\s+(?P<value>.+?)\.?$",
        flags=re.IGNORECASE,
    )
    _repair_reveal_pattern = re.compile(
        r"\bI need\s+(?P<slot>[\w\s_-]+?)\s+to be\s+(?P<value>.+?)\.?$",
        flags=re.IGNORECASE,
    )
    _dismissed_pattern = re.compile(
        r"I do not have a particular preference about\s+(?P<slot>[\w\s_-]+?)\.?$",
        flags=re.IGNORECASE,
    )
    _number = r"(?P<number>\d+(?:\.\d+)?)"

    def __init__(self, ontology: DomainSlotOntology | None = None) -> None:
        self.ontology = ontology

    def build(self, state: BeliefState, action: SystemAction, public_user_text: str) -> UserObservation:
        del state  # Explicitly unused: no latent state is available to this parser.
        text = public_user_text.strip()
        folded = text.casefold()
        extracted: dict[str, ConstraintPredicate] = {}
        dismissed: list[str] = []
        importance = (
            ConstraintImportance.SOFT if "ideally" in folded else ConstraintImportance.HARD
        )

        dismissed_match = self._dismissed_pattern.search(text)
        if dismissed_match:
            raw_slot = dismissed_match.group("slot")
            slot = self.ontology.canonicalize(raw_slot) if self.ontology is not None else raw_slot
            dismissed.append(slot)

        rating_match = re.search(
            r"rating\s+of\s+at\s+least\s+(?P<number>\d+(?:\.\d+)?)\s+stars?",
            text,
            flags=re.IGNORECASE,
        )
        if rating_match:
            extracted["rating"] = ConstraintPredicate(
                operator=ConstraintOperator.GTE,
                value=float(rating_match.group("number")),
                importance=importance,
                unit="stars",
            )

        popularity_match = re.search(
            r"top\s+(?P<number>\d+(?:\.\d+)?)%\s+by\s+popularity",
            text,
            flags=re.IGNORECASE,
        )
        if popularity_match:
            percentile = 1.0 - float(popularity_match.group("number")) / 100.0
            extracted["popularity_percentile"] = ConstraintPredicate(
                operator=ConstraintOperator.GTE,
                value=max(0.0, min(1.0, percentile)),
                importance=importance,
                unit="percentile",
            )

        price_between_match = re.search(
            r"spend\s+between\s+\$?(?P<minimum>\d+(?:\.\d+)?)\s+and\s+"
            r"\$?(?P<maximum>\d+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if price_between_match:
            extracted["price"] = ConstraintPredicate(
                operator=ConstraintOperator.BETWEEN,
                value=NumericRange(
                    minimum=float(price_between_match.group("minimum")),
                    maximum=float(price_between_match.group("maximum")),
                ),
                importance=importance,
                unit="USD",
            )
        else:
            price_max_match = re.search(
                r"(?:budget\s+is\s+at\s+most|stay\s+under)\s+\$?(?P<number>\d+(?:\.\d+)?)",
                text,
                flags=re.IGNORECASE,
            )
            if price_max_match:
                extracted["price"] = ConstraintPredicate(
                    operator=ConstraintOperator.LTE,
                    value=float(price_max_match.group("number")),
                    importance=importance,
                    unit="USD",
                )

        preference_match = self._preference_pattern.search(text)
        if preference_match:
            raw_slot = preference_match.group("slot")
            slot = self.ontology.canonicalize(raw_slot) if self.ontology is not None else raw_slot.strip().casefold()
            raw_value = preference_match.group("value").strip()
            operator = ConstraintOperator.EQ
            for candidate in ConstraintOperator:
                prefix = f"{candidate.value.upper()} "
                if raw_value.startswith(prefix):
                    operator = candidate
                    raw_value = raw_value[len(prefix) :]
                    break
            extracted[slot] = ConstraintPredicate(
                operator=operator,
                value=raw_value,
                importance=importance,
            )
        repair_reveal_match = self._repair_reveal_pattern.search(text)
        if repair_reveal_match:
            raw_slot = repair_reveal_match.group("slot")
            slot = self.ontology.canonicalize(raw_slot) if self.ontology is not None else raw_slot.strip().casefold()
            extracted[slot] = ConstraintPredicate(
                operator=ConstraintOperator.EQ,
                value=repair_reveal_match.group("value").strip(),
                importance=ConstraintImportance.HARD,
            )

        if "works for me" in folded:
            predicted_act = UserAct.ACCEPT
        elif "taking too long" in folded or "we can stop" in folded:
            predicted_act = UserAct.ABANDON
        elif "do not fit" in folded or "not available" in folded:
            predicted_act = UserAct.REJECT
        elif "could you" in folded and action.action_type != SystemActionType.ASK:
            predicted_act = UserAct.REQUEST_EXPLANATION
        elif action.action_type == SystemActionType.ASK or preference_match or extracted or dismissed:
            predicted_act = UserAct.INFORM
        else:
            predicted_act = UserAct.OTHER

        return UserObservation(
            text=text,
            predicted_act=predicted_act,
            extracted_constraints=extracted,
            dismissed_slots=dismissed,
        )
