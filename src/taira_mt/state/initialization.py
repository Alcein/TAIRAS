"""Initialize belief from public ontology and extracted request evidence only."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field, field_validator, model_validator

from taira_mt.models import (
    AttributeValue,
    BeliefState,
    ConstraintPredicate,
    StrictModel,
    SystemActionType,
    normalize_slot_name,
)
from taira_mt.recommender.catalog import CandidatePool


class SlotDefinition(StrictModel):
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    askable: bool = True
    priority: int = 0

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return normalize_slot_name(value)

    @field_validator("aliases", mode="before")
    @classmethod
    def normalize_aliases(cls, value: list[str]) -> list[str]:
        return [normalize_slot_name(alias) for alias in value]


class DomainSlotOntology(StrictModel):
    domain: str
    slots: list[SlotDefinition]

    @model_validator(mode="after")
    def validate_slots(self) -> DomainSlotOntology:
        names = [slot.name for slot in self.slots]
        if len(names) != len(set(names)):
            raise ValueError("ontology slot names must be unique")
        all_labels = [label for slot in self.slots for label in [slot.name, *slot.aliases]]
        if len(all_labels) != len(set(all_labels)):
            raise ValueError("ontology names and aliases must be globally unique")
        return self

    def canonicalize(self, slot_name: str) -> str:
        normalized = normalize_slot_name(slot_name)
        for slot in self.slots:
            if normalized == slot.name or normalized in slot.aliases:
                return slot.name
        return normalized


def initialize_belief(
    observed_constraints: dict[str, AttributeValue | ConstraintPredicate],
    ontology: DomainSlotOntology,
    candidate_ids: Iterable[str] | None = None,
    patience_prior: int = 5,
    *,
    candidate_pool: CandidatePool | None = None,
) -> BeliefState:
    """Build the initial policy state without access to ``HiddenGoal``."""

    if candidate_pool is not None and candidate_ids is not None:
        raise ValueError("provide candidate_ids or candidate_pool, not both")
    materialized_candidate_ids = (
        candidate_pool.item_ids if candidate_pool is not None else tuple(candidate_ids or ())
    )
    if not materialized_candidate_ids:
        raise ValueError("candidate_ids cannot be empty at dialogue start")
    if len(materialized_candidate_ids) != len(set(materialized_candidate_ids)):
        raise ValueError("candidate_ids must be unique")
    canonical_observed: dict[str, AttributeValue | ConstraintPredicate] = {}
    for slot, value in observed_constraints.items():
        canonical = ontology.canonicalize(slot)
        if canonical in canonical_observed:
            raise ValueError(f"duplicate observed constraint after alias normalization: {canonical}")
        canonical_observed[canonical] = value

    ordered_askable = [
        slot.name
        for slot in sorted(ontology.slots, key=lambda item: (-item.priority, item.name))
        if slot.askable
    ]
    missing = [slot for slot in ordered_askable if slot not in canonical_observed]
    denominator = len(canonical_observed) + len(missing)
    readiness = len(canonical_observed) / denominator if denominator else 1.0
    legal_actions = [SystemActionType.RECOMMEND]
    if missing:
        legal_actions.insert(0, SystemActionType.ASK)
    belief = BeliefState(
        known_constraints=canonical_observed,
        missing_slots=missing,
        uncertainty=1.0 - readiness,
        readiness=readiness,
        patience_estimate=patience_prior,
        available_candidate_count=len(materialized_candidate_ids),
        legal_actions=legal_actions,
    )
    if candidate_pool is not None:
        from taira_mt.recommender.oracle import posterior_catalog_from_belief

        posterior_count = len(posterior_catalog_from_belief(belief, candidate_pool))
        belief = belief.model_copy(update={"available_candidate_count": posterior_count})
    return belief
