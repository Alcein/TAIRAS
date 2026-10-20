"""Whole-trajectory language rendering for frozen logical event plans."""

from __future__ import annotations

import json
import re

from pydantic import Field, field_validator, model_validator

from taira_mt.models import StrictModel
from taira_mt.state_training.authoring import RichLogicalTrajectoryPlan, RichSourceRoot

_FORBIDDEN_SURFACE_MARKERS = (
    "benchmark",
    "evaluator",
    "fact_id",
    "gold state",
    "hidden state",
    "mechanism",
    "optimal action",
    "schema",
    "specific_category",
    "popularity_percentile",
)


class FactSurfaceEvidence(StrictModel):
    fact_id: str = Field(min_length=1)
    exact_quotes: list[str] = Field(min_length=1, max_length=8)

    @field_validator("exact_quotes")
    @classmethod
    def normalize_quotes(cls, values: list[str]) -> list[str]:
        normalized = [" ".join(value.split()) for value in values]
        if any(len(value) < 2 for value in normalized):
            raise ValueError("surface evidence quotes cannot be empty")
        if len(normalized) != len(set(value.casefold() for value in normalized)):
            raise ValueError("surface evidence quotes must be unique")
        return normalized


class RenderedTurn(StrictModel):
    turn_index: int = Field(ge=1, le=8)
    system_utterance: str = Field(min_length=2, max_length=600)
    user_utterance: str = Field(min_length=2, max_length=600)
    user_event_evidence: list[str] = Field(min_length=1, max_length=3)
    presented_option_evidence: list[FactSurfaceEvidence] = Field(default_factory=list, max_length=4)
    revealed_fact_evidence: list[FactSurfaceEvidence] = Field(default_factory=list, max_length=4)

    @field_validator("system_utterance", "user_utterance", mode="before")
    @classmethod
    def normalize_utterance(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("rendered utterance must be text")
        return " ".join(value.split())

    @model_validator(mode="after")
    def reject_construction_language(self) -> RenderedTurn:
        text = f"{self.system_utterance} {self.user_utterance}".casefold()
        marker = next((value for value in _FORBIDDEN_SURFACE_MARKERS if value in text), None)
        if marker is not None:
            raise ValueError(f"rendered dialogue contains forbidden construction marker: {marker}")
        normalized_event_quotes = [" ".join(value.split()) for value in self.user_event_evidence]
        if any(
            not value or value.casefold() not in self.user_utterance.casefold() for value in normalized_event_quotes
        ):
            raise ValueError("user-event evidence must be an exact user-utterance substring")
        for evidence in self.presented_option_evidence:
            if any(quote.casefold() not in self.system_utterance.casefold() for quote in evidence.exact_quotes):
                raise ValueError("presented-option evidence must be an exact system-utterance substring")
        self.user_event_evidence = normalized_event_quotes
        return self


class RenderedTrajectoryVariant(StrictModel):
    root_id: str = Field(min_length=1)
    variant_id: str = Field(pattern=r"^v[12]$")
    initial_user_utterance: str = Field(min_length=2, max_length=800)
    initial_fact_evidence: list[FactSurfaceEvidence] = Field(default_factory=list, max_length=12)
    turns: list[RenderedTurn] = Field(min_length=1, max_length=8)

    @field_validator("initial_user_utterance", mode="before")
    @classmethod
    def normalize_initial_utterance(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("initial user utterance must be text")
        return " ".join(value.split())


class RenderedTrajectoryBatch(StrictModel):
    variants: list[RenderedTrajectoryVariant] = Field(min_length=2, max_length=40)


def _normalized_transcript(variant: RenderedTrajectoryVariant) -> str:
    values = [variant.initial_user_utterance]
    for turn in variant.turns:
        values.extend((turn.system_utterance, turn.user_utterance))
    return re.sub(r"\W+", " ", " ".join(values).casefold()).strip()


def validate_rendered_variant(
    variant: RenderedTrajectoryVariant,
    *,
    source: RichSourceRoot,
    plan: RichLogicalTrajectoryPlan,
) -> None:
    if variant.root_id != source.root_id or variant.root_id != plan.root_id:
        raise ValueError("rendered trajectory root_id does not match source and logical plan")
    if [turn.turn_index for turn in variant.turns] != list(range(1, len(plan.turns) + 1)):
        raise ValueError("rendered turn indices do not match the logical plan")
    if len(variant.turns) != len(plan.turns):
        raise ValueError("rendered trajectory turn count does not match the logical plan")

    fact_ids = {fact.fact_id for fact in plan.semantic_facts}
    initially_public = {fact.fact_id for fact in plan.semantic_facts if fact.initially_public}
    initial_by_fact = {evidence.fact_id: evidence for evidence in variant.initial_fact_evidence}
    if len(initial_by_fact) != len(variant.initial_fact_evidence):
        raise ValueError("initial utterance repeats one fact evidence record")
    if set(initial_by_fact) != initially_public:
        raise ValueError("initial evidence fact IDs do not exactly match initially public facts")
    initial_text = variant.initial_user_utterance.casefold()
    for evidence in initial_by_fact.values():
        for quote in evidence.exact_quotes:
            if quote.casefold() not in initial_text:
                raise ValueError(f"initial evidence quote is absent from initial utterance: {quote}")
    revealed_so_far = set(initially_public)

    for rendered, logical in zip(variant.turns, plan.turns, strict=True):
        option_by_fact = {evidence.fact_id: evidence for evidence in rendered.presented_option_evidence}
        if len(option_by_fact) != len(rendered.presented_option_evidence):
            raise ValueError("rendered turn repeats one presented option evidence record")
        expected_options = set(logical.choice_fact_ids)
        if set(option_by_fact) != expected_options:
            raise ValueError("presented option fact IDs do not exactly match planned alternatives")
        if not expected_options <= fact_ids:
            raise ValueError("rendered turn presents an option outside the logical plan")

        evidence_by_fact = {evidence.fact_id: evidence for evidence in rendered.revealed_fact_evidence}
        if len(evidence_by_fact) != len(rendered.revealed_fact_evidence):
            raise ValueError("rendered turn repeats one fact evidence record")
        expected = set(logical.reveal_fact_ids)
        if set(evidence_by_fact) != expected:
            raise ValueError("rendered evidence fact IDs do not exactly match authorized reveals")
        if not expected <= fact_ids:
            raise ValueError("rendered turn references a fact outside the logical plan")
        user_text = rendered.user_utterance.casefold()
        for evidence in evidence_by_fact.values():
            for quote in evidence.exact_quotes:
                if quote.casefold() not in user_text:
                    raise ValueError(f"surface evidence quote is absent from user utterance: {quote}")
        revealed_so_far.update(expected | expected_options)

    never_revealed = fact_ids - revealed_so_far
    if never_revealed:
        raise ValueError(f"logical semantic facts never become public: {sorted(never_revealed)}")


def renderer_messages(
    source: RichSourceRoot,
    plan: RichLogicalTrajectoryPlan,
) -> tuple[str, dict[str, object]]:
    """Ask for two state-preserving whole-trajectory realizations."""

    system_prompt = (
        "You render one already-frozen logical shopping trajectory into two natural dialogue variants. "
        "Do not change, omit, anticipate, or add semantic events. Write the whole trajectory coherently, "
        "not isolated sentences. A fact may appear only on the turn whose reveal_fact_ids authorizes it. "
        "For each initially public or newly revealed fact, return one or more exact substrings copied from "
        "the corresponding user utterance. When choice_fact_ids is non-empty, the system must explicitly "
        "contrast every listed alternative and presented_option_evidence must quote exact system spans. "
        "System suggestions are not user commitments: only user-utterance spans count as revealed facts. "
        "Do not mention construction terminology, labels, IDs, states, schemas, evaluators, or benchmarks. "
        "The two variants must differ materially in wording/register while preserving identical events. "
        "Reply with exactly one JSON object containing variants."
    )
    public_plan = {
        "root_id": plan.root_id,
        "mechanism_hidden_from_surface": plan.mechanism.value,
        "source_shopping_need": source.query,
        "semantic_facts": [
            {
                "fact_id": fact.fact_id,
                "name": fact.name,
                "value": fact.value,
                "polarity": fact.polarity.value,
                "importance": fact.importance.value,
                "initially_public": fact.initially_public,
            }
            for fact in plan.semantic_facts
        ],
        "turns": [turn.model_dump(mode="json") for turn in plan.turns],
    }
    schema = {
        "variants": [
            {
                "root_id": "copy exactly",
                "variant_id": "v1 or v2",
                "initial_user_utterance": "natural initial shopping request",
                "initial_fact_evidence": [
                    {"fact_id": "initially_public fact only", "exact_quotes": ["exact initial-user span"]}
                ],
                "turns": [
                    {
                        "turn_index": "copy exactly",
                        "system_utterance": "natural system action matching system_intent",
                        "user_utterance": "natural response matching user_event",
                        "user_event_evidence": ["exact span expressing the user event"],
                        "presented_option_evidence": [
                            {
                                "fact_id": "every choice_fact_id, otherwise none",
                                "exact_quotes": ["exact system span expressing the alternative"],
                            }
                        ],
                        "revealed_fact_evidence": [
                            {"fact_id": "authorized reveal only", "exact_quotes": ["exact user span"]}
                        ],
                    }
                ],
            }
        ]
    }
    return system_prompt, {"logical_plan": public_plan, "required_output_schema": schema}


def parse_rendered_batch(
    content: str,
    *,
    source: RichSourceRoot,
    plan: RichLogicalTrajectoryPlan,
) -> list[RenderedTrajectoryVariant]:
    value = json.loads(content)
    # Utterance-level evidence is exact, public, and deterministic.  Models
    # frequently return punctuation-normalized near-spans, so code owns span
    # alignment while the independent semantic judge owns entailment.
    if isinstance(value, dict) and isinstance(value.get("variants"), list):
        for raw_variant in value["variants"]:
            if not isinstance(raw_variant, dict) or not isinstance(raw_variant.get("turns"), list):
                continue
            initial_text = raw_variant.get("initial_user_utterance")
            if isinstance(initial_text, str):
                raw_variant["initial_fact_evidence"] = [
                    {"fact_id": fact.fact_id, "exact_quotes": [initial_text]}
                    for fact in plan.semantic_facts
                    if fact.initially_public
                ]
            for raw_turn, logical_turn in zip(raw_variant["turns"], plan.turns, strict=False):
                if not isinstance(raw_turn, dict):
                    continue
                user_text = raw_turn.get("user_utterance")
                system_text = raw_turn.get("system_utterance")
                if isinstance(user_text, str):
                    raw_turn["user_event_evidence"] = [user_text]
                    raw_turn["revealed_fact_evidence"] = [
                        {"fact_id": fact_id, "exact_quotes": [user_text]}
                        for fact_id in logical_turn.reveal_fact_ids
                    ]
                if isinstance(system_text, str):
                    raw_turn["presented_option_evidence"] = [
                        {"fact_id": fact_id, "exact_quotes": [system_text]}
                        for fact_id in logical_turn.choice_fact_ids
                    ]
    batch = RenderedTrajectoryBatch.model_validate(value)
    variants = [variant for variant in batch.variants if variant.root_id == source.root_id]
    if len(variants) != 2 or {variant.variant_id for variant in variants} != {"v1", "v2"}:
        raise ValueError("renderer must return exactly v1 and v2 for the requested root")
    if len(batch.variants) != 2:
        raise ValueError("renderer returned unexpected extra trajectory variants")
    for variant in variants:
        validate_rendered_variant(variant, source=source, plan=plan)
    if _normalized_transcript(variants[0]) == _normalized_transcript(variants[1]):
        raise ValueError("renderer variants are textually identical")
    return sorted(variants, key=lambda variant: variant.variant_id)
