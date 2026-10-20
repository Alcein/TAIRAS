"""State-addressed Thought Patterns.

The library preserves TAIRA's three-scale Pattern semantics: task description
is the retrieval key, solution description holds transferable insight, and a
typed Thought Template is the executable SOP.  In TAIRAS a Thought Pattern is matched
against public dialogue state as well as complex-intent language; it does not
directly see a benchmark mechanism or private transition.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from taira_mt.benchmark.schemas import BenchmarkActionType
from taira_mt.models import StrictModel
from taira_mt.mt_agent.memory import PublicDialogueMemory
from taira_mt.mt_agent.state import (
    ComplexIntentShape,
    DecisionSufficientUserState,
    DialoguePhase,
    DisclosureStatus,
    RecommendationReadiness,
)
from taira_mt.provenance import content_sha256

THOUGHT_PATTERN_SKILL_SCHEMA_VERSION = "mt_state_conditioned_thought_skill_v1"
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9&'-]{1,}")


class SkillLayer(str, Enum):
    SEMANTIC = "semantic_grounding"
    INTERACTION = "interaction_control"


class QueryPolicy(str, Enum):
    SCENARIO_FUNCTIONAL = "scenario_functional"
    AUDIENCE_FIT = "audience_fit"
    MULTI_OCCASION = "multi_occasion"
    DECOMPOSE = "decompose"
    TAXONOMY_CLARIFY = "taxonomy_clarify"
    FEEDBACK_REPAIR = "feedback_repair"
    PRESERVE = "preserve"


class ThoughtTemplateStep(StrictModel):
    step_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    guard: str = Field(min_length=1)
    on_success: str = Field(min_length=1)
    on_failure: str = Field(min_length=1)


class SkillInsight(StrictModel):
    trigger: str = Field(min_length=1)
    principle: str = Field(min_length=1)
    intended_effect: str = Field(min_length=1)


class SkillApplicability(StrictModel):
    disclosure_statuses: list[DisclosureStatus] = Field(default_factory=list)
    dialogue_phases: list[DialoguePhase] = Field(default_factory=list)
    readiness: list[RecommendationReadiness] = Field(default_factory=list)
    intent_shapes: list[ComplexIntentShape] = Field(default_factory=list)
    public_cues: list[str] = Field(default_factory=list)

    @field_validator("public_cues")
    @classmethod
    def unique_public_cues(cls, value: list[str]) -> list[str]:
        normalized = [" ".join(item.split()) for item in value if item.strip()]
        if len(normalized) != len(set(item.casefold() for item in normalized)):
            raise ValueError("Skill public cues must be unique")
        return normalized


class ThoughtPatternSkill(StrictModel):
    """One frozen, auditable TAIRA Thought Pattern represented as an Agent Skill."""

    schema_version: str = THOUGHT_PATTERN_SKILL_SCHEMA_VERSION
    pattern_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    layer: SkillLayer
    strategy_family: str = Field(min_length=1)
    task_description: str = Field(min_length=1)
    state_description: str = Field(min_length=1)
    solution_description: str = Field(min_length=1)
    applicability: SkillApplicability
    thought_template: list[ThoughtTemplateStep] = Field(min_length=2)
    insights: list[SkillInsight] = Field(min_length=1)
    query_policy: QueryPolicy
    action_types: list[BenchmarkActionType] = Field(default_factory=list)
    provenance: str = Field(min_length=1)

    @field_validator("action_types")
    @classmethod
    def unique_action_types(cls, value: list[BenchmarkActionType]) -> list[BenchmarkActionType]:
        if len(value) != len(set(value)):
            raise ValueError("Skill action types must be unique")
        return value

    @model_validator(mode="after")
    def validate_layer_contract(self) -> ThoughtPatternSkill:
        if self.layer == SkillLayer.INTERACTION and not self.action_types:
            raise ValueError("interaction Skills must emit at least one typed action")
        if self.layer == SkillLayer.SEMANTIC and self.action_types:
            raise ValueError("semantic grounding Skills cannot directly emit an action")
        return self


class SkillMatch(StrictModel):
    """Transparent retrieval evidence for one Pattern candidate."""

    pattern_id: str
    score: float
    state_score: float
    lexical_score: float
    matched_state_features: list[str] = Field(default_factory=list)
    matched_public_cues: list[str] = Field(default_factory=list)


class SkillRetrieval(StrictModel):
    library_id: str
    state_key: str
    semantic_matches: list[SkillMatch]
    interaction_matches: list[SkillMatch]


def _token_set(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.casefold()))


def _cue_matches(dialogue: str, cues: list[str]) -> list[str]:
    folded = dialogue.casefold()
    return [cue for cue in cues if cue.casefold() in folded]


class ThoughtPatternLibrary:
    """Frozen Skill library with deterministic public state/query retrieval."""

    def __init__(self, skills: list[ThoughtPatternSkill], *, source: str) -> None:
        if not skills:
            raise ValueError("Thought Pattern library cannot be empty")
        ids = [skill.pattern_id for skill in skills]
        if len(ids) != len(set(ids)):
            raise ValueError("Thought Pattern IDs must be unique")
        self.skills = tuple(skills)
        self.source = source
        self.library_id = f"sctp::{content_sha256([skill.model_dump(mode='json') for skill in self.skills])[:16]}"
        self._by_id = {skill.pattern_id: skill for skill in self.skills}

    @classmethod
    def load(cls, path: Path) -> ThoughtPatternLibrary:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != THOUGHT_PATTERN_SKILL_SCHEMA_VERSION:
            raise ValueError(f"unsupported Thought Pattern Skill library: {path}")
        raw_skills = payload.get("skills")
        if not isinstance(raw_skills, list):
            raise ValueError("Thought Pattern Skill library has no skills list")
        return cls([ThoughtPatternSkill.model_validate(item) for item in raw_skills], source=str(path))

    @classmethod
    def default(cls) -> ThoughtPatternLibrary:
        return cls.load(Path(__file__).with_name("thought_pattern_skills_v1.json"))

    def skill(self, pattern_id: str) -> ThoughtPatternSkill:
        try:
            return self._by_id[pattern_id]
        except KeyError as exc:
            raise KeyError(f"unknown Thought Pattern Skill: {pattern_id}") from exc

    @staticmethod
    def _state_match(skill: ThoughtPatternSkill, state: DecisionSufficientUserState) -> tuple[float, list[str]]:
        applicability = skill.applicability
        score = 0.0
        matched: list[str] = []
        if applicability.disclosure_statuses:
            if state.disclosure_status in applicability.disclosure_statuses:
                score += 2.0
                matched.append(f"disclosure={state.disclosure_status.value}")
            else:
                score -= 1.4
        if applicability.dialogue_phases:
            if state.dialogue_phase in applicability.dialogue_phases:
                score += 1.5
                matched.append(f"phase={state.dialogue_phase.value}")
            else:
                score -= 0.8
        if applicability.readiness:
            if state.recommendation_readiness in applicability.readiness:
                score += 1.8
                matched.append(f"readiness={state.recommendation_readiness.value}")
            else:
                score -= 1.0
        if applicability.intent_shapes:
            overlap = set(applicability.intent_shapes).intersection(state.intent_shapes)
            if overlap:
                # A broad fallback Skill must not outrank a more specific
                # repair/ambiguity Skill merely because it lists many generic
                # shapes.  Shape evidence is one applicability factor, not a
                # reward for enumerating aliases.
                score += 0.8 * len(overlap) / len(applicability.intent_shapes)
                matched.extend(f"shape={item.value}" for item in sorted(overlap, key=lambda item: item.value))
        return score, matched

    @staticmethod
    def _lexical_match(skill: ThoughtPatternSkill, memory: PublicDialogueMemory) -> tuple[float, list[str]]:
        dialogue = memory.serialized_dialogue
        cues = _cue_matches(dialogue, skill.applicability.public_cues)
        query_tokens = _token_set(dialogue)
        skill_tokens = _token_set(
            " ".join((skill.task_description, skill.state_description, skill.solution_description))
        )
        overlap = len(query_tokens.intersection(skill_tokens))
        normalized_overlap = overlap / max(1, min(len(query_tokens), len(skill_tokens)))
        return (0.45 * len(cues) + normalized_overlap), cues

    def retrieve(
        self,
        *,
        memory: PublicDialogueMemory,
        state: DecisionSufficientUserState,
        limit_per_layer: int = 3,
    ) -> SkillRetrieval:
        if limit_per_layer < 1:
            raise ValueError("limit_per_layer must be positive")
        ranked: dict[SkillLayer, list[SkillMatch]] = defaultdict(list)
        for skill in self.skills:
            state_score, state_features = self._state_match(skill, state)
            lexical_score, cues = self._lexical_match(skill, memory)
            ranked[skill.layer].append(
                SkillMatch(
                    pattern_id=skill.pattern_id,
                    score=state_score + lexical_score,
                    state_score=state_score,
                    lexical_score=lexical_score,
                    matched_state_features=state_features,
                    matched_public_cues=cues,
                )
            )
        for layer in ranked:
            ranked[layer].sort(key=lambda item: (-item.score, item.pattern_id))
        return SkillRetrieval(
            library_id=self.library_id,
            state_key=state.state_key,
            semantic_matches=ranked[SkillLayer.SEMANTIC][:limit_per_layer],
            interaction_matches=ranked[SkillLayer.INTERACTION][:limit_per_layer],
        )
