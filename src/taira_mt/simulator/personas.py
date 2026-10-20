"""Versioned persona templates for non-degenerate transition supervision."""

from __future__ import annotations

from enum import Enum

from taira_mt.models import StrictModel, UserPersona

PERSONA_SCHEDULE_VERSION = "persona_schedule_v1"


class PersonaProfile(str, Enum):
    FLEXIBLE_PATIENT = "flexible_patient"
    NORMAL_COOPERATIVE = "normal_cooperative"
    SELECTIVE_EXACTING = "selective_exacting"
    GUARDED_IMPATIENT = "guarded_impatient"


class PersonaTemplate(StrictModel):
    profile: PersonaProfile
    patience_turns: int
    cooperativeness: float
    acceptance_threshold: float
    question_cost: float
    misstatement_probability: float
    self_correction_probability: float

    def instantiate(self, case_key: str) -> UserPersona:
        return UserPersona(
            persona_id=f"{self.profile.value}-{case_key}",
            patience_turns=self.patience_turns,
            cooperativeness=self.cooperativeness,
            acceptance_threshold=self.acceptance_threshold,
            question_cost=self.question_cost,
            misstatement_probability=self.misstatement_probability,
            self_correction_probability=self.self_correction_probability,
        )


PERSONA_TEMPLATES = (
    PersonaTemplate(
        profile=PersonaProfile.FLEXIBLE_PATIENT,
        patience_turns=7,
        cooperativeness=0.98,
        acceptance_threshold=0.72,
        question_cost=0.03,
        misstatement_probability=0.02,
        self_correction_probability=0.95,
    ),
    PersonaTemplate(
        profile=PersonaProfile.NORMAL_COOPERATIVE,
        patience_turns=6,
        cooperativeness=0.90,
        acceptance_threshold=0.80,
        question_cost=0.04,
        misstatement_probability=0.05,
        self_correction_probability=0.90,
    ),
    PersonaTemplate(
        profile=PersonaProfile.SELECTIVE_EXACTING,
        patience_turns=5,
        cooperativeness=0.82,
        acceptance_threshold=0.88,
        question_cost=0.06,
        misstatement_probability=0.08,
        self_correction_probability=0.85,
    ),
    PersonaTemplate(
        profile=PersonaProfile.GUARDED_IMPATIENT,
        patience_turns=3,
        cooperativeness=0.62,
        acceptance_threshold=0.80,
        question_cost=0.09,
        misstatement_probability=0.12,
        self_correction_probability=0.75,
    ),
)


def persona_variants(case_key: str) -> tuple[UserPersona, ...]:
    """Return all versioned variants; dataset selection chooses them publicly."""

    return tuple(template.instantiate(case_key) for template in PERSONA_TEMPLATES)
