"""Schema for the multi-turn evolution of TAIRA Thought Patterns."""

from __future__ import annotations

from pydantic import Field, model_validator

from taira_mt.models import DialoguePhase, StrictModel, SystemActionType, UserAct


class PatternOutcome(StrictModel):
    """Executable transition/termination rule for an observed user act."""

    user_act: UserAct
    next_phase: DialoguePhase | None = None
    terminal: bool = False


class StateAwareThoughtPattern(StrictModel):
    """A reusable strategy grounded in the current belief state.

    Unlike the legacy query-level templates, a pattern declares when it applies,
    what information it needs, which action it proposes, and when it should stop.
    """

    pattern_id: str
    version: str = "1.0.0"
    name: str
    task_description: str
    state_description: str
    solution_description: str
    thought_template: list[str] = Field(min_length=1)
    applicable_phases: list[DialoguePhase]
    preferred_action: SystemActionType
    information_targets: list[str] = Field(default_factory=list)
    requires_missing_slot: bool = False
    requires_no_candidates: bool = False
    required_feedback: list[UserAct] = Field(default_factory=list)
    min_readiness: float = Field(default=0.0, ge=0.0, le=1.0)
    max_readiness: float = Field(default=1.0, ge=0.0, le=1.0)
    priority: int = 0
    action_template: str
    outcome_rules: list[PatternOutcome]
    fallback_outcome: PatternOutcome
    expected_utility: float = Field(default=0.0, ge=-1.0, le=1.0)
    provenance: str = "expert_seed"

    @model_validator(mode="after")
    def validate_pattern(self) -> StateAwareThoughtPattern:
        if self.min_readiness > self.max_readiness:
            raise ValueError("min_readiness cannot exceed max_readiness")
        if not self.outcome_rules:
            raise ValueError("at least one executable outcome rule is required")
        acts = [rule.user_act for rule in self.outcome_rules]
        if len(acts) != len(set(acts)):
            raise ValueError("outcome rules must have unique user acts")
        for rule in self.outcome_rules:
            if rule.terminal != (rule.next_phase == DialoguePhase.TERMINAL):
                raise ValueError("terminal outcome rules must transition to terminal and vice versa")
        if self.fallback_outcome.user_act != UserAct.OTHER:
            raise ValueError("fallback_outcome must use UserAct.OTHER")
        return self
