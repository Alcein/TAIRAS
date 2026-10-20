"""Research-facing contracts for executable TAIRAS Thought Patterns."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from taira_mt.benchmark.schemas import (
    BenchmarkActionType,
    BenchmarkUserAct,
    CandidateFeasibility,
    DialoguePhaseV1,
    IntentCompleteness,
    IntentStability,
)
from taira_mt.dsus.schemas import DSUSStateLabel, PatienceRisk
from taira_mt.models import StrictModel

PATTERN_LIBRARY_VERSION = "state_conditioned_pattern_skills_v2"
PATTERN_INDUCTION_VERSION = "bounded_success_failure_induction_v2"
PATTERN_EXECUTION_CONTRACT_VERSION = "pattern_execution_contract_v1"


class PatternStrategyFamily(str, Enum):
    VERIFIED_FULFILLMENT = "verified_fulfillment"
    PROGRESSIVE_ELICITATION = "progressive_elicitation"
    AMBIGUITY_RESOLUTION = "ambiguity_resolution"
    GOAL_REVISION_RECOVERY = "goal_revision_recovery"
    INFEASIBILITY_NEGOTIATION = "infeasibility_negotiation"
    GROUNDED_EXPLANATION = "grounded_explanation"
    REJECTION_REPAIR = "rejection_repair"
    DIALOGUE_REORIENTATION = "dialogue_reorientation"
    PATIENCE_PRESERVATION = "patience_preservation"


class PatternOperation(str, Enum):
    INSPECT_STATE = "inspect_state"
    INSPECT_CATALOG = "inspect_catalog"
    ASK = "ask"
    RECOMMEND = "recommend"
    EXPLAIN = "explain"
    PROPOSE_RELAX = "propose_relax"
    UPDATE_STATE = "update_state"
    INVALIDATE_STALE_PLAN = "invalidate_stale_plan"
    REMATCH_PATTERN = "rematch_pattern"
    STOP = "stop"


class PatternProvenance(str, Enum):
    LEGACY_MIGRATED_SUCCESS_DISTILLED = "legacy_migrated_success_distilled"
    SUCCESS_DISTILLED = "success_distilled"
    FAILURE_CORRECTED = "failure_corrected"


class PatternExecutionRole(str, Enum):
    """Whether a Skill emits the next action or constrains another Skill.

    Overlay Skills are still executable: their bounded adaptations can remove
    unsafe optional branches from a primary Skill.  They do not get to invent
    a separate action outside the primary Skill's typed SOP templates.
    """

    PRIMARY = "primary"
    OVERLAY = "overlay"


class PatternInsightAdaptationKind(str, Enum):
    """Small, versioned primitives available to executable insight guidance.

    These are deliberately *not* strategy-family names.  The runtime dispatches
    only on this closed primitive vocabulary; which primitive is legal in a
    state is supplied by the matched Skill's serialized contract.
    """

    RETAIN_ALTERNATIVE_QUESTIONS = "retain_alternative_questions"
    ALLOW_ONE_EARLY_VERIFIED_RECOMMENDATION = "allow_one_early_verified_recommendation"
    BLOCK_ADAPTATIONS = "block_adaptations"


class PatternQuestionTargetPolicy(StrictModel):
    """Public, deterministic binding for an ASK SOP template."""

    policy_id: str = Field(min_length=1)
    target_order: list[str] = Field(min_length=1)
    require_unobserved_slot: bool = False
    require_disclosed_slot: bool = False

    @model_validator(mode="after")
    def validate_target_policy(self) -> PatternQuestionTargetPolicy:
        if len(self.target_order) != len(set(self.target_order)):
            raise ValueError("question target policy must not repeat targets")
        if self.require_unobserved_slot and self.require_disclosed_slot:
            raise ValueError("question target policy cannot require both unobserved and disclosed slots")
        return self


class PatternCandidateEvidencePolicy(StrictModel):
    """A public retrieval contract for a typed RECOMMEND template.

    v1 intentionally permits only the conservative verified-only, unseen
    top-1 retrieval path.  It prevents an insight branch from silently gaining
    a weaker evidence rule than its SOP counterpart.
    """

    policy_id: str = Field(min_length=1)
    verified_only: bool = True
    exclude_rejected: bool = True
    maximum_items: int = Field(default=1, ge=1, le=1)
    ranking_rule: str = "public_catalog_default"

    @model_validator(mode="after")
    def validate_safe_evidence_policy(self) -> PatternCandidateEvidencePolicy:
        if not self.verified_only:
            raise ValueError("v1 Pattern contracts require verified-only recommendation evidence")
        if not self.exclude_rejected:
            raise ValueError("v1 Pattern contracts must exclude public rejections")
        if self.ranking_rule != "public_catalog_default":
            raise ValueError("v1 Pattern contracts support only the public catalog ranking rule")
        return self


class PatternSOPActionTemplate(StrictModel):
    """A typed action that an SOP step explicitly authorizes."""

    template_id: str = Field(min_length=1)
    sop_step_id: str = Field(min_length=1)
    action_type: BenchmarkActionType
    question_target_policy_id: str | None = None
    candidate_evidence_policy_id: str | None = None
    maximum_primary_options: int = Field(default=1, ge=1, le=5)

    @model_validator(mode="after")
    def validate_template_bindings(self) -> PatternSOPActionTemplate:
        if self.action_type == BenchmarkActionType.ASK:
            if not self.question_target_policy_id or self.candidate_evidence_policy_id is not None:
                raise ValueError("ASK template requires one question-target policy only")
        elif self.action_type == BenchmarkActionType.RECOMMEND:
            if not self.candidate_evidence_policy_id or self.question_target_policy_id is not None:
                raise ValueError("RECOMMEND template requires one evidence policy only")
        elif self.question_target_policy_id is not None or self.candidate_evidence_policy_id is not None:
            raise ValueError("only ASK/RECOMMEND templates may carry a binding policy")
        return self


class PatternFallbackContract(StrictModel):
    """Fail-closed fallback authorized by a Skill contract."""

    fallback_id: str = Field(min_length=1)
    action_type: BenchmarkActionType = BenchmarkActionType.STOP
    utterance: str = Field(min_length=1)
    global_safe_stop_authority: bool = False

    @model_validator(mode="after")
    def validate_fail_closed_fallback(self) -> PatternFallbackContract:
        if self.action_type != BenchmarkActionType.STOP:
            raise ValueError("v1 Pattern fallback contracts may only emit STOP")
        return self


class PatternInsightAdaptation(StrictModel):
    """An auditable, bounded operational use of one Pattern insight."""

    adaptation_id: str = Field(min_length=1)
    kind: PatternInsightAdaptationKind
    source_insight_id: str = Field(min_length=1)
    authorizing_sop_step_id: str = Field(min_length=1)
    enabled: bool = True
    question_template_id: str | None = None
    question_target_policy_id: str | None = None
    candidate_evidence_policy_id: str | None = None
    maximum_additional_options: int = Field(default=1, ge=1, le=4)
    blocked_adaptation_kinds: list[PatternInsightAdaptationKind] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_adaptation_bindings(self) -> PatternInsightAdaptation:
        if self.kind == PatternInsightAdaptationKind.RETAIN_ALTERNATIVE_QUESTIONS:
            if not self.question_template_id or not self.question_target_policy_id:
                raise ValueError("alternative-question adaptation needs a question template and target policy")
            if self.candidate_evidence_policy_id is not None or self.blocked_adaptation_kinds:
                raise ValueError("alternative-question adaptation has incompatible bindings")
        elif self.kind == PatternInsightAdaptationKind.ALLOW_ONE_EARLY_VERIFIED_RECOMMENDATION:
            if not self.candidate_evidence_policy_id:
                raise ValueError("early-recommend adaptation needs an evidence policy")
            if self.question_template_id is not None or self.question_target_policy_id is not None:
                raise ValueError("early-recommend adaptation cannot bind a question template")
            if self.blocked_adaptation_kinds:
                raise ValueError("early-recommend adaptation cannot block other adaptations")
        elif self.kind == PatternInsightAdaptationKind.BLOCK_ADAPTATIONS:
            if not self.blocked_adaptation_kinds:
                raise ValueError("adaptation blocker must name at least one blocked adaptation kind")
            if self.question_template_id is not None or self.question_target_policy_id is not None:
                raise ValueError("adaptation blocker cannot bind a question template")
            if self.candidate_evidence_policy_id is not None:
                raise ValueError("adaptation blocker cannot bind an evidence policy")
        return self


class PatternExecutionContract(StrictModel):
    """Machine-executable projection of a Thought Pattern's SOP and insight.

    It deliberately contains only public-state rules and typed action
    primitives.  It has no case, goal, reward, evaluator, or simulator field.
    """

    contract_version: str = PATTERN_EXECUTION_CONTRACT_VERSION
    contract_id: str = Field(min_length=1)
    role: PatternExecutionRole
    priority: int = Field(ge=0, le=1000)
    minimum_match_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    sop_action_templates: list[PatternSOPActionTemplate] = Field(default_factory=list)
    question_target_policies: list[PatternQuestionTargetPolicy] = Field(default_factory=list)
    candidate_evidence_policies: list[PatternCandidateEvidencePolicy] = Field(default_factory=list)
    fallback_contract: PatternFallbackContract
    insight_adaptations: list[PatternInsightAdaptation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contract_ids(self) -> PatternExecutionContract:
        for values, label in (
            (self.sop_action_templates, "SOP action template"),
            (self.question_target_policies, "question target policy"),
            (self.candidate_evidence_policies, "candidate evidence policy"),
            (self.insight_adaptations, "insight adaptation"),
        ):
            ids = [
                getattr(item, "template_id", None)
                or getattr(item, "policy_id", None)
                or getattr(item, "adaptation_id", None)
                for item in values
            ]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Pattern execution contract repeats a {label} ID")
        return self


class PatternApplicability(StrictModel):
    allowed_phases: list[DialoguePhaseV1] = Field(default_factory=list)
    allowed_completeness: list[IntentCompleteness] = Field(default_factory=list)
    allowed_stability: list[IntentStability] = Field(default_factory=list)
    allowed_feasibility: list[CandidateFeasibility] = Field(default_factory=list)
    allowed_last_user_acts: list[BenchmarkUserAct] = Field(default_factory=list)
    allowed_patience_risks: list[PatienceRisk] = Field(default_factory=list)
    requires_explanation_pending: bool | None = None

    def matches(self, label: DSUSStateLabel) -> bool:
        checks = (
            not self.allowed_phases or label.dialogue_phase in self.allowed_phases,
            not self.allowed_completeness
            or label.intent_completeness in self.allowed_completeness,
            not self.allowed_stability
            or label.intent_stability in self.allowed_stability,
            not self.allowed_feasibility
            or label.candidate_feasibility in self.allowed_feasibility,
            not self.allowed_last_user_acts
            or label.last_user_act in self.allowed_last_user_acts,
            not self.allowed_patience_risks
            or label.patience_risk in self.allowed_patience_risks,
            self.requires_explanation_pending is None
            or label.explanation_pending == self.requires_explanation_pending,
        )
        return all(checks)


class PatternSOPStep(StrictModel):
    step_id: str
    operation: PatternOperation
    objective: str
    guard: str
    binding_rule: str
    system_action_type: BenchmarkActionType | None = None
    next_step_on_success: str | None = None
    next_step_on_failure: str | None = None

    @model_validator(mode="after")
    def validate_action_operation(self) -> PatternSOPStep:
        operation_to_action = {
            PatternOperation.ASK: BenchmarkActionType.ASK,
            PatternOperation.RECOMMEND: BenchmarkActionType.RECOMMEND,
            PatternOperation.EXPLAIN: BenchmarkActionType.EXPLAIN,
            PatternOperation.PROPOSE_RELAX: BenchmarkActionType.PROPOSE_RELAX,
            PatternOperation.STOP: BenchmarkActionType.STOP,
        }
        expected = operation_to_action.get(self.operation)
        if expected is None and self.system_action_type is not None:
            raise ValueError("non-action SOP step cannot declare a system action")
        if expected is not None and self.system_action_type != expected:
            raise ValueError("SOP action operation and system action type differ")
        return self


class PatternInsight(StrictModel):
    insight_id: str
    trigger: str
    principle: str
    intended_effect: str
    evidence_signature: str = Field(min_length=64, max_length=64)
    support_count: int = Field(ge=1)
    source: str


class PatternExecutionSignature(StrictModel):
    action_sequence: list[BenchmarkActionType] = Field(min_length=1)
    support_count: int = Field(ge=1)
    source: str


class PatternSkill(StrictModel):
    schema_version: str = PATTERN_LIBRARY_VERSION
    pattern_id: str
    version: str = "1.0.0"
    name: str
    strategy_family: PatternStrategyFamily
    task_description: str
    state_description: str
    solution_description: str
    applicability: PatternApplicability
    sop_steps: list[PatternSOPStep] = Field(min_length=2)
    insights: list[PatternInsight] = Field(min_length=1)
    forbidden_operations: list[str] = Field(min_length=1)
    fallback_policy: str
    # v1 artifacts intentionally remain loadable for historical debugging, but
    # the online v4 compiler refuses to execute a Skill without this contract.
    # Keeping the compatibility boundary in the schema lets the loader inspect
    # legacy artifacts without making them eligible for a paper result.
    execution_contract: PatternExecutionContract | None = None
    execution_signatures: list[PatternExecutionSignature] = Field(min_length=1)
    legacy_seed_refs: list[str] = Field(default_factory=list)
    provenance: PatternProvenance
    supporting_expert_episode_count: int = Field(ge=0)
    supporting_correction_state_count: int = Field(ge=0)
    covered_state_count: int = Field(ge=1)
    covered_archetype_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_sop_graph(self) -> PatternSkill:
        step_ids = [item.step_id for item in self.sop_steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Pattern SOP step IDs must be unique")
        known = set(step_ids)
        for step in self.sop_steps:
            for target in (step.next_step_on_success, step.next_step_on_failure):
                if target is not None and target not in known:
                    raise ValueError("Pattern SOP transition references an unknown step")
        if len({item.insight_id for item in self.insights}) != len(self.insights):
            raise ValueError("Pattern insight IDs must be unique")
        contract = self.execution_contract
        if contract is not None:
            steps = {item.step_id: item for item in self.sop_steps}
            question_policies = {item.policy_id for item in contract.question_target_policies}
            evidence_policies = {item.policy_id for item in contract.candidate_evidence_policies}
            templates = {item.template_id: item for item in contract.sop_action_templates}
            insight_ids = {item.insight_id for item in self.insights}
            for template in contract.sop_action_templates:
                step = steps.get(template.sop_step_id)
                if step is None:
                    raise ValueError("Pattern action template references an unknown SOP step")
                if step.system_action_type != template.action_type:
                    raise ValueError("Pattern action template type differs from its SOP action step")
                if (
                    template.question_target_policy_id is not None
                    and template.question_target_policy_id not in question_policies
                ):
                    raise ValueError("Pattern action template references an unknown question policy")
                if (
                    template.candidate_evidence_policy_id is not None
                    and template.candidate_evidence_policy_id not in evidence_policies
                ):
                    raise ValueError("Pattern action template references an unknown evidence policy")
            for adaptation in contract.insight_adaptations:
                if adaptation.source_insight_id not in insight_ids:
                    raise ValueError("Pattern insight adaptation references an unknown insight")
                if adaptation.authorizing_sop_step_id not in steps:
                    raise ValueError("Pattern insight adaptation references an unknown SOP step")
                if (
                    adaptation.question_template_id is not None
                    and adaptation.question_template_id not in templates
                ):
                    raise ValueError("Pattern insight adaptation references an unknown question template")
                if (
                    adaptation.question_target_policy_id is not None
                    and adaptation.question_target_policy_id not in question_policies
                ):
                    raise ValueError("Pattern insight adaptation references an unknown question policy")
                if (
                    adaptation.candidate_evidence_policy_id is not None
                    and adaptation.candidate_evidence_policy_id not in evidence_policies
                ):
                    raise ValueError("Pattern insight adaptation references an unknown evidence policy")
        return self


class StateConditionedPatternLibrary(StrictModel):
    schema_version: str = PATTERN_LIBRARY_VERSION
    induction_version: str = PATTERN_INDUCTION_VERSION
    library_id: str
    parent_dsus_build_id: str
    patterns: list[PatternSkill] = Field(min_length=1)


class ArchetypePatternCoverage(StrictModel):
    state_archetype_id: str = Field(min_length=64, max_length=64)
    state_count: int = Field(ge=1)
    pattern_ids: list[str] = Field(min_length=1)
    decision_obligation_counts: dict[str, int]


class PatternInductionSummary(StrictModel):
    induction_id: str
    parent_dsus_build_id: str
    pattern_library_id: str
    pattern_count: int = Field(ge=1)
    strategy_family_counts: dict[str, int]
    sop_step_count: int = Field(ge=1)
    insight_count: int = Field(ge=1)
    correction_state_count: int = Field(ge=1)
    state_count: int = Field(ge=1)
    archetype_count: int = Field(ge=1)
    covered_state_count: int = Field(ge=1)
    covered_archetype_count: int = Field(ge=1)
    uncovered_state_count: int = Field(ge=0)
    uncovered_archetype_count: int = Field(ge=0)
    mean_patterns_per_state: float = Field(gt=0.0)
    mean_patterns_per_archetype: float = Field(gt=0.0)
    multi_pattern_state_rate: float = Field(ge=0.0, le=1.0)
    expert_execution_signature_counts: dict[str, int]
    pattern_support: dict[str, dict[str, int]]
    all_patterns_have_sop_and_insights: bool
    all_states_covered: bool
    all_archetypes_covered: bool


class PatternArtifactFile(StrictModel):
    relative_path: str
    sha256: str = Field(min_length=64, max_length=64)
    byte_size: int = Field(ge=0)
    line_count: int | None = Field(default=None, ge=0)


class PatternInductionManifest(StrictModel):
    schema_version: str = PATTERN_LIBRARY_VERSION
    artifact_role: str = "state_conditioned_sop_insight_pattern_library"
    induction_id: str
    library_id: str
    parent_dsus_build_id: str
    parent_dsus_manifest_sha256: str = Field(min_length=64, max_length=64)
    library_sha256: str = Field(min_length=64, max_length=64)
    coverage_sha256: str = Field(min_length=64, max_length=64)
    summary_sha256: str = Field(min_length=64, max_length=64)
    source_tree_sha256: str = Field(min_length=64, max_length=64)
    outputs: list[PatternArtifactFile]
