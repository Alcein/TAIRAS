# ruff: noqa: E501
"""Bounded success-distillation and failure-correction Pattern induction."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taira_mt.benchmark.schemas import (
    BenchmarkActionType,
    BenchmarkUserAct,
    CandidateFeasibility,
    IntentCompleteness,
    IntentStability,
)
from taira_mt.dsus import LoadedDSUSDevelopmentData, load_dsus_development_artifact
from taira_mt.dsus.schemas import (
    DecisionObligation,
    DSUSBehaviorPolicy,
    DSUSStateLabel,
    PatienceRisk,
)
from taira_mt.pattern_planner.schemas import (
    PATTERN_INDUCTION_VERSION,
    ArchetypePatternCoverage,
    PatternApplicability,
    PatternArtifactFile,
    PatternCandidateEvidencePolicy,
    PatternExecutionContract,
    PatternExecutionRole,
    PatternExecutionSignature,
    PatternFallbackContract,
    PatternInductionManifest,
    PatternInductionSummary,
    PatternInsight,
    PatternInsightAdaptation,
    PatternInsightAdaptationKind,
    PatternOperation,
    PatternProvenance,
    PatternQuestionTargetPolicy,
    PatternSkill,
    PatternSOPActionTemplate,
    PatternSOPStep,
    PatternStrategyFamily,
    StateConditionedPatternLibrary,
)
from taira_mt.provenance import canonical_json, content_sha256, source_tree_sha256


@dataclass(frozen=True)
class _PatternBlueprint:
    slug: str
    name: str
    family: PatternStrategyFamily
    task_description: str
    state_description: str
    solution_description: str
    applicability: PatternApplicability
    sop_steps: tuple[PatternSOPStep, ...]
    base_insight: tuple[str, str, str]
    forbidden_operations: tuple[str, ...]
    fallback_policy: str
    execution_contract: PatternExecutionContract
    provenance: PatternProvenance
    legacy_seed_refs: tuple[str, ...] = ()
    source_scenario: str | None = None


@dataclass(frozen=True)
class PatternInductionResult:
    induction_id: str
    parent_dsus_manifest_sha256: str
    library: StateConditionedPatternLibrary
    coverage: tuple[ArchetypePatternCoverage, ...]
    summary: PatternInductionSummary


def _step(
    step_id: str,
    operation: PatternOperation,
    objective: str,
    guard: str,
    binding_rule: str,
    *,
    success: str | None = None,
    failure: str | None = None,
) -> PatternSOPStep:
    action_types = {
        PatternOperation.ASK: BenchmarkActionType.ASK,
        PatternOperation.RECOMMEND: BenchmarkActionType.RECOMMEND,
        PatternOperation.EXPLAIN: BenchmarkActionType.EXPLAIN,
        PatternOperation.PROPOSE_RELAX: BenchmarkActionType.PROPOSE_RELAX,
        PatternOperation.STOP: BenchmarkActionType.STOP,
    }
    return PatternSOPStep(
        step_id=step_id,
        operation=operation,
        objective=objective,
        guard=guard,
        binding_rule=binding_rule,
        system_action_type=action_types.get(operation),
        next_step_on_success=success,
        next_step_on_failure=failure,
    )


def _base_insight_id(slug: str) -> str:
    """Return the stable base-insight identifier emitted by ``_base_insight``."""

    return f"insight-{slug}-base"


def _execution_contract(
    *,
    slug: str,
    role: PatternExecutionRole,
    priority: int,
    fallback_utterance: str,
    action_templates: tuple[PatternSOPActionTemplate, ...] = (),
    question_policies: tuple[PatternQuestionTargetPolicy, ...] = (),
    evidence_policies: tuple[PatternCandidateEvidencePolicy, ...] = (),
    insight_adaptations: tuple[PatternInsightAdaptation, ...] = (),
    global_safe_stop_authority: bool = False,
) -> PatternExecutionContract:
    """Build one explicit, public execution contract for a Pattern blueprint.

    The helper intentionally performs no strategy-family dispatch.  Each
    blueprint below declares its actual action templates, target ordering,
    evidence policy, fallback, and bounded insight adaptations in the frozen
    artifact consumed by the online compiler.
    """

    return PatternExecutionContract(
        contract_id=f"contract-{slug}-v1",
        role=role,
        priority=priority,
        sop_action_templates=list(action_templates),
        question_target_policies=list(question_policies),
        candidate_evidence_policies=list(evidence_policies),
        fallback_contract=PatternFallbackContract(
            fallback_id=f"fallback-{slug}-stop-v1",
            utterance=fallback_utterance,
            global_safe_stop_authority=global_safe_stop_authority,
        ),
        insight_adaptations=list(insight_adaptations),
    )


def _ask_template(
    *,
    template_id: str,
    sop_step_id: str,
    policy_id: str,
) -> PatternSOPActionTemplate:
    return PatternSOPActionTemplate(
        template_id=template_id,
        sop_step_id=sop_step_id,
        action_type=BenchmarkActionType.ASK,
        question_target_policy_id=policy_id,
    )


def _recommend_template(
    *,
    template_id: str,
    sop_step_id: str,
    policy_id: str,
) -> PatternSOPActionTemplate:
    return PatternSOPActionTemplate(
        template_id=template_id,
        sop_step_id=sop_step_id,
        action_type=BenchmarkActionType.RECOMMEND,
        candidate_evidence_policy_id=policy_id,
    )


def _explain_template(*, template_id: str, sop_step_id: str) -> PatternSOPActionTemplate:
    return PatternSOPActionTemplate(
        template_id=template_id,
        sop_step_id=sop_step_id,
        action_type=BenchmarkActionType.EXPLAIN,
    )


def _question_policy(
    *,
    policy_id: str,
    targets: tuple[str, ...],
    require_unobserved_slot: bool = False,
    require_disclosed_slot: bool = False,
) -> PatternQuestionTargetPolicy:
    return PatternQuestionTargetPolicy(
        policy_id=policy_id,
        target_order=list(targets),
        require_unobserved_slot=require_unobserved_slot,
        require_disclosed_slot=require_disclosed_slot,
    )


def _verified_evidence_policy(*, policy_id: str) -> PatternCandidateEvidencePolicy:
    return PatternCandidateEvidencePolicy(policy_id=policy_id)


def _blueprints() -> tuple[_PatternBlueprint, ...]:
    return (
        _PatternBlueprint(
            slug="verified_intent_fulfillment",
            name="Verified-Intent Direct Fulfillment",
            family=PatternStrategyFamily.VERIFIED_FULFILLMENT,
            task_description=(
                "Fulfill a sufficiently specified single-target request without paying for an unnecessary dialogue turn."
            ),
            state_description=(
                "Intent is sufficient, the catalog has verifiable candidates, and no explanation obligation is pending."
            ),
            solution_description=(
                "Treat known hard preferences as executable filters, rank only verifiable unseen items, and recommend promptly."
            ),
            applicability=PatternApplicability(
                allowed_completeness=[IntentCompleteness.SUFFICIENT],
                allowed_feasibility=[
                    CandidateFeasibility.BROAD,
                    CandidateFeasibility.NARROW,
                ],
                requires_explanation_pending=False,
            ),
            sop_steps=(
                _step(
                    "inspect",
                    PatternOperation.INSPECT_STATE,
                    "Confirm that every active hard preference is public.",
                    "Intent is sufficient and no response obligation is pending.",
                    "Read DSUS completeness plus the public predicate store.",
                    success="retrieve",
                    failure="rematch",
                ),
                _step(
                    "retrieve",
                    PatternOperation.INSPECT_CATALOG,
                    "Form a verifiable candidate slate.",
                    "At least one unseen candidate satisfies every public hard predicate.",
                    "Filter UNKNOWN and violated hard evidence, then rank verified unseen items.",
                    success="recommend",
                    failure="rematch",
                ),
                _step(
                    "recommend",
                    PatternOperation.RECOMMEND,
                    "Deliver the shortest safe recommendation.",
                    "The verified slate is non-empty.",
                    "Recommend one or two highest-ranked item IDs with concise public evidence.",
                    success="rematch",
                    failure="rematch",
                ),
                _step(
                    "rematch",
                    PatternOperation.REMATCH_PATTERN,
                    "Route the observed feedback to the next state-conditioned Skill.",
                    "A user response has updated the public state.",
                    "Retain, switch, or rematch from the updated DSUS posterior.",
                ),
            ),
            base_insight=(
                "sufficient_intent",
                "Once all active hard preferences are known, another generic question spends patience without reducing decision risk.",
                "Reduce unnecessary turns while preserving hard-constraint safety.",
            ),
            forbidden_operations=(
                "Do not ask a generic question after sufficient intent is established.",
                "Do not recommend an item with UNKNOWN evidence for a public hard preference.",
            ),
            fallback_policy="rematch from the updated DSUS state; negotiate if the verified set becomes empty",
            execution_contract=_execution_contract(
                slug="verified_intent_fulfillment",
                role=PatternExecutionRole.PRIMARY,
                priority=60,
                fallback_utterance=(
                    "I cannot safely identify a publicly verified option from the current requirements."
                ),
                action_templates=(
                    _recommend_template(
                        template_id="recommend_verified",
                        sop_step_id="recommend",
                        policy_id="verified_unseen_top1",
                    ),
                ),
                evidence_policies=(
                    _verified_evidence_policy(policy_id="verified_unseen_top1"),
                ),
            ),
            provenance=PatternProvenance.LEGACY_MIGRATED_SUCCESS_DISTILLED,
            legacy_seed_refs=("template_0", "template_1", "template_8", "template_11"),
            source_scenario="direct",
        ),
        _PatternBlueprint(
            slug="progressive_constraint_elicitation",
            name="Progressive Decisive-Constraint Elicitation",
            family=PatternStrategyFamily.PROGRESSIVE_ELICITATION,
            task_description=(
                "Recover the hidden hard preferences needed to make a safe recommendation with as few questions as possible."
            ),
            state_description="Intent is incomplete because at least one active hard preference remains undisclosed.",
            solution_description=(
                "Ask one unrepeated question with the highest expected candidate reduction, update the public store, and stop eliciting as soon as intent is sufficient."
            ),
            applicability=PatternApplicability(
                allowed_completeness=[IntentCompleteness.INCOMPLETE],
                requires_explanation_pending=False,
            ),
            sop_steps=(
                _step(
                    "inspect_missing",
                    PatternOperation.INSPECT_STATE,
                    "Identify unresolved hard-preference roles without guessing their values.",
                    "DSUS marks intent incomplete.",
                    "Compare ontology slots, asked targets, and public candidate uncertainty.",
                    success="ask_decisive",
                    failure="rematch",
                ),
                _step(
                    "ask_decisive",
                    PatternOperation.ASK,
                    "Reveal exactly one decision-relevant preference.",
                    "The target has not already been asked and can change the viable set.",
                    "Choose the unrevealed hard slot with greatest verified candidate reduction.",
                    success="update",
                    failure="rematch",
                ),
                _step(
                    "update",
                    PatternOperation.UPDATE_STATE,
                    "Commit the authorized answer and recompute viability.",
                    "The response contains an authorized public preference or no-preference signal.",
                    "Update the predicate store, DSUS posterior, and candidate summary once.",
                    success="rematch",
                    failure="rematch",
                ),
                _step(
                    "rematch",
                    PatternOperation.REMATCH_PATTERN,
                    "Continue elicitation only if incompleteness remains.",
                    "The public state was recomputed.",
                    "Retain this Skill if incomplete; otherwise switch to fulfillment or negotiation.",
                ),
            ),
            base_insight=(
                "one_decisive_question",
                "Progressive disclosure is solved by information value, not by enumerating every possible preference slot.",
                "Increase informative-question rate and protect patience.",
            ),
            forbidden_operations=(
                "Do not repeat an already answered question.",
                "Do not recommend while an undisclosed hard preference can invalidate the slate.",
            ),
            fallback_policy="use the shortest verifiable recommendation path when patience risk becomes high",
            execution_contract=_execution_contract(
                slug="progressive_constraint_elicitation",
                role=PatternExecutionRole.PRIMARY,
                priority=40,
                fallback_utterance=(
                    "I need one more public preference before I can make a verified recommendation."
                ),
                action_templates=(
                    _ask_template(
                        template_id="ask_decisive",
                        sop_step_id="ask_decisive",
                        policy_id="decisive_unobserved_order",
                    ),
                ),
                question_policies=(
                    _question_policy(
                        policy_id="decisive_unobserved_order",
                        targets=("specific_category", "price", "rating", "popularity_percentile"),
                        require_unobserved_slot=True,
                    ),
                ),
                evidence_policies=(
                    _verified_evidence_policy(policy_id="early_verified_unseen_top1"),
                ),
                insight_adaptations=(
                    PatternInsightAdaptation(
                        adaptation_id="retain_alternative_decisive_questions",
                        kind=PatternInsightAdaptationKind.RETAIN_ALTERNATIVE_QUESTIONS,
                        source_insight_id=_base_insight_id("progressive_constraint_elicitation"),
                        authorizing_sop_step_id="ask_decisive",
                        question_template_id="ask_decisive",
                        question_target_policy_id="decisive_unobserved_order",
                        maximum_additional_options=3,
                    ),
                    PatternInsightAdaptation(
                        adaptation_id="offer_one_early_verified_alternative",
                        kind=PatternInsightAdaptationKind.ALLOW_ONE_EARLY_VERIFIED_RECOMMENDATION,
                        source_insight_id=_base_insight_id("progressive_constraint_elicitation"),
                        authorizing_sop_step_id="ask_decisive",
                        candidate_evidence_policy_id="early_verified_unseen_top1",
                    ),
                ),
            ),
            provenance=PatternProvenance.SUCCESS_DISTILLED,
            source_scenario="progressive",
        ),
        _PatternBlueprint(
            slug="hierarchical_ambiguity_resolution",
            name="Hierarchical Occasion-to-Preference Resolution",
            family=PatternStrategyFamily.AMBIGUITY_RESOLUTION,
            task_description=(
                "Resolve which catalog interpretation an underspecified shopping occasion refers to before eliciting lower-level preferences."
            ),
            state_description="The public need admits multiple catalog-level interpretations and DSUS marks intent ambiguous.",
            solution_description=(
                "Resolve the high-level interpretation first, then ask only the remaining decisive preference and reuse direct catalog fulfillment."
            ),
            applicability=PatternApplicability(
                allowed_completeness=[IntentCompleteness.AMBIGUOUS],
                requires_explanation_pending=False,
            ),
            sop_steps=(
                _step(
                    "enumerate_interpretations",
                    PatternOperation.INSPECT_STATE,
                    "Form a public, generic occasion-or-subtype clarification.",
                    "DSUS ambiguity is unresolved.",
                    "Use only the public domain ontology and the user's wording; do not enumerate or infer a private registered option.",
                    success="ask_interpretation",
                    failure="rematch",
                ),
                _step(
                    "ask_interpretation",
                    PatternOperation.ASK,
                    "Resolve the top-level meaning before item retrieval.",
                    "More than one catalog interpretation remains plausible.",
                    "Ask one generic occasion or subtype clarification, not a low-level numeric question or a private-option menu.",
                    success="update",
                    failure="rematch",
                ),
                _step(
                    "update",
                    PatternOperation.UPDATE_STATE,
                    "Bind the selected interpretation and expose its relevant preference roles.",
                    "The user selected one authorized interpretation.",
                    "Update DSUS, then rematch progressive elicitation if a hard preference remains hidden.",
                    success="rematch",
                    failure="rematch",
                ),
                _step(
                    "rematch",
                    PatternOperation.REMATCH_PATTERN,
                    "Continue with the appropriate lower-level Skill.",
                    "The ambiguity flag has been recomputed.",
                    "Switch to progressive elicitation or verified fulfillment.",
                ),
            ),
            base_insight=(
                "resolve_high_level_first",
                "A low-level preference answer is not useful until the intended shopping interpretation is fixed.",
                "Prevent locally valid but globally irrelevant questions and recommendations.",
            ),
            forbidden_operations=(
                "Do not collapse multiple interpretations into one guessed category.",
                "Do not retrieve a final slate before the high-level ambiguity is resolved.",
            ),
            fallback_policy="ask one interpretation contrast or safely defer when no option can be grounded",
            execution_contract=_execution_contract(
                slug="hierarchical_ambiguity_resolution",
                role=PatternExecutionRole.PRIMARY,
                priority=30,
                fallback_utterance=(
                    "I cannot safely resolve the remaining interpretation from the public request."
                ),
                action_templates=(
                    _ask_template(
                        template_id="ask_interpretation",
                        sop_step_id="ask_interpretation",
                        policy_id="top_level_interpretation_first",
                    ),
                ),
                question_policies=(
                    _question_policy(
                        policy_id="top_level_interpretation_first",
                        targets=("occasion_interpretation",),
                    ),
                    _question_policy(
                        policy_id="post_interpretation_unobserved_order",
                        targets=("specific_category", "price", "rating", "popularity_percentile"),
                        require_unobserved_slot=True,
                    ),
                ),
                evidence_policies=(
                    _verified_evidence_policy(policy_id="early_verified_unseen_top1"),
                ),
                insight_adaptations=(
                    PatternInsightAdaptation(
                        adaptation_id="retain_post_interpretation_questions",
                        kind=PatternInsightAdaptationKind.RETAIN_ALTERNATIVE_QUESTIONS,
                        source_insight_id=_base_insight_id("hierarchical_ambiguity_resolution"),
                        authorizing_sop_step_id="ask_interpretation",
                        question_template_id="ask_interpretation",
                        question_target_policy_id="post_interpretation_unobserved_order",
                        maximum_additional_options=4,
                    ),
                    PatternInsightAdaptation(
                        adaptation_id="offer_one_early_verified_alternative",
                        kind=PatternInsightAdaptationKind.ALLOW_ONE_EARLY_VERIFIED_RECOMMENDATION,
                        source_insight_id=_base_insight_id("hierarchical_ambiguity_resolution"),
                        authorizing_sop_step_id="ask_interpretation",
                        candidate_evidence_policy_id="early_verified_unseen_top1",
                    ),
                ),
            ),
            provenance=PatternProvenance.LEGACY_MIGRATED_SUCCESS_DISTILLED,
            legacy_seed_refs=("template_4",),
            source_scenario="ambiguous",
        ),
        _PatternBlueprint(
            slug="feedback_triggered_goal_revision",
            name="Feedback-Triggered Goal Revision Recovery",
            family=PatternStrategyFamily.GOAL_REVISION_RECOVERY,
            task_description=(
                "Recover when recommendation feedback reveals that the active need has tightened or been corrected."
            ),
            state_description="DSUS detects corrected/tightened intent or a repair-phase clarification after a recommendation.",
            solution_description=(
                "Invalidate the stale candidate rationale, bind the corrected public preference, re-retrieve, and route any acceptance-contingent explanation to the evidence Skill."
            ),
            applicability=PatternApplicability(
                allowed_stability=[IntentStability.CORRECTED, IntentStability.TIGHTENED],
                requires_explanation_pending=False,
            ),
            sop_steps=(
                _step(
                    "attribute_feedback",
                    PatternOperation.INSPECT_STATE,
                    "Distinguish a true goal update from ordinary item rejection.",
                    "The latest public feedback changes an active preference.",
                    "Compare the previous public predicates with the newly authorized correction.",
                    success="invalidate",
                    failure="rematch",
                ),
                _step(
                    "invalidate",
                    PatternOperation.INVALIDATE_STALE_PLAN,
                    "Prevent reuse of a recommendation justified under the old goal.",
                    "At least one active predicate or goal version changed.",
                    "Clear stale candidate rankings and explanation evidence; retain only still-valid public facts.",
                    success="retrieve",
                    failure="rematch",
                ),
                _step(
                    "retrieve",
                    PatternOperation.INSPECT_CATALOG,
                    "Recompute candidates under the revised public goal.",
                    "The revised predicate store is consistent.",
                    "Filter and rank verifiable unseen items under the corrected constraints.",
                    success="recommend",
                    failure="rematch",
                ),
                _step(
                    "recommend",
                    PatternOperation.RECOMMEND,
                    "Present a recommendation that visibly responds to the correction.",
                    "A revised verified slate is available.",
                    "Recommend one or two revised items and retain their evidence for explanation.",
                    success="rematch",
                    failure="rematch",
                ),
                _step(
                    "rematch",
                    PatternOperation.REMATCH_PATTERN,
                    "Handle the next feedback obligation.",
                    "The user responded to the revised slate.",
                    "Switch to grounded explanation when requested; otherwise continue from DSUS.",
                ),
            ),
            base_insight=(
                "invalidate_stale_plan",
                "A corrected preference changes the justification as well as the candidate set; appending it to a stale plan is unsafe.",
                "Reduce repeated bad recommendations after goal revision.",
            ),
            forbidden_operations=(
                "Do not reuse the pre-revision slate without rechecking every active hard preference.",
                "Do not treat corrective feedback as a generic rejection.",
            ),
            fallback_policy="rematch negotiation if the revised verified set is empty",
            execution_contract=_execution_contract(
                slug="feedback_triggered_goal_revision",
                role=PatternExecutionRole.PRIMARY,
                priority=50,
                fallback_utterance=(
                    "I cannot safely recommend a revised option from the currently public requirements."
                ),
                action_templates=(
                    _recommend_template(
                        template_id="recommend_revised_verified",
                        sop_step_id="recommend",
                        policy_id="revised_verified_unseen_top1",
                    ),
                ),
                evidence_policies=(
                    _verified_evidence_policy(policy_id="revised_verified_unseen_top1"),
                ),
            ),
            provenance=PatternProvenance.SUCCESS_DISTILLED,
            source_scenario="revision",
        ),
        _PatternBlueprint(
            slug="joint_infeasibility_negotiation",
            name="Joint-Infeasibility Trade-off Negotiation",
            family=PatternStrategyFamily.INFEASIBILITY_NEGOTIATION,
            task_description=(
                "Recover a recommendation task whose individually plausible requirements are jointly infeasible."
            ),
            state_description="DSUS and the public catalog summary indicate an empty verified feasible set.",
            solution_description=(
                "Diagnose the joint conflict, ask which disclosed requirement can move and what boundary is acceptable, then re-enter verified fulfillment from the user's public answer."
            ),
            applicability=PatternApplicability(
                allowed_feasibility=[CandidateFeasibility.EMPTY],
                requires_explanation_pending=False,
            ),
            sop_steps=(
                _step(
                    "verify_conflict",
                    PatternOperation.INSPECT_CATALOG,
                    "Confirm joint infeasibility rather than blaming one plausible condition.",
                    "No item verifiably satisfies all active public hard predicates.",
                    "Inspect per-constraint survivor sets and UNKNOWN evidence separately.",
                    success="select_tradeoff_question",
                    failure="rematch",
                ),
                _step(
                    "select_tradeoff_question",
                    PatternOperation.INSPECT_STATE,
                    "Choose one public condition whose flexibility must be elicited.",
                    "A disclosed numeric hard preference has not yet been queried for flexibility.",
                    "Select a conflict-relevant public slot, but never infer the user's acceptable boundary.",
                    success="ask_boundary",
                    failure="rematch",
                ),
                _step(
                    "ask_boundary",
                    PatternOperation.ASK,
                    "Elicit an explicit, user-authored trade-off boundary.",
                    "The selected slot and its current requirement are already public.",
                    "Ask whether that requirement is flexible and let the response supply the new boundary.",
                    success="update",
                    failure="rematch",
                ),
                _step(
                    "update",
                    PatternOperation.UPDATE_STATE,
                    "Bind the public answer and recompute the verified candidate set.",
                    "The user explicitly states an acceptable relaxed requirement.",
                    "Mark intent relaxed, replace only the answered predicate, and rematch fulfillment.",
                    success="rematch",
                    failure="rematch",
                ),
                _step(
                    "rematch",
                    PatternOperation.REMATCH_PATTERN,
                    "Resume recommendation only after the conflict is resolved.",
                    "Feasibility has been recomputed.",
                    "Switch to verified fulfillment, explanation, or safe termination.",
                ),
            ),
            base_insight=(
                "elicit_user_authored_boundary",
                "An empty joint set calls for a targeted flexibility question; the Agent may identify the conflict but must not guess the user's acceptable boundary.",
                "Increase conflict recovery while preserving user control.",
            ),
            forbidden_operations=(
                "Do not recommend while the verified feasible set is empty.",
                "Do not infer, invent, or silently apply the user's acceptable relaxation boundary.",
            ),
            fallback_policy="explain the joint conflict and stop transparently when the user declines to relax a condition",
            execution_contract=_execution_contract(
                slug="joint_infeasibility_negotiation",
                role=PatternExecutionRole.PRIMARY,
                priority=20,
                fallback_utterance=(
                    "No publicly verified option satisfies all current requirements, so I cannot safely continue."
                ),
                action_templates=(
                    _ask_template(
                        template_id="ask_tradeoff_boundary",
                        sop_step_id="ask_boundary",
                        policy_id="disclosed_numeric_tradeoff_order",
                    ),
                ),
                question_policies=(
                    _question_policy(
                        policy_id="disclosed_numeric_tradeoff_order",
                        targets=("price", "rating", "popularity_percentile"),
                        require_disclosed_slot=True,
                    ),
                ),
                insight_adaptations=(
                    PatternInsightAdaptation(
                        adaptation_id="retain_alternative_tradeoff_questions",
                        kind=PatternInsightAdaptationKind.RETAIN_ALTERNATIVE_QUESTIONS,
                        source_insight_id=_base_insight_id("joint_infeasibility_negotiation"),
                        authorizing_sop_step_id="ask_boundary",
                        question_template_id="ask_tradeoff_boundary",
                        question_target_policy_id="disclosed_numeric_tradeoff_order",
                        maximum_additional_options=2,
                    ),
                ),
            ),
            provenance=PatternProvenance.LEGACY_MIGRATED_SUCCESS_DISTILLED,
            legacy_seed_refs=("template_5",),
            source_scenario="conflict",
        ),
        _PatternBlueprint(
            slug="preference_grounded_explanation",
            name="Preference-Grounded Explanation Closure",
            family=PatternStrategyFamily.GROUNDED_EXPLANATION,
            task_description="Close an acceptance-contingent explanation request with verifiable preference evidence.",
            state_description="A previously recommended acceptable item is pending a grounded explanation.",
            solution_description=(
                "Collect catalog evidence for the disclosed specific category and every active numeric hard preference, then explain the pending item once."
            ),
            applicability=PatternApplicability(requires_explanation_pending=True),
            sop_steps=(
                _step(
                    "collect_evidence",
                    PatternOperation.INSPECT_CATALOG,
                    "Assemble the complete public justification contract.",
                    "One pending item and its active public preferences are known.",
                    "Read the item's specific category and each stated numeric hard-preference field.",
                    success="explain",
                    failure="rematch",
                ),
                _step(
                    "explain",
                    PatternOperation.EXPLAIN,
                    "Explain why the pending item matches the user's own preferences.",
                    "Every required evidence field is verifiable in the catalog.",
                    "Cite typed item-field references; title or generic assurance alone is insufficient.",
                    success="rematch",
                    failure="collect_evidence",
                ),
                _step(
                    "rematch",
                    PatternOperation.REMATCH_PATTERN,
                    "Close or repair the interaction based on the response.",
                    "The user evaluated the explanation.",
                    "Terminate on acceptance; otherwise return to evidence-safe repair.",
                ),
            ),
            base_insight=(
                "explanation_is_obligation",
                "Once acceptance depends on an explanation, recommending again is not progress; the pending evidence contract must be completed.",
                "Eliminate ignored explanation requests and title-only shortcuts.",
            ),
            forbidden_operations=(
                "Do not recommend another item while a valid explanation request is pending.",
                "Do not cite a field that is absent or not tied to the pending item.",
            ),
            fallback_policy="choose a verifiable alternate only after explicitly acknowledging missing evidence",
            execution_contract=_execution_contract(
                slug="preference_grounded_explanation",
                role=PatternExecutionRole.PRIMARY,
                priority=10,
                fallback_utterance=(
                    "I cannot ground an explanation for a currently public recommendation."
                ),
                action_templates=(
                    _explain_template(
                        template_id="explain_with_visible_evidence",
                        sop_step_id="explain",
                    ),
                ),
            ),
            provenance=PatternProvenance.FAILURE_CORRECTED,
            source_scenario=None,
        ),
        _PatternBlueprint(
            slug="rejection_aware_candidate_repair",
            name="Rejection-Aware Candidate Repair",
            family=PatternStrategyFamily.REJECTION_REPAIR,
            task_description="Recover after a rejected or unverifiable recommendation without repeating failed evidence.",
            state_description="The latest public user act is rejection and the dialogue remains recoverable.",
            solution_description=(
                "Attribute rejection to a known violation, missing evidence, or unresolved preference; exclude rejected items and choose the corresponding ask, negotiate, or verified-recommend branch."
            ),
            applicability=PatternApplicability(
                allowed_last_user_acts=[BenchmarkUserAct.REJECT],
                requires_explanation_pending=False,
            ),
            sop_steps=(
                _step(
                    "diagnose",
                    PatternOperation.INSPECT_STATE,
                    "Identify why the previous slate failed.",
                    "A public rejection response is available.",
                    "Separate known hard violation, UNKNOWN evidence, incompleteness, and joint infeasibility.",
                    success="exclude",
                    failure="rematch",
                ),
                _step(
                    "exclude",
                    PatternOperation.INVALIDATE_STALE_PLAN,
                    "Prevent repetition of the failed item and rationale.",
                    "The rejected item IDs are public history.",
                    "Exclude rejected IDs and any evidence-equivalent failed slate.",
                    success="rematch",
                    failure="rematch",
                ),
                _step(
                    "rematch",
                    PatternOperation.REMATCH_PATTERN,
                    "Select the correction implied by the diagnosed state.",
                    "DSUS and catalog viability are recomputed.",
                    "Ask if incomplete, negotiate if empty, otherwise recommend a verified unseen item.",
                ),
            ),
            base_insight=(
                "diagnose_before_retry",
                "A rejection is evidence about the failed slate, not permission to sample another item under the same broken plan.",
                "Reduce repeated rejections and hard-evidence violations.",
            ),
            forbidden_operations=(
                "Do not repeat a rejected item.",
                "Do not treat UNKNOWN catalog evidence as proof of preference satisfaction.",
            ),
            fallback_policy="ask one diagnostic question only when the rejection leaves intent incomplete",
            execution_contract=_execution_contract(
                slug="rejection_aware_candidate_repair",
                role=PatternExecutionRole.OVERLAY,
                priority=0,
                fallback_utterance=(
                    "I cannot safely repair the rejected recommendation from the public information."
                ),
                insight_adaptations=(
                    PatternInsightAdaptation(
                        adaptation_id="block_optional_retry_shortcuts",
                        kind=PatternInsightAdaptationKind.BLOCK_ADAPTATIONS,
                        source_insight_id=_base_insight_id("rejection_aware_candidate_repair"),
                        authorizing_sop_step_id="rematch",
                        blocked_adaptation_kinds=[
                            PatternInsightAdaptationKind.RETAIN_ALTERNATIVE_QUESTIONS,
                            PatternInsightAdaptationKind.ALLOW_ONE_EARLY_VERIFIED_RECOMMENDATION,
                        ],
                    ),
                ),
            ),
            provenance=PatternProvenance.FAILURE_CORRECTED,
        ),
        _PatternBlueprint(
            slug="dialogue_contract_reorientation",
            name="Dialogue-Contract Reorientation",
            family=PatternStrategyFamily.DIALOGUE_REORIENTATION,
            task_description="Recover after a redundant question or irrelevant interaction branch.",
            state_description="The user reminds the Agent of known information or reports no preference for an irrelevant target.",
            solution_description=(
                "Acknowledge the failed interaction move, preserve the existing public store, and immediately execute the still-pending decision obligation."
            ),
            applicability=PatternApplicability(
                allowed_last_user_acts=[
                    BenchmarkUserAct.REMIND,
                    BenchmarkUserAct.NO_PREFERENCE,
                    BenchmarkUserAct.CORRECT_INVALID_ACTION,
                ]
            ),
            sop_steps=(
                _step(
                    "locate_obligation",
                    PatternOperation.INSPECT_STATE,
                    "Recover the obligation that the last move failed to address.",
                    "The latest user act signals redundancy, irrelevance, or invalidity.",
                    "Read explanation pending, completeness, feasibility, and asked-target history in priority order.",
                    success="rematch",
                    failure="rematch",
                ),
                _step(
                    "rematch",
                    PatternOperation.REMATCH_PATTERN,
                    "Return to the shortest valid Skill path.",
                    "A decision obligation is identifiable from public state.",
                    "Prefer explain, negotiate, decisive ask, then verified recommend in that order.",
                ),
            ),
            base_insight=(
                "do_not_compound_contract_error",
                "After a redundant or irrelevant move, another exploratory move compounds patience loss; restore the pending obligation directly.",
                "Improve recovery speed and reduce abandonment.",
            ),
            forbidden_operations=(
                "Do not repeat the failed question target.",
                "Do not erase previously authorized public preferences.",
            ),
            fallback_policy="choose the lowest-turn safe action under the current DSUS posterior",
            execution_contract=_execution_contract(
                slug="dialogue_contract_reorientation",
                role=PatternExecutionRole.OVERLAY,
                priority=0,
                fallback_utterance=(
                    "I cannot safely recover the pending public dialogue obligation."
                ),
                insight_adaptations=(
                    PatternInsightAdaptation(
                        adaptation_id="block_optional_reorientation_detours",
                        kind=PatternInsightAdaptationKind.BLOCK_ADAPTATIONS,
                        source_insight_id=_base_insight_id("dialogue_contract_reorientation"),
                        authorizing_sop_step_id="rematch",
                        blocked_adaptation_kinds=[
                            PatternInsightAdaptationKind.RETAIN_ALTERNATIVE_QUESTIONS,
                            PatternInsightAdaptationKind.ALLOW_ONE_EARLY_VERIFIED_RECOMMENDATION,
                        ],
                    ),
                ),
            ),
            provenance=PatternProvenance.FAILURE_CORRECTED,
        ),
        _PatternBlueprint(
            slug="low_patience_shortest_path",
            name="Low-Patience Shortest Safe Path",
            family=PatternStrategyFamily.PATIENCE_PRESERVATION,
            task_description="Complete or safely end a recoverable recommendation before further interaction causes abandonment.",
            state_description="DSUS estimates high near-term abandonment risk from consumed patience.",
            solution_description=(
                "Remove optional dialogue steps while preserving hard safety: satisfy a pending explanation, make the one decisive repair, or recommend the best verified slate now."
            ),
            applicability=PatternApplicability(
                allowed_patience_risks=[PatienceRisk.HIGH]
            ),
            sop_steps=(
                _step(
                    "rank_obligations",
                    PatternOperation.INSPECT_STATE,
                    "Find the minimum-turn safe completion path.",
                    "Patience risk is high and the episode is nonterminal.",
                    "Prioritize mandatory explanation/negotiation, then one decisive ask, then verified recommendation.",
                    success="rematch",
                    failure="stop",
                ),
                _step(
                    "rematch",
                    PatternOperation.REMATCH_PATTERN,
                    "Execute exactly one mandatory next operation.",
                    "A safe completion branch exists.",
                    "Disable optional questions and rematch the highest-priority obligation Skill.",
                ),
                _step(
                    "stop",
                    PatternOperation.STOP,
                    "Avoid further harm when no safe branch remains.",
                    "No verifiable recommendation or authorized recovery is available.",
                    "Explain the limitation briefly and stop.",
                ),
            ),
            base_insight=(
                "patience_changes_value",
                "The same information-gathering action can become harmful after repeated failures because its abandonment cost rises with consumed patience.",
                "Trade a small information gain for lower abandonment risk when necessary.",
            ),
            forbidden_operations=(
                "Do not ask an optional or repeated question at high patience risk.",
                "Do not sacrifice hard-constraint verification merely to end faster.",
            ),
            fallback_policy="stop safely if neither a mandatory response nor a verified slate is available",
            execution_contract=_execution_contract(
                slug="low_patience_shortest_path",
                role=PatternExecutionRole.OVERLAY,
                priority=0,
                fallback_utterance=(
                    "I cannot identify a safe next step from the currently public requirements."
                ),
                global_safe_stop_authority=True,
                insight_adaptations=(
                    PatternInsightAdaptation(
                        adaptation_id="block_optional_question_branches",
                        kind=PatternInsightAdaptationKind.BLOCK_ADAPTATIONS,
                        source_insight_id=_base_insight_id("low_patience_shortest_path"),
                        authorizing_sop_step_id="rematch",
                        blocked_adaptation_kinds=[
                            PatternInsightAdaptationKind.RETAIN_ALTERNATIVE_QUESTIONS,
                        ],
                    ),
                ),
            ),
            provenance=PatternProvenance.FAILURE_CORRECTED,
        ),
    )


def _actual_action_by_state(
    data: LoadedDSUSDevelopmentData,
) -> dict[str, tuple[BenchmarkActionType, str | None]]:
    episode_by_id = {item.episode_id: item for item in data.episodes}
    actions = {}
    for feature in data.features:
        episode = episode_by_id[feature.episode_id]
        if feature.turn_index >= len(episode.turns):
            raise ValueError("DSUS state has no matching behavior turn")
        action = episode.turns[feature.turn_index].system_action
        actions[feature.state_id] = (action.action_type, action.question_target)
    return actions


def _expert_signature_counts(
    data: LoadedDSUSDevelopmentData,
) -> dict[str, Counter[tuple[BenchmarkActionType, ...]]]:
    label_by_episode: dict[str, DSUSStateLabel] = {}
    for feature, label in zip(data.features, data.labels, strict=True):
        label_by_episode.setdefault(feature.episode_id, label)
    result: dict[str, Counter[tuple[BenchmarkActionType, ...]]] = defaultdict(Counter)
    for episode in data.episodes:
        first = label_by_episode[episode.episode_id]
        if first.behavior_policy != DSUSBehaviorPolicy.CORRECTIVE_EXPERT:
            continue
        signature = tuple(turn.system_action.action_type for turn in episode.turns)
        result[first.scenario_family.value][signature] += 1
    return result


def _contrast_principle(
    label: DSUSStateLabel,
    actual_action: BenchmarkActionType,
) -> tuple[str, str, str]:
    optimal = label.optimal_action_type
    trigger = (
        f"phase={label.dialogue_phase.value}; completeness={label.intent_completeness.value}; "
        f"feasibility={label.candidate_feasibility.value}; last_act="
        f"{label.last_user_act.value if label.last_user_act else 'none'}"
    )
    if optimal == BenchmarkActionType.EXPLAIN:
        principle = "Complete the pending preference-grounded explanation instead of taking another dialogue action."
        effect = "Avoid ignored explanation requests and dialogue-contract cost."
    elif optimal == BenchmarkActionType.PROPOSE_RELAX:
        principle = "Negotiate one exact feasible trade-off instead of acting on an empty candidate set."
        effect = "Restore feasibility without hard-constraint violations."
    elif label.decision_obligation == DecisionObligation.NEGOTIATE_TRADEOFF:
        principle = "Ask for a public, user-authored flexibility boundary before attempting to restore feasibility."
        effect = "Recover from joint infeasibility without leaking or guessing a hidden relaxation."
    elif optimal == BenchmarkActionType.ASK:
        principle = "Ask the one unresolved decision-relevant target before committing to a slate."
        effect = "Reduce premature recommendations while limiting question cost."
    elif optimal == BenchmarkActionType.RECOMMEND:
        principle = "Act on the verified public state now instead of spending another low-value turn."
        effect = "Protect patience and shorten successful interactions."
    else:
        principle = "Stop safely because no productive public action remains."
        effect = "Avoid repeated invalid or harmful actions."
    return trigger, f"Prefer {optimal.value} over {actual_action.value}: {principle}", effect


def _base_insight(
    blueprint: _PatternBlueprint,
    *,
    support_count: int,
) -> PatternInsight:
    trigger, principle, effect = blueprint.base_insight
    evidence = {
        "pattern_slug": blueprint.slug,
        "trigger": trigger,
        "support_count": support_count,
        "source": "successful_and_corrected_training_states",
    }
    return PatternInsight(
        insight_id=f"insight-{blueprint.slug}-base",
        trigger=trigger,
        principle=principle,
        intended_effect=effect,
        evidence_signature=content_sha256(evidence),
        support_count=max(1, support_count),
        source="success_distillation",
    )


def _correction_insights(
    blueprint: _PatternBlueprint,
    matching: list[DSUSStateLabel],
    actual_by_state: dict[str, tuple[BenchmarkActionType, str | None]],
    *,
    maximum: int = 3,
) -> tuple[list[PatternInsight], int]:
    signatures: Counter[tuple[str, str, str, str]] = Counter()
    labels_by_signature: dict[tuple[str, str, str, str], DSUSStateLabel] = {}
    for label in matching:
        actual_type, actual_target = actual_by_state[label.state_id]
        target_differs = (
            actual_type == BenchmarkActionType.ASK
            and actual_target != label.optimal_question_target
        )
        if actual_type == label.optimal_action_type and not target_differs:
            continue
        key = (
            actual_type.value,
            label.optimal_action_type.value,
            label.dialogue_phase.value,
            label.last_user_act.value if label.last_user_act else "none",
        )
        signatures[key] += 1
        labels_by_signature.setdefault(key, label)
    insights = []
    for index, (signature, count) in enumerate(signatures.most_common(maximum), start=1):
        label = labels_by_signature[signature]
        actual_type = BenchmarkActionType(signature[0])
        trigger, principle, effect = _contrast_principle(label, actual_type)
        evidence = {
            "pattern_slug": blueprint.slug,
            "contrast_signature": list(signature),
            "support_count": count,
        }
        insights.append(
            PatternInsight(
                insight_id=f"insight-{blueprint.slug}-contrast-{index}",
                trigger=trigger,
                principle=principle,
                intended_effect=effect,
                evidence_signature=content_sha256(evidence),
                support_count=count,
                source="off_policy_failure_correction",
            )
        )
    return insights, sum(signatures.values())


def _execution_signatures(
    blueprint: _PatternBlueprint,
    expert_signatures: dict[str, Counter[tuple[BenchmarkActionType, ...]]],
    matching: list[DSUSStateLabel],
) -> list[PatternExecutionSignature]:
    if blueprint.source_scenario is not None:
        values = expert_signatures[blueprint.source_scenario]
        return [
            PatternExecutionSignature(
                action_sequence=list(signature),
                support_count=count,
            source=f"public_corrective_expert:{blueprint.source_scenario}",
            )
            for signature, count in values.most_common()
        ]
    optimal_counts = Counter(label.optimal_action_type for label in matching)
    return [
        PatternExecutionSignature(
            action_sequence=[action_type],
            support_count=count,
            source="corrected_off_policy_state",
        )
        for action_type, count in optimal_counts.most_common()
    ]


def _pattern_id_payload(
    blueprint: _PatternBlueprint,
    *,
    applicability_support: int,
    correction_support: int,
) -> dict[str, Any]:
    return {
        "induction_version": PATTERN_INDUCTION_VERSION,
        "slug": blueprint.slug,
        "family": blueprint.family.value,
        "applicability": blueprint.applicability.model_dump(mode="json"),
        "sop": [item.model_dump(mode="json") for item in blueprint.sop_steps],
        "execution_contract": blueprint.execution_contract.model_dump(mode="json"),
        "base_insight": list(blueprint.base_insight),
        "applicability_support": applicability_support,
        "correction_support": correction_support,
    }


def induce_pattern_library(dsus_dir: Path) -> PatternInductionResult:
    data = load_dsus_development_artifact(dsus_dir)
    feature_by_state = {item.state_id: item for item in data.features}
    label_by_state = {item.state_id: item for item in data.labels}
    if set(feature_by_state) != set(label_by_state):
        raise ValueError("Pattern induction requires exact DSUS public/private joins")
    actual_by_state = _actual_action_by_state(data)
    expert_signatures = _expert_signature_counts(data)
    correction_state_ids = {
        state_id
        for state_id, label in label_by_state.items()
        if actual_by_state[state_id][0] != label.optimal_action_type
        or (
            label.optimal_action_type == BenchmarkActionType.ASK
            and actual_by_state[state_id][1] != label.optimal_question_target
        )
    }

    patterns: list[PatternSkill] = []
    for blueprint in _blueprints():
        matching = [
            label
            for label in data.labels
            if blueprint.applicability.matches(label)
        ]
        if not matching:
            continue
        archetypes = {item.state_archetype_id for item in matching}
        correction_insights, correction_support = _correction_insights(
            blueprint,
            matching,
            actual_by_state,
        )
        expert_episode_ids = {
            feature_by_state[item.state_id].episode_id
            for item in matching
            if item.behavior_policy == DSUSBehaviorPolicy.CORRECTIVE_EXPERT
        }
        pattern_id = (
            f"pattern-{blueprint.slug}-"
            + content_sha256(
                _pattern_id_payload(
                    blueprint,
                    applicability_support=len(matching),
                    correction_support=correction_support,
                )
            )[:12]
        )
        patterns.append(
            PatternSkill(
                pattern_id=pattern_id,
                name=blueprint.name,
                strategy_family=blueprint.family,
                task_description=blueprint.task_description,
                state_description=blueprint.state_description,
                solution_description=blueprint.solution_description,
                applicability=blueprint.applicability,
                sop_steps=list(blueprint.sop_steps),
                insights=[
                    _base_insight(blueprint, support_count=len(matching)),
                    *correction_insights,
                ],
                forbidden_operations=list(blueprint.forbidden_operations),
                fallback_policy=blueprint.fallback_policy,
                execution_contract=blueprint.execution_contract,
                execution_signatures=_execution_signatures(
                    blueprint,
                    expert_signatures,
                    matching,
                ),
                legacy_seed_refs=list(blueprint.legacy_seed_refs),
                provenance=blueprint.provenance,
                supporting_expert_episode_count=len(expert_episode_ids),
                supporting_correction_state_count=correction_support,
                covered_state_count=len(matching),
                covered_archetype_count=len(archetypes),
            )
        )

    library_payload = [
        item.model_dump(mode="json") for item in sorted(patterns, key=lambda value: value.pattern_id)
    ]
    library_id = f"pattern-library-{content_sha256(library_payload)[:16]}"
    library = StateConditionedPatternLibrary(
        library_id=library_id,
        parent_dsus_build_id=data.manifest.build_id,
        patterns=patterns,
    )
    archetype_labels: dict[str, list[DSUSStateLabel]] = defaultdict(list)
    for label in data.labels:
        archetype_labels[label.state_archetype_id].append(label)
    coverage = tuple(
        ArchetypePatternCoverage(
            state_archetype_id=archetype_id,
            state_count=len(labels),
            pattern_ids=sorted(
                pattern.pattern_id
                for pattern in patterns
                if any(pattern.applicability.matches(label) for label in labels)
            ),
            decision_obligation_counts=dict(
                sorted(Counter(item.decision_obligation.value for item in labels).items())
            ),
        )
        for archetype_id, labels in sorted(archetype_labels.items())
    )
    covered_state_counts = [
        sum(pattern.applicability.matches(label) for pattern in patterns)
        for label in data.labels
    ]
    uncovered_states = sum(value == 0 for value in covered_state_counts)
    uncovered_archetypes = sum(not item.pattern_ids for item in coverage)
    signature_summary = {
        f"{scenario}:{'->'.join(action.value for action in signature)}": count
        for scenario, values in sorted(expert_signatures.items())
        for signature, count in values.items()
    }
    summary_identity = {
        "induction_version": PATTERN_INDUCTION_VERSION,
        "parent_dsus_build_id": data.manifest.build_id,
        "parent_dsus_manifest_sha256": data.manifest_sha256,
        "library_id": library_id,
        "coverage_sha256": content_sha256(
            [item.model_dump(mode="json") for item in coverage]
        ),
    }
    induction_id = f"pattern-induction-{content_sha256(summary_identity)[:16]}"
    summary = PatternInductionSummary(
        induction_id=induction_id,
        parent_dsus_build_id=data.manifest.build_id,
        pattern_library_id=library_id,
        pattern_count=len(patterns),
        strategy_family_counts=dict(
            sorted(Counter(item.strategy_family.value for item in patterns).items())
        ),
        sop_step_count=sum(len(item.sop_steps) for item in patterns),
        insight_count=sum(len(item.insights) for item in patterns),
        correction_state_count=len(correction_state_ids),
        state_count=len(data.labels),
        archetype_count=len(archetype_labels),
        covered_state_count=len(data.labels) - uncovered_states,
        covered_archetype_count=len(coverage) - uncovered_archetypes,
        uncovered_state_count=uncovered_states,
        uncovered_archetype_count=uncovered_archetypes,
        mean_patterns_per_state=sum(covered_state_counts) / len(covered_state_counts),
        mean_patterns_per_archetype=(
            sum(len(item.pattern_ids) for item in coverage) / len(coverage)
        ),
        multi_pattern_state_rate=(
            sum(value > 1 for value in covered_state_counts) / len(covered_state_counts)
        ),
        expert_execution_signature_counts=dict(sorted(signature_summary.items())),
        pattern_support={
            item.pattern_id: {
                "states": item.covered_state_count,
                "archetypes": item.covered_archetype_count,
                "expert_episodes": item.supporting_expert_episode_count,
                "correction_states": item.supporting_correction_state_count,
            }
            for item in patterns
        },
        all_patterns_have_sop_and_insights=all(
            item.sop_steps and item.insights for item in patterns
        ),
        all_states_covered=uncovered_states == 0,
        all_archetypes_covered=uncovered_archetypes == 0,
    )
    return PatternInductionResult(
        induction_id=induction_id,
        parent_dsus_manifest_sha256=data.manifest_sha256,
        library=library,
        coverage=coverage,
        summary=summary,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_idempotently(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different Pattern file: {path}")
        return
    path.write_text(text, encoding="utf-8")


def write_pattern_library_artifact(
    result: PatternInductionResult,
    output_root: Path,
    *,
    project_root: Path,
) -> Path:
    run_dir = output_root / result.induction_id
    run_dir.mkdir(parents=True, exist_ok=True)
    library_payload = result.library.model_dump(mode="json")
    coverage_payload = [item.model_dump(mode="json") for item in result.coverage]
    summary_payload = result.summary.model_dump(mode="json")
    outputs = {
        "pattern_library.public.json": json.dumps(
            library_payload, ensure_ascii=False, indent=2
        )
        + "\n",
        "archetype_coverage.private.jsonl": "".join(
            canonical_json(item) + "\n" for item in coverage_payload
        ),
        "summary.json": json.dumps(summary_payload, ensure_ascii=False, indent=2)
        + "\n",
    }
    for filename, text in outputs.items():
        _write_idempotently(run_dir / filename, text)
    files = [
        PatternArtifactFile(
            relative_path=filename,
            sha256=_file_sha256(run_dir / filename),
            byte_size=(run_dir / filename).stat().st_size,
            line_count=text.count("\n") if filename.endswith(".jsonl") else None,
        )
        for filename, text in sorted(outputs.items())
    ]
    manifest = PatternInductionManifest(
        induction_id=result.induction_id,
        library_id=result.library.library_id,
        parent_dsus_build_id=result.library.parent_dsus_build_id,
        parent_dsus_manifest_sha256=result.parent_dsus_manifest_sha256,
        library_sha256=content_sha256(library_payload),
        coverage_sha256=content_sha256(coverage_payload),
        summary_sha256=content_sha256(summary_payload),
        source_tree_sha256=source_tree_sha256(project_root),
        outputs=files,
    )
    _write_idempotently(
        run_dir / "manifest.json",
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
    )
    return run_dir
