"""Contract-driven public option compilation for TAIRAS Thought Patterns.

The online compiler receives only Agent-visible memory, a public DSUS
posterior, and a public catalog.  It deliberately has no environment,
private-case, reward, or evaluator argument.  Most importantly, it does not
dispatch on a library taxonomy: a matched Skill's serialized execution contract
determines priority, typed SOP actions, evidence policy, fallback, and bounded
insight adaptations.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import Field, model_validator

from taira_mt.agent.catalog_tools import PublicCatalogTool
from taira_mt.agent.memory import PublicInteractionMemory
from taira_mt.benchmark.schemas import (
    BenchmarkActionType,
    BenchmarkSystemAction,
    EvidenceReference,
)
from taira_mt.dsus.schemas import DSUSPosterior, DSUSStateFeature, PublicConstraintObservation
from taira_mt.models import ConstraintImportance, StrictModel
from taira_mt.pattern_planner.matching import PatternMatch, match_patterns
from taira_mt.pattern_planner.schemas import (
    PATTERN_EXECUTION_CONTRACT_VERSION,
    PatternCandidateEvidencePolicy,
    PatternExecutionContract,
    PatternExecutionRole,
    PatternInsightAdaptation,
    PatternInsightAdaptationKind,
    PatternQuestionTargetPolicy,
    PatternSkill,
    PatternSOPActionTemplate,
    PatternSOPStep,
    StateConditionedPatternLibrary,
)
from taira_mt.provenance import content_sha256

# v4 binds every public option to a serializable execution contract.  Any CIV
# records generated under v2 are therefore development-only and must not be
# mixed with a v4 selector/ranker artifact.
PATTERN_PLAN_INSTANTIATOR_VERSION = "public_pattern_plan_instantiator_v4_contract_compiler"
_MAX_PATTERN_OPTIONS = 5


class PatternBranchKind(str):
    """String constants keep option artifacts legible without a macro action."""

    SOP = "sop"
    INSIGHT_ADAPTATION = "insight_adaptation"
    FALLBACK = "fallback"


class PublicInteractionState(StrictModel):
    """Whitelisted online state; no evaluator/private-case field is permitted."""

    domain: str
    serialized_dialogue: str
    public_constraints: list[PublicConstraintObservation]
    asked_question_targets: list[str] = Field(default_factory=list)
    rejected_item_ids: list[str] = Field(default_factory=list)
    latest_recommendation_item_ids: list[str] = Field(default_factory=list)
    public_verified_candidate_count: int = Field(ge=0)
    public_unknown_candidate_count: int = Field(ge=0)
    dsus_posterior: DSUSPosterior

    @model_validator(mode="after")
    def validate_unique_public_memory(self) -> PublicInteractionState:
        for field_name in (
            "asked_question_targets",
            "rejected_item_ids",
            "latest_recommendation_item_ids",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
        slots = [item.slot for item in self.public_constraints]
        if len(slots) != len(set(slots)):
            raise ValueError("public constraint observations must have unique slots")
        return self

    @property
    def public_state_hash(self) -> str:
        return content_sha256(self.model_dump(mode="json"))

    @classmethod
    def from_dsus_feature(
        cls,
        feature: DSUSStateFeature,
        posterior: DSUSPosterior,
        *,
        rejected_item_ids: Iterable[str] = (),
        latest_recommendation_item_ids: Iterable[str] = (),
    ) -> PublicInteractionState:
        """Adapt a whitelisted DSUS feature to the online state interface."""

        return cls(
            domain=feature.domain,
            serialized_dialogue=feature.serialized_dialogue,
            public_constraints=feature.public_constraints,
            asked_question_targets=feature.asked_question_targets,
            rejected_item_ids=list(rejected_item_ids),
            latest_recommendation_item_ids=list(latest_recommendation_item_ids),
            public_verified_candidate_count=feature.public_verified_candidate_count,
            public_unknown_candidate_count=feature.public_unknown_candidate_count,
            dsus_posterior=posterior,
        )

    @classmethod
    def from_memory(
        cls,
        memory: PublicInteractionMemory,
        catalog: PublicCatalogTool,
        posterior: DSUSPosterior,
    ) -> PublicInteractionState:
        """Build planner state from actual public online memory, not a trace feature."""

        feature = memory.to_dsus_feature(catalog)
        return cls(
            domain=feature.domain,
            serialized_dialogue=feature.serialized_dialogue,
            public_constraints=feature.public_constraints,
            asked_question_targets=feature.asked_question_targets,
            rejected_item_ids=memory.rejected_item_ids,
            latest_recommendation_item_ids=memory.latest_recommendation_item_ids,
            public_verified_candidate_count=feature.public_verified_candidate_count,
            public_unknown_candidate_count=feature.public_unknown_candidate_count,
            dsus_posterior=posterior,
        )


class PatternPlanOption(StrictModel):
    """A typed action bound to a Skill contract, SOP step, and public evidence."""

    option_id: str = Field(min_length=64, max_length=64)
    primary_pattern_id: str
    overlay_pattern_ids: list[str] = Field(default_factory=list)
    sop_step_id: str
    branch_kind: str
    action: BenchmarkSystemAction
    rationale: str
    public_option_features: dict[str, str | float | bool]

    def model_inputs(self) -> dict[str, object]:
        return {
            "primary_pattern_id": self.primary_pattern_id,
            "overlay_pattern_ids": self.overlay_pattern_ids,
            "sop_step_id": self.sop_step_id,
            "branch_kind": self.branch_kind,
            "action": self.action.model_dump(mode="json"),
            "rationale": self.rationale,
            "public_option_features": self.public_option_features,
        }


@dataclass(frozen=True)
class _MatchedContract:
    """A library Skill whose public DSUS applicability exceeds its own gate."""

    pattern: PatternSkill
    contract: PatternExecutionContract
    structural_probability: float


def _posterior_map_value(posterior: DSUSPosterior, factor: str, value: str) -> float:
    return float(getattr(posterior, factor).get(value, 0.0))


def _visible_hard_constraints(
    state: PublicInteractionState,
) -> list[PublicConstraintObservation]:
    return [
        item
        for item in state.public_constraints
        if item.predicate.importance == ConstraintImportance.HARD
    ]


class PublicPatternPlanInstantiator:
    """Compile matched executable Thought-Pattern contracts into legal options.

    The implementation owns only generic public primitives (ASK, verified-only
    RECOMMEND, grounded EXPLAIN, STOP) and a five-option budget.  Which
    primitive is allowed, how it is bound, and whether a bounded insight
    adaptation may add an alternative are supplied by the serialized library.
    """

    def __init__(self, library: StateConditionedPatternLibrary) -> None:
        self.library = library
        if len({item.pattern_id for item in library.patterns}) != len(library.patterns):
            raise ValueError("Pattern library repeats a Pattern ID")
        missing_contract = [item.pattern_id for item in library.patterns if item.execution_contract is None]
        if missing_contract:
            raise ValueError(
                "Pattern runtime requires executable v4 contracts; legacy library is debug-only: "
                + ", ".join(sorted(missing_contract))
            )
        self._by_pattern_id = {item.pattern_id: item for item in library.patterns}
        self._contract_by_pattern_id = {
            item.pattern_id: self._require_contract(item) for item in library.patterns
        }
        contract_ids = [contract.contract_id for contract in self._contract_by_pattern_id.values()]
        if len(contract_ids) != len(set(contract_ids)):
            raise ValueError("Pattern library repeats an execution contract ID")
        global_fallbacks = [
            item
            for item in library.patterns
            if self._contract_by_pattern_id[item.pattern_id]
            .fallback_contract.global_safe_stop_authority
        ]
        if len(global_fallbacks) != 1:
            raise ValueError("Pattern library requires exactly one global safe-stop contract")
        self._global_fallback_pattern = global_fallbacks[0]

    @staticmethod
    def _require_contract(pattern: PatternSkill) -> PatternExecutionContract:
        contract = pattern.execution_contract
        if contract is None:
            raise ValueError(f"Pattern {pattern.pattern_id} lacks an executable contract")
        if contract.contract_version != PATTERN_EXECUTION_CONTRACT_VERSION:
            raise ValueError(
                f"Pattern {pattern.pattern_id} uses unsupported execution contract "
                f"{contract.contract_version}"
            )
        return contract

    @staticmethod
    def _deduplicate(options: list[PatternPlanOption]) -> tuple[PatternPlanOption, ...]:
        unique: dict[str, PatternPlanOption] = {}
        for option in options:
            action_key = content_sha256(option.action.model_dump(mode="json"))
            unique.setdefault(action_key, option)
        return tuple(unique.values())

    def _matched_contracts(self, state: PublicInteractionState) -> tuple[_MatchedContract, ...]:
        match_by_pattern_id: dict[str, PatternMatch] = {
            item.pattern_id: item for item in match_patterns(self.library, state.dsus_posterior).matches
        }
        matched: list[_MatchedContract] = []
        for pattern in self.library.patterns:
            contract = self._contract_by_pattern_id[pattern.pattern_id]
            structural_probability = match_by_pattern_id[pattern.pattern_id].structural_probability
            if structural_probability >= contract.minimum_match_probability:
                matched.append(
                    _MatchedContract(
                        pattern=pattern,
                        contract=contract,
                        structural_probability=structural_probability,
                    )
                )
        return tuple(matched)

    @staticmethod
    def _select_primary(matched: Iterable[_MatchedContract]) -> _MatchedContract | None:
        candidates = [
            item for item in matched if item.contract.role == PatternExecutionRole.PRIMARY
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                item.contract.priority,
                -item.structural_probability,
                item.pattern.pattern_id,
            ),
        )

    @staticmethod
    def _overlays(matched: Iterable[_MatchedContract]) -> tuple[_MatchedContract, ...]:
        return tuple(
            sorted(
                (
                    item
                    for item in matched
                    if item.contract.role == PatternExecutionRole.OVERLAY
                ),
                key=lambda item: item.pattern.pattern_id,
            )
        )

    @staticmethod
    def _blocked_adaptation_kinds(
        overlays: Iterable[_MatchedContract],
    ) -> frozenset[PatternInsightAdaptationKind]:
        blocked: set[PatternInsightAdaptationKind] = set()
        for overlay in overlays:
            for adaptation in overlay.contract.insight_adaptations:
                if (
                    adaptation.enabled
                    and adaptation.kind == PatternInsightAdaptationKind.BLOCK_ADAPTATIONS
                ):
                    blocked.update(adaptation.blocked_adaptation_kinds)
        return frozenset(blocked)

    @staticmethod
    def _sop_step(pattern: PatternSkill, step_id: str) -> PatternSOPStep:
        for step in pattern.sop_steps:
            if step.step_id == step_id:
                return step
        raise ValueError(f"Pattern {pattern.pattern_id} lacks bound SOP step: {step_id}")

    @staticmethod
    def _policy_by_id(
        contract: PatternExecutionContract,
        policy_id: str,
        *,
        evidence: bool,
    ) -> PatternQuestionTargetPolicy | PatternCandidateEvidencePolicy:
        values = (
            contract.candidate_evidence_policies
            if evidence
            else contract.question_target_policies
        )
        for policy in values:
            if policy.policy_id == policy_id:
                return policy
        label = "evidence" if evidence else "question-target"
        raise ValueError(f"Pattern contract lacks bound {label} policy: {policy_id}")

    @staticmethod
    def _template_by_id(
        contract: PatternExecutionContract,
        template_id: str,
    ) -> PatternSOPActionTemplate:
        for template in contract.sop_action_templates:
            if template.template_id == template_id:
                return template
        raise ValueError(f"Pattern contract lacks bound SOP action template: {template_id}")

    @staticmethod
    def _action_rationale(
        *,
        matched: _MatchedContract,
        sop_step_id: str,
        adaptation: PatternInsightAdaptation | None,
    ) -> str:
        step = PublicPatternPlanInstantiator._sop_step(matched.pattern, sop_step_id)
        if adaptation is None:
            return (
                f"{matched.pattern.name}: execute the contract-bound SOP step "
                f"'{step.objective}'."
            )
        return (
            f"{matched.pattern.name}: apply bounded insight adaptation "
            f"'{adaptation.adaptation_id}' under SOP step '{step.objective}'."
        )

    def _make_option(
        self,
        state: PublicInteractionState,
        matched: _MatchedContract,
        overlays: tuple[_MatchedContract, ...],
        action: BenchmarkSystemAction,
        *,
        sop_step_id: str,
        binding_id: str,
        branch_kind: str,
        rationale: str,
        features: dict[str, str | float | bool],
        adaptation: PatternInsightAdaptation | None = None,
    ) -> PatternPlanOption:
        overlay_ids = [item.pattern.pattern_id for item in overlays]
        contract = matched.contract
        merged_features: dict[str, str | float | bool] = {
            "execution_contract_id": contract.contract_id,
            "execution_contract_version": contract.contract_version,
            "execution_binding_id": binding_id,
            "primary_match_probability": matched.structural_probability,
            "active_overlay_count": float(len(overlays)),
            "insight_adaptation_enabled": adaptation is not None,
            **features,
        }
        if adaptation is not None:
            merged_features["insight_adaptation_id"] = adaptation.adaptation_id
        payload = {
            "instantiator_version": PATTERN_PLAN_INSTANTIATOR_VERSION,
            "public_state_hash": state.public_state_hash,
            "primary_pattern_id": matched.pattern.pattern_id,
            "execution_contract_id": contract.contract_id,
            "execution_contract_version": contract.contract_version,
            "overlay_pattern_ids": overlay_ids,
            "sop_step_id": sop_step_id,
            "binding_id": binding_id,
            "adaptation_id": None if adaptation is None else adaptation.adaptation_id,
            "branch_kind": branch_kind,
            "action": action.model_dump(mode="json"),
        }
        return PatternPlanOption(
            option_id=content_sha256(payload),
            primary_pattern_id=matched.pattern.pattern_id,
            overlay_pattern_ids=overlay_ids,
            sop_step_id=sop_step_id,
            branch_kind=branch_kind,
            action=action,
            rationale=rationale,
            public_option_features=merged_features,
        )

    @staticmethod
    def _question_targets(
        state: PublicInteractionState,
        policy: PatternQuestionTargetPolicy,
    ) -> tuple[str, ...]:
        asked = set(state.asked_question_targets)
        observed_slots = {item.slot for item in state.public_constraints}
        targets: list[str] = []
        for target in policy.target_order:
            if target in asked:
                continue
            if policy.require_unobserved_slot and target in observed_slots:
                continue
            if policy.require_disclosed_slot and target not in observed_slots:
                continue
            targets.append(target)
        return tuple(targets)

    def _ask_option(
        self,
        state: PublicInteractionState,
        matched: _MatchedContract,
        overlays: tuple[_MatchedContract, ...],
        template: PatternSOPActionTemplate,
        target: str,
        *,
        adaptation: PatternInsightAdaptation | None = None,
    ) -> PatternPlanOption:
        return self._make_option(
            state,
            matched,
            overlays,
            BenchmarkSystemAction(
                action_type=BenchmarkActionType.ASK,
                question_target=target,
                utterance=f"Could you clarify your preference about {target.replace('_', ' ')}?",
            ),
            sop_step_id=template.sop_step_id,
            binding_id=template.template_id if adaptation is None else adaptation.adaptation_id,
            branch_kind=(
                PatternBranchKind.SOP
                if adaptation is None
                else PatternBranchKind.INSIGHT_ADAPTATION
            ),
            rationale=self._action_rationale(
                matched=matched,
                sop_step_id=template.sop_step_id,
                adaptation=adaptation,
            ),
            features={
                "action_type": "ask",
                "question_target": target,
                "was_already_asked": target in state.asked_question_targets,
                "empty_feasibility_probability": _posterior_map_value(
                    state.dsus_posterior, "candidate_feasibility", "empty"
                ),
            },
            adaptation=adaptation,
        )

    def _recommend_option(
        self,
        state: PublicInteractionState,
        catalog: PublicCatalogTool,
        matched: _MatchedContract,
        overlays: tuple[_MatchedContract, ...],
        evidence_policy: PatternCandidateEvidencePolicy,
        *,
        sop_step_id: str,
        binding_id: str,
        adaptation: PatternInsightAdaptation | None = None,
    ) -> PatternPlanOption | None:
        """Compile the only v1 recommendation primitive: verified unseen top-1."""

        # The policy schema itself rejects weaker evidence, but these checks
        # keep the compiler fail-closed if a future deserialized object bypasses
        # schema construction.
        if not evidence_policy.verified_only or not evidence_policy.exclude_rejected:
            raise ValueError("Pattern recommendation contract weakens the public evidence guard")
        result = catalog.search(
            state.public_constraints,
            excluded_item_ids=state.rejected_item_ids,
        )
        if not result.verified_item_ids:
            return None
        item_id = result.verified_item_ids[0]
        features: dict[str, str | float | bool] = {
            "action_type": "recommend",
            "public_rank_position": 1.0,
            "slate_size": float(evidence_policy.maximum_items),
            "verified_candidate_count": float(len(result.verified_item_ids)),
            "unknown_candidate_count": float(len(result.unknown_evidence_item_ids)),
            "rejected_item_exclusion_count": float(len(state.rejected_item_ids)),
        }
        if adaptation is not None:
            features["is_early_verified_recommendation"] = True
        return self._make_option(
            state,
            matched,
            overlays,
            BenchmarkSystemAction(
                action_type=BenchmarkActionType.RECOMMEND,
                item_ids=[item_id],
                utterance="I found a publicly verified option that fits the details you shared.",
            ),
            sop_step_id=sop_step_id,
            binding_id=binding_id,
            branch_kind=(
                PatternBranchKind.SOP
                if adaptation is None
                else PatternBranchKind.INSIGHT_ADAPTATION
            ),
            rationale=self._action_rationale(
                matched=matched,
                sop_step_id=sop_step_id,
                adaptation=adaptation,
            ),
            features=features,
            adaptation=adaptation,
        )

    def _explain_option(
        self,
        state: PublicInteractionState,
        catalog: PublicCatalogTool,
        matched: _MatchedContract,
        overlays: tuple[_MatchedContract, ...],
        template: PatternSOPActionTemplate,
    ) -> PatternPlanOption | None:
        if not state.latest_recommendation_item_ids:
            return None
        item_id = state.latest_recommendation_item_ids[0]
        if item_id not in set(catalog.item_ids):
            return None
        item = catalog.inspect(item_id)
        fields = [
            constraint.slot
            for constraint in _visible_hard_constraints(state)
            if constraint.slot != "broad_category" and constraint.slot in item.attributes
        ]
        if not fields:
            fields = ["title"]
        return self._make_option(
            state,
            matched,
            overlays,
            BenchmarkSystemAction(
                action_type=BenchmarkActionType.EXPLAIN,
                item_ids=[item_id],
                evidence_refs=[
                    EvidenceReference(item_id=item_id, catalog_field=field) for field in fields
                ],
                utterance="Here is how that option matches the preferences you have shared.",
            ),
            sop_step_id=template.sop_step_id,
            binding_id=template.template_id,
            branch_kind=PatternBranchKind.SOP,
            rationale=self._action_rationale(
                matched=matched,
                sop_step_id=template.sop_step_id,
                adaptation=None,
            ),
            features={
                "action_type": "explain",
                "evidence_field_count": float(len(fields)),
                "pending_explanation_probability": _posterior_map_value(
                    state.dsus_posterior, "explanation_pending", "true"
                ),
            },
        )

    def _fallback_stop_option(
        self,
        state: PublicInteractionState,
        matched: _MatchedContract,
        overlays: tuple[_MatchedContract, ...],
        *,
        rationale: str,
    ) -> PatternPlanOption:
        fallback = matched.contract.fallback_contract
        return self._make_option(
            state,
            matched,
            overlays,
            BenchmarkSystemAction(action_type=BenchmarkActionType.STOP, utterance=fallback.utterance),
            sop_step_id=f"fallback:{fallback.fallback_id}",
            binding_id=fallback.fallback_id,
            branch_kind=PatternBranchKind.FALLBACK,
            rationale=rationale,
            features={
                "action_type": "stop",
                "patience_high_risk_probability": _posterior_map_value(
                    state.dsus_posterior, "patience_risk", "high"
                ),
                "global_safe_stop_authority": fallback.global_safe_stop_authority,
            },
        )

    def _compile_sop_templates(
        self,
        state: PublicInteractionState,
        catalog: PublicCatalogTool,
        primary: _MatchedContract,
        overlays: tuple[_MatchedContract, ...],
    ) -> list[PatternPlanOption]:
        options: list[PatternPlanOption] = []
        for template in primary.contract.sop_action_templates:
            if template.action_type == BenchmarkActionType.ASK:
                assert template.question_target_policy_id is not None
                policy = self._policy_by_id(
                    primary.contract,
                    template.question_target_policy_id,
                    evidence=False,
                )
                assert isinstance(policy, PatternQuestionTargetPolicy)
                for target in self._question_targets(state, policy)[: template.maximum_primary_options]:
                    options.append(self._ask_option(state, primary, overlays, template, target))
            elif template.action_type == BenchmarkActionType.RECOMMEND:
                assert template.candidate_evidence_policy_id is not None
                policy = self._policy_by_id(
                    primary.contract,
                    template.candidate_evidence_policy_id,
                    evidence=True,
                )
                assert isinstance(policy, PatternCandidateEvidencePolicy)
                option = self._recommend_option(
                    state,
                    catalog,
                    primary,
                    overlays,
                    policy,
                    sop_step_id=template.sop_step_id,
                    binding_id=template.template_id,
                )
                if option is not None:
                    options.append(option)
            elif template.action_type == BenchmarkActionType.EXPLAIN:
                option = self._explain_option(state, catalog, primary, overlays, template)
                if option is not None:
                    options.append(option)
            elif template.action_type == BenchmarkActionType.STOP:
                options.append(
                    self._fallback_stop_option(
                        state,
                        primary,
                        overlays,
                        rationale=(
                            f"{primary.pattern.name}: execute the contract-bound STOP template."
                        ),
                    )
                )
            else:
                raise ValueError(
                    f"v1 Pattern compiler cannot execute action type {template.action_type.value}"
                )
        return options

    @staticmethod
    def _adaptation_order(adaptation: PatternInsightAdaptation) -> tuple[int, str]:
        """Reserve a legal early recommendation before optional ASK branches.

        This ordering is global over the primitive vocabulary, not a
        strategy-family rule.  If the verified candidate primitive cannot run,
        the remaining option budget naturally goes to alternative questions.
        """

        priority = {
            PatternInsightAdaptationKind.ALLOW_ONE_EARLY_VERIFIED_RECOMMENDATION: 0,
            PatternInsightAdaptationKind.RETAIN_ALTERNATIVE_QUESTIONS: 1,
            PatternInsightAdaptationKind.BLOCK_ADAPTATIONS: 2,
        }[adaptation.kind]
        return priority, adaptation.adaptation_id

    def _compile_insight_adaptations(
        self,
        state: PublicInteractionState,
        catalog: PublicCatalogTool,
        primary: _MatchedContract,
        overlays: tuple[_MatchedContract, ...],
        ordinary_options: list[PatternPlanOption],
    ) -> list[PatternPlanOption]:
        blocked_kinds = self._blocked_adaptation_kinds(overlays)
        if not ordinary_options:
            return []
        additional: list[PatternPlanOption] = []
        existing_question_targets = {
            item.action.question_target
            for item in ordinary_options
            if item.action.action_type == BenchmarkActionType.ASK
        }
        has_sop_ask = bool(existing_question_targets)
        for adaptation in sorted(primary.contract.insight_adaptations, key=self._adaptation_order):
            if not adaptation.enabled or adaptation.kind in blocked_kinds:
                continue
            if adaptation.kind == PatternInsightAdaptationKind.ALLOW_ONE_EARLY_VERIFIED_RECOMMENDATION:
                if not has_sop_ask:
                    continue
                assert adaptation.candidate_evidence_policy_id is not None
                policy = self._policy_by_id(
                    primary.contract,
                    adaptation.candidate_evidence_policy_id,
                    evidence=True,
                )
                assert isinstance(policy, PatternCandidateEvidencePolicy)
                option = self._recommend_option(
                    state,
                    catalog,
                    primary,
                    overlays,
                    policy,
                    sop_step_id=adaptation.authorizing_sop_step_id,
                    binding_id=adaptation.adaptation_id,
                    adaptation=adaptation,
                )
                if option is not None:
                    additional.append(option)
            elif adaptation.kind == PatternInsightAdaptationKind.RETAIN_ALTERNATIVE_QUESTIONS:
                assert adaptation.question_template_id is not None
                assert adaptation.question_target_policy_id is not None
                template = self._template_by_id(primary.contract, adaptation.question_template_id)
                policy = self._policy_by_id(
                    primary.contract,
                    adaptation.question_target_policy_id,
                    evidence=False,
                )
                assert isinstance(policy, PatternQuestionTargetPolicy)
                targets = [
                    target
                    for target in self._question_targets(state, policy)
                    if target not in existing_question_targets
                ]
                for target in targets[: adaptation.maximum_additional_options]:
                    additional.append(
                        self._ask_option(
                            state,
                            primary,
                            overlays,
                            template,
                            target,
                            adaptation=adaptation,
                        )
                    )
                    existing_question_targets.add(target)
            elif adaptation.kind == PatternInsightAdaptationKind.BLOCK_ADAPTATIONS:
                # Blockers belong to matched overlay contracts and were
                # consumed before compiling this primary contract.
                continue
            else:  # pragma: no cover - enum exhaustiveness guard
                raise ValueError(f"unsupported Pattern insight primitive: {adaptation.kind}")
        return additional

    def instantiate(
        self,
        state: PublicInteractionState,
        catalog: PublicCatalogTool,
    ) -> tuple[PatternPlanOption, ...]:
        """Return up to five legal options compiled from matched contracts.

        If no primary Skill can prove public applicability, the one library
        designated global safe-stop contract is used.  The compiler never
        falls back to a strategy-family switch or a free-form LLM action.
        """

        if state.domain != catalog.domain:
            raise ValueError("Pattern state domain differs from public catalog domain")
        matched = self._matched_contracts(state)
        primary = self._select_primary(matched)
        overlays = self._overlays(matched)
        if primary is None:
            fallback_pattern = self._global_fallback_pattern
            fallback_contract = self._contract_by_pattern_id[fallback_pattern.pattern_id]
            global_match = _MatchedContract(
                pattern=fallback_pattern,
                contract=fallback_contract,
                structural_probability=0.0,
            )
            return (
                self._fallback_stop_option(
                    state,
                    global_match,
                    overlays,
                    rationale=(
                        "No primary execution contract met its public DSUS applicability gate; "
                        "use the library-authorized safe stop."
                    ),
                ),
            )

        ordinary = self._compile_sop_templates(state, catalog, primary, overlays)
        if not ordinary:
            return (
                self._fallback_stop_option(
                    state,
                    primary,
                    overlays,
                    rationale=(
                        f"{primary.pattern.name} has no legal public action under its bound SOP templates."
                    ),
                ),
            )
        adapted = self._compile_insight_adaptations(
            state,
            catalog,
            primary,
            overlays,
            ordinary,
        )
        options = self._deduplicate([*ordinary, *adapted])
        return options[:_MAX_PATTERN_OPTIONS]
