"""Pattern-constrained action generation and transparent net-utility ranking."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Protocol

from pydantic import Field, model_validator

from taira_mt.constraints.predicates import public_posterior_items
from taira_mt.dialogue_policy.pattern_policy import PatternConditionedPolicy
from taira_mt.models import (
    BeliefState,
    StrictModel,
    SystemAction,
    SystemActionType,
)
from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.recommender.catalog import CandidatePool
from taira_mt.recommender.oracle import rank_catalog_from_belief
from taira_mt.thought_patterns import ThoughtPatternLibrary
from taira_mt.thought_patterns.models import StateAwareThoughtPattern

PublicFeatureValue = float | int | str | bool
PATTERN_INSTANTIATOR_VERSION = "pattern_action_instantiator_v3"


class PatternInstantiation(StrictModel):
    instantiation_id: str
    pattern_id: str
    pattern_version: str
    pattern_library_hash: str
    candidate_pool_id: str
    instantiator_version: str = PATTERN_INSTANTIATOR_VERSION
    action: SystemAction
    public_features: dict[str, PublicFeatureValue]
    public_features_hash: str

    @model_validator(mode="after")
    def validate_public_feature_identity(self) -> PatternInstantiation:
        if self.public_features_hash != content_sha256(self.public_features):
            raise ValueError("public_features_hash does not match public_features")
        return self


class ValueEstimate(StrictModel):
    satisfaction_delta: float
    accept_probability: float = Field(ge=0.0, le=1.0)
    abandon_probability: float = Field(ge=0.0, le=1.0)
    hard_violation_probability: float = Field(ge=0.0, le=1.0)
    continuation_value: float


class ActionValueModel(Protocol):
    """Online value estimators receive public belief and Pattern instantiations only."""

    name: str

    def predict(
        self,
        belief: BeliefState,
        candidates: list[PatternInstantiation],
    ) -> list[ValueEstimate]: ...


class CostConfig(StrictModel):
    acceptance_weight: float = 1.0
    abandonment_weight: float = 1.0
    hard_violation_weight: float = 1.0
    continuation_discount: float = Field(default=0.8, ge=0.0, le=1.0)
    turn_cost: float = Field(default=0.02, ge=0.0)
    ask_cost: float = Field(default=0.04, ge=0.0)
    repair_cost: float = Field(default=0.02, ge=0.0)
    stop_cost: float = Field(default=0.0, ge=0.0)
    pattern_prior_weight: float = 0.1


class ScoreBreakdown(StrictModel):
    satisfaction_component: float
    acceptance_component: float
    abandonment_component: float
    hard_violation_component: float
    continuation_component: float
    turn_cost_component: float
    action_cost_component: float
    pattern_prior_component: float
    total: float


class ScoredInstantiation(StrictModel):
    instantiation: PatternInstantiation
    value_estimate: ValueEstimate
    score: ScoreBreakdown


class PolicyDecision(StrictModel):
    belief_hash: str
    pattern_library_hash: str
    value_model_name: str
    cost_config_hash: str
    candidates: list[ScoredInstantiation]
    selected_instantiation_id: str
    selected_action: SystemAction
    tie_break_rule: str = "highest_total_then_lexicographic_instantiation_id"


def _normalized_entropy(values: list[str]) -> float:
    if len(values) <= 1:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    maximum = math.log(len(counts))
    return 0.0 if maximum == 0.0 else entropy / maximum


class PatternActionInstantiator:
    """Instantiate actions exclusively from applicable Thought Patterns."""

    def __init__(
        self,
        patterns: ThoughtPatternLibrary,
        candidate_pool: CandidatePool,
        slate_size: int = 2,
    ) -> None:
        if slate_size <= 0:
            raise ValueError("slate_size must be positive")
        self.patterns = patterns
        self.candidate_pool = candidate_pool
        self.slate_size = slate_size

    def instantiate(self, state: BeliefState) -> list[PatternInstantiation]:
        actual_count = len(
            public_posterior_items(
                state.known_constraints,
                self.candidate_pool.remaining_items(set(state.rejected_item_ids)),
            )
        )
        if state.available_candidate_count != actual_count:
            raise RuntimeError("belief candidate count does not match candidate pool")
        candidates: list[PatternInstantiation] = []
        seen_action_keys: set[str] = set()
        for pattern in self.patterns.applicable(state):
            actions = self._actions_for_pattern(pattern, state)
            for action in actions:
                action_key = canonical_json(
                    {
                        "action_type": action.action_type.value,
                        "item_ids": action.item_ids,
                        "question_slot": action.question_slot,
                    }
                )
                if action_key in seen_action_keys:
                    continue
                seen_action_keys.add(action_key)
                features = self._public_features(pattern, action, state)
                feature_hash = content_sha256(features)
                payload = {
                    "instantiator_version": PATTERN_INSTANTIATOR_VERSION,
                    "pattern_id": pattern.pattern_id,
                    "pattern_version": pattern.version,
                    "library_hash": self.patterns.content_hash,
                    "candidate_pool_id": self.candidate_pool.pool_id,
                    "belief_hash": content_sha256(state.model_dump(mode="json")),
                    "action": action.model_dump(mode="json"),
                    "public_features_hash": feature_hash,
                }
                candidates.append(
                    PatternInstantiation(
                        instantiation_id=hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
                        pattern_id=pattern.pattern_id,
                        pattern_version=pattern.version,
                        pattern_library_hash=self.patterns.content_hash,
                        candidate_pool_id=self.candidate_pool.pool_id,
                        action=action,
                        public_features=features,
                        public_features_hash=feature_hash,
                    )
                )
        if not candidates:
            raise LookupError("no Pattern-constrained action can be instantiated")
        return sorted(candidates, key=lambda candidate: candidate.instantiation_id)

    def _actions_for_pattern(
        self,
        pattern: StateAwareThoughtPattern,
        state: BeliefState,
    ) -> list[SystemAction]:
        action_type = pattern.preferred_action
        if action_type == SystemActionType.ASK:
            explicit = [target for target in pattern.information_targets if target in state.missing_slots]
            slots = explicit or list(state.missing_slots)
            return [
                SystemAction(
                    action_type=SystemActionType.ASK,
                    thought_pattern_id=pattern.pattern_id,
                    question_slot=slot,
                    rationale=pattern.action_template.format(slot=slot),
                )
                for slot in slots
            ]
        if action_type == SystemActionType.STOP:
            return [
                SystemAction(
                    action_type=SystemActionType.STOP,
                    thought_pattern_id=pattern.pattern_id,
                    rationale=pattern.action_template,
                )
            ]
        if action_type not in {SystemActionType.RECOMMEND, SystemActionType.REPAIR}:
            raise NotImplementedError(f"unsupported Pattern macro-action: {action_type}")
        ranked = rank_catalog_from_belief(state, self.candidate_pool, set(state.rejected_item_ids))
        if not ranked:
            return []
        return [
            SystemAction(
                action_type=action_type,
                thought_pattern_id=pattern.pattern_id,
                item_ids=[item.item_id for item in ranked[: self.slate_size]],
                rationale=pattern.action_template,
            )
        ]

    def _public_features(
        self,
        pattern: StateAwareThoughtPattern,
        action: SystemAction,
        state: BeliefState,
    ) -> dict[str, PublicFeatureValue]:
        features: dict[str, PublicFeatureValue] = {
            "turn_index": state.turn_index,
            "phase": state.phase.value,
            "readiness": state.readiness,
            "uncertainty": state.uncertainty,
            "satisfaction_estimate": state.satisfaction_estimate,
            "patience_estimate": state.patience_estimate,
            "missing_slot_count": len(state.missing_slots),
            "remaining_candidate_count": state.available_candidate_count,
            "rejected_item_count": len(state.rejected_item_ids),
            "pattern_expected_utility": pattern.expected_utility,
            "action_type": action.action_type.value,
        }
        if action.question_slot is not None:
            serialized_values: list[str] = []
            numeric_values: list[float] = []
            missing_count = 0
            posterior = public_posterior_items(
                state.known_constraints,
                self.candidate_pool.remaining_items(set(state.rejected_item_ids)),
            )
            for item in posterior:
                value = item.attributes.get(action.question_slot)
                if value is None:
                    missing_count += 1
                elif isinstance(value, list):
                    serialized_values.append("|".join(sorted(str(part).casefold() for part in value)))
                    numeric_values.extend(
                        float(part)
                        for part in value
                        if isinstance(part, (int, float)) and not isinstance(part, bool)
                    )
                else:
                    serialized_values.append(str(value).casefold())
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        numeric_values.append(float(value))
            total = len(posterior)
            features["question_slot"] = action.question_slot
            features["slot_missing_rate"] = missing_count / total if total else 1.0
            if numeric_values and len(numeric_values) == len(serialized_values):
                ordered = sorted(numeric_values)
                lower = ordered[(len(ordered) - 1) // 4]
                upper = ordered[(3 * (len(ordered) - 1)) // 4]
                span = ordered[-1] - ordered[0]
                features["numeric_coverage"] = len(numeric_values) / total if total else 0.0
                features["normalized_iqr"] = (upper - lower) / span if span > 0 else 0.0
                features["candidate_reduction_if_revealed"] = 1.0 - (
                    max(Counter(round(value, 2) for value in numeric_values).values()) / len(numeric_values)
                )
            else:
                features["slot_value_entropy"] = _normalized_entropy(serialized_values)
        if action.item_ids:
            features["slate_size"] = len(action.item_ids)
        return features


class HeuristicPublicValueModel:
    """Zero-training baseline used to validate the cost-aware interfaces."""

    name = "heuristic_public_v1"

    def predict(
        self,
        belief: BeliefState,
        candidates: list[PatternInstantiation],
    ) -> list[ValueEstimate]:
        estimates: list[ValueEstimate] = []
        base_abandonment = min(1.0, 1.0 / max(1, belief.patience_estimate + 1))
        for candidate in candidates:
            action_type = candidate.action.action_type
            if action_type == SystemActionType.ASK:
                information_value = max(
                    float(candidate.public_features.get("slot_value_entropy", 0.0)),
                    float(candidate.public_features.get("candidate_reduction_if_revealed", 0.0)),
                )
                estimates.append(
                    ValueEstimate(
                        satisfaction_delta=-0.02,
                        accept_probability=0.0,
                        abandon_probability=min(1.0, base_abandonment + 0.05),
                        hard_violation_probability=0.0,
                        continuation_value=belief.uncertainty * (0.5 + 0.5 * information_value),
                    )
                )
            elif action_type in {SystemActionType.RECOMMEND, SystemActionType.REPAIR}:
                repair_bonus = 0.15 if action_type == SystemActionType.REPAIR else 0.0
                accept_probability = min(1.0, belief.readiness + repair_bonus)
                violation_probability = belief.uncertainty * (0.5 if repair_bonus else 1.0)
                estimates.append(
                    ValueEstimate(
                        satisfaction_delta=0.2 * accept_probability - 0.1 * violation_probability,
                        accept_probability=accept_probability,
                        abandon_probability=base_abandonment,
                        hard_violation_probability=violation_probability,
                        continuation_value=0.05,
                    )
                )
            else:
                estimates.append(
                    ValueEstimate(
                        satisfaction_delta=-0.05,
                        accept_probability=0.0,
                        abandon_probability=0.0,
                        hard_violation_probability=0.0,
                        continuation_value=0.0,
                    )
                )
        return estimates


class NetUtilityScorer:
    def __init__(self, config: CostConfig) -> None:
        self.config = config

    def score(self, candidate: PatternInstantiation, estimate: ValueEstimate) -> ScoreBreakdown:
        action_cost = {
            SystemActionType.ASK: self.config.ask_cost,
            SystemActionType.REPAIR: self.config.repair_cost,
            SystemActionType.STOP: self.config.stop_cost,
            SystemActionType.RECOMMEND: 0.0,
            SystemActionType.EXPLAIN: 0.0,
        }[candidate.action.action_type]
        satisfaction = estimate.satisfaction_delta
        acceptance = self.config.acceptance_weight * estimate.accept_probability
        abandonment = -self.config.abandonment_weight * estimate.abandon_probability
        hard_violation = -self.config.hard_violation_weight * estimate.hard_violation_probability
        continuation = self.config.continuation_discount * estimate.continuation_value
        turn_cost = -self.config.turn_cost
        action_cost_component = -action_cost
        pattern_prior = self.config.pattern_prior_weight * float(
            candidate.public_features["pattern_expected_utility"]
        )
        total = sum(
            [
                satisfaction,
                acceptance,
                abandonment,
                hard_violation,
                continuation,
                turn_cost,
                action_cost_component,
                pattern_prior,
            ]
        )
        return ScoreBreakdown(
            satisfaction_component=satisfaction,
            acceptance_component=acceptance,
            abandonment_component=abandonment,
            hard_violation_component=hard_violation,
            continuation_component=continuation,
            turn_cost_component=turn_cost,
            action_cost_component=action_cost_component,
            pattern_prior_component=pattern_prior,
            total=total,
        )


class CostAwarePatternPolicy(PatternConditionedPolicy):
    """Select only Pattern-instantiated actions using a public-state value model."""

    def __init__(
        self,
        patterns: ThoughtPatternLibrary,
        candidate_pool: CandidatePool,
        *,
        value_model: ActionValueModel | None = None,
        cost_config: CostConfig | None = None,
        slate_size: int = 2,
    ) -> None:
        super().__init__(patterns, candidate_pool, slate_size)
        self.value_model = value_model or HeuristicPublicValueModel()
        self.cost_config = cost_config or CostConfig()
        self.instantiator = PatternActionInstantiator(patterns, candidate_pool, slate_size)
        self.scorer = NetUtilityScorer(self.cost_config)
        self.last_decision: PolicyDecision | None = None

    def choose_decision(self, state: BeliefState) -> PolicyDecision:
        candidates = self.instantiator.instantiate(state)
        estimates = self.value_model.predict(state, candidates)
        if len(estimates) != len(candidates):
            raise ValueError("value model must return one estimate per Pattern instantiation")
        scored = [
            ScoredInstantiation(
                instantiation=candidate,
                value_estimate=estimate,
                score=self.scorer.score(candidate, estimate),
            )
            for candidate, estimate in zip(candidates, estimates, strict=True)
        ]
        ranked = sorted(
            scored,
            key=lambda item: (-item.score.total, item.instantiation.instantiation_id),
        )
        selected = ranked[0]
        decision = PolicyDecision(
            belief_hash=content_sha256(state.model_dump(mode="json")),
            pattern_library_hash=self.patterns.content_hash,
            value_model_name=self.value_model.name,
            cost_config_hash=content_sha256(self.cost_config.model_dump(mode="json")),
            candidates=scored,
            selected_instantiation_id=selected.instantiation.instantiation_id,
            selected_action=selected.instantiation.action,
        )
        self.last_decision = decision
        return decision

    def choose_action(self, state: BeliefState) -> SystemAction:
        return self.choose_decision(state).selected_action
