"""Scripted-policy sanity checks for the zero-API symbolic user benchmark."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field

from taira_mt.benchmark.construction import (
    PilotCandidatePoolRecord,
    SymbolicPilotManifest,
)
from taira_mt.benchmark.development import (
    CATALOG_SURFACE_DEVELOPMENT_BENCHMARK_VERSION,
    DEVELOPMENT_BENCHMARK_VERSION,
    JOINT_CONFLICT_DEVELOPMENT_BENCHMARK_VERSION,
    LEGACY_DEVELOPMENT_BENCHMARK_VERSION,
    MULTISTAGE_DEVELOPMENT_BENCHMARK_VERSION,
    DevelopmentBenchmarkManifest,
)
from taira_mt.benchmark.environment import DeterministicEvolvingUserEnvironment
from taira_mt.benchmark.interaction_quality import derive_interaction_quality
from taira_mt.benchmark.schemas import (
    BenchmarkActionType,
    BenchmarkCasePair,
    BenchmarkSystemAction,
    CandidateFeasibility,
    EvidenceReference,
    IntentCompleteness,
    PrivateBenchmarkCase,
    PublicBenchmarkCase,
    RelaxationProposal,
    ScenarioFamily,
    TerminationKind,
)
from taira_mt.models import ConstraintOperator, ConstraintPredicate, StrictModel
from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.recommender.catalog import CandidatePool

DIAGNOSTIC_VERSION = "user_symbolic_diagnostics_v3_preference_grounded_explanation"


class DiagnosticPolicyName(str, Enum):
    ALWAYS_ASK = "always_ask"
    ALWAYS_RECOMMEND = "always_recommend"
    DETERMINISTIC_RANDOM = "deterministic_random"
    PUBLIC_HEURISTIC = "public_heuristic"
    PRIVATE_ORACLE = "private_oracle_diagnostic"


class PublicDialogueContext(StrictModel):
    case: PublicBenchmarkCase
    user_utterances: list[str] = Field(default_factory=list)
    actions: list[BenchmarkSystemAction] = Field(default_factory=list)

    @property
    def transcript_text(self) -> str:
        return " ".join([self.case.initial_query, *self.user_utterances]).casefold()


class DiagnosticRunRecord(StrictModel):
    episode_id: str
    case_id: str
    domain: str
    scenario_family: ScenarioFamily
    policy: DiagnosticPolicyName
    policy_uses_private_state: bool
    success: bool
    termination_kind: TerminationKind
    turn_count: int = Field(ge=0)
    ask_count: int = Field(ge=0)
    recommend_count: int = Field(ge=0)
    explain_count: int = Field(ge=0)
    relaxation_count: int = Field(ge=0)
    stop_count: int = Field(ge=0)
    invalid_action_count: int = Field(ge=0)
    hard_violation_turn_count: int = Field(ge=0)
    informative_ask_count: int = Field(default=0, ge=0)
    noninformative_ask_count: int = Field(default=0, ge=0)
    repeated_ask_count: int = Field(default=0, ge=0)
    irrelevant_ask_count: int = Field(default=0, ge=0)
    rejected_recommendation_count: int = Field(default=0, ge=0)
    premature_recommendation_count: int = Field(default=0, ge=0)
    ignored_explanation_request_count: int = Field(default=0, ge=0)
    explanation_pending_turn_count: int = Field(default=0, ge=0)
    unverifiable_item_count: int = Field(default=0, ge=0)
    final_satisfaction: float = Field(default=0.5, ge=0.0, le=1.0)
    total_reward: float
    trace_sha256: str = Field(min_length=64, max_length=64)


class PolicyAggregate(StrictModel):
    policy: DiagnosticPolicyName
    episode_count: int = Field(ge=1)
    success_rate: float = Field(ge=0.0, le=1.0)
    abandonment_rate: float = Field(ge=0.0, le=1.0)
    graceful_no_solution_rate: float = Field(ge=0.0, le=1.0)
    mean_turns: float = Field(ge=0.0)
    mean_questions: float = Field(ge=0.0)
    mean_reward: float
    invalid_action_rate: float = Field(ge=0.0, le=1.0)
    hard_violation_turn_rate: float = Field(ge=0.0, le=1.0)
    informative_ask_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    noninformative_ask_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    repeated_ask_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    premature_recommendation_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    ignored_explanation_request_rate: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    mean_final_satisfaction: float = Field(default=0.5, ge=0.0, le=1.0)


class DiagnosticSummary(StrictModel):
    diagnostic_version: str = DIAGNOSTIC_VERSION
    diagnostic_id: str
    parent_build_id: str
    episode_count: int = Field(ge=1)
    policy_aggregates: list[PolicyAggregate]
    policy_scenario_success: dict[str, float]
    checks: dict[str, bool]


@dataclass(frozen=True)
class LoadedPilotCase:
    pair: BenchmarkCasePair
    candidate_pool: CandidatePool


@dataclass(frozen=True)
class DiagnosticResult:
    diagnostic_id: str
    parent_manifest: SymbolicPilotManifest | DevelopmentBenchmarkManifest
    records: tuple[DiagnosticRunRecord, ...]
    summary: DiagnosticSummary


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_symbolic_pilot(
    run_dir: Path,
) -> tuple[SymbolicPilotManifest | DevelopmentBenchmarkManifest, tuple[LoadedPilotCase, ...]]:
    run_dir = run_dir.resolve()
    manifest_payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest_payload.get("schema_version") == "user_symbolic_pilot_v1":
        manifest: SymbolicPilotManifest | DevelopmentBenchmarkManifest = (
            SymbolicPilotManifest.model_validate(manifest_payload)
        )
    elif manifest_payload.get("schema_version") in {
        LEGACY_DEVELOPMENT_BENCHMARK_VERSION,
        JOINT_CONFLICT_DEVELOPMENT_BENCHMARK_VERSION,
        MULTISTAGE_DEVELOPMENT_BENCHMARK_VERSION,
        CATALOG_SURFACE_DEVELOPMENT_BENCHMARK_VERSION,
        DEVELOPMENT_BENCHMARK_VERSION,
    }:
        manifest = DevelopmentBenchmarkManifest.model_validate(manifest_payload)
    else:
        raise ValueError("unsupported symbolic benchmark manifest version")
    for item in manifest.files:
        path = (run_dir / item.relative_path).resolve()
        if run_dir not in path.parents:
            raise ValueError("symbolic benchmark manifest path escapes its artifact")
        if not path.is_file() or _file_sha256(path) != item.sha256:
            raise ValueError(f"symbolic benchmark file hash mismatch: {item.relative_path}")
        if path.stat().st_size != item.byte_size:
            raise ValueError(f"symbolic benchmark file size mismatch: {item.relative_path}")
    public = {
        item["case_id"]: PublicBenchmarkCase.model_validate(item)
        for item in _load_jsonl(run_dir / "cases.public.jsonl")
    }
    private = {
        item["case_id"]: PrivateBenchmarkCase.model_validate(item)
        for item in _load_jsonl(run_dir / "cases.private.jsonl")
    }
    pools = {
        item["case_id"]: PilotCandidatePoolRecord.model_validate(item)
        for item in _load_jsonl(run_dir / "candidate_pools.public.jsonl")
    }
    if set(public) != set(private) or set(public) != set(pools):
        raise ValueError("pilot public/private/pool case IDs differ")
    cases = tuple(
        LoadedPilotCase(
            pair=BenchmarkCasePair(public=public[case_id], private=private[case_id]),
            candidate_pool=pools[case_id].materialize(),
        )
        for case_id in sorted(public)
    )
    if content_sha256([item.pair.public.model_dump(mode="json") for item in cases]) != (
        manifest.public_cases_sha256
    ):
        raise ValueError("pilot public cases differ from manifest")
    if content_sha256([item.pair.private.model_dump(mode="json") for item in cases]) != (
        manifest.private_cases_sha256
    ):
        raise ValueError("pilot private cases differ from manifest")
    if isinstance(manifest, DevelopmentBenchmarkManifest):
        evidence = _load_jsonl(run_dir / "case_evidence.private.jsonl")
        if content_sha256(evidence) != manifest.case_evidence_sha256:
            raise ValueError("development case evidence differs from manifest")
        if len(evidence) != len(public) or {item["case_id"] for item in evidence} != set(public):
            raise ValueError("development case evidence does not cover every case exactly")
    return manifest, cases


def _untried_items(context: PublicDialogueContext, pool: CandidatePool) -> list[str]:
    tried = {item_id for action in context.actions for item_id in action.item_ids}
    return [item_id for item_id in pool.item_ids if item_id not in tried]


def _always_ask(context: PublicDialogueContext) -> BenchmarkSystemAction:
    targets = (
        "occasion_interpretation",
        "specific_category",
        "price",
        "rating",
        "popularity_percentile",
    )
    target = targets[len(context.actions) % len(targets)]
    return BenchmarkSystemAction(
        action_type=BenchmarkActionType.ASK,
        question_target=target,
        utterance=f"Could you clarify your preference about {target}?",
    )


def _always_recommend(context: PublicDialogueContext, pool: CandidatePool) -> BenchmarkSystemAction:
    remaining = _untried_items(context, pool)
    if not remaining:
        return BenchmarkSystemAction(
            action_type=BenchmarkActionType.STOP,
            utterance="I do not have any unseen options left.",
        )
    return BenchmarkSystemAction(
        action_type=BenchmarkActionType.RECOMMEND,
        item_ids=remaining[:2],
        utterance="Here are two options from the available catalog.",
    )


def _deterministic_random(
    context: PublicDialogueContext,
    pool: CandidatePool,
    root_seed: int,
) -> BenchmarkSystemAction:
    seed = int(
        hashlib.sha256(
            f"{root_seed}|{context.case.case_id}|{len(context.actions)}".encode()
        ).hexdigest()[:16],
        16,
    )
    rng = random.Random(seed)
    if rng.random() < 0.45:
        targets = [
            "occasion_interpretation",
            "specific_category",
            "price",
            "rating",
            "popularity_percentile",
        ]
        target = targets[rng.randrange(len(targets))]
        return BenchmarkSystemAction(
            action_type=BenchmarkActionType.ASK,
            question_target=target,
            utterance=f"Could you say more about {target}?",
        )
    return _always_recommend(context, pool)


def _mentioned_specific_tag(context: PublicDialogueContext, pool: CandidatePool) -> str | None:
    transcript = context.transcript_text
    tags = {
        str(tag)
        for item in pool.remaining_items()
        for tag in (item.attributes.get("specific_category") or [])
    }
    mentioned = [tag for tag in tags if tag.casefold() in transcript]
    return max(mentioned, key=lambda value: (len(value), value), default=None)


def _public_ranked_items(context: PublicDialogueContext, pool: CandidatePool) -> list[str]:
    transcript = context.transcript_text
    tag = _mentioned_specific_tag(context, pool)
    budget_matches = re.findall(r"(?:at most|under)\s*\$\s*(\d+(?:\.\d+)?)", transcript)
    budget = float(budget_matches[-1]) if budget_matches else None
    rating_matches = re.findall(r"rating (?:of )?at least\s*(\d+(?:\.\d+)?)", transcript)
    minimum_rating = float(rating_matches[-1]) if rating_matches else None
    popularity_matches = re.findall(
        r"top\s*(\d+(?:\.\d+)?)%\s*by popularity",
        transcript,
    )
    minimum_popularity = (
        1.0 - float(popularity_matches[-1]) / 100.0
        if popularity_matches
        else None
    )
    tried = {item_id for action in context.actions for item_id in action.item_ids}

    def score(item_id: str) -> tuple[float, str]:
        item = pool.index.get(item_id)
        value = 0.0
        tags = item.attributes.get("specific_category") or []
        if tag is not None and tag in tags:
            value += 10.0
        price = item.attributes.get("price")
        if budget is not None:
            if isinstance(price, (int, float)) and price <= budget:
                value += 4.0
            else:
                value -= 4.0
        rating = item.attributes.get("rating")
        if minimum_rating is not None:
            if isinstance(rating, (int, float)) and rating >= minimum_rating:
                value += 4.0
            else:
                value -= 4.0
        popularity = item.attributes.get("popularity_percentile")
        if minimum_popularity is not None:
            if isinstance(popularity, (int, float)) and popularity >= minimum_popularity:
                value += 4.0
            else:
                value -= 4.0
        title_tokens = set(re.findall(r"\w+", item.title.casefold()))
        value += 0.1 * len(title_tokens.intersection(set(re.findall(r"\w+", transcript))))
        return (-value, item_id)

    return sorted((item_id for item_id in pool.item_ids if item_id not in tried), key=score)


def _public_relaxation(context: PublicDialogueContext, pool: CandidatePool) -> BenchmarkSystemAction | None:
    tag = _mentioned_specific_tag(context, pool)
    if tag is None:
        return None
    values = sorted(
        float(value)
        for item in pool.remaining_items()
        if tag in (item.attributes.get("specific_category") or [])
        and isinstance((value := item.attributes.get("price")), (int, float))
        and not isinstance(value, bool)
    )
    if len(values) < 4:
        return None
    thresholds = sorted(set(values))
    candidates = [
        (threshold, sum(value <= threshold for value in values))
        for threshold in thresholds
        if 2 <= sum(value <= threshold for value in values) < len(values)
    ]
    if not candidates:
        return None
    threshold, _ = min(
        candidates,
        key=lambda pair: (abs(pair[1] / len(values) - 0.5), pair[0]),
    )
    return BenchmarkSystemAction(
        action_type=BenchmarkActionType.PROPOSE_RELAX,
        relaxation_proposal=RelaxationProposal(
            constraint_id="c-tradeoff",
            proposed_predicate=ConstraintPredicate(
                operator=ConstraintOperator.LTE,
                value=threshold,
            ),
        ),
        utterance=f"Would you be willing to relax the budget to ${threshold:g}?",
    )


def _public_heuristic(context: PublicDialogueContext, pool: CandidatePool) -> BenchmarkSystemAction:
    transcript = context.transcript_text
    asked_targets = {action.question_target for action in context.actions if action.question_target}
    if not context.actions:
        tradeoff_target = None
        if re.search(r"(?:at most|under)\s*\$", transcript):
            tradeoff_target = "price"
        elif "rating of at least" in transcript:
            tradeoff_target = "rating"
        elif "by popularity" in transcript:
            tradeoff_target = "popularity_percentile"
        if tradeoff_target is not None:
            return BenchmarkSystemAction(
                action_type=BenchmarkActionType.ASK,
                question_target=tradeoff_target,
                utterance=(
                    "That requirement may leave no feasible option. "
                    f"How flexible can you be about {tradeoff_target}?"
                ),
            )
    if "several subtypes" in context.case.initial_query.casefold() and "occasion_interpretation" not in asked_targets:
        return BenchmarkSystemAction(
            action_type=BenchmarkActionType.ASK,
            question_target="occasion_interpretation",
            utterance="Which interpretation or specific use do you have in mind?",
        )
    if "narrowing down" in context.case.initial_query.casefold() and "specific_category" not in asked_targets:
        return BenchmarkSystemAction(
            action_type=BenchmarkActionType.ASK,
            question_target="specific_category",
            utterance="Which specific type would suit you best?",
        )
    ranked = _public_ranked_items(context, pool)
    if not ranked:
        return BenchmarkSystemAction(
            action_type=BenchmarkActionType.STOP,
            utterance="I cannot find another suitable item in this candidate set.",
        )
    return BenchmarkSystemAction(
        action_type=BenchmarkActionType.RECOMMEND,
        item_ids=ranked[:2],
        utterance="Based on the details you have shared, these are the closest available options.",
    )


def _private_oracle_action(environment: DeterministicEvolvingUserEnvironment) -> BenchmarkSystemAction:
    state = environment.runtime
    private = environment.case.private.initial_state
    if state.pending_acceptable_item_id is not None:
        item_id = state.pending_acceptable_item_id
        evidence_fields = environment.required_explanation_fields(item_id)
        return BenchmarkSystemAction(
            action_type=BenchmarkActionType.EXPLAIN,
            item_ids=[item_id],
            evidence_refs=[
                EvidenceReference(item_id=item_id, catalog_field=field)
                for field in evidence_fields
            ],
            utterance=(
                "This option matches the preferences you shared according to its "
                + ", ".join(evidence_fields)
                + "."
            ),
        )
    if state.feasibility == CandidateFeasibility.EMPTY:
        event = next(
            (
                item
                for item in private.evolution_plan.events
                if item.trigger.value == "allowed_relaxation"
                and item.from_goal_version == state.goal_version
                and item.event_id not in state.fired_event_ids
            ),
            None,
        )
        if event is None or event.trigger_ref is None:
            return BenchmarkSystemAction(
                action_type=BenchmarkActionType.STOP,
                utterance="There is no compatible item under the current requirements.",
            )
        target = private.latent_goal.version(event.to_goal_version)
        constraint = next(
            item for item in target.constraints if item.constraint_id == event.trigger_ref
        )
        return BenchmarkSystemAction(
            action_type=BenchmarkActionType.PROPOSE_RELAX,
            relaxation_proposal=RelaxationProposal(
                constraint_id=constraint.constraint_id,
                proposed_predicate=constraint.predicate,
            ),
            utterance="Would you accept the registered feasible trade-off?",
        )
    if state.completeness == IntentCompleteness.AMBIGUOUS:
        event = next(
            item
            for item in private.evolution_plan.events
            if item.trigger.value == "ask_interpretation"
            and item.event_id not in state.fired_event_ids
        )
        assert event.trigger_ref is not None
        return BenchmarkSystemAction(
            action_type=BenchmarkActionType.ASK,
            question_target=event.trigger_ref,
            utterance="Which interpretation do you have in mind?",
        )
    if state.completeness == IntentCompleteness.INCOMPLETE:
        active = private.latent_goal.version(state.goal_version)
        constraint = next(
            item
            for item in active.constraints
            if item.constraint_id not in state.disclosed_constraint_ids
            and item.predicate.importance.value == "hard"
        )
        return BenchmarkSystemAction(
            action_type=BenchmarkActionType.ASK,
            question_target=constraint.slot,
            utterance=f"Could you clarify {constraint.slot}?",
        )
    goal = private.latent_goal.version(state.goal_version)
    candidates = [
        item_id
        for item_id in goal.acceptable_item_ids
        if item_id not in state.rejected_item_ids
    ]
    if not candidates:
        return BenchmarkSystemAction(
            action_type=BenchmarkActionType.STOP,
            utterance="There is no unseen compatible item left.",
        )
    return BenchmarkSystemAction(
        action_type=BenchmarkActionType.RECOMMEND,
        item_ids=candidates[:2],
        utterance="These options satisfy the complete evaluator-only goal.",
    )


def private_oracle_diagnostic_action(
    environment: DeterministicEvolvingUserEnvironment,
) -> BenchmarkSystemAction:
    """Return the evaluator-private expert action for offline supervision only.

    The action is intentionally kept out of Agent observations.  Dataset
    builders may use it to establish solvability, state targets, and corrected
    continuations after an isolated counterfactual branch.
    """

    return _private_oracle_action(environment)


def _public_action(
    policy: DiagnosticPolicyName,
    context: PublicDialogueContext,
    pool: CandidatePool,
    root_seed: int,
) -> BenchmarkSystemAction:
    if policy == DiagnosticPolicyName.ALWAYS_ASK:
        return _always_ask(context)
    if policy == DiagnosticPolicyName.ALWAYS_RECOMMEND:
        return _always_recommend(context, pool)
    if policy == DiagnosticPolicyName.DETERMINISTIC_RANDOM:
        return _deterministic_random(context, pool, root_seed)
    if policy == DiagnosticPolicyName.PUBLIC_HEURISTIC:
        return _public_heuristic(context, pool)
    raise ValueError(f"policy is not public-only: {policy}")


def diagnostic_public_action(
    policy: DiagnosticPolicyName,
    context: PublicDialogueContext,
    pool: CandidatePool,
    *,
    root_seed: int,
) -> BenchmarkSystemAction:
    """Expose the fixed public-only diagnostic policies for data collection."""

    return _public_action(policy, context, pool, root_seed)


def _run_one(
    loaded: LoadedPilotCase,
    policy: DiagnosticPolicyName,
    *,
    root_seed: int,
    max_turns: int,
) -> DiagnosticRunRecord:
    environment = DeterministicEvolvingUserEnvironment(loaded.pair, loaded.candidate_pool)
    context = PublicDialogueContext(case=loaded.pair.public)
    while not environment.terminal and len(context.actions) < max_turns:
        if policy == DiagnosticPolicyName.PRIVATE_ORACLE:
            action = _private_oracle_action(environment)
        else:
            action = _public_action(policy, context, loaded.candidate_pool, root_seed)
        response = environment.step(action)
        context.actions.append(action)
        context.user_utterances.append(response.utterance)
    if not environment.terminal:
        environment.step(
            BenchmarkSystemAction(
                action_type=BenchmarkActionType.STOP,
                utterance="The diagnostic turn limit has been reached.",
            )
        )
    trace = environment.episode_trace(
        f"diagnostic-{content_sha256({'case': loaded.pair.public.case_id, 'policy': policy.value})[:24]}"
    )
    quality = derive_interaction_quality(trace)
    action_counts = Counter(turn.system_action.action_type for turn in trace.turns)
    hard_violations = sum(
        bool(turn.oracle_assessments)
        and all(assessment.violated_hard_slots for assessment in turn.oracle_assessments)
        for turn in trace.turns
    )
    return DiagnosticRunRecord(
        episode_id=trace.episode_id,
        case_id=trace.case_id,
        domain=loaded.pair.public.domain,
        scenario_family=loaded.pair.private.initial_state.scenario_family,
        policy=policy,
        policy_uses_private_state=policy == DiagnosticPolicyName.PRIVATE_ORACLE,
        success=trace.success,
        termination_kind=trace.termination_kind,
        turn_count=len(trace.turns),
        ask_count=action_counts[BenchmarkActionType.ASK],
        recommend_count=action_counts[BenchmarkActionType.RECOMMEND],
        explain_count=action_counts[BenchmarkActionType.EXPLAIN],
        relaxation_count=action_counts[BenchmarkActionType.PROPOSE_RELAX],
        stop_count=action_counts[BenchmarkActionType.STOP],
        invalid_action_count=quality.invalid_action_count,
        hard_violation_turn_count=hard_violations,
        informative_ask_count=quality.informative_ask_count,
        noninformative_ask_count=quality.noninformative_ask_count,
        repeated_ask_count=quality.repeated_ask_count,
        irrelevant_ask_count=quality.irrelevant_ask_count,
        rejected_recommendation_count=quality.rejected_recommendation_count,
        premature_recommendation_count=quality.premature_recommendation_count,
        ignored_explanation_request_count=quality.ignored_explanation_request_count,
        explanation_pending_turn_count=quality.explanation_pending_turn_count,
        unverifiable_item_count=quality.unverifiable_item_count,
        final_satisfaction=quality.final_satisfaction,
        total_reward=sum(turn.reward.total for turn in trace.turns),
        trace_sha256=content_sha256(trace.model_dump(mode="json")),
    )


def _aggregate(policy: DiagnosticPolicyName, records: list[DiagnosticRunRecord]) -> PolicyAggregate:
    count = len(records)
    total_turns = sum(item.turn_count for item in records)
    total_recommend_turns = sum(item.recommend_count for item in records)
    total_questions = sum(item.ask_count for item in records)
    total_pending_explanation = sum(
        item.explanation_pending_turn_count for item in records
    )
    return PolicyAggregate(
        policy=policy,
        episode_count=count,
        success_rate=sum(item.success for item in records) / count,
        abandonment_rate=sum(item.termination_kind == TerminationKind.ABANDONED for item in records) / count,
        graceful_no_solution_rate=sum(
            item.termination_kind == TerminationKind.GRACEFUL_NO_SOLUTION for item in records
        )
        / count,
        mean_turns=total_turns / count,
        mean_questions=sum(item.ask_count for item in records) / count,
        mean_reward=sum(item.total_reward for item in records) / count,
        invalid_action_rate=sum(item.invalid_action_count for item in records) / max(total_turns, 1),
        hard_violation_turn_rate=sum(item.hard_violation_turn_count for item in records)
        / max(total_recommend_turns, 1),
        informative_ask_rate=(
            sum(item.informative_ask_count for item in records) / total_questions
            if total_questions
            else None
        ),
        noninformative_ask_rate=(
            sum(item.noninformative_ask_count for item in records) / total_questions
            if total_questions
            else None
        ),
        repeated_ask_rate=(
            sum(item.repeated_ask_count for item in records) / total_questions
            if total_questions
            else None
        ),
        premature_recommendation_rate=(
            sum(item.premature_recommendation_count for item in records)
            / total_recommend_turns
            if total_recommend_turns
            else None
        ),
        ignored_explanation_request_rate=(
            sum(item.ignored_explanation_request_count for item in records)
            / total_pending_explanation
            if total_pending_explanation
            else None
        ),
        mean_final_satisfaction=sum(item.final_satisfaction for item in records) / count,
    )


def run_symbolic_diagnostics(
    pilot_dir: Path,
    *,
    root_seed: int = 20260721,
    max_turns: int = 8,
) -> DiagnosticResult:
    manifest, cases = load_symbolic_pilot(pilot_dir)
    policies = tuple(DiagnosticPolicyName)
    records = [
        _run_one(case, policy, root_seed=root_seed, max_turns=max_turns)
        for policy in policies
        for case in cases
    ]
    grouped: dict[DiagnosticPolicyName, list[DiagnosticRunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.policy].append(record)
    aggregates = [_aggregate(policy, grouped[policy]) for policy in policies]
    scenario_success: dict[str, float] = {}
    for policy in policies:
        for scenario in ScenarioFamily:
            subset = [item for item in grouped[policy] if item.scenario_family == scenario]
            if subset:
                scenario_success[f"{policy.value}:{scenario.value}"] = sum(
                    item.success for item in subset
                ) / len(subset)
    aggregate_map = {item.policy: item for item in aggregates}
    checks = {
        "private_oracle_solves_all_cases": aggregate_map[
            DiagnosticPolicyName.PRIVATE_ORACLE
        ].success_rate
        == 1.0,
        "always_ask_is_not_sufficient": aggregate_map[DiagnosticPolicyName.ALWAYS_ASK].success_rate < 0.5,
        "always_recommend_is_not_sufficient": aggregate_map[
            DiagnosticPolicyName.ALWAYS_RECOMMEND
        ].success_rate
        < 0.9,
        "public_heuristic_beats_random": aggregate_map[DiagnosticPolicyName.PUBLIC_HEURISTIC].success_rate
        > aggregate_map[DiagnosticPolicyName.DETERMINISTIC_RANDOM].success_rate,
        "conflict_requires_distinct_behavior": scenario_success.get(
            f"{DiagnosticPolicyName.ALWAYS_RECOMMEND.value}:{ScenarioFamily.CONFLICT.value}",
            1.0,
        )
        < scenario_success.get(
            f"{DiagnosticPolicyName.PRIVATE_ORACLE.value}:{ScenarioFamily.CONFLICT.value}",
            0.0,
        ),
    }
    identity = {
        "version": DIAGNOSTIC_VERSION,
        "parent_build_id": manifest.build_id,
        "root_seed": root_seed,
        "max_turns": max_turns,
        "record_sha256": content_sha256([record.model_dump(mode="json") for record in records]),
    }
    diagnostic_id = f"diagnostic-{content_sha256(identity)[:16]}"
    summary = DiagnosticSummary(
        diagnostic_id=diagnostic_id,
        parent_build_id=manifest.build_id,
        episode_count=len(records),
        policy_aggregates=aggregates,
        policy_scenario_success=dict(sorted(scenario_success.items())),
        checks=checks,
    )
    return DiagnosticResult(
        diagnostic_id=diagnostic_id,
        parent_manifest=manifest,
        records=tuple(records),
        summary=summary,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_diagnostic_artifact(result: DiagnosticResult, output_root: Path) -> Path:
    run_dir = output_root / result.diagnostic_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "diagnostic_runs.jsonl"
    summary_path = run_dir / "summary.json"
    records_text = "".join(
        canonical_json(record.model_dump(mode="json")) + "\n" for record in result.records
    )
    summary_text = json.dumps(result.summary.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    for path, text in ((records_path, records_text), (summary_path, summary_text)):
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different diagnostic file: {path}")
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    manifest = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "diagnostic_id": result.diagnostic_id,
        "parent_build_id": result.parent_manifest.build_id,
        "files": [
            {
                "relative_path": path.name,
                "sha256": _file_sha256(path),
                "byte_size": path.stat().st_size,
            }
            for path in (records_path, summary_path)
        ],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != manifest_text:
        raise FileExistsError(f"refusing to overwrite a different diagnostic manifest: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_text(manifest_text, encoding="utf-8")
    return run_dir
