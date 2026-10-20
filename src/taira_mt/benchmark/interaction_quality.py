"""Trace-derived interaction quality metrics independent of policy internals."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from taira_mt.benchmark.schemas import (
    BenchmarkActionType,
    BenchmarkEpisodeTrace,
    BenchmarkUserAct,
    CandidateFeasibility,
    IntentCompleteness,
    TerminationKind,
)
from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json, content_sha256

INTERACTION_QUALITY_VERSION = "interaction_quality_metrics_v4_preference_grounded_explanation"

_INFORMATIVE_ASK_GOALS = {
    "resolve_ambiguity",
    "answer_registered_tradeoff_question",
    "answer_relevant_question",
}
_NONINFORMATIVE_ASK_GOALS = {
    "dismiss_irrelevant_question",
    "remind_already_disclosed_constraint",
}
_INVALID_GOALS = {
    "recommendation_contains_out_of_pool_item",
    "explanation_contains_out_of_pool_item",
    "explanation_uses_ungrounded_catalog_evidence",
}
class InteractionQualityRecord(StrictModel):
    schema_version: str = INTERACTION_QUALITY_VERSION
    episode_id: str
    case_id: str
    success: bool
    termination_kind: TerminationKind
    turn_count: int = Field(ge=1)
    ask_count: int = Field(ge=0)
    informative_ask_count: int = Field(ge=0)
    noninformative_ask_count: int = Field(ge=0)
    repeated_ask_count: int = Field(ge=0)
    irrelevant_ask_count: int = Field(ge=0)
    recommend_count: int = Field(ge=0)
    explain_count: int = Field(default=0, ge=0)
    explanation_pending_turn_count: int = Field(default=0, ge=0)
    ignored_explanation_request_count: int = Field(default=0, ge=0)
    rejected_recommendation_count: int = Field(ge=0)
    premature_recommendation_count: int = Field(ge=0)
    recommended_item_count: int = Field(ge=0)
    hard_violating_item_count: int = Field(ge=0)
    unverifiable_item_count: int = Field(ge=0)
    all_hard_violating_slate_count: int = Field(ge=0)
    all_unverifiable_slate_count: int = Field(ge=0)
    invalid_action_count: int = Field(ge=0)
    patience_consumed: int = Field(ge=0)
    final_satisfaction: float = Field(ge=0.0, le=1.0)
    reward_v1_total: float
    reward_v1_components: dict[str, float]
    trace_sha256: str = Field(min_length=64, max_length=64)


class InteractionQualityAggregate(StrictModel):
    schema_version: str = INTERACTION_QUALITY_VERSION
    evaluation_id: str
    parent_benchmark_build_id: str
    parent_run_id: str
    policy_or_model: str
    episode_count: int = Field(ge=1)
    success_rate: float = Field(ge=0.0, le=1.0)
    abandonment_rate: float = Field(ge=0.0, le=1.0)
    mean_turns_all: float = Field(ge=0.0)
    mean_turns_success: float | None = Field(default=None, ge=0.0)
    mean_ask_count: float = Field(ge=0.0)
    mean_explain_count: float = Field(default=0.0, ge=0.0)
    informative_ask_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    noninformative_ask_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    repeated_ask_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    rejected_recommendation_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    premature_recommendation_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    hard_violating_item_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    unverifiable_item_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    all_hard_violating_slate_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    all_unverifiable_slate_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    invalid_action_turn_rate: float = Field(ge=0.0, le=1.0)
    ignored_explanation_request_rate: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    mean_reward_v1: float
    mean_final_satisfaction: float = Field(ge=0.0, le=1.0)
    termination_counts: dict[str, int]


@dataclass(frozen=True)
class InteractionQualityEvaluation:
    evaluation_id: str
    records: tuple[InteractionQualityRecord, ...]
    summary: InteractionQualityAggregate


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def derive_interaction_quality(trace: BenchmarkEpisodeTrace) -> InteractionQualityRecord:
    """Derive quality labels only from the evaluator trace after an episode ends."""

    if not trace.turns:
        raise ValueError("interaction quality requires at least one executed turn")
    ask_turns = [
        turn
        for turn in trace.turns
        if turn.system_action.action_type == BenchmarkActionType.ASK
    ]
    recommend_turns = [
        turn
        for turn in trace.turns
        if turn.system_action.action_type == BenchmarkActionType.RECOMMEND
    ]
    explain_turns = [
        turn
        for turn in trace.turns
        if turn.system_action.action_type == BenchmarkActionType.EXPLAIN
    ]
    explanation_pending_turns = [
        turn
        for turn in trace.turns
        if turn.state_before.pending_acceptable_item_id is not None
    ]
    ignored_explanation = sum(
        turn.system_action.action_type != BenchmarkActionType.EXPLAIN
        or turn.response_frame.user_act == BenchmarkUserAct.REQUEST_EXPLANATION
        for turn in explanation_pending_turns
    )
    informative = sum(
        turn.response_frame.communicative_goal in _INFORMATIVE_ASK_GOALS
        for turn in ask_turns
    )
    noninformative = sum(
        turn.response_frame.communicative_goal in _NONINFORMATIVE_ASK_GOALS
        for turn in ask_turns
    )
    repeated = sum(
        turn.system_action.question_target in set(turn.state_before.asked_question_targets)
        for turn in ask_turns
    )
    irrelevant = sum(
        turn.response_frame.communicative_goal == "dismiss_irrelevant_question"
        for turn in ask_turns
    )
    rejected = sum(
        turn.response_frame.user_act == BenchmarkUserAct.REJECT
        and turn.response_frame.communicative_goal
        in {
            "reject_unsatisfactory_recommendation",
            "reject_unverifiable_recommendation",
        }
        for turn in recommend_turns
    )
    premature = sum(
        turn.state_before.completeness != IntentCompleteness.SUFFICIENT
        or turn.state_before.feasibility == CandidateFeasibility.EMPTY
        for turn in recommend_turns
    )
    assessments = [assessment for turn in recommend_turns for assessment in turn.oracle_assessments]
    hard_items = sum(bool(item.violated_hard_slots) for item in assessments)
    unverifiable_items = sum(bool(item.unknown_hard_slots) for item in assessments)
    all_hard_slates = sum(
        bool(turn.oracle_assessments)
        and all(item.violated_hard_slots for item in turn.oracle_assessments)
        for turn in recommend_turns
    )
    all_unverifiable_slates = sum(
        bool(turn.oracle_assessments)
        and all(
            item.unknown_hard_slots and not item.violated_hard_slots
            for item in turn.oracle_assessments
        )
        for turn in recommend_turns
    )
    invalid = sum(
        turn.response_frame.communicative_goal in _INVALID_GOALS
        for turn in trace.turns
    )
    component_names = (
        "recommendation_utility",
        "acceptance_bonus",
        "question_cost",
        "invalid_action_cost",
        "dialogue_contract_cost",
        "hard_violation_cost",
        "abandonment_cost",
    )
    reward_components = {
        name: sum(float(getattr(turn.reward, name)) for turn in trace.turns)
        for name in component_names
    }
    first_patience = trace.turns[0].state_before.patience_remaining
    final_patience = trace.turns[-1].state_after.patience_remaining
    return InteractionQualityRecord(
        episode_id=trace.episode_id,
        case_id=trace.case_id,
        success=trace.success,
        termination_kind=trace.termination_kind,
        turn_count=len(trace.turns),
        ask_count=len(ask_turns),
        informative_ask_count=informative,
        noninformative_ask_count=noninformative,
        repeated_ask_count=repeated,
        irrelevant_ask_count=irrelevant,
        recommend_count=len(recommend_turns),
        explain_count=len(explain_turns),
        explanation_pending_turn_count=len(explanation_pending_turns),
        ignored_explanation_request_count=ignored_explanation,
        rejected_recommendation_count=rejected,
        premature_recommendation_count=premature,
        recommended_item_count=len(assessments),
        hard_violating_item_count=hard_items,
        unverifiable_item_count=unverifiable_items,
        all_hard_violating_slate_count=all_hard_slates,
        all_unverifiable_slate_count=all_unverifiable_slates,
        invalid_action_count=invalid,
        patience_consumed=first_patience - final_patience,
        final_satisfaction=trace.turns[-1].state_after.satisfaction,
        reward_v1_total=sum(turn.reward.total for turn in trace.turns),
        reward_v1_components=reward_components,
        trace_sha256=content_sha256(trace.model_dump(mode="json")),
    )


def aggregate_interaction_quality(
    records: tuple[InteractionQualityRecord, ...],
    *,
    parent_benchmark_build_id: str,
    parent_run_id: str,
    policy_or_model: str,
) -> InteractionQualityEvaluation:
    if not records:
        raise ValueError("interaction quality aggregation requires records")
    if len({item.case_id for item in records}) != len(records):
        raise ValueError("interaction quality records contain duplicate case IDs")
    count = len(records)
    total_turns = sum(item.turn_count for item in records)
    total_asks = sum(item.ask_count for item in records)
    total_recommends = sum(item.recommend_count for item in records)
    total_explanation_pending = sum(
        item.explanation_pending_turn_count for item in records
    )
    total_items = sum(item.recommended_item_count for item in records)
    successful = [item for item in records if item.success]
    identity = {
        "schema_version": INTERACTION_QUALITY_VERSION,
        "parent_benchmark_build_id": parent_benchmark_build_id,
        "parent_run_id": parent_run_id,
        "policy_or_model": policy_or_model,
        "records_sha256": content_sha256([item.model_dump(mode="json") for item in records]),
    }
    evaluation_id = f"interaction-quality-{content_sha256(identity)[:16]}"
    summary = InteractionQualityAggregate(
        evaluation_id=evaluation_id,
        parent_benchmark_build_id=parent_benchmark_build_id,
        parent_run_id=parent_run_id,
        policy_or_model=policy_or_model,
        episode_count=count,
        success_rate=sum(item.success for item in records) / count,
        abandonment_rate=sum(
            item.termination_kind == TerminationKind.ABANDONED for item in records
        )
        / count,
        mean_turns_all=total_turns / count,
        mean_turns_success=(
            sum(item.turn_count for item in successful) / len(successful)
            if successful
            else None
        ),
        mean_ask_count=total_asks / count,
        mean_explain_count=sum(item.explain_count for item in records) / count,
        informative_ask_rate=_safe_rate(
            sum(item.informative_ask_count for item in records), total_asks
        ),
        noninformative_ask_rate=_safe_rate(
            sum(item.noninformative_ask_count for item in records), total_asks
        ),
        repeated_ask_rate=_safe_rate(
            sum(item.repeated_ask_count for item in records), total_asks
        ),
        rejected_recommendation_rate=_safe_rate(
            sum(item.rejected_recommendation_count for item in records), total_recommends
        ),
        premature_recommendation_rate=_safe_rate(
            sum(item.premature_recommendation_count for item in records), total_recommends
        ),
        hard_violating_item_rate=_safe_rate(
            sum(item.hard_violating_item_count for item in records), total_items
        ),
        unverifiable_item_rate=_safe_rate(
            sum(item.unverifiable_item_count for item in records), total_items
        ),
        all_hard_violating_slate_rate=_safe_rate(
            sum(item.all_hard_violating_slate_count for item in records), total_recommends
        ),
        all_unverifiable_slate_rate=_safe_rate(
            sum(item.all_unverifiable_slate_count for item in records), total_recommends
        ),
        invalid_action_turn_rate=sum(item.invalid_action_count for item in records)
        / total_turns,
        ignored_explanation_request_rate=_safe_rate(
            sum(item.ignored_explanation_request_count for item in records),
            total_explanation_pending,
        ),
        mean_reward_v1=sum(item.reward_v1_total for item in records) / count,
        mean_final_satisfaction=sum(item.final_satisfaction for item in records) / count,
        termination_counts=dict(
            sorted(Counter(item.termination_kind.value for item in records).items())
        ),
    )
    return InteractionQualityEvaluation(
        evaluation_id=evaluation_id,
        records=records,
        summary=summary,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_interaction_quality_artifact(
    result: InteractionQualityEvaluation,
    output_root: Path,
    *,
    benchmark_manifest_sha256: str,
    parent_run_manifest_sha256: str,
) -> Path:
    run_dir = output_root / result.evaluation_id
    run_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "episodes.private_metrics.jsonl": "".join(
            canonical_json(item.model_dump(mode="json")) + "\n" for item in result.records
        ),
        "summary.json": json.dumps(
            result.summary.model_dump(mode="json"), ensure_ascii=False, indent=2
        )
        + "\n",
    }
    for filename, value in outputs.items():
        (run_dir / filename).write_text(value, encoding="utf-8")
    manifest = {
        "schema_version": INTERACTION_QUALITY_VERSION,
        "artifact_role": "posthoc_private_interaction_quality_evaluation",
        "evaluation_id": result.evaluation_id,
        "status": "complete",
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "parent_run_manifest_sha256": parent_run_manifest_sha256,
        "private_labels_exposed_to_policy": False,
        "outputs": [
            {
                "relative_path": filename,
                "sha256": _file_sha256(run_dir / filename),
                "byte_size": (run_dir / filename).stat().st_size,
                "line_count": value.count("\n") if filename.endswith(".jsonl") else None,
            }
            for filename, value in sorted(outputs.items())
        ],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
