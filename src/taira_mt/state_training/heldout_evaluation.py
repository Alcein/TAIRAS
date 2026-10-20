"""Paper-facing held-out evaluation for DSUS state-compiler rollouts.

This layer never trains or calls a model.  It joins frozen public prompts,
trainer-private verifiers, sample strata, and already-generated rollouts; invokes
the executable verifier chain; and materializes auditable JSON/CSV summaries.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import Field, JsonValue, field_validator, model_validator

from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.state_compiler import EvidenceReference, PublicDSUSState, reduce_state
from taira_mt.state_training.dataset_assembly import (
    DatasetSplit,
    PrivateRLVerifierRecord,
    PublicRLPromptRecord,
    SampleRegistryRecord,
)
from taira_mt.state_training.verifier_chain import (
    VERIFIER_CHAIN_VERSION,
    TVRLVRRewardConfig,
    VerifierScoreRecord,
    parse_model_patch,
    score_verifier_sample,
)

HELDOUT_EVALUATION_SPEC_VERSION = "dsus_heldout_evaluation_spec_v1"
HELDOUT_EVALUATION_VERSION = "dsus_heldout_evaluation_v1"

RecordT = TypeVar("RecordT", bound=StrictModel)

BOOTSTRAP_METRICS = (
    "schema_validity",
    "evidence_f1",
    "control_exact",
    "constraint_f1",
    "joint_state_accuracy",
    "replay_rate",
    "transition_consistency",
    "persistence",
    "accepted_rate",
    "mean_total_reward",
)

SEED_SUMMARY_METRICS = (
    "schema_validity",
    "evidence_f1_micro",
    "evidence_f1_macro",
    "control_exact",
    "constraint_f1_micro",
    "constraint_f1_macro",
    "joint_state_accuracy",
    "operation_legality",
    "revision_validity",
    "replay_rate",
    "transition_consistency",
    "persistence",
    "accepted_rate",
    "mean_total_reward",
)

STRATIFICATION_AXES = (
    "difficulty",
    "mechanism",
    "domain",
    "source_family",
    "curriculum_stage",
)


class RolloutRuntimeMetadata(StrictModel):
    """Optional measured runtime fields; absence remains null in paper outputs."""

    latency_ms: float | None = Field(default=None, ge=0.0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    json_retry_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_or_derive_total(self) -> RolloutRuntimeMetadata:
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            derived = self.prompt_tokens + self.completion_tokens
            if self.total_tokens is None:
                self.total_tokens = derived
            elif self.total_tokens != derived:
                raise ValueError("total_tokens disagrees with prompt_tokens + completion_tokens")
        return self


class HeldoutRolloutRecord(StrictModel):
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_output: str | dict[str, JsonValue]
    runtime: RolloutRuntimeMetadata | None = None


class EvaluationSeedSpec(StrictModel):
    seed_id: str = Field(min_length=1, max_length=128)
    rollout_path: str = Field(min_length=1)

    @field_validator("seed_id")
    @classmethod
    def normalize_seed_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("seed_id cannot be blank")
        return value.strip()


class EvaluationSystemSpec(StrictModel):
    system_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    display_name: str = Field(min_length=1, max_length=256)
    seeds: list[EvaluationSeedSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_seeds(self) -> EvaluationSystemSpec:
        identifiers = [seed.seed_id for seed in self.seeds]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"system {self.system_id} has duplicate seed_id values")
        return self


class PairedComparisonSpec(StrictModel):
    challenger_system_id: str
    reference_system_id: str

    @model_validator(mode="after")
    def validate_distinct_systems(self) -> PairedComparisonSpec:
        if self.challenger_system_id == self.reference_system_id:
            raise ValueError("paired comparison requires two distinct systems")
        return self


class HeldoutEvaluationSpec(StrictModel):
    schema_version: Literal[HELDOUT_EVALUATION_SPEC_VERSION] = HELDOUT_EVALUATION_SPEC_VERSION
    dataset_dir: str = Field(min_length=1)
    split: DatasetSplit = DatasetSplit.TEST
    systems: list[EvaluationSystemSpec] = Field(min_length=1)
    comparisons: list[PairedComparisonSpec] = Field(default_factory=list)
    require_complete: bool = True
    bootstrap_resamples: int = Field(default=5000, ge=100, le=100000)
    bootstrap_seed: int = 20260809
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)

    @model_validator(mode="after")
    def validate_systems_and_comparisons(self) -> HeldoutEvaluationSpec:
        system_ids = [system.system_id for system in self.systems]
        if len(system_ids) != len(set(system_ids)):
            raise ValueError("evaluation spec has duplicate system_id values")
        known = set(system_ids)
        comparison_pairs = []
        for comparison in self.comparisons:
            if comparison.challenger_system_id not in known or comparison.reference_system_id not in known:
                raise ValueError("paired comparison references an unknown system")
            pair = (comparison.challenger_system_id, comparison.reference_system_id)
            comparison_pairs.append(pair)
        if len(comparison_pairs) != len(set(comparison_pairs)):
            raise ValueError("evaluation spec has duplicate paired comparisons")
        return self


class PerExampleEvaluation(StrictModel):
    system_id: str
    seed_id: str
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    difficulty: str
    mechanism: str
    domain: str
    source_family: str
    curriculum_stage: str
    verifier: VerifierScoreRecord
    evidence_tp: int = Field(ge=0)
    evidence_fp: int = Field(ge=0)
    evidence_fn: int = Field(ge=0)
    evidence_precision: float = Field(ge=0.0, le=1.0)
    evidence_recall: float = Field(ge=0.0, le=1.0)
    evidence_f1: float = Field(ge=0.0, le=1.0)
    constraint_tp: int = Field(ge=0)
    constraint_fp: int = Field(ge=0)
    constraint_fn: int = Field(ge=0)
    constraint_precision: float = Field(ge=0.0, le=1.0)
    constraint_recall: float = Field(ge=0.0, le=1.0)
    constraint_f1: float = Field(ge=0.0, le=1.0)
    transition_consistent: bool
    runtime: RolloutRuntimeMetadata | None = None


class MetricSummary(StrictModel):
    prediction_count: int = Field(ge=1)
    unique_sample_count: int = Field(ge=1)
    schema_validity: float = Field(ge=0.0, le=1.0)
    evidence_precision_micro: float = Field(ge=0.0, le=1.0)
    evidence_recall_micro: float = Field(ge=0.0, le=1.0)
    evidence_f1_micro: float = Field(ge=0.0, le=1.0)
    evidence_f1_macro: float = Field(ge=0.0, le=1.0)
    control_exact: float = Field(ge=0.0, le=1.0)
    constraint_precision_micro: float = Field(ge=0.0, le=1.0)
    constraint_recall_micro: float = Field(ge=0.0, le=1.0)
    constraint_f1_micro: float = Field(ge=0.0, le=1.0)
    constraint_f1_macro: float = Field(ge=0.0, le=1.0)
    joint_state_accuracy: float = Field(ge=0.0, le=1.0)
    operation_legality: float = Field(ge=0.0, le=1.0)
    revision_validity: float = Field(ge=0.0, le=1.0)
    replay_rate: float = Field(ge=0.0, le=1.0)
    transition_consistency: float = Field(ge=0.0, le=1.0)
    persistence: float = Field(ge=0.0, le=1.0)
    accepted_rate: float = Field(ge=0.0, le=1.0)
    mean_total_reward: float = Field(ge=0.0, le=1.0)
    latency_coverage_count: int = Field(ge=0)
    latency_mean_ms: float | None = Field(default=None, ge=0.0)
    latency_p50_ms: float | None = Field(default=None, ge=0.0)
    latency_p95_ms: float | None = Field(default=None, ge=0.0)
    token_coverage_count: int = Field(ge=0)
    prompt_tokens_mean: float | None = Field(default=None, ge=0.0)
    completion_tokens_mean: float | None = Field(default=None, ge=0.0)
    total_tokens_mean: float | None = Field(default=None, ge=0.0)
    json_retry_coverage_count: int = Field(ge=0)
    json_retry_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    json_retry_mean: float | None = Field(default=None, ge=0.0)


class SeedEvaluationSummary(StrictModel):
    seed_id: str
    rollout_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: MetricSummary


class SystemEvaluationSummary(StrictModel):
    system_id: str
    display_name: str
    seed_count: int = Field(ge=1)
    pooled_metrics: MetricSummary
    seed_summaries: list[SeedEvaluationSummary]
    seed_metric_mean: dict[str, float | None]
    seed_metric_std: dict[str, float | None]


class StratumEvaluationSummary(StrictModel):
    system_id: str
    axis: str
    value: str
    metrics: MetricSummary


class BootstrapMetricResult(StrictModel):
    metric: str
    paired_sample_count: int = Field(ge=1)
    challenger_mean: float
    reference_mean: float
    observed_delta: float
    ci_low: float
    ci_high: float
    two_sided_p: float = Field(ge=0.0, le=1.0)
    confidence_level: float
    bootstrap_resamples: int = Field(ge=100)


class PairedComparisonResult(StrictModel):
    challenger_system_id: str
    reference_system_id: str
    metrics: list[BootstrapMetricResult]


class HeldoutEvaluationReport(StrictModel):
    schema_version: Literal[HELDOUT_EVALUATION_VERSION] = HELDOUT_EVALUATION_VERSION
    dataset_build_name: str
    split: DatasetSplit
    dataset_sample_count: int = Field(ge=1)
    systems: list[SystemEvaluationSummary]
    strata: list[StratumEvaluationSummary]
    paired_comparisons: list[PairedComparisonResult]
    source_artifacts_unchanged: bool
    external_calls: Literal[False] = False
    training_started: Literal[False] = False


class HeldoutEvaluationResult(StrictModel):
    run_id: str
    run_dir: str
    report: HeldoutEvaluationReport


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _load_models(path: Path, model_type: type[RecordT]) -> list[RecordT]:
    return [model_type.model_validate(row) for row in _read_jsonl(path)]


def _index(records: Iterable[RecordT], *, label: str) -> dict[str, RecordT]:
    output: dict[str, RecordT] = {}
    for record in records:
        sample_id = str(record.sample_id)
        if sample_id in output:
            raise ValueError(f"duplicate {label} sample_id: {sample_id}")
        output[sample_id] = record
    return output


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_idempotently(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"refusing to overwrite non-identical evaluation artifact: {path}")
    path.write_text(text, encoding="utf-8")


def _counter_overlap(
    predicted: Counter[str],
    gold: Counter[str],
) -> tuple[int, int, int, float, float, float]:
    true_positive = sum((predicted & gold).values())
    false_positive = sum(predicted.values()) - true_positive
    false_negative = sum(gold.values()) - true_positive
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else float(not gold)
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 1.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return true_positive, false_positive, false_negative, precision, recall, f1


def _model_output_json(model_output: str | Mapping[str, JsonValue]) -> object | None:
    try:
        return json.loads(model_output) if isinstance(model_output, str) else dict(model_output)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _evidence_counter_from_output(model_output: str | Mapping[str, JsonValue]) -> Counter[str]:
    """Extract citations independently of whole-patch validity for diagnostic P/R/F1."""

    value = _model_output_json(model_output)
    if not isinstance(value, dict) or not isinstance(value.get("evidence"), list):
        return Counter()
    output: Counter[str] = Counter()
    for index, item in enumerate(value["evidence"]):
        try:
            evidence = EvidenceReference.model_validate(item)
            key = canonical_json(
                {
                    "turn_id": evidence.turn_id,
                    "source": evidence.source.value,
                    "quote": evidence.quote,
                }
            )
        except (TypeError, ValueError):
            key = f"__invalid_evidence__:{index}:{content_sha256(item)}"
        output[key] += 1
    return output


def _gold_evidence_counter(private: PrivateRLVerifierRecord) -> Counter[str]:
    return Counter(
        canonical_json(
            {
                "turn_id": item.turn_id,
                "source": item.source.value,
                "quote": item.quote,
            }
        )
        for item in private.trainer_private.allowed_evidence
    )


def _constraint_counter(state: PublicDSUSState | None) -> Counter[str]:
    if state is None:
        return Counter()
    return Counter(
        canonical_json(
            {
                "name": item.name,
                "value": item.value,
                "polarity": item.polarity.value,
                "importance": item.importance.value,
                "status": item.status.value,
            }
        )
        for item in state.constraints
    )


def _candidate_next_state(
    public: PublicRLPromptRecord,
    rollout: HeldoutRolloutRecord,
) -> PublicDSUSState | None:
    try:
        predicted, _ = parse_model_patch(rollout.model_output)
        return reduce_state(public.public_prompt.previous_public_state, predicted.as_patch())
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _evaluate_one(
    *,
    system_id: str,
    seed_id: str,
    public: PublicRLPromptRecord,
    private: PrivateRLVerifierRecord,
    registry: SampleRegistryRecord,
    rollout: HeldoutRolloutRecord,
    reward_config: TVRLVRRewardConfig,
) -> PerExampleEvaluation:
    verifier = score_verifier_sample(
        public_record=public,
        private_record=private,
        rollout=rollout,
        config=reward_config,
    )
    evidence_metrics = _counter_overlap(
        _evidence_counter_from_output(rollout.model_output),
        _gold_evidence_counter(private),
    )
    candidate = _candidate_next_state(public, rollout)
    constraint_metrics = _counter_overlap(
        _constraint_counter(candidate),
        _constraint_counter(private.trainer_private.gold_next_state),
    )
    components = verifier.components
    transition_consistent = all(
        components[name] == 1.0
        for name in ("operation_legality", "revision", "replay")
    )
    return PerExampleEvaluation(
        system_id=system_id,
        seed_id=seed_id,
        sample_id=rollout.sample_id,
        difficulty=registry.difficulty.value,
        mechanism=registry.mechanism.value,
        domain=registry.domain,
        source_family=registry.source_family.value,
        curriculum_stage=registry.curriculum_stage.value,
        verifier=verifier,
        evidence_tp=evidence_metrics[0],
        evidence_fp=evidence_metrics[1],
        evidence_fn=evidence_metrics[2],
        evidence_precision=evidence_metrics[3],
        evidence_recall=evidence_metrics[4],
        evidence_f1=evidence_metrics[5],
        constraint_tp=constraint_metrics[0],
        constraint_fp=constraint_metrics[1],
        constraint_fn=constraint_metrics[2],
        constraint_precision=constraint_metrics[3],
        constraint_recall=constraint_metrics[4],
        constraint_f1=constraint_metrics[5],
        transition_consistent=transition_consistent,
        runtime=rollout.runtime,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _optional_mean(values: Sequence[float]) -> float | None:
    return _mean(values) if values else None


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _micro_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else float(fn == 0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _aggregate(rows: Sequence[PerExampleEvaluation]) -> MetricSummary:
    if not rows:
        raise ValueError("cannot aggregate an empty evaluation slice")
    evidence = _micro_metrics(
        sum(row.evidence_tp for row in rows),
        sum(row.evidence_fp for row in rows),
        sum(row.evidence_fn for row in rows),
    )
    constraints = _micro_metrics(
        sum(row.constraint_tp for row in rows),
        sum(row.constraint_fp for row in rows),
        sum(row.constraint_fn for row in rows),
    )
    latency = [row.runtime.latency_ms for row in rows if row.runtime and row.runtime.latency_ms is not None]
    prompt_tokens = [
        float(row.runtime.prompt_tokens)
        for row in rows
        if row.runtime and row.runtime.prompt_tokens is not None
    ]
    completion_tokens = [
        float(row.runtime.completion_tokens)
        for row in rows
        if row.runtime and row.runtime.completion_tokens is not None
    ]
    total_tokens = [
        float(row.runtime.total_tokens)
        for row in rows
        if row.runtime and row.runtime.total_tokens is not None
    ]
    token_coverage_count = sum(
        bool(
            row.runtime
            and any(
                value is not None
                for value in (
                    row.runtime.prompt_tokens,
                    row.runtime.completion_tokens,
                    row.runtime.total_tokens,
                )
            )
        )
        for row in rows
    )
    retries = [
        float(row.runtime.json_retry_count)
        for row in rows
        if row.runtime and row.runtime.json_retry_count is not None
    ]
    return MetricSummary(
        prediction_count=len(rows),
        unique_sample_count=len({row.sample_id for row in rows}),
        schema_validity=_mean([row.verifier.components["schema"] for row in rows]),
        evidence_precision_micro=evidence[0],
        evidence_recall_micro=evidence[1],
        evidence_f1_micro=evidence[2],
        evidence_f1_macro=_mean([row.evidence_f1 for row in rows]),
        control_exact=_mean([row.verifier.components["control_match"] for row in rows]),
        constraint_precision_micro=constraints[0],
        constraint_recall_micro=constraints[1],
        constraint_f1_micro=constraints[2],
        constraint_f1_macro=_mean([row.constraint_f1 for row in rows]),
        joint_state_accuracy=_mean([row.verifier.components["state_equivalence"] for row in rows]),
        operation_legality=_mean([row.verifier.components["operation_legality"] for row in rows]),
        revision_validity=_mean([row.verifier.components["revision"] for row in rows]),
        replay_rate=_mean([row.verifier.components["replay"] for row in rows]),
        transition_consistency=_mean([float(row.transition_consistent) for row in rows]),
        persistence=_mean([row.verifier.components["persistence"] for row in rows]),
        accepted_rate=_mean([float(row.verifier.accepted) for row in rows]),
        mean_total_reward=_mean([row.verifier.total_reward for row in rows]),
        latency_coverage_count=len(latency),
        latency_mean_ms=_optional_mean(latency),
        latency_p50_ms=_quantile(latency, 0.5),
        latency_p95_ms=_quantile(latency, 0.95),
        token_coverage_count=token_coverage_count,
        prompt_tokens_mean=_optional_mean(prompt_tokens),
        completion_tokens_mean=_optional_mean(completion_tokens),
        total_tokens_mean=_optional_mean(total_tokens),
        json_retry_coverage_count=len(retries),
        json_retry_rate=(
            _mean([float(value > 0) for value in retries]) if retries else None
        ),
        json_retry_mean=_optional_mean(retries),
    )


def _population_std(values: Sequence[float]) -> float:
    mean = _mean(values)
    return math.sqrt(_mean([(value - mean) ** 2 for value in values]))


def _system_summary(
    system: EvaluationSystemSpec,
    rows_by_seed: Mapping[str, Sequence[PerExampleEvaluation]],
    rollout_hashes: Mapping[str, str],
) -> SystemEvaluationSummary:
    seed_summaries = [
        SeedEvaluationSummary(
            seed_id=seed.seed_id,
            rollout_file_sha256=rollout_hashes[seed.seed_id],
            metrics=_aggregate(rows_by_seed[seed.seed_id]),
        )
        for seed in system.seeds
    ]
    means: dict[str, float | None] = {}
    standard_deviations: dict[str, float | None] = {}
    for metric in SEED_SUMMARY_METRICS:
        values = [getattr(seed.metrics, metric) for seed in seed_summaries]
        numeric = [float(value) for value in values if value is not None]
        means[metric] = _optional_mean(numeric)
        standard_deviations[metric] = _population_std(numeric) if len(numeric) >= 2 else None
    return SystemEvaluationSummary(
        system_id=system.system_id,
        display_name=system.display_name,
        seed_count=len(seed_summaries),
        pooled_metrics=_aggregate(
            [row for seed in system.seeds for row in rows_by_seed[seed.seed_id]]
        ),
        seed_summaries=seed_summaries,
        seed_metric_mean=means,
        seed_metric_std=standard_deviations,
    )


def _stratified_summaries(
    systems: Sequence[EvaluationSystemSpec],
    rows_by_system_seed: Mapping[tuple[str, str], Sequence[PerExampleEvaluation]],
) -> list[StratumEvaluationSummary]:
    output: list[StratumEvaluationSummary] = []
    for system in systems:
        rows = [
            row
            for seed in system.seeds
            for row in rows_by_system_seed[(system.system_id, seed.seed_id)]
        ]
        for axis in STRATIFICATION_AXES:
            grouped: dict[str, list[PerExampleEvaluation]] = defaultdict(list)
            for row in rows:
                grouped[str(getattr(row, axis))].append(row)
            output.extend(
                StratumEvaluationSummary(
                    system_id=system.system_id,
                    axis=axis,
                    value=value,
                    metrics=_aggregate(grouped[value]),
                )
                for value in sorted(grouped)
            )
    return output


def _example_metric(row: PerExampleEvaluation, metric: str) -> float:
    mapping = {
        "schema_validity": row.verifier.components["schema"],
        "evidence_f1": row.evidence_f1,
        "control_exact": row.verifier.components["control_match"],
        "constraint_f1": row.constraint_f1,
        "joint_state_accuracy": row.verifier.components["state_equivalence"],
        "replay_rate": row.verifier.components["replay"],
        "transition_consistency": float(row.transition_consistent),
        "persistence": row.verifier.components["persistence"],
        "accepted_rate": float(row.verifier.accepted),
        "mean_total_reward": row.verifier.total_reward,
    }
    return mapping[metric]


def _sample_metric_map(
    rows: Sequence[PerExampleEvaluation],
    metric: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.sample_id].append(_example_metric(row, metric))
    return {sample_id: _mean(values) for sample_id, values in grouped.items()}


def _bootstrap_result(
    *,
    metric: str,
    challenger: Mapping[str, float],
    reference: Mapping[str, float],
    resamples: int,
    confidence_level: float,
    seed: int,
) -> BootstrapMetricResult:
    sample_ids = sorted(set(challenger).intersection(reference))
    if not sample_ids:
        raise ValueError("paired bootstrap systems share no evaluated sample_ids")
    challenger_values = [challenger[sample_id] for sample_id in sample_ids]
    reference_values = [reference[sample_id] for sample_id in sample_ids]
    differences = [left - right for left, right in zip(challenger_values, reference_values, strict=True)]
    observed = _mean(differences)
    generator = random.Random(seed)
    bootstrap = [
        _mean([differences[generator.randrange(len(differences))] for _ in differences])
        for _ in range(resamples)
    ]
    alpha = 1.0 - confidence_level
    ci_low = _quantile(bootstrap, alpha / 2)
    ci_high = _quantile(bootstrap, 1 - alpha / 2)
    assert ci_low is not None and ci_high is not None
    non_positive = (sum(value <= 0 for value in bootstrap) + 1) / (resamples + 1)
    non_negative = (sum(value >= 0 for value in bootstrap) + 1) / (resamples + 1)
    return BootstrapMetricResult(
        metric=metric,
        paired_sample_count=len(sample_ids),
        challenger_mean=_mean(challenger_values),
        reference_mean=_mean(reference_values),
        observed_delta=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        two_sided_p=min(1.0, 2 * min(non_positive, non_negative)),
        confidence_level=confidence_level,
        bootstrap_resamples=resamples,
    )


def _paired_comparisons(
    spec: HeldoutEvaluationSpec,
    rows_by_system_seed: Mapping[tuple[str, str], Sequence[PerExampleEvaluation]],
) -> list[PairedComparisonResult]:
    system_rows = {
        system.system_id: [
            row
            for seed in system.seeds
            for row in rows_by_system_seed[(system.system_id, seed.seed_id)]
        ]
        for system in spec.systems
    }
    output: list[PairedComparisonResult] = []
    for comparison in spec.comparisons:
        metric_results = []
        for metric in BOOTSTRAP_METRICS:
            derived_seed = spec.bootstrap_seed ^ int(
                content_sha256(
                    {
                        "challenger": comparison.challenger_system_id,
                        "reference": comparison.reference_system_id,
                        "metric": metric,
                    }
                )[:16],
                16,
            )
            metric_results.append(
                _bootstrap_result(
                    metric=metric,
                    challenger=_sample_metric_map(
                        system_rows[comparison.challenger_system_id],
                        metric,
                    ),
                    reference=_sample_metric_map(
                        system_rows[comparison.reference_system_id],
                        metric,
                    ),
                    resamples=spec.bootstrap_resamples,
                    confidence_level=spec.confidence_level,
                    seed=derived_seed,
                )
            )
        output.append(
            PairedComparisonResult(
                challenger_system_id=comparison.challenger_system_id,
                reference_system_id=comparison.reference_system_id,
                metrics=metric_results,
            )
        )
    return output


def _flatten_metrics(metrics: MetricSummary) -> dict[str, JsonValue]:
    return metrics.model_dump(mode="json")


def _csv_text(
    rows: Sequence[Mapping[str, object]],
    *,
    empty_fieldnames: Sequence[str] | None = None,
) -> str:
    if not rows:
        if empty_fieldnames is None:
            return ""
        buffer = io.StringIO(newline="")
        csv.DictWriter(
            buffer,
            fieldnames=list(empty_fieldnames),
            lineterminator="\n",
        ).writeheader()
        return buffer.getvalue()
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("CSV rows do not share a stable column order")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _paper_csv_tables(report: HeldoutEvaluationReport) -> dict[str, str]:
    overall_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for system in report.systems:
        overall_rows.append(
            {
                "system_id": system.system_id,
                "display_name": system.display_name,
                "seed_count": system.seed_count,
                **_flatten_metrics(system.pooled_metrics),
                **{
                    f"seed_std_{metric}": system.seed_metric_std[metric]
                    for metric in SEED_SUMMARY_METRICS
                },
            }
        )
        seed_rows.extend(
            {
                "system_id": system.system_id,
                "display_name": system.display_name,
                "seed_id": seed.seed_id,
                "rollout_file_sha256": seed.rollout_file_sha256,
                **_flatten_metrics(seed.metrics),
            }
            for seed in system.seed_summaries
        )
    stratum_rows = [
        {
            "system_id": row.system_id,
            "axis": row.axis,
            "value": row.value,
            **_flatten_metrics(row.metrics),
        }
        for row in report.strata
    ]
    bootstrap_rows = [
        {
            "challenger_system_id": comparison.challenger_system_id,
            "reference_system_id": comparison.reference_system_id,
            **metric.model_dump(mode="json"),
        }
        for comparison in report.paired_comparisons
        for metric in comparison.metrics
    ]
    return {
        "paper_overall.csv": _csv_text(overall_rows),
        "paper_seeds.csv": _csv_text(seed_rows),
        "paper_strata.csv": _csv_text(stratum_rows),
        "paper_paired_bootstrap.csv": _csv_text(
            bootstrap_rows,
            empty_fieldnames=(
                "challenger_system_id",
                "reference_system_id",
                "metric",
                "paired_sample_count",
                "challenger_mean",
                "reference_mean",
                "observed_delta",
                "ci_low",
                "ci_high",
                "two_sided_p",
                "confidence_level",
                "bootstrap_resamples",
            ),
        ),
    }


def _output_ledger(run_dir: Path) -> list[dict[str, JsonValue]]:
    return [
        {
            "relative_path": str(path.relative_to(run_dir)),
            "sha256": _file_sha256(path),
            "byte_size": path.stat().st_size,
            "row_count": (
                sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
                if path.suffix == ".jsonl"
                else None
            ),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path != run_dir / "manifest.json"
    ]


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def load_heldout_evaluation_spec(path: Path) -> tuple[HeldoutEvaluationSpec, Path]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("held-out evaluation spec must be one JSON object")
    return HeldoutEvaluationSpec.model_validate(value), path.resolve().parent


def run_heldout_evaluation(
    *,
    spec: HeldoutEvaluationSpec,
    spec_base_dir: Path,
    output_root: Path,
    reward_config: TVRLVRRewardConfig | None = None,
    progress: Callable[[str], None] = print,
) -> HeldoutEvaluationResult:
    """Evaluate frozen rollouts and write only evidence-backed paper tables."""

    reward_config = reward_config or TVRLVRRewardConfig()
    dataset_dir = _resolve_path(spec_base_dir, spec.dataset_dir)
    output_root = output_root.resolve()
    if dataset_dir == output_root or dataset_dir in output_root.parents:
        raise ValueError("evaluation output_root must be outside the immutable dataset directory")
    dataset_paths = {
        "manifest": dataset_dir / "manifest.json",
        "public": dataset_dir / f"rl_prompt_pool.{spec.split.value}.public.jsonl",
        "private": dataset_dir / f"rl_verifier.{spec.split.value}.private.jsonl",
        "registry": dataset_dir / "sample_registry.private.jsonl",
    }
    rollout_paths = {
        (system.system_id, seed.seed_id): _resolve_path(spec_base_dir, seed.rollout_path)
        for system in spec.systems
        for seed in system.seeds
    }
    input_hashes_before = {
        "dataset": {name: _file_sha256(path) for name, path in dataset_paths.items()},
        "rollouts": {
            f"{system_id}/{seed_id}": _file_sha256(path)
            for (system_id, seed_id), path in sorted(rollout_paths.items())
        },
    }
    identity = {
        "schema_version": HELDOUT_EVALUATION_VERSION,
        "verifier_chain_version": VERIFIER_CHAIN_VERSION,
        "dataset_build_name": dataset_dir.name,
        "split": spec.split.value,
        "systems": [
            {
                "system_id": system.system_id,
                "display_name": system.display_name,
                "seeds": [
                    {
                        "seed_id": seed.seed_id,
                        "rollout_sha256": input_hashes_before["rollouts"][
                            f"{system.system_id}/{seed.seed_id}"
                        ],
                    }
                    for seed in system.seeds
                ],
            }
            for system in spec.systems
        ],
        "comparisons": [comparison.model_dump(mode="json") for comparison in spec.comparisons],
        "require_complete": spec.require_complete,
        "bootstrap_resamples": spec.bootstrap_resamples,
        "bootstrap_seed": spec.bootstrap_seed,
        "confidence_level": spec.confidence_level,
        "reward_config": reward_config.model_dump(mode="json"),
        "dataset_hashes": input_hashes_before["dataset"],
    }
    run_id = f"dsus-heldout-eval-{content_sha256(identity)[:16]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    progress("步骤 1/5：只读加载 held-out public/private/registry，核对 split 与样本身份。")
    public_by_id = _index(_load_models(dataset_paths["public"], PublicRLPromptRecord), label="public")
    private_by_id = _index(_load_models(dataset_paths["private"], PrivateRLVerifierRecord), label="private")
    registry_by_id = _index(
        [
            record
            for record in _load_models(dataset_paths["registry"], SampleRegistryRecord)
            if record.split == spec.split
        ],
        label="registry",
    )
    if not public_by_id:
        raise ValueError(f"dataset split {spec.split.value} has no evaluation samples")
    if not (set(public_by_id) == set(private_by_id) == set(registry_by_id)):
        raise ValueError("public/private/registry sample_id sets differ for the selected split")
    for sample_id in public_by_id:
        if public_by_id[sample_id].public_prompt_sha256 != private_by_id[sample_id].public_prompt_sha256:
            raise ValueError(f"public/private prompt hash mismatch: {sample_id}")
    progress(f"步骤 1/5 完成：冻结 split={spec.split.value}，样本={len(public_by_id)}。")

    progress("步骤 2/5：读取已生成 rollouts；运行 verifier 并计算 evidence/constraint 分解指标。")
    rows_by_system_seed: dict[tuple[str, str], list[PerExampleEvaluation]] = {}
    rollout_hashes_by_system: dict[str, dict[str, str]] = defaultdict(dict)
    for system in spec.systems:
        for seed in system.seeds:
            path = rollout_paths[(system.system_id, seed.seed_id)]
            rollout_by_id = _index(_load_models(path, HeldoutRolloutRecord), label="rollout")
            if spec.require_complete and set(rollout_by_id) != set(public_by_id):
                raise ValueError(
                    f"{system.system_id}/{seed.seed_id} is not complete for split {spec.split.value}"
                )
            if not set(rollout_by_id) <= set(public_by_id):
                raise ValueError(f"{system.system_id}/{seed.seed_id} contains unknown sample_ids")
            evaluated = []
            for sample_id in sorted(rollout_by_id):
                rollout = rollout_by_id[sample_id]
                if rollout.public_prompt_sha256 != public_by_id[sample_id].public_prompt_sha256:
                    raise ValueError(
                        f"rollout prompt hash mismatch: {system.system_id}/{seed.seed_id}/{sample_id}"
                    )
                evaluated.append(
                    _evaluate_one(
                        system_id=system.system_id,
                        seed_id=seed.seed_id,
                        public=public_by_id[sample_id],
                        private=private_by_id[sample_id],
                        registry=registry_by_id[sample_id],
                        rollout=rollout,
                        reward_config=reward_config,
                    )
                )
            rows_by_system_seed[(system.system_id, seed.seed_id)] = evaluated
            rollout_hashes_by_system[system.system_id][seed.seed_id] = _file_sha256(path)
            progress(
                f"  {system.system_id}/{seed.seed_id}: {len(evaluated)}/{len(public_by_id)} 条完成。"
            )
    progress("步骤 2/5 完成：逐样本 verifier 与细粒度 P/R/F1 已计算。")

    progress("步骤 3/5：合并多个 seed，按 difficulty/mechanism/domain/source/curriculum 分层。")
    systems = [
        _system_summary(
            system,
            {
                seed.seed_id: rows_by_system_seed[(system.system_id, seed.seed_id)]
                for seed in system.seeds
            },
            rollout_hashes_by_system[system.system_id],
        )
        for system in spec.systems
    ]
    strata = _stratified_summaries(spec.systems, rows_by_system_seed)
    progress(f"步骤 3/5 完成：system={len(systems)}，stratum rows={len(strata)}。")

    progress("步骤 4/5：在 shared sample IDs 上执行确定性 paired bootstrap。")
    comparisons = _paired_comparisons(spec, rows_by_system_seed)
    progress(
        f"步骤 4/5 完成：comparison={len(comparisons)}，"
        f"bootstrap metrics={sum(len(item.metrics) for item in comparisons)}。"
    )

    progress("步骤 5/5：写入 paper-facing JSON/CSV、逐样本 private ledger 与内容寻址 manifest。")
    input_hashes_after = {
        "dataset": {name: _file_sha256(path) for name, path in dataset_paths.items()},
        "rollouts": {
            f"{system_id}/{seed_id}": _file_sha256(path)
            for (system_id, seed_id), path in sorted(rollout_paths.items())
        },
    }
    unchanged = input_hashes_before == input_hashes_after
    if not unchanged:
        raise ValueError("dataset or rollout source changed during held-out evaluation")
    report = HeldoutEvaluationReport(
        dataset_build_name=dataset_dir.name,
        split=spec.split,
        dataset_sample_count=len(public_by_id),
        systems=systems,
        strata=strata,
        paired_comparisons=comparisons,
        source_artifacts_unchanged=True,
    )
    per_example_rows = [
        row
        for system in spec.systems
        for seed in system.seeds
        for row in rows_by_system_seed[(system.system_id, seed.seed_id)]
    ]
    _write_idempotently(
        run_dir / "per_example.private.jsonl",
        "".join(canonical_json(row.model_dump(mode="json")) + "\n" for row in per_example_rows),
    )
    _write_idempotently(
        run_dir / "paper_results.json",
        canonical_json(report.model_dump(mode="json")) + "\n",
    )
    for filename, text in _paper_csv_tables(report).items():
        _write_idempotently(run_dir / filename, text)
    resolved_spec = {
        "schema_version": HELDOUT_EVALUATION_SPEC_VERSION,
        "dataset_build_name": dataset_dir.name,
        "split": spec.split.value,
        "systems": identity["systems"],
        "comparisons": identity["comparisons"],
        "require_complete": spec.require_complete,
        "bootstrap_resamples": spec.bootstrap_resamples,
        "bootstrap_seed": spec.bootstrap_seed,
        "confidence_level": spec.confidence_level,
    }
    _write_idempotently(
        run_dir / "resolved_spec.private.json",
        canonical_json(resolved_spec) + "\n",
    )
    _write_idempotently(
        run_dir / "summary.json",
        canonical_json(
            {
                "schema_version": HELDOUT_EVALUATION_VERSION,
                "run_id": run_id,
                "dataset_build_name": dataset_dir.name,
                "split": spec.split.value,
                "dataset_sample_count": len(public_by_id),
                "system_count": len(systems),
                "seed_count": sum(system.seed_count for system in systems),
                "comparison_count": len(comparisons),
                "runtime_metadata_note": (
                    "null means not provided by rollout; missing runtime values are never imputed"
                ),
                "source_artifacts_unchanged": True,
                "external_calls": False,
                "training_started": False,
            }
        )
        + "\n",
    )
    manifest = {
        "schema_version": HELDOUT_EVALUATION_VERSION,
        "artifact_role": "dsus_state_compiler_heldout_paper_evaluation",
        "run_id": run_id,
        "identity": identity,
        "source_artifacts_mutated": False,
        "runtime_missing_value_policy": "preserve_null_and_report_coverage",
        "external_calls": False,
        "training_started": False,
        "outputs": _output_ledger(run_dir),
    }
    _write_idempotently(run_dir / "manifest.json", canonical_json(manifest) + "\n")
    progress(f"步骤 5/5 完成：paper-facing 评测工件={run_dir}。")
    return HeldoutEvaluationResult(run_id=run_id, run_dir=str(run_dir), report=report)


__all__ = [
    "BOOTSTRAP_METRICS",
    "HELDOUT_EVALUATION_SPEC_VERSION",
    "HELDOUT_EVALUATION_VERSION",
    "BootstrapMetricResult",
    "EvaluationSeedSpec",
    "EvaluationSystemSpec",
    "HeldoutEvaluationReport",
    "HeldoutEvaluationResult",
    "HeldoutEvaluationSpec",
    "HeldoutRolloutRecord",
    "MetricSummary",
    "PairedComparisonResult",
    "PairedComparisonSpec",
    "PerExampleEvaluation",
    "RolloutRuntimeMetadata",
    "SeedEvaluationSummary",
    "StratumEvaluationSummary",
    "SystemEvaluationSummary",
    "load_heldout_evaluation_spec",
    "run_heldout_evaluation",
]
