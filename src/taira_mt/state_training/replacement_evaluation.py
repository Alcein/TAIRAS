"""Offline near-lossless replacement evaluation for the trainable DSUS compiler.

The evaluator is deliberately split into two physical stages.  Stage A reads
only a frozen public sample manifest, public compiler prompts and already
generated model rollouts.  It validates/replays each patch and runs the same
frozen SCTP/CIV/Thought-Pattern stack.  Only after those downstream snapshots
have been frozen does Stage B open the trainer-private gold state and score the
systems.  No model generation, API call or training occurs in this module.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, JsonValue, field_validator, model_validator

from taira_mt.benchmark.schemas import BenchmarkActionType, BenchmarkSystemAction
from taira_mt.models import StrictModel
from taira_mt.mt_agent.civ import CIVPairwiseLinearRanker, CIVRanker
from taira_mt.mt_agent.memory import PublicDialogueMemory
from taira_mt.mt_agent.retrieval import LexicalCatalogRetriever
from taira_mt.mt_agent.sctp import SCTPSkillCompiler
from taira_mt.mt_agent.skills import ThoughtPatternLibrary
from taira_mt.mt_agent.state import DSUSDecisionPosterior, RuleDSUSEstimator
from taira_mt.mt_agent.state_compiler_adapter import project_compiler_state_for_sctp
from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.state_compiler import ControlField, PublicDSUSState, reduce_state
from taira_mt.state_compiler.runtime import validate_public_evidence
from taira_mt.state_training.dataset_assembly import (
    DatasetSplit,
    PrivateRLVerifierRecord,
    PublicRLPromptRecord,
    SampleRegistryRecord,
)
from taira_mt.state_training.heldout_evaluation import RolloutRuntimeMetadata
from taira_mt.state_training.training_views import forbidden_public_key_paths
from taira_mt.state_training.verifier_chain import parse_model_patch

REPLACEMENT_SAMPLE_MANIFEST_VERSION = "dsus_replacement_sample_manifest_v1"
REPLACEMENT_EVALUATION_SPEC_VERSION = "dsus_compiler_replacement_evaluation_spec_v1"
REPLACEMENT_EVALUATION_VERSION = "dsus_compiler_replacement_evaluation_v1"

PRIMARY_METRICS = (
    "json_schema_validity",
    "evidence_grounding",
    "next_state_exact",
    "field_f1",
    "control_state_accuracy",
    "pattern_frontier_agreement",
    "action_agreement",
    "end_to_end_success_proxy",
    "fallback_free",
)

_PUBLIC_DENIED_KEYS = {
    "case_id",
    "episode_id",
    "gold_delta",
    "gold_next_state",
    "gold_patch",
    "hidden_goal",
    "private_target",
    "reward",
    "root_id",
    "trainer_private",
}


def _normalized_key(value: object) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value))
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _denied_key_paths(payload: object) -> list[str]:
    paths: list[str] = []

    def visit(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, child_value in value.items():
                child = f"{path}.{key}" if path else str(key)
                if _normalized_key(key) in _PUBLIC_DENIED_KEYS:
                    paths.append(child)
                visit(child_value, child)
        elif isinstance(value, list):
            for index, child_value in enumerate(value):
                visit(child_value, f"{path}[{index}]")

    visit(payload, "")
    return sorted(paths)


class FrozenCompilerSample(StrictModel):
    """Opaque sample identity plus the two public routing fields needed online."""

    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    domain: str = Field(min_length=1, max_length=128)


class ReplacementSampleManifest(StrictModel):
    schema_version: Literal[REPLACEMENT_SAMPLE_MANIFEST_VERSION] = REPLACEMENT_SAMPLE_MANIFEST_VERSION
    dataset_build_name: str = Field(min_length=1)
    split: DatasetSplit
    samples: list[FrozenCompilerSample] = Field(min_length=1)
    source_public_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_manifest_id: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> ReplacementSampleManifest:
        identifiers = [item.sample_id for item in self.samples]
        if identifiers != sorted(identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError("replacement sample manifest must have sorted unique sample IDs")
        identity = self.model_dump(mode="json", exclude={"sample_manifest_id"})
        if self.sample_manifest_id != content_sha256(identity):
            raise ValueError("replacement sample manifest identity mismatch")
        if _denied_key_paths(self.model_dump(mode="json")):
            raise ValueError("replacement sample manifest exposes a denied private/episode key")
        return self


class CompilerReplacementRollout(StrictModel):
    """One already-generated compiler output; raw private labels are impossible."""

    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_output: str | dict[str, JsonValue]
    runtime: RolloutRuntimeMetadata | None = None
    fallback_used: bool = False


class ReplacementSystemSpec(StrictModel):
    system_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    display_name: str = Field(min_length=1, max_length=256)
    checkpoint_stage: Literal["reference", "sft", "tv_rlvr"]
    rollout_path: str = Field(min_length=1)


class FrozenDownstreamSpec(StrictModel):
    """Secret-free deterministic downstream shared by all compiler systems."""

    retriever_root: str = Field(min_length=1)
    pattern_library_path: str = Field(min_length=1)
    civ_model_path: str | None = None
    include_initial_counterfactuals: bool = True
    semantic_limit: int = Field(default=2, ge=1, le=10)
    interaction_limit: int = Field(default=3, ge=1, le=10)
    shortlist_k: int = Field(default=50, ge=10, le=50)
    recommendation_k: int = Field(default=10, ge=1, le=10)
    fallback_to_rule: bool = True

    @model_validator(mode="after")
    def validate_retrieval_sizes(self) -> FrozenDownstreamSpec:
        if self.recommendation_k > self.shortlist_k:
            raise ValueError("recommendation_k cannot exceed shortlist_k")
        return self


class PairedReplacementSpec(StrictModel):
    challenger_system_id: str
    reference_system_id: str

    @model_validator(mode="after")
    def validate_distinct(self) -> PairedReplacementSpec:
        if self.challenger_system_id == self.reference_system_id:
            raise ValueError("paired replacement comparison requires distinct systems")
        return self


class CompilerReplacementEvaluationSpec(StrictModel):
    schema_version: Literal[REPLACEMENT_EVALUATION_SPEC_VERSION] = REPLACEMENT_EVALUATION_SPEC_VERSION
    dataset_dir: str = Field(min_length=1)
    split: DatasetSplit = DatasetSplit.TEST
    sample_manifest_path: str = Field(min_length=1)
    systems: list[ReplacementSystemSpec] = Field(min_length=2)
    reference_system_id: str = Field(min_length=1)
    downstream: FrozenDownstreamSpec
    comparisons: list[PairedReplacementSpec] = Field(default_factory=list)
    # A near-lossless paired claim is meaningful only on an exactly matched
    # panel; partial system-specific coverage is therefore not supported.
    require_complete: Literal[True] = True
    bootstrap_resamples: int = Field(default=5000, ge=100, le=100000)
    bootstrap_seed: int = 20260809
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)

    @model_validator(mode="after")
    def validate_systems(self) -> CompilerReplacementEvaluationSpec:
        identifiers = [item.system_id for item in self.systems]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("replacement evaluation has duplicate system IDs")
        if self.reference_system_id not in identifiers:
            raise ValueError("reference_system_id is not one of the configured systems")
        reference = next(item for item in self.systems if item.system_id == self.reference_system_id)
        if reference.checkpoint_stage != "reference":
            raise ValueError("reference_system_id must point to checkpoint_stage=reference")
        known = set(identifiers)
        pairs = []
        for item in self.comparisons:
            if item.challenger_system_id not in known or item.reference_system_id not in known:
                raise ValueError("paired comparison references an unknown system")
            pairs.append((item.challenger_system_id, item.reference_system_id))
        if len(pairs) != len(set(pairs)):
            raise ValueError("replacement evaluation has duplicate paired comparisons")
        return self


class DownstreamDecisionSnapshot(StrictModel):
    pattern_frontier: list[str] = Field(min_length=1)
    selected_pattern_signature: str = Field(min_length=1)
    selected_action_type: str = Field(min_length=1)
    selected_item_ids: list[str] = Field(default_factory=list)
    selected_question_target: str | None = None
    used_rule_fallback: bool = False

    @field_validator("pattern_frontier")
    @classmethod
    def canonicalize_frontier(cls, value: list[str]) -> list[str]:
        values = sorted(set(value))
        if not values:
            raise ValueError("downstream Pattern frontier cannot be empty")
        return values


class DownstreamEvaluator(Protocol):
    identity: Mapping[str, JsonValue]

    def evaluate(
        self,
        *,
        public_record: PublicRLPromptRecord,
        domain: str,
        compiler_state: PublicDSUSState | None,
        use_rule_fallback: bool,
    ) -> DownstreamDecisionSnapshot | None: ...


class PublicStageEvaluation(StrictModel):
    system_id: str
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    json_schema_valid: bool
    evidence_grounded: bool
    reducer_executable: bool
    fallback_used: bool
    public_error_codes: list[str]
    candidate_next_state: PublicDSUSState | None = None
    downstream: DownstreamDecisionSnapshot | None = None
    runtime: RolloutRuntimeMetadata | None = None


class PerExampleReplacementEvaluation(StrictModel):
    system_id: str
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    json_schema_validity: float = Field(ge=0.0, le=1.0)
    evidence_grounding: float = Field(ge=0.0, le=1.0)
    next_state_exact: float = Field(ge=0.0, le=1.0)
    field_f1: float = Field(ge=0.0, le=1.0)
    control_state_accuracy: float = Field(ge=0.0, le=1.0)
    pattern_frontier_agreement: float = Field(ge=0.0, le=1.0)
    action_agreement: float = Field(ge=0.0, le=1.0)
    pattern_frontier_agreement_to_gold: float = Field(ge=0.0, le=1.0)
    action_accuracy_to_gold: float = Field(ge=0.0, le=1.0)
    end_to_end_success_proxy: float = Field(ge=0.0, le=1.0)
    fallback_free: float = Field(ge=0.0, le=1.0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    total_tokens: int | None = Field(default=None, ge=0)


class ReplacementMetricSummary(StrictModel):
    sample_count: int = Field(ge=1)
    json_schema_validity: float = Field(ge=0.0, le=1.0)
    evidence_grounding: float = Field(ge=0.0, le=1.0)
    next_state_exact: float = Field(ge=0.0, le=1.0)
    field_f1: float = Field(ge=0.0, le=1.0)
    control_state_accuracy: float = Field(ge=0.0, le=1.0)
    pattern_frontier_agreement: float = Field(ge=0.0, le=1.0)
    action_agreement: float = Field(ge=0.0, le=1.0)
    pattern_frontier_agreement_to_gold: float = Field(ge=0.0, le=1.0)
    action_accuracy_to_gold: float = Field(ge=0.0, le=1.0)
    end_to_end_success_proxy: float = Field(ge=0.0, le=1.0)
    fallback_rate: float = Field(ge=0.0, le=1.0)
    latency_coverage_count: int = Field(ge=0)
    latency_mean_ms: float | None = Field(default=None, ge=0.0)
    latency_p50_ms: float | None = Field(default=None, ge=0.0)
    latency_p95_ms: float | None = Field(default=None, ge=0.0)
    token_coverage_count: int = Field(ge=0)
    total_tokens_mean: float | None = Field(default=None, ge=0.0)


class ReplacementSystemSummary(StrictModel):
    system_id: str
    display_name: str
    checkpoint_stage: str
    rollout_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics: ReplacementMetricSummary


class PairedReplacementMetric(StrictModel):
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


class PairedReplacementResult(StrictModel):
    challenger_system_id: str
    reference_system_id: str
    metrics: list[PairedReplacementMetric] = Field(min_length=1)


class CompilerReplacementEvaluationReport(StrictModel):
    schema_version: Literal[REPLACEMENT_EVALUATION_VERSION] = REPLACEMENT_EVALUATION_VERSION
    dataset_build_name: str
    split: DatasetSplit
    sample_manifest_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_count: int = Field(ge=1)
    reference_system_id: str
    downstream_identity: dict[str, JsonValue]
    systems: list[ReplacementSystemSummary]
    paired_comparisons: list[PairedReplacementResult]
    public_stage_completed_before_private_join: Literal[True] = True
    trainer_private_join_stage: Literal["post_public_downstream_freeze"] = "post_public_downstream_freeze"
    episode_id_exposed_to_compiler: Literal[False] = False
    gold_exposed_to_compiler: Literal[False] = False
    source_artifacts_unchanged: Literal[True] = True
    external_calls: Literal[False] = False
    training_started: Literal[False] = False


class CompilerReplacementEvaluationResult(StrictModel):
    run_id: str
    run_dir: str
    report: CompilerReplacementEvaluationReport


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    output: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        output.append(value)
    return output


def _load_index(path: Path, model_type: type[StrictModel], *, label: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for row in _read_jsonl(path):
        record = model_type.model_validate(row)
        sample_id = str(record.sample_id)
        if sample_id in output:
            raise ValueError(f"duplicate {label} sample_id: {sample_id}")
        output[sample_id] = record
    return output


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _write_idempotently(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"refusing to overwrite a non-identical artifact: {path}")
    path.write_text(text, encoding="utf-8")


def freeze_replacement_sample_manifest(
    *,
    dataset_dir: Path,
    split: DatasetSplit,
    output_path: Path,
    sample_ids: Sequence[str] | None = None,
) -> ReplacementSampleManifest:
    """Freeze opaque IDs/hash/domain only; episode IDs and gold are never serialized."""

    dataset_dir = dataset_dir.resolve()
    output_path = output_path.resolve()
    if dataset_dir == output_path.parent or dataset_dir in output_path.parents:
        raise ValueError("sample manifest must be written outside the immutable dataset directory")
    public_path = dataset_dir / f"rl_prompt_pool.{split.value}.public.jsonl"
    registry_path = dataset_dir / "sample_registry.private.jsonl"
    public_by_id = _load_index(public_path, PublicRLPromptRecord, label="public")
    registry_by_id = {
        sample_id: record
        for sample_id, record in _load_index(
            registry_path, SampleRegistryRecord, label="registry"
        ).items()
        if record.split == split
    }
    if set(public_by_id) != set(registry_by_id):
        raise ValueError("selected split public/registry sample sets differ")
    selected = sorted(public_by_id) if sample_ids is None else list(sample_ids)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("sample_ids must be non-empty and unique")
    unknown = sorted(set(selected).difference(public_by_id))
    if unknown:
        raise ValueError(f"sample manifest requested unknown IDs: {unknown[:3]}")
    samples = [
        FrozenCompilerSample(
            sample_id=sample_id,
            public_prompt_sha256=public_by_id[sample_id].public_prompt_sha256,
            domain=registry_by_id[sample_id].domain,
        )
        for sample_id in sorted(selected)
    ]
    base = {
        "schema_version": REPLACEMENT_SAMPLE_MANIFEST_VERSION,
        "dataset_build_name": dataset_dir.name,
        "split": split.value,
        "samples": [item.model_dump(mode="json") for item in samples],
        "source_public_sha256": _file_sha256(public_path),
        "source_registry_sha256": _file_sha256(registry_path),
    }
    manifest = ReplacementSampleManifest.model_validate(
        {**base, "sample_manifest_id": content_sha256(base)}
    )
    _write_idempotently(output_path, canonical_json(manifest.model_dump(mode="json")) + "\n")
    return manifest


def load_replacement_sample_manifest(path: Path) -> ReplacementSampleManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("replacement sample manifest must be one JSON object")
    denied = _denied_key_paths(raw)
    if denied:
        raise ValueError(f"replacement sample manifest contains denied keys: {denied}")
    return ReplacementSampleManifest.model_validate(raw)


def load_replacement_evaluation_spec(
    path: Path,
) -> tuple[CompilerReplacementEvaluationSpec, Path]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("replacement evaluation spec must be one JSON object")
    return CompilerReplacementEvaluationSpec.model_validate(raw), path.resolve().parent


def _history_system_action(text: str, index: int) -> BenchmarkSystemAction:
    folded = text.casefold()
    if "?" in text or folded.startswith(("which ", "what ", "would ", "do you ", "could you ")):
        return BenchmarkSystemAction(
            action_type=BenchmarkActionType.ASK,
            question_target=" ".join(text.split())[:500],
            utterance=text,
        )
    if any(cue in folded for cue in ("recommend", "suggest", "my leading option", "i found")):
        return BenchmarkSystemAction(
            action_type=BenchmarkActionType.RECOMMEND,
            item_ids=[f"public-history-item-{content_sha256({'index': index, 'text': text})[:16]}"],
            utterance=text,
        )
    return BenchmarkSystemAction(action_type=BenchmarkActionType.STOP, utterance=text)


def public_prompt_to_memory(*, public_record: PublicRLPromptRecord, domain: str) -> PublicDialogueMemory:
    """Deterministically reconstruct the public user/system memory used downstream."""

    observations = public_record.public_prompt.public_history
    if observations is None:
        raise ValueError("replacement downstream evaluation requires the full public_history view")
    dialogue = [item for item in observations if item.role in {"user", "system"}]
    if not dialogue or dialogue[0].role != "user" or dialogue[-1].role != "user":
        raise ValueError("public history must start/end with user text for a compiler decision")
    if any(item.role != ("user" if index % 2 == 0 else "system") for index, item in enumerate(dialogue)):
        raise ValueError("public history user/system observations must alternate")
    memory = PublicDialogueMemory.begin(domain=domain, initial_user_utterance=dialogue[0].text)
    for index in range(1, len(dialogue), 2):
        system = dialogue[index]
        user = dialogue[index + 1]
        memory = memory.record_system_action(_history_system_action(system.text, index))
        memory = memory.record_user_response(user.text, terminal=False)
    return memory


class FrozenTAIRAMTDownstream:
    """One pinned deterministic SCTP/CIV/Pattern stack shared by every compiler."""

    def __init__(
        self,
        *,
        retrievers: Mapping[str, LexicalCatalogRetriever],
        library: ThoughtPatternLibrary,
        sctp: SCTPSkillCompiler,
        civ: CIVRanker | CIVPairwiseLinearRanker,
        identity: Mapping[str, JsonValue],
    ) -> None:
        self.retrievers = dict(retrievers)
        self.library = library
        self.sctp = sctp
        self.civ = civ
        self.identity = dict(identity)

    @classmethod
    def from_spec(
        cls,
        spec: FrozenDownstreamSpec,
        *,
        base_dir: Path,
        domains: Sequence[str],
    ) -> FrozenTAIRAMTDownstream:
        retriever_root = _resolve(base_dir, spec.retriever_root)
        pattern_path = _resolve(base_dir, spec.pattern_library_path)
        library = ThoughtPatternLibrary.load(pattern_path)
        retrievers = {
            domain: LexicalCatalogRetriever.load(retriever_root / domain / "lexical_index.joblib")
            for domain in sorted(set(domains))
        }
        if spec.civ_model_path is None:
            civ: CIVRanker | CIVPairwiseLinearRanker = CIVRanker()
            civ_artifact_sha256 = None
        else:
            civ_path = _resolve(base_dir, spec.civ_model_path)
            civ = CIVPairwiseLinearRanker.load(civ_path)
            civ_artifact_sha256 = _file_sha256(civ_path)
        sctp = SCTPSkillCompiler(
            library=library,
            semantic_limit=spec.semantic_limit,
            interaction_limit=spec.interaction_limit,
            shortlist_k=spec.shortlist_k,
            recommendation_k=spec.recommendation_k,
            include_initial_counterfactuals=spec.include_initial_counterfactuals,
        )
        identity: dict[str, JsonValue] = {
            "pattern_library_id": library.library_id,
            "pattern_library_sha256": _file_sha256(pattern_path),
            "sctp_compiler_id": sctp.compiler_id,
            "civ_model_id": civ.model_id,
            "civ_artifact_sha256": civ_artifact_sha256,
            "retriever_ids": {
                domain: retriever.retriever_id for domain, retriever in sorted(retrievers.items())
            },
            "semantic_limit": spec.semantic_limit,
            "interaction_limit": spec.interaction_limit,
            "shortlist_k": spec.shortlist_k,
            "recommendation_k": spec.recommendation_k,
            "include_initial_counterfactuals": spec.include_initial_counterfactuals,
            "external_calls": False,
        }
        return cls(retrievers=retrievers, library=library, sctp=sctp, civ=civ, identity=identity)

    def evaluate(
        self,
        *,
        public_record: PublicRLPromptRecord,
        domain: str,
        compiler_state: PublicDSUSState | None,
        use_rule_fallback: bool,
    ) -> DownstreamDecisionSnapshot | None:
        retriever = self.retrievers[domain]
        memory = public_prompt_to_memory(public_record=public_record, domain=domain)
        if use_rule_fallback:
            inference = RuleDSUSEstimator().infer(memory=memory, retriever=retriever)
            state = inference.state
            posterior = inference.decision_posterior
        else:
            if compiler_state is None:
                return None
            state = project_compiler_state_for_sctp(
                compiler_state,
                memory=memory,
                retriever=retriever,
                estimator_id="replacement-evaluation-compiler",
            )
            posterior = DSUSDecisionPosterior.from_raw(
                None,
                selected=state.recommendation_readiness,
                selected_probability=state.confidence,
            )
        frontier = self.sctp.compile(
            memory=memory,
            state=state,
            retriever=retriever,
            decision_posterior=posterior,
        )
        selection = self.civ.select(frontier, memory=memory)
        proposal = self.civ.selected_proposal(frontier, selection)
        signatures = [
            f"{item.semantic_skill_id}::{item.interaction_skill_id}::{item.action.action_type.value}"
            for item in frontier.proposals
        ]
        selected_signature = (
            f"{proposal.semantic_skill_id}::{proposal.interaction_skill_id}::"
            f"{proposal.action.action_type.value}"
        )
        return DownstreamDecisionSnapshot(
            pattern_frontier=signatures,
            selected_pattern_signature=selected_signature,
            selected_action_type=proposal.action.action_type.value,
            selected_item_ids=list(proposal.action.item_ids),
            selected_question_target=proposal.action.question_target,
            used_rule_fallback=use_rule_fallback,
        )


def _evaluate_public_rollout(
    *,
    system_id: str,
    public: PublicRLPromptRecord,
    domain: str,
    rollout: CompilerReplacementRollout,
    downstream: DownstreamEvaluator,
    fallback_to_rule: bool,
) -> PublicStageEvaluation:
    errors: list[str] = []
    parsed = None
    candidate: PublicDSUSState | None = None
    try:
        parsed, _ = parse_model_patch(rollout.model_output)
        schema_valid = True
    except (json.JSONDecodeError, TypeError, ValueError):
        schema_valid = False
        errors.append("json_schema_invalid")
    grounded = False
    reducer_executable = False
    if parsed is not None:
        try:
            validate_public_evidence(parsed.evidence, prompt=public.public_prompt)
            grounded = True
        except (TypeError, ValueError):
            errors.append("evidence_not_publicly_grounded")
        if parsed.base_revision != public.public_prompt.previous_public_state.revision:
            errors.append("base_revision_mismatch")
        else:
            try:
                candidate = reduce_state(
                    public.public_prompt.previous_public_state,
                    parsed.as_patch(),
                )
                reducer_executable = True
            except (TypeError, ValueError):
                errors.append("reducer_not_executable")
    compiler_failure = not (schema_valid and grounded and reducer_executable)
    effective_fallback = bool(rollout.fallback_used or compiler_failure) and fallback_to_rule
    effective_state = None if effective_fallback or compiler_failure else candidate
    try:
        snapshot = downstream.evaluate(
            public_record=public,
            domain=domain,
            compiler_state=effective_state,
            use_rule_fallback=effective_fallback,
        )
    except (TypeError, ValueError):
        # A reducer-executable state may still violate a stricter downstream
        # decision invariant (for example repair_recommend without visible
        # rejection/revision evidence).  Treat this as a verified compiler
        # failure and exercise the same bounded rule fallback as runtime.
        errors.append("downstream_state_invalid")
        effective_fallback = fallback_to_rule
        snapshot = downstream.evaluate(
            public_record=public,
            domain=domain,
            compiler_state=None,
            use_rule_fallback=effective_fallback,
        )
    return PublicStageEvaluation(
        system_id=system_id,
        sample_id=rollout.sample_id,
        public_prompt_sha256=rollout.public_prompt_sha256,
        json_schema_valid=schema_valid,
        evidence_grounded=grounded,
        reducer_executable=reducer_executable,
        fallback_used=effective_fallback,
        public_error_codes=sorted(set(errors)),
        candidate_next_state=candidate,
        downstream=snapshot,
        runtime=rollout.runtime,
    )


def _evidence_alpha_payload(state: PublicDSUSState) -> dict[str, JsonValue]:
    payload = state.model_dump(mode="json")
    id_to_token = {
        item.evidence_id: "evidence-"
        + content_sha256(
            {"turn_id": item.turn_id, "source": item.source.value, "quote": item.quote}
        )
        for item in state.evidence
    }
    for row in payload["evidence"]:
        row["evidence_id"] = id_to_token[str(row["evidence_id"])]
    payload["evidence"].sort(key=canonical_json)
    for collection_name in ("constraints", "unresolved_axes", "rejected_items"):
        for row in payload[collection_name]:
            row["evidence_ids"] = sorted(id_to_token[str(value)] for value in row["evidence_ids"])
    for field, values in payload["control_evidence"].items():
        payload["control_evidence"][field] = sorted(id_to_token[str(value)] for value in values)
    payload["termination_evidence_ids"] = sorted(
        id_to_token[str(value)] for value in payload["termination_evidence_ids"]
    )
    return payload


def _state_field_atoms(state: PublicDSUSState | None) -> Counter[str]:
    if state is None:
        return Counter()
    atoms: list[str] = [f"initialized={state.initialized}"]
    atoms.extend(
        f"control.{field.value}={getattr(state.control, field.value).value}"
        for field in ControlField
    )
    atoms.extend(
        "constraint="
        + canonical_json(item.model_dump(mode="json", exclude={"constraint_id", "evidence_ids"}))
        for item in state.constraints
    )
    atoms.extend(
        "axis=" + canonical_json(item.model_dump(mode="json", exclude={"axis_id", "evidence_ids"}))
        for item in state.unresolved_axes
    )
    atoms.extend(
        "rejected=" + canonical_json(item.model_dump(mode="json", exclude={"evidence_ids"}))
        for item in state.rejected_items
    )
    atoms.append(
        "termination=" + ("none" if state.termination_reason is None else state.termination_reason.value)
    )
    return Counter(atoms)


def _counter_f1(left: Counter[str], right: Counter[str]) -> float:
    overlap = sum((left & right).values())
    precision = overlap / sum(left.values()) if left else float(not right)
    recall = overlap / sum(right.values()) if right else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _control_accuracy(candidate: PublicDSUSState | None, gold: PublicDSUSState) -> float:
    if candidate is None:
        return 0.0
    return sum(
        getattr(candidate.control, field.value) == getattr(gold.control, field.value)
        for field in ControlField
    ) / len(ControlField)


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _frontier_agreement(
    left: DownstreamDecisionSnapshot | None,
    right: DownstreamDecisionSnapshot | None,
) -> float:
    if left is None or right is None:
        return 0.0
    return _jaccard(left.pattern_frontier, right.pattern_frontier)


def _action_agreement(
    left: DownstreamDecisionSnapshot | None,
    right: DownstreamDecisionSnapshot | None,
) -> float:
    return float(
        left is not None
        and right is not None
        and left.selected_action_type == right.selected_action_type
    )


def _end_to_end_proxy(
    candidate: DownstreamDecisionSnapshot | None,
    gold: DownstreamDecisionSnapshot | None,
) -> float:
    if candidate is None or gold is None or candidate.selected_action_type != gold.selected_action_type:
        return 0.0
    if gold.selected_action_type == BenchmarkActionType.RECOMMEND.value:
        return float(bool(set(candidate.selected_item_ids).intersection(gold.selected_item_ids)))
    if gold.selected_action_type == BenchmarkActionType.ASK.value:
        return float(candidate.selected_pattern_signature == gold.selected_pattern_signature)
    return 1.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _aggregate(rows: Sequence[PerExampleReplacementEvaluation]) -> ReplacementMetricSummary:
    if not rows:
        raise ValueError("cannot aggregate an empty replacement evaluation")
    latency = [row.latency_ms for row in rows if row.latency_ms is not None]
    tokens = [float(row.total_tokens) for row in rows if row.total_tokens is not None]
    return ReplacementMetricSummary(
        sample_count=len(rows),
        **{metric: _mean([getattr(row, metric) for row in rows]) for metric in PRIMARY_METRICS[:-1]},
        pattern_frontier_agreement_to_gold=_mean(
            [row.pattern_frontier_agreement_to_gold for row in rows]
        ),
        action_accuracy_to_gold=_mean([row.action_accuracy_to_gold for row in rows]),
        fallback_rate=1.0 - _mean([row.fallback_free for row in rows]),
        latency_coverage_count=len(latency),
        latency_mean_ms=_mean(latency) if latency else None,
        latency_p50_ms=_quantile(latency, 0.5),
        latency_p95_ms=_quantile(latency, 0.95),
        token_coverage_count=len(tokens),
        total_tokens_mean=_mean(tokens) if tokens else None,
    )


def _bootstrap_metric(
    *,
    metric: str,
    challenger: Mapping[str, PerExampleReplacementEvaluation],
    reference: Mapping[str, PerExampleReplacementEvaluation],
    resamples: int,
    confidence_level: float,
    seed: int,
) -> PairedReplacementMetric | None:
    sample_ids = sorted(set(challenger).intersection(reference))
    values: list[tuple[float, float]] = []
    for sample_id in sample_ids:
        left = getattr(challenger[sample_id], metric)
        right = getattr(reference[sample_id], metric)
        if left is not None and right is not None:
            values.append((float(left), float(right)))
    if not values:
        return None
    differences = [left - right for left, right in values]
    generator = random.Random(seed)
    bootstrap = [
        _mean([differences[generator.randrange(len(differences))] for _ in differences])
        for _ in range(resamples)
    ]
    alpha = 1.0 - confidence_level
    low, high = _quantile(bootstrap, alpha / 2), _quantile(bootstrap, 1 - alpha / 2)
    assert low is not None and high is not None
    non_positive = (sum(value <= 0 for value in bootstrap) + 1) / (resamples + 1)
    non_negative = (sum(value >= 0 for value in bootstrap) + 1) / (resamples + 1)
    return PairedReplacementMetric(
        metric=metric,
        paired_sample_count=len(values),
        challenger_mean=_mean([left for left, _ in values]),
        reference_mean=_mean([right for _, right in values]),
        observed_delta=_mean(differences),
        ci_low=low,
        ci_high=high,
        two_sided_p=min(1.0, 2 * min(non_positive, non_negative)),
        confidence_level=confidence_level,
        bootstrap_resamples=resamples,
    )


def _csv_text(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return ""
    fieldnames = list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _output_ledger(run_dir: Path) -> list[dict[str, JsonValue]]:
    return [
        {
            "relative_path": str(path.relative_to(run_dir)),
            "sha256": _file_sha256(path),
            "byte_size": path.stat().st_size,
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path != run_dir / "manifest.json"
    ]


def run_compiler_replacement_evaluation(
    *,
    spec: CompilerReplacementEvaluationSpec,
    spec_base_dir: Path,
    output_root: Path,
    downstream_evaluator: DownstreamEvaluator | None = None,
    progress: Callable[[str], None] = print,
) -> CompilerReplacementEvaluationResult:
    """Run an API-free, training-free, fixed-manifest paired replacement test."""

    dataset_dir = _resolve(spec_base_dir, spec.dataset_dir)
    sample_manifest_path = _resolve(spec_base_dir, spec.sample_manifest_path)
    output_root = output_root.resolve()
    if dataset_dir == output_root or dataset_dir in output_root.parents:
        raise ValueError("replacement evaluation output must be outside the immutable dataset")
    public_path = dataset_dir / f"rl_prompt_pool.{spec.split.value}.public.jsonl"
    private_path = dataset_dir / f"rl_verifier.{spec.split.value}.private.jsonl"
    registry_path = dataset_dir / "sample_registry.private.jsonl"
    rollout_paths = {
        item.system_id: _resolve(spec_base_dir, item.rollout_path) for item in spec.systems
    }
    public_source_paths = {
        "dataset_manifest": dataset_dir / "manifest.json",
        "public": public_path,
        "sample_manifest": sample_manifest_path,
        **{f"rollout/{key}": value for key, value in rollout_paths.items()},
    }
    private_source_paths = {
        "private": private_path,
        "registry": registry_path,
    }
    # Do not even hash a trainer-private file before the public decisions have
    # been frozen: hashing would still read its bytes and weaken the isolation
    # claim.  The private hashes are captured at the start of Stage B instead.
    hashes_before = {key: _file_sha256(value) for key, value in public_source_paths.items()}

    progress("步骤 1/6：加载固定 public manifest 与 public prompts；此阶段不打开 trainer-private。")
    sample_manifest = load_replacement_sample_manifest(sample_manifest_path)
    if sample_manifest.dataset_build_name != dataset_dir.name or sample_manifest.split != spec.split:
        raise ValueError("sample manifest does not identify the configured dataset split")
    if sample_manifest.source_public_sha256 != hashes_before["public"]:
        raise ValueError("sample manifest public source hash no longer matches the dataset")
    public_all = _load_index(public_path, PublicRLPromptRecord, label="public")
    selected = {item.sample_id: item for item in sample_manifest.samples}
    if not set(selected) <= set(public_all):
        raise ValueError("sample manifest contains unknown public sample IDs")
    public_by_id = {sample_id: public_all[sample_id] for sample_id in selected}
    for sample_id, item in selected.items():
        public = public_by_id[sample_id]
        if item.public_prompt_sha256 != public.public_prompt_sha256:
            raise ValueError(f"sample manifest prompt hash mismatch: {sample_id}")
        forbidden = forbidden_public_key_paths(public.public_prompt.model_dump(mode="json", by_alias=True))
        denied = _denied_key_paths(public.public_prompt.model_dump(mode="json", by_alias=True))
        if forbidden or denied:
            raise ValueError(f"public compiler prompt contains private keys: {forbidden + denied}")
    progress(f"步骤 1/6 完成：固定样本={len(selected)}，episode ID=0，gold 字段=0。")

    progress("步骤 2/6：固定唯一 SCTP/CIV/Pattern/retrieval 下游，三套编译器共享同一实例。")
    if downstream_evaluator is None:
        downstream_evaluator = FrozenTAIRAMTDownstream.from_spec(
            spec.downstream,
            base_dir=spec_base_dir,
            domains=[item.domain for item in sample_manifest.samples],
        )
    downstream_identity = dict(downstream_evaluator.identity)
    denied_downstream = _denied_key_paths(downstream_identity)
    if denied_downstream:
        raise ValueError(f"downstream identity contains private/episode keys: {denied_downstream}")
    progress(
        "步骤 2/6 完成："
        f"Pattern={downstream_identity.get('pattern_library_id', 'injected')}，"
        f"CIV={downstream_identity.get('civ_model_id', 'injected')}。"
    )

    progress("步骤 3/6：只用 public history 重放三套输出并冻结 Pattern frontier 与最终动作。")
    public_rows: dict[tuple[str, str], PublicStageEvaluation] = {}
    for system in spec.systems:
        rollout_by_id = _load_index(
            rollout_paths[system.system_id], CompilerReplacementRollout, label=f"rollout/{system.system_id}"
        )
        if spec.require_complete and set(rollout_by_id) != set(selected):
            raise ValueError(f"{system.system_id} rollout does not exactly cover the fixed manifest")
        if not set(rollout_by_id) <= set(selected):
            raise ValueError(f"{system.system_id} rollout contains samples outside the fixed manifest")
        for sample_id in sorted(rollout_by_id):
            rollout = rollout_by_id[sample_id]
            public = public_by_id[sample_id]
            if rollout.public_prompt_sha256 != public.public_prompt_sha256:
                raise ValueError(f"rollout/public prompt hash mismatch: {system.system_id}/{sample_id}")
            public_rows[(system.system_id, sample_id)] = _evaluate_public_rollout(
                system_id=system.system_id,
                public=public,
                domain=selected[sample_id].domain,
                rollout=rollout,
                downstream=downstream_evaluator,
                fallback_to_rule=spec.downstream.fallback_to_rule,
            )
        progress(f"  {system.system_id}: public downstream={len(rollout_by_id)} 条。")
    progress("步骤 3/6 完成：public 下游快照已冻结；现在才允许进入 private 计分阶段。")

    progress("步骤 4/6：首次打开 trainer-private，只连接 gold next state 并计算状态/下游指标。")
    hashes_before.update(
        {key: _file_sha256(value) for key, value in private_source_paths.items()}
    )
    if sample_manifest.source_registry_sha256 != hashes_before["registry"]:
        raise ValueError("sample manifest registry source hash no longer matches the dataset")
    private_all = _load_index(private_path, PrivateRLVerifierRecord, label="private")
    if not set(selected) <= set(private_all):
        raise ValueError("trainer-private verifier misses fixed manifest samples")
    reference_id = spec.reference_system_id
    gold_downstream: dict[str, DownstreamDecisionSnapshot | None] = {}
    for sample_id in sorted(selected):
        private = private_all[sample_id]
        public = public_by_id[sample_id]
        if private.public_prompt_sha256 != public.public_prompt_sha256:
            raise ValueError(f"public/private prompt hash mismatch: {sample_id}")
        gold_downstream[sample_id] = downstream_evaluator.evaluate(
            public_record=public,
            domain=selected[sample_id].domain,
            compiler_state=private.trainer_private.gold_next_state,
            use_rule_fallback=False,
        )
    scored_by_system: dict[str, list[PerExampleReplacementEvaluation]] = {}
    for system in spec.systems:
        rows: list[PerExampleReplacementEvaluation] = []
        for sample_id in sorted(selected):
            stage = public_rows[(system.system_id, sample_id)]
            reference = public_rows[(reference_id, sample_id)]
            gold = private_all[sample_id].trainer_private.gold_next_state
            candidate = stage.candidate_next_state
            next_exact = float(
                candidate is not None
                and canonical_json(_evidence_alpha_payload(candidate))
                == canonical_json(_evidence_alpha_payload(gold))
            )
            rows.append(
                PerExampleReplacementEvaluation(
                    system_id=system.system_id,
                    sample_id=sample_id,
                    json_schema_validity=float(stage.json_schema_valid),
                    evidence_grounding=float(stage.evidence_grounded),
                    next_state_exact=next_exact,
                    field_f1=_counter_f1(_state_field_atoms(candidate), _state_field_atoms(gold)),
                    control_state_accuracy=_control_accuracy(candidate, gold),
                    pattern_frontier_agreement=_frontier_agreement(stage.downstream, reference.downstream),
                    action_agreement=_action_agreement(stage.downstream, reference.downstream),
                    pattern_frontier_agreement_to_gold=_frontier_agreement(
                        stage.downstream, gold_downstream[sample_id]
                    ),
                    action_accuracy_to_gold=_action_agreement(
                        stage.downstream, gold_downstream[sample_id]
                    ),
                    end_to_end_success_proxy=_end_to_end_proxy(
                        stage.downstream, gold_downstream[sample_id]
                    ),
                    fallback_free=float(not stage.fallback_used),
                    latency_ms=None if stage.runtime is None else stage.runtime.latency_ms,
                    total_tokens=None if stage.runtime is None else stage.runtime.total_tokens,
                )
            )
        scored_by_system[system.system_id] = rows
    progress("步骤 4/6 完成：gold 只参与 post-freeze scoring，没有回流到编译或下游决策。")

    progress("步骤 5/6：汇总三系统，并在相同 sample IDs 上做 paired bootstrap。")
    systems = [
        ReplacementSystemSummary(
            system_id=system.system_id,
            display_name=system.display_name,
            checkpoint_stage=system.checkpoint_stage,
            rollout_file_sha256=hashes_before[f"rollout/{system.system_id}"],
            metrics=_aggregate(scored_by_system[system.system_id]),
        )
        for system in spec.systems
    ]
    comparisons = spec.comparisons or [
        PairedReplacementSpec(
            challenger_system_id=system.system_id,
            reference_system_id=reference_id,
        )
        for system in spec.systems
        if system.system_id != reference_id
    ]
    paired_results: list[PairedReplacementResult] = []
    for comparison in comparisons:
        challenger = {row.sample_id: row for row in scored_by_system[comparison.challenger_system_id]}
        reference = {row.sample_id: row for row in scored_by_system[comparison.reference_system_id]}
        metrics = []
        for metric in (*PRIMARY_METRICS, "latency_ms", "total_tokens"):
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
            result = _bootstrap_metric(
                metric=metric,
                challenger=challenger,
                reference=reference,
                resamples=spec.bootstrap_resamples,
                confidence_level=spec.confidence_level,
                seed=derived_seed,
            )
            if result is not None:
                metrics.append(result)
        paired_results.append(
            PairedReplacementResult(
                challenger_system_id=comparison.challenger_system_id,
                reference_system_id=comparison.reference_system_id,
                metrics=metrics,
            )
        )
    progress(f"步骤 5/6 完成：系统={len(systems)}，paired comparisons={len(paired_results)}。")

    progress("步骤 6/6：核对源文件未变，写入 paper 表、private ledger 与内容寻址 manifest。")
    source_paths = {**public_source_paths, **private_source_paths}
    hashes_after = {key: _file_sha256(value) for key, value in source_paths.items()}
    if hashes_after != hashes_before:
        raise ValueError("dataset/manifest/rollout changed during replacement evaluation")
    identity = {
        "schema_version": REPLACEMENT_EVALUATION_VERSION,
        "dataset_build_name": dataset_dir.name,
        "split": spec.split.value,
        "sample_manifest_id": sample_manifest.sample_manifest_id,
        "systems": [item.model_dump(mode="json") for item in spec.systems],
        "reference_system_id": reference_id,
        "downstream_identity": downstream_identity,
        "source_hashes": hashes_before,
        "bootstrap_resamples": spec.bootstrap_resamples,
        "bootstrap_seed": spec.bootstrap_seed,
        "confidence_level": spec.confidence_level,
    }
    run_id = f"dsus-compiler-swap-{content_sha256(identity)[:16]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report = CompilerReplacementEvaluationReport(
        dataset_build_name=dataset_dir.name,
        split=spec.split,
        sample_manifest_id=sample_manifest.sample_manifest_id,
        sample_count=len(selected),
        reference_system_id=reference_id,
        downstream_identity=downstream_identity,
        systems=systems,
        paired_comparisons=paired_results,
    )
    public_stage_rows = [
        public_rows[(system.system_id, sample_id)]
        for system in spec.systems
        for sample_id in sorted(selected)
    ]
    private_rows = [
        row for system in spec.systems for row in scored_by_system[system.system_id]
    ]
    _write_idempotently(
        run_dir / "public_stage.audit.jsonl",
        "".join(canonical_json(row.model_dump(mode="json")) + "\n" for row in public_stage_rows),
    )
    _write_idempotently(
        run_dir / "per_example.private.jsonl",
        "".join(canonical_json(row.model_dump(mode="json")) + "\n" for row in private_rows),
    )
    _write_idempotently(
        run_dir / "paper_results.json",
        canonical_json(report.model_dump(mode="json")) + "\n",
    )
    overall_rows = [
        {
            "system_id": item.system_id,
            "display_name": item.display_name,
            "checkpoint_stage": item.checkpoint_stage,
            **item.metrics.model_dump(mode="json"),
        }
        for item in systems
    ]
    paired_rows = [
        {
            "challenger_system_id": item.challenger_system_id,
            "reference_system_id": item.reference_system_id,
            **metric.model_dump(mode="json"),
        }
        for item in paired_results
        for metric in item.metrics
    ]
    _write_idempotently(run_dir / "paper_overall.csv", _csv_text(overall_rows))
    _write_idempotently(run_dir / "paper_paired_bootstrap.csv", _csv_text(paired_rows))
    _write_idempotently(
        run_dir / "resolved_spec.private.json",
        canonical_json({**identity, "run_id": run_id}) + "\n",
    )
    _write_idempotently(
        run_dir / "summary.json",
        canonical_json(
            {
                "schema_version": REPLACEMENT_EVALUATION_VERSION,
                "run_id": run_id,
                "sample_count": len(selected),
                "system_count": len(systems),
                "comparison_count": len(paired_results),
                "public_stage_completed_before_private_join": True,
                "episode_id_exposed_to_compiler": False,
                "gold_exposed_to_compiler": False,
                "external_calls": False,
                "training_started": False,
            }
        )
        + "\n",
    )
    manifest = {
        "schema_version": REPLACEMENT_EVALUATION_VERSION,
        "artifact_role": "dsus_state_compiler_near_lossless_replacement_evaluation",
        "run_id": run_id,
        "identity": identity,
        "source_artifacts_mutated": False,
        "external_calls": False,
        "training_started": False,
        "outputs": _output_ledger(run_dir),
    }
    _write_idempotently(run_dir / "manifest.json", canonical_json(manifest) + "\n")
    progress(f"步骤 6/6 完成：近无损替换评测工件={run_dir}。")
    return CompilerReplacementEvaluationResult(run_id=run_id, run_dir=str(run_dir), report=report)


__all__ = [
    "PRIMARY_METRICS",
    "REPLACEMENT_EVALUATION_SPEC_VERSION",
    "REPLACEMENT_EVALUATION_VERSION",
    "REPLACEMENT_SAMPLE_MANIFEST_VERSION",
    "CompilerReplacementEvaluationReport",
    "CompilerReplacementEvaluationResult",
    "CompilerReplacementEvaluationSpec",
    "CompilerReplacementRollout",
    "DownstreamDecisionSnapshot",
    "DownstreamEvaluator",
    "FrozenCompilerSample",
    "FrozenDownstreamSpec",
    "FrozenTAIRAMTDownstream",
    "PairedReplacementResult",
    "PairedReplacementSpec",
    "ReplacementMetricSummary",
    "ReplacementSampleManifest",
    "ReplacementSystemSpec",
    "freeze_replacement_sample_manifest",
    "load_replacement_evaluation_spec",
    "load_replacement_sample_manifest",
    "public_prompt_to_memory",
    "run_compiler_replacement_evaluation",
]
