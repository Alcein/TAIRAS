"""Validated small-batch execution for the fixed benchmark language plan."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from taira_mt.benchmark.diagnostics import LoadedPilotCase
from taira_mt.benchmark.language_plan import (
    BENCHMARK_LANGUAGE_PLAN_VERSION,
    BENCHMARK_LANGUAGE_SYSTEM_PROMPT,
    BenchmarkLanguageTask,
    BenchmarkLanguageTaskKind,
    language_task_reuse_signature,
    provider_payload,
    unmask_query_variant,
    validate_language_variants,
)
from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.providers import JsonCompletion, JsonCompletionTransport, JsonTokenUsage

BENCHMARK_LANGUAGE_EXECUTION_VERSION = "user_benchmark_language_execution_v1"


class BenchmarkLanguageAttempt(StrictModel):
    attempt_index: int = Field(ge=1)
    status: str
    provider_request_id: str | None = None
    requested_model: str | None = None
    returned_model: str | None = None
    finish_reason: str | None = None
    raw_content: str | None = None
    raw_content_sha256: str | None = None
    usage: JsonTokenUsage | None = None
    error_type: str | None = None
    error_message: str | None = None


class BenchmarkLanguageExecution(StrictModel):
    schema_version: str = BENCHMARK_LANGUAGE_EXECUTION_VERSION
    task_id: str
    task_kind: BenchmarkLanguageTaskKind
    linkage_id: str
    status: str
    attempts: list[BenchmarkLanguageAttempt]
    accepted_masked_variants: list[str] = Field(default_factory=list)
    realized_query_variants: list[str] = Field(default_factory=list)
    realization_origin: str = "external_generation"
    source_task_id: str | None = None
    source_execution_id: str | None = None
    reuse_signature_sha256: str | None = None


class BenchmarkLanguageExecutionSummary(StrictModel):
    schema_version: str = BENCHMARK_LANGUAGE_EXECUTION_VERSION
    execution_id: str
    parent_language_plan_build_id: str
    parent_benchmark_build_id: str
    selected_task_count: int = Field(ge=1)
    selected_query_task_count: int = Field(ge=0)
    selected_response_task_count: int = Field(ge=0)
    committed_task_count: int = Field(ge=0)
    failed_task_count: int = Field(ge=0)
    first_attempt_commit_rate: float = Field(ge=0.0, le=1.0)
    total_attempt_count: int = Field(ge=1)
    usage: JsonTokenUsage
    returned_model_counts: dict[str, int]
    error_type_counts: dict[str, int]
    external_calls: bool = True


@dataclass(frozen=True)
class BenchmarkLanguageExecutionResult:
    execution_id: str
    records: tuple[BenchmarkLanguageExecution, ...]
    summary: BenchmarkLanguageExecutionSummary


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark_language_plan(
    plan_dir: Path,
) -> tuple[dict[str, Any], tuple[BenchmarkLanguageTask, ...]]:
    """Fail closed on an incomplete or modified language-plan artifact."""

    plan_dir = plan_dir.resolve()
    manifest_path = plan_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BENCHMARK_LANGUAGE_PLAN_VERSION:
        raise ValueError("unsupported benchmark language plan manifest")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("benchmark language plan manifest lacks files")
    declared: dict[str, dict[str, Any]] = {}
    for record in files:
        if not isinstance(record, dict) or not isinstance(record.get("relative_path"), str):
            raise ValueError("invalid language-plan file record")
        relative_path = record["relative_path"]
        path = (plan_dir / relative_path).resolve()
        if plan_dir not in path.parents or not path.is_file():
            raise ValueError("language-plan file escapes or is missing")
        if _file_sha256(path) != record.get("sha256"):
            raise ValueError(f"language-plan file hash mismatch: {relative_path}")
        if path.stat().st_size != record.get("byte_size"):
            raise ValueError(f"language-plan file size mismatch: {relative_path}")
        declared[relative_path] = record
    if "tasks.public.jsonl" not in declared:
        raise ValueError("language plan does not declare tasks.public.jsonl")
    tasks = tuple(
        BenchmarkLanguageTask.model_validate(json.loads(line))
        for line in (plan_dir / "tasks.public.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    )
    if not tasks or len({item.task_id for item in tasks}) != len(tasks):
        raise ValueError("language-plan tasks are empty or duplicated")
    if content_sha256([item.model_dump(mode="json") for item in tasks]) != manifest.get(
        "tasks_sha256"
    ):
        raise ValueError("language-plan task content differs from manifest")
    return manifest, tasks


def select_language_calibration_tasks(
    tasks: tuple[BenchmarkLanguageTask, ...],
    cases: tuple[LoadedPilotCase, ...],
    *,
    seed: int,
    queries_per_domain_scenario: int,
    responses_per_register: int,
    required_response_keys: tuple[str, ...] = (),
) -> tuple[BenchmarkLanguageTask, ...]:
    """Select fixed semantic strata without consulting outcomes or model output."""

    if queries_per_domain_scenario < 1 or responses_per_register < 1:
        raise ValueError("language calibration stratum counts must be positive")
    if len(required_response_keys) != len(set(required_response_keys)):
        raise ValueError("required response keys must be unique")
    if len(required_response_keys) > responses_per_register:
        raise ValueError("response calibration budget is smaller than the required keys")
    case_strata = {
        item.pair.public.case_id: (
            item.pair.public.domain,
            item.pair.private.initial_state.scenario_family.value,
        )
        for item in cases
    }
    query_groups: dict[tuple[str, str], list[BenchmarkLanguageTask]] = defaultdict(list)
    response_groups: dict[str, list[BenchmarkLanguageTask]] = defaultdict(list)
    for task in tasks:
        if not task.pilot_selected:
            continue
        if task.task_kind == BenchmarkLanguageTaskKind.INITIAL_QUERY:
            try:
                stratum = case_strata[task.linkage_id]
            except KeyError as exc:
                raise ValueError("language query task does not link to the benchmark") from exc
            query_groups[stratum].append(task)
        else:
            response_groups[task.language_register.value].append(task)

    selected: list[BenchmarkLanguageTask] = []
    for stratum, values in sorted(query_groups.items()):
        ordered = sorted(
            values,
            key=lambda item: content_sha256(
                {"seed": seed, "stratum": stratum, "task_id": item.task_id}
            ),
        )
        if len(ordered) < queries_per_domain_scenario:
            raise ValueError(f"insufficient query calibration tasks for {stratum}")
        selected.extend(ordered[:queries_per_domain_scenario])
    if len(query_groups) != 15:
        raise ValueError("language calibration must cover three domains by five scenarios")

    for register, values in sorted(response_groups.items()):
        by_key = {item.linkage_id: item for item in values}
        missing = [key for key in required_response_keys if key not in by_key]
        if missing:
            raise ValueError(
                f"required response calibration keys are absent for {register}: {missing}"
            )
        required = [by_key[key] for key in required_response_keys]
        required_ids = {item.task_id for item in required}
        ordered = sorted(
            [item for item in values if item.task_id not in required_ids],
            key=lambda item: content_sha256(
                {"seed": seed, "register": register, "task_id": item.task_id}
            ),
        )
        remaining_count = responses_per_register - len(required)
        if len(ordered) < remaining_count:
            raise ValueError(f"insufficient response calibration tasks for {register}")
        selected.extend([*required, *ordered[:remaining_count]])
    if len(response_groups) != 4:
        raise ValueError("language calibration must cover four language registers")
    return tuple(sorted(selected, key=lambda item: item.task_id))


def _parse_json_object(content: str) -> dict[str, object]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("provider content is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("provider JSON root must be an object")
    return value


def execute_benchmark_language_tasks(
    tasks: tuple[BenchmarkLanguageTask, ...],
    transport: JsonCompletionTransport,
    *,
    parent_language_plan_build_id: str,
    parent_benchmark_build_id: str,
    max_attempts: int = 2,
    progress: Callable[[int, int, BenchmarkLanguageExecution], None] | None = None,
) -> BenchmarkLanguageExecutionResult:
    if not tasks or max_attempts < 1:
        raise ValueError("language execution needs tasks and at least one attempt")
    records: list[BenchmarkLanguageExecution] = []
    for task in tasks:
        attempts: list[BenchmarkLanguageAttempt] = []
        accepted: list[str] = []
        retry_feedback: str | None = None
        for attempt_index in range(1, max_attempts + 1):
            completion: JsonCompletion | None = None
            try:
                completion = transport.complete_json(
                    system_prompt=BENCHMARK_LANGUAGE_SYSTEM_PROMPT,
                    user_payload=provider_payload(task),
                    retry_feedback=retry_feedback,
                )
                accepted = validate_language_variants(
                    task,
                    _parse_json_object(completion.content),
                )
                attempts.append(
                    BenchmarkLanguageAttempt(
                        attempt_index=attempt_index,
                        status="committed",
                        provider_request_id=completion.provider_request_id,
                        requested_model=completion.requested_model,
                        returned_model=completion.returned_model,
                        finish_reason=completion.finish_reason,
                        raw_content=completion.content,
                        raw_content_sha256=content_sha256(completion.content),
                        usage=completion.usage,
                    )
                )
                break
            except Exception as exc:
                attempts.append(
                    BenchmarkLanguageAttempt(
                        attempt_index=attempt_index,
                        status="rejected",
                        provider_request_id=(
                            completion.provider_request_id if completion is not None else None
                        ),
                        requested_model=(
                            completion.requested_model if completion is not None else None
                        ),
                        returned_model=(
                            completion.returned_model if completion is not None else None
                        ),
                        finish_reason=(completion.finish_reason if completion is not None else None),
                        raw_content=(completion.content if completion is not None else None),
                        raw_content_sha256=(
                            content_sha256(completion.content) if completion is not None else None
                        ),
                        usage=(completion.usage if completion is not None else None),
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:500],
                    )
                )
                complete_sentence_tokens = [
                    item.token
                    for item in task.placeholders
                    if item.semantic_type == "authorized_constraint"
                ]
                semantic_retry = (
                    " For an unverifiable-response task, every variant must explicitly say "
                    "that available evidence is insufficient using wording such as cannot "
                    "verify, cannot confirm, cannot determine, or cannot check. Never say or "
                    "imply that the item itself fails, does not match, or does not satisfy."
                    if task.linkage_id.startswith("reject_unverifiable_")
                    else ""
                )
                answer_retry = (
                    " For an answer-response task, write a shopper's direct statement of the "
                    "protected preference. Do not ask a question, request clarification, seek "
                    "confirmation, or use instruction labels such as authorized, precommitted, "
                    "fact, answer, or constraint."
                    if task.linkage_id
                    in {"answer_relevant_question", "answer_registered_tradeoff_question"}
                    else ""
                )
                sentence_placeholder_retry = (
                    " A protected complete shopper sentence is already a full clause. Put "
                    "its token at the start or after a colon, dash, or sentence boundary. "
                    f"Valid: `For clarity: {complete_sentence_tokens[0]}.` and "
                    f"`{complete_sentence_tokens[0]}.` Invalid: "
                    f"`I prefer {complete_sentence_tokens[0]}.`, "
                    f"`that {complete_sentence_tokens[0]}.`, and "
                    f"`{complete_sentence_tokens[0]} works.` "
                    "Apply this boundary rule to every variant, while preserving the original "
                    "dialogue act."
                    if complete_sentence_tokens
                    else ""
                )
                response_act_retry = (
                    " Every reminder variant must explicitly signal that the preference was "
                    "said earlier, using wording such as already, earlier, previously, "
                    "reiterate, or remind."
                    if task.linkage_id == "remind_already_disclosed_constraint"
                    else (
                        " Every revision variant must explicitly signal that seeing the option "
                        "clarified, updated, changed, or refined the shopper's need."
                        if task.linkage_id == "apply_precommitted_goal_revision"
                        else ""
                    )
                )
                retry_feedback = (
                    "Your previous JSON failed a local invariant. Return the complete JSON again; "
                    "preserve every placeholder exactly and introduce no new fact or number. "
                    f"Failure: {type(exc).__name__}: {str(exc)[:220]}"
                    f"{semantic_retry}{answer_retry}{sentence_placeholder_retry}"
                    f"{response_act_retry}"
                )
        record = BenchmarkLanguageExecution(
            task_id=task.task_id,
            task_kind=task.task_kind,
            linkage_id=task.linkage_id,
            status="committed" if accepted else "failed",
            attempts=attempts,
            accepted_masked_variants=accepted,
            realized_query_variants=(
                [unmask_query_variant(task, item) for item in accepted]
                if accepted and task.task_kind == BenchmarkLanguageTaskKind.INITIAL_QUERY
                else []
            ),
        )
        records.append(record)
        if progress is not None:
            progress(len(records), len(tasks), record)

    usage_records = [attempt.usage for item in records for attempt in item.attempts if attempt.usage]
    usage = JsonTokenUsage(
        input_tokens=sum(item.input_tokens for item in usage_records),
        output_tokens=sum(item.output_tokens for item in usage_records),
        total_tokens=sum(item.total_tokens for item in usage_records),
        reasoning_tokens=(
            sum(item.reasoning_tokens or 0 for item in usage_records)
            if any(item.reasoning_tokens is not None for item in usage_records)
            else None
        ),
    )
    returned_models = Counter(
        attempt.returned_model
        for item in records
        for attempt in item.attempts
        if attempt.returned_model is not None
    )
    errors = Counter(
        attempt.error_type
        for item in records
        for attempt in item.attempts
        if attempt.error_type is not None
    )
    identity = {
        "schema_version": BENCHMARK_LANGUAGE_EXECUTION_VERSION,
        "parent_language_plan_build_id": parent_language_plan_build_id,
        "parent_benchmark_build_id": parent_benchmark_build_id,
        "records_sha256": content_sha256([item.model_dump(mode="json") for item in records]),
    }
    execution_id = f"benchmark-language-execution-{content_sha256(identity)[:16]}"
    summary = BenchmarkLanguageExecutionSummary(
        execution_id=execution_id,
        parent_language_plan_build_id=parent_language_plan_build_id,
        parent_benchmark_build_id=parent_benchmark_build_id,
        selected_task_count=len(records),
        selected_query_task_count=sum(
            item.task_kind == BenchmarkLanguageTaskKind.INITIAL_QUERY for item in records
        ),
        selected_response_task_count=sum(
            item.task_kind == BenchmarkLanguageTaskKind.RESPONSE_TEMPLATE for item in records
        ),
        committed_task_count=sum(item.status == "committed" for item in records),
        failed_task_count=sum(item.status == "failed" for item in records),
        first_attempt_commit_rate=sum(
            item.attempts[0].status == "committed" for item in records
        )
        / len(records),
        total_attempt_count=sum(len(item.attempts) for item in records),
        usage=usage,
        returned_model_counts=dict(sorted(returned_models.items())),
        error_type_counts=dict(sorted(errors.items())),
    )
    return BenchmarkLanguageExecutionResult(
        execution_id=execution_id,
        records=tuple(records),
        summary=summary,
    )


def reuse_benchmark_language_tasks(
    target_tasks: tuple[BenchmarkLanguageTask, ...],
    source_tasks: tuple[BenchmarkLanguageTask, ...],
    source_summary: BenchmarkLanguageExecutionSummary,
    source_records: tuple[BenchmarkLanguageExecution, ...],
    *,
    parent_language_plan_build_id: str,
    parent_benchmark_build_id: str,
) -> BenchmarkLanguageExecutionResult:
    """Reuse validated masked wording when provider-visible semantics are identical."""

    if not target_tasks or not source_tasks or not source_records:
        raise ValueError("language reuse requires target tasks and complete source inputs")
    source_task_by_id = {item.task_id: item for item in source_tasks}
    source_record_by_id = {item.task_id: item for item in source_records}
    if len(source_task_by_id) != len(source_tasks) or len(source_record_by_id) != len(
        source_records
    ):
        raise ValueError("language reuse source contains duplicate task IDs")
    if set(source_task_by_id) != set(source_record_by_id):
        raise ValueError("language reuse source plan and execution coverage differ")
    if source_summary.committed_task_count != len(source_records):
        raise ValueError("language reuse requires a completely committed source execution")

    by_signature: dict[str, list[BenchmarkLanguageTask]] = defaultdict(list)
    for source_task in source_tasks:
        source_record = source_record_by_id[source_task.task_id]
        if source_record.status != "committed":
            raise ValueError("language reuse source includes a failed task")
        by_signature[language_task_reuse_signature(source_task)].append(source_task)

    records: list[BenchmarkLanguageExecution] = []
    reuse_kinds: Counter[str] = Counter()
    for target in sorted(target_tasks, key=lambda item: item.task_id):
        signature = language_task_reuse_signature(target)
        if target.task_id in source_task_by_id:
            candidates = [source_task_by_id[target.task_id]]
            origin = "exact_task_reuse"
        else:
            candidates = sorted(by_signature.get(signature, []), key=lambda item: item.task_id)
            origin = "semantic_signature_reuse"
        accepted: list[str] | None = None
        selected_source: BenchmarkLanguageTask | None = None
        for candidate in candidates:
            source_record = source_record_by_id[candidate.task_id]
            try:
                accepted = validate_language_variants(
                    target,
                    {"variants": source_record.accepted_masked_variants},
                )
            except Exception:
                continue
            selected_source = candidate
            break
        if selected_source is None or accepted is None:
            raise ValueError(f"no validated language reuse source for target task: {target.task_id}")
        reuse_kinds[origin] += 1
        records.append(
            BenchmarkLanguageExecution(
                task_id=target.task_id,
                task_kind=target.task_kind,
                linkage_id=target.linkage_id,
                status="committed",
                attempts=[
                    BenchmarkLanguageAttempt(
                        attempt_index=1,
                        status="reused",
                        raw_content_sha256=content_sha256(accepted),
                    )
                ],
                accepted_masked_variants=accepted,
                realized_query_variants=(
                    [unmask_query_variant(target, item) for item in accepted]
                    if target.task_kind == BenchmarkLanguageTaskKind.INITIAL_QUERY
                    else []
                ),
                realization_origin=origin,
                source_task_id=selected_source.task_id,
                source_execution_id=source_summary.execution_id,
                reuse_signature_sha256=signature,
            )
        )

    identity = {
        "schema_version": BENCHMARK_LANGUAGE_EXECUTION_VERSION,
        "realization_origin": "offline_validated_reuse",
        "parent_language_plan_build_id": parent_language_plan_build_id,
        "parent_benchmark_build_id": parent_benchmark_build_id,
        "source_execution_id": source_summary.execution_id,
        "records_sha256": content_sha256([item.model_dump(mode="json") for item in records]),
    }
    execution_id = f"benchmark-language-reuse-{content_sha256(identity)[:16]}"
    summary = BenchmarkLanguageExecutionSummary(
        execution_id=execution_id,
        parent_language_plan_build_id=parent_language_plan_build_id,
        parent_benchmark_build_id=parent_benchmark_build_id,
        selected_task_count=len(records),
        selected_query_task_count=sum(
            item.task_kind == BenchmarkLanguageTaskKind.INITIAL_QUERY for item in records
        ),
        selected_response_task_count=sum(
            item.task_kind == BenchmarkLanguageTaskKind.RESPONSE_TEMPLATE for item in records
        ),
        committed_task_count=len(records),
        failed_task_count=0,
        first_attempt_commit_rate=1.0,
        total_attempt_count=len(records),
        usage=JsonTokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        returned_model_counts={},
        error_type_counts={},
        external_calls=False,
    )
    if sum(reuse_kinds.values()) != len(records):
        raise AssertionError("language reuse accounting differs from target coverage")
    return BenchmarkLanguageExecutionResult(
        execution_id=execution_id,
        records=tuple(records),
        summary=summary,
    )


def complete_benchmark_language_tasks_with_supplement(
    target_tasks: tuple[BenchmarkLanguageTask, ...],
    source_tasks: tuple[BenchmarkLanguageTask, ...],
    source_summary: BenchmarkLanguageExecutionSummary,
    source_records: tuple[BenchmarkLanguageExecution, ...],
    supplement_summary: BenchmarkLanguageExecutionSummary,
    supplement_records: tuple[BenchmarkLanguageExecution, ...],
    *,
    parent_language_plan_build_id: str,
    parent_benchmark_build_id: str,
) -> BenchmarkLanguageExecutionResult:
    """Complete semantic reuse with a small exact target-plan external supplement."""

    if supplement_summary.parent_language_plan_build_id != parent_language_plan_build_id:
        raise ValueError("language supplement and target plan build IDs differ")
    if supplement_summary.parent_benchmark_build_id != parent_benchmark_build_id:
        raise ValueError("language supplement and target benchmark build IDs differ")
    if not supplement_summary.external_calls:
        raise ValueError("language supplement must retain its external-call provenance")
    if supplement_summary.committed_task_count != len(supplement_records):
        raise ValueError("language supplement must contain only committed records")
    supplement_by_id = {item.task_id: item for item in supplement_records}
    if len(supplement_by_id) != len(supplement_records):
        raise ValueError("language supplement contains duplicate task IDs")
    target_by_id = {item.task_id: item for item in target_tasks}
    if not supplement_by_id or not set(supplement_by_id).issubset(target_by_id):
        raise ValueError("language supplement contains a task outside the target plan")
    source_ids = {item.task_id for item in source_tasks}
    if source_ids.intersection(supplement_by_id):
        raise ValueError("language supplement redundantly overlaps the reuse source")
    for task_id, record in supplement_by_id.items():
        task = target_by_id[task_id]
        if (
            record.status != "committed"
            or record.task_kind != task.task_kind
            or record.linkage_id != task.linkage_id
        ):
            raise ValueError("language supplement record differs from its target task")
        validate_language_variants(
            task,
            {"variants": record.accepted_masked_variants},
        )

    combined_tasks = (*source_tasks, *(target_by_id[value] for value in supplement_by_id))
    combined_records = (*source_records, *supplement_records)
    combined_summary = source_summary.model_copy(
        update={
            "execution_id": (
                f"combined-{source_summary.execution_id}-{supplement_summary.execution_id}"
            ),
            "selected_task_count": len(combined_records),
            "committed_task_count": len(combined_records),
            "failed_task_count": 0,
        },
        deep=True,
    )
    reused = reuse_benchmark_language_tasks(
        target_tasks,
        combined_tasks,
        combined_summary,
        combined_records,
        parent_language_plan_build_id=parent_language_plan_build_id,
        parent_benchmark_build_id=parent_benchmark_build_id,
    )
    records: list[BenchmarkLanguageExecution] = []
    for record in reused.records:
        supplement = supplement_by_id.get(record.task_id)
        if supplement is not None:
            records.append(
                supplement.model_copy(
                    update={
                        "realization_origin": "target_external_supplement",
                        "source_task_id": supplement.task_id,
                        "source_execution_id": supplement_summary.execution_id,
                        "reuse_signature_sha256": language_task_reuse_signature(
                            target_by_id[supplement.task_id]
                        ),
                    },
                    deep=True,
                )
            )
        else:
            records.append(
                record.model_copy(
                    update={"source_execution_id": source_summary.execution_id},
                    deep=True,
                )
            )
    identity = {
        "schema_version": BENCHMARK_LANGUAGE_EXECUTION_VERSION,
        "realization_origin": "offline_reuse_with_external_supplement",
        "parent_language_plan_build_id": parent_language_plan_build_id,
        "parent_benchmark_build_id": parent_benchmark_build_id,
        "source_execution_id": source_summary.execution_id,
        "supplement_execution_id": supplement_summary.execution_id,
        "source_summary_sha256": content_sha256(source_summary.model_dump(mode="json")),
        "supplement_summary_sha256": content_sha256(
            supplement_summary.model_dump(mode="json")
        ),
        "records_sha256": content_sha256(
            [item.model_dump(mode="json") for item in records]
        ),
    }
    execution_id = f"benchmark-language-hybrid-{content_sha256(identity)[:16]}"
    source_usage = source_summary.usage
    supplement_usage = supplement_summary.usage
    combined_usage = JsonTokenUsage(
        input_tokens=source_usage.input_tokens + supplement_usage.input_tokens,
        output_tokens=source_usage.output_tokens + supplement_usage.output_tokens,
        total_tokens=source_usage.total_tokens + supplement_usage.total_tokens,
        reasoning_tokens=(
            (source_usage.reasoning_tokens or 0)
            + (supplement_usage.reasoning_tokens or 0)
            if source_usage.reasoning_tokens is not None
            or supplement_usage.reasoning_tokens is not None
            else None
        ),
    )
    returned_models = Counter(source_summary.returned_model_counts)
    returned_models.update(supplement_summary.returned_model_counts)
    error_types = Counter(source_summary.error_type_counts)
    error_types.update(supplement_summary.error_type_counts)
    summary = BenchmarkLanguageExecutionSummary(
        execution_id=execution_id,
        parent_language_plan_build_id=parent_language_plan_build_id,
        parent_benchmark_build_id=parent_benchmark_build_id,
        selected_task_count=len(records),
        selected_query_task_count=sum(
            item.task_kind == BenchmarkLanguageTaskKind.INITIAL_QUERY for item in records
        ),
        selected_response_task_count=sum(
            item.task_kind == BenchmarkLanguageTaskKind.RESPONSE_TEMPLATE for item in records
        ),
        committed_task_count=len(records),
        failed_task_count=0,
        first_attempt_commit_rate=sum(
            item.attempts and item.attempts[0].status in {"committed", "reused"}
            for item in records
        )
        / len(records),
        total_attempt_count=sum(len(item.attempts) for item in records),
        usage=combined_usage,
        returned_model_counts=dict(sorted(returned_models.items())),
        error_type_counts=dict(sorted(error_types.items())),
        external_calls=True,
    )
    return BenchmarkLanguageExecutionResult(
        execution_id=execution_id,
        records=tuple(records),
        summary=summary,
    )


def write_benchmark_language_execution_artifact(
    result: BenchmarkLanguageExecutionResult,
    output_root: Path,
    *,
    config: dict[str, object],
    artifact_role: str = "small_batch_grounded_language_capability_calibration",
) -> Path:
    run_dir = output_root / result.execution_id
    run_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "executions.jsonl": "".join(
            canonical_json(item.model_dump(mode="json")) + "\n" for item in result.records
        ),
        "summary.json": json.dumps(
            result.summary.model_dump(mode="json"), ensure_ascii=False, indent=2
        )
        + "\n",
    }
    for filename, text in outputs.items():
        (run_dir / filename).write_text(text, encoding="utf-8")
    manifest = {
        "schema_version": BENCHMARK_LANGUAGE_EXECUTION_VERSION,
        "artifact_role": artifact_role,
        "execution_id": result.execution_id,
        "status": "complete",
        "config": config,
        "external_calls": result.summary.external_calls,
        "outputs": [
            {
                "relative_path": filename,
                "sha256": _file_sha256(run_dir / filename),
                "byte_size": (run_dir / filename).stat().st_size,
                "line_count": text.count("\n") if filename.endswith(".jsonl") else None,
            }
            for filename, text in sorted(outputs.items())
        ],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir
