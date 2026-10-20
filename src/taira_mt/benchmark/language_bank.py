"""Compile validated external realizations into a frozen offline language bank."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, model_validator

from taira_mt.artifacts import read_jsonl_unique, verify_artifact_bundle
from taira_mt.benchmark.language_execution import (
    BENCHMARK_LANGUAGE_EXECUTION_VERSION,
    BenchmarkLanguageExecution,
    BenchmarkLanguageExecutionSummary,
)
from taira_mt.benchmark.language_plan import (
    BenchmarkLanguageRegister,
    BenchmarkLanguageTask,
    BenchmarkLanguageTaskKind,
    LanguagePlaceholder,
)
from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json, content_sha256

BENCHMARK_LANGUAGE_BANK_VERSION = "user_benchmark_language_bank_v1"


class FrozenInitialQueryVariants(StrictModel):
    task_id: str
    case_id: str
    language_register: BenchmarkLanguageRegister
    variants: list[str] = Field(min_length=1)


class FrozenResponseTemplateVariants(StrictModel):
    task_id: str
    template_key: str
    language_register: BenchmarkLanguageRegister
    placeholders: list[LanguagePlaceholder]
    masked_variants: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_runtime_placeholders(self) -> FrozenResponseTemplateVariants:
        if any(item.replacement_surface is not None for item in self.placeholders):
            raise ValueError("response-bank placeholders cannot contain fixed replacement values")
        return self


class BenchmarkLanguageBankSummary(StrictModel):
    schema_version: str = BENCHMARK_LANGUAGE_BANK_VERSION
    bank_id: str
    parent_language_plan_build_id: str
    parent_language_execution_id: str
    parent_benchmark_build_id: str
    initial_query_count: int = Field(ge=1)
    response_template_count: int = Field(ge=1)
    initial_query_variant_count: int = Field(ge=1)
    response_variant_count: int = Field(ge=1)
    response_key_count: int = Field(ge=1)
    register_count: int = Field(ge=1)
    complete_plan_coverage: bool
    runtime_external_calls: bool = False


@dataclass(frozen=True)
class BenchmarkLanguageBankResult:
    bank_id: str
    queries: tuple[FrozenInitialQueryVariants, ...]
    responses: tuple[FrozenResponseTemplateVariants, ...]
    summary: BenchmarkLanguageBankSummary


def load_benchmark_language_execution(
    execution_dir: Path,
) -> tuple[dict[str, object], BenchmarkLanguageExecutionSummary, tuple[BenchmarkLanguageExecution, ...]]:
    bundle = verify_artifact_bundle(execution_dir)
    if bundle.manifest.get("schema_version") != BENCHMARK_LANGUAGE_EXECUTION_VERSION:
        raise ValueError("unsupported benchmark language execution artifact")
    summary = BenchmarkLanguageExecutionSummary.model_validate(
        json.loads(bundle.require_output("summary.json").read_text(encoding="utf-8"))
    )
    records = read_jsonl_unique(
        bundle.require_output("executions.jsonl"),
        BenchmarkLanguageExecution.model_validate,
        lambda item: item.task_id,
        record_name="benchmark language execution",
    )
    if len(records) != summary.selected_task_count:
        raise ValueError("language execution record count differs from summary")
    return bundle.manifest, summary, tuple(records)


def compile_benchmark_language_bank(
    tasks: tuple[BenchmarkLanguageTask, ...],
    execution_summary: BenchmarkLanguageExecutionSummary,
    executions: tuple[BenchmarkLanguageExecution, ...],
    *,
    parent_language_plan_build_id: str,
    require_complete: bool = True,
) -> BenchmarkLanguageBankResult:
    task_by_id = {item.task_id: item for item in tasks}
    execution_by_id = {item.task_id: item for item in executions}
    if len(task_by_id) != len(tasks) or len(execution_by_id) != len(executions):
        raise ValueError("language bank inputs contain duplicate task IDs")
    if not set(execution_by_id).issubset(task_by_id):
        raise ValueError("language execution contains a task outside the plan")
    complete = set(execution_by_id) == set(task_by_id)
    if require_complete and not complete:
        raise ValueError("final language bank requires complete plan coverage")
    if execution_summary.parent_language_plan_build_id != parent_language_plan_build_id:
        raise ValueError("language execution and plan build IDs differ")

    queries: list[FrozenInitialQueryVariants] = []
    responses: list[FrozenResponseTemplateVariants] = []
    for task_id in sorted(execution_by_id):
        task = task_by_id[task_id]
        execution = execution_by_id[task_id]
        if execution.status != "committed" or not execution.accepted_masked_variants:
            raise ValueError(f"language bank cannot admit failed task: {task_id}")
        if task.task_kind != execution.task_kind or task.linkage_id != execution.linkage_id:
            raise ValueError("language task and execution linkage differs")
        if task.task_kind == BenchmarkLanguageTaskKind.INITIAL_QUERY:
            if len(execution.realized_query_variants) != task.required_variant_count:
                raise ValueError("initial query execution lacks all realized variants")
            queries.append(
                FrozenInitialQueryVariants(
                    task_id=task.task_id,
                    case_id=task.linkage_id,
                    language_register=task.language_register,
                    variants=execution.realized_query_variants,
                )
            )
        else:
            responses.append(
                FrozenResponseTemplateVariants(
                    task_id=task.task_id,
                    template_key=task.linkage_id,
                    language_register=task.language_register,
                    placeholders=task.placeholders,
                    masked_variants=execution.accepted_masked_variants,
                )
            )
    if not queries or not responses:
        raise ValueError("language bank must contain query and response entries")
    identity = {
        "schema_version": BENCHMARK_LANGUAGE_BANK_VERSION,
        "parent_language_plan_build_id": parent_language_plan_build_id,
        "parent_language_execution_id": execution_summary.execution_id,
        "queries_sha256": content_sha256([item.model_dump(mode="json") for item in queries]),
        "responses_sha256": content_sha256([item.model_dump(mode="json") for item in responses]),
        "complete_plan_coverage": complete,
    }
    bank_id = f"benchmark-language-bank-{content_sha256(identity)[:16]}"
    summary = BenchmarkLanguageBankSummary(
        bank_id=bank_id,
        parent_language_plan_build_id=parent_language_plan_build_id,
        parent_language_execution_id=execution_summary.execution_id,
        parent_benchmark_build_id=execution_summary.parent_benchmark_build_id,
        initial_query_count=len(queries),
        response_template_count=len(responses),
        initial_query_variant_count=sum(len(item.variants) for item in queries),
        response_variant_count=sum(len(item.masked_variants) for item in responses),
        response_key_count=len({item.template_key for item in responses}),
        register_count=len(
            {
                item.language_register
                for item in [*queries, *responses]
            }
        ),
        complete_plan_coverage=complete,
    )
    return BenchmarkLanguageBankResult(
        bank_id=bank_id,
        queries=tuple(queries),
        responses=tuple(responses),
        summary=summary,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_benchmark_language_bank_artifact(
    result: BenchmarkLanguageBankResult,
    output_root: Path,
    *,
    plan_manifest_sha256: str,
    execution_manifest_sha256: str,
) -> Path:
    run_dir = output_root / result.bank_id
    run_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "initial_query_bank.public.jsonl": "".join(
            canonical_json(item.model_dump(mode="json")) + "\n" for item in result.queries
        ),
        "response_template_bank.public.jsonl": "".join(
            canonical_json(item.model_dump(mode="json")) + "\n" for item in result.responses
        ),
        "summary.json": json.dumps(
            result.summary.model_dump(mode="json"), ensure_ascii=False, indent=2
        )
        + "\n",
    }
    for filename, text in outputs.items():
        (run_dir / filename).write_text(text, encoding="utf-8")
    manifest = {
        "schema_version": BENCHMARK_LANGUAGE_BANK_VERSION,
        "artifact_role": "frozen_offline_evolving_user_language_bank",
        "bank_id": result.bank_id,
        "status": "complete",
        "plan_manifest_sha256": plan_manifest_sha256,
        "execution_manifest_sha256": execution_manifest_sha256,
        "runtime_external_calls": False,
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
