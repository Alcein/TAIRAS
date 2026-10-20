"""Seeded, offline rendering over a verified frozen benchmark language bank.

The renderer is deliberately a surface-only component: it receives an already
authorized semantic response context and cannot inspect latent goals or alter
the user-state transition.  Variant selection is content-addressed rather than
stateful, so reset, checkpoint restore, and branch replay produce identical
utterances.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from taira_mt.artifacts import read_jsonl_unique, verify_artifact_bundle
from taira_mt.benchmark.language_bank import (
    BENCHMARK_LANGUAGE_BANK_VERSION,
    BenchmarkLanguageBankSummary,
    FrozenInitialQueryVariants,
    FrozenResponseTemplateVariants,
)
from taira_mt.benchmark.language_plan import BenchmarkLanguageRegister
from taira_mt.benchmark.schemas import BenchmarkUserAct
from taira_mt.provenance import content_sha256

BENCHMARK_LANGUAGE_RENDERER_VERSION = "frozen_benchmark_language_renderer_v1"

_PLACEHOLDER_RE = re.compile(r"\[\[F[A-Z]{3}\]\]")
_INVALID_GOALS = {
    "recommendation_contains_out_of_pool_item",
    "explanation_contains_out_of_pool_item",
    "explanation_uses_ungrounded_catalog_evidence",
}


class BenchmarkLanguageRenderer(Protocol):
    """Narrow interface consumed by the deterministic user environment."""

    @property
    def renderer_id(self) -> str: ...

    def render_initial_query(self, *, case_id: str, language_style_id: str) -> str: ...

    def render_response(self, context: RuntimeResponseLanguageContext) -> str: ...


@dataclass(frozen=True)
class RuntimeResponseLanguageContext:
    """Only public/authorized facts made available to the surface renderer."""

    case_id: str
    turn_index: int
    language_style_id: str
    user_act: BenchmarkUserAct
    communicative_goal: str
    canonical_text: str
    authorized_constraint_surfaces: tuple[str, ...] = ()
    referenced_item_titles: tuple[str, ...] = ()
    selected_interpretation: str | None = None
    question_target: str | None = None


@dataclass(frozen=True)
class LoadedBenchmarkLanguageBank:
    manifest_sha256: str
    summary: BenchmarkLanguageBankSummary
    queries: tuple[FrozenInitialQueryVariants, ...]
    responses: tuple[FrozenResponseTemplateVariants, ...]


def load_benchmark_language_bank(bank_dir: Path) -> LoadedBenchmarkLanguageBank:
    """Load a complete bank only after manifest and per-file hash verification."""

    bundle = verify_artifact_bundle(bank_dir)
    if bundle.manifest.get("schema_version") != BENCHMARK_LANGUAGE_BANK_VERSION:
        raise ValueError("unsupported benchmark language bank artifact")
    if bundle.manifest.get("runtime_external_calls") is not False:
        raise ValueError("runtime language bank must explicitly disable external calls")
    summary = BenchmarkLanguageBankSummary.model_validate(
        json.loads(bundle.require_output("summary.json").read_text(encoding="utf-8"))
    )
    if summary.bank_id != bundle.manifest.get("bank_id"):
        raise ValueError("language bank summary and manifest IDs differ")
    if not summary.complete_plan_coverage or summary.runtime_external_calls:
        raise ValueError("runtime language bank must be complete and offline")
    queries = tuple(
        read_jsonl_unique(
            bundle.require_output("initial_query_bank.public.jsonl"),
            FrozenInitialQueryVariants.model_validate,
            lambda item: item.case_id,
            record_name="frozen initial query",
        )
    )
    responses = tuple(
        read_jsonl_unique(
            bundle.require_output("response_template_bank.public.jsonl"),
            FrozenResponseTemplateVariants.model_validate,
            lambda item: f"{item.template_key}:{item.language_register.value}",
            record_name="frozen response template",
        )
    )
    if len(queries) != summary.initial_query_count:
        raise ValueError("language bank initial-query count differs from summary")
    if len(responses) != summary.response_template_count:
        raise ValueError("language bank response-template count differs from summary")
    if sum(len(item.variants) for item in queries) != summary.initial_query_variant_count:
        raise ValueError("language bank initial-query variant count differs from summary")
    if sum(len(item.masked_variants) for item in responses) != summary.response_variant_count:
        raise ValueError("language bank response variant count differs from summary")
    return LoadedBenchmarkLanguageBank(
        manifest_sha256=bundle.manifest_sha256,
        summary=summary,
        queries=queries,
        responses=responses,
    )


def response_template_key(context: RuntimeResponseLanguageContext) -> str:
    """Map a semantic response frame to exactly one frozen response family."""

    goal = context.communicative_goal
    if context.user_act == BenchmarkUserAct.CORRECT_INVALID_ACTION or goal in _INVALID_GOALS:
        return "invalid_action"
    if goal == "reject_unverifiable_recommendation":
        return (
            "reject_unverifiable_with_authorized_reason"
            if context.authorized_constraint_surfaces
            else "reject_unverifiable_without_detail"
        )
    if goal == "reject_unsatisfactory_recommendation":
        return (
            "reject_with_authorized_reason"
            if context.authorized_constraint_surfaces
            else "reject_without_detail"
        )
    if goal == "respond_to_system_stop":
        return (
            "graceful_no_solution"
            if context.user_act == BenchmarkUserAct.GRACEFUL_END
            else "abandon_after_premature_stop"
        )
    return goal


def _sentence_block(values: tuple[str, ...]) -> str:
    if not values:
        raise ValueError("authorized-constraint placeholder has no authorized surface")
    cleaned = [value.strip().rstrip(". ") for value in values]
    if any(not value for value in cleaned):
        raise ValueError("authorized-constraint surface cannot be blank")
    return ". ".join(cleaned)


def _placeholder_value(
    semantic_type: str,
    context: RuntimeResponseLanguageContext,
) -> str:
    if semantic_type == "authorized_constraint":
        return _sentence_block(context.authorized_constraint_surfaces)
    if semantic_type == "referenced_item_title":
        if len(context.referenced_item_titles) != 1:
            raise ValueError("item-title response requires exactly one referenced item")
        return context.referenced_item_titles[0]
    if semantic_type == "selected_interpretation":
        if not context.selected_interpretation:
            raise ValueError("ambiguity response lacks the selected interpretation")
        return context.selected_interpretation
    if semantic_type == "question_target":
        if not context.question_target:
            raise ValueError("irrelevant-question response lacks the question target")
        return context.question_target
    raise ValueError(f"unsupported runtime language placeholder type: {semantic_type}")


class FrozenBenchmarkLanguageRenderer:
    """Content-addressed deterministic renderer with zero runtime model calls."""

    def __init__(self, bank: LoadedBenchmarkLanguageBank, *, seed: int = 20260726) -> None:
        if seed < 0:
            raise ValueError("language renderer seed must be non-negative")
        self.bank = bank
        self.seed = seed
        self._queries = {item.case_id: item for item in bank.queries}
        self._responses = {
            (item.template_key, item.language_register.value): item
            for item in bank.responses
        }
        if len(self._queries) != len(bank.queries) or len(self._responses) != len(bank.responses):
            raise ValueError("language bank contains duplicate runtime lookup keys")
        self._renderer_id = (
            "language-renderer-"
            + content_sha256(
                {
                    "version": BENCHMARK_LANGUAGE_RENDERER_VERSION,
                    "bank_id": bank.summary.bank_id,
                    "bank_manifest_sha256": bank.manifest_sha256,
                    "seed": seed,
                }
            )[:16]
        )

    @classmethod
    def from_artifact(
        cls,
        bank_dir: Path,
        *,
        seed: int = 20260726,
    ) -> FrozenBenchmarkLanguageRenderer:
        return cls(load_benchmark_language_bank(bank_dir), seed=seed)

    @property
    def renderer_id(self) -> str:
        return self._renderer_id

    def _index(self, *, purpose: str, identity: dict[str, object], count: int) -> int:
        if count < 1:
            raise ValueError("language bank entry has no variants")
        digest = content_sha256(
            {
                "renderer_version": BENCHMARK_LANGUAGE_RENDERER_VERSION,
                "renderer_seed": self.seed,
                "bank_id": self.bank.summary.bank_id,
                "purpose": purpose,
                **identity,
            }
        )
        return int(digest[:16], 16) % count

    def render_initial_query(self, *, case_id: str, language_style_id: str) -> str:
        entry = self._queries.get(case_id)
        if entry is None:
            raise KeyError(f"language bank lacks initial query for case: {case_id}")
        if entry.language_register.value != language_style_id:
            raise ValueError("initial-query language register differs from benchmark behavior")
        index = self._index(
            purpose="initial_query",
            identity={"case_id": case_id, "language_style_id": language_style_id},
            count=len(entry.variants),
        )
        value = entry.variants[index].strip()
        if not value or _PLACEHOLDER_RE.search(value):
            raise ValueError("frozen initial query is blank or contains an unresolved placeholder")
        return value

    def render_response(self, context: RuntimeResponseLanguageContext) -> str:
        register = BenchmarkLanguageRegister(context.language_style_id)
        template_key = response_template_key(context)
        entry = self._responses.get((template_key, register.value))
        if entry is None:
            raise KeyError(
                "language bank lacks response template: "
                f"{template_key}/{register.value}"
            )
        index = self._index(
            purpose="response",
            identity={
                "case_id": context.case_id,
                "turn_index": context.turn_index,
                "template_key": template_key,
                "language_style_id": register.value,
            },
            count=len(entry.masked_variants),
        )
        result = entry.masked_variants[index]
        expected_tokens = {item.token for item in entry.placeholders}
        observed_tokens = set(_PLACEHOLDER_RE.findall(result))
        if observed_tokens != expected_tokens:
            raise ValueError("frozen response variant and placeholder contract differ")
        for placeholder in entry.placeholders:
            value = _placeholder_value(placeholder.semantic_type, context)
            if not value.strip() or _PLACEHOLDER_RE.search(value):
                raise ValueError("runtime placeholder value is blank or contains a placeholder")
            if result.count(placeholder.token) != 1:
                raise ValueError("runtime response placeholder must occur exactly once")
            result = result.replace(placeholder.token, value)
        result = result.strip()
        if not result or _PLACEHOLDER_RE.search(result):
            raise ValueError("rendered response is blank or contains an unresolved placeholder")
        return result
