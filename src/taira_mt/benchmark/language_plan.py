"""Provider-neutral natural-language plan for the fixed evolving-user benchmark."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from taira_mt.benchmark.diagnostics import load_symbolic_pilot
from taira_mt.benchmark.schemas import BenchmarkConstraint, ScenarioFamily
from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json, content_sha256

BENCHMARK_LANGUAGE_PLAN_VERSION = "user_benchmark_language_plan_v1"
BENCHMARK_LANGUAGE_SYSTEM_PROMPT = """You rewrite one synthetic shopping utterance template as natural English.
The dialogue act and authorized facts are fixed; change wording only.
Return exactly one JSON object with a `variants` array and no markdown.
Every [[FAAA]]-style placeholder must remain byte-identical exactly once in every variant.
Do not add product properties, preferences, constraints, reasons, numbers, brands, or named entities.
Do not mention simulators, states, scenarios, policies, hidden goals, prompts, or Thought Patterns.
Make every variant semantically equivalent, self-contained, concise, and distinct.
Follow the requested language register without caricature or unnecessary verbosity.
"""

_PLACEHOLDER_RE = re.compile(r"\[\[F[A-Z]{3}\]\]")
_NUMBER_RE = re.compile(r"(?<![\w])\$?\d+(?:\.\d+)?%?(?![\w])")
_FORBIDDEN_META = (
    "thought pattern",
    "hidden goal",
    "latent state",
    "simulator",
    "scenario family",
    "policy label",
    "authorized constraint",
    "authorized requirement",
    "precommitted",
    "semantic goal",
    "placeholder",
    "complete shopper sentence",
)
_UNVERIFIABLE_EPISTEMIC_MARKERS = (
    "verify",
    "confirm",
    "determine",
    "ascertain",
    "validate",
    "establish",
    "check whether",
    "check if",
    "cannot check",
    "can't check",
    "unable to check",
    "unable to tell",
    "cannot tell",
    "can't tell",
    "not enough information",
    "enough information",
    "sufficient information",
    "sufficient evidence",
    "enough product information",
    "information does not show",
    "information doesn't show",
)
_UNVERIFIABLE_SEMANTIC_DRIFT = (
    "do not appear to meet",
    "don't appear to meet",
    "does not appear to meet",
    "doesn't appear to meet",
    "do not seem to match",
    "don't seem to match",
    "does not seem to match",
    "doesn't seem to match",
    "do not seem to meet",
    "don't seem to meet",
    "does not seem to meet",
    "doesn't seem to meet",
    "do not appear to cover",
    "don't appear to cover",
    "does not appear to cover",
    "doesn't appear to cover",
    "do not fit",
    "don't fit",
    "does not fit",
    "doesn't fit",
    "do not satisfy",
    "don't satisfy",
    "does not satisfy",
    "doesn't satisfy",
    "do not address",
    "don't address",
    "does not address",
    "doesn't address",
    "looks like",
)
_ANSWER_ROLE_REVERSAL_MARKERS = (
    "could you",
    "would you",
    "can you",
    "might i ask",
    "please confirm",
    "please clarify",
    "please elaborate",
    "tell me",
    "what about",
    "is it true",
    "i need to know if",
    "let's verify",
    "i seek clarification",
    "ask for further details",
)
_ANSWER_META_MARKERS = (
    "authorized",
    "precommitted",
    "the fact",
    "the answer",
    "constraint",
)
_PROVIDER_SEMANTIC_TYPE_LABELS = {
    "authorized_constraint": "complete shopper sentence",
    "public_constraint_value": "stated preference phrase",
    "public_numeric_surface": "stated numeric expression",
    "public_target_type": "requested product type",
    "question_target": "attribute mentioned by the recommender",
    "referenced_item_title": "recommended item title",
    "selected_interpretation": "shopper's chosen meaning",
}
_PROVIDER_RESPONSE_GOALS = {
    "answer_relevant_question": (
        "reply directly to the recommender by preserving the protected complete shopper "
        "sentence as a standalone clause"
    ),
    "answer_registered_tradeoff_question": (
        "reply directly with the chosen compromise by preserving the protected complete "
        "shopper sentence as a standalone clause"
    ),
    "remind_already_disclosed_constraint": (
        "explicitly remind the recommender that the shopper stated this earlier, then "
        "preserve the protected complete shopper sentence as a standalone clause"
    ),
    "apply_precommitted_goal_revision": (
        "explicitly say that seeing the option clarified, updated, or changed the shopper's "
        "need, then preserve the protected complete shopper sentence as a standalone clause"
    ),
}
_RESPONSE_ACT_MARKERS = {
    "remind_already_disclosed_constraint": (
        "already",
        "earlier",
        "before",
        "previously",
        "reiterate",
        "remind",
        "recall",
        "as noted",
        "as stated",
        "mentioned",
    ),
    "apply_precommitted_goal_revision": (
        "option",
        "choice",
        "seeing",
        "having seen",
        "after seeing",
        "now that",
        "revise",
        "revision",
        "update",
        "change",
        "refine",
        "clarif",
        "reconsider",
    ),
}


class BenchmarkLanguageTaskKind(str, Enum):
    INITIAL_QUERY = "initial_query"
    RESPONSE_TEMPLATE = "response_template"


class BenchmarkLanguageRegister(str, Enum):
    CONCISE = "concise"
    CONVERSATIONAL = "conversational"
    DIRECT = "direct"
    DELIBERATIVE = "deliberative"


class LanguagePlaceholder(StrictModel):
    token: str
    semantic_type: str = Field(min_length=1)
    replacement_surface: str | None = None

    @model_validator(mode="after")
    def validate_token(self) -> LanguagePlaceholder:
        if _PLACEHOLDER_RE.fullmatch(self.token) is None:
            raise ValueError("language placeholder must match [[FAAA]]")
        if self.replacement_surface is not None and not self.replacement_surface.strip():
            raise ValueError("placeholder replacement cannot be blank")
        return self


class BenchmarkLanguageTask(StrictModel):
    schema_version: str = BENCHMARK_LANGUAGE_PLAN_VERSION
    task_id: str
    task_kind: BenchmarkLanguageTaskKind
    linkage_id: str
    language_register: BenchmarkLanguageRegister
    semantic_goal: str = Field(min_length=1)
    masked_source_text: str = Field(min_length=1)
    placeholders: list[LanguagePlaceholder]
    required_variant_count: int = Field(ge=1, le=10)
    pilot_selected: bool = False

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"task_id"})

    @model_validator(mode="after")
    def validate_task(self) -> BenchmarkLanguageTask:
        if self.schema_version != BENCHMARK_LANGUAGE_PLAN_VERSION:
            raise ValueError(f"unsupported benchmark language plan: {self.schema_version}")
        tokens = [item.token for item in self.placeholders]
        if len(tokens) != len(set(tokens)):
            raise ValueError("language task placeholders must be unique")
        observed = _PLACEHOLDER_RE.findall(self.masked_source_text)
        if sorted(observed) != sorted(tokens):
            raise ValueError("masked source and placeholder requirements differ")
        if any(self.masked_source_text.count(token) != 1 for token in tokens):
            raise ValueError("each language placeholder must occur exactly once")
        expected = f"benchmark-language-{content_sha256(self.identity_payload())[:24]}"
        if self.task_id != expected:
            raise ValueError("benchmark language task ID differs from its content")
        return self


class BenchmarkLanguagePlanSummary(StrictModel):
    build_id: str
    parent_build_id: str
    task_count: int = Field(ge=1)
    initial_query_task_count: int = Field(ge=1)
    response_template_task_count: int = Field(ge=1)
    pilot_task_count: int = Field(ge=1)
    requested_variant_count: int = Field(ge=1)
    query_domain_scenario_counts: dict[str, int]
    register_task_counts: dict[str, int]
    response_template_key_count: int = Field(ge=1)
    estimated_input_tokens_char_div_4: int = Field(ge=1)
    estimated_output_tokens_char_div_4: int = Field(ge=1)
    external_calls: bool = False


@dataclass(frozen=True)
class BenchmarkLanguagePlanResult:
    build_id: str
    parent_build_id: str
    tasks: tuple[BenchmarkLanguageTask, ...]
    summary: BenchmarkLanguagePlanSummary


@dataclass(frozen=True)
class _ResponseTemplateSpec:
    key: str
    semantic_goal: str
    source: str
    placeholder_types: tuple[str, ...] = ()


_RESPONSE_SPECS = (
    _ResponseTemplateSpec(
        "invalid_action",
        "politely request a valid recommendation-system action",
        "I cannot respond to that as a valid recommendation step. Please try again.",
    ),
    _ResponseTemplateSpec(
        "resolve_ambiguity",
        "select and state one intended interpretation without repeating it",
        "I mean [[FAAA]].",
        ("selected_interpretation",),
    ),
    _ResponseTemplateSpec(
        "dismiss_irrelevant_question",
        "say that the asked attribute is not a preference",
        "I do not have a particular preference about [[FAAA]].",
        ("question_target",),
    ),
    _ResponseTemplateSpec(
        "answer_registered_tradeoff_question",
        "state the precommitted feasible trade-off",
        "[[FAAA]].",
        ("authorized_constraint",),
    ),
    _ResponseTemplateSpec(
        "remind_already_disclosed_constraint",
        "remind the system of an already stated requirement without changing it",
        "I already mentioned that [[FAAA]].",
        ("authorized_constraint",),
    ),
    _ResponseTemplateSpec(
        "answer_relevant_question",
        "answer one relevant clarification question",
        "[[FAAA]].",
        ("authorized_constraint",),
    ),
    _ResponseTemplateSpec(
        "apply_precommitted_goal_revision",
        "revise the need after seeing an option and state the newly active requirement",
        "Seeing that option helps me clarify my need. [[FAAA]].",
        ("authorized_constraint",),
    ),
    _ResponseTemplateSpec(
        "request_explanation_before_acceptance",
        "request a grounded explanation before accepting a plausible option",
        "That option seems plausible. Could you explain why it fits my needs?",
    ),
    _ResponseTemplateSpec(
        "accept_satisfactory_recommendation",
        "accept one satisfactory recommended item",
        "[[FAAA]] works for me.",
        ("referenced_item_title",),
    ),
    _ResponseTemplateSpec(
        "reject_without_detail",
        "reject the recommended options without inventing a new reason",
        "Those options do not fit what I need.",
    ),
    _ResponseTemplateSpec(
        "reject_with_authorized_reason",
        "reject the options and state exactly one authorized unmet requirement",
        "Those options do not fit what I need. [[FAAA]].",
        ("authorized_constraint",),
    ),
    _ResponseTemplateSpec(
        "reject_unverifiable_without_detail",
        "decline options whose stated hard requirements cannot be verified from catalog evidence",
        (
            "I cannot verify that those options meet my stated requirements from "
            "the available product information."
        ),
    ),
    _ResponseTemplateSpec(
        "reject_unverifiable_with_authorized_reason",
        (
            "decline unverifiable options and restate exactly one authorized requirement "
            "whose evidence is missing"
        ),
        (
            "I cannot verify that those options meet my stated requirements from "
            "the available product information. [[FAAA]]."
        ),
        ("authorized_constraint",),
    ),
    _ResponseTemplateSpec(
        "accept_after_grounded_explanation",
        "accept the pending item after a helpful grounded explanation",
        "That explanation helps. [[FAAA]] works for me.",
        ("referenced_item_title",),
    ),
    _ResponseTemplateSpec(
        "acknowledge_grounded_explanation",
        "acknowledge the explanation but continue requesting a suitable item",
        "That explanation is helpful. I would still like a suitable recommendation.",
    ),
    _ResponseTemplateSpec(
        "refuse_unregistered_relaxation",
        "refuse a relaxation that is not part of the fixed trade-off plan",
        "I would rather not relax that requirement in this way.",
    ),
    _ResponseTemplateSpec(
        "refuse_mismatched_relaxation",
        "refuse a proposed relaxation that does not match the acceptable trade-off",
        "That is not the trade-off I am willing to make.",
    ),
    _ResponseTemplateSpec(
        "accept_registered_relaxation",
        "agree to the registered relaxation and state the updated requirement",
        "That trade-off works for me. [[FAAA]].",
        ("authorized_constraint",),
    ),
    _ResponseTemplateSpec(
        "graceful_no_solution",
        "acknowledge that no item can satisfy the current requirements",
        "I understand that there is no suitable option under those requirements.",
    ),
    _ResponseTemplateSpec(
        "abandon_after_premature_stop",
        "say that a recommendation was still needed and end the conversation",
        "I still needed a recommendation, so I will stop here.",
    ),
    _ResponseTemplateSpec(
        "patience_exhausted",
        "end the conversation because it has taken too long",
        "This is taking too long, so I will stop here.",
    ),
)


def _placeholder(index: int) -> str:
    if index < 0 or index >= 26**3:
        raise ValueError("too many placeholders")
    letters = []
    remainder = index
    for power in (26**2, 26, 1):
        digit, remainder = divmod(remainder, power)
        letters.append(chr(ord("A") + digit))
    return f"[[F{''.join(letters)}]]"


def _query_surfaces(
    initial_query: str,
    constraints: list[BenchmarkConstraint],
    target_type: str,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for constraint in constraints:
        value = constraint.predicate.value
        if isinstance(value, str) and value in initial_query:
            candidates.append(("public_constraint_value", value))
        for prefix in ("I am looking within ", "More specifically, I need "):
            if constraint.surface_form.startswith(prefix):
                surface = constraint.surface_form.removeprefix(prefix).rstrip(". ")
                if surface in initial_query:
                    candidates.append(("public_constraint_value", surface))
    if target_type in initial_query:
        candidates.append(("public_target_type", target_type))
    for match in _NUMBER_RE.finditer(initial_query):
        candidates.append(("public_numeric_surface", match.group(0)))
    deduplicated: dict[str, str] = {}
    for semantic_type, surface in sorted(
        candidates,
        key=lambda item: (-len(item[1]), item[1].casefold(), item[0]),
    ):
        deduplicated.setdefault(surface, semantic_type)
    return [(kind, surface) for surface, kind in deduplicated.items()]


def _mask_query(
    initial_query: str,
    constraints: list[BenchmarkConstraint],
    target_type: str,
) -> tuple[str, list[LanguagePlaceholder]]:
    masked = initial_query
    placeholders: list[LanguagePlaceholder] = []
    for semantic_type, surface in _query_surfaces(initial_query, constraints, target_type):
        # A public fact may legitimately occur more than once (for example,
        # "a clothing item in clothing").  Give every occurrence its own
        # protected token: the provider contract intentionally requires each
        # token exactly once, which is easier to validate than permitting a
        # variable placeholder multiplicity.
        while surface in masked:
            token = _placeholder(len(placeholders))
            masked = masked.replace(surface, token, 1)
            placeholders.append(
                LanguagePlaceholder(
                    token=token,
                    semantic_type=semantic_type,
                    replacement_surface=surface,
                )
            )
    return masked, placeholders


def _task(**payload: Any) -> BenchmarkLanguageTask:
    identity = {"schema_version": BENCHMARK_LANGUAGE_PLAN_VERSION, **payload}
    return BenchmarkLanguageTask(
        task_id=f"benchmark-language-{content_sha256(identity)[:24]}",
        **identity,
    )


def provider_payload(task: BenchmarkLanguageTask) -> dict[str, Any]:
    """Remove dataset linkage and replacement values before a model sees the task."""

    complete_sentence_tokens = [
        item.token
        for item in task.placeholders
        if item.semantic_type == "authorized_constraint"
    ]
    payload: dict[str, Any] = {
        "task_kind": task.task_kind.value,
        "language_register": task.language_register.value,
        "semantic_goal": _PROVIDER_RESPONSE_GOALS.get(task.linkage_id, task.semantic_goal),
        "masked_source_text": task.masked_source_text,
        "placeholder_requirements": [
            {
                "token": item.token,
                "semantic_type": _PROVIDER_SEMANTIC_TYPE_LABELS.get(
                    item.semantic_type, item.semantic_type
                ),
            }
            for item in task.placeholders
        ],
        "utterance_rule": (
            "Write only what a shopper would naturally say; never echo instruction labels. "
            "A protected complete shopper sentence must start the utterance or follow "
            "punctuation as its own clause; never place it directly after a word or make "
            "later words depend on it."
        ),
        "required_variant_count": task.required_variant_count,
        "required_output": {"variants": ["natural placeholder-preserving utterance"]},
    }
    if complete_sentence_tokens:
        token_examples = ", ".join(
            f"`For clarity: {token}.`" for token in complete_sentence_tokens
        )
        payload["complete_sentence_boundary_contract"] = (
            "For each complete-sentence token, the character before it must be sentence "
            "punctuation (:, dash, period, !, or ?) or there must be no preceding text; "
            "nothing except punctuation may follow the token. Valid shape(s): "
            f"{token_examples}. Invalid shapes include `I prefer TOKEN.` and "
            "`TOKEN works for me.`"
        )
    return payload


def language_task_reuse_signature(task: BenchmarkLanguageTask) -> str:
    """Hash reusable semantics, excluding case IDs and protected values.

    A requested category can be discovered through either the explicit target
    type or an equivalent public category constraint.  Both placeholders obey
    the same preservation contract and are therefore one reuse class.
    """

    def reuse_semantic_type(semantic_type: str) -> str:
        if (
            task.task_kind == BenchmarkLanguageTaskKind.INITIAL_QUERY
            and semantic_type
            in {"public_constraint_value", "public_target_type"}
        ):
            return "public_category_value"
        return semantic_type

    payload = {
        "reuse_contract_version": "provider_visible_semantic_signature_v2_category_alias",
        "task_kind": task.task_kind.value,
        "language_register": task.language_register.value,
        "semantic_goal": task.semantic_goal,
        "masked_source_text": task.masked_source_text,
        "placeholder_requirements": [
            {
                "token": item.token,
                "semantic_type": reuse_semantic_type(item.semantic_type),
            }
            for item in task.placeholders
        ],
        "required_variant_count": task.required_variant_count,
    }
    return content_sha256(payload)


def _scenario_query_goal(scenario: ScenarioFamily) -> str:
    return {
        ScenarioFamily.DIRECT: "state a sufficiently specific current shopping need",
        ScenarioFamily.PROGRESSIVE: (
            "state a broad shopping need and naturally ask for help narrowing the exact type"
        ),
        ScenarioFamily.AMBIGUOUS: (
            "state a broad need while making clear that the intended subtype is unresolved"
        ),
        ScenarioFamily.REVISION: (
            "state only the user's currently known shopping need without foreshadowing a later revision"
        ),
        ScenarioFamily.CONFLICT: (
            "state the exact current type and numeric requirement without claiming it is feasible"
        ),
    }[scenario]


def build_benchmark_language_plan(
    benchmark_dir: Path,
    *,
    schedule_seed: int = 20260725,
    pilot_queries_per_domain_scenario: int = 2,
    query_variant_count: int = 2,
    response_variant_count: int = 5,
) -> BenchmarkLanguagePlanResult:
    manifest, cases = load_symbolic_pilot(benchmark_dir)
    if pilot_queries_per_domain_scenario < 1:
        raise ValueError("language pilot requires at least one query per domain/scenario")
    grouped: dict[tuple[str, ScenarioFamily], list] = {}
    for case in cases:
        key = (case.pair.public.domain, case.pair.private.initial_state.scenario_family)
        grouped.setdefault(key, []).append(case)
    pilot_case_ids: set[str] = set()
    for key, values in grouped.items():
        ordered = sorted(
            values,
            key=lambda item: content_sha256(
                {"schedule_seed": schedule_seed, "case_id": item.pair.public.case_id}
            ),
        )
        if len(ordered) < pilot_queries_per_domain_scenario:
            raise ValueError(f"insufficient query pilot capacity for {key}")
        pilot_case_ids.update(
            item.pair.public.case_id
            for item in ordered[:pilot_queries_per_domain_scenario]
        )

    tasks: list[BenchmarkLanguageTask] = []
    for loaded in sorted(cases, key=lambda item: item.pair.public.case_id):
        public = loaded.pair.public
        private = loaded.pair.private
        state = private.initial_state
        initial_goal = state.latent_goal.version(0)
        public_ids = set(state.disclosure_plan.initially_public)
        public_constraints = [
            item for item in initial_goal.constraints if item.constraint_id in public_ids
        ]
        masked, placeholders = _mask_query(
            public.initial_query,
            public_constraints,
            state.latent_goal.target_type,
        )
        register = BenchmarkLanguageRegister(state.behavior.language_style_id)
        tasks.append(
            _task(
                task_kind=BenchmarkLanguageTaskKind.INITIAL_QUERY,
                linkage_id=public.case_id,
                language_register=register,
                semantic_goal=_scenario_query_goal(state.scenario_family),
                masked_source_text=masked,
                placeholders=[item.model_dump(mode="json") for item in placeholders],
                required_variant_count=query_variant_count,
                pilot_selected=public.case_id in pilot_case_ids,
            )
        )

    for spec in _RESPONSE_SPECS:
        placeholders = [
            LanguagePlaceholder(token=_placeholder(index), semantic_type=semantic_type)
            for index, semantic_type in enumerate(spec.placeholder_types)
        ]
        if _PLACEHOLDER_RE.findall(spec.source) != [item.token for item in placeholders]:
            raise AssertionError(f"response template placeholder mismatch: {spec.key}")
        for register in BenchmarkLanguageRegister:
            tasks.append(
                _task(
                    task_kind=BenchmarkLanguageTaskKind.RESPONSE_TEMPLATE,
                    linkage_id=spec.key,
                    language_register=register,
                    semantic_goal=spec.semantic_goal,
                    masked_source_text=spec.source,
                    placeholders=[item.model_dump(mode="json") for item in placeholders],
                    required_variant_count=response_variant_count,
                    pilot_selected=True,
                )
            )
    tasks.sort(key=lambda item: item.task_id)
    task_payload = [item.model_dump(mode="json") for item in tasks]
    identity = {
        "schema_version": BENCHMARK_LANGUAGE_PLAN_VERSION,
        "parent_build_id": manifest.build_id,
        "schedule_seed": schedule_seed,
        "pilot_queries_per_domain_scenario": pilot_queries_per_domain_scenario,
        "query_variant_count": query_variant_count,
        "response_variant_count": response_variant_count,
        "tasks_sha256": content_sha256(task_payload),
    }
    build_id = f"benchmark-language-plan-{content_sha256(identity)[:16]}"
    request_characters = sum(
        len(BENCHMARK_LANGUAGE_SYSTEM_PROMPT) + len(canonical_json(provider_payload(item)))
        for item in tasks
    )
    output_characters = sum(
        item.required_variant_count * (len(item.masked_source_text) + 32) + 24
        for item in tasks
    )
    query_counts = Counter(
        f"{item.pair.public.domain}:{item.pair.private.initial_state.scenario_family.value}"
        for item in cases
    )
    summary = BenchmarkLanguagePlanSummary(
        build_id=build_id,
        parent_build_id=manifest.build_id,
        task_count=len(tasks),
        initial_query_task_count=sum(
            item.task_kind == BenchmarkLanguageTaskKind.INITIAL_QUERY for item in tasks
        ),
        response_template_task_count=sum(
            item.task_kind == BenchmarkLanguageTaskKind.RESPONSE_TEMPLATE for item in tasks
        ),
        pilot_task_count=sum(item.pilot_selected for item in tasks),
        requested_variant_count=sum(item.required_variant_count for item in tasks),
        query_domain_scenario_counts=dict(sorted(query_counts.items())),
        register_task_counts=dict(
            sorted(Counter(item.language_register.value for item in tasks).items())
        ),
        response_template_key_count=len(_RESPONSE_SPECS),
        estimated_input_tokens_char_div_4=(request_characters + 3) // 4,
        estimated_output_tokens_char_div_4=(output_characters + 3) // 4,
    )
    return BenchmarkLanguagePlanResult(
        build_id=build_id,
        parent_build_id=manifest.build_id,
        tasks=tuple(tasks),
        summary=summary,
    )


class BenchmarkLanguageValidationError(ValueError):
    pass


def validate_language_variants(
    task: BenchmarkLanguageTask,
    raw_response: object,
) -> list[str]:
    if not isinstance(raw_response, dict) or set(raw_response) != {"variants"}:
        raise BenchmarkLanguageValidationError("language response must contain only variants")
    variants = raw_response["variants"]
    if not isinstance(variants, list) or len(variants) != task.required_variant_count:
        raise BenchmarkLanguageValidationError("language response variant count differs from task")
    if any(not isinstance(item, str) or not item.strip() for item in variants):
        raise BenchmarkLanguageValidationError("language variants must be non-empty strings")
    normalized = [item.strip() for item in variants]
    if len(set(normalized)) != len(normalized):
        raise BenchmarkLanguageValidationError("language variants must be distinct")
    expected = sorted(item.token for item in task.placeholders)
    for variant in normalized:
        if len(variant) > 320:
            raise BenchmarkLanguageValidationError("language variant is too long")
        observed = _PLACEHOLDER_RE.findall(variant)
        if sorted(observed) != expected or any(variant.count(token) != 1 for token in expected):
            raise BenchmarkLanguageValidationError("language variant changed protected placeholders")
        if _NUMBER_RE.search(variant):
            raise BenchmarkLanguageValidationError("language variant introduced an unprotected number")
        lowered = variant.casefold()
        if any(term in lowered for term in _FORBIDDEN_META):
            raise BenchmarkLanguageValidationError("language variant exposed forbidden metadata")
        if task.linkage_id in {
            "answer_relevant_question",
            "answer_registered_tradeoff_question",
        }:
            if "?" in variant or any(
                marker in lowered for marker in _ANSWER_ROLE_REVERSAL_MARKERS
            ):
                raise BenchmarkLanguageValidationError(
                    "answer response reversed roles instead of stating the authorized answer"
                )
            frame = lowered
            for token in expected:
                frame = frame.replace(token.casefold(), "")
            if any(marker in frame for marker in _ANSWER_META_MARKERS):
                raise BenchmarkLanguageValidationError(
                    "answer response exposed an internal label instead of natural shopper speech"
                )
        for placeholder in task.placeholders:
            if placeholder.semantic_type != "authorized_constraint":
                continue
            token_start = variant.index(placeholder.token)
            before = variant[:token_start].rstrip()
            after = variant[token_start + len(placeholder.token) :].lstrip()
            if before and before[-1] not in ".!?:;\N{EM DASH}\N{EN DASH}-":
                raise BenchmarkLanguageValidationError(
                    "complete-sentence placeholder was embedded after a word"
                )
            if after and after[0].isalnum():
                raise BenchmarkLanguageValidationError(
                    "complete-sentence placeholder was embedded before dependent text"
                )
        act_markers = _RESPONSE_ACT_MARKERS.get(task.linkage_id)
        if act_markers is not None and not any(marker in lowered for marker in act_markers):
            raise BenchmarkLanguageValidationError(
                "response lost the observable multi-turn dialogue act"
            )
        if task.linkage_id.startswith("reject_unverifiable_"):
            if any(term in lowered for term in _UNVERIFIABLE_SEMANTIC_DRIFT):
                raise BenchmarkLanguageValidationError(
                    "unverifiable response changed missing evidence into item mismatch"
                )
            if not any(marker in lowered for marker in _UNVERIFIABLE_EPISTEMIC_MARKERS):
                raise BenchmarkLanguageValidationError(
                    "unverifiable response lost its missing-evidence uncertainty"
                )
    return normalized


def unmask_query_variant(task: BenchmarkLanguageTask, masked_variant: str) -> str:
    if task.task_kind != BenchmarkLanguageTaskKind.INITIAL_QUERY:
        raise ValueError("only initial-query tasks have fixed replacement surfaces")
    result = masked_variant
    for item in task.placeholders:
        if item.replacement_surface is None:
            raise ValueError("query placeholder lacks its replacement surface")
        result = result.replace(item.token, item.replacement_surface)
    if _PLACEHOLDER_RE.search(result):
        raise ValueError("unresolved query placeholder")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_idempotently(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different language plan file: {path}")
        return
    path.write_text(text, encoding="utf-8")


def write_benchmark_language_plan_artifact(
    result: BenchmarkLanguagePlanResult,
    output_root: Path,
) -> Path:
    run_dir = output_root / result.build_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = [item.model_dump(mode="json") for item in result.tasks]
    pilot = [item for item in tasks if item["pilot_selected"]]
    outputs = {
        "tasks.public.jsonl": "".join(canonical_json(item) + "\n" for item in tasks),
        "pilot_tasks.public.jsonl": "".join(canonical_json(item) + "\n" for item in pilot),
        "summary.json": json.dumps(
            result.summary.model_dump(mode="json"), ensure_ascii=False, indent=2
        )
        + "\n",
    }
    for filename, text in outputs.items():
        _write_idempotently(run_dir / filename, text)
    manifest = {
        "schema_version": BENCHMARK_LANGUAGE_PLAN_VERSION,
        "artifact_role": "provider_neutral_user_benchmark_language_plan",
        "build_id": result.build_id,
        "parent_build_id": result.parent_build_id,
        "tasks_sha256": content_sha256(tasks),
        "system_prompt_sha256": content_sha256(BENCHMARK_LANGUAGE_SYSTEM_PROMPT),
        "external_calls": False,
        "files": [
            {
                "relative_path": filename,
                "sha256": _file_sha256(run_dir / filename),
                "byte_size": (run_dir / filename).stat().st_size,
                "row_count": text.count("\n") if filename.endswith(".jsonl") else None,
            }
            for filename, text in sorted(outputs.items())
        ],
    }
    _write_idempotently(
        run_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return run_dir
