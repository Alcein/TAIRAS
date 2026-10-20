"""LLM execution of a CIV-selected Thought-Pattern Skill.

SCTP retrieves an SOP-plus-insight Skill and CIV chooses its legal operation.
This module is the final public executor: it follows the selected SOP to map
visible need language (including a newly revealed preference) into catalog
vocabulary, invokes the shared full-catalog retriever, and selects a grounded
Top-k slate.  It never sees the benchmark mechanism, target item, or evaluator.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from pydantic import Field, field_validator

from taira_mt.benchmark.schemas import BenchmarkActionType, BenchmarkSystemAction
from taira_mt.models import StrictModel
from taira_mt.mt_agent.direct_llm import DirectCandidateSelection
from taira_mt.mt_agent.memory import PublicDialogueMemory
from taira_mt.mt_agent.retrieval import (
    CatalogRetrievalItem,
    CatalogRetrievalResult,
    LexicalCatalogRetriever,
    SearchSpec,
)
from taira_mt.mt_agent.runner import LLMCompletionAudit
from taira_mt.mt_agent.sctp import SkillActionProposal
from taira_mt.mt_agent.skills import ThoughtPatternLibrary
from taira_mt.mt_agent.state import RecommendationReadiness, visible_choice_axis
from taira_mt.provenance import content_sha256
from taira_mt.providers.openai_compatible import JsonCompletion, JsonCompletionTransport, JsonResponseError

LEGACY_THOUGHT_PATTERN_EXECUTOR_VERSION = "thought_pattern_sop_executor_v6_catalog_native_grounding"
THOUGHT_PATTERN_EXECUTOR_VERSION = (
    "thought_pattern_sop_executor_v18_public_head_guard_deep_tail_calibration"
)

_GROUND_PROMPT = """You execute one Thought-Pattern Skill for a shopping recommendation Agent.
The Skill has already been retrieved and its action has already been chosen. Follow its SOP and transferable
insights using only the public dialogue and public DSUS state. Do not change the action, infer a hidden user,
or use benchmark/evaluator information.

For a recommendation, convert scenario and preference language into concise vocabulary likely to occur in
product titles, categories, or descriptions. Pay special attention to the latest user reply. Map a natural
preference to useful catalog synonyms (for example, a strong fragrance may map to concentrated/parfum terms),
but do not invent a factual requirement. For the latest decisive preference, include one to three catalog-native
synonyms or evidence cues in preferred_terms when they are supported by ordinary product vocabulary (for example,
arch support may map to orthotic/supportive insole, and fragrance-free may map to unscented). Keep must_terms to
truly necessary product identifiers; put alternative expressions and inferred functional attributes in
preferred_terms. Avoid broad social words such as confidence or professional as hard requirements when they are
unlikely to appear in product metadata.

Return only one JSON object with exactly these keys:
query_focus, target_type, must_terms, preferred_terms, avoid_terms.
All term fields must be arrays of short strings.
"""

_DIALOGUE_GROUND_PROMPT = """You are the independent semantic retrieval view of a shopping recommendation Agent.
Translate only the complete public dialogue into concise vocabulary likely to occur in product titles, categories,
or descriptions. Preserve the literal product identity and give the latest user answer priority. Do not see or
imitate a Thought Pattern: this view exists to keep a useful catalog interpretation when a specialized SOP becomes
too narrow. Keep must_terms to essential product identifiers; put synonyms, scenario implications, and alternative
expressions in preferred_terms. Do not invent a factual requirement or use benchmark/evaluator information.

Return only one JSON object with exactly these keys:
query_focus, target_type, must_terms, preferred_terms, avoid_terms.
All term fields must be arrays of short strings.
"""

_PATTERN_CANDIDATE_SELECTION_PROMPT = """You are executing the final catalog-selection step of a shopping
Thought Pattern. Use only the supplied public dialogue, target type, retrieval plans, and candidate metadata.

First enforce product identity: a candidate whose title/category is a different product type must rank below
candidates of the requested type, even when it contains a preferred attribute word. Then rank the type-correct
candidates by hard public constraints, the latest decisive user answer, functional/scenario fit, and supporting
catalog evidence, in that order. An attribute cue never compensates for the wrong product type. Select and rank
up to ten supplied candidate IDs; never invent an ID.

Return only one JSON object with exactly these keys: item_ids, utterance.
The utterance should name the leading item and briefly ground it in visible catalog evidence.
"""

_PUBLIC_SEMANTIC_VERIFIER_PROMPT = """You are the independent public-evidence verifier for a shopping
recommendation slate. A specialized Thought Pattern has proposed a provisional ranking. Audit and rerank it using
only the complete public dialogue and supplied catalog metadata; do not assume the provisional ranking is correct.

Apply these checks in order:
1. Product identity: demote candidates whose title/category is a different product type.
2. Latest public answer: when the user resolves a conflict or revises a preference, it overrides the earlier
   unresolved wording.
3. Hard constraints, then functional/scenario fit, then supporting evidence strength.
4. Prefer candidates with explicit catalog evidence over candidates that merely share a broad attribute word.

Return up to ten supplied IDs in best-first order and never invent an ID. Do not use or mention a hidden target,
benchmark mechanism, evaluator, expected answer, or method identity.

Return only one JSON object with exactly these keys: item_ids, utterance.
The utterance should briefly ground the leading item in visible evidence.
"""

_FOCUSED_HEAD3_CONSENSUS_PROMPT = """You are the focused public-evidence head calibrator for a shopping
recommendation slate. The action and the three head candidates are already fixed. Return an exact permutation of
those same three IDs and independently assess every candidate's visible support. Never add, remove, or duplicate
an ID.

Treat the current order as a useful prior and change the first item only when another existing Top-3 candidate has
clearly stronger public evidence. Compare, in order: literal product identity, the latest visible user answer,
coverage of the public DSUS constraints, absence of public avoid violations, functional/scenario fit, and catalog
evidence strength. Retrieval rank is supporting evidence, not authority. Use only the supplied public dialogue,
public DSUS retrieval frame, and public catalog fields. Do not infer a hidden user or use any benchmark, evaluator,
expected answer, method identity, or private state.

Assign public_evidence_grade conservatively: 2 only when the visible catalog fields explicitly satisfy the literal
product identity and decisive latest public requirement; 1 for plausible or partial public support; 0 for a wrong
identity, a visible violation, or no material support. The grades assess public evidence, not hidden relevance.

Return only one JSON object with exactly these keys: item_ids, assessments. item_ids must contain exactly the
supplied three IDs once each in best-first order. assessments must contain exactly one object for each supplied ID,
with exactly these keys: item_id, public_evidence_grade.
"""

_DEEP_TAIL_COVERAGE_PROMPT = """You are the focused public-evidence tail calibrator for a shopping recommendation
slate. The action and fixed Top-3 are outside your control. Rank exactly seven IDs for positions 4 through 10 from
the supplied public reservoir.

The current tail is a strong prior. Introduce at most three IDs that are not in current_tail, and only when their
visible product identity, latest-answer fit, public DSUS constraint coverage, avoid safety, and catalog evidence
are clearly stronger. Seek complementary relevant coverage, not diversity for its own sake. Deep retrieval rank
only expands the evidence surface; it is not a relevance label. Use only the supplied public dialogue, public DSUS
retrieval frame, and public catalog fields. Do not infer a hidden user or use any benchmark, evaluator, expected
answer, method identity, or private state.

Return only one JSON object with exactly one key: item_ids. item_ids must contain exactly seven distinct supplied
tail-candidate IDs, must exclude every fixed_head ID, and may contain at most three IDs absent from current_tail.
"""

_QUESTION_PROMPT = """You execute the ASK operation of one selected Thought-Pattern Skill for a shopping Agent.
Use only the public dialogue, public DSUS state, and selected Skill. Realize one concise question whose answer would
materially change the product ranking. The question must name a concrete customer-facing distinction grounded in
the current product type, preferably as two plausible alternatives (for example, cushioning versus breathability).
Never ask a generic question such as "what property matters", "what do you prefer", or "can you clarify".
Do not infer or mention a hidden preference, target item, benchmark mechanism, evaluator, or expected answer.

Return only one JSON object with exactly these keys: question_target, utterance.
question_target must be a short concrete distinction. utterance must be a single natural question.
"""

_GENERIC_QUESTION_TARGETS = (
    "deciding preference",
    "single product property",
    "what property",
    "what do you prefer",
    "key preference",
)

_PUBLIC_HEAD_GUARD_VERSION = "public_top3_deterministic_guard_v18"
_DECISIVE_LATEST_MARKERS = frozenset(
    {
        "maximum",
        "minimize",
        "minimizing",
        "prioritize",
        "priority",
        "instead",
        "rather",
        "must",
        "without",
    }
)
_HEAD_GUARD_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "like",
        "me",
        "more",
        "must",
        "my",
        "need",
        "of",
        "on",
        "one",
        "option",
        "or",
        "our",
        "pack",
        "please",
        "prioritize",
        "recommend",
        "should",
        "something",
        "specific",
        "still",
        "suggest",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "these",
        "they",
        "this",
        "those",
        "to",
        "use",
        "want",
        "was",
        "we",
        "were",
        "while",
        "with",
        "without",
        "would",
        "yes",
        "you",
        "your",
    }
)
_AUDIENCE_LEXICON: dict[str, frozenset[str]] = {
    "female": frozenset({"female", "woman", "women", "womens", "girl", "girls"}),
    "male": frozenset({"male", "man", "men", "mens", "boy", "boys"}),
    "boy": frozenset({"boy", "boys"}),
    "girl": frozenset({"girl", "girls"}),
    "baby": frozenset({"baby", "babies"}),
    "newborn": frozenset({"newborn", "newborns"}),
    "youth": frozenset(
        {"youth", "kid", "kids", "child", "children", "toddler", "toddlers"}
    ),
    "adult": frozenset({"adult", "adults"}),
    "mature": frozenset({"mature"}),
    "teen": frozenset({"teen", "teens", "teenage"}),
}


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").replace("_", " ").split())


def _clean_terms(values: object) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError("executor term fields must be lists")
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _clean_text(raw)
        if value and value.casefold() not in seen:
            output.append(value)
            seen.add(value.casefold())
    return output


def _question_for_visible_axis(axis: str) -> SkillQuestionPlan:
    target = _clean_text(axis).strip(" ,;:.?")
    has_alternatives = " or " in target.casefold() or " versus " in target.casefold() or "," in target
    if has_alternatives:
        utterance = f"Which should I prioritize: {target}?"
    else:
        utterance = f"Should I prioritize {target}?"
    return SkillQuestionPlan(question_target=target, utterance=utterance)


class SkillGroundingPlan(StrictModel):
    query_focus: str = Field(min_length=1, max_length=500)
    target_type: str = Field(default="", max_length=160)
    must_terms: list[str] = Field(default_factory=list, max_length=10)
    preferred_terms: list[str] = Field(default_factory=list, max_length=16)
    avoid_terms: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("query_focus", "target_type", mode="before")
    @classmethod
    def clean_scalars(cls, value: object) -> str:
        return _clean_text(value)

    @field_validator("must_terms", "preferred_terms", "avoid_terms", mode="before")
    @classmethod
    def clean_term_fields(cls, value: object) -> list[str]:
        return _clean_terms(value)


class SkillQuestionPlan(StrictModel):
    question_target: str = Field(min_length=3, max_length=240)
    utterance: str = Field(min_length=6, max_length=500)

    @field_validator("question_target", "utterance", mode="before")
    @classmethod
    def clean_question_text(cls, value: object) -> str:
        return _clean_text(value)

    @field_validator("question_target")
    @classmethod
    def reject_generic_target(cls, value: str) -> str:
        folded = value.casefold()
        if any(marker in folded for marker in _GENERIC_QUESTION_TARGETS):
            raise ValueError("question_target must name a concrete ranking-changing distinction")
        return value

    @field_validator("utterance")
    @classmethod
    def require_question(cls, value: str) -> str:
        if "?" not in value:
            raise ValueError("utterance must be a question")
        return value


class FocusedHeadAssessment(StrictModel):
    """Method-blind visible-evidence assessment for one fixed head candidate."""

    item_id: str = Field(min_length=1)
    public_evidence_grade: int = Field(ge=0, le=2)

    @field_validator("item_id", mode="before")
    @classmethod
    def clean_item_id(cls, value: object) -> str:
        return _clean_text(value)


class FocusedHead3Selection(StrictModel):
    """Exact permutation returned by the public Top-3 calibrator."""

    item_ids: list[str] = Field(min_length=3, max_length=3)
    assessments: list[FocusedHeadAssessment] = Field(min_length=3, max_length=3)

    @field_validator("item_ids")
    @classmethod
    def validate_exact_unique_ids(cls, value: list[str]) -> list[str]:
        cleaned = [_clean_text(item_id) for item_id in value]
        if any(not item_id for item_id in cleaned):
            raise ValueError("focused head IDs cannot be blank")
        if len(set(cleaned)) != 3:
            raise ValueError("focused head must contain three distinct IDs")
        return cleaned

    @field_validator("assessments")
    @classmethod
    def validate_unique_assessment_ids(
        cls, value: list[FocusedHeadAssessment]
    ) -> list[FocusedHeadAssessment]:
        item_ids = [assessment.item_id for assessment in value]
        if len(set(item_ids)) != 3:
            raise ValueError("focused head assessments must cover three distinct IDs")
        return value


class FocusedTailSelection(StrictModel):
    """Exact seven-position tail returned by the public tail calibrator."""

    item_ids: list[str] = Field(min_length=7, max_length=7)

    @field_validator("item_ids")
    @classmethod
    def validate_exact_unique_ids(cls, value: list[str]) -> list[str]:
        cleaned = [_clean_text(item_id) for item_id in value]
        if any(not item_id for item_id in cleaned):
            raise ValueError("focused tail IDs cannot be blank")
        if len(set(cleaned)) != 7:
            raise ValueError("focused tail must contain seven distinct IDs")
        return cleaned


class PublicHeadGuardTrace(StrictModel):
    """Replayable audit of the deterministic public Top-3 guard."""

    rule_version: str = Field(default=_PUBLIC_HEAD_GUARD_VERSION, min_length=1)
    applied: bool = False
    reason: str = Field(min_length=1)
    original_head: list[str] = Field(min_length=3, max_length=3)
    calibrated_head: list[str] = Field(min_length=3, max_length=3)

    @field_validator("original_head", "calibrated_head")
    @classmethod
    def validate_head_permutation(cls, value: list[str]) -> list[str]:
        cleaned = [_clean_text(item_id) for item_id in value]
        if len(set(cleaned)) != 3 or any(not item_id for item_id in cleaned):
            raise ValueError("public head guard requires three distinct item IDs")
        return cleaned


class SkillExecutionResult(StrictModel):
    action: BenchmarkSystemAction
    search_spec: SearchSpec | None = None
    retrieval: CatalogRetrievalResult | None = None
    llm_audits: list[LLMCompletionAudit] = Field(default_factory=list)
    used_fallback: bool = False
    grounding_plan: SkillGroundingPlan | None = None
    question_plan: SkillQuestionPlan | None = None
    candidate_fusion: str | None = None
    pattern_candidate_count: int = 0
    dialogue_candidate_count: int = 0
    direct_candidate_count: int = 0
    fused_candidate_count: int = 0
    identity_anchor_applied: bool = False
    focused_head3_applied: bool = False
    focused_tail_applied: bool = False
    focused_tail_new_item_count: int = Field(default=0, ge=0, le=3)
    public_head_guard: PublicHeadGuardTrace | None = None
    executor_id: str = Field(min_length=1)


def _strict_json_object(content: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"executor JSON repeats key: {key}")
            output[key] = value
        return output

    payload = json.loads(content, object_pairs_hook=no_duplicates)
    if not isinstance(payload, dict):
        raise ValueError("executor completion must be a JSON object")
    return payload


def _audit(
    *,
    stage: str,
    attempt_index: int,
    completion: JsonCompletion | None,
    failure: Exception | None,
) -> LLMCompletionAudit:
    if completion is None and isinstance(failure, JsonResponseError):
        return LLMCompletionAudit(stage=stage, attempt_index=attempt_index,
            status="rejected", failure_kind=type(failure).__name__, **failure.audit_metadata)
    return LLMCompletionAudit(
        stage=stage,
        attempt_index=attempt_index,
        status="rejected" if failure is not None else "committed",
        requested_model=None if completion is None else completion.requested_model,
        returned_model=None if completion is None else completion.returned_model,
        finish_reason=None if completion is None else completion.finish_reason,
        raw_content_sha256=None if completion is None else content_sha256(completion.content),
        input_tokens=None if completion is None else completion.usage.input_tokens,
        output_tokens=None if completion is None else completion.usage.output_tokens,
        total_tokens=None if completion is None else completion.usage.total_tokens,
        reasoning_tokens=None if completion is None else completion.usage.reasoning_tokens,
        failure_kind=None if failure is None else type(failure).__name__,
    )


class LLMThoughtPatternExecutor:
    """Execute the selected Skill through one bounded public grounding stage."""

    def __init__(
        self,
        *,
        transport: JsonCompletionTransport,
        library: ThoughtPatternLibrary,
        planner_id: str,
        max_attempts: int = 2,
        shortlist_k: int = 50,
        recommendation_k: int = 10,
        enable_dialogue_view: bool = True,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("executor max_attempts must be positive")
        if not 1 <= recommendation_k <= shortlist_k <= 50:
            raise ValueError("executor requires 1 <= recommendation_k <= shortlist_k <= 50")
        self.transport = transport
        self.library = library
        self.max_attempts = max_attempts
        self.shortlist_k = shortlist_k
        self.recommendation_k = recommendation_k
        self.enable_dialogue_view = enable_dialogue_view
        version = (
            THOUGHT_PATTERN_EXECUTOR_VERSION
            if enable_dialogue_view
            else LEGACY_THOUGHT_PATTERN_EXECUTOR_VERSION
        )
        self.executor_id = f"{version}:{planner_id}"

    def _request(
        self,
        *,
        stage: str,
        system_prompt: str,
        payload: Mapping[str, object],
        parser: (
            type[SkillGroundingPlan]
            | type[SkillQuestionPlan]
            | type[DirectCandidateSelection]
            | type[FocusedHead3Selection]
            | type[FocusedTailSelection]
        ),
    ) -> tuple[
        SkillGroundingPlan
        | SkillQuestionPlan
        | DirectCandidateSelection
        | FocusedHead3Selection
        | FocusedTailSelection
        | None,
        list[LLMCompletionAudit],
    ]:
        audits: list[LLMCompletionAudit] = []
        retry_feedback: str | None = None
        for attempt_index in range(1, self.max_attempts + 1):
            completion: JsonCompletion | None = None
            try:
                completion = self.transport.complete_json(
                    system_prompt=system_prompt,
                    user_payload=payload,
                    retry_feedback=retry_feedback,
                )
                value = parser.model_validate(_strict_json_object(completion.content))
                audits.append(
                    _audit(
                        stage=stage,
                        attempt_index=attempt_index,
                        completion=completion,
                        failure=None,
                    )
                )
                return value, audits
            except Exception as exc:
                audits.append(
                    _audit(
                        stage=stage,
                        attempt_index=attempt_index,
                        completion=completion,
                        failure=exc,
                    )
                )
                retry_feedback = (
                    "Return only one valid JSON object with the exact requested keys and no markdown or prose."
                )
        return None, audits

    @staticmethod
    def _identity_safe_grounding(plan: SkillGroundingPlan) -> SkillGroundingPlan:
        """Keep hard retrieval terms for identity, not soft shopping attributes.

        The public prompt already asks the planner to use ``must_terms`` only
        for product identifiers, but compatible LLM endpoints can still move
        an attribute such as ``anti-aging`` between must and preferred across
        deterministic requests.  That small formatting choice should not
        replace the candidate universe.  Preserve target-type phrases,
        measurements/model strings, and proper-name literals as hard terms;
        demote ordinary descriptive attributes to preferred evidence.
        """

        target = plan.target_type.casefold().strip()

        def is_identity(term: str) -> bool:
            folded = term.casefold().strip()
            if target and (folded in target or target in folded):
                return True
            if any(character.isdigit() for character in term):
                return True
            words = [word.strip("\"'()[]{}.,:;") for word in term.split()]
            return any(word[:1].isupper() for word in words if word)

        hard = [term for term in plan.must_terms if is_identity(term)]
        demoted = [term for term in plan.must_terms if term not in hard]
        preferred = list(dict.fromkeys([*demoted, *plan.preferred_terms]))
        return plan.model_copy(
            update={"must_terms": hard, "preferred_terms": preferred[:16]},
            deep=True,
        )

    @staticmethod
    def _needs_public_identity_anchor(
        *,
        memory: PublicDialogueMemory,
        grounding_plan: SkillGroundingPlan,
        pattern_result: CatalogRetrievalResult,
        purity_threshold: float = 0.30,
    ) -> bool:
        """Detect query dilution before retrying one public retrieval view.

        An identity anchor is deliberately conditional.  Adding every target
        type as a hard term hurts media catalogs whose titles omit words such
        as ``album``.  We retry only outside the music domain, when a compact
        target is literally stated by the user, no equivalent hard identity
        already exists, and fewer than 30% of the first ten retrieved items
        even match that public product identity.
        """

        if memory.domain == "amazon_music":
            return False
        target = grounding_plan.target_type.casefold().strip()
        target_words = re.findall(r"[a-z0-9]+", target)
        if not 1 <= len(target_words) <= 4:
            return False
        if set(target_words).issubset({"product", "item", "goods", "option", "thing"}):
            return False

        def singular(word: str) -> str:
            if len(word) > 4 and word.endswith("ies"):
                return word[:-3] + "y"
            if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
                return word[:-1]
            return word

        target_tokens = [singular(word) for word in target_words]

        def contains_target(text: str) -> bool:
            tokens = [singular(word) for word in re.findall(r"[a-z0-9]+", text.casefold())]
            return any(
                tokens[index : index + len(target_tokens)] == target_tokens
                for index in range(max(0, len(tokens) - len(target_tokens) + 1))
            )

        if not contains_target(memory.retrieval_text):
            return False
        if any(contains_target(term) for term in grounding_plan.must_terms):
            return False
        head = pattern_result.results[:10]
        if not head:
            return False
        identity_matches = sum(
            contains_target(f"{item.title} {item.category}")
            for item in head
        )
        return identity_matches / len(head) < purity_threshold

    @staticmethod
    def _deterministic_public_grounding(
        *,
        proposal: SkillActionProposal,
        memory: PublicDialogueMemory,
    ) -> SkillGroundingPlan:
        """Recover a TAIRAS grounding plan from already-public state.

        A transient executor-model failure must not erase the catalog plan that
        DSUS and the selected Thought Pattern have already compiled.  Prefer
        the Skill-specific ``SearchSpec``, fill any missing fields from the
        public DSUS retrieval frame, and retain the raw public dialogue as the
        lexical query.  This helper is used only by the multi-view TAIRAS
        executor; the Legacy baseline keeps its original early-return path.
        """

        compiled = proposal.search_spec
        frame = proposal.state_hypothesis.retrieval_frame

        def unique_terms(*groups: list[str], limit: int) -> list[str]:
            values: list[str] = []
            seen: set[str] = set()
            for group in groups:
                for raw in group:
                    value = _clean_text(raw)
                    folded = value.casefold()
                    if value and folded not in seen:
                        values.append(value)
                        seen.add(folded)
                        if len(values) >= limit:
                            return values
            return values

        target_type = _clean_text(
            ("" if compiled is None else compiled.target_type) or frame.target_type
        )
        must_terms = unique_terms(
            [] if compiled is None else compiled.must_terms,
            frame.must_terms,
            limit=10,
        )
        preferred_terms = unique_terms(
            [] if compiled is None else compiled.preferred_terms,
            frame.preferred_terms,
            limit=16,
        )
        avoid_terms = unique_terms(
            [] if compiled is None else compiled.avoid_terms,
            frame.avoid_terms,
            limit=10,
        )
        query_parts = [
            memory.retrieval_text,
            "" if compiled is None else compiled.query,
            target_type,
            *must_terms,
            *preferred_terms,
        ]
        query_focus = _clean_text(" ".join(part for part in query_parts if part))
        if len(query_focus) > 500:
            bounded = query_focus[:500]
            query_focus = bounded.rsplit(" ", 1)[0] or bounded
        if not query_focus:
            query_focus = target_type or "catalog product"
        return SkillGroundingPlan(
            query_focus=query_focus,
            target_type=target_type,
            must_terms=must_terms,
            preferred_terms=preferred_terms,
            avoid_terms=avoid_terms,
        )

    @staticmethod
    def _merge_public_grounding_plans(
        *,
        primary: SkillGroundingPlan,
        state_anchor: SkillGroundingPlan,
    ) -> SkillGroundingPlan:
        """Keep DSUS public constraints when the LLM supplies synonyms.

        The executor model is useful for catalog-native expansion, but one
        stochastic paraphrase must not erase a literal term already compiled
        by DSUS.  The state plan therefore anchors target, constraints, and
        avoid evidence; the LLM plan remains as an additive semantic view.
        """

        def unique_terms(*groups: list[str], limit: int) -> list[str]:
            output: list[str] = []
            seen: set[str] = set()
            for group in groups:
                for raw in group:
                    value = _clean_text(raw)
                    folded = value.casefold()
                    if value and folded not in seen:
                        output.append(value)
                        seen.add(folded)
                        if len(output) >= limit:
                            return output
            return output

        target_type = state_anchor.target_type or primary.target_type
        must_terms = unique_terms(
            state_anchor.must_terms,
            primary.must_terms,
            limit=10,
        )
        preferred_terms = unique_terms(
            state_anchor.preferred_terms,
            primary.preferred_terms,
            limit=16,
        )
        avoid_terms = unique_terms(
            state_anchor.avoid_terms,
            primary.avoid_terms,
            limit=10,
        )
        anchor_terms = [target_type, *must_terms, *preferred_terms]
        query_focus = _clean_text(
            " ".join([primary.query_focus, *(term for term in anchor_terms if term)])
        )
        if len(query_focus) > 500:
            bounded = query_focus[:500]
            query_focus = bounded.rsplit(" ", 1)[0] or bounded
        return SkillGroundingPlan(
            query_focus=query_focus,
            target_type=target_type,
            must_terms=must_terms,
            preferred_terms=preferred_terms,
            avoid_terms=avoid_terms,
        )

    @staticmethod
    def _fuse_candidates(
        results: tuple[CatalogRetrievalResult, ...],
        *,
        limit: int,
    ) -> list[CatalogRetrievalItem]:
        """Interleave specialized, semantic, and raw-query public evidence.

        The Pattern path remains first. The independent dialogue view protects
        semantic catalog translation, while the original public query protects
        literal evidence that either LLM rewrite accidentally filtered out.
        """

        fused: list[CatalogRetrievalItem] = []
        seen: set[str] = set()
        width = max((len(result.results) for result in results), default=0)
        for position in range(width):
            for result in results:
                if position >= len(result.results):
                    continue
                item = result.results[position]
                if item.item_id in seen:
                    continue
                fused.append(item)
                seen.add(item.item_id)
                if len(fused) >= limit:
                    return fused
        return fused

    @staticmethod
    def _rank_aware_evidence_diverse_slate(
        *,
        selected_ids: list[str],
        pattern_result: CatalogRetrievalResult,
        dialogue_result: CatalogRetrievalResult | None,
        direct_result: CatalogRetrievalResult,
        fused_items: list[CatalogRetrievalItem],
        limit: int,
        protected_prefix: int = 5,
    ) -> list[str]:
        """Stabilize the semantic head while protecting Top-k evidence recall.

        V7 placed one candidate from every retrieval view directly after the
        LLM's first item. That improved broad Top-10 coverage but could
        overwrite an otherwise useful Top-3 ranking. V11 keeps the first five
        LLM-ranked items intact and fills the recall half from the head of the
        already interleaved multi-view retrieval. This retains independent
        public evidence without letting one stochastic verifier response erase
        every high-ranked lexical/semantic candidate from the final Top-10.

        V12 adds a deliberately bounded head consensus.  It reorders only the
        verifier's existing Top-3 using three public ranks: the verifier rank,
        the catalog retrieval score rank, and the first position in the
        Pattern/dialogue/raw-query fusion.  The verifier receives triple
        weight, so a challenger can replace its first item only when both
        independent retrieval signals agree and its public retrieval score is
        at least 1.5 times the verifier head's score.  The Top-3 and Top-10
        sets are unchanged; no evaluator label or hidden benchmark state is
        available.
        """

        by_id = {item.item_id: item for item in fused_items}
        fusion_position = {
            item.item_id: position
            for position, item in enumerate(fused_items, start=1)
        }
        head = [item_id for item_id in selected_ids[:3] if item_id in by_id]
        if len(head) >= 2:
            verifier_rank = {
                item_id: position
                for position, item_id in enumerate(head, start=1)
            }
            score_order = sorted(
                head,
                key=lambda item_id: (
                    -float(by_id[item_id].score),
                    verifier_rank[item_id],
                ),
            )
            score_rank = {
                item_id: position
                for position, item_id in enumerate(score_order, start=1)
            }
            fusion_order = sorted(
                head,
                key=lambda item_id: (
                    fusion_position[item_id],
                    verifier_rank[item_id],
                ),
            )
            fusion_rank = {
                item_id: position
                for position, item_id in enumerate(fusion_order, start=1)
            }
            consensus_order = sorted(
                head,
                key=lambda item_id: (
                    3 * verifier_rank[item_id]
                    + score_rank[item_id]
                    + fusion_rank[item_id],
                    verifier_rank[item_id],
                ),
            )
            consensus_head = head
            challenger = consensus_order[0]
            verifier_head = head[0]
            verifier_score = float(by_id[verifier_head].score)
            challenger_score = float(by_id[challenger].score)
            high_confidence_promotion = (
                challenger != verifier_head
                and challenger_score > verifier_score
                and challenger_score >= 1.5 * max(verifier_score, 1e-12)
            )
            if high_confidence_promotion:
                consensus_head = [
                    challenger,
                    *[item_id for item_id in head if item_id != challenger],
                ]
            selected_ids = [
                *consensus_head,
                *selected_ids[len(head) :],
            ]

        prefix = selected_ids[:protected_prefix]
        sources: list[str] = [*prefix]
        sources.extend(item.item_id for item in fused_items[:protected_prefix])
        if dialogue_result is not None and dialogue_result.results:
            sources.append(dialogue_result.results[0].item_id)
        if pattern_result.results:
            sources.append(pattern_result.results[0].item_id)
        if direct_result.results:
            sources.append(direct_result.results[0].item_id)
        sources.extend(selected_ids[protected_prefix:])
        sources.extend(item.item_id for item in fused_items[protected_prefix:])
        return list(dict.fromkeys(sources))[:limit]

    @staticmethod
    def _complete_public_multiview_slate(
        *,
        selected_ids: list[str],
        pattern_result: CatalogRetrievalResult,
        dialogue_result: CatalogRetrievalResult | None,
        direct_result: CatalogRetrievalResult,
        fused_items: list[CatalogRetrievalItem],
        limit: int,
        protected_prefix: int = 5,
    ) -> list[str]:
        """Complete a verifier slate without comparing cross-query scores.

        The verifier may return fewer than ten IDs. Preserve its leading five,
        then fill the remaining positions from the already public multi-view
        retrieval. This is the set-construction portion of the historical v14
        helper, separated so online v15 never inherits its incomparable
        raw-score head heuristic.
        """

        prefix = selected_ids[:protected_prefix]
        sources: list[str] = [*prefix]
        sources.extend(item.item_id for item in fused_items[:protected_prefix])
        if dialogue_result is not None and dialogue_result.results:
            sources.append(dialogue_result.results[0].item_id)
        if pattern_result.results:
            sources.append(pattern_result.results[0].item_id)
        if direct_result.results:
            sources.append(direct_result.results[0].item_id)
        sources.extend(selected_ids[protected_prefix:])
        sources.extend(item.item_id for item in fused_items[protected_prefix:])
        return list(dict.fromkeys(sources))[:limit]

    @staticmethod
    def _matches_public_target_identity(
        *,
        target_type: str,
        item: CatalogRetrievalItem,
    ) -> bool:
        """Check public product identity with token boundaries.

        Head promotion is optional, so this check favors abstention over a
        loose substring match. A tiny morphology map covers ordinary catalog
        variants such as ``face``/``facial`` without reading private targets.
        """

        generic = {
            "a",
            "an",
            "catalog",
            "choice",
            "goods",
            "item",
            "kind",
            "one",
            "option",
            "product",
            "recommendation",
            "some",
            "the",
            "thing",
            "type",
        }
        aliases = {
            "facial": "face",
            "faces": "face",
            "tshirt": "tee",
            "tshirts": "tee",
            "straightening": "straightener",
        }

        def canonical_tokens(text: str) -> list[str]:
            output: list[str] = []
            for raw in re.findall(r"[a-z0-9]+", text.casefold()):
                if raw == "s":
                    continue
                token = aliases.get(raw, raw)
                if len(token) > 4 and token.endswith("ies"):
                    token = token[:-3] + "y"
                elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                    token = token[:-1]
                output.append(aliases.get(token, token))
            return output

        target_tokens = [
            token for token in canonical_tokens(target_type) if token not in generic
        ]
        if not target_tokens:
            return False
        item_tokens = set(canonical_tokens(f"{item.title} {item.category}"))
        return all(token in item_tokens for token in target_tokens)

    @classmethod
    def _evidence_consistent_existing_slate(
        cls,
        *,
        selected_ids: list[str],
        pattern_result: CatalogRetrievalResult,
        dialogue_result: CatalogRetrievalResult | None,
        direct_result: CatalogRetrievalResult,
        fused_items: list[CatalogRetrievalItem],
        target_type: str,
        domain: str,
        limit: int,
    ) -> list[str]:
        """Promote one independently supported item inside an existing Top-10.

        Scores from different ``SearchSpec`` objects are not calibrated and
        never enter this decision. A candidate must match the public target
        identity, avoid every public negative term, and satisfy a conservative
        Pattern- or Direct-rank gate. No candidate is inserted or removed.
        """

        slate = list(dict.fromkeys(selected_ids))[:limit]
        if len(slate) < 2 or domain == "amazon_music":
            return slate

        views = [pattern_result]
        if dialogue_result is not None:
            views.append(dialogue_result)
        views.append(direct_result)
        ranks = [
            {
                item.item_id: position
                for position, item in enumerate(result.results, start=1)
            }
            for result in views
        ]
        pattern_rank = ranks[0]
        direct_rank = ranks[-1]

        preferred: dict[str, set[str]] = {}
        avoided: dict[str, set[str]] = {}
        public_items: dict[str, CatalogRetrievalItem] = {}
        for item in [
            *(item for result in views for item in result.results),
            *fused_items,
        ]:
            public_items.setdefault(item.item_id, item)
            preferred.setdefault(item.item_id, set()).update(
                term.casefold().strip()
                for term in item.matched_preferred_terms
                if term.strip()
            )
            avoided.setdefault(item.item_id, set()).update(
                term.casefold().strip()
                for term in item.matched_avoid_terms
                if term.strip()
            )

        head_id = slate[0]
        head_preferred_count = len(preferred.get(head_id, set()))
        original_rank = {
            item_id: position for position, item_id in enumerate(slate, start=1)
        }
        eligible: list[str] = []
        for item_id in slate[1:]:
            item = public_items.get(item_id)
            if item is None or avoided.get(item_id):
                continue
            if not cls._matches_public_target_identity(
                target_type=target_type,
                item=item,
            ):
                continue
            preferred_count = len(preferred.get(item_id, set()))
            inside_top3_gate = (
                original_rank[item_id] <= 3
                and pattern_rank.get(item_id, 10**9) <= 3
                and preferred_count >= head_preferred_count + 1
            )
            direct_head_gate = (
                original_rank[item_id] > 3
                and direct_rank.get(item_id) == 1
                and preferred_count >= 1
            )
            if inside_top3_gate or direct_head_gate:
                eligible.append(item_id)

        if not eligible:
            return slate

        def support_key(item_id: str) -> tuple[float, ...]:
            view_ranks = [rank[item_id] for rank in ranks if item_id in rank]
            support_at_10 = sum(position <= 10 for position in view_ranks)
            reciprocal_rank = sum(1.0 / (60 + position) for position in view_ranks)
            return (
                float(len(preferred.get(item_id, set()))),
                float(support_at_10),
                reciprocal_rank,
                float(-min(view_ranks, default=10**9)),
                float(-original_rank[item_id]),
            )

        winner = max(eligible, key=support_key)
        reranked = [winner, *[item_id for item_id in slate if item_id != winner]]
        if len(reranked) != len(slate) or set(reranked) != set(slate):
            raise AssertionError("public head consensus changed the completed slate set")
        return reranked

    @staticmethod
    def _selection_payload(
        *,
        memory: PublicDialogueMemory,
        result: CatalogRetrievalResult,
        grounding_plan: SkillGroundingPlan,
        dialogue_grounding_plan: SkillGroundingPlan | None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "task": "select_ranked_catalog_items_after_thought_pattern_execution",
            "domain": memory.domain,
            "public_dialogue": memory.serialized_dialogue,
            "thought_pattern_grounding_plan": grounding_plan.model_dump(mode="json"),
            "search_spec": result.search_spec.model_dump(mode="json"),
            "candidates": [
                {
                    "item_id": item.item_id,
                    "title": item.title,
                    "category": item.category,
                    "matched_must_terms": item.matched_must_terms,
                    "matched_preferred_terms": item.matched_preferred_terms,
                }
                for item in result.results
            ],
            "instruction": "Select up to ten IDs from the supplied public candidates only.",
        }
        if dialogue_grounding_plan is not None:
            payload["independent_dialogue_grounding_plan"] = dialogue_grounding_plan.model_dump(mode="json")
        return payload

    @staticmethod
    def _verification_candidates(
        *,
        provisional_ids: list[str],
        pattern_result: CatalogRetrievalResult,
        dialogue_result: CatalogRetrievalResult | None,
        direct_result: CatalogRetrievalResult,
        fused_items: list[CatalogRetrievalItem],
        limit: int = 30,
    ) -> list[CatalogRetrievalItem]:
        """Build a bounded public candidate union for independent verification."""

        by_id = {item.item_id: item for item in fused_items}
        ordered_ids: list[str] = [*provisional_ids]
        for result in (pattern_result, dialogue_result, direct_result):
            if result is not None:
                ordered_ids.extend(item.item_id for item in result.results[:10])
        ordered_ids.extend(item.item_id for item in fused_items)
        return [
            by_id[item_id]
            for item_id in dict.fromkeys(ordered_ids)
            if item_id in by_id
        ][:limit]

    @staticmethod
    def _declare_final_public_slate(
        *,
        fused_retrieval: CatalogRetrievalResult,
        selected_ids: list[str],
        pattern_result: CatalogRetrievalResult,
        dialogue_result: CatalogRetrievalResult | None,
        direct_result: CatalogRetrievalResult,
    ) -> CatalogRetrievalResult:
        """Declare every selected public candidate without changing the fused prefix.

        The focused tail verifier may select a deep item from any public
        retrieval view.  Such an item is grounded, but it can sit beyond the
        bounded interleaved ``fused_retrieval``.  Append only missing selected
        items, in final-slate order, so the runner's action/retrieval contract
        remains explicit while the original fused evidence prefix is intact.
        """

        public_by_id: dict[str, CatalogRetrievalItem] = {}
        for result in (pattern_result, dialogue_result, direct_result):
            if result is None:
                continue
            for item in result.results:
                public_by_id.setdefault(item.item_id, item)
        declared_items = list(fused_retrieval.results)
        declared_ids = {item.item_id for item in declared_items}
        for item_id in selected_ids:
            if item_id in declared_ids:
                continue
            item = public_by_id.get(item_id)
            if item is None:
                raise AssertionError(
                    f"selected item {item_id!r} is absent from every public retrieval view"
                )
            declared_items.append(item)
            declared_ids.add(item_id)
        if not set(selected_ids).issubset(declared_ids):
            raise AssertionError("declared retrieval does not cover the final public slate")
        if declared_items == fused_retrieval.results:
            return fused_retrieval
        return fused_retrieval.model_copy(update={"results": declared_items}, deep=True)

    @staticmethod
    def _deep_tail_reservoir_ids(
        *,
        current_slate: list[str],
        provisional_ids: list[str],
        pattern_result: CatalogRetrievalResult,
        dialogue_result: CatalogRetrievalResult | None,
        direct_result: CatalogRetrievalResult,
        depth_per_view: int = 30,
    ) -> list[str]:
        """Expose deeper public evidence while leaving the current head fixed.

        The ordinary verifier is deliberately head-oriented: it sees the
        provisional ranking plus only the first ten results of each retrieval
        view.  Semantic NDCG@10 additionally needs a useful coverage tail.
        This reservoir therefore retains the current tail and then interleaves
        the first thirty public results of every view.  It never reads an
        evaluator label or compares scores produced by different queries.
        """

        if depth_per_view < 1:
            raise ValueError("deep tail retrieval depth must be positive")
        fixed_head = set(current_slate[:3])
        ordered_ids = [*current_slate[3:], *provisional_ids[3:]]
        views = [pattern_result]
        if dialogue_result is not None:
            views.append(dialogue_result)
        views.append(direct_result)
        for position in range(depth_per_view):
            for result in views:
                if position < len(result.results):
                    ordered_ids.append(result.results[position].item_id)
        return [
            item_id
            for item_id in dict.fromkeys(ordered_ids)
            if item_id not in fixed_head
        ]

    @staticmethod
    def _public_term_matches(text: str, terms: list[str]) -> list[str]:
        """Return literal public catalog coverage under punctuation folding."""

        folded_text = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
        matched: list[str] = []
        for term in terms:
            folded_term = " ".join(re.findall(r"[a-z0-9]+", term.casefold()))
            if folded_term and folded_term in folded_text:
                matched.append(term)
        return matched

    @classmethod
    def _public_exact_value_literals(
        cls,
        text: str,
        *,
        domain: str | None = None,
    ) -> list[str]:
        """Extract narrow, customer-visible numeric requirements.

        This guard intentionally ignores ordinary lexical overlap.  It covers
        only measurements/capacities and explicit ``size N`` expressions, for
        example ``40-inch`` or ``size 10``.  Those values can be lost by a
        compact state compiler even though they remain plainly available in
        the latest public utterance.
        """

        normalized = " ".join(cls._public_guard_tokens(text))
        literals: list[str] = []
        measurement = re.compile(
            r"\b(\d+(?:\.\d+)?)\s+"
            r"(inch|foot|mm|cm|meter|ounce|ml|liter|pound|gb|tb|watt|month|year)\b"
        )
        for match in measurement.finditer(normalized):
            literals.append(match.group(0))
        for match in re.finditer(r"\bsize\s+(\d+(?:\.\d+)?)\b", normalized):
            literals.append(f"size {match.group(1)}")
        # Counted attribute structures are stronger than bare package counts.
        # The final relational noun is a fixed, domain-independent vocabulary;
        # this captures expressions such as ``two sleeve lengths`` without
        # treating every ``pack of six`` as a discriminating attribute.
        relational_nouns = {
            "compartment",
            "feature",
            "layer",
            "length",
            "mode",
            "setting",
            "size",
            "speed",
        }
        for match in re.finditer(r"\b(\d+)\s+([a-z]+)\s+([a-z]+)\b", normalized):
            if match.group(3) in relational_nouns:
                literals.append(match.group(0))
        if domain == "amazon_clothing":
            # Apparel catalogs commonly compress waist/inseam as ``40x32``,
            # while customers say ``40-inch waist``.  Normalize both public
            # expressions to the same role-aware literal.
            for match in re.finditer(
                r"\b(\d+(?:\.\d+)?)\s+(?:inch\s+)?waist\b", normalized
            ):
                literals.append(f"waist:{match.group(1)}")
            for match in re.finditer(
                r"\bwaist(?:\s+of)?\s+(\d+(?:\.\d+)?)(?:\s+inch)?\b",
                normalized,
            ):
                literals.append(f"waist:{match.group(1)}")
            for match in re.finditer(
                r"\b(\d+(?:\.\d+)?)\s+(?:inch\s+)?inseam\b", normalized
            ):
                literals.append(f"inseam:{match.group(1)}")
            for match in re.finditer(
                r"\binseam(?:\s+of)?\s+(\d+(?:\.\d+)?)(?:\s+inch)?\b",
                normalized,
            ):
                literals.append(f"inseam:{match.group(1)}")
            for match in re.finditer(
                r"\b(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)\b", normalized
            ):
                literals.extend(
                    (f"waist:{match.group(1)}", f"inseam:{match.group(2)}")
                )
        return list(dict.fromkeys(literals))

    @staticmethod
    def _public_guard_tokens(text: str) -> list[str]:
        """Normalize only deterministic public literals used by the v18 guard."""

        number_words = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
        }
        aliases = {
            "feet": "foot",
            "fl": "",
            "fluid": "",
            "in": "inch",
            "inches": "inch",
            "lbs": "pound",
            "lb": "pound",
            "liters": "liter",
            "meters": "meter",
            "months": "month",
            "ounces": "ounce",
            "oz": "ounce",
            "pounds": "pound",
            "watts": "watt",
            "years": "year",
        }

        def singular(token: str) -> str:
            if token.endswith("ies") and len(token) > 4:
                return f"{token[:-3]}y"
            if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
                return token[:-1]
            return token

        raw_tokens = re.findall(
            r"\d+(?:\.\d+)?(?:x\d+(?:\.\d+)?)?|[a-z]+",
            text.casefold().replace("×", "x"),
        )
        normalized: list[str] = []
        for raw_token in raw_tokens:
            token = aliases.get(number_words.get(raw_token, raw_token), number_words.get(raw_token, raw_token))
            if token:
                normalized.append(singular(token))
        return normalized

    @classmethod
    def _public_audience_groups(cls, text: str) -> set[str]:
        tokens = set(cls._public_guard_tokens(text))
        return {
            group
            for group, forms in _AUDIENCE_LEXICON.items()
            if tokens.intersection(forms)
        }

    @staticmethod
    def _public_audience_conflicts(desired: set[str], candidate: set[str]) -> int:
        conflicts = 0
        if "boy" in desired and candidate.intersection({"female", "adult"}):
            conflicts += 1
        if "girl" in desired and candidate.intersection({"male", "adult"}):
            conflicts += 1
        if desired.intersection({"baby", "newborn", "youth"}) and "adult" in candidate:
            conflicts += 1
        if desired == {"male"} and "female" in candidate:
            conflicts += 1
        if desired == {"female"} and "male" in candidate:
            conflicts += 1
        return conflicts

    @classmethod
    def _public_deterministic_head_guard(
        cls,
        *,
        current_head: list[str],
        candidate_rows: list[dict[str, object]],
        memory: PublicDialogueMemory,
        proposal: SkillActionProposal,
    ) -> tuple[list[str], PublicHeadGuardTrace]:
        """Apply the frozen v18 method-blind Top-3 public-evidence guard."""

        original_head = list(current_head)
        if len(original_head) != 3 or len(set(original_head)) != 3:
            raise ValueError("public head guard requires an exact Top-3")
        rows = {str(row["item_id"]): row for row in candidate_rows}
        if set(rows) != set(original_head):
            raise ValueError("public head guard rows must exactly cover current_head")

        frame = proposal.state_hypothesis.retrieval_frame
        domain = memory.domain
        public_user_text = memory.retrieval_text
        public_values = set(cls._public_exact_value_literals(public_user_text, domain=domain))
        if not public_values:
            compiler_text = " ".join(
                [
                    frame.target_type,
                    *frame.must_terms,
                    *frame.preferred_terms,
                ]
            )
            public_values = set(
                cls._public_exact_value_literals(compiler_text, domain=domain)
            )
        desired_audience = cls._public_audience_groups(public_user_text)

        latest_text = ""
        if memory.completed_action_count > 0:
            candidate_latest = memory.latest_user_utterance
            generic_markers = (
                "these options do not meet",
                "not quite right",
                "this recommendation fits",
                "thank you",
            )
            if not any(marker in candidate_latest.casefold() for marker in generic_markers):
                latest_text = candidate_latest
        latest_tokens = {
            token
            for token in cls._public_guard_tokens(latest_text)
            if token not in _HEAD_GUARD_STOPWORDS
            and len(token) > 2
            and not token.replace(".", "", 1).isdigit()
        }
        ordered_latest_tokens = cls._public_guard_tokens(latest_text)
        raw_latest_tokens = set(ordered_latest_tokens)
        decisive_latest = bool(raw_latest_tokens.intersection(_DECISIVE_LATEST_MARKERS)) or (
            "care more" in " ".join(ordered_latest_tokens)
        )

        target_signature = [
            token
            for token in cls._public_guard_tokens(frame.target_type)
            if token not in _HEAD_GUARD_STOPWORDS
            and not token.replace(".", "", 1).isdigit()
        ]
        target_signature_set = set(target_signature)
        target_head = target_signature[-1] if target_signature else ""

        features: dict[str, dict[str, object]] = {}
        for item_id in original_head:
            row = rows[item_id]
            catalog_text = f"{row.get('title', '')} {row.get('category', '')}"
            catalog_tokens = set(cls._public_guard_tokens(catalog_text))
            value_coverage = len(
                public_values.intersection(
                    cls._public_exact_value_literals(catalog_text, domain=domain)
                )
            )
            candidate_audience = cls._public_audience_groups(catalog_text)
            audience_coverage = len(desired_audience.intersection(candidate_audience))
            latest_coverage = len(latest_tokens.intersection(catalog_tokens))
            audience_conflicts = cls._public_audience_conflicts(
                desired_audience, candidate_audience
            )
            avoid_count = len(cls._public_term_matches(catalog_text, frame.avoid_terms))
            features[item_id] = {
                "q": value_coverage,
                "a": audience_coverage,
                "l": latest_coverage,
                "c": audience_conflicts,
                "score": (
                    4 * value_coverage
                    + 3 * audience_coverage
                    + 2 * latest_coverage
                    - 8 * audience_conflicts
                ),
                "avoid": avoid_count,
                "identity_head": bool(target_head and target_head in catalog_tokens),
                "identity_signature": len(target_signature_set.intersection(catalog_tokens)),
            }

        base_id = original_head[0]
        base = features[base_id]
        eligible: list[tuple[int, int, str, str]] = []
        for rank, item_id in enumerate(original_head[1:], start=2):
            challenger = features[item_id]
            improves_anchor = (
                int(challenger["q"]) > int(base["q"])
                or int(challenger["a"]) > int(base["a"])
                or int(challenger["c"]) < int(base["c"])
                or (
                    decisive_latest
                    and int(challenger["l"]) > int(base["l"])
                )
            )
            safe = (
                int(challenger["avoid"]) <= int(base["avoid"])
                and int(challenger["c"]) <= int(base["c"])
                and int(challenger["q"]) >= int(base["q"])
                and int(challenger["a"]) >= int(base["a"])
                and (
                    not bool(base["identity_head"])
                    or bool(challenger["identity_head"])
                )
            )
            if (
                safe
                and improves_anchor
                and int(challenger["score"]) - int(base["score"]) >= 2
            ):
                eligible.append(
                    (
                        int(challenger["score"]),
                        -rank,
                        item_id,
                        "decisive_public_anchor",
                    )
                )

        # A long, concrete target phrase can mechanically distinguish a wrong
        # product form (for example jars) from the requested form (bottles).
        # Never apply this rule to a one-word label such as album or record.
        if len(target_signature_set) >= 3:
            identity_candidates: list[str] = []
            for item_id in original_head[1:]:
                challenger = features[item_id]
                if (
                    int(challenger["identity_signature"])
                    >= int(base["identity_signature"]) + 2
                    and bool(challenger["identity_head"])
                    and int(challenger["avoid"]) <= int(base["avoid"])
                    and int(challenger["c"]) <= int(base["c"])
                    and int(challenger["q"]) >= int(base["q"])
                    and int(challenger["a"]) >= int(base["a"])
                ):
                    identity_candidates.append(item_id)
            if len(identity_candidates) == 1:
                item_id = identity_candidates[0]
                eligible.append(
                    (
                        100 + int(features[item_id]["identity_signature"]),
                        -original_head.index(item_id),
                        item_id,
                        "specific_identity_signature",
                    )
                )

        if not eligible:
            trace = PublicHeadGuardTrace(
                applied=False,
                reason="no_decisive_public_advantage",
                original_head=original_head,
                calibrated_head=original_head,
            )
            return original_head, trace
        _, _, winner, reason = max(eligible)
        calibrated = [winner, *[item_id for item_id in original_head if item_id != winner]]
        trace = PublicHeadGuardTrace(
            applied=True,
            reason=reason,
            original_head=original_head,
            calibrated_head=calibrated,
        )
        return calibrated, trace

    @classmethod
    def _public_candidate_evidence(
        cls,
        *,
        candidate_ids: list[str],
        current_slate: list[str],
        provisional_ids: list[str],
        proposal: SkillActionProposal,
        grounding_plan: SkillGroundingPlan,
        pattern_result: CatalogRetrievalResult,
        dialogue_result: CatalogRetrievalResult | None,
        direct_result: CatalogRetrievalResult,
        fused_items: list[CatalogRetrievalItem],
    ) -> list[dict[str, object]]:
        """Build method-blind evidence rows from the public catalog surface."""

        named_views: list[tuple[str, CatalogRetrievalResult]] = [("pattern", pattern_result)]
        if dialogue_result is not None:
            named_views.append(("dialogue", dialogue_result))
        named_views.append(("direct", direct_result))

        public_items: dict[str, CatalogRetrievalItem] = {}
        view_ranks: dict[str, dict[str, int]] = {}
        retrieval_must: dict[str, set[str]] = {}
        retrieval_preferred: dict[str, set[str]] = {}
        retrieval_avoid: dict[str, set[str]] = {}
        for view_name, result in named_views:
            for rank, item in enumerate(result.results, start=1):
                public_items.setdefault(item.item_id, item)
                view_ranks.setdefault(item.item_id, {})[view_name] = rank
                retrieval_must.setdefault(item.item_id, set()).update(item.matched_must_terms)
                retrieval_preferred.setdefault(item.item_id, set()).update(
                    item.matched_preferred_terms
                )
                retrieval_avoid.setdefault(item.item_id, set()).update(item.matched_avoid_terms)
        for item in fused_items:
            public_items.setdefault(item.item_id, item)
            retrieval_must.setdefault(item.item_id, set()).update(item.matched_must_terms)
            retrieval_preferred.setdefault(item.item_id, set()).update(
                item.matched_preferred_terms
            )
            retrieval_avoid.setdefault(item.item_id, set()).update(item.matched_avoid_terms)

        frame = proposal.state_hypothesis.retrieval_frame
        required_terms = list(dict.fromkeys([*frame.must_terms, *grounding_plan.must_terms]))
        preferred_terms = list(
            dict.fromkeys([*frame.preferred_terms, *grounding_plan.preferred_terms])
        )
        avoid_terms = list(dict.fromkeys([*frame.avoid_terms, *grounding_plan.avoid_terms]))
        current_rank = {
            item_id: rank for rank, item_id in enumerate(current_slate, start=1)
        }
        provisional_rank = {
            item_id: rank for rank, item_id in enumerate(provisional_ids, start=1)
        }

        rows: list[dict[str, object]] = []
        for item_id in candidate_ids:
            item = public_items.get(item_id)
            if item is None:
                continue
            catalog_text = f"{item.title} {item.category}"
            literal_required = cls._public_term_matches(catalog_text, required_terms)
            literal_preferred = cls._public_term_matches(catalog_text, preferred_terms)
            literal_avoid = cls._public_term_matches(catalog_text, avoid_terms)
            rows.append(
                {
                    "item_id": item.item_id,
                    "title": item.title,
                    "category": item.category,
                    "current_rank": current_rank.get(item.item_id),
                    "provisional_rank": provisional_rank.get(item.item_id),
                    "retrieval_view_ranks": view_ranks.get(item.item_id, {}),
                    "public_constraint_coverage": {
                        "required_terms": literal_required,
                        "preferred_terms": literal_preferred,
                        "retrieval_matched_required_terms": sorted(
                            retrieval_must.get(item.item_id, set()),
                            key=str.casefold,
                        ),
                        "retrieval_matched_preferred_terms": sorted(
                            retrieval_preferred.get(item.item_id, set()),
                            key=str.casefold,
                        ),
                    },
                    "public_avoid_violations": sorted(
                        set(literal_avoid).union(retrieval_avoid.get(item.item_id, set())),
                        key=str.casefold,
                    ),
                }
            )
        return rows

    def _focused_semantic_slate_calibration(
        self,
        *,
        current_slate: list[str],
        provisional_ids: list[str],
        proposal: SkillActionProposal,
        memory: PublicDialogueMemory,
        grounding_plan: SkillGroundingPlan,
        dialogue_grounding_plan: SkillGroundingPlan | None,
        pattern_result: CatalogRetrievalResult,
        dialogue_result: CatalogRetrievalResult | None,
        direct_result: CatalogRetrievalResult,
        fused_items: list[CatalogRetrievalItem],
    ) -> tuple[
        list[str],
        list[LLMCompletionAudit],
        bool,
        bool,
        int,
        PublicHeadGuardTrace | None,
        bool,
    ]:
        """Atomically calibrate an exact Top-3 permutation and seven-item tail.

        The protocol is intentionally fail-closed.  Any parser error, unknown
        ID, set-size violation, or excessive tail replacement returns the
        complete v16 slate unchanged.  The focused calls are attempted only on
        a complete Top-10, so smaller catalogs keep their established path.
        """

        original_slate = list(dict.fromkeys(current_slate))[: self.recommendation_k]
        if self.recommendation_k != 10 or len(original_slate) != 10:
            return original_slate, [], False, False, 0, None, False

        fixed_head = original_slate[:3]
        head_rows = self._public_candidate_evidence(
            candidate_ids=fixed_head,
            current_slate=original_slate,
            provisional_ids=provisional_ids,
            proposal=proposal,
            grounding_plan=grounding_plan,
            pattern_result=pattern_result,
            dialogue_result=dialogue_result,
            direct_result=direct_result,
            fused_items=fused_items,
        )
        if len(head_rows) != 3:
            return original_slate, [], False, False, 0, None, True
        public_frame = proposal.state_hypothesis.retrieval_frame.model_dump(mode="json")
        head_value, head_audits = self._request(
            stage="skill_head3_consensus",
            system_prompt=_FOCUSED_HEAD3_CONSENSUS_PROMPT,
            payload={
                "task": "rank_existing_public_head3_only",
                "domain": memory.domain,
                "public_dialogue": memory.serialized_dialogue,
                "latest_user_utterance": memory.latest_user_utterance,
                "public_dsus_retrieval_frame": public_frame,
                "public_grounding_plan": grounding_plan.model_dump(mode="json"),
                "independent_dialogue_grounding_plan": (
                    None
                    if dialogue_grounding_plan is None
                    else dialogue_grounding_plan.model_dump(mode="json")
                ),
                "current_head": fixed_head,
                "candidates": head_rows,
            },
            parser=FocusedHead3Selection,
        )
        if not isinstance(head_value, FocusedHead3Selection):
            return original_slate, head_audits, False, False, 0, None, True
        assessment_ids = {assessment.item_id for assessment in head_value.assessments}
        if set(head_value.item_ids) != set(fixed_head) or assessment_ids != set(fixed_head):
            return original_slate, head_audits, False, False, 0, None, True

        # A three-way LLM permutation is useful evidence but is not, by itself,
        # permission to disturb a strong existing head.  Require a decisive
        # public-evidence grade transition.  This makes the focused pass a
        # conservative calibrator instead of a second unconstrained reranker.
        grades = {
            assessment.item_id: assessment.public_evidence_grade
            for assessment in head_value.assessments
        }
        proposed_head = list(head_value.item_ids)
        current_head_id = fixed_head[0]
        challenger_id = proposed_head[0]
        calibrated_head = list(fixed_head)
        if (
            challenger_id == current_head_id
            or (
                grades[challenger_id] == 2
                and grades[current_head_id] < 2
            )
        ):
            calibrated_head = proposed_head

        calibrated_head, public_head_guard = self._public_deterministic_head_guard(
            current_head=calibrated_head,
            candidate_rows=head_rows,
            memory=memory,
            proposal=proposal,
        )

        reservoir_ids = self._deep_tail_reservoir_ids(
            current_slate=original_slate,
            provisional_ids=provisional_ids,
            pattern_result=pattern_result,
            dialogue_result=dialogue_result,
            direct_result=direct_result,
        )
        tail_rows = self._public_candidate_evidence(
            candidate_ids=reservoir_ids,
            current_slate=original_slate,
            provisional_ids=provisional_ids,
            proposal=proposal,
            grounding_plan=grounding_plan,
            pattern_result=pattern_result,
            dialogue_result=dialogue_result,
            direct_result=direct_result,
            fused_items=fused_items,
        )
        visible_tail_ids = {str(row["item_id"]) for row in tail_rows}
        current_tail = original_slate[3:]
        if not set(current_tail).issubset(visible_tail_ids):
            return original_slate, head_audits, False, False, 0, None, True
        tail_value, tail_audits = self._request(
            stage="skill_tail_coverage_verify",
            system_prompt=_DEEP_TAIL_COVERAGE_PROMPT,
            payload={
                "task": "rank_public_positions_4_through_10_only",
                "domain": memory.domain,
                "public_dialogue": memory.serialized_dialogue,
                "latest_user_utterance": memory.latest_user_utterance,
                "public_dsus_retrieval_frame": public_frame,
                "public_grounding_plan": grounding_plan.model_dump(mode="json"),
                "independent_dialogue_grounding_plan": (
                    None
                    if dialogue_grounding_plan is None
                    else dialogue_grounding_plan.model_dump(mode="json")
                ),
                "fixed_head": calibrated_head,
                "current_tail": current_tail,
                "maximum_new_tail_ids": 3,
                "candidates": tail_rows,
            },
            parser=FocusedTailSelection,
        )
        audits = [*head_audits, *tail_audits]
        if not isinstance(tail_value, FocusedTailSelection):
            return original_slate, audits, False, False, 0, None, True
        calibrated_tail = list(tail_value.item_ids)
        new_item_count = sum(item_id not in current_tail for item_id in calibrated_tail)
        valid_tail = (
            set(calibrated_tail).issubset(visible_tail_ids)
            and set(calibrated_tail).isdisjoint(calibrated_head)
            and new_item_count <= 3
        )
        calibrated_slate = [*calibrated_head, *calibrated_tail]
        if (
            not valid_tail
            or len(calibrated_slate) != 10
            or len(set(calibrated_slate)) != 10
        ):
            return original_slate, audits, False, False, 0, None, True
        return (
            calibrated_slate,
            audits,
            True,
            True,
            new_item_count,
            public_head_guard,
            False,
        )

    @staticmethod
    def _public_leading_utterance(item: CatalogRetrievalItem) -> str:
        evidence = list(
            dict.fromkeys([*item.matched_must_terms, *item.matched_preferred_terms])
        )[:3]
        if evidence:
            evidence_text = ", ".join(evidence)
            return (
                f"{item.title} is my leading recommendation because its public "
                f"catalog evidence best matches {evidence_text} after checking the "
                "strongest alternatives."
            )
        return (
            f"{item.title} is my leading recommendation after comparing the "
            "strongest catalog-grounded alternatives against your stated need."
        )

    def _skill_payload(
        self,
        *,
        proposal: SkillActionProposal,
        memory: PublicDialogueMemory,
    ) -> dict[str, object]:
        semantic = self.library.skill(proposal.semantic_skill_id)
        interaction = self.library.skill(proposal.interaction_skill_id)
        return {
            "task": "execute_selected_thought_pattern_sop",
            "domain": memory.domain,
            "public_dialogue": memory.serialized_dialogue,
            "public_dsus_state": proposal.state_hypothesis.model_dump(mode="json"),
            "selected_operation": proposal.action.action_type.value,
            "semantic_skill": {
                "name": semantic.name,
                "task_description": semantic.task_description,
                "solution_description": semantic.solution_description,
                "thought_template": [step.model_dump(mode="json") for step in semantic.thought_template],
                "insights": [insight.model_dump(mode="json") for insight in semantic.insights],
            },
            "interaction_skill": {
                "name": interaction.name,
                "task_description": interaction.task_description,
                "solution_description": interaction.solution_description,
                "thought_template": [step.model_dump(mode="json") for step in interaction.thought_template],
                "insights": [insight.model_dump(mode="json") for insight in interaction.insights],
            },
            "compiler_search_spec": (
                None if proposal.search_spec is None else proposal.search_spec.model_dump(mode="json")
            ),
        }

    def execute(
        self,
        *,
        proposal: SkillActionProposal,
        memory: PublicDialogueMemory,
        retriever: LexicalCatalogRetriever,
    ) -> SkillExecutionResult:
        if proposal.action.action_type == BenchmarkActionType.ASK:
            question_value, question_audits = self._request(
                stage="skill_question",
                system_prompt=_QUESTION_PROMPT,
                payload=self._skill_payload(proposal=proposal, memory=memory),
                parser=SkillQuestionPlan,
            )
            if (
                not isinstance(question_value, SkillQuestionPlan)
                or question_value.question_target in memory.asked_question_targets
            ):
                return SkillExecutionResult(
                    action=proposal.action,
                    search_spec=proposal.search_spec,
                    retrieval=proposal.retrieval,
                    llm_audits=question_audits,
                    used_fallback=True,
                    executor_id=self.executor_id,
                )
            # Frozen benchmark sessions materialize the first user event before
            # committing ``initial_user_utterance``.  Use the latest public
            # utterance/retrieval text as a causal fallback so the visible
            # opening choice is not lost at the executor boundary.
            visible_text = (
                memory.initial_user_utterance
                or memory.latest_user_utterance
                or memory.retrieval_text
            )
            visible_axis = visible_choice_axis(visible_text)
            if visible_axis:
                # The DSUS state already exposes this unresolved public choice.
                # Preserve it through Skill execution instead of letting the
                # realization model invent a different clarification axis.  The
                # public transition guard normally makes this opening-only;
                # keeping the override unconditional also protects against a
                # stale action counter in a replayed frozen benchmark trace.
                question_value = _question_for_visible_axis(visible_axis)
            return SkillExecutionResult(
                action=BenchmarkSystemAction(
                    action_type=BenchmarkActionType.ASK,
                    question_target=question_value.question_target,
                    utterance=question_value.utterance,
                ),
                search_spec=proposal.search_spec,
                retrieval=proposal.retrieval,
                llm_audits=question_audits,
                question_plan=question_value,
                executor_id=self.executor_id,
            )

        if proposal.action.action_type != BenchmarkActionType.RECOMMEND:
            return SkillExecutionResult(
                action=proposal.action,
                search_spec=proposal.search_spec,
                retrieval=proposal.retrieval,
                executor_id=self.executor_id,
            )

        base_payload = self._skill_payload(proposal=proposal, memory=memory)
        grounding_value, grounding_audits = self._request(
            stage="skill_ground",
            system_prompt=_GROUND_PROMPT,
            payload=base_payload,
            parser=SkillGroundingPlan,
        )
        grounding_used_fallback = False
        if not isinstance(grounding_value, SkillGroundingPlan):
            if not self.enable_dialogue_view:
                return SkillExecutionResult(
                    action=proposal.action,
                    search_spec=proposal.search_spec,
                    retrieval=proposal.retrieval,
                    llm_audits=grounding_audits,
                    used_fallback=True,
                    executor_id=self.executor_id,
                )
            grounding_value = self._deterministic_public_grounding(
                proposal=proposal,
                memory=memory,
            )
            grounding_used_fallback = True
        if self.enable_dialogue_view:
            state_anchor = self._deterministic_public_grounding(
                proposal=proposal,
                memory=memory,
            )
            grounding_value = self._merge_public_grounding_plans(
                primary=grounding_value,
                state_anchor=state_anchor,
            )
            grounding_value = self._identity_safe_grounding(grounding_value)
        dialogue_value: SkillGroundingPlan | None = None
        dialogue_audits: list[LLMCompletionAudit] = []
        if self.enable_dialogue_view:
            raw_dialogue_value, dialogue_audits = self._request(
                stage="dialogue_ground",
                system_prompt=_DIALOGUE_GROUND_PROMPT,
                payload={
                    "task": "compile_independent_public_dialogue_catalog_view",
                    "domain": memory.domain,
                    "public_dialogue": memory.serialized_dialogue,
                    "latest_user_utterance": memory.latest_user_utterance,
                },
                parser=SkillGroundingPlan,
            )
            if isinstance(raw_dialogue_value, SkillGroundingPlan):
                dialogue_value = self._identity_safe_grounding(raw_dialogue_value)

        # A visible correction can make an item from the first slate newly
        # acceptable. Re-rank it under the refined state instead of banning
        # every previously displayed item. Other phases retain the ordinary
        # novelty constraint.
        retain_for_repair = proposal.state_hypothesis.recommendation_readiness == (
            RecommendationReadiness.REPAIR_RECOMMEND
        ) or memory.last_system_action_type == BenchmarkActionType.RECOMMEND
        excluded_item_ids = [] if retain_for_repair else memory.previously_recommended_item_ids
        refined_spec = SearchSpec(
            query=f"{memory.retrieval_text} {grounding_value.query_focus}",
            target_type=grounding_value.target_type,
            must_terms=grounding_value.must_terms,
            preferred_terms=grounding_value.preferred_terms,
            avoid_terms=grounding_value.avoid_terms,
            excluded_item_ids=excluded_item_ids,
            candidate_k=100,
            ranking_k=self.shortlist_k,
        )
        pattern_retrieval = retriever.search(refined_spec)
        identity_anchor_applied = False
        if self.enable_dialogue_view and self._needs_public_identity_anchor(
            memory=memory,
            grounding_plan=grounding_value,
            pattern_result=pattern_retrieval,
        ):
            identity_anchor_applied = True
            grounding_value = grounding_value.model_copy(
                update={
                    "must_terms": list(
                        dict.fromkeys([grounding_value.target_type, *grounding_value.must_terms])
                    )[:10]
                },
                deep=True,
            )
            refined_spec = refined_spec.model_copy(
                update={"must_terms": grounding_value.must_terms},
                deep=True,
            )
            pattern_retrieval = retriever.search(refined_spec)
        dialogue_retrieval = None
        if dialogue_value is not None:
            dialogue_spec = SearchSpec(
                query=f"{memory.retrieval_text} {dialogue_value.query_focus}",
                target_type=dialogue_value.target_type,
                must_terms=dialogue_value.must_terms,
                preferred_terms=dialogue_value.preferred_terms,
                avoid_terms=dialogue_value.avoid_terms,
                excluded_item_ids=excluded_item_ids,
                candidate_k=100,
                ranking_k=self.shortlist_k,
            )
            dialogue_retrieval = retriever.search(dialogue_spec)
        direct_spec = SearchSpec(
            query=memory.retrieval_text,
            excluded_item_ids=excluded_item_ids,
            candidate_k=100,
            ranking_k=self.shortlist_k,
        )
        direct_retrieval = retriever.search(direct_spec)
        retrieval_views = (
            (pattern_retrieval, dialogue_retrieval, direct_retrieval)
            if dialogue_retrieval is not None
            else (pattern_retrieval, direct_retrieval)
        )
        fused_items = self._fuse_candidates(retrieval_views, limit=self.shortlist_k)
        if not fused_items:
            return SkillExecutionResult(
                action=proposal.action,
                search_spec=proposal.search_spec,
                retrieval=proposal.retrieval,
                llm_audits=[*grounding_audits, *dialogue_audits],
                used_fallback=True,
                grounding_plan=grounding_value,
                executor_id=self.executor_id,
            )

        fused_retrieval = CatalogRetrievalResult(
            retriever_id=pattern_retrieval.retriever_id,
            domain=pattern_retrieval.domain,
            pool_id=pattern_retrieval.pool_id,
            search_spec=refined_spec,
            results=fused_items,
        )
        selection_value, selection_audits = self._request(
            stage="skill_candidate_select",
            system_prompt=_PATTERN_CANDIDATE_SELECTION_PROMPT,
            payload=self._selection_payload(
                memory=memory,
                result=fused_retrieval,
                grounding_plan=grounding_value,
                dialogue_grounding_plan=dialogue_value,
            ),
            parser=DirectCandidateSelection,
        )
        audits = [*grounding_audits, *dialogue_audits, *selection_audits]
        retrieved_ids = {item.item_id for item in fused_items}
        if isinstance(selection_value, DirectCandidateSelection):
            selected_ids = [item_id for item_id in selection_value.item_ids if item_id in retrieved_ids]
            utterance = selection_value.utterance
        else:
            selected_ids = []
            utterance = ""
        selection_used_fallback = not selected_ids
        used_fallback = grounding_used_fallback or selection_used_fallback
        if selection_used_fallback:
            selected_ids = [item.item_id for item in fused_items[: self.recommendation_k]]
            utterance = (
                f"I found {len(selected_ids)} catalog-grounded options after applying the selected "
                f"recommendation strategy. My leading option is {fused_items[0].title}."
            )
        provisional_ids = list(selected_ids)
        # A Thought Pattern is a residual procedural guide, not authority to
        # override the user's literal product identity or latest public reply.
        # Independently verify the specialized ranking against a bounded union
        # of all three public retrieval views before it becomes an action.
        if self.enable_dialogue_view:
            verification_items = self._verification_candidates(
                provisional_ids=selected_ids,
                pattern_result=pattern_retrieval,
                dialogue_result=dialogue_retrieval,
                direct_result=direct_retrieval,
                fused_items=fused_items,
            )
            verification_retrieval = CatalogRetrievalResult(
                retriever_id=pattern_retrieval.retriever_id,
                domain=pattern_retrieval.domain,
                pool_id=pattern_retrieval.pool_id,
                search_spec=refined_spec,
                results=verification_items,
            )
            verification_payload = self._selection_payload(
                memory=memory,
                result=verification_retrieval,
                grounding_plan=grounding_value,
                dialogue_grounding_plan=dialogue_value,
            )
            verification_payload["provisional_pattern_ranking"] = selected_ids
            verification_value, verification_audits = self._request(
                stage="skill_candidate_verify",
                system_prompt=_PUBLIC_SEMANTIC_VERIFIER_PROMPT,
                payload=verification_payload,
                parser=DirectCandidateSelection,
            )
            audits.extend(verification_audits)
            verification_ids = {item.item_id for item in verification_items}
            if isinstance(verification_value, DirectCandidateSelection):
                verified_ids = [
                    item_id
                    for item_id in verification_value.item_ids
                    if item_id in verification_ids
                ]
                if verified_ids:
                    selected_ids = verified_ids
                    utterance = verification_value.utterance
        focused_head3_applied = False
        focused_tail_applied = False
        focused_tail_new_item_count = 0
        public_head_guard: PublicHeadGuardTrace | None = None
        if self.enable_dialogue_view:
            verifier_head_id = selected_ids[0] if selected_ids else None
            selected_ids = self._complete_public_multiview_slate(
                selected_ids=selected_ids,
                pattern_result=pattern_retrieval,
                dialogue_result=dialogue_retrieval,
                direct_result=direct_retrieval,
                fused_items=fused_items,
                limit=self.recommendation_k,
            )
            selected_ids = self._evidence_consistent_existing_slate(
                selected_ids=selected_ids,
                pattern_result=pattern_retrieval,
                dialogue_result=dialogue_retrieval,
                direct_result=direct_retrieval,
                fused_items=fused_items,
                target_type=grounding_value.target_type,
                domain=memory.domain,
                limit=self.recommendation_k,
            )
            (
                selected_ids,
                calibration_audits,
                focused_head3_applied,
                focused_tail_applied,
                focused_tail_new_item_count,
                public_head_guard,
                calibration_failed,
            ) = self._focused_semantic_slate_calibration(
                current_slate=selected_ids,
                provisional_ids=provisional_ids,
                proposal=proposal,
                memory=memory,
                grounding_plan=grounding_value,
                dialogue_grounding_plan=dialogue_value,
                pattern_result=pattern_retrieval,
                dialogue_result=dialogue_retrieval,
                direct_result=direct_retrieval,
                fused_items=fused_items,
            )
            audits.extend(calibration_audits)
            used_fallback = used_fallback or calibration_failed
            if selected_ids and selected_ids[0] != verifier_head_id:
                public_item = next(
                    item
                    for item in [
                        *fused_items,
                        *pattern_retrieval.results,
                        *([] if dialogue_retrieval is None else dialogue_retrieval.results),
                        *direct_retrieval.results,
                    ]
                    if item.item_id == selected_ids[0]
                )
                utterance = self._public_leading_utterance(public_item)
        else:
            # The Legacy baseline must preserve its original single-view,
            # LLM-selected slate instead of inheriting TAIRAS's tri-view
            # completion or verifier.
            selected_ids = selected_ids[: self.recommendation_k]
        declared_retrieval = self._declare_final_public_slate(
            fused_retrieval=fused_retrieval,
            selected_ids=selected_ids[: self.recommendation_k],
            pattern_result=pattern_retrieval,
            dialogue_result=dialogue_retrieval,
            direct_result=direct_retrieval,
        )
        return SkillExecutionResult(
            action=BenchmarkSystemAction(
                action_type=BenchmarkActionType.RECOMMEND,
                item_ids=selected_ids[: self.recommendation_k],
                utterance=utterance,
            ),
            search_spec=refined_spec,
            retrieval=declared_retrieval,
            llm_audits=audits,
            used_fallback=used_fallback,
            grounding_plan=grounding_value,
            candidate_fusion=(
                "public_v18_head_guard_deep_tail_calibration_v10"
                if focused_head3_applied and focused_tail_applied
                else (
                    "public_verified_pattern_dialogue_lexical_tri_view_v8_state_anchored_consensus"
                    if dialogue_retrieval is not None
                    else "pattern_first_interleave_direct_v1"
                )
            ),
            pattern_candidate_count=len(pattern_retrieval.results),
            dialogue_candidate_count=(
                0 if dialogue_retrieval is None else len(dialogue_retrieval.results)
            ),
            direct_candidate_count=len(direct_retrieval.results),
            fused_candidate_count=len(fused_items),
            identity_anchor_applied=identity_anchor_applied,
            focused_head3_applied=focused_head3_applied,
            focused_tail_applied=focused_tail_applied,
            focused_tail_new_item_count=focused_tail_new_item_count,
            public_head_guard=public_head_guard,
            executor_id=self.executor_id,
        )
