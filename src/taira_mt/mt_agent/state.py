"""Decision-Oriented User State (DOUS) for the multi-turn Agent.

DSUS identifiers below remain serialization/checkpoint compatibility names.

DSUS is deliberately a perception layer, not an action policy.  It turns
observable dialogue into a compact state that tells SCTP what kind of
complex-intent plan is applicable and tells CIV which legal plan operation is
worth considering.  A deterministic estimator provides a runnable local
control; an OpenAI-compatible estimator can replace it without changing the
state contract.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from enum import Enum
from typing import Any, Protocol

from pydantic import Field, field_validator, model_validator

from taira_mt.benchmark.schemas import BenchmarkActionType
from taira_mt.models import StrictModel
from taira_mt.mt_agent.dsus_prior import DSUSInitialStatePrior, DSUSPriorPrediction
from taira_mt.mt_agent.memory import PublicDialogueMemory
from taira_mt.mt_agent.prototypes import StatePrototypeBank
from taira_mt.mt_agent.retrieval import CatalogRetrievalItem, LexicalCatalogRetriever, SearchSpec
from taira_mt.mt_agent.runner import LLMCompletionAudit
from taira_mt.provenance import content_sha256
from taira_mt.providers.openai_compatible import JsonCompletion, JsonCompletionTransport

DSUS_VERSION = "decision_sufficient_user_state_v1"


class DialoguePhase(str, Enum):
    INITIAL = "initial"
    ELICITATION = "elicitation"
    RECOMMENDATION = "recommendation"
    FEEDBACK = "feedback"
    REPAIR = "repair"


class DisclosureStatus(str, Enum):
    SUFFICIENT = "sufficient"
    MISSING_DECISIVE = "missing_decisive"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"
    REVISED = "revised"


class FeedbackSignal(str, Enum):
    NONE = "none"
    CLARIFICATION = "clarification"
    REVISION = "revision"
    REJECTION = "rejection"
    ACCEPTANCE = "acceptance"


class RecommendationReadiness(str, Enum):
    RECOMMEND_NOW = "recommend_now"
    ASK_DECISIVE = "ask_decisive"
    PROBE_RECOMMEND = "probe_recommend"
    REPAIR_RECOMMEND = "repair_recommend"
    SAFE_STOP = "safe_stop"


class DSUSDecisionPosterior(StrictModel):
    """Normalized public uncertainty over the next interaction operation."""

    recommend_now: float = Field(ge=0.0, le=1.0)
    ask_decisive: float = Field(ge=0.0, le=1.0)
    probe_recommend: float = Field(ge=0.0, le=1.0)
    repair_recommend: float = Field(ge=0.0, le=1.0)
    safe_stop: float = Field(ge=0.0, le=1.0)

    @classmethod
    def from_raw(
        cls,
        raw: object,
        *,
        selected: RecommendationReadiness,
        selected_probability: float | None = None,
    ) -> DSUSDecisionPosterior:
        if not isinstance(raw, Mapping):
            confidence = min(0.95, max(0.55, float(selected_probability or 0.75)))
            values = {item.value: 0.0 for item in RecommendationReadiness}
            values[selected.value] = confidence
            alternative = (
                RecommendationReadiness.RECOMMEND_NOW
                if selected == RecommendationReadiness.ASK_DECISIVE
                else RecommendationReadiness.ASK_DECISIVE
            )
            values[alternative.value] = 1.0 - confidence
            return cls(**values)
        values: dict[str, float] = {}
        for item in RecommendationReadiness:
            try:
                value = float(raw.get(item.value, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            values[item.value] = min(1.0, max(0.0, value))
        total = sum(values.values())
        if total <= 0.0:
            values = {item.value: float(item == selected) for item in RecommendationReadiness}
        else:
            values = {key: value / total for key, value in values.items()}
        return cls(**values)

    def probability(self, readiness: RecommendationReadiness) -> float:
        return float(getattr(self, readiness.value))

    def public_action_probabilities(self) -> tuple[float, float, float]:
        recommend = self.recommend_now + self.probe_recommend + self.repair_recommend
        return self.ask_decisive, recommend, self.safe_stop


class ComplexIntentShape(str, Enum):
    DIRECT_ATTRIBUTE = "direct_attribute"
    OCCASION_TO_ATTRIBUTE = "occasion_to_attribute"
    FIT_OR_AUDIENCE = "fit_or_audience"
    MULTI_OCCASION = "multi_occasion"
    MULTI_TARGET = "multi_target"
    PAIRING_OR_MATCHING = "pairing_or_matching"
    BUNDLE = "bundle"
    TAXONOMY_AMBIGUITY = "taxonomy_ambiguity"
    FEEDBACK_REPAIR = "feedback_repair"


class RetrievalFrame(StrictModel):
    """Public scenario-to-catalog conversion before a Skill compiles a search."""

    target_type: str = Field(default="", max_length=160)
    must_terms: list[str] = Field(default_factory=list, max_length=12)
    preferred_terms: list[str] = Field(default_factory=list, max_length=12)
    avoid_terms: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("target_type", mode="before")
    @classmethod
    def clean_target_type(cls, value: object) -> str:
        return " ".join(str(value or "").replace("_", " ").split())

    @field_validator("must_terms", "preferred_terms", "avoid_terms", mode="before")
    @classmethod
    def clean_terms(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("retrieval frame term fields must be lists")
        output: list[str] = []
        seen: set[str] = set()
        for raw in value:
            term = " ".join(str(raw or "").replace("_", " ").split())
            if term and term.casefold() not in seen:
                seen.add(term.casefold())
                output.append(term)
        return output


class DecisionSufficientUserState(StrictModel):
    """Compact public state passed to SCTP and CIV, never a hidden goal proxy."""

    schema_version: str = DSUS_VERSION
    estimator_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    dialogue_phase: DialoguePhase
    disclosure_status: DisclosureStatus
    intent_shapes: list[ComplexIntentShape] = Field(min_length=1, max_length=4)
    active_subgoal: str = Field(min_length=1, max_length=240)
    unresolved_axes: list[str] = Field(default_factory=list, max_length=5)
    feedback_signal: FeedbackSignal
    recommendation_readiness: RecommendationReadiness
    retrieval_frame: RetrievalFrame
    confidence: float = Field(ge=0.0, le=1.0)
    public_evidence_summary: str = Field(min_length=1, max_length=800)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: object) -> float:
        if isinstance(value, str):
            qualitative = {"low": 0.42, "medium": 0.64, "high": 0.84}
            normalized = value.strip().casefold()
            if normalized in qualitative:
                return qualitative[normalized]
        return float(value)

    @field_validator("unresolved_axes", mode="before")
    @classmethod
    def clean_axes(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("unresolved_axes must be a list")
        return [" ".join(str(item or "").split()) for item in value if str(item or "").strip()]

    @model_validator(mode="after")
    def validate_decision_state(self) -> DecisionSufficientUserState:
        if self.recommendation_readiness == RecommendationReadiness.ASK_DECISIVE and not self.unresolved_axes:
            raise ValueError("ask_decisive state requires at least one unresolved public axis")
        if self.recommendation_readiness == RecommendationReadiness.REPAIR_RECOMMEND:
            if self.feedback_signal not in {FeedbackSignal.REVISION, FeedbackSignal.REJECTION}:
                raise ValueError("repair_recommend requires visible revision or rejection feedback")
        return self

    @property
    def state_key(self) -> str:
        return ":".join(
            (
                self.dialogue_phase.value,
                self.disclosure_status.value,
                self.recommendation_readiness.value,
            )
        )


class DSUSInference(StrictModel):
    state: DecisionSufficientUserState
    llm_audits: list[LLMCompletionAudit] = Field(default_factory=list)
    used_fallback: bool = False
    initial_state_prior: DSUSPriorPrediction | None = None
    pre_state_catalog_evidence: list[CatalogRetrievalItem] = Field(default_factory=list)
    decision_posterior: DSUSDecisionPosterior | None = None


class DSUSEstimator(Protocol):
    estimator_id: str

    def infer(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> DSUSInference: ...


_TARGET_CUES = (
    # Longer, domain-specific noun phrases come first conceptually; the
    # inference function selects the longest matching phrase.  These cues are
    # public vocabulary only and are used as a conservative retrieval/readiness
    # prior when a learned state omits the product noun.
    "hand and cuticle salve",
    "eau de toilette spray",
    "face moisturizer",
    "biotin supplement",
    "deodorant spray",
    "athletic dance shoes",
    "baseball cap",
    "live performance recording",
    "music collection",
    "music album",
    "music track",
    "t-shirt",
    "tee shirt",
    "shirt",
    "jeans",
    "dress",
    "skirt",
    "shorts",
    "sweater",
    "scarf",
    "shoes",
    "socks",
    "hat",
    "bag",
    "pumps",
    "loafers",
    "boots",
    "earmuffs",
    "lingerie",
    "bracelet",
    "lotion",
    "shampoo",
    "serum",
    "lipstick",
    "perfume",
    "album",
    "cd",
    "vinyl",
    "music",
    "moisturizer",
    "supplement",
    "deodorant",
    "salve",
    "fragrance",
    "mules",
    "belt",
    "cap",
    "track",
)
_FIT_OR_AUDIENCE_CUES = (
    "men's",
    "women's",
    "girls",
    "boys",
    "baby",
    "newborn",
    "big & tall",
    "plus-size",
    "plus size",
    "larger body",
    "post-mastectomy",
)
_OCCASION_CUES = (
    "for ",
    "outdoor",
    "travel",
    "party",
    "wedding",
    "casual",
    "formal",
    "beach",
    "work",
    "sport",
    "concert",
    "road trip",
    "gathering",
)
_AMBIGUITY_CUES = (
    "men and women",
    "women and children",
    "girls'",
    "both",
    "either",
    "casual and dress",
    "best suited",
    "what's the best",
)
_CONFLICT_CUES = (
    "while also",
    "but also",
    "formal event and",
    "formal gatherings and casual",
    "comfort during travel",
    "post-mastectomy",
)
_REVISION_CUES = (
    "first option",
    "the first",
    "but i need",
    "prioritize",
    "instead",
    "does not meet",
    "do not meet",
    "none of those",
)
_FUNCTIONAL_CUES = (
    "comfortable",
    "comfort",
    "breathable",
    "wireless",
    "lace",
    "durable",
    "functional",
    "support",
    "anti-itch",
    "soothing",
    "layered",
    "warmth",
    "flexibility",
)


def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in text for cue in cues)


def _unique_terms(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        term = " ".join(value.split())
        if term and term.casefold() not in seen:
            seen.add(term.casefold())
            output.append(term)
    return output


def _infer_target_type(text: str) -> str:
    lowered = text.casefold()
    matches = [cue for cue in _TARGET_CUES if re.search(rf"\b{re.escape(cue)}\b", lowered)]
    return max(matches, key=len) if matches else "product"


_VISIBLE_UNCERTAINTY_CUES = (
    "not sure",
    "unsure",
    "wondering",
    "whether",
    "open to",
    "consider",
    "haven't decided",
    "have not decided",
    "haven't chosen",
    "have not chosen",
    "still deciding",
    "can't decide",
    "cannot decide",
    "haven't settled",
    "have not settled",
    "trying to determine",
    "trying to choose",
    "trying to decide",
    "deciding between",
    "trying to balance",
    "struggling to choose",
    "struggling to decide",
    "conflicted between",
    "undecided between",
    "weighing the need",
    "weighing the importance",
    "weigh the need",
    "weigh the importance",
    "figure out if",
    "unclear if",
    "clarify if",
    "without sacrificing",
    "haven't determined",
    "have not determined",
    "haven't specified",
    "have not specified",
    "need to decide",
    "determine if",
    "debating",
    "torn between",
    "equally important",
    "equally",
    "perhaps",
)

_VISIBLE_PROBE_CUES = (
    "one concrete option",
    "one option to begin",
    "one option to start",
    "particular model",
    "one specific album",
    "one specific option",
    "one specific pair",
    "one initial option",
    "to look at initially",
    "concrete suggestion first",
    "one concrete suggestion",
    "one suggestion",
    "one recommendation",
    "show me one",
)

_VISIBLE_PROBE_PATTERNS = (
    # A concrete-first request is a recommendation contract, even when the
    # user intends to refine the need after seeing the first slate.
    r"\b(?:one|a)\s+(?:specific|concrete)\s+"
    r"(?:option|suggestion|recommendation|pair|album|track|item|product|example)\b",
    r"\b(?:specific|concrete)\s+"
    r"(?:option|suggestion|recommendation|pair|album|track|item|product|example)\b"
    r".{0,80}\b(?:first|initially|before\s+(?:deciding|refining|moving))\b",
    r"\bwhich\s+specific\b",
    r"\bplease\s+recommend\s+one\b",
    r"\bshow\s+me\s+(?:one|a)\s+(?:specific|concrete)?\s*"
    r"(?:option|suggestion|recommendation|pair|album|track|item|product|example)?\b",
    r"\bone\b.{0,70}\b"
    r"(?:option|suggestion|recommendation|pair|album|track|item|product|example)\b",
    r"\bwhat\s+is\s+one\s+good\b",
)


def visible_choice_axis(text: str) -> str | None:
    """Extract an explicitly unresolved public choice without inferring its answer."""

    normalized = " ".join(text.split())
    lowered = normalized.casefold()
    balance_match = re.search(
        r"\bbalanc(?:e|es|ing)\s+(.+?)\s+with\s+(.+?)(?:,\s*(?:as|since|because)\b|[?.]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    clarify_type_match = re.search(
        r"\b(?:recommend|suggest)\s+(?:an?\s+)?(.+?),\s*or\s+clarify\s+if\s+.*?"
        r"\b(?:refer|refers)\s+to\s+(.+?)(?:[?.]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    missing_choice_match = re.search(
        r"\bhave(?:n't|\s+not)\s+(?:selected|specified|settled\s+on|determined)\s+"
        r"(?!if\b)(?:a\s+|the\s+)?(?:specific\s+|preferred\s+)?(.+?)\s+yet(?:[?.]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    competing_use_match = re.search(
        r"\bworks?\s+for\s+(.+?)\s+but\s+(?:is\s+)?still\s+(.+?)(?:[?.]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    against_match = re.search(
        r"\bweigh(?:ing)?\s+(?:the\s+(?:need|importance)\s+(?:for|of)\s+)?"
        r"(.+?)\s+against\s+(.+?)(?:[?.]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    just_as_match = re.search(
        r"\bjust\s+as\s+(.+?)\s+as\s+(?:they|it)\s+(?:are|is)\s+"
        r"(.+?)(?:,\s*without\b|[?.]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    blend_match = re.search(
        r"\b(?:blend(?:s|ing)?|combin(?:e|es|ing))\s+(.+?)\s+with\s+(.+?)(?:[?.]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    but_also_match = re.search(
        r"\b(.+?)\s+but\s+(?:(?:is|are)\s+)?also\s+(.+?)(?:[?.]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    offers_with_match = re.search(
        r"\boffers?\s+(?:the\s+)?(.+?)\s+with\s+(?:the\s+)?(.+?)(?:[?.]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    either_or_match = re.search(
        r"\beither\s+(.+?)\s+or\s+(.+?)(?:[?.]|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    # ``balances A with B`` is itself an explicit public trade-off even when
    # the sentence does not also contain a hedge such as "not sure".  Keep
    # that structured cue narrow so an ordinary request for a "balanced"
    # product is not turned into a clarification question.
    if (
        not _contains_any(lowered, _VISIBLE_UNCERTAINTY_CUES)
        and balance_match is None
        and clarify_type_match is None
        and missing_choice_match is None
        and competing_use_match is None
        and against_match is None
        and just_as_match is None
        and blend_match is None
        and but_also_match is None
        and offers_with_match is None
        and either_or_match is None
    ):
        return None

    if balance_match:
        axis = f"{balance_match.group(1)} versus {balance_match.group(2)}"
    elif clarify_type_match:
        requested = clarify_type_match.group(1)
        requested = re.sub(
            r"^\d+(?:[.-]\d+)?[- ]?(?:ounce|oz)\s+",
            "",
            requested,
            flags=re.IGNORECASE,
        )
        requested = re.split(r"\s+for\s+", requested, maxsplit=1, flags=re.IGNORECASE)[0]
        axis = f"{requested} versus {clarify_type_match.group(2)}"
    elif competing_use_match:
        axis = f"{competing_use_match.group(1)} versus {competing_use_match.group(2)}"
    elif against_match:
        axis = f"{against_match.group(1)} versus {against_match.group(2)}"
    elif just_as_match:
        axis = f"{just_as_match.group(1)} versus {just_as_match.group(2)}"
    elif blend_match:
        axis = f"{blend_match.group(1)} versus {blend_match.group(2)}"
    elif but_also_match:
        left = re.split(r"\b(?:that|which)\s+", but_also_match.group(1), flags=re.IGNORECASE)[-1]
        axis = f"{left} versus {but_also_match.group(2)}"
    elif offers_with_match:
        axis = f"{offers_with_match.group(1)} versus {offers_with_match.group(2)}"
    elif either_or_match:
        axis = f"{either_or_match.group(1)} versus {either_or_match.group(2)}"
    elif missing_choice_match:
        missing_dimension = missing_choice_match.group(1).strip(" ,;:.")
        axis = (
            f"{missing_dimension} family"
            if missing_dimension.casefold() in {"shade", "color", "colour"}
            else missing_dimension
        )
    else:
        between_match = re.search(
            r"\bbetween\s+(.+?)(?:[?.]|$)",
            normalized,
            flags=re.IGNORECASE,
        )
        if between_match:
            axis = between_match.group(1)
        else:
            whether_match = re.search(
                r"\b(?:whether|if)\s+(.+?)(?:[?.]|$)",
                normalized,
                flags=re.IGNORECASE,
            )
            if whether_match:
                axis = whether_match.group(1)
            else:
                perhaps_match = re.search(
                    r"\bperhaps\s+(.+?\s+or\s+.+?)(?:[?.]|$)",
                    normalized,
                    flags=re.IGNORECASE,
                )
                if not perhaps_match:
                    return None
                axis = perhaps_match.group(1)

    axis = re.sub(
        r"^(?:i|we)\s+(?:should\s+)?(?:prefer|need|want|choose|prioritize)\s+",
        "",
        axis,
        flags=re.IGNORECASE,
    )
    axis = re.sub(
        r"^(?:to\s+)?(?:look\s+for|stick\s+(?:strictly\s+)?to)\s+",
        "",
        axis,
        flags=re.IGNORECASE,
    )
    axis = re.sub(r"^the\s+style\s+should\s+be\s+", "", axis, flags=re.IGNORECASE)
    axis = re.sub(r",?\s+(?:as|since)\s+both\b.*$", "", axis, flags=re.IGNORECASE)
    axis = axis.strip(" ,;:.?")
    return axis[:220] if axis else None


def prefers_initial_probe(text: str) -> bool:
    """Return whether the user visibly asks to react to one initial option."""

    lowered = text.casefold()
    return _contains_any(lowered, _VISIBLE_PROBE_CUES) or any(
        re.search(pattern, text, flags=re.IGNORECASE) is not None
        for pattern in _VISIBLE_PROBE_PATTERNS
    )


_EXPLICIT_DIRECT_REQUEST_CUES = (
    "please suggest",
    "suggest a",
    "suggest one",
    "can you recommend",
    "recommend a",
    "recommend one",
    "i'm after",
    "i’m after",
    "i am after",
    "i'm interested in",
    "i am interested in",
    "i need",
    "i'm looking",
    "i’m looking",
    "i am looking",
    "i'm in need",
    "i’m in need",
    "i am in need",
    "i want to find",
    "help me find",
    "help me choose",
    "can you help me choose",
    "do you have any suggestions",
    "do you have suggestions",
    "i'd like to find",
    "i’d like to find",
    "i would like to find",
    "what would you recommend",
    "can you point me to",
    "could you point me to",
    "could you guide me to",
    "can you guide me to",
    "can you help me pick",
    "could you help me pick",
    "what's the best",
    "what’s the best",
    "do you know any",
    "do you have a",
    "do you have an",
    "what should i go for",
    "i'm searching for",
    "i’m searching for",
    "i am searching for",
)

_EXPLICIT_DIRECT_REQUEST_PATTERNS = (
    r"\bi\s+want\s+(?:a|an|some|one)\b",
    r"\bwhat\b.{0,50}\bshould\s+i\s+go\s+for\b",
)


def explicit_direct_request(text: str) -> bool:
    """Return whether public wording explicitly requests a first recommendation.

    This is narrower than ``prefers_initial_probe``: it must not classify an
    ambiguous audience/trade-off request such as ``women and children`` as a
    direct recommendation merely because it starts with ``I'm after``.
    """

    lowered = text.casefold()
    if visible_choice_axis(text) is not None:
        return False
    if _contains_any(lowered, _VISIBLE_UNCERTAINTY_CUES):
        return False
    # Broad surface words such as ``both`` and ``best`` are not uncertainty
    # by themselves.  A user may explicitly ask for one product satisfying
    # two compatible requirements.  Structured public uncertainty has already
    # been rejected above; retain only explicit conflict constructions here.
    if _contains_any(lowered, (*_CONFLICT_CUES, "both", "either", "but still")):
        return False
    return _contains_any(lowered, _EXPLICIT_DIRECT_REQUEST_CUES) or any(
        re.search(pattern, text, flags=re.IGNORECASE) is not None
        for pattern in _EXPLICIT_DIRECT_REQUEST_PATTERNS
    )


class RuleDSUSEstimator:
    """Deterministic state estimator used for offline controls and safe fallback."""

    estimator_id = "rule_dsus_public_v1"

    def infer(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> DSUSInference:
        del retriever  # The rule fallback intentionally avoids a hidden retrieval-derived label.
        latest = memory.latest_user_utterance
        text = memory.retrieval_text.casefold()
        latest_lower = latest.casefold()
        target_type = _infer_target_type(text)
        # Word boundaries matter for audience identity: the substring
        # ``men's`` occurs inside ``women's`` and previously polluted the
        # retrieval frame with mutually exclusive hard terms.
        must_terms = [
            cue
            for cue in _FIT_OR_AUDIENCE_CUES
            if re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", text)
        ]
        must_terms.extend(cue for cue in _FUNCTIONAL_CUES if cue in text and cue in {"wireless", "lace", "anti-itch"})
        preferred_terms = [cue for cue in _FUNCTIONAL_CUES if cue in text and cue not in must_terms]
        shapes: list[ComplexIntentShape] = [ComplexIntentShape.DIRECT_ATTRIBUTE]
        if _contains_any(text, _OCCASION_CUES):
            shapes.append(ComplexIntentShape.OCCASION_TO_ATTRIBUTE)
        if _contains_any(text, _FIT_OR_AUDIENCE_CUES):
            shapes.append(ComplexIntentShape.FIT_OR_AUDIENCE)
        if _contains_any(text, (" and ", "multiple", "several", "bundle", "set")):
            shapes.append(ComplexIntentShape.MULTI_OCCASION)

        if memory.last_system_action_type == BenchmarkActionType.RECOMMEND and _contains_any(
            latest_lower, _REVISION_CUES
        ):
            state = DecisionSufficientUserState(
                estimator_id=self.estimator_id,
                domain=memory.domain,
                turn_index=memory.completed_action_count,
                dialogue_phase=DialoguePhase.REPAIR,
                disclosure_status=DisclosureStatus.REVISED,
                intent_shapes=[ComplexIntentShape.FEEDBACK_REPAIR, *shapes[:2]],
                active_subgoal="Repair the recommendation using the newly stated visible priority.",
                unresolved_axes=[],
                feedback_signal=FeedbackSignal.REVISION,
                recommendation_readiness=RecommendationReadiness.REPAIR_RECOMMEND,
                retrieval_frame=RetrievalFrame(
                    target_type=target_type,
                    must_terms=_unique_terms(must_terms),
                    preferred_terms=_unique_terms(preferred_terms),
                ),
                confidence=0.82,
                public_evidence_summary=(
                    "A prior recommendation received an explicit visible correction or priority update."
                ),
            )
            return DSUSInference(state=state)

        if memory.last_system_action_type == BenchmarkActionType.ASK:
            state = DecisionSufficientUserState(
                estimator_id=self.estimator_id,
                domain=memory.domain,
                turn_index=memory.completed_action_count,
                dialogue_phase=DialoguePhase.RECOMMENDATION,
                disclosure_status=DisclosureStatus.SUFFICIENT,
                intent_shapes=shapes[:4],
                active_subgoal="Use the newly supplied clarification to ground a recommendation.",
                unresolved_axes=[],
                feedback_signal=FeedbackSignal.CLARIFICATION,
                recommendation_readiness=RecommendationReadiness.RECOMMEND_NOW,
                retrieval_frame=RetrievalFrame(
                    target_type=target_type,
                    must_terms=_unique_terms(must_terms),
                    preferred_terms=_unique_terms(preferred_terms),
                ),
                confidence=0.74,
                public_evidence_summary="The user has answered a prior public clarification question.",
            )
            return DSUSInference(state=state)

        visible_axis = visible_choice_axis(memory.initial_user_utterance)
        if visible_axis:
            is_conflict = _contains_any(text, ("balance", "equally important", "prioritize"))
            disclosure = DisclosureStatus.CONFLICT if is_conflict else DisclosureStatus.AMBIGUOUS
            readiness = RecommendationReadiness.ASK_DECISIVE
            axis = [visible_axis]
            phase = DialoguePhase.ELICITATION
            summary = "The initial request explicitly leaves a public ranking choice unresolved."
            confidence = 0.9
        elif prefers_initial_probe(memory.initial_user_utterance):
            disclosure = DisclosureStatus.SUFFICIENT
            readiness = RecommendationReadiness.PROBE_RECOMMEND
            axis = []
            phase = DialoguePhase.RECOMMENDATION
            summary = "The user explicitly prefers to react to one concrete initial option."
            confidence = 0.9
        elif _contains_any(text, _CONFLICT_CUES):
            disclosure = DisclosureStatus.CONFLICT
            readiness = RecommendationReadiness.ASK_DECISIVE
            axis = ["which visible requirement should dominate the trade-off"]
            phase = DialoguePhase.ELICITATION
            summary = "The initial request combines potentially competing visible use requirements."
            confidence = 0.64
        elif _contains_any(text, _AMBIGUITY_CUES):
            disclosure = DisclosureStatus.AMBIGUOUS
            readiness = RecommendationReadiness.ASK_DECISIVE
            axis = ["the intended audience, interpretation, or use context"]
            phase = DialoguePhase.ELICITATION
            summary = "The initial request has a visibly ambiguous audience or interpretation."
            confidence = 0.62
        elif (
            target_type == "product"
            or _contains_any(text, _VISIBLE_UNCERTAINTY_CUES)
            or ("best" in text and len(must_terms) + len(preferred_terms) < 1)
        ):
            disclosure = DisclosureStatus.MISSING_DECISIVE
            readiness = RecommendationReadiness.ASK_DECISIVE
            axis = ["the single product property that should decide among plausible options"]
            phase = DialoguePhase.ELICITATION
            summary = "The initial request lacks a public product type or a decisive product property."
            confidence = 0.58
        else:
            disclosure = DisclosureStatus.SUFFICIENT
            readiness = RecommendationReadiness.RECOMMEND_NOW
            axis = []
            phase = DialoguePhase.RECOMMENDATION
            summary = "The initial request states a product target plus usable public scenario or attribute evidence."
            confidence = 0.66
        return DSUSInference(
            state=DecisionSufficientUserState(
                estimator_id=self.estimator_id,
                domain=memory.domain,
                turn_index=memory.completed_action_count,
                dialogue_phase=phase,
                disclosure_status=disclosure,
                intent_shapes=shapes[:4],
                active_subgoal="Convert the visible complex shopping scenario into catalog-grounded attributes.",
                unresolved_axes=axis,
                feedback_signal=FeedbackSignal.NONE,
                recommendation_readiness=readiness,
                retrieval_frame=RetrievalFrame(
                    target_type=target_type,
                    must_terms=_unique_terms(must_terms),
                    preferred_terms=_unique_terms(preferred_terms),
                ),
                confidence=confidence,
                public_evidence_summary=summary,
            )
        )


DSUS_LLM_CALIBRATION_VERSION = "dsus_llm_calibration_v5_public_uncertainty_evidence"

_DSUS_PROMPT = """You are the Decision-Sufficient User State (DSUS) module for a multi-turn shopping Agent.
Read only the supplied public dialogue and the Agent's own public action history. Do not infer a hidden target,
benchmark mechanism, evaluator rule, or private user state. Do not choose an item or formulate a system action.

Describe the decision-relevant public state: whether the current intent is sufficient, missing one decisive property,
ambiguous, conflicting, or revised after feedback; which complex-intent structures are present; and whether the
next state is ready to recommend, needs one decisive clarification, or needs a repair recommendation. Translate
visible scenario language into a compact public catalog retrieval frame.

Return only one JSON object with exactly these keys:
dialogue_phase, disclosure_status, intent_shapes, active_subgoal, unresolved_axes, feedback_signal,
recommendation_readiness, retrieval_frame, confidence, public_evidence_summary.
Valid intent_shapes are: direct_attribute, occasion_to_attribute, fit_or_audience, multi_occasion, multi_target,
pairing_or_matching, bundle, taxonomy_ambiguity, feedback_repair.
Valid dialogue_phase values are: initial, elicitation, recommendation, feedback, repair.
Valid disclosure_status values are: sufficient, missing_decisive, ambiguous, conflict, revised.
Valid feedback_signal values are: none, clarification, revision, rejection, acceptance.
Valid recommendation_readiness values are: recommend_now, ask_decisive, probe_recommend, repair_recommend, safe_stop.
retrieval_frame must have target_type, must_terms, preferred_terms, avoid_terms. Do not use private information.

Important calibration rules: infer uncertainty from public evidence rather than manufacturing a latent
preference. A coherent product request may remain actionable even when it uses subjective scenario language.
On the first turn, compare the value of one focused answer against the cost of a premature slate:
- use recommend_now when the request names a coherent product route and concrete ranking constraints such as
  audience, product type, occasion, functional use, form, material, size, price, artist/genre, or other
  catalog-groundable attributes;
- words such as comfortable, stylish, professional, nostalgic, cozy, sensitive, or versatile are not by
  themselves evidence of an unresolved choice. Do not invent alternatives such as cushioning versus
  breathability unless the dialogue or supplied catalog evidence actually makes that distinction public;
- use ask_decisive when the user publicly signals uncertainty, an undecided attribute, multiple interpretations,
  competing requirements, an unresolved audience/product type, or a choice whose answer would materially rerank
  products;
- when the user explicitly says two alternatives are still open, or that two competing goals both matter without
  a priority, use ask_decisive and name that already-visible axis;
- use conflict whenever two stated goals favor different product characteristics, even if both goals sound
  individually reasonable; ask which characteristic dominates;
- use probe_recommend only when showing a concrete option is itself more informative than asking a focused
  question, not merely because a recommendation is possible.
- when the user explicitly prefers to react to one concrete initial option before refining details, use
  probe_recommend; this is a visible interaction preference, not an inferred hidden product preference.
Do not ask about an optional detail that would not change the ranking. When asking is useful, unresolved_axes must
name a short concrete customer-facing distinction (for example "cushioning versus breathability"), never a generic
phrase such as "the deciding preference".

The optional source-disjoint initial-state prior is weak calibration evidence produced only from development
queries. It is not a benchmark label and may be wrong. Use it to reconsider an over-confident first-turn decision
when the public request contains a real unresolved trade-off; never copy it blindly and never mention it to users.
Retrieved development prototypes are stronger procedural demonstrations than the weak classifier prior. When a
prototype is semantically close, transfer its distinction and interaction route, but adapt the wording to the
current public query.
The optional public catalog preview comes from the same full-catalog retriever available to every compared Agent.
Use it only to judge whether the visible need maps to one coherent product route or several materially different
routes. A high lexical score is evidence of catalog grounding, not evidence about an unspoken private preference.

Development-only calibration examples (these are not evaluated cases):
- "men's sun hat ... practical, durable, outdoor activities" -> sufficient / recommend_now.
- "men's big & tall running sock ... exercise and fitness" -> missing_decisive / ask_decisive;
  unresolved axis: cushioning versus breathability.
- "a bodysuit ... family gatherings and playdates" -> ambiguous / ask_decisive;
  unresolved axis: presentable gathering wear versus ease of changing for play.
- "men's polo ... family gatherings and layering under blazers" -> sufficient / probe_recommend;
  a plausible initial option may expose the slim-versus-bulky layering priority.
- "women's running shoes ... outdoor adventures, casual outings, and warmth" -> conflict / ask_decisive;
  unresolved axis: lightweight running agility versus warmth and protection.
"""


def _strict_json_object(content: str) -> dict[str, Any]:
    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"DSUS JSON repeats key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(content, object_pairs_hook=no_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError("DSUS completion is not JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("DSUS completion must be a JSON object")
    return payload


def _llm_audit(
    *, attempt_index: int, completion: JsonCompletion | None, failure: Exception | None = None
) -> LLMCompletionAudit:
    return LLMCompletionAudit(
        stage="dsus",
        attempt_index=attempt_index,
        status="rejected" if failure is not None else "committed",
        requested_model=None if completion is None else completion.requested_model,
        returned_model=None if completion is None else completion.returned_model,
        finish_reason=None if completion is None else completion.finish_reason,
        raw_content_sha256=None if completion is None else content_sha256(completion.content),
        input_tokens=None if completion is None else completion.usage.input_tokens,
        output_tokens=None if completion is None else completion.usage.output_tokens,
        total_tokens=None if completion is None else completion.usage.total_tokens,
        failure_kind=None if failure is None else type(failure).__name__,
    )


class LLMDSUSEstimator:
    """Constrained JSON DSUS parser with an explicit rule-only fallback."""

    def __init__(
        self,
        *,
        transport: JsonCompletionTransport,
        estimator_id: str,
        max_attempts: int = 2,
        fallback: RuleDSUSEstimator | None = None,
        prototype_bank: StatePrototypeBank | None = None,
        prototype_k: int = 5,
        initial_state_prior: DSUSInitialStatePrior | None = None,
        include_catalog_preview: bool = False,
    ) -> None:
        if not estimator_id.strip():
            raise ValueError("estimator_id cannot be blank")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if prototype_k < 1:
            raise ValueError("prototype_k must be positive")
        self.transport = transport
        self.estimator_id = estimator_id
        self.max_attempts = max_attempts
        self.fallback = fallback or RuleDSUSEstimator()
        self.prototype_bank = prototype_bank
        self.prototype_k = prototype_k
        self.initial_state_prior = initial_state_prior
        self.include_catalog_preview = include_catalog_preview

    @staticmethod
    def _apply_public_transition_guard(
        state: DecisionSufficientUserState,
        memory: PublicDialogueMemory,
    ) -> DecisionSufficientUserState:
        """Apply only deterministic implications of the Agent's visible history.

        This is state-transition validation, not an evaluator label. A user
        response to the Agent's immediately preceding focused question should
        be used before asking again; an explicit visible correction after a
        recommendation should create a repair state even if a language model
        described it with a weaker synonym.
        """

        if memory.last_system_action_type == BenchmarkActionType.ASK:
            return state.model_copy(
                update={
                    "dialogue_phase": DialoguePhase.RECOMMENDATION,
                    "disclosure_status": DisclosureStatus.SUFFICIENT,
                    "unresolved_axes": [],
                    "feedback_signal": FeedbackSignal.CLARIFICATION,
                    "recommendation_readiness": RecommendationReadiness.RECOMMEND_NOW,
                    "public_evidence_summary": (
                        "The user has supplied a public answer to the immediately preceding clarification. "
                        + state.public_evidence_summary
                    )[:800],
                },
                deep=True,
            )
        if memory.last_system_action_type == BenchmarkActionType.RECOMMEND:
            shapes = list(state.intent_shapes)
            if ComplexIntentShape.FEEDBACK_REPAIR not in shapes:
                shapes.insert(0, ComplexIntentShape.FEEDBACK_REPAIR)
            return state.model_copy(
                update={
                    "dialogue_phase": DialoguePhase.REPAIR,
                    "disclosure_status": DisclosureStatus.REVISED,
                    "intent_shapes": shapes[:4],
                    "unresolved_axes": [],
                    "feedback_signal": FeedbackSignal.REVISION,
                    "recommendation_readiness": RecommendationReadiness.REPAIR_RECOMMEND,
                    "public_evidence_summary": (
                        "A continuing visible response after a recommendation corrects, clarifies, or "
                        "reprioritizes the prior slate. "
                        + state.public_evidence_summary
                    )[:800],
                },
                deep=True,
            )
        if memory.completed_action_count == 0:
            visible_axis = visible_choice_axis(memory.initial_user_utterance)
            if visible_axis:
                is_conflict = _contains_any(
                    memory.initial_user_utterance.casefold(),
                    ("balance", "equally important", "prioritize"),
                )
                return state.model_copy(
                    update={
                        "dialogue_phase": DialoguePhase.ELICITATION,
                        "disclosure_status": (
                            DisclosureStatus.CONFLICT if is_conflict else DisclosureStatus.AMBIGUOUS
                        ),
                        "unresolved_axes": [visible_axis],
                        "recommendation_readiness": RecommendationReadiness.ASK_DECISIVE,
                        "confidence": max(state.confidence, 0.9),
                        "public_evidence_summary": (
                            "The initial request explicitly leaves this public ranking choice unresolved: "
                            f"{visible_axis}."
                        )[:800],
                    },
                    deep=True,
                )
            if prefers_initial_probe(memory.initial_user_utterance):
                return state.model_copy(
                    update={
                        "dialogue_phase": DialoguePhase.RECOMMENDATION,
                        "disclosure_status": DisclosureStatus.SUFFICIENT,
                        "unresolved_axes": [],
                        "recommendation_readiness": RecommendationReadiness.PROBE_RECOMMEND,
                        "confidence": max(state.confidence, 0.9),
                        "public_evidence_summary": (
                            "The user explicitly asks to react to one concrete initial option. "
                            + state.public_evidence_summary
                        )[:800],
                    },
                    deep=True,
                )
        return state

    def infer(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> DSUSInference:
        pre_state_catalog_evidence: list[CatalogRetrievalItem] = []
        if self.include_catalog_preview:
            pre_state_retrieval = retriever.search(SearchSpec.from_public_query(memory.retrieval_text, ranking_k=5))
            pre_state_catalog_evidence = pre_state_retrieval.results
        payload: Mapping[str, object] = {
            "task": "infer_decision_sufficient_user_state",
            "domain": memory.domain,
            "public_dialogue": memory.serialized_dialogue,
            "asked_question_targets": memory.asked_question_targets,
            "previously_recommended_item_ids": memory.previously_recommended_item_ids,
        }
        if pre_state_catalog_evidence:
            payload = {
                **payload,
                "public_catalog_preview": [
                    {
                        "item_id": item.item_id,
                        "title": item.title,
                        "category": item.category,
                        "score": round(item.score, 4),
                    }
                    for item in pre_state_catalog_evidence
                ],
            }
        initial_state_prior: DSUSPriorPrediction | None = None
        if self.initial_state_prior is not None and memory.completed_action_count == 0:
            initial_state_prior = self.initial_state_prior.predict(
                domain=memory.domain,
                query=memory.initial_user_utterance,
            )
            payload = {
                **payload,
                "source_disjoint_initial_state_prior": initial_state_prior.model_dump(mode="json"),
            }
        if self.prototype_bank is not None:
            payload = {
                **payload,
                "development_state_prototypes": self.prototype_bank.examples(
                    domain=memory.domain,
                    query=memory.latest_user_utterance,
                    k=self.prototype_k,
                ),
            }
        audits: list[LLMCompletionAudit] = []
        retry_feedback: str | None = None
        for attempt_index in range(1, self.max_attempts + 1):
            completion: JsonCompletion | None = None
            try:
                completion = self.transport.complete_json(
                    system_prompt=_DSUS_PROMPT,
                    user_payload=payload,
                    retry_feedback=retry_feedback,
                )
                raw = _strict_json_object(completion.content)
                raw_posterior = raw.pop("decision_posterior", None)
                state = DecisionSufficientUserState(
                    estimator_id=self.estimator_id,
                    domain=memory.domain,
                    turn_index=memory.completed_action_count,
                    **raw,
                )
                parsed_readiness = state.recommendation_readiness
                state = self._apply_public_transition_guard(state, memory)
                posterior = DSUSDecisionPosterior.from_raw(
                    None if parsed_readiness != state.recommendation_readiness else raw_posterior,
                    selected=state.recommendation_readiness,
                    selected_probability=state.confidence,
                )
                if memory.completed_action_count > 0:
                    posterior = DSUSDecisionPosterior.from_raw(
                        None,
                        selected=state.recommendation_readiness,
                        selected_probability=0.999,
                    )
                audits.append(_llm_audit(attempt_index=attempt_index, completion=completion))
                return DSUSInference(
                    state=state,
                    llm_audits=audits,
                    initial_state_prior=initial_state_prior,
                    pre_state_catalog_evidence=pre_state_catalog_evidence,
                    decision_posterior=posterior,
                )
            except Exception as exc:
                audits.append(_llm_audit(attempt_index=attempt_index, completion=completion, failure=exc))
                retry_feedback = (
                    "Your previous response could not be validated. Return only the exact DSUS JSON object and use "
                    "only the supplied public dialogue."
                )
        fallback = self.fallback.infer(memory=memory, retriever=retriever)
        return DSUSInference(
            state=fallback.state.model_copy(update={"estimator_id": f"{self.estimator_id}:rule_fallback"}),
            llm_audits=audits,
            used_fallback=True,
            initial_state_prior=initial_state_prior,
            pre_state_catalog_evidence=pre_state_catalog_evidence,
            decision_posterior=DSUSDecisionPosterior.from_raw(
                None,
                selected=fallback.state.recommendation_readiness,
            ),
        )
