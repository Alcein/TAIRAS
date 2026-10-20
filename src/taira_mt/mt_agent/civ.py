"""Counterfactual Interaction Value (CIV) ranking over SCTP's legal frontier.

CIV never invents a Skill, hidden state, catalog item, or action.  It evaluates
only state-conditioned proposals already compiled by SCTP, making its role
different from both DSUS perception and Thought-Pattern retrieval.  The seeded
weights provide a runnable development control; the same feature contract is
intended for train-only pairwise calibration in the next phase.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import Field
from sklearn.linear_model import LogisticRegression

from taira_mt.benchmark.schemas import BenchmarkActionType
from taira_mt.models import StrictModel
from taira_mt.mt_agent.memory import PublicDialogueMemory
from taira_mt.mt_agent.runner import LLMCompletionAudit
from taira_mt.mt_agent.sctp import SCTPActionFrontier, SkillActionProposal
from taira_mt.mt_agent.state import RecommendationReadiness
from taira_mt.provenance import content_sha256
from taira_mt.providers.openai_compatible import JsonCompletion, JsonCompletionTransport

CIV_VERSION = "counterfactual_interaction_value_v1"
CIV_PAIRWISE_LINEAR_VERSION = "civ_public_pairwise_linear_v1"
CIV_PUBLIC_FEATURE_NAMES = (
    "interaction_match",
    "semantic_match",
    "log_state_probability",
    "state_action_alignment",
    "retrieval_top_score",
    "candidate_available",
    "question_target_present",
    "is_ask",
    "is_recommend",
    "is_stop",
    "repair_novelty",
)


class CIVWeights(StrictModel):
    """Small transparent linear prior; later train-only fitting preserves these features."""

    schema_version: str = CIV_VERSION
    model_id: str = "civ_seeded_public_value_prior_v1"
    interaction_match_weight: float = 0.18
    semantic_match_weight: float = 0.07
    retrieval_quality_weight: float = 0.60
    candidate_availability_bonus: float = 0.20
    targeted_question_bonus: float = 0.20
    ask_cost: float = -0.12
    recommend_cost: float = -0.03
    stop_cost: float = -1.40
    recommend_when_ready: float = 1.20
    recommend_repair: float = 1.45
    recommend_probe: float = 0.50
    recommend_when_ask_needed: float = -1.25
    ask_when_needed: float = 1.25
    ask_when_ready: float = -1.20
    ask_after_repair: float = -0.90
    repair_exclusion_bonus: float = 0.30
    state_hypothesis_prior_weight: float = 1.10


class CIVProposalScore(StrictModel):
    proposal_id: str
    total_value: float
    components: dict[str, float]


class CIVSelection(StrictModel):
    model_id: str
    selected_proposal_id: str
    scores: list[CIVProposalScore] = Field(min_length=1)
    decision_summary: str = ""
    llm_audits: list[LLMCompletionAudit] = Field(default_factory=list)


class SOPFirstRanker:
    """Fixed-priority control that selects the first compiled legal directive.

    Unlike heuristic or learned CIV, this selector assigns no counterfactual
    value to asking, recommending, or repairing.  It isolates whether value
    ranking contributes beyond Pattern matching and frontier construction.
    """

    model_id = "sop_first_fixed_priority_v1"

    def select(
        self,
        frontier: SCTPActionFrontier,
        *,
        memory: PublicDialogueMemory | None = None,
    ) -> CIVSelection:
        del memory
        scores = [
            CIVProposalScore(
                proposal_id=proposal.proposal_id,
                total_value=-float(index),
                components={"fixed_frontier_priority": -float(index)},
            )
            for index, proposal in enumerate(frontier.proposals)
        ]
        return CIVSelection(
            model_id=self.model_id,
            selected_proposal_id=frontier.proposals[0].proposal_id,
            scores=scores,
            decision_summary="Fixed control selected the first legal Pattern directive without value ranking.",
        )

    @staticmethod
    def selected_proposal(
        frontier: SCTPActionFrontier,
        selection: CIVSelection,
    ) -> SkillActionProposal:
        for proposal in frontier.proposals:
            if proposal.proposal_id == selection.selected_proposal_id:
                return proposal
        raise ValueError("fixed selector refers to a proposal absent from its SCTP frontier")


class CIVRanker:
    """Rank existing legal Skills/actions with explicit public value components."""

    def __init__(self, *, weights: CIVWeights | None = None) -> None:
        self.weights = weights or CIVWeights()
        self.model_id = self.weights.model_id

    def _readiness_component(self, proposal: SkillActionProposal, readiness: RecommendationReadiness) -> float:
        action_type = proposal.action.action_type
        if action_type == BenchmarkActionType.RECOMMEND:
            if readiness == RecommendationReadiness.RECOMMEND_NOW:
                return self.weights.recommend_when_ready
            if readiness == RecommendationReadiness.REPAIR_RECOMMEND:
                return self.weights.recommend_repair
            if readiness == RecommendationReadiness.PROBE_RECOMMEND:
                return self.weights.recommend_probe
            if readiness == RecommendationReadiness.ASK_DECISIVE:
                return self.weights.recommend_when_ask_needed
        if action_type == BenchmarkActionType.ASK:
            if readiness == RecommendationReadiness.ASK_DECISIVE:
                return self.weights.ask_when_needed
            if readiness == RecommendationReadiness.REPAIR_RECOMMEND:
                return self.weights.ask_after_repair
            return self.weights.ask_when_ready
        if action_type == BenchmarkActionType.STOP:
            return self.weights.stop_cost
        return -1.0

    def score(self, proposal: SkillActionProposal, frontier: SCTPActionFrontier) -> CIVProposalScore:
        weights = self.weights
        components: dict[str, float] = {
            "state_action_alignment": self._readiness_component(
                proposal, proposal.state_hypothesis.recommendation_readiness
            ),
            "interaction_skill_match": weights.interaction_match_weight * proposal.interaction_match_score,
            "semantic_skill_match": weights.semantic_match_weight * proposal.semantic_match_score,
            "state_hypothesis_prior": weights.state_hypothesis_prior_weight
            * math.log(max(proposal.state_hypothesis_probability, 1e-6)),
        }
        if proposal.action.action_type == BenchmarkActionType.ASK:
            components["interaction_cost"] = weights.ask_cost
            components["targeted_question"] = weights.targeted_question_bonus * float(
                bool(proposal.action.question_target)
            )
        elif proposal.action.action_type == BenchmarkActionType.RECOMMEND:
            components["interaction_cost"] = weights.recommend_cost
            if proposal.retrieval is not None and proposal.retrieval.results:
                top_score = max(0.0, proposal.retrieval.results[0].score)
                components["retrieval_quality"] = weights.retrieval_quality_weight * min(top_score, 1.0)
                components["candidate_availability"] = weights.candidate_availability_bonus
            else:
                components["retrieval_quality"] = -weights.candidate_availability_bonus
            if (
                proposal.state_hypothesis.recommendation_readiness == RecommendationReadiness.REPAIR_RECOMMEND
                and proposal.search_spec is not None
                and proposal.search_spec.excluded_item_ids
            ):
                components["repair_novelty"] = weights.repair_exclusion_bonus
        else:
            components["interaction_cost"] = weights.stop_cost
        return CIVProposalScore(
            proposal_id=proposal.proposal_id,
            total_value=sum(components.values()),
            components=components,
        )

    def select(
        self,
        frontier: SCTPActionFrontier,
        *,
        memory: PublicDialogueMemory | None = None,
    ) -> CIVSelection:
        del memory
        scores = [self.score(proposal, frontier) for proposal in frontier.proposals]
        selected = max(scores, key=lambda item: (item.total_value, item.proposal_id))
        return CIVSelection(
            model_id=self.model_id,
            selected_proposal_id=selected.proposal_id,
            scores=sorted(scores, key=lambda item: (-item.total_value, item.proposal_id)),
        )

    @staticmethod
    def selected_proposal(frontier: SCTPActionFrontier, selection: CIVSelection) -> SkillActionProposal:
        for proposal in frontier.proposals:
            if proposal.proposal_id == selection.selected_proposal_id:
                return proposal
        raise ValueError("CIV selection refers to a proposal absent from its SCTP frontier")


def civ_public_feature_values(proposal: SkillActionProposal) -> dict[str, float]:
    """Frozen public proposal features shared by training and online ranking."""

    readiness = proposal.state_hypothesis.recommendation_readiness
    action_type = proposal.action.action_type
    aligned = (
        (action_type == BenchmarkActionType.ASK and readiness == RecommendationReadiness.ASK_DECISIVE)
        or (
            action_type == BenchmarkActionType.RECOMMEND
            and readiness
            in {
                RecommendationReadiness.RECOMMEND_NOW,
                RecommendationReadiness.PROBE_RECOMMEND,
                RecommendationReadiness.REPAIR_RECOMMEND,
            }
        )
        or (action_type == BenchmarkActionType.STOP and readiness == RecommendationReadiness.SAFE_STOP)
    )
    retrieval_top_score = (
        max(0.0, proposal.retrieval.results[0].score)
        if proposal.retrieval is not None and proposal.retrieval.results
        else 0.0
    )
    return {
        "interaction_match": proposal.interaction_match_score,
        "semantic_match": proposal.semantic_match_score,
        "log_state_probability": math.log(max(proposal.state_hypothesis_probability, 1e-6)),
        "state_action_alignment": 1.0 if aligned else -1.0,
        "retrieval_top_score": min(retrieval_top_score, 2.0),
        "candidate_available": float(bool(proposal.retrieval and proposal.retrieval.results)),
        "question_target_present": float(bool(proposal.action.question_target)),
        "is_ask": float(action_type == BenchmarkActionType.ASK),
        "is_recommend": float(action_type == BenchmarkActionType.RECOMMEND),
        "is_stop": float(action_type == BenchmarkActionType.STOP),
        "repair_novelty": float(
            bool(
                proposal.search_spec
                and proposal.search_spec.excluded_item_ids
                and readiness == RecommendationReadiness.REPAIR_RECOMMEND
            )
        ),
    }


def civ_public_feature_vector(proposal: SkillActionProposal) -> np.ndarray:
    values = civ_public_feature_values(proposal)
    return np.asarray([values[name] for name in CIV_PUBLIC_FEATURE_NAMES], dtype=np.float64)


@dataclass
class CIVPairwiseLinearRanker:
    """Small train-only pairwise value model over the frozen public feature contract."""

    coefficients: np.ndarray
    model_id: str
    feature_names: tuple[str, ...] = CIV_PUBLIC_FEATURE_NAMES
    model_version: str = CIV_PAIRWISE_LINEAR_VERSION

    @classmethod
    def fit(cls, differences: np.ndarray, labels: np.ndarray) -> CIVPairwiseLinearRanker:
        if differences.ndim != 2 or differences.shape[1] != len(CIV_PUBLIC_FEATURE_NAMES):
            raise ValueError("CIV pairwise differences do not match the public feature contract")
        if len(differences) < 2 or len(labels) != len(differences) or set(labels.tolist()) != {0, 1}:
            raise ValueError("CIV pairwise training needs matched positive and negative comparisons")
        model = LogisticRegression(
            C=0.75,
            class_weight="balanced",
            fit_intercept=False,
            max_iter=2_000,
            random_state=23,
        )
        model.fit(differences, labels)
        coefficients = model.coef_[0].astype(np.float64, copy=True)
        identity = content_sha256(
            {
                "version": CIV_PAIRWISE_LINEAR_VERSION,
                "feature_names": list(CIV_PUBLIC_FEATURE_NAMES),
                "coefficients": coefficients.round(10).tolist(),
                "training_rows": len(differences),
            }
        )
        return cls(
            coefficients=coefficients,
            model_id=f"civ_pairwise_linear::{identity[:16]}",
        )

    def _score(self, proposal: SkillActionProposal) -> CIVProposalScore:
        values = civ_public_feature_values(proposal)
        components = {
            name: float(self.coefficients[index] * values[name]) for index, name in enumerate(self.feature_names)
        }
        return CIVProposalScore(
            proposal_id=proposal.proposal_id,
            total_value=sum(components.values()),
            components=components,
        )

    def select(
        self,
        frontier: SCTPActionFrontier,
        *,
        memory: PublicDialogueMemory | None = None,
    ) -> CIVSelection:
        del memory
        scores = [self._score(proposal) for proposal in frontier.proposals]
        selected = max(scores, key=lambda item: (item.total_value, item.proposal_id))
        return CIVSelection(
            model_id=self.model_id,
            selected_proposal_id=selected.proposal_id,
            scores=sorted(scores, key=lambda item: (-item.total_value, item.proposal_id)),
            decision_summary="Pairwise public-value model selected the highest-valued legal Skill action.",
        )

    selected_proposal = staticmethod(CIVRanker.selected_proposal)

    def save(self, path: Path) -> None:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("saving a CIV pairwise model requires ML dependencies") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_version": self.model_version,
                "model_id": self.model_id,
                "feature_names": list(self.feature_names),
                "coefficients": self.coefficients,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> CIVPairwiseLinearRanker:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("loading a CIV pairwise model requires ML dependencies") from exc
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("model_version") != CIV_PAIRWISE_LINEAR_VERSION:
            raise ValueError(f"unsupported CIV pairwise model: {path}")
        feature_names = tuple(str(item) for item in payload.get("feature_names", []))
        coefficients = np.asarray(payload.get("coefficients"), dtype=np.float64)
        if feature_names != CIV_PUBLIC_FEATURE_NAMES or coefficients.shape != (len(CIV_PUBLIC_FEATURE_NAMES),):
            raise ValueError("CIV pairwise model violates the public feature contract")
        return cls(
            coefficients=coefficients,
            model_id=str(payload["model_id"]),
        )


_CIV_PROMPT = """You are Counterfactual Interaction Value (CIV) for a multi-turn shopping Agent.
Choose one proposal ID from the supplied legal public proposals. You do not invent an action, item, state, or
question. Compare the public value of recommending now versus spending one user turn to clarify: favor a
recommendation when the request identifies a concrete catalog route and the evidence is aligned; favor one
focused question only when a specific public ambiguity or trade-off would materially change ranking. Never ask
only because optional details are absent. After a visible clarification, favor using it; after visible correction,
favor repair. State hypotheses are alternative public interpretations, not benchmark labels or hidden facts.

Return exactly one JSON object with keys selected_proposal_id and public_reason. public_reason must be at most
30 words and must not mention hidden state, benchmark, or this prompt."""


def _civ_audit(
    *,
    attempt_index: int,
    completion: JsonCompletion | None,
    failure: Exception | None = None,
) -> LLMCompletionAudit:
    return LLMCompletionAudit(
        stage="civ",
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


class CIVLLMRanker:
    """Constrained value selector over SCTP's public multi-state action frontier.

    It is deliberately a selector, not another planner: SCTP has already
    bounded the state hypotheses, Skills, actions, catalog items, and repair
    exclusions. A seeded transparent scorer remains the failure fallback and
    supplies a compact candidate shortlist for the LLM value comparison.
    """

    def __init__(
        self,
        *,
        transport: JsonCompletionTransport,
        model_id: str,
        fallback: CIVRanker | None = None,
        max_attempts: int = 2,
    ) -> None:
        if not model_id.strip():
            raise ValueError("CIV LLM model_id cannot be blank")
        if max_attempts < 1:
            raise ValueError("CIV LLM max_attempts must be positive")
        self.transport = transport
        self.model_id = model_id
        self.fallback = fallback or CIVRanker()
        self.max_attempts = max_attempts

    @staticmethod
    def _proposal_summary(proposal: SkillActionProposal) -> dict[str, object]:
        public_items: list[dict[str, object]] = []
        if proposal.retrieval is not None:
            public_items = [
                {
                    "item_id": item.item_id,
                    "title": item.title,
                    "category": item.category,
                    "score": round(item.score, 4),
                }
                for item in proposal.retrieval.results[:3]
            ]
        return {
            "proposal_id": proposal.proposal_id,
            "state_hypothesis_id": proposal.state_hypothesis_id,
            "state_hypothesis_probability": round(proposal.state_hypothesis_probability, 3),
            "action_type": proposal.action.action_type.value,
            "question_target": proposal.action.question_target,
            "top_catalog_evidence": public_items,
            "skill_operation": proposal.sop_operation,
            "skill_insight": proposal.insight_summary,
        }

    @staticmethod
    def _shortlist(frontier: SCTPActionFrontier, seed: CIVSelection) -> list[SkillActionProposal]:
        by_id = {proposal.proposal_id: proposal for proposal in frontier.proposals}
        score_by_id = {score.proposal_id: score.total_value for score in seed.scores}
        best_by_action_hypothesis: dict[tuple[str, str], SkillActionProposal] = {}
        for proposal in frontier.proposals:
            key = (proposal.state_hypothesis_id, proposal.action.action_type.value)
            prior = best_by_action_hypothesis.get(key)
            if prior is None or (score_by_id[proposal.proposal_id], proposal.proposal_id) > (
                score_by_id[prior.proposal_id],
                prior.proposal_id,
            ):
                best_by_action_hypothesis[key] = proposal
        selected = list(best_by_action_hypothesis.values())
        selected_ids = {proposal.proposal_id for proposal in selected}
        for scored in seed.scores:
            if len(selected) >= 8:
                break
            proposal = by_id[scored.proposal_id]
            if proposal.proposal_id not in selected_ids:
                selected.append(proposal)
                selected_ids.add(proposal.proposal_id)
        return sorted(
            selected,
            key=lambda proposal: (-score_by_id[proposal.proposal_id], proposal.proposal_id),
        )

    def select(self, frontier: SCTPActionFrontier, *, memory: PublicDialogueMemory | None = None) -> CIVSelection:
        if memory is None:
            raise ValueError("CIVLLMRanker requires public dialogue memory")
        seed = self.fallback.select(frontier)
        candidates = self._shortlist(frontier, seed)
        payload = {
            "task": "select_counterfactual_interaction_value",
            "public_dialogue": memory.serialized_dialogue,
            "state_hypotheses": [
                {
                    "hypothesis_id": item.hypothesis_id,
                    "probability": round(item.probability, 3),
                    "disclosure_status": item.state.disclosure_status.value,
                    "recommendation_readiness": item.state.recommendation_readiness.value,
                    "unresolved_axes": item.state.unresolved_axes,
                }
                for item in frontier.state_hypotheses
            ],
            "legal_proposals": [self._proposal_summary(proposal) for proposal in candidates],
        }
        audits: list[LLMCompletionAudit] = []
        retry_feedback: str | None = None
        candidate_ids = {proposal.proposal_id for proposal in candidates}
        for attempt_index in range(1, self.max_attempts + 1):
            completion: JsonCompletion | None = None
            try:
                completion = self.transport.complete_json(
                    system_prompt=_CIV_PROMPT,
                    user_payload=payload,
                    retry_feedback=retry_feedback,
                )
                raw = json.loads(completion.content)
                if not isinstance(raw, dict):
                    raise ValueError("CIV completion must be a JSON object")
                proposal_id = str(raw.get("selected_proposal_id", "")).strip()
                if proposal_id not in candidate_ids:
                    raise ValueError("CIV completion selected an absent proposal")
                public_reason = " ".join(str(raw.get("public_reason", "")).split())[:300]
                audits.append(_civ_audit(attempt_index=attempt_index, completion=completion))
                return CIVSelection(
                    model_id=self.model_id,
                    selected_proposal_id=proposal_id,
                    scores=seed.scores,
                    decision_summary=public_reason,
                    llm_audits=audits,
                )
            except Exception as exc:
                audits.append(_civ_audit(attempt_index=attempt_index, completion=completion, failure=exc))
                retry_feedback = "Return only valid JSON and select exactly one listed proposal_id."
        return seed.model_copy(
            update={
                "model_id": f"{self.model_id}:seeded_fallback",
                "decision_summary": "The constrained value selector fell back to the seeded public scorer.",
                "llm_audits": audits,
            },
            deep=True,
        )

    selected_proposal = staticmethod(CIVRanker.selected_proposal)
