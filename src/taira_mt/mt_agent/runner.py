"""A public-only rollout bridge for the frozen full-catalog benchmark.

The runner is deliberately thin: policies receive public dialogue memory and a
shared catalogue retriever, while the benchmark session remains responsible for
the hidden simulator state and post-hoc evaluation.  This keeps direct LLM,
static-pattern, and TAIRAS comparisons on exactly the same tool surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import Field, model_validator

from taira_mt.benchmark.schemas import BenchmarkActionType, BenchmarkSystemAction, PublicEnvironmentResponse
from taira_mt.models import StrictModel
from taira_mt.mt_agent.memory import PublicDialogueMemory
from taira_mt.mt_agent.retrieval import CatalogRetrievalResult, LexicalCatalogRetriever, SearchSpec


class CompatiblePublicSession(Protocol):
    """The only session operations available to a paper-facing Agent."""

    domain: str
    candidate_pool_id: str

    @property
    def terminal(self) -> bool: ...

    def reset(self) -> str: ...

    def step(self, action: BenchmarkSystemAction) -> PublicEnvironmentResponse: ...


class LLMCompletionAudit(StrictModel):
    """Secret-free telemetry for one constrained LLM decision substep."""

    stage: str = Field(min_length=1)
    attempt_index: int = Field(ge=1)
    status: str = Field(min_length=1)
    requested_model: str | None = None
    returned_model: str | None = None
    finish_reason: str | None = None
    raw_content_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    backend_id: str | None = None
    failure_kind: str | None = None
    failure_code: str | None = None
    reasoning_tokens: int | None = Field(default=None, ge=0)


class MTAgentDecision(StrictModel):
    """One typed decision plus the public retrieval evidence that grounded it."""

    action: BenchmarkSystemAction
    search_spec: SearchSpec | None = None
    retrieval: CatalogRetrievalResult | None = None
    decision_source: str = Field(min_length=1)
    selected_skill_id: str | None = None
    rationale: str = Field(default="")
    llm_audits: list[LLMCompletionAudit] = Field(default_factory=list)
    method_trace: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_recommendation_grounding(self) -> MTAgentDecision:
        if self.retrieval is not None and self.search_spec != self.retrieval.search_spec:
            raise ValueError("decision search_spec does not match its retrieval evidence")
        if self.action.action_type == BenchmarkActionType.RECOMMEND:
            if self.retrieval is None:
                raise ValueError("a recommendation must carry public retrieval evidence")
            retrieved_item_ids = {item.item_id for item in self.retrieval.results}
            if not set(self.action.item_ids).issubset(retrieved_item_ids):
                raise ValueError("recommended IDs must come from the declared public retrieval result")
        return self


class MTAgentTurn(StrictModel):
    turn_index: int = Field(ge=1)
    decision: MTAgentDecision
    public_response: PublicEnvironmentResponse


class MTAgentEpisode(StrictModel):
    runner_id: str
    policy_id: str
    domain: str
    initial_user_utterance: str
    turns: list[MTAgentTurn]
    final_memory: PublicDialogueMemory
    terminal: bool
    reached_turn_limit: bool


class MTAgentPolicy(Protocol):
    """Policy contract shared by all paper baselines and the proposed method."""

    policy_id: str

    def decide(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> MTAgentDecision: ...


class MultiDomainRetrieverRegistry:
    """Lazy loader for the common, pre-built complete-catalog retrieval layer."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._retrievers: dict[str, LexicalCatalogRetriever] = {}

    def get(self, domain: str) -> LexicalCatalogRetriever:
        if domain not in self._retrievers:
            path = self.root / domain / "lexical_index.joblib"
            self._retrievers[domain] = LexicalCatalogRetriever.load(path)
        return self._retrievers[domain]


class MTAgentRunner:
    """Run a policy without exposing evaluator-owned state to that policy."""

    runner_id = "mt_agent_public_rollout_runner_v1"

    def __init__(self, *, retrievers: MultiDomainRetrieverRegistry) -> None:
        self.retrievers = retrievers

    @staticmethod
    def _budget_stop() -> MTAgentDecision:
        return MTAgentDecision(
            action=BenchmarkSystemAction(
                action_type=BenchmarkActionType.STOP,
                utterance="I will stop here because the public interaction budget has been reached.",
            ),
            decision_source="runner_turn_budget_guard",
            rationale="Safety termination after the configured public rollout limit.",
        )

    def _run_turn(
        self,
        *,
        session: CompatiblePublicSession,
        memory: PublicDialogueMemory,
        decision: MTAgentDecision,
        turn_index: int,
    ) -> tuple[PublicDialogueMemory, MTAgentTurn]:
        if decision.retrieval is not None and decision.retrieval.pool_id != session.candidate_pool_id:
            raise ValueError("Agent decision retrieval belongs to a different candidate pool")
        response = session.step(decision.action)
        updated_memory = memory.record_system_action(decision.action).record_user_response(
            response.utterance,
            terminal=response.terminal,
        )
        return updated_memory, MTAgentTurn(
            turn_index=turn_index,
            decision=decision,
            public_response=response,
        )

    def run(
        self,
        session: CompatiblePublicSession,
        *,
        policy: MTAgentPolicy,
        maximum_turns: int = 3,
    ) -> MTAgentEpisode:
        if maximum_turns < 1:
            raise ValueError("maximum_turns must be positive")
        initial_utterance = session.reset()
        memory = PublicDialogueMemory.begin(domain=session.domain, initial_user_utterance=initial_utterance)
        retriever = self.retrievers.get(session.domain)
        if retriever.pool_id != session.candidate_pool_id:
            raise ValueError("shared retriever and public session use different candidate pools")

        turns: list[MTAgentTurn] = []
        for turn_index in range(1, maximum_turns + 1):
            if session.terminal:
                break
            decision = policy.decide(memory=memory, retriever=retriever)
            memory, turn = self._run_turn(
                session=session,
                memory=memory,
                decision=decision,
                turn_index=turn_index,
            )
            turns.append(turn)
            if session.terminal:
                break

        reached_turn_limit = not session.terminal
        if reached_turn_limit:
            memory, turn = self._run_turn(
                session=session,
                memory=memory,
                decision=self._budget_stop(),
                turn_index=len(turns) + 1,
            )
            turns.append(turn)
        return MTAgentEpisode(
            runner_id=self.runner_id,
            policy_id=policy.policy_id,
            domain=session.domain,
            initial_user_utterance=initial_utterance,
            turns=turns,
            final_memory=memory,
            terminal=session.terminal,
            reached_turn_limit=reached_turn_limit,
        )
