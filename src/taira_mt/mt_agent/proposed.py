"""TAIRAS policy: DOUS/history planner view -> SCTP -> CIV -> executor."""

from __future__ import annotations

from taira_mt.mt_agent.civ import CIVLLMRanker, CIVPairwiseLinearRanker, CIVRanker, SOPFirstRanker
from taira_mt.mt_agent.executor import LLMThoughtPatternExecutor
from taira_mt.mt_agent.memory import PublicDialogueMemory
from taira_mt.mt_agent.retrieval import LexicalCatalogRetriever
from taira_mt.mt_agent.runner import MTAgentDecision
from taira_mt.mt_agent.sctp import SCTPSkillCompiler
from taira_mt.mt_agent.state import DSUSEstimator

TAIRA_MT_POLICY_VERSION = "taira_mt_state_skill_civ_policy_v1"


class TAIRAMTPolicy:
    """Compose the three claimed modules without leaking evaluator information.

    - the state estimator sees public dialogue and produces decision-oriented state;
    - state-addressed planning retrieves TAIRA's SOP-plus-insight Thought
      Patterns and compiles legal actions;
    - CIV ranks those legal proposals only; and
    - this policy executes the selected typed public action.
    """

    def __init__(
        self,
        *,
        dsus: DSUSEstimator,
        sctp: SCTPSkillCompiler,
        civ: CIVRanker | CIVPairwiseLinearRanker | CIVLLMRanker | SOPFirstRanker,
        executor: LLMThoughtPatternExecutor | None = None,
    ) -> None:
        self.dsus = dsus
        self.sctp = sctp
        self.civ = civ
        self.executor = executor
        self.policy_id = (
            f"{TAIRA_MT_POLICY_VERSION}:{dsus.estimator_id}:{sctp.library.library_id}:"
            f"{sctp.compiler_id}:{civ.model_id}:"
            f"{'deterministic_executor' if executor is None else executor.executor_id}"
        )

    def decide(self, *, memory: PublicDialogueMemory, retriever: LexicalCatalogRetriever) -> MTAgentDecision:
        inference = self.dsus.infer(memory=memory, retriever=retriever)
        frontier = self.sctp.compile(
            memory=memory,
            state=inference.state,
            retriever=retriever,
            decision_posterior=inference.decision_posterior,
        )
        selection = self.civ.select(frontier, memory=memory)
        proposal = self.civ.selected_proposal(frontier, selection)
        execution = (
            None
            if self.executor is None
            else self.executor.execute(
                proposal=proposal,
                memory=memory,
                retriever=retriever,
            )
        )
        return MTAgentDecision(
            action=proposal.action if execution is None else execution.action,
            search_spec=proposal.search_spec if execution is None else execution.search_spec,
            retrieval=proposal.retrieval if execution is None else execution.retrieval,
            decision_source=TAIRA_MT_POLICY_VERSION,
            selected_skill_id=f"{proposal.semantic_skill_id} + {proposal.interaction_skill_id}",
            rationale=proposal.insight_summary,
            llm_audits=[
                *inference.llm_audits,
                *selection.llm_audits,
                *([] if execution is None else execution.llm_audits),
            ],
            method_trace={
                "dsus_state": inference.state.model_dump(mode="json"),
                "dsus_used_fallback": inference.used_fallback,
                "dsus_initial_state_prior": (
                    None
                    if inference.initial_state_prior is None
                    else inference.initial_state_prior.model_dump(mode="json")
                ),
                "dsus_pre_state_catalog_evidence": [
                    item.model_dump(mode="json") for item in inference.pre_state_catalog_evidence
                ],
                "dsus_decision_posterior": (
                    None
                    if inference.decision_posterior is None
                    else inference.decision_posterior.model_dump(mode="json")
                ),
                "sctp_frontier": frontier.model_dump(mode="json"),
                "civ_selection": selection.model_dump(mode="json"),
                "selected_proposal_id": proposal.proposal_id,
                "skill_execution": (
                    {
                        "executor_id": "deterministic_executor",
                        "used_fallback": False,
                        "grounding_plan": None,
                        "question_plan": None,
                        "candidate_fusion": None,
                        "pattern_candidate_count": 0,
                        "dialogue_candidate_count": 0,
                        "direct_candidate_count": 0,
                        "fused_candidate_count": 0,
                        "focused_head3_applied": False,
                        "focused_tail_applied": False,
                        "focused_tail_new_item_count": 0,
                        "public_head_guard": None,
                    }
                    if execution is None
                    else {
                        "executor_id": execution.executor_id,
                        "used_fallback": execution.used_fallback,
                        "candidate_fusion": execution.candidate_fusion,
                        "pattern_candidate_count": execution.pattern_candidate_count,
                        "dialogue_candidate_count": execution.dialogue_candidate_count,
                        "direct_candidate_count": execution.direct_candidate_count,
                        "fused_candidate_count": execution.fused_candidate_count,
                        "focused_head3_applied": execution.focused_head3_applied,
                        "focused_tail_applied": execution.focused_tail_applied,
                        "focused_tail_new_item_count": execution.focused_tail_new_item_count,
                        "public_head_guard": (
                            None
                            if execution.public_head_guard is None
                            else execution.public_head_guard.model_dump(mode="json")
                        ),
                        "grounding_plan": (
                            None
                            if execution.grounding_plan is None
                            else execution.grounding_plan.model_dump(mode="json")
                        ),
                        "question_plan": (
                            None
                            if execution.question_plan is None
                            else execution.question_plan.model_dump(mode="json")
                        ),
                    }
                ),
            },
        )


# Public API name.  The historical class remains available because saved
# experiment scripts import it directly.
TAIRASPolicy = TAIRAMTPolicy
