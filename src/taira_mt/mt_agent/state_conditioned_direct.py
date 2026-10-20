"""Architecture control: the same learned DSUS, with free tool planning."""
from taira_mt.mt_agent.direct_llm import DirectLLMToolPolicy


class StateConditionedDirectPolicy(DirectLLMToolPolicy):
    """Remove the Pattern frontier and CIV together, retaining state and tools.

    This is explicitly a procedural-controller architecture contrast, not a
    single SCTP-component ablation with an unchanged executor.
    """

    def __init__(self, *, dsus, **kwargs):
        super().__init__(**kwargs)
        self.dsus = dsus
        self._current_state = None
        self.policy_id = f"sft_state_free_tool_planning_v1:{dsus.estimator_id}:{self.planner_id}"

    def _plan_payload(self, memory):
        payload = super()._plan_payload(memory)
        payload["compiled_public_user_state"] = self._current_state
        payload["state_instruction"] = (
            "Use the compiled public user state and the complete public dialogue together. "
            "Resolve any discrepancy using explicit user statements. Choose the next action "
            "and translate the visible needs into catalog attributes."
        )
        return payload

    def decide(self, *, memory, retriever):
        inference = self.dsus.infer(memory=memory, retriever=retriever)
        self._current_state = inference.state.model_dump(mode="json")
        decision = super().decide(memory=memory, retriever=retriever)
        return decision.model_copy(update={
            "llm_audits": [*inference.llm_audits, *decision.llm_audits],
            "method_trace": {
                **decision.method_trace,
                "dsus_state": self._current_state,
                "dsus_used_fallback": inference.used_fallback,
                "architecture_control": "same_dsus_free_planning_without_pattern_frontier_or_civ",
            },
        })
