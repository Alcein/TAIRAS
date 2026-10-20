"""Agent-safe facade around the deterministic evaluator-owned user environment."""

from __future__ import annotations

from taira_mt.agent.catalog_tools import PublicCatalogTool
from taira_mt.benchmark.environment import DeterministicEvolvingUserEnvironment
from taira_mt.benchmark.schemas import (
    BenchmarkSystemAction,
    PublicBenchmarkCase,
    PublicEnvironmentResponse,
)


class PublicAgentSession:
    """Expose only a public query, typed actions, user text, and catalog tools.

    The wrapped evaluator remains private by construction.  This is an API
    discipline rather than a Python sandbox: production Agent code must accept
    this facade (or its compatible protocol), never a raw environment.
    """

    def __init__(
        self,
        *,
        environment: DeterministicEvolvingUserEnvironment,
        public_case: PublicBenchmarkCase,
    ) -> None:
        if environment.case.public.case_id != public_case.case_id:
            raise ValueError("public session facade received a mismatched public case")
        self.__environment = environment
        self.domain = public_case.domain
        self.candidate_pool_id = public_case.candidate_pool_id
        self.catalog = PublicCatalogTool(environment.candidate_pool)
        self._terminal = False

    @property
    def terminal(self) -> bool:
        """The terminal bit is public because it is returned after every step."""

        return self._terminal

    def reset(self) -> str:
        """Start a fresh deterministic conversation and return only its public query."""

        self._terminal = False
        return self.__environment.reset()

    def step(self, action: BenchmarkSystemAction) -> PublicEnvironmentResponse:
        """Submit one typed public action and receive only the public user response."""

        if self._terminal:
            raise RuntimeError("cannot step a public session after a terminal response")
        response = self.__environment.step(action)
        self._terminal = response.terminal
        return response
