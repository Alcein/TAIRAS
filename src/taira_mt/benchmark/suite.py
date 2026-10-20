"""Public loading interface for the frozen evolving-user benchmark."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from taira_mt.benchmark.diagnostics import LoadedPilotCase, load_symbolic_pilot
from taira_mt.benchmark.environment import DeterministicEvolvingUserEnvironment
from taira_mt.benchmark.language_renderer import FrozenBenchmarkLanguageRenderer
from taira_mt.benchmark.schemas import PublicBenchmarkCase, ScenarioFamily


class FrozenEvolvingUserBenchmark:
    """Verified benchmark bundle that opens fresh, zero-API user sessions.

    The class deliberately exposes only public case metadata.  Private state is
    retained internally by the simulator and can be recovered from an episode
    trace for evaluation after an interaction, rather than handed to the agent.
    """

    def __init__(
        self,
        *,
        benchmark_dir: Path,
        language_bank_dir: Path,
        renderer_seed: int = 20260726,
    ) -> None:
        manifest, cases = load_symbolic_pilot(benchmark_dir)
        renderer = FrozenBenchmarkLanguageRenderer.from_artifact(
            language_bank_dir,
            seed=renderer_seed,
        )
        if renderer.bank.summary.parent_benchmark_build_id != manifest.build_id:
            raise ValueError("language bank and benchmark build IDs differ")
        by_id = {item.pair.public.case_id: item for item in cases}
        if len(by_id) != len(cases):
            raise ValueError("frozen benchmark contains duplicate case IDs")
        self._manifest = manifest
        self._cases = by_id
        self._renderer = renderer

    @classmethod
    def from_config(cls, config_path: Path) -> FrozenEvolvingUserBenchmark:
        """Load an exact benchmark/language pair from a release freeze file."""

        config_path = config_path.resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("benchmark freeze config must contain a mapping")
        benchmark = _required_mapping(payload, "benchmark")
        language = _required_mapping(payload, "language_bank")
        runtime = _required_mapping(payload, "runtime")
        benchmark_dir = _artifact_path(config_path, benchmark)
        language_bank_dir = _artifact_path(config_path, language)
        _verify_manifest_hash(benchmark_dir, benchmark)
        _verify_manifest_hash(language_bank_dir, language)
        suite = cls(
            benchmark_dir=benchmark_dir,
            language_bank_dir=language_bank_dir,
            renderer_seed=int(runtime["renderer_seed"]),
        )
        if suite.benchmark_build_id != benchmark.get("build_id"):
            raise ValueError("loaded benchmark does not match freeze build_id")
        if suite.language_bank_id != language.get("bank_id"):
            raise ValueError("loaded language bank does not match freeze bank_id")
        expected_renderer_id = runtime.get("renderer_id")
        if expected_renderer_id and suite.renderer_id != expected_renderer_id:
            raise ValueError("loaded renderer does not match freeze renderer_id")
        return suite

    @property
    def benchmark_build_id(self) -> str:
        return self._manifest.build_id

    @property
    def language_bank_id(self) -> str:
        return self._renderer.bank.summary.bank_id

    @property
    def renderer_id(self) -> str:
        return self._renderer.renderer_id

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._cases))

    def public_case(self, case_id: str) -> PublicBenchmarkCase:
        return self._loaded_case(case_id).pair.public.model_copy(deep=True)

    def select_case_ids(
        self,
        *,
        domain: str | None = None,
        scenario_family: ScenarioFamily | str | None = None,
    ) -> tuple[str, ...]:
        scenario = (
            ScenarioFamily(scenario_family)
            if isinstance(scenario_family, str)
            else scenario_family
        )
        return tuple(
            case_id
            for case_id in self.case_ids
            if (domain is None or self._cases[case_id].pair.public.domain == domain)
            and (
                scenario is None
                or self._cases[case_id].pair.private.initial_state.scenario_family
                == scenario
            )
        )

    def open_session(self, case_id: str) -> DeterministicEvolvingUserEnvironment:
        """Return a fresh branchable session with deterministic natural language."""

        loaded = self._loaded_case(case_id)
        return DeterministicEvolvingUserEnvironment(
            loaded.pair,
            loaded.candidate_pool,
            language_renderer=self._renderer,
        )

    def open_agent_session(self, case_id: str):
        """Return an Agent-safe facade that cannot be mistaken for evaluator state."""

        from taira_mt.agent.session import PublicAgentSession

        loaded = self._loaded_case(case_id)
        environment = DeterministicEvolvingUserEnvironment(
            loaded.pair,
            loaded.candidate_pool,
            language_renderer=self._renderer,
        )
        return PublicAgentSession(environment=environment, public_case=loaded.pair.public)

    def _loaded_case(self, case_id: str) -> LoadedPilotCase:
        try:
            return self._cases[case_id]
        except KeyError as exc:
            raise KeyError(f"unknown frozen benchmark case: {case_id}") from exc


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"benchmark freeze config is missing mapping: {key}")
    return value


def _artifact_path(config_path: Path, descriptor: dict[str, Any]) -> Path:
    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("artifact descriptor requires a non-empty path")
    path = (config_path.parent / raw_path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"frozen artifact directory not found: {path}")
    return path


def _verify_manifest_hash(artifact_dir: Path, descriptor: dict[str, Any]) -> None:
    expected = descriptor.get("manifest_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("artifact descriptor requires manifest_sha256")
    actual = hashlib.sha256((artifact_dir / "manifest.json").read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"frozen manifest hash mismatch: {artifact_dir}")
