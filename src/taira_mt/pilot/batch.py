"""Batch execution, projection, auditing, and manifests for the offline pilot."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taira_mt.counterfactual import CounterfactualBranchEvaluator, CounterfactualRewardConfig
from taira_mt.counterfactual.schemas import CounterfactualDecisionLabel, CounterfactualPublicState
from taira_mt.dialogue_policy import PatternConditionedPolicy
from taira_mt.dialogue_policy.cost_aware import PatternActionInstantiator
from taira_mt.models import DialogueEpisode
from taira_mt.pilot.construction import (
    DEFAULT_ANCHOR_SPECS,
    AnchorSpec,
    BuiltPilotCase,
    build_domain_cases,
)
from taira_mt.pilot.schemas import (
    OutputFileRecord,
    PatternDecisionTrace,
    PilotFailure,
    PilotRunManifest,
    PilotScenario,
    PilotSummary,
    PrivateEpisode,
    PrivateTurnRecord,
    PublicEpisode,
    PublicTurnRecord,
    SourceFileRecord,
)
from taira_mt.provenance import canonical_json, content_sha256, source_tree_sha256
from taira_mt.recommender import IndependentCompatibilityOracle
from taira_mt.runner import run_episode
from taira_mt.simulator import DeterministicUserSimulator
from taira_mt.state import RuleBasedObservationBuilder
from taira_mt.thought_patterns import ThoughtPatternLibrary, default_pattern_library

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class PilotRunResult:
    output_dir: Path
    manifest: PilotRunManifest
    summary: PilotSummary
    public_episodes: tuple[PublicEpisode, ...]
    private_episodes: tuple[PrivateEpisode, ...]


def derive_episode_seed(root_seed: int, domain: str, case_id: str, replicate_index: int = 0) -> int:
    payload = f"{root_seed}|{domain}|{case_id}|{replicate_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def project_episode_views(
    episode: DialogueEpisode,
    case_id: str,
    patterns: ThoughtPatternLibrary,
    trajectory_id: str | None = None,
) -> tuple[PublicEpisode, PrivateEpisode]:
    public_turns: list[PublicTurnRecord] = []
    private_turns: list[PrivateTurnRecord] = []
    for turn in episode.turns:
        matched = patterns.get(turn.matched_thought_pattern_id)
        applicable = patterns.applicable(turn.belief_before)
        decision = PatternDecisionTrace(
            pattern_id=matched.pattern_id,
            pattern_version=matched.version,
            pattern_library_hash=patterns.content_hash,
            belief_before_hash=content_sha256(turn.belief_before.model_dump(mode="json")),
            applicable_pattern_ids=[pattern.pattern_id for pattern in applicable],
            selected_action_type=turn.system_action.action_type.value,
            instantiated_information_target=turn.system_action.question_slot,
        )
        public_turn = PublicTurnRecord(
            turn_index=turn.turn_index,
            belief_before=turn.belief_before,
            pattern_decision=decision,
            system_action=turn.system_action,
            public_user_text=turn.public_user_text,
            policy_observation=turn.policy_observation,
            belief_after=turn.belief_after,
        )
        public_turns.append(public_turn)
        private_turns.append(
            PrivateTurnRecord(
                turn_index=turn.turn_index,
                public_record_hash=content_sha256(public_turn.model_dump(mode="json")),
                reward=turn.reward,
                evaluation_trace=turn.evaluation_trace,
            )
        )
    public = PublicEpisode(
        episode_id=episode.episode_id,
        seed=episode.seed,
        case_id=case_id,
        trajectory_id=trajectory_id,
        domain=episode.domain,
        public_initial_request=episode.public_initial_request,
        candidate_pool_id=episode.candidate_pool_id,
        pattern_library_hash=patterns.content_hash,
        turns=public_turns,
    )
    private = PrivateEpisode(
        episode_id=episode.episode_id,
        case_id=case_id,
        trajectory_id=trajectory_id,
        goal_id=episode.goal_id,
        persona_id=episode.persona_id,
        turns=private_turns,
        termination_reason=episode.termination_reason,
        success=episode.success,
        metrics=episode.metrics,
    )
    return public, private


def _walk_keys(payload: Any) -> Iterable[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield str(key)
            yield from _walk_keys(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _walk_keys(value)


def audit_public_private_pair(
    case: BuiltPilotCase,
    public: PublicEpisode,
    private: PrivateEpisode,
    patterns: ThoughtPatternLibrary,
) -> None:
    """Fail closed if private construction/evaluator state enters a public record."""

    public_payload = public.model_dump(mode="json")
    public_text = canonical_json(public_payload).casefold()
    forbidden_keys = {
        "goal_id",
        "persona_id",
        "reward",
        "evaluation_trace",
        "oracle_assessments",
        "latent_satisfaction",
        "true_user_act",
        "true_revealed_constraints",
        "success",
    }
    leaked_keys = forbidden_keys.intersection(_walk_keys(public_payload))
    if leaked_keys:
        raise AssertionError(f"private field names leaked into public episode: {sorted(leaked_keys)}")

    provenance = case.private_case.provenance
    for sentinel_name, sentinel in {
        "anchor item ID": provenance.anchor_item_id,
        "anchor title": provenance.anchor_title,
        "raw user ID": provenance.raw_user_id,
    }.items():
        if sentinel and sentinel.casefold() in public_text:
            raise AssertionError(f"{sentinel_name} leaked into public episode")

    if case.public_case.scenario != PilotScenario.READY:
        hidden = str(case.private_case.hidden_goal.constraint_map()["specific_category"].value).casefold()
        if hidden in public.public_initial_request.casefold():
            raise AssertionError("hidden specific category leaked into the initial request")
        if "specific_category" in public.turns[0].belief_before.known_constraints:
            raise AssertionError("hidden specific category leaked into the initial belief")

    if public.episode_id != private.episode_id or len(public.turns) != len(private.turns):
        raise AssertionError("public/private episode linkage is inconsistent")
    for public_turn, private_turn in zip(public.turns, private.turns, strict=True):
        if public_turn.turn_index != private_turn.turn_index:
            raise AssertionError("public/private turn indices do not match")
        expected_hash = content_sha256(public_turn.model_dump(mode="json"))
        if private_turn.public_record_hash != expected_hash:
            raise AssertionError("private turn references the wrong public record")
        rematched = patterns.match(public_turn.belief_before)
        if rematched.pattern_id != public_turn.pattern_decision.pattern_id:
            raise AssertionError("saved Thought Pattern does not match deterministic replay")
        if public_turn.system_action.thought_pattern_id != rematched.pattern_id:
            raise AssertionError("system action is not controlled by the matched Thought Pattern")
        if public_turn.system_action.action_type != rematched.preferred_action:
            raise AssertionError("system action type violates the matched Thought Pattern")
        if any(item_id not in case.candidate_pool.item_ids for item_id in public_turn.system_action.item_ids):
            raise AssertionError("system action recommends an item outside the candidate pool")

    if private.success:
        accepted_ids = [
            turn.evaluation_trace.accepted_item_id
            for turn in private.turns
            if turn.evaluation_trace.accepted_item_id is not None
        ]
        if len(accepted_ids) != 1 or accepted_ids[0] not in case.private_case.acceptable_item_ids:
            raise AssertionError("successful episode did not accept a member of the set-valued goal")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[Any]) -> None:
    lines = []
    for record in records:
        payload = record.model_dump(mode="json") if hasattr(record, "model_dump") else record
        lines.append(canonical_json(payload))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _file_record(path: Path, relative_to: Path) -> OutputFileRecord:
    content = path.read_bytes()
    return OutputFileRecord(
        relative_path=str(path.relative_to(relative_to)),
        sha256=hashlib.sha256(content).hexdigest(),
        line_count=len(content.splitlines()),
    )


def _run_config(
    root_seed: int,
    specs: tuple[AnchorSpec, ...],
    pattern_hash: str,
    sources: list[SourceFileRecord],
    counterfactual_config: CounterfactualRewardConfig,
) -> dict[str, Any]:
    return {
        "root_seed": root_seed,
        "case_construction": [
            {
                "case_id": spec.case_id,
                "domain": spec.domain,
                "scenario": spec.scenario.value,
                "anchor_item_id": spec.anchor_item_id,
                "visible_tag": spec.visible_tag,
                "hidden_tag": spec.hidden_tag,
            }
            for spec in sorted(specs, key=lambda item: item.case_id)
        ],
        "source_files": [source.model_dump(mode="json") for source in sources],
        "pattern_library_hash": pattern_hash,
        "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parents[3]),
        "simulator": "deterministic_transition_v1",
        "tracker": "rule_based_public_text",
        "counterfactual_reward": counterfactual_config.model_dump(mode="json"),
        "external_calls": False,
    }


def run_offline_pilot(
    legacy_root: Path,
    output_root: Path,
    root_seed: int = 20260714,
    specs: tuple[AnchorSpec, ...] = DEFAULT_ANCHOR_SPECS,
    progress: ProgressCallback | None = None,
) -> PilotRunResult:
    progress = progress or (lambda _: None)
    legacy_root = legacy_root.resolve()
    patterns = default_pattern_library()
    counterfactual_config = CounterfactualRewardConfig()

    progress("Step 1/5: reading frozen catalogs for three domains and constructing target-free candidate sets.")
    cases: list[BuiltPilotCase] = []
    sources: list[SourceFileRecord] = []
    for domain in sorted({spec.domain for spec in specs}):
        domain_cases, domain_sources = build_domain_cases(legacy_root, domain, specs)
        cases.extend(domain_cases)
        sources.extend(domain_sources)
        progress(f"Step 1/5 progress: constructed {len(domain_cases)} deterministic cases for {domain}.")
    cases.sort(key=lambda case: case.public_case.case_id)
    sources.sort(key=lambda source: (source.domain, source.kind, source.relative_path))
    config = _run_config(root_seed, specs, patterns.content_hash, sources, counterfactual_config)
    config_hash = content_sha256(config)
    run_id = f"pilot-{config_hash[:16]}"
    output_dir = output_root.resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    progress(f"Step 1/5 complete: constructed {len(cases)} cases, excluding all anchors and historical items.")

    progress("Step 2/5: matching Thought Patterns at each turn and running multi-turn dialogues.")
    public_episodes: list[PublicEpisode] = []
    private_episodes: list[PrivateEpisode] = []
    failures: list[PilotFailure] = []
    for case in cases:
        case_id = case.public_case.case_id
        seed = derive_episode_seed(root_seed, case.public_case.domain, case_id)
        try:
            policy = PatternConditionedPolicy(patterns, case.candidate_pool, slate_size=2)
            simulator = DeterministicUserSimulator(
                goal=case.private_case.hidden_goal,
                persona=case.private_case.persona,
                catalog_index=case.catalog_index,
                oracle=IndependentCompatibilityOracle(),
                seed=seed,
            )
            episode = run_episode(
                episode_id=f"episode-{case_id}",
                seed=seed,
                initial_belief=case.initial_belief,
                policy=policy,
                simulator=simulator,
                observation_builder=RuleBasedObservationBuilder(case.ontology),
                max_turns=8,
            )
            public, private = project_episode_views(episode, case_id, patterns)
            audit_public_private_pair(case, public, private, patterns)
            public_episodes.append(public)
            private_episodes.append(private)
            actions = "→".join(turn.system_action.action_type.value for turn in public.turns)
            progress(f"Step 2/5 progress: {case_id} complete, action sequence={actions}.")
        except Exception as exc:  # noqa: BLE001 - failures are explicitly persisted for batch audit
            failures.append(PilotFailure(case_id=case_id, error_type=type(exc).__name__, message=str(exc)))
    progress(
        f"Step 2/5 complete: ran {len(public_episodes)} episodes successfully; {len(failures)} failed."
    )

    progress("Step 3/5: generating private Ask/Recommend/Repair counterfactual labels and auditing leakage.")
    counterfactual_public: list[CounterfactualPublicState] = []
    counterfactual_private: list[CounterfactualDecisionLabel] = []
    for case in cases:
        seed = derive_episode_seed(root_seed, case.public_case.domain, case.public_case.case_id)
        simulator = DeterministicUserSimulator(
            goal=case.private_case.hidden_goal,
            persona=case.private_case.persona,
            catalog_index=case.catalog_index,
            oracle=IndependentCompatibilityOracle(),
            seed=seed,
        )
        evaluator = CounterfactualBranchEvaluator(
            patterns,
            PatternActionInstantiator(patterns, case.candidate_pool, slate_size=2),
            RuleBasedObservationBuilder(case.ontology),
            counterfactual_config,
        )
        public_state, private_label = evaluator.evaluate(case.initial_belief, simulator, seed)
        serialized_public = canonical_json(public_state.model_dump(mode="json")).casefold()
        for sentinel in (
            case.private_case.provenance.anchor_item_id,
            case.private_case.provenance.anchor_title,
            case.private_case.provenance.raw_user_id,
        ):
            if sentinel and sentinel.casefold() in serialized_public:
                raise AssertionError("private construction sentinel leaked into counterfactual public features")
        counterfactual_public.append(public_state)
        counterfactual_private.append(private_label)
    if len(public_episodes) != len(private_episodes):
        raise AssertionError("public/private episode counts differ")
    if failures:
        progress("Step 3/5 warning: some cases failed; see failures.jsonl.")
    else:
        progress(
            f"Step 3/5 complete: generated {sum(len(label.action_labels) for label in counterfactual_private)} "
            "counterfactual action labels; target identifier leakage is 0."
        )

    domain_counts = Counter(episode.domain for episode in public_episodes)
    scenario_by_case = {case.public_case.case_id: case.public_case.scenario.value for case in cases}
    scenario_counts = Counter(scenario_by_case[episode.case_id] for episode in public_episodes)
    action_counts = Counter(
        turn.system_action.action_type.value for episode in public_episodes for turn in episode.turns
    )
    pattern_counts = Counter(
        turn.pattern_decision.pattern_id for episode in public_episodes for turn in episode.turns
    )
    success_count = sum(episode.success for episode in private_episodes)
    summary = PilotSummary(
        case_count=len(cases),
        episode_count=len(public_episodes),
        failure_count=len(failures),
        success_count=success_count,
        domain_counts=dict(sorted(domain_counts.items())),
        scenario_counts=dict(sorted(scenario_counts.items())),
        action_counts=dict(sorted(action_counts.items())),
        pattern_counts=dict(sorted(pattern_counts.items())),
        counterfactual_state_count=len(counterfactual_public),
        counterfactual_branch_count=sum(len(label.action_labels) for label in counterfactual_private),
    )

    progress("Step 4/5: saving allowlisted public trajectories, private evaluation trajectories, and content-addressed manifests.")
    core_outputs: list[Path] = []
    config_path = output_dir / "config.private.snapshot.json"
    _write_json(config_path, config)
    core_outputs.append(config_path)
    source_path = output_dir / "source_manifest.json"
    _write_jsonl(source_path, sources)
    core_outputs.append(source_path)
    pattern_path = output_dir / "pattern_library.json"
    _write_jsonl(pattern_path, sorted(patterns.patterns, key=lambda pattern: pattern.pattern_id))
    core_outputs.append(pattern_path)
    pool_path = output_dir / "candidate_pools.jsonl"
    _write_jsonl(pool_path, [case.candidate_pool_record for case in cases])
    core_outputs.append(pool_path)
    public_cases_path = output_dir / "cases.public.jsonl"
    _write_jsonl(public_cases_path, [case.public_case for case in cases])
    core_outputs.append(public_cases_path)
    private_cases_path = output_dir / "cases.private.jsonl"
    _write_jsonl(private_cases_path, [case.private_case for case in cases])
    core_outputs.append(private_cases_path)
    public_episode_path = output_dir / "episodes.public.jsonl"
    _write_jsonl(public_episode_path, public_episodes)
    core_outputs.append(public_episode_path)
    private_episode_path = output_dir / "episodes.private.jsonl"
    _write_jsonl(private_episode_path, private_episodes)
    core_outputs.append(private_episode_path)
    counterfactual_public_path = output_dir / "counterfactual.public.jsonl"
    _write_jsonl(counterfactual_public_path, counterfactual_public)
    core_outputs.append(counterfactual_public_path)
    counterfactual_private_path = output_dir / "counterfactual.private.jsonl"
    _write_jsonl(counterfactual_private_path, counterfactual_private)
    core_outputs.append(counterfactual_private_path)
    failure_path = output_dir / "failures.jsonl"
    _write_jsonl(failure_path, failures)
    core_outputs.append(failure_path)
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary.model_dump(mode="json"))
    core_outputs.append(summary_path)

    output_records = [_file_record(path, output_dir) for path in core_outputs]
    checksum_path = output_dir / "checksums.sha256"
    checksum_lines = [f"{record.sha256}  {record.relative_path}" for record in output_records]
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    output_records.append(_file_record(checksum_path, output_dir))
    manifest = PilotRunManifest(
        run_id=run_id,
        config_hash=config_hash,
        root_seed=root_seed,
        episode_seed_derivation="uint64_be(sha256(root_seed|domain|case_id|replicate_index)[:8])",
        pattern_library_hash=patterns.content_hash,
        source_files=sources,
        output_files=output_records,
        case_count=len(cases),
        episode_count=len(public_episodes),
        failure_count=len(failures),
    )
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    progress(f"Step 4/5 complete: artifacts saved to {output_dir}.")
    progress(
        f"Step 5/5 complete: {summary.success_count}/{summary.episode_count} succeeded, "
        f"external_calls={summary.external_calls}."
    )
    return PilotRunResult(
        output_dir=output_dir,
        manifest=manifest,
        summary=summary,
        public_episodes=tuple(public_episodes),
        private_episodes=tuple(private_episodes),
    )
