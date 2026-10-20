"""Run paper-canonical TAIRAS baselines on the frozen full-catalog test set."""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from taira_mt.mt_agent.ablations import (
    CoarsePublicStateEstimator,
    FieldMaskedDSUSEstimator,
    OpeningFrozenDSUSEstimator,
    OpeningFrozenSCTPCompiler,
    mismatched_pattern_library,
)
from taira_mt.mt_agent.baselines import (
    AskFirstLexicalPolicy,
    LegacyTAIRALLMPolicy,
    OneShotLexicalPolicy,
    ReactiveLexicalPolicy,
    StaticTAIRAPolicy,
)
from taira_mt.mt_agent.civ import CIVLLMRanker, CIVPairwiseLinearRanker, CIVRanker, SOPFirstRanker
from taira_mt.mt_agent.direct_llm import DirectLLMToolPolicy, ReActLLMToolPolicy
from taira_mt.mt_agent.dsus_prior import DSUSInitialStatePrior
from taira_mt.mt_agent.executor import THOUGHT_PATTERN_EXECUTOR_VERSION, LLMThoughtPatternExecutor
from taira_mt.mt_agent.literature_baselines import (
    ChatCRSGoalGuidedPolicy,
    ItemCenteredIncrementalPolicy,
    LLMAskThenRecommendPolicy,
    RecMindSelfInspiringPolicy,
)
from taira_mt.mt_agent.proposed import TAIRAMTPolicy
from taira_mt.mt_agent.prototypes import StatePrototypeBank
from taira_mt.mt_agent.runner import MTAgentRunner, MultiDomainRetrieverRegistry
from taira_mt.mt_agent.sctp import SCTPSkillCompiler
from taira_mt.mt_agent.skills import ThoughtPatternLibrary
from taira_mt.mt_agent.state import DSUS_LLM_CALIBRATION_VERSION, LLMDSUSEstimator, RuleDSUSEstimator
from taira_mt.mt_agent.state_compiler_adapter import StateCompilerDSUSEstimator
from taira_mt.mt_agent.state_conditioned_direct import StateConditionedDirectPolicy
from taira_mt.multiturn_benchmark.runtime import FrozenMultiTurnBenchmark
from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.providers.openai_compatible import OpenAICompatibleTransportProfile
from taira_mt.state_compiler.runtime import (
    DSUSStateCompilerRuntime,
    build_state_compiler_backend,
    load_state_compiler_profile,
)

BASELINE_ROLLOUT_VERSION = "mt_agent_baseline_rollout_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a shared-tool TAIRAS baseline on frozen full-catalog cases.")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("data/benchmark/episodes.jsonl"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/catalogs"))
    parser.add_argument("--retriever-root", type=Path, default=Path("artifacts/mt_catalog_indexes_v1"))
    parser.add_argument(
        "--policy",
        choices=(
            "one-shot",
            "reactive",
            "ask-first",
            "ask-first-llm",
            "static-taira",
            "direct-llm",
            "unstructured-tool-llm",
            "react-llm",
            "chatcrs-llm",
            "item-centered-llm",
            "legacy-taira-llm",
            "recmind-llm",
            "taira-mt-rule",
            "taira-mt-llm",
            "taira-mt-llm-civ",
            "taira-mt-compiler",
        ),
        required=True,
    )
    parser.add_argument("--maximum-turns", type=int, default=2)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional deterministic mechanism/domain-balanced prefix; use for API pilots.",
    )
    parser.add_argument(
        "--episode-manifest",
        type=Path,
        default=None,
        help="Optional public JSONL or one-ID-per-line manifest of the exact episode set.",
    )
    parser.add_argument("--run-label", default="default")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/mt_agent_baseline_rollouts_v1"))
    parser.add_argument(
        "--state-prototype-bank",
        type=Path,
        default=None,
        help="Optional source-disjoint DSUS development prototype bank for an LLM-backed TAIRAS policy.",
    )
    parser.add_argument(
        "--dsus-prior",
        type=Path,
        default=None,
        help="Optional source-disjoint learned initial-state prior for an LLM-backed TAIRAS policy.",
    )
    parser.add_argument(
        "--dsus-catalog-preview",
        action="store_true",
        help="Expose a public full-catalog Top-5 preview to DSUS; off by default after development gating.",
    )
    parser.add_argument(
        "--state-compiler-profile",
        type=Path,
        default=None,
        help=(
            "Secret-free DSUS v2 runtime profile. Required only by taira-mt-compiler; "
            "switch this file to compare Luna/API and local Qwen LoRA under identical downstream modules."
        ),
    )
    parser.add_argument(
        "--civ-model",
        type=Path,
        default=None,
        help="Optional source-disjoint pairwise CIV model for TAIRAS.",
    )
    parser.add_argument(
        "--pattern-library",
        type=Path,
        default=None,
        help="Optional pinned Thought Pattern Skill repository checkpoint.",
    )
    parser.add_argument(
        "--dsus-mode",
        choices=(
            "dynamic",
            "coarse",
            "opening-frozen",
            "mask-feedback",
            "mask-unresolved-axis",
            "mask-readiness",
        ),
        default="dynamic",
        help="Matched control: update DSUS after observations or reuse the opening state.",
    )
    parser.add_argument(
        "--pattern-mode",
        choices=("state-rematched", "opening-frozen", "mismatched", "none-state-direct"),
        default="state-rematched",
        help="Matched control for observation-boundary rematching and Pattern-content specificity.",
    )
    parser.add_argument(
        "--selector-mode",
        choices=("auto", "learned", "heuristic", "sop-first"),
        default="auto",
        help="Value selector over the identical legal Pattern frontier.",
    )
    parser.add_argument(
        "--executor-mode",
        choices=("llm", "deterministic"),
        default="llm",
        help="Use the constrained LLM Pattern realizer or the compiled typed action directly.",
    )

    # Direct LLM route arguments. Values are environment-variable names, never credentials.
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider-profile-id", default=None)
    parser.add_argument("--api-key-env", default="TAIRA_MT_API_KEY")
    parser.add_argument("--base-url-env", default="TAIRA_MT_BASE_URL")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=900)
    parser.add_argument(
        "--completion-token-parameter",
        choices=("max_tokens", "max_completion_tokens"),
        default="max_tokens",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Optional OpenAI-compatible reasoning effort, e.g. low; omitted for providers that do not support it.",
    )
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--allow-large-api-run", action="store_true")
    return parser.parse_args()


def _balanced_selection(manifest: list[dict[str, str]], limit: int | None) -> list[dict[str, str]]:
    if limit is None or limit >= len(manifest):
        return manifest
    if limit < 1:
        raise ValueError("max-cases must be positive when provided")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        grouped[(row["mechanism"], row["domain"])].append(row)
    selected: list[dict[str, str]] = []
    mechanisms = sorted({mechanism for mechanism, _ in grouped})
    domains = sorted({domain for _, domain in grouped})
    consumed: dict[tuple[str, str], int] = defaultdict(int)
    round_index = 0
    while len(selected) < limit:
        emitted = False
        # The first small pilot round covers every interaction mechanism once,
        # while rotating domains. Subsequent rounds keep the same deterministic
        # mechanism/domain balance instead of exhausting one mechanism first.
        for mechanism_index, mechanism in enumerate(mechanisms):
            domain = domains[(round_index + mechanism_index) % len(domains)]
            key = (mechanism, domain)
            rows = grouped[key]
            position = consumed[key]
            if position < len(rows) and len(selected) < limit:
                selected.append(rows[position])
                consumed[key] += 1
                emitted = True
        if not emitted:
            break
        round_index += 1
    return selected


def _manifest_selection(
    manifest: list[dict[str, str]],
    episode_manifest: Path,
) -> list[dict[str, str]]:
    requested_rows: list[dict[str, str]] = []
    for line in episode_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            value = {"episode_id": line.strip()}
        if isinstance(value, str):
            value = {"episode_id": value}
        if not isinstance(value, dict) or not isinstance(value.get("episode_id"), str):
            raise ValueError("episode manifest lines must be JSON objects/strings or plain episode IDs")
        requested_rows.append({"episode_id": value["episode_id"]})
    requested_ids = [str(row["episode_id"]) for row in requested_rows]
    if not requested_ids or len(requested_ids) != len(set(requested_ids)):
        raise ValueError("episode manifest must contain unique episode IDs")
    by_id = {row["episode_id"]: row for row in manifest}
    missing = [episode_id for episode_id in requested_ids if episode_id not in by_id]
    if missing:
        raise ValueError(f"episode manifest contains unknown IDs: {missing[:3]}")
    return [by_id[episode_id] for episode_id in requested_ids]


def _build_policy(args: argparse.Namespace):
    if args.policy == "one-shot":
        return OneShotLexicalPolicy(), {"runtime_external_calls": False}
    if args.policy == "reactive":
        return ReactiveLexicalPolicy(), {"runtime_external_calls": False}
    if args.policy == "ask-first":
        return AskFirstLexicalPolicy(), {"runtime_external_calls": False}
    if args.policy == "static-taira":
        library = ThoughtPatternLibrary.default()
        return (
            StaticTAIRAPolicy(library=library),
            {
                "runtime_external_calls": False,
                "thought_pattern_library_id": library.library_id,
                "state_updates": False,
                "civ_model": None,
            },
        )
    if args.policy == "taira-mt-rule":
        library = ThoughtPatternLibrary.default()
        return (
            TAIRAMTPolicy(
                dsus=RuleDSUSEstimator(),
                sctp=SCTPSkillCompiler(library=library),
                civ=CIVRanker(),
            ),
            {
                "runtime_external_calls": False,
                "thought_pattern_library_id": library.library_id,
                "dsus_estimator": "rule_dsus_public_v1",
                "civ_model": "civ_seeded_public_value_prior_v1",
            },
        )
    if not args.model:
        raise ValueError("--model is required for an LLM-backed policy")
    profile = OpenAICompatibleTransportProfile(
        provider_profile_id=args.provider_profile_id or f"mt-direct:{args.model}",
        model=args.model,
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        completion_token_parameter=args.completion_token_parameter,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=False if args.disable_thinking else None,
    )
    # Reuse the secret-free adapter; the endpoint/key enter process memory only here.
    from taira_mt.agent.direct_llm import DirectOpenAICompatiblePlanner

    planner = DirectOpenAICompatiblePlanner.from_environment(profile)
    route_telemetry = {
        "runtime_external_calls": True,
        "provider_profile": profile.secret_free_telemetry(),
        "planner": planner.public_telemetry(),
    }
    if args.policy in {"direct-llm", "unstructured-tool-llm"}:
        return (
            DirectLLMToolPolicy(
                transport=planner,
                planner_id=planner.planner_id,
                max_attempts=args.max_attempts,
            ),
            route_telemetry,
        )
    if args.policy == "react-llm":
        return (
            ReActLLMToolPolicy(
                transport=planner,
                planner_id=planner.planner_id,
                max_attempts=args.max_attempts,
            ),
            route_telemetry,
        )
    if args.policy == "ask-first-llm":
        return (
            LLMAskThenRecommendPolicy(
                transport=planner,
                planner_id=planner.planner_id,
                max_attempts=args.max_attempts,
            ),
            {**route_telemetry, "fixed_timing_baseline": "LLM Ask-then-Recommend"},
        )
    if args.policy == "chatcrs-llm":
        return (
            ChatCRSGoalGuidedPolicy(
                transport=planner,
                planner_id=planner.planner_id,
                max_attempts=args.max_attempts,
            ),
            {**route_telemetry, "adapted_baseline": "ChatCRS"},
        )
    if args.policy == "item-centered-llm":
        return (
            ItemCenteredIncrementalPolicy(
                transport=planner,
                planner_id=planner.planner_id,
                max_attempts=args.max_attempts,
            ),
            {**route_telemetry, "adapted_baseline": "A Flash in the Pan"},
        )
    if args.policy == "legacy-taira-llm":
        library = ThoughtPatternLibrary.default()
        executor = LLMThoughtPatternExecutor(
            transport=planner,
            library=library,
            planner_id=planner.planner_id,
            max_attempts=args.max_attempts,
            enable_dialogue_view=False,
        )
        return (
            LegacyTAIRALLMPolicy(
                executor=executor,
                library=library,
            ),
            {
                **route_telemetry,
                "continuity_baseline": "Legacy TAIRA recommend-first with visible-feedback replan",
                "thought_pattern_library_id": library.library_id,
                "thought_pattern_executor": executor.executor_id.split(":", maxsplit=1)[0],
            },
        )
    if args.policy == "recmind-llm":
        return (
            RecMindSelfInspiringPolicy(
                transport=planner,
                planner_id=planner.planner_id,
                max_attempts=args.max_attempts,
            ),
            {**route_telemetry, "adapted_baseline": "RecMind"},
        )
    library = (
        ThoughtPatternLibrary.default()
        if args.pattern_library is None
        else ThoughtPatternLibrary.load(args.pattern_library)
    )
    if args.pattern_mode == "mismatched":
        library = mismatched_pattern_library(library)
    prototype_bank = None if args.state_prototype_bank is None else StatePrototypeBank.load(args.state_prototype_bank)
    initial_state_prior = None if args.dsus_prior is None else DSUSInitialStatePrior.load(args.dsus_prior)
    if args.selector_mode == "sop-first":
        civ = SOPFirstRanker()
    elif args.selector_mode == "heuristic":
        civ = CIVRanker()
    elif args.selector_mode == "learned":
        if args.civ_model is None:
            raise ValueError("--selector-mode learned requires --civ-model")
        civ = CIVPairwiseLinearRanker.load(args.civ_model)
    elif args.policy == "taira-mt-llm-civ":
        civ = CIVLLMRanker(
            transport=planner,
            model_id=f"llm_civ::{planner.planner_id}",
            max_attempts=args.max_attempts,
        )
    elif args.civ_model is not None:
        civ = CIVPairwiseLinearRanker.load(args.civ_model)
    else:
        civ = CIVRanker()
    compiler_profile_telemetry = None
    if args.policy == "taira-mt-compiler":
        if args.state_compiler_profile is None:
            raise ValueError("taira-mt-compiler requires --state-compiler-profile")
        compiler_profile, compiler_profile_base = load_state_compiler_profile(
            args.state_compiler_profile
        )
        compiler_backend = build_state_compiler_backend(
            compiler_profile,
            profile_base_dir=compiler_profile_base,
        )
        dsus = StateCompilerDSUSEstimator(
            runtime=DSUSStateCompilerRuntime(
                compiler_id=compiler_profile.compiler_id,
                backend=compiler_backend,
                max_attempts=compiler_profile.max_attempts,
            ),
            fallback_to_rule=compiler_profile.fallback_to_rule,
        )
        compiler_profile_telemetry = compiler_profile.secret_free_telemetry()
    else:
        if args.state_compiler_profile is not None:
            raise ValueError("--state-compiler-profile is supported only by taira-mt-compiler")
        dsus = LLMDSUSEstimator(
            transport=planner,
            estimator_id=f"{DSUS_LLM_CALIBRATION_VERSION}::{planner.planner_id}",
            max_attempts=args.max_attempts,
            prototype_bank=prototype_bank,
            initial_state_prior=initial_state_prior,
            include_catalog_preview=args.dsus_catalog_preview,
        )
    if args.dsus_mode == "opening-frozen":
        dsus = OpeningFrozenDSUSEstimator(dsus)
    elif args.dsus_mode == "coarse":
        dsus = CoarsePublicStateEstimator(dsus)
    elif args.dsus_mode.startswith("mask-"):
        dsus = FieldMaskedDSUSEstimator(
            dsus,
            factor=args.dsus_mode.removeprefix("mask-"),
        )
    if args.pattern_mode == "none-state-direct":
        if args.policy != "taira-mt-compiler":
            raise ValueError("none-state-direct requires the same learned state compiler")
        return (
            StateConditionedDirectPolicy(
                dsus=dsus, transport=planner, planner_id=planner.planner_id,
                max_attempts=args.max_attempts, shortlist_k=50,
            ),
            {**route_telemetry, "state_compiler_profile": compiler_profile_telemetry,
             "pattern_mode": args.pattern_mode, "dsus_mode": args.dsus_mode,
             "architecture_control": "same_dsus_free_planning_without_pattern_frontier_or_civ"},
        )
    sctp = SCTPSkillCompiler(
        library=library,
        include_initial_counterfactuals=args.policy
        in {"taira-mt-llm", "taira-mt-llm-civ", "taira-mt-compiler"},
    )
    if args.pattern_mode == "opening-frozen":
        sctp = OpeningFrozenSCTPCompiler(sctp)
    executor = (
        None
        if args.executor_mode == "deterministic"
        else LLMThoughtPatternExecutor(
            transport=planner,
            library=library,
            planner_id=planner.planner_id,
            max_attempts=args.max_attempts,
        )
    )
    return (
        TAIRAMTPolicy(
            dsus=dsus,
            sctp=sctp,
            civ=civ,
            executor=executor,
        ),
        {
            **route_telemetry,
            "thought_pattern_library_id": library.library_id,
            "civ_model": civ.model_id,
            "state_prototype_bank_id": None if prototype_bank is None else prototype_bank.bank_id,
            "initial_state_prior_id": None if initial_state_prior is None else initial_state_prior.model_id,
            "dsus_catalog_preview": args.dsus_catalog_preview,
            "dsus_mode": args.dsus_mode,
            "pattern_mode": args.pattern_mode,
            "selector_mode": args.selector_mode,
            "executor_mode": args.executor_mode,
            "pattern_library_source": library.source,
            "thought_pattern_executor": (
                "deterministic_executor" if executor is None else THOUGHT_PATTERN_EXECUTOR_VERSION
            ),
            "dsus_state_compiler_profile": compiler_profile_telemetry,
        },
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty rollout")
    success = [bool(row["metrics"]["success"]) for row in rows]
    success_at_1 = [bool(row["metrics"]["success_at_1"]) for row in rows]
    success_at_3 = [bool(row["metrics"]["success_at_3"]) for row in rows]
    success_at_10 = [bool(row["metrics"]["success_at_10"]) for row in rows]
    turns = [int(row["metrics"]["agent_turn_count"]) for row in rows]
    unnecessary = [int(row["metrics"]["unnecessary_action_count"]) for row in rows]
    utilities = [float(row["metrics"]["interaction_utility"]) for row in rows]
    rejected = [int(row["metrics"]["rejected_recommendation_count"]) for row in rows]
    targeted = [int(row["metrics"]["targeted_question_count"]) for row in rows]
    ask_count = [int(row["metrics"]["ask_count"]) for row in rows]
    return {
        "case_count": len(rows),
        "success_rate": mean(float(value) for value in success),
        "success_at_1_rate": mean(float(value) for value in success_at_1),
        "success_at_3_rate": mean(float(value) for value in success_at_3),
        "success_at_10_rate": mean(float(value) for value in success_at_10),
        "mean_agent_turn_count": mean(turns),
        "mean_unnecessary_action_count": mean(unnecessary),
        "mean_rejected_recommendation_count": mean(rejected),
        "mean_interaction_utility": mean(utilities),
        "targeted_question_rate": (sum(targeted) / sum(ask_count)) if sum(ask_count) else None,
    }


def _grouped_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {group: _summary(groups[group]) for group in sorted(groups)}


def _usage(rows: list[dict[str, Any]]) -> dict[str, int]:
    total = {"attempt_count": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for row in rows:
        for turn in row["episode"]["turns"]:
            for audit in turn["decision"]["llm_audits"]:
                total["attempt_count"] += 1
                total["input_tokens"] += audit["input_tokens"] or 0
                total["output_tokens"] += audit["output_tokens"] or 0
                total["total_tokens"] += audit["total_tokens"] or 0
    return total


def main() -> None:
    args = parse_args()
    if args.maximum_turns < 1:
        raise ValueError("maximum-turns must be positive")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be positive")
    if not args.run_label.strip():
        raise ValueError("run-label cannot be blank")
    ablation_requested = (
        args.pattern_library is not None
        or args.dsus_mode != "dynamic"
        or args.pattern_mode != "state-rematched"
        or args.selector_mode != "auto"
        or args.executor_mode != "llm"
    )
    if ablation_requested and args.policy not in {
        "taira-mt-llm",
        "taira-mt-llm-civ",
        "taira-mt-compiler",
    }:
        raise ValueError("matched ablation flags are supported only for LLM-backed TAIRAS policies")

    print("Step 1/5: loading the frozen test benchmark and public manifest.", flush=True)
    benchmark = FrozenMultiTurnBenchmark.from_artifact(args.artifact, args.data_root)
    if args.episode_manifest is not None and args.max_cases is not None:
        raise ValueError("--episode-manifest and --max-cases are mutually exclusive")
    public_manifest = benchmark.public_manifest()
    selected = (
        _balanced_selection(public_manifest, args.max_cases)
        if args.episode_manifest is None
        else _manifest_selection(public_manifest, args.episode_manifest)
    )
    llm_backed_policy = args.policy in {
        "direct-llm",
        "unstructured-tool-llm",
        "react-llm",
        "ask-first-llm",
        "chatcrs-llm",
        "item-centered-llm",
        "legacy-taira-llm",
        "recmind-llm",
        "taira-mt-llm",
        "taira-mt-llm-civ",
    }
    if llm_backed_policy and len(selected) > 20 and not args.allow_large_api_run:
        raise RuntimeError(
            "Direct LLM is limited to 20 development/pilot cases by default; "
            "add --allow-large-api-run explicitly after confirming a large API run."
        )
    print(f"Step 1/5 complete: {len(selected)} public cases selected for this run.", flush=True)

    print("Step 2/5: creating the shared full-catalog retriever and the selected baseline.", flush=True)
    policy, route_telemetry = _build_policy(args)
    print(f"Step 2/5 complete: policy={policy.policy_id}; all methods use the same full-catalog retrieval interface.", flush=True)

    print("Step 3/5: running public agent rollouts; evaluator metrics are read only after each rollout completes.", flush=True)
    rows: list[dict[str, Any]] = []
    selected_by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)
    for public_case in selected:
        selected_by_domain[public_case["domain"]].append(public_case)
    # A full Amazon domain is intentionally large.  Load the evaluator catalog
    # and its shared retriever one domain at a time, then release both before
    # moving on; this preserves the exact tool surface while avoiding a
    # three-domain double-residency peak in a single rollout process.
    completed = 0
    del benchmark
    for domain in sorted(selected_by_domain):
        print(f"Stage start: loading the full catalog for {domain} and running {len(selected_by_domain[domain])} cases.", flush=True)
        domain_benchmark = FrozenMultiTurnBenchmark.from_artifact(args.artifact, args.data_root)
        domain_runner = MTAgentRunner(retrievers=MultiDomainRetrieverRegistry(args.retriever_root))
        for public_case in selected_by_domain[domain]:
            session = domain_benchmark.session(public_case["episode_id"])
            episode = domain_runner.run(session, policy=policy, maximum_turns=args.maximum_turns)
            rows.append(
                {
                    **public_case,
                    "episode": episode.model_dump(mode="json"),
                    "metrics": session.evaluation_summary(),
                }
            )
            completed += 1
            if completed == len(selected) or completed % 25 == 0:
                print(f"Stage complete: finished {completed}/{len(selected)} rollouts.", flush=True)
        del domain_runner
        del domain_benchmark
        gc.collect()
        print(f"Stage complete: released the runtime catalog and index for {domain}.", flush=True)

    print("Step 4/5: summarizing path success, turn counts, invalid interactions, and LLM token telemetry.", flush=True)
    config = {
        "version": BASELINE_ROLLOUT_VERSION,
        "run_label": args.run_label,
        "policy": args.policy,
        "maximum_turns": args.maximum_turns,
        "episode_manifest": None if args.episode_manifest is None else str(args.episode_manifest),
        "selected_episode_ids": [row["episode_id"] for row in selected],
        "artifact": str(args.artifact),
        "retriever_root": str(args.retriever_root),
        "dsus_mode": args.dsus_mode,
        "pattern_mode": args.pattern_mode,
        "selector_mode": args.selector_mode,
        "executor_mode": args.executor_mode,
        "pattern_library": None if args.pattern_library is None else str(args.pattern_library),
        "route_telemetry": route_telemetry,
    }
    run_id = f"{args.policy}-{content_sha256(config)[:16]}"
    summary = {
        "run_id": run_id,
        "config": config,
        "overall": _summary(rows),
        "by_mechanism": _grouped_summary(rows, "mechanism"),
        "by_domain": _grouped_summary(rows, "domain"),
        "llm_usage": _usage(rows),
    }
    print(
        f"Step 4/5 complete: success={summary['overall']['success_rate']:.4f}, "
        f"mean turns={summary['overall']['mean_agent_turn_count']:.3f}.",
        flush=True,
    )

    print("Step 5/5: saving auditable public trajectories and evaluator metrics.", flush=True)
    output_dir = args.output_root / run_id
    if output_dir.exists():
        raise FileExistsError(f"run artifact already exists: {output_dir}; choose a new --run-label")
    output_dir.mkdir(parents=True)
    public_rows = [
        {
            "episode_id": row["episode_id"],
            "domain": row["domain"],
            "mechanism": row["mechanism"],
            "initial_query": row["initial_query"],
            "episode": row["episode"],
        }
        for row in rows
    ]
    evaluator_rows = [
        {
            "episode_id": row["episode_id"],
            "domain": row["domain"],
            "mechanism": row["mechanism"],
            "metrics": row["metrics"],
        }
        for row in rows
    ]
    (output_dir / "episodes.public.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in public_rows), encoding="utf-8"
    )
    (output_dir / "metrics.evaluator.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in evaluator_rows), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
    print(f"Complete: baseline artifacts saved to {output_dir.resolve()}.", flush=True)


if __name__ == "__main__":
    main()
