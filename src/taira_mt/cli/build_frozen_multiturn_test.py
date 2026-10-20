"""Build the globally balanced, frozen TAIRAS Agent test benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from taira_mt.multiturn_benchmark.development import attach_catalog_evidence, openai_json_caller
from taira_mt.multiturn_benchmark.frozen_test import (
    FROZEN_TEST_VERSION,
    allocate_balanced_mechanisms,
    generate_test_trajectories,
    make_test_case_inputs,
    plan_source_pool,
    select_source_pool,
    write_frozen_test_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frozen globally balanced multi-turn Agent test benchmark.")
    parser.add_argument("--data-root", type=Path, default=Path("data/catalogs"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/frozen_multiturn_test_v1"))
    parser.add_argument("--api-base-url", default=os.getenv("TAIRA_MT_LLM_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="TAIRA_MT_LLM_API_KEY")
    parser.add_argument("--model", default=os.getenv("TAIRA_MT_AUTHOR_MODEL", "qwen3-max"))
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--source-pool-per-domain", type=int, default=160)
    parser.add_argument("--cases-per-mechanism", type=int, default=20)
    parser.add_argument("--plan-batch-size", type=int, default=8)
    parser.add_argument("--stage", choices=("plan", "all"), default="all")
    return parser.parse_args()


def _write_plan_artifacts(output_dir: Path, sources: list[dict], allocations: list[dict], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_pool.jsonl").write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in sources), encoding="utf-8"
    )
    (output_dir / "allocation_manifest.jsonl").write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in allocations), encoding="utf-8"
    )
    manifest = {
        "version": FROZEN_TEST_VERSION,
        "stage": "planned",
        "seed": args.seed,
        "source_pool_per_domain": args.source_pool_per_domain,
        "cases_per_domain_mechanism": args.cases_per_mechanism,
        "source_count": len(sources),
        "allocated_count": len(allocations),
        "global_allocator": "score_then_exact_quota_greedy_v1",
        "development_source_disjoint": True,
    }
    (output_dir / "planning_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    api_key = os.getenv(args.api_key_env, "").strip()
    if not args.api_base_url or not api_key:
        raise SystemExit(f"Missing model configuration: set --api-base-url and environment variable {args.api_key_env}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Step 1/4: extracting a fixed candidate pool from legacy TAIRA queries, disjoint from the 30 development cases.")
    sources = select_source_pool(args.data_root, seed=args.seed, pool_per_domain=args.source_pool_per_domain)
    print(f"Stage complete: fixed {len(sources)} candidate source queries; multi-turn trajectories have not been generated.")
    planner = openai_json_caller(args.api_base_url, api_key, args.model, temperature=0.0, max_tokens=2400)
    print("Step 2/4: using the LLM to score five multi-turn mechanisms independently; global ratios are hidden from the model.")
    planned = plan_source_pool(
        sources,
        planner,
        cache_path=args.output_dir / "planning_cache.jsonl",
        batch_size=args.plan_batch_size,
    )
    allocations = allocate_balanced_mechanisms(planned, cases_per_mechanism=args.cases_per_mechanism)
    _write_plan_artifacts(args.output_dir, sources, allocations, args)
    print(f"Stage complete: the global allocator assigned {len(allocations)} records; mechanism quotas are fixed.")
    if args.stage == "plan":
        print(f"Planning complete: pre-freeze manifest is at {args.output_dir.resolve()}.")
        return
    print("Step 3/4: generating compact user states, one multi-turn transition, and item evidence for each assigned mechanism.")
    case_inputs = attach_catalog_evidence(args.data_root, make_test_case_inputs(allocations))
    author = openai_json_caller(args.api_base_url, api_key, args.model, temperature=0.0, max_tokens=1400)
    records = generate_test_trajectories(case_inputs, author, cache_path=args.output_dir / "generation_cache.jsonl")
    write_frozen_test_artifacts(
        args.output_dir,
        source_pool=sources,
        allocations=allocations,
        records=records,
        seed=args.seed,
        pool_per_domain=args.source_pool_per_domain,
        cases_per_mechanism=args.cases_per_mechanism,
    )
    print("Step 4/4: froze agent runtime episodes, source pools, score caches, and the global allocation manifest.")
    print(f"Complete: {len(records)} frozen test episodes written to {args.output_dir.resolve()}.")


if __name__ == "__main__":
    main()
