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
        raise SystemExit(f"缺少模型配置：请设置 --api-base-url 与环境变量 {args.api_key_env}。")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("步骤 1/4：从老 TAIRA 成型 query 中抽取与 30 条开发集不重叠的固定候选池。")
    sources = select_source_pool(args.data_root, seed=args.seed, pool_per_domain=args.source_pool_per_domain)
    print(f"阶段完成：已固定 {len(sources)} 条候选 source query，尚未生成多轮轨迹。")
    planner = openai_json_caller(args.api_base_url, api_key, args.model, temperature=0.0, max_tokens=2400)
    print("步骤 2/4：由 LLM 对五种多轮机制分别评分；模型不接触全局比例。")
    planned = plan_source_pool(
        sources,
        planner,
        cache_path=args.output_dir / "planning_cache.jsonl",
        batch_size=args.plan_batch_size,
    )
    allocations = allocate_balanced_mechanisms(planned, cases_per_mechanism=args.cases_per_mechanism)
    _write_plan_artifacts(args.output_dir, sources, allocations, args)
    print(f"阶段完成：全局 allocator 已分配 {len(allocations)} 条，不同机制配额已固定。")
    if args.stage == "plan":
        print(f"规划完成：冻结前 manifest 位于 {args.output_dir.resolve()}。")
        return
    print("步骤 3/4：按已固定机制生成紧凑用户状态、一次多轮转移和商品证据。")
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
    print("步骤 4/4：已冻结 Agent runtime episode、来源池、评分缓存与全局分配 manifest。")
    print(f"全部完成：{len(records)} 条冻结测试 episode 已写入 {args.output_dir.resolve()}。")


if __name__ == "__main__":
    main()
