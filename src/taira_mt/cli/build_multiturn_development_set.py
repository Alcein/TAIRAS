"""Construct and quality-gate the fixed 30-case V2 trajectory development set."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from taira_mt.multiturn_benchmark.development import (
    DEVELOPMENT_BLUEPRINTS,
    build_development_cases,
    openai_json_caller,
    render_development_audit,
    write_development_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the quality-gated TAIRAS V2 development set.")
    parser.add_argument("--data-root", type=Path, default=Path("data/catalogs"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/multiturn_development_v2"))
    parser.add_argument("--api-base-url", default=os.getenv("TAIRA_MT_LLM_BASE_URL", ""))
    parser.add_argument("--api-key-env", default="TAIRA_MT_LLM_API_KEY")
    parser.add_argument("--author-model", default=os.getenv("TAIRA_MT_AUTHOR_MODEL", "qwen3-max"))
    parser.add_argument("--judge-model", default=os.getenv("TAIRA_MT_JUDGE_MODEL", "qwen3-max"))
    parser.add_argument("--case-limit", type=int, default=0, help="Use only a prefix for a low-cost quality gate.")
    parser.add_argument("--case-ids", default="", help="Comma-separated fixed case ids for a representative quality gate.")
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--generation-only", action="store_true", help="Generate every runnable trajectory without per-case judge gating.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv(args.api_key_env, "").strip()
    if not args.api_base_url or not api_key:
        raise SystemExit(
            "缺少模型配置：请通过 --api-base-url 和环境变量 "
            f"{args.api_key_env} 提供兼容 OpenAI 的 endpoint 与密钥；程序不会把密钥写入 artifact。"
        )
    selected_case_ids = {value.strip() for value in args.case_ids.split(",") if value.strip()}
    case_count = len(selected_case_ids) or args.case_limit or len(DEVELOPMENT_BLUEPRINTS)
    print("步骤 1/4：正在读取固定的三领域×五状态发展轨迹蓝图。")
    mode = "仅生成可运行轨迹" if args.generation_only else "作者、独立审校和必要修订"
    print(f"阶段完成：本次将处理 {case_count} 条轨迹；模式：{mode}。")
    # The compact benchmark contract is intentionally short enough to avoid
    # truncated JSON while retaining state, evidence, and Thought Pattern.
    author_caller = openai_json_caller(args.api_base_url, api_key, args.author_model, temperature=0.0, max_tokens=1400)
    judge_caller = None if args.generation_only else openai_json_caller(args.api_base_url, api_key, args.judge_model, temperature=0.0, max_tokens=900)
    print("步骤 2/4：正在用强模型生成严格遵循 source-query 的轨迹契约。")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint(records: list[dict[str, object]]) -> None:
        write_development_cases(records, args.output_dir)
        (args.output_dir / "quality_audit.md").write_text(render_development_audit(records), encoding="utf-8")
        print(f"阶段完成：已持久化 {len(records)} 条轨迹的作者稿、审校和准入状态。", flush=True)

    records = build_development_cases(
        args.data_root,
        author_caller,
        judge_caller=judge_caller,
        max_repairs=args.max_repairs,
        case_limit=args.case_limit,
        case_ids=selected_case_ids or None,
        checkpoint=checkpoint,
        generation_only=args.generation_only,
    )
    print("步骤 3/4：正在冻结轨迹记录与运行所需证据。")
    write_development_cases(records, args.output_dir)
    (args.output_dir / "quality_audit.md").write_text(render_development_audit(records), encoding="utf-8")
    admitted = sum(record["admitted"] for record in records)
    print("步骤 4/4：已写出可人工展示和复核的开发轨迹 artifact。")
    label = "已生成可运行轨迹" if args.generation_only else "通过全部质量门槛"
    print(f"全部完成：{admitted}/{len(records)} 条{label}，输出目录为 {args.output_dir.resolve()}。")


if __name__ == "__main__":
    main()
