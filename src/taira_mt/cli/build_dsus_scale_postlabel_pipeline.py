"""Preflight or run the offline Scale teacher-label postprocessing pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from taira_mt.state_training.scale_postlabel_pipeline import (
    load_scale_postlabel_config,
    preflight_scale_postlabel_pipeline,
    run_scale_postlabel_pipeline,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate frozen inputs and print the plan; a missing future teacher file is expected.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path | None:
    args = parse_args(argv)
    print("预检步骤 1/2：读取 typed config 并解析 project-root-relative 路径。", flush=True)
    config = load_scale_postlabel_config(args.config)
    print(f"预检步骤 1/2 完成：输出根目录={config.output_dir}。", flush=True)

    print("预检步骤 2/2：核对 public/routing/manifest、完整 trajectory 与 OOD 隔离证据。", flush=True)
    report = preflight_scale_postlabel_pipeline(config)
    print(
        "预检步骤 2/2 完成："
        f"status={report.status}，prefix={report.public_count}，"
        f"trajectory={report.trajectory_count}，teacher={report.teacher_validated_count}/"
        f"{report.manifest_count}。",
        flush=True,
    )
    if args.dry_run:
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        print("dry-run 完成：未写文件、未调用 API/GPU、未启动训练。", flush=True)
        return None
    return run_scale_postlabel_pipeline(config)


if __name__ == "__main__":
    main()
