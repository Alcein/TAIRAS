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
    print("Preflight step 1/2: reading the typed config and resolving project-root-relative paths.", flush=True)
    config = load_scale_postlabel_config(args.config)
    print(f"Preflight step 1/2 complete: output root={config.output_dir}.", flush=True)

    print("Preflight step 2/2: checking public/routing/manifest files, complete trajectories, and OOD isolation evidence.", flush=True)
    report = preflight_scale_postlabel_pipeline(config)
    print(
        "Preflight step 2/2 complete: "
        f"status={report.status}, prefix={report.public_count}, "
        f"trajectory={report.trajectory_count}, teacher={report.teacher_validated_count}/"
        f"{report.manifest_count}.",
        flush=True,
    )
    if args.dry_run:
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        print("Dry-run complete: no files written, no API/GPU calls, and no training started.", flush=True)
        return None
    return run_scale_postlabel_pipeline(config)


if __name__ == "__main__":
    main()
