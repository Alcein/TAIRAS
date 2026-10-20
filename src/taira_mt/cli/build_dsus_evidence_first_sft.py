"""Build the frozen evidence-first Real500 SFT continuation views offline."""

from __future__ import annotations

import argparse
from pathlib import Path

from taira_mt.state_training.evidence_first_sft import build_evidence_first_real_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mix-dir", type=Path, required=True)
    parser.add_argument("--clean-retention-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cold-repeat-limit", type=int, default=1)
    parser.add_argument("--clean-per-mechanism-role", type=int, default=12)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    output = build_evidence_first_real_sft(
        source_mix_dir=args.source_mix_dir,
        clean_retention_dir=args.clean_retention_dir,
        output_dir=args.output_dir,
        cold_repeat_limit=args.cold_repeat_limit,
        clean_per_mechanism_role=args.clean_per_mechanism_role,
    )
    print("Complete: only data and audit files were built; no GPU use, training, or API calls.", flush=True)
    return output


if __name__ == "__main__":
    main()
