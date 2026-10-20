"""Materialize the deterministic dialogue protocol for a frozen Agent artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taira_mt.multiturn_benchmark.runtime import load_frozen_agent_records, materialize_runtime_protocols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize deterministic turn schedules for frozen TAIRAS Agent episodes."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("data/benchmark/episodes.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to dialogue_protocols.jsonl beside --artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.artifact.with_name("dialogue_protocols.jsonl")
    print("Step 1/2: reading frozen agent episodes and deriving deterministic per-turn action protocols.", flush=True)
    records = load_frozen_agent_records(args.artifact)
    protocols = materialize_runtime_protocols(records)
    print(f"Stage complete: generated runtime protocols for {len(protocols)} episodes.", flush=True)
    print("Step 2/2: writing the evaluator-side protocol ledger; no LLM calls are made.", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(protocol, ensure_ascii=False, sort_keys=True) + "\n" for protocol in protocols),
        encoding="utf-8",
    )
    print(f"Complete: runtime protocols written to {output.resolve()}.", flush=True)


if __name__ == "__main__":
    main()
