"""Greedy public-only DSUS State Compiler generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from taira_mt.state_training.sft_harness import (
    greedy_generate_records,
    load_model_and_adapter_for_generation,
    load_public_generation_inputs,
    write_rollout_records,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy non-thinking JSON generation from public RL prompt records only."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-model-revision")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--public-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    print("Step 1/3: reading only the public prompt pool; trainer-private verifier data is not opened at this stage.")
    records = load_public_generation_inputs(args.public_prompts, limit=args.limit)
    print(f"Step 1/3 complete: read {len(records)} public prompts.")
    print("Step 2/3: loading the text-only base model and LoRA adapter.")
    model, tokenizer = load_model_and_adapter_for_generation(
        base_model_name_or_path=args.base_model,
        adapter_path=args.adapter,
        base_model_revision=args.base_model_revision,
    )
    print("Step 2/3 complete: starting JSON generation with temperature=0 / do_sample=False.")
    outputs = greedy_generate_records(
        model=model,
        tokenizer=tokenizer,
        records=records,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    path = write_rollout_records(args.output, outputs)
    print(f"Step 3/3 complete: wrote {len(outputs)} rollouts to {path}.")
    return path


if __name__ == "__main__":
    main()
