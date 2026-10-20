"""Generate and verifier-score a held-out DSUS State Compiler split."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from taira_mt.provenance import content_sha256
from taira_mt.state_training.sft_harness import (
    greedy_generate_records,
    load_model_and_adapter_for_generation,
    load_public_generation_inputs,
    write_rollout_records,
)
from taira_mt.state_training.verifier_chain import VerifierBatchResult, score_verifier_files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Held-out DSUS evaluation: generate from public prompts first, then join the "
            "separate trainer-private verifier after generation."
        )
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-model-revision")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--public-prompts", type=Path, required=True)
    parser.add_argument("--trainer-private", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> VerifierBatchResult:
    args = parse_args(argv)
    print("Stage A: public-only greedy generation; trainer-private files have not been read.")
    public_records = load_public_generation_inputs(args.public_prompts, limit=args.limit)
    model, tokenizer = load_model_and_adapter_for_generation(
        base_model_name_or_path=args.base_model,
        adapter_path=args.adapter,
        base_model_revision=args.base_model_revision,
        load_in_4bit=args.load_in_4bit,
    )
    rollouts = greedy_generate_records(
        model=model,
        tokenizer=tokenizer,
        records=public_records,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    run_identity = {
        "base_model": args.base_model,
        "base_model_revision": args.base_model_revision,
        "adapter": str(args.adapter.resolve()),
        "public_prompt_count": len(public_records),
    }
    generation_dir = args.output_root / f"greedy-{content_sha256(run_identity)[:16]}"
    rollout_path = write_rollout_records(generation_dir / "rollout_outputs.jsonl", rollouts)
    print(f"Stage A complete: {len(rollouts)} records frozen at {rollout_path}.")

    print("Stage B: generation is complete; now joining the trainer-private verifier by sample_id.")
    result = score_verifier_files(
        public_path=args.public_prompts,
        private_path=args.trainer_private,
        rollout_path=rollout_path,
        require_complete=args.limit is None,
        output_root=args.output_root / "verifier_scores",
    )
    print(
        "Stage B complete: "
        f"accepted={result.summary.acceptance_rate:.4f}, "
        f"state_equivalence={result.summary.component_means.get('state_equivalence', 0.0):.4f}, "
        f"mean_reward={result.summary.mean_total_reward:.4f}."
    )
    return result


if __name__ == "__main__":
    main()
