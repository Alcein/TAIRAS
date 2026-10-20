"""Run Qwen3.5/Qwen3 LoRA SFT for the DSUS State Compiler."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from taira_mt.state_training.sft_harness import (
    DSUSLoRASFTConfig,
    load_sft_config,
    run_lora_sft,
)

THREE_SOURCE_FAMILIES = {"legacy_rich", "complex_multiturn", "multifamily_warmup"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assistant-only LoRA SFT for DSUS state deltas.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint")
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument(
        "--disable-deepspeed",
        action="store_true",
        help="Disable optional ZeRO-2 for plain four-GPU DDP or a single-card smoke.",
    )
    execution_group.add_argument(
        "--deepspeed-config",
        type=Path,
        help="Opt into a ZeRO config; plain DDP/single GPU is the primary default.",
    )
    parser.add_argument("--max-steps", type=int, help="Override max_steps, e.g. 2 for smoke.")
    parser.add_argument("--train-limit", type=int, help="Override train rows for smoke.")
    parser.add_argument("--dev-limit", type=int, help="Override dev rows for smoke.")
    parser.add_argument("--output-dir", help="Override adapter output directory.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help=(
            "Frozen assembler build directory. Overrides train/dev with "
            "tokenizer_sft.{train,dev}.private.jsonl from this directory."
        ),
    )
    parser.add_argument(
        "--base-model-revision",
        help="Immutable Hugging Face base-model commit used for the paper run.",
    )
    return parser.parse_args(argv)


def _overrides(config: DSUSLoRASFTConfig, args: argparse.Namespace) -> DSUSLoRASFTConfig:
    values: dict[str, object] = {}
    for field in ("max_steps", "train_limit", "dev_limit", "output_dir"):
        value = getattr(args, field)
        if value is not None:
            values[field] = value
    if args.base_model_revision is not None:
        values["base_model_revision"] = args.base_model_revision
    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir.resolve()
        train_path = dataset_dir / "tokenizer_sft.train.private.jsonl"
        dev_path = dataset_dir / "tokenizer_sft.dev.private.jsonl"
        manifest_path = dataset_dir / "manifest.json"
        missing = [str(path) for path in (train_path, dev_path, manifest_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"frozen dataset directory is incomplete: {missing}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        families = {
            item.get("source_family")
            for item in manifest.get("identity", {}).get("source_ledger", [])
            if isinstance(item, dict)
        }
        if manifest.get("status") != "complete" or families != THREE_SOURCE_FAMILIES:
            raise ValueError(
                "--dataset-dir must be a complete three-source assembler build; "
                f"status={manifest.get('status')!r}, source_families={sorted(families)}"
            )
        values["train_file"] = str(train_path)
        values["dev_file"] = str(dev_path)
    if args.deepspeed_config is not None:
        deepspeed_path = args.deepspeed_config.resolve()
        if not deepspeed_path.is_file():
            raise FileNotFoundError(deepspeed_path)
        values["deepspeed_config"] = str(deepspeed_path)
    return DSUSLoRASFTConfig.model_validate({**config.model_dump(mode="python"), **values})


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    config, base_dir = load_sft_config(args.config)
    config = _overrides(config, args)
    use_deepspeed = config.deepspeed_config is not None and not args.disable_deepspeed
    mode = "ZeRO-2" if use_deepspeed else "plain DDP/单卡"
    print(f"训练模式：{mode}；packing={config.packing}；LoRA r={config.lora_rank}。")
    output = run_lora_sft(
        config=config,
        config_base_dir=base_dir,
        disable_deepspeed=args.disable_deepspeed,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(f"DSUS State Compiler SFT 完成：{output}")
    return output


if __name__ == "__main__":
    main()
