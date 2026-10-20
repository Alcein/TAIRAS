"""Leakage-safe LoRA SFT harness for the neural DSUS state compiler.

The dataset assembler deliberately writes a file named
``tokenizer_sft.<split>.private.jsonl`` because the assistant target is a
private supervision signal.  This module enforces a narrower runtime contract:

* only ``model_input`` is rendered as the user message;
* ``assistant_target`` is used only to build assistant-token labels;
* every prompt token and every padding token receives label ``-100``;
* trainer-private verifier payloads are rejected by the SFT loader;
* held-out generation reads the *public* RL prompt pool, before the existing
  verifier chain opens its separate private file.

Heavy ML dependencies are imported lazily so dataset construction and normal
TAIRAS execution do not require a training environment.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json, content_sha256, source_tree_sha256
from taira_mt.state_training.dataset_assembly import (
    PublicRLPromptRecord,
    SFTTokenizerRecord,
)
from taira_mt.state_training.training_views import forbidden_public_key_paths
from taira_mt.state_training.verifier_chain import RolloutOutputRecord

SFT_CONFIG_VERSION = "dsus_state_compiler_lora_sft_v1"
SFT_PREFLIGHT_VERSION = "dsus_state_compiler_sft_preflight_v1"
TOKEN_REPORT_VERSION = "dsus_state_compiler_token_report_v1"
CHECKPOINT_PROVENANCE_VERSION = "dsus_state_compiler_checkpoint_provenance_v1"

# This is an experiment runtime lock, not merely a minimum compatibility claim.
# Qwen3.5 first appeared in an earlier Transformers v5 release, but later releases
# fixed several hybrid Gated-DeltaNet/cache paths.  Keeping one audited triplet
# makes the paper checkpoint reproducible and prevents a broad ``>=5.4`` range
# from silently selecting a materially different Trainer/model implementation.
PRIMARY_RUNTIME_PINS = {
    "transformers": "5.14.1",
    "peft": "0.19.1",
    "accelerate": "1.14.0",
}
PRIMARY_CONDA_ENVIRONMENT = "taira_dsus_qwen35"

ProgressCallback = Callable[[str], None]


class DSUSLoRASFTConfig(StrictModel):
    """Versioned training contract; paths are resolved relative to its JSON file."""

    schema_version: Literal[SFT_CONFIG_VERSION] = SFT_CONFIG_VERSION
    base_model_name_or_path: str = Field(min_length=1)
    base_model_revision: str | None = None
    tokenizer_name_or_path: str | None = None
    initial_adapter_path: str | None = None
    train_file: str = Field(min_length=1)
    dev_file: str | None = None
    output_dir: str = Field(min_length=1)
    max_seq_length: int = Field(default=2048, ge=256, le=32768)
    train_limit: int | None = Field(default=None, ge=1)
    dev_limit: int | None = Field(default=None, ge=1)
    num_train_epochs: float = Field(default=3.0, gt=0)
    max_steps: int = Field(default=-1, ge=-1)
    per_device_train_batch_size: int = Field(default=2, ge=1)
    per_device_eval_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.05, ge=0, lt=1)
    logging_steps: int = Field(default=5, ge=1)
    eval_steps: int = Field(default=50, ge=1)
    save_steps: int = Field(default=50, ge=1)
    save_total_limit: int = Field(default=2, ge=1)
    seed: int = Field(default=42, ge=0)
    lora_rank: int = Field(default=32, ge=1)
    lora_alpha: int = Field(default=64, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    lora_target_modules: Literal["all-linear"] | list[str] = "all-linear"
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    packing: Literal[False] = False
    deepspeed_config: str | None = None
    dataloader_num_workers: int = Field(default=2, ge=0, le=32)
    attn_implementation: str | None = None
    trust_remote_code: Literal[False] = False

    @field_validator(
        "base_model_name_or_path",
        "tokenizer_name_or_path",
        "initial_adapter_path",
        "train_file",
        "dev_file",
        "output_dir",
        "deepspeed_config",
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("lora_target_modules")
    @classmethod
    def validate_target_modules(
        cls,
        value: Literal["all-linear"] | list[str],
    ) -> Literal["all-linear"] | list[str]:
        if isinstance(value, list):
            normalized = sorted({item.strip() for item in value if item.strip()})
            if not normalized:
                raise ValueError("lora_target_modules cannot be empty")
            return normalized
        return value

    @model_validator(mode="after")
    def validate_steps(self) -> DSUSLoRASFTConfig:
        if self.max_steps == 0:
            raise ValueError("max_steps must be -1 or positive")
        return self


class PackageCheck(StrictModel):
    package: str
    installed: bool
    version: str | None = None
    required_for_primary: bool
    note: str


class GPUCheck(StrictModel):
    visible_gpu_count: int = Field(ge=0)
    names: list[str]
    memory_mib: list[int]
    four_gpu_plan_available: bool
    single_gpu_smoke_available: bool


class SFTPreflightReport(StrictModel):
    schema_version: Literal[SFT_PREFLIGHT_VERSION] = SFT_PREFLIGHT_VERSION
    conda_environment: str | None
    python_executable: str
    packages: list[PackageCheck]
    gpu: GPUCheck
    qwen35_transformers_support: bool
    qwen3_transformers_support: bool
    trainer_api_compatible: bool
    primary_runtime_pins_match: bool
    primary_qwen35_ready: bool
    fallback_qwen3_ready: bool
    recommended_isolated_install: list[str]
    blocking_reasons: list[str]
    warnings: list[str]


class LengthDistribution(StrictModel):
    minimum: int = Field(ge=0)
    mean: float = Field(ge=0)
    p50: int = Field(ge=0)
    p90: int = Field(ge=0)
    p95: int = Field(ge=0)
    p99: int = Field(ge=0)
    maximum: int = Field(ge=0)


class TokenLengthReport(StrictModel):
    schema_version: Literal[TOKEN_REPORT_VERSION] = TOKEN_REPORT_VERSION
    dataset_path: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_name_or_path: str
    sample_count: int = Field(ge=0)
    max_seq_length: int = Field(ge=1)
    prompt_tokens: LengthDistribution
    assistant_tokens: LengthDistribution
    total_tokens: LengthDistribution
    over_limit_count: int = Field(ge=0)
    over_limit_sample_ids: list[str]
    prompt_prefix_mismatch_count: int = Field(ge=0)
    public_input_leakage_count: int = Field(ge=0)
    trainer_private_rows_seen: Literal[0] = 0


@dataclass(frozen=True)
class TokenizedSFTExample:
    sample_id: str
    input_ids: list[int]
    labels: list[int]
    prompt_length: int


@dataclass(frozen=True)
class PublicGenerationInput:
    sample_id: str
    public_prompt_sha256: str
    model_input: str


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _major_minor(version: str | None) -> tuple[int, int] | None:
    if version is None:
        return None
    try:
        major, minor, *_ = version.split("+", maxsplit=1)[0].split(".")
        return int(major), int(minor)
    except (TypeError, ValueError):
        return None


def _trainer_api_compatible() -> bool:
    """Check only interfaces used by this harness; never load a model or GPU."""

    try:
        from transformers import Trainer, TrainingArguments
    except (ImportError, ModuleNotFoundError):
        return False
    training_parameters = inspect.signature(TrainingArguments).parameters
    trainer_parameters = inspect.signature(Trainer).parameters
    required_training_parameters = {
        "output_dir",
        "eval_strategy",
        "gradient_checkpointing_kwargs",
        "include_num_input_tokens_seen",
    }
    return required_training_parameters <= set(training_parameters) and "processing_class" in trainer_parameters


def _gpu_check() -> GPUCheck:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return GPUCheck(
            visible_gpu_count=0,
            names=[],
            memory_mib=[],
            four_gpu_plan_available=False,
            single_gpu_smoke_available=False,
        )
    names: list[str] = []
    memory: list[int] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        name, separator, raw_memory = line.rpartition(",")
        if not separator:
            continue
        names.append(name.strip())
        try:
            memory.append(int(raw_memory.strip()))
        except ValueError:
            memory.append(0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        requested = [item.strip() for item in visible.split(",") if item.strip()]
        if requested and all(item.isdigit() for item in requested):
            indices = [int(item) for item in requested if int(item) < len(names)]
            names = [names[index] for index in indices]
            memory = [memory[index] for index in indices]
        elif not requested or visible.strip() == "-1":
            names, memory = [], []
    return GPUCheck(
        visible_gpu_count=len(names),
        names=names,
        memory_mib=memory,
        four_gpu_plan_available=len(names) >= 4 and all(value >= 24_000 for value in memory[:4]),
        single_gpu_smoke_available=bool(names) and memory[0] >= 16_000,
    )


def collect_sft_preflight() -> SFTPreflightReport:
    """Inspect the environment without loading a model or allocating GPU tensors."""

    package_specs = [
        ("torch", True, "BF16 model and training runtime"),
        ("transformers", True, "model/Trainer runtime"),
        ("peft", True, "LoRA adapter injection"),
        ("accelerate", True, "distributed Trainer launcher"),
        ("deepspeed", False, "optional four-GPU ZeRO-2 backend"),
        ("trl", False, "not used: custom assistant-only Trainer avoids Qwen3.5 packing path"),
        ("ms-swift", False, "not used by the lowest-risk in-repository harness"),
        ("flash-attn", False, "optional full-attention acceleration; not required by the harness"),
        ("causal-conv1d", False, "optional Qwen3.5 Gated-DeltaNet kernel"),
        ("flash-linear-attention", False, "optional Qwen3.5 Gated-DeltaNet kernel"),
    ]
    packages = [
        PackageCheck(
            package=name,
            installed=_package_version(name) is not None,
            version=_package_version(name),
            required_for_primary=required,
            note=note,
        )
        for name, required, note in package_specs
    ]
    qwen35_support = _module_available("transformers.models.qwen3_5")
    qwen3_support = _module_available("transformers.models.qwen3")
    trainer_compatible = _trainer_api_compatible()
    versions = {item.package: item.version for item in packages}
    pins_match = all(versions.get(package) == expected for package, expected in PRIMARY_RUNTIME_PINS.items())
    torch_version = _major_minor(versions.get("torch"))
    torch_runtime_ok = torch_version is not None and torch_version >= (2, 7)
    required_installed = all(item.installed for item in packages if item.required_for_primary)
    gpu = _gpu_check()
    primary_ready = (
        required_installed
        and qwen35_support
        and trainer_compatible
        and pins_match
        and torch_runtime_ok
        and gpu.single_gpu_smoke_available
    )
    fallback_ready = required_installed and qwen3_support
    blocking: list[str] = []
    if not qwen35_support:
        blocking.append("当前 transformers 未注册 qwen3_5，不能直接加载 Qwen3.5-4B。")
    if not pins_match:
        mismatch = ", ".join(
            f"{package}={versions.get(package) or 'missing'}（需 {expected}）"
            for package, expected in PRIMARY_RUNTIME_PINS.items()
            if versions.get(package) != expected
        )
        blocking.append(f"论文训练 runtime 未命中已审计精确版本：{mismatch}。")
    if not trainer_compatible:
        blocking.append("当前 Transformers Trainer/TrainingArguments 接口与 harness 不兼容。")
    if not torch_runtime_ok:
        blocking.append(f"当前 PyTorch={versions.get('torch') or 'missing'}；5090 训练环境要求保留 2.7+ CUDA 栈。")
    if not required_installed:
        blocking.append("缺少 torch/transformers/peft/accelerate 中至少一个必需包。")
    if not gpu.single_gpu_smoke_available:
        blocking.append("没有可见的 >=16GB GPU，不能通过 Qwen3.5 单卡 smoke gate。")
    warnings: list[str] = []
    if not gpu.four_gpu_plan_available:
        warnings.append("当前可见 GPU 不满足默认 4 卡计划；仍可使用单卡 smoke 配置。")
    current_environment = os.environ.get("CONDA_DEFAULT_ENV")
    if current_environment not in {"cot_com", PRIMARY_CONDA_ENVIRONMENT}:
        warnings.append(
            f"当前进程不在 cot_com 或隔离训练环境 {PRIMARY_CONDA_ENVIRONMENT}；请显式选择环境。"
        )
    warnings.append("Qwen3.5 是 hybrid Gated-DeltaNet；本 harness 禁用 sequence packing，并使用动态 padding。")
    if not _module_available("causal_conv1d") or not _module_available("fla"):
        warnings.append(
            "causal_conv1d/FLA 未齐备，Qwen3.5 Gated-DeltaNet 会走较慢的 PyTorch fallback；"
            "先 smoke 再决定是否装 kernel。"
        )
    return SFTPreflightReport(
        conda_environment=os.environ.get("CONDA_DEFAULT_ENV"),
        python_executable=os.sys.executable,
        packages=packages,
        gpu=gpu,
        qwen35_transformers_support=qwen35_support,
        qwen3_transformers_support=qwen3_support,
        trainer_api_compatible=trainer_compatible,
        primary_runtime_pins_match=pins_match,
        primary_qwen35_ready=primary_ready,
        fallback_qwen3_ready=fallback_ready,
        recommended_isolated_install=[
            "bash scripts/create_dsus_qwen35_env.sh",
            f"conda create -n {PRIMARY_CONDA_ENVIRONMENT} --clone cot_com",
            (
                f"conda run -n {PRIMARY_CONDA_ENVIRONMENT} python -m pip install --upgrade "
                "'transformers==5.14.1' 'peft==0.19.1' 'accelerate==1.14.0'"
            ),
        ],
        blocking_reasons=blocking,
        warnings=warnings,
    )


def load_sft_records(path: Path, *, limit: int | None = None) -> list[SFTTokenizerRecord]:
    """Strictly load paired public-input/private-target rows.

    ``SFTTokenizerRecord`` forbids extra fields, so a row carrying
    ``trainer_private`` or verifier state fails before tokenization.
    """

    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[SFTTokenizerRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        record = SFTTokenizerRecord.model_validate(payload)
        try:
            public_payload = json.loads(record.model_input)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} model_input is not canonical JSON") from exc
        leaked_paths = forbidden_public_key_paths(public_payload)
        if leaked_paths:
            raise ValueError(f"{path}:{line_number} public input contains private keys: {leaked_paths[:5]}")
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise ValueError(f"SFT dataset is empty: {path}")
    return records


def _as_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected exactly one encoded conversation")
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise TypeError("tokenizer returned an unsupported input_ids representation")
    return value


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    value = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    return _as_token_ids(value)


def _render_generation_prompt(tokenizer: Any, model_input: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": model_input}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _assistant_boundary_and_full_ids(
    tokenizer: Any,
    record: SFTTokenizerRecord,
) -> tuple[int, list[int]]:
    """Locate the assistant content boundary without assuming generation-template identity.

    Some reasoning-model templates append a thinking marker only when
    ``add_generation_prompt=True``.  Comparing a full target conversation with
    the same conversation whose assistant content is empty gives the actual
    loss boundary and never labels the user prompt.
    """

    empty_assistant_ids = _apply_chat_template(
        tokenizer,
        [
            {"role": "user", "content": record.model_input},
            {"role": "assistant", "content": ""},
        ],
        add_generation_prompt=False,
    )
    full_ids = _apply_chat_template(
        tokenizer,
        [
            {"role": "user", "content": record.model_input},
            {"role": "assistant", "content": record.assistant_target},
        ],
        add_generation_prompt=False,
    )
    boundary = 0
    for empty_token, full_token in zip(empty_assistant_ids, full_ids, strict=False):
        if empty_token != full_token:
            break
        boundary += 1
    if boundary == 0 or boundary >= len(full_ids):
        raise ValueError(f"cannot locate assistant target boundary for sample {record.sample_id}")
    return boundary, full_ids


def tokenize_sft_record(
    record: SFTTokenizerRecord,
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> TokenizedSFTExample:
    """Tokenize one row and mask every non-assistant token from the loss."""

    prompt_length, full_ids = _assistant_boundary_and_full_ids(tokenizer, record)
    if len(full_ids) > max_seq_length:
        raise ValueError(
            f"sample {record.sample_id} needs {len(full_ids)} tokens, exceeding max_seq_length={max_seq_length}; "
            "run the token-length dry-run and increase the limit instead of silently truncating gold state"
        )
    labels = [-100] * prompt_length + full_ids[prompt_length:]
    if not labels or all(value == -100 for value in labels):
        raise ValueError(f"sample {record.sample_id} contains no assistant target tokens")
    return TokenizedSFTExample(
        sample_id=record.sample_id,
        input_ids=full_ids,
        labels=labels,
        prompt_length=prompt_length,
    )


class DSUSTokenizedDataset:
    """Small eager dataset whose items contain only tensors understood by Trainer."""

    def __init__(
        self,
        records: Sequence[SFTTokenizerRecord],
        tokenizer: Any,
        *,
        max_seq_length: int,
        progress: ProgressCallback = print,
    ) -> None:
        self.examples: list[TokenizedSFTExample] = []
        for index, record in enumerate(records, start=1):
            self.examples.append(
                tokenize_sft_record(record, tokenizer, max_seq_length=max_seq_length)
            )
            if index % 1000 == 0:
                progress(f"分词进度：已完成 {index}/{len(records)} 条。")
        progress(f"分词完成：{len(self.examples)} 条，未发生静默截断。")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        return {"input_ids": example.input_ids, "labels": example.labels}


class AssistantOnlyDataCollator:
    """Dynamic right-padding collator; prompt and padding labels remain -100."""

    def __init__(self, *, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        if pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")
        if pad_to_multiple_of < 1:
            raise ValueError("pad_to_multiple_of must be positive")
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty batch")
        import torch

        longest = max(len(feature["input_ids"]) for feature in features)
        padded_length = (
            (longest + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
        ) * self.pad_to_multiple_of
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        for feature in features:
            inputs = list(feature["input_ids"])
            labels = list(feature["labels"])
            if len(inputs) != len(labels):
                raise ValueError("input_ids and labels must have equal lengths")
            padding = padded_length - len(inputs)
            input_rows.append(inputs + [self.pad_token_id] * padding)
            label_rows.append(labels + [-100] * padding)
            attention_rows.append([1] * len(inputs) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }


def _distribution(values: Sequence[int]) -> LengthDistribution:
    if not values:
        return LengthDistribution(minimum=0, mean=0.0, p50=0, p90=0, p95=0, p99=0, maximum=0)
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
        return ordered[index]

    return LengthDistribution(
        minimum=ordered[0],
        mean=sum(ordered) / len(ordered),
        p50=percentile(0.50),
        p90=percentile(0.90),
        p95=percentile(0.95),
        p99=percentile(0.99),
        maximum=ordered[-1],
    )


def build_token_length_report(
    *,
    dataset_path: Path,
    tokenizer: Any,
    tokenizer_name_or_path: str,
    max_seq_length: int,
    limit: int | None = None,
    progress: ProgressCallback = print,
) -> TokenLengthReport:
    progress("步骤 1/3：严格读取 SFT tokenizer export，并检查 public input 泄漏。")
    records = load_sft_records(dataset_path, limit=limit)
    progress(f"步骤 1/3 完成：读取 {len(records)} 条；trainer_private 行数为 0。")
    prompts: list[int] = []
    assistants: list[int] = []
    totals: list[int] = []
    over_limit: list[str] = []
    prefix_mismatches = 0
    progress("步骤 2/3：仅按 user=model_input / assistant=assistant_target 构造 chat tokens。")
    for index, record in enumerate(records, start=1):
        generation_prompt_ids = _apply_chat_template(
            tokenizer,
            [{"role": "user", "content": record.model_input}],
            add_generation_prompt=True,
        )
        prompt_length, full_ids = _assistant_boundary_and_full_ids(tokenizer, record)
        stable = full_ids[: len(generation_prompt_ids)] == generation_prompt_ids
        prefix_mismatches += int(not stable)
        target_length = len(full_ids) - prompt_length
        prompts.append(prompt_length)
        assistants.append(target_length)
        totals.append(len(full_ids))
        if len(full_ids) > max_seq_length:
            over_limit.append(record.sample_id)
        if index % 1000 == 0:
            progress(f"token dry-run：已处理 {index}/{len(records)} 条。")
    progress("步骤 2/3 完成：长度统计与 prefix-stability 检查完成。")
    report = TokenLengthReport(
        dataset_path=str(dataset_path.resolve()),
        dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        tokenizer_name_or_path=tokenizer_name_or_path,
        sample_count=len(records),
        max_seq_length=max_seq_length,
        prompt_tokens=_distribution(prompts),
        assistant_tokens=_distribution(assistants),
        total_tokens=_distribution(totals),
        over_limit_count=len(over_limit),
        over_limit_sample_ids=over_limit[:100],
        prompt_prefix_mismatch_count=prefix_mismatches,
        public_input_leakage_count=0,
    )
    progress(
        "步骤 3/3 完成："
        f"total p99={report.total_tokens.p99}, max={report.total_tokens.maximum}, "
        f"over_limit={report.over_limit_count}, prefix_mismatch={prefix_mismatches}。"
    )
    return report


def load_sft_config(path: Path) -> tuple[DSUSLoRASFTConfig, Path]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DSUSLoRASFTConfig.model_validate(payload), path.parent


def _resolved_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _tokenizer_pad_id(tokenizer: Any) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    if pad is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    return int(pad)


def _model_family_supported(model_name: str) -> tuple[bool, str]:
    normalized = model_name.casefold()
    if "qwen3.5" in normalized or "qwen3_5" in normalized:
        return _module_available("transformers.models.qwen3_5"), "qwen3.5"
    if "qwen3" in normalized:
        return _module_available("transformers.models.qwen3"), "qwen3"
    return True, "auto"


def _load_causal_lm(config: DSUSLoRASFTConfig) -> Any:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    supported, family = _model_family_supported(config.base_model_name_or_path)
    if not supported:
        raise RuntimeError(
            f"当前 Transformers 不支持 {family}；先运行 preflight，并在隔离环境执行其最小升级命令。"
        )
    kwargs: dict[str, Any] = {
        "revision": config.base_model_revision,
        "trust_remote_code": config.trust_remote_code,
        "dtype": torch.bfloat16 if config.bf16 else None,
    }
    if config.attn_implementation:
        kwargs["attn_implementation"] = config.attn_implementation
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if family == "qwen3.5" and hasattr(transformers, "Qwen3_5ForCausalLM"):
        model_class = transformers.Qwen3_5ForCausalLM
    else:
        model_class = AutoModelForCausalLM
    model = model_class.from_pretrained(config.base_model_name_or_path, **kwargs)
    if family == "qwen3.5":
        if type(model).__name__ != "Qwen3_5ForCausalLM":
            raise RuntimeError(f"Qwen3.5 text-only loader returned unexpected class: {type(model).__name__}")
        if hasattr(model, "visual") or hasattr(model, "vision_model"):
            raise RuntimeError("Qwen3.5 SFT must not retain a vision tower")
    return model


def _runtime_versions() -> dict[str, str | None]:
    return {
        name: _package_version(name)
        for name in ("torch", "transformers", "peft", "accelerate", "deepspeed", "trl")
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(item) for item in value), key=lambda item: canonical_json(item))
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _file_ledger(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_size": path.stat().st_size,
        "row_count": sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()),
    }


def _git_commit(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "src", "pyproject.toml"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        return {"commit": commit, "source_dirty": dirty}
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"commit": None, "source_dirty": None}


def write_checkpoint_provenance(
    *,
    output_dir: Path,
    config: DSUSLoRASFTConfig,
    train_path: Path,
    dev_path: Path | None,
    model: Any,
    tokenizer: Any,
    metrics: Mapping[str, Any],
    project_root: Path,
) -> Path:
    """Freeze model, data, source, dependency, and LoRA identities beside the adapter."""

    model_config = getattr(model, "config", None)
    resolved_commit = getattr(model_config, "_commit_hash", None)
    peft_config = getattr(model, "peft_config", {})
    peft_payload = {
        name: value.to_dict() if hasattr(value, "to_dict") else str(value)
        for name, value in peft_config.items()
    }
    payload = {
        "schema_version": CHECKPOINT_PROVENANCE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_role": "dsus_state_compiler_lora_adapter",
        "base_model": {
            "name_or_path": config.base_model_name_or_path,
            "requested_revision": config.base_model_revision,
            "resolved_commit": resolved_commit,
            "model_type": getattr(model_config, "model_type", None),
            "architectures": getattr(model_config, "architectures", None),
        },
        "tokenizer": {
            "name_or_path": getattr(tokenizer, "name_or_path", config.tokenizer_name_or_path),
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "pad_token_id": _tokenizer_pad_id(tokenizer),
        },
        "datasets": {
            "train": _file_ledger(train_path),
            "dev": _file_ledger(dev_path) if dev_path is not None else None,
        },
        "training_config": config.model_dump(mode="json"),
        "training_config_sha256": content_sha256(config.model_dump(mode="json")),
        "peft_config": _json_safe(peft_payload),
        "runtime_versions": _runtime_versions(),
        "source": {
            **_git_commit(project_root),
            "source_tree_sha256": source_tree_sha256(project_root),
        },
        "metrics": _json_safe(dict(metrics)),
        "external_calls": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "checkpoint_provenance.json"
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return path


def run_lora_sft(
    *,
    config: DSUSLoRASFTConfig,
    config_base_dir: Path,
    disable_deepspeed: bool = False,
    resume_from_checkpoint: str | None = None,
    progress: ProgressCallback = print,
) -> Path:
    """Run LoRA SFT. Formal calls should use ``torchrun --nproc_per_node=4``."""

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

    train_path = _resolved_path(config_base_dir, config.train_file)
    dev_path = _resolved_path(config_base_dir, config.dev_file)
    output_dir = _resolved_path(config_base_dir, config.output_dir)
    deepspeed_path = _resolved_path(config_base_dir, config.deepspeed_config)
    initial_adapter_path = _resolved_path(config_base_dir, config.initial_adapter_path)
    assert train_path is not None and output_dir is not None
    if disable_deepspeed:
        deepspeed_path = None

    progress("步骤 1/6：读取 leakage-safe SFT 数据；任何 trainer_private 字段都会直接失败。")
    train_records = load_sft_records(train_path, limit=config.train_limit)
    dev_records = (
        load_sft_records(dev_path, limit=config.dev_limit) if dev_path is not None else []
    )
    progress(f"步骤 1/6 完成：train={len(train_records)}，dev={len(dev_records)}。")

    progress("步骤 2/6：加载 tokenizer，并执行 assistant-only tokenization。")
    tokenizer_name = config.tokenizer_name_or_path or config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=config.base_model_revision,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = DSUSTokenizedDataset(
        train_records,
        tokenizer,
        max_seq_length=config.max_seq_length,
        progress=progress,
    )
    dev_dataset = (
        DSUSTokenizedDataset(
            dev_records,
            tokenizer,
            max_seq_length=config.max_seq_length,
            progress=progress,
        )
        if dev_records
        else None
    )
    progress("步骤 2/6 完成：prompt/padding labels 均为 -100，未做 sequence packing。")

    progress("步骤 3/6：以 BF16 加载 text-only causal LM，并准备可训练 LoRA。")
    set_seed(config.seed)
    model = _load_causal_lm(config)
    model.config.use_cache = False
    if config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if initial_adapter_path is None:
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora)
        adapter_mode = "fresh"
    else:
        if not (initial_adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"initial LoRA adapter is incomplete: {initial_adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            initial_adapter_path,
            is_trainable=True,
        )
        adapter_mode = f"continued:{initial_adapter_path}"
    trainable, total = model.get_nb_trainable_parameters()
    progress(
        f"步骤 3/6 完成：mode={adapter_mode}；trainable={trainable:,}/{total:,} "
        f"({100 * trainable / total:.3f}%)。"
    )

    progress("步骤 4/6：构造 Trainer；默认 4 卡 ZeRO-2，可由 smoke 显式降级单卡。")
    eval_strategy = "steps" if dev_dataset is not None else "no"
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=dev_dataset is not None,
        eval_strategy=eval_strategy,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps if dev_dataset is not None else None,
        save_steps=config.save_steps,
        save_strategy="steps",
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        tf32=config.tf32,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        deepspeed=str(deepspeed_path) if deepspeed_path is not None else None,
        ddp_find_unused_parameters=False,
        include_num_input_tokens_seen=True,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=AssistantOnlyDataCollator(pad_token_id=_tokenizer_pad_id(tokenizer)),
        processing_class=tokenizer,
    )
    progress("步骤 4/6 完成：Trainer 就绪。")

    progress("步骤 5/6：开始 LoRA SFT；每个 checkpoint 仅保存 adapter 与训练状态。")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics: dict[str, Any] = dict(train_result.metrics)
    if dev_dataset is not None:
        metrics.update({f"final_{key}": value for key, value in trainer.evaluate().items()})
    progress(f"步骤 5/6 完成：train_loss={metrics.get('train_loss', 'n/a')}。")

    progress("步骤 6/6：保存最终 adapter、tokenizer 与可追溯 provenance。")
    if trainer.is_world_process_zero():
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(output_dir)
        project_root = Path(__file__).resolve().parents[3]
        provenance_path = write_checkpoint_provenance(
            output_dir=output_dir,
            config=config,
            train_path=train_path,
            dev_path=dev_path,
            model=model,
            tokenizer=tokenizer,
            metrics=metrics,
            project_root=project_root,
        )
        progress(f"步骤 6/6 完成：provenance={provenance_path}。")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    return output_dir


def load_public_generation_inputs(
    path: Path,
    *,
    limit: int | None = None,
) -> list[PublicGenerationInput]:
    """Load only the public RL pool; no target/verifier file is accepted here."""

    if not path.is_file():
        raise FileNotFoundError(path)
    output: list[PublicGenerationInput] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        record = PublicRLPromptRecord.model_validate(payload)
        public_payload = record.public_prompt.canonical_payload()
        leaked = forbidden_public_key_paths(public_payload)
        if leaked:
            raise ValueError(f"public generation input contains private keys: {leaked[:5]}")
        output.append(
            PublicGenerationInput(
                sample_id=record.sample_id,
                public_prompt_sha256=record.public_prompt_sha256,
                model_input=record.public_prompt.canonical_json(),
            )
        )
        if limit is not None and len(output) >= limit:
            break
    if not output:
        raise ValueError(f"public generation pool is empty: {path}")
    return output


def greedy_generate_records(
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[PublicGenerationInput],
    batch_size: int = 4,
    max_new_tokens: int = 1024,
    progress: ProgressCallback = print,
) -> list[RolloutOutputRecord]:
    """Generate deterministic, non-thinking JSON candidates from public prompts only."""

    import torch

    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("batch_size and max_new_tokens must be positive")
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")
    prior_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    outputs: list[RolloutOutputRecord] = []
    try:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            prompt_texts = [_render_generation_prompt(tokenizer, item.model_input) for item in batch]
            encoded = tokenizer(
                prompt_texts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_width = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=_tokenizer_pad_id(tokenizer),
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                )
            completions = tokenizer.batch_decode(
                generated[:, prompt_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outputs.extend(
                RolloutOutputRecord(
                    sample_id=item.sample_id,
                    public_prompt_sha256=item.public_prompt_sha256,
                    model_output=completion.strip(),
                )
                for item, completion in zip(batch, completions, strict=True)
            )
            progress(f"greedy JSON 生成：已完成 {min(start + len(batch), len(records))}/{len(records)} 条。")
    finally:
        tokenizer.padding_side = prior_padding_side
    return outputs


def load_model_and_adapter_for_generation(
    *,
    base_model_name_or_path: str,
    adapter_path: Path,
    base_model_revision: str | None = None,
    load_in_4bit: bool = False,
) -> tuple[Any, Any]:
    """Load a text-only causal LM plus LoRA adapter for held-out generation."""

    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # This loader is used only for one-example-at-a-time held-out generation,
    # never for training.  Recent Transformers can call ``torch.compile``
    # implicitly from ``generate`` for Qwen3.5's cache.  Make that optional
    # optimization a no-op in this process so a fresh rollout worker does not
    # block on an inductor build; all model weights and greedy operations are
    # otherwise unchanged.
    torch.compile = lambda function, *args, **kwargs: function

    supported, family = _model_family_supported(base_model_name_or_path)
    if not supported:
        raise RuntimeError(f"当前 Transformers 不支持 {family}；请先执行 preflight 建议的隔离升级。")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_class = (
        transformers.Qwen3_5ForCausalLM
        if family == "qwen3.5" and hasattr(transformers, "Qwen3_5ForCausalLM")
        else AutoModelForCausalLM
    )
    model_kwargs: dict[str, Any] = {
        "revision": base_model_revision,
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "sdpa",
        "trust_remote_code": False,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = model_class.from_pretrained(base_model_name_or_path, **model_kwargs)
    if family == "qwen3.5" and type(model).__name__ != "Qwen3_5ForCausalLM":
        raise RuntimeError(f"Qwen3.5 generation loaded unexpected class: {type(model).__name__}")
    model = PeftModel.from_pretrained(model, adapter_path)
    # Transformers 5.14 may auto-compile Qwen3.5's cache-backed forward pass on
    # the first ``generate`` call.  The compiler is useful for long batched
    # serving, but it can spend several minutes constructing an inductor graph
    # for this short, one-example-at-a-time state-compilation workload.  Keep
    # the runtime contract deterministic and avoid a hidden, per-process
    # compilation barrier; greedy decoding remains unchanged.
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "disable_compile"):
        generation_config.disable_compile = True
    # PEFT forwards ``generate`` to the wrapped base model and Transformers
    # still evaluates the model-level auto-compile predicate in that path.
    # Override the predicate as a second guard: short state patches are not a
    # serving benchmark and should never trigger an implicit inductor build.
    def _never_auto_compile(*_args: Any, **_kwargs: Any) -> bool:
        return False

    for candidate in (
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ):
        if candidate is not None and hasattr(candidate, "_valid_auto_compile_criteria"):
            candidate._valid_auto_compile_criteria = _never_auto_compile
    model.eval()
    return model, tokenizer


def write_rollout_records(path: Path, records: Sequence[RolloutOutputRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(record.model_dump(mode="json")) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


__all__ = [
    "AssistantOnlyDataCollator",
    "CHECKPOINT_PROVENANCE_VERSION",
    "DSUSLoRASFTConfig",
    "DSUSTokenizedDataset",
    "GPUCheck",
    "LengthDistribution",
    "PackageCheck",
    "PublicGenerationInput",
    "SFT_CONFIG_VERSION",
    "SFT_PREFLIGHT_VERSION",
    "SFTPreflightReport",
    "TOKEN_REPORT_VERSION",
    "TokenLengthReport",
    "TokenizedSFTExample",
    "build_token_length_report",
    "collect_sft_preflight",
    "greedy_generate_records",
    "load_model_and_adapter_for_generation",
    "load_public_generation_inputs",
    "load_sft_config",
    "load_sft_records",
    "run_lora_sft",
    "tokenize_sft_record",
    "write_checkpoint_provenance",
    "write_rollout_records",
]
