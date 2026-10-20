"""Load immutable counterfactual reward configurations."""

from __future__ import annotations

from pathlib import Path

import yaml

from taira_mt.counterfactual.schemas import CounterfactualRewardConfig
from taira_mt.provenance import content_sha256


def load_reward_config(path: Path) -> tuple[CounterfactualRewardConfig, str]:
    """Parse a strict YAML/JSON config and return its semantic content hash."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reward config must contain one mapping")
    config = CounterfactualRewardConfig.model_validate(payload)
    return config, content_sha256(config.model_dump(mode="json"))
