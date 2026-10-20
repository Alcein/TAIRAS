"""Canonical serialization and hashing shared by manifests and decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def source_tree_sha256(project_root: Path) -> str:
    """Hash executable package sources and packaging metadata, excluding artifacts."""

    project_root = project_root.resolve()
    paths = sorted((project_root / "src").rglob("*.py"))
    pyproject = project_root / "pyproject.toml"
    if pyproject.is_file():
        paths.append(pyproject)
    records = [
        {
            "path": str(path.relative_to(project_root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(paths)
    ]
    return content_sha256(records)
