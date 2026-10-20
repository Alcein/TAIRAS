"""Verified loading for frozen state-conditioned Pattern libraries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from taira_mt.artifacts import verify_artifact_bundle
from taira_mt.pattern_planner.schemas import (
    PatternInductionManifest,
    PatternInductionSummary,
    StateConditionedPatternLibrary,
)
from taira_mt.provenance import content_sha256


@dataclass(frozen=True)
class LoadedPatternLibrary:
    manifest_sha256: str
    manifest: PatternInductionManifest
    library: StateConditionedPatternLibrary
    summary: PatternInductionSummary


def load_pattern_library_artifact(run_dir: Path) -> LoadedPatternLibrary:
    """Load a Pattern library only after manifest and content checks pass."""

    bundle = verify_artifact_bundle(run_dir)
    manifest = PatternInductionManifest.model_validate(bundle.manifest)
    library_payload = json.loads(
        bundle.require_output("pattern_library.public.json").read_text(encoding="utf-8")
    )
    summary_payload = json.loads(
        bundle.require_output("summary.json").read_text(encoding="utf-8")
    )
    library = StateConditionedPatternLibrary.model_validate(library_payload)
    summary = PatternInductionSummary.model_validate(summary_payload)
    if library.library_id != manifest.library_id or summary.pattern_library_id != library.library_id:
        raise ValueError("Pattern library IDs differ across manifest, library, and summary")
    if library.parent_dsus_build_id != manifest.parent_dsus_build_id:
        raise ValueError("Pattern library parent DSUS build differs from manifest")
    if content_sha256(library_payload) != manifest.library_sha256:
        raise ValueError("Pattern library content hash differs from manifest")
    if content_sha256(summary_payload) != manifest.summary_sha256:
        raise ValueError("Pattern summary content hash differs from manifest")
    return LoadedPatternLibrary(
        manifest_sha256=bundle.manifest_sha256,
        manifest=manifest,
        library=library,
        summary=summary,
    )
