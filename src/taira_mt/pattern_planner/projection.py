"""Executable Thought-Pattern ablation projections.

The important property of the SOP-only ablation is operational: it disables
the exact serialized insight adaptations consumed by the v4 compiler while
retaining the same DSUS matcher, Pattern applicability, SOP action templates,
and public evidence guards.  It is therefore not a text/provenance ablation.
"""

from __future__ import annotations

from enum import Enum

from taira_mt.pattern_planner.schemas import StateConditionedPatternLibrary
from taira_mt.provenance import content_sha256

PATTERN_EXECUTION_PROJECTION_VERSION = "pattern_execution_projection_v1"


class PatternExecutionProjection(str, Enum):
    FULL = "full"
    SOP_ONLY = "sop_only"


def project_pattern_library_execution(
    library: StateConditionedPatternLibrary,
    projection: PatternExecutionProjection,
) -> StateConditionedPatternLibrary:
    """Return a deterministic library whose contract semantics match an ablation.

    ``FULL`` is a deep copy retaining its frozen library identity; ``SOP_ONLY`` keeps
    every Skill and SOP action template but disables each bounded insight
    adaptation.  The v4 runtime subsequently observes the changed ``enabled``
    bits and emits a different public option space where adaptations were
    previously legal.
    """

    if projection == PatternExecutionProjection.FULL:
        return library.model_copy(deep=True)

    projected_patterns = []
    for pattern in library.patterns:
        contract = pattern.execution_contract
        if contract is None:
            raise ValueError("cannot project a legacy Pattern library without executable contracts")
        if projection == PatternExecutionProjection.SOP_ONLY:
            contract = contract.model_copy(
                update={
                    "contract_id": f"{contract.contract_id}-sop-only",
                    "insight_adaptations": [
                        adaptation.model_copy(update={"enabled": False})
                        for adaptation in contract.insight_adaptations
                    ],
                },
                deep=True,
            )
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ValueError(f"unsupported Pattern execution projection: {projection}")
        projected_patterns.append(
            pattern.model_copy(update={"execution_contract": contract}, deep=True)
        )

    identity = {
        "projection_version": PATTERN_EXECUTION_PROJECTION_VERSION,
        "base_library_id": library.library_id,
        "projection": projection.value,
        "patterns": [
            item.model_dump(mode="json")
            for item in sorted(projected_patterns, key=lambda item: item.pattern_id)
        ],
    }
    projected_id = f"pattern-library-{content_sha256(identity)[:16]}"
    return library.model_copy(
        update={"library_id": projected_id, "patterns": projected_patterns},
        deep=True,
    )
