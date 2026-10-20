"""Validate the paper-family to runtime-entry granularity map."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taira_mt.provenance import content_sha256

MAPPING_SCHEMA_VERSION = "pattern_family_runtime_mapping_v1"
ALLOWED_RELATIONS = {
    "primary_control",
    "secondary_control",
    "semantic_binding_support",
    "evidence_binding_support",
    "embedded_evidence_realization",
}


@dataclass(frozen=True)
class LoadedFamilyRuntimeMapping:
    """A source-locked many-to-many correspondence and its derived views."""

    payload: dict[str, Any]
    mapping_file_sha256: str
    mapping_content_sha256: str
    runtime_file_sha256: str
    runtime_content_sha256: str
    family_to_runtime: dict[str, list[dict[str, str]]]
    runtime_to_family: dict[str, list[dict[str, str]]]
    relation_counts: dict[str, int]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must have a JSON object at the top level")
    return payload


def load_family_runtime_mapping(
    *,
    mapping_path: Path,
    paper_library_path: Path,
    runtime_library_path: Path,
) -> LoadedFamilyRuntimeMapping:
    """Fail closed if the 9-family/14-entry correspondence has drifted."""

    mapping_path = mapping_path.resolve()
    paper_library_path = paper_library_path.resolve()
    runtime_library_path = runtime_library_path.resolve()
    mapping = _load_object(mapping_path)
    paper_library = _load_object(paper_library_path)
    runtime_library = _load_object(runtime_library_path)
    if mapping.get("schema_version") != MAPPING_SCHEMA_VERSION:
        raise ValueError("Unsupported Pattern family/runtime mapping schema")
    if mapping.get("granularity_contract", {}).get("counts_are_not_additive") is not True:
        raise ValueError("mapping must explicitly declare that 9 and 14 cannot be added together")

    sources = mapping.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("mapping is missing sources")
    paper_source = sources.get("paper_family_library", {})
    runtime_source = sources.get("runtime_entry_library", {})
    actual_paper_file_hash = _file_sha256(paper_library_path)
    actual_paper_content_hash = content_sha256(paper_library)
    actual_runtime_file_hash = _file_sha256(runtime_library_path)
    actual_runtime_content_hash = content_sha256(runtime_library)
    expected_actual_pairs = (
        (paper_source.get("file_sha256"), actual_paper_file_hash, "paper family file"),
        (paper_source.get("content_sha256"), actual_paper_content_hash, "paper family content"),
        (runtime_source.get("file_sha256"), actual_runtime_file_hash, "runtime entry file"),
        (runtime_source.get("content_sha256"), actual_runtime_content_hash, "runtime entry content"),
    )
    for expected, actual, label in expected_actual_pairs:
        if expected != actual:
            raise ValueError(f"{label} hash does not match the mapping declaration")
    if paper_source.get("library_id") != paper_library.get("library_id"):
        raise ValueError("paper family library_id does not match the mapping declaration")
    if runtime_source.get("schema_version") != runtime_library.get("schema_version"):
        raise ValueError("runtime schema does not match the mapping declaration")

    actual_paper = {
        item["strategy_family"]: item for item in paper_library.get("patterns", []) if isinstance(item, dict)
    }
    actual_runtime = {item["pattern_id"]: item for item in runtime_library.get("skills", []) if isinstance(item, dict)}
    declared_family_rows = mapping.get("paper_procedural_families")
    declared_runtime_rows = mapping.get("runtime_executable_entries")
    if not isinstance(declared_family_rows, list) or not isinstance(declared_runtime_rows, list):
        raise ValueError("mapping must contain paper_procedural_families and runtime_executable_entries")
    declared_families = {item["family_id"]: item for item in declared_family_rows}
    declared_runtime = {item["runtime_entry_id"]: item for item in declared_runtime_rows}
    if len(declared_families) != len(declared_family_rows):
        raise ValueError("mapping contains duplicate family_id values")
    if len(declared_runtime) != len(declared_runtime_rows):
        raise ValueError("mapping contains duplicate runtime_entry_id values")
    if set(declared_families) != set(actual_paper):
        raise ValueError("mapping paper family set does not match the actual induction artifact")
    if set(declared_runtime) != set(actual_runtime):
        raise ValueError("mapping runtime entry set does not match the actual executable library")
    for family_id, row in declared_families.items():
        source_row = actual_paper[family_id]
        if row.get("induced_pattern_id") != source_row.get("pattern_id"):
            raise ValueError(f"{family_id} has an inconsistent induced_pattern_id")
        if row.get("name") != source_row.get("name"):
            raise ValueError(f"{family_id} has an inconsistent paper name")

    family_to_runtime: dict[str, list[dict[str, str]]] = defaultdict(list)
    runtime_to_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    relation_counts: Counter[str] = Counter()
    for runtime_id, row in declared_runtime.items():
        source_row = actual_runtime[runtime_id]
        layer = row.get("layer")
        role = row.get("implementation_role")
        if layer != source_row.get("layer"):
            raise ValueError(f"{runtime_id} layer does not match the actual runtime entry")
        expected_role = "semantic_binding_adapter" if layer == "semantic_grounding" else "procedural_control_adapter"
        if role != expected_role:
            raise ValueError(f"{runtime_id} implementation_role does not match its layer")
        links = row.get("paper_family_links")
        if not isinstance(links, list) or not links:
            raise ValueError(f"{runtime_id} has no paper_family_links")
        seen_links: set[tuple[str, str]] = set()
        for link in links:
            family_id = link.get("family_id")
            relation = link.get("relation")
            if family_id not in declared_families:
                raise ValueError(f"{runtime_id} references unknown family {family_id}")
            if relation not in ALLOWED_RELATIONS:
                raise ValueError(f"{runtime_id} uses unknown relation {relation}")
            if layer == "semantic_grounding" and relation not in {
                "semantic_binding_support",
                "evidence_binding_support",
            }:
                raise ValueError(f"{runtime_id} semantic entry cannot declare a control relation")
            if layer == "interaction_control" and relation in {
                "semantic_binding_support",
                "evidence_binding_support",
            }:
                raise ValueError(f"{runtime_id} interaction entry cannot declare a semantic relation")
            pair = (family_id, relation)
            if pair in seen_links:
                raise ValueError(f"{runtime_id} declares the same family relation more than once")
            seen_links.add(pair)
            record = {
                "runtime_entry_id": runtime_id,
                "layer": layer,
                "relation": relation,
            }
            family_to_runtime[family_id].append(record)
            runtime_to_family[runtime_id].append(
                {
                    "family_id": family_id,
                    "relation": relation,
                }
            )
            relation_counts[relation] += 1

    if set(family_to_runtime) != set(declared_families):
        raise ValueError("At least one paper family has no runtime correspondence")
    for family_id, family in declared_families.items():
        relations = {item["relation"] for item in family_to_runtime[family_id]}
        status = str(family.get("runtime_realization_status", ""))
        if status != "embedded_not_standalone_in_main_controller" and "primary_control" not in relations:
            raise ValueError(f"{family_id} is missing a primary runtime control")
        if status == "embedded_not_standalone_in_main_controller" and "primary_control" in relations:
            raise ValueError(f"{family_id} claims embedded-only but has a primary control")

    layer_counts = Counter(item["layer"] for item in actual_runtime.values())
    count_contract = mapping.get("count_contract", {})
    count_checks = {
        "paper_procedural_family_count": len(actual_paper),
        "runtime_executable_entry_count": len(actual_runtime),
        "runtime_semantic_binding_entry_count": layer_counts["semantic_grounding"],
        "runtime_interaction_control_entry_count": layer_counts["interaction_control"],
    }
    for key, actual in count_checks.items():
        if count_contract.get(key) != actual:
            raise ValueError(f"count_contract.{key} does not match the actual source")

    return LoadedFamilyRuntimeMapping(
        payload=mapping,
        mapping_file_sha256=_file_sha256(mapping_path),
        mapping_content_sha256=content_sha256(mapping),
        runtime_file_sha256=actual_runtime_file_hash,
        runtime_content_sha256=actual_runtime_content_hash,
        family_to_runtime={
            key: sorted(value, key=lambda item: (item["layer"], item["runtime_entry_id"], item["relation"]))
            for key, value in sorted(family_to_runtime.items())
        },
        runtime_to_family={
            key: sorted(value, key=lambda item: (item["family_id"], item["relation"]))
            for key, value in sorted(runtime_to_family.items())
        },
        relation_counts=dict(sorted(relation_counts.items())),
    )
