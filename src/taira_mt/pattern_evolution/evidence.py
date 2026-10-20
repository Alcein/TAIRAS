"""Build honest, offline evidence for Thought Pattern repository evolution.

The current Pattern artifact is a final induced library, not a persisted
training-time repository history.  This module therefore distinguishes:

* retrospective diagnostic projections (``seed`` and ``+success``);
* the observed final induced library (``+correction``); and
* an identity checkpoint (``full-gated``) when no admitted post-hoc candidate
  or paired replay evidence exists.

That distinction is deliberately machine-readable.  In particular, the tool
never turns missing replay measurements into an apparent improvement.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from taira_mt.pattern_evolution.mapping import load_family_runtime_mapping
from taira_mt.pattern_planner.schemas import StateConditionedPatternLibrary
from taira_mt.provenance import canonical_json, content_sha256

EVIDENCE_SCHEMA_VERSION = "pattern_repository_evolution_evidence_v1"
CHECKPOINT_ORDER = ("seed", "+success", "+correction", "full-gated")
PRIVATE_KEY_MARKERS = (
    "private",
    "hidden",
    "latent",
    "oracle",
    "simulator",
    "ground_truth",
    "groundtruth",
    "secret",
    "reward",
)
PRIVATE_VALUE_SENTINEL = "__PRIVATE_SENTINEL__"
APPLICABILITY_AXES = (
    "allowed_phases",
    "allowed_completeness",
    "allowed_stability",
    "allowed_feasibility",
    "allowed_last_user_acts",
    "allowed_patience_risks",
)


@dataclass(frozen=True)
class LoadedPatternSource:
    """Verified source files needed by the offline evidence builder."""

    artifact_dir: Path
    manifest: dict[str, Any]
    library: dict[str, Any]
    library_file_sha256: str
    library_content_sha256: str
    manifest_file_sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_idempotently(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite evolution evidence with different content: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"{path} must contain an array of objects or JSONL")
        return payload
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        records.append(payload)
    return records


def load_pattern_source(artifact_dir: Path) -> LoadedPatternSource:
    """Load and verify the real v4 Pattern induction artifact."""

    artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / "manifest.json"
    library_path = artifact_dir / "pattern_library.public.json"
    if not manifest_path.is_file() or not library_path.is_file():
        raise FileNotFoundError("Pattern artifact must contain both manifest.json and pattern_library.public.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    library = json.loads(library_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(library, dict):
        raise ValueError("Pattern manifest/library must have a JSON object at the top level")

    expected_outputs = {
        item["relative_path"]: item["sha256"]
        for item in manifest.get("outputs", [])
        if isinstance(item, dict) and "relative_path" in item and "sha256" in item
    }
    actual_file_sha256 = _file_sha256(library_path)
    expected_file_sha256 = expected_outputs.get("pattern_library.public.json")
    if expected_file_sha256 != actual_file_sha256:
        raise ValueError("Pattern library file hash does not match the manifest")
    actual_content_sha256 = content_sha256(library)
    if manifest.get("library_sha256") != actual_content_sha256:
        raise ValueError("Pattern library content hash does not match the manifest")

    parsed = StateConditionedPatternLibrary.model_validate(library)
    pattern_ids = [item.pattern_id for item in parsed.patterns]
    if len(pattern_ids) != len(set(pattern_ids)):
        raise ValueError("Pattern library contains duplicate pattern_id values")
    return LoadedPatternSource(
        artifact_dir=artifact_dir,
        manifest=manifest,
        library=library,
        library_file_sha256=actual_file_sha256,
        library_content_sha256=actual_content_sha256,
        manifest_file_sha256=_file_sha256(manifest_path),
    )


def _filter_success_insights(pattern: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(pattern))
    retained = [insight for insight in projected["insights"] if insight.get("source") == "success_distillation"]
    if not retained:
        raise ValueError(f"{pattern.get('pattern_id')} has no success_distillation insight")
    projected["insights"] = retained
    retained_ids = {item["insight_id"] for item in retained}
    contract = projected.get("execution_contract")
    if isinstance(contract, dict):
        contract["insight_adaptations"] = [
            item for item in contract.get("insight_adaptations", []) if item.get("source_insight_id") in retained_ids
        ]
    return projected


def _diagnostic_library(
    source: Mapping[str, Any],
    *,
    stage: str,
    patterns: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": source["schema_version"],
        "induction_version": source["induction_version"],
        "library_id": "pending",
        "parent_dsus_build_id": source["parent_dsus_build_id"],
        "patterns": [copy.deepcopy(dict(item)) for item in patterns],
    }
    identity = {
        "stage": stage,
        "parent_dsus_build_id": payload["parent_dsus_build_id"],
        "patterns": payload["patterns"],
    }
    payload["library_id"] = f"diagnostic-pattern-library-{content_sha256(identity)[:16]}"
    StateConditionedPatternLibrary.model_validate(payload)
    return payload


def _checkpoint_stats(library: Mapping[str, Any]) -> dict[str, Any]:
    patterns = library["patterns"]
    return {
        "pattern_count": len(patterns),
        "strategy_family_count": len({item["strategy_family"] for item in patterns}),
        "sop_step_count": sum(len(item["sop_steps"]) for item in patterns),
        "insight_count": sum(len(item["insights"]) for item in patterns),
        "success_insight_count": sum(
            insight.get("source") == "success_distillation" for item in patterns for insight in item["insights"]
        ),
        "correction_insight_count": sum(
            insight.get("source") == "off_policy_failure_correction"
            for item in patterns
            for insight in item["insights"]
        ),
        "supporting_expert_episode_sum": sum(int(item.get("supporting_expert_episode_count", 0)) for item in patterns),
        "supporting_correction_state_sum": sum(
            int(item.get("supporting_correction_state_count", 0)) for item in patterns
        ),
    }


def build_staged_checkpoints(source: LoadedPatternSource) -> list[dict[str, Any]]:
    """Create four explicitly qualified repository checkpoints."""

    all_patterns = list(source.library["patterns"])
    seed_patterns = [_filter_success_insights(item) for item in all_patterns if item.get("legacy_seed_refs")]
    success_patterns = [
        _filter_success_insights(item)
        for item in all_patterns
        if int(item.get("supporting_expert_episode_count", 0)) > 0
    ]
    if not seed_patterns or not success_patterns:
        raise ValueError("Existing Pattern fields are insufficient for the seed/+success diagnostic projection")

    seed_library = _diagnostic_library(source.library, stage="seed", patterns=seed_patterns)
    success_library = _diagnostic_library(
        source.library,
        stage="+success",
        patterns=success_patterns,
    )
    correction_library = copy.deepcopy(source.library)
    StateConditionedPatternLibrary.model_validate(correction_library)
    gated_library = copy.deepcopy(correction_library)

    raw = (
        (
            "seed",
            seed_library,
            "retrospective_diagnostic_projection",
            "legacy_seed_refs_present AND success_distillation insights only",
            [
                "not_a_historical_training_checkpoint",
                "final_SOP_structure_cannot_be_temporally_rolled_back",
            ],
        ),
        (
            "+success",
            success_library,
            "retrospective_diagnostic_projection",
            "supporting_expert_episode_count > 0 AND success_distillation insights only",
            [
                "not_a_historical_training_checkpoint",
                "final_SOP_structure_cannot_be_temporally_rolled_back",
            ],
        ),
        (
            "+correction",
            correction_library,
            "observed_final_induction_artifact",
            "exact verified pattern_library.public.json payload",
            ["chronological_candidate_admission_log_not_present_in_source_artifact"],
        ),
        (
            "full-gated",
            gated_library,
            "diagnostic_identity_checkpoint",
            "byte-identical to +correction because no admitted gated candidate is evidenced",
            [
                "paired_target_boundary_retention_replay_not_available",
                "no_post_correction_admission_may_be_claimed",
            ],
        ),
    )
    checkpoints: list[dict[str, Any]] = []
    parent_id: str | None = None
    parent_hash: str | None = None
    for ordinal, (stage, library, evidence_status, rule, limitations) in enumerate(raw):
        payload_hash = content_sha256(library)
        checkpoint_id = f"pattern-repository-{ordinal:02d}-{payload_hash[:16]}"
        identical_to_parent = parent_hash == payload_hash
        checkpoints.append(
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "checkpoint_id": checkpoint_id,
                "stage": stage,
                "ordinal": ordinal,
                "parent_checkpoint_id": parent_id,
                "evidence_status": evidence_status,
                "selection_rule": rule,
                "library": library,
                "library_payload_sha256": payload_hash,
                "identical_to_parent": identical_to_parent,
                "stats": _checkpoint_stats(library),
                "limitations": limitations,
            }
        )
        parent_id = checkpoint_id
        parent_hash = payload_hash
    return checkpoints


def _procedure_signature(pattern: Mapping[str, Any]) -> dict[str, Any]:
    contract = pattern.get("execution_contract")
    return {
        "sop_steps": pattern.get("sop_steps"),
        "forbidden_operations": pattern.get("forbidden_operations"),
        "fallback_policy": pattern.get("fallback_policy"),
        "execution_contract": contract,
    }


def _semantic_signature_payload(pattern: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(pattern))
    for key in (
        "pattern_id",
        "version",
        "provenance",
        "legacy_seed_refs",
        "supporting_expert_episode_count",
        "supporting_correction_state_count",
        "covered_state_count",
        "covered_archetype_count",
    ):
        payload.pop(key, None)
    for insight in payload.get("insights", []):
        insight.pop("evidence_signature", None)
        insight.pop("support_count", None)
    for signature in payload.get("execution_signatures", []):
        signature.pop("support_count", None)
        signature.pop("source", None)
    return payload


def _candidate_signature(pattern: Mapping[str, Any]) -> str:
    return content_sha256(_semantic_signature_payload(pattern))


def _candidate_wrapper(
    pattern: Mapping[str, Any],
    *,
    corruption_type: str,
    ordinal: int,
    mutation: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_pattern = copy.deepcopy(dict(pattern))
    candidate_id = (
        f"corrupt-{corruption_type}-{ordinal:03d}-"
        f"{content_sha256({'parent': pattern['pattern_id'], 'mutation': mutation})[:12]}"
    )
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "candidate_kind": "deterministic_corruption_injection",
        "corruption_type": corruption_type,
        "parent_pattern_id": pattern["pattern_id"],
        "mutation": dict(mutation),
        "candidate_pattern": candidate_pattern,
    }


def _flip_action(action: str) -> str:
    mapping = {
        "ask": "recommend",
        "recommend": "ask",
        "explain": "recommend",
        "propose_relax": "recommend",
        "stop": "ask",
    }
    return mapping.get(action, "stop")


def generate_corruption_candidates(library: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Generate deterministic scope, action, leakage, and duplicate injections."""

    patterns = sorted(library["patterns"], key=lambda item: item["pattern_id"])
    candidates: list[dict[str, Any]] = []
    ordinal = 0
    for pattern in patterns:
        widened = _candidate_wrapper(
            pattern,
            corruption_type="scope_widening",
            ordinal=ordinal,
            mutation={"kind": "pending"},
        )
        applicability = widened["candidate_pattern"]["applicability"]
        widened_axis: str | None = None
        previous_value: Any = None
        for axis in APPLICABILITY_AXES:
            if applicability.get(axis):
                widened_axis = axis
                previous_value = copy.deepcopy(applicability[axis])
                applicability[axis] = []
                break
        if widened_axis is None and applicability.get("requires_explanation_pending") is not None:
            widened_axis = "requires_explanation_pending"
            previous_value = applicability[widened_axis]
            applicability[widened_axis] = None
        if widened_axis is None:
            raise ValueError(f"{pattern['pattern_id']} has no applicability scope that can be widened")
        widened["mutation"] = {
            "kind": "remove_applicability_guard",
            "axis": widened_axis,
            "before": previous_value,
            "after": applicability[widened_axis],
        }
        widened["declared_candidate_signature_sha256"] = _candidate_signature(widened["candidate_pattern"])
        candidates.append(widened)
        ordinal += 1

    for pattern in patterns:
        templates = pattern.get("execution_contract", {}).get("sop_action_templates", [])
        if not templates:
            continue
        flipped = _candidate_wrapper(
            pattern,
            corruption_type="action_flip",
            ordinal=ordinal,
            mutation={"kind": "flip_first_typed_action"},
        )
        template = flipped["candidate_pattern"]["execution_contract"]["sop_action_templates"][0]
        before = template["action_type"]
        template["action_type"] = _flip_action(before)
        flipped["mutation"] = {
            "kind": "flip_first_typed_action",
            "template_id": template["template_id"],
            "before": before,
            "after": template["action_type"],
        }
        flipped["declared_candidate_signature_sha256"] = _candidate_signature(flipped["candidate_pattern"])
        candidates.append(flipped)
        ordinal += 1

    for pattern in patterns:
        leaked = _candidate_wrapper(
            pattern,
            corruption_type="private_field",
            ordinal=ordinal,
            mutation={"kind": "inject_private_state_key"},
        )
        leaked["candidate_pattern"]["private_user_state"] = PRIVATE_VALUE_SENTINEL
        leaked["declared_candidate_signature_sha256"] = _candidate_signature(leaked["candidate_pattern"])
        candidates.append(leaked)
        ordinal += 1

    for pattern in patterns:
        duplicate = _candidate_wrapper(
            pattern,
            corruption_type="duplicate_copy",
            ordinal=ordinal,
            mutation={"kind": "semantic_copy_with_same_payload"},
        )
        duplicate["declared_candidate_signature_sha256"] = _candidate_signature(duplicate["candidate_pattern"])
        candidates.append(duplicate)
        ordinal += 1
    return candidates


def _find_private_paths(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            child_path = f"{path}.{key}"
            if any(marker in lowered for marker in PRIVATE_KEY_MARKERS):
                findings.append(child_path)
            findings.extend(_find_private_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_private_paths(child, f"{path}[{index}]"))
    elif isinstance(value, str) and PRIVATE_VALUE_SENTINEL in value:
        findings.append(path)
    return sorted(set(findings))


def evaluate_candidate_gates(
    library: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply structural gates; behavioral gates remain explicitly pending."""

    patterns = {item["pattern_id"]: item for item in library["patterns"]}
    repository_signatures: dict[str, list[str]] = defaultdict(list)
    for pattern in patterns.values():
        repository_signatures[_candidate_signature(pattern)].append(pattern["pattern_id"])

    decisions: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", ""))
        parent_id = str(candidate.get("parent_pattern_id", ""))
        candidate_pattern = candidate.get("candidate_pattern")
        if not candidate_id or not isinstance(candidate_pattern, dict):
            raise ValueError("Each candidate must provide candidate_id and candidate_pattern")
        parent = patterns.get(parent_id)

        procedure_pass = parent is not None and canonical_json(_procedure_signature(parent)) == canonical_json(
            _procedure_signature(candidate_pattern)
        )
        private_paths = _find_private_paths(candidate_pattern)
        actual_signature = _candidate_signature(candidate_pattern)
        declared_signature = candidate.get("declared_candidate_signature_sha256")
        signature_matches = declared_signature == actual_signature
        duplicate_ids = repository_signatures.get(actual_signature, [])
        is_unique = not duplicate_ids
        gate_results = [
            {
                "gate": "procedure_invariant",
                "status": "pass" if procedure_pass else "fail",
                "reason": (
                    "candidate_preserves_parent_SOP_and_execution_contract"
                    if procedure_pass
                    else "candidate_changes_or_lacks_parent_procedure"
                ),
            },
            {
                "gate": "private_leakage",
                "status": "pass" if not private_paths else "fail",
                "reason": (
                    "no_private_key_or_sentinel_detected"
                    if not private_paths
                    else "private_paths=" + ",".join(private_paths)
                ),
            },
            {
                "gate": "declared_signature",
                "status": "pass" if signature_matches else "fail",
                "reason": (
                    "declared_signature_matches_candidate"
                    if signature_matches
                    else "declared_signature_missing_or_mismatched"
                ),
            },
            {
                "gate": "repository_duplicate",
                "status": "pass" if is_unique else "fail",
                "reason": (
                    "candidate_signature_is_new" if is_unique else "duplicates=" + ",".join(sorted(duplicate_ids))
                ),
            },
        ]
        structural_pass = all(item["status"] == "pass" for item in gate_results)
        behavioral_gates = [
            {
                "gate": name,
                "status": "pending" if structural_pass else "not_run",
                "reason": (
                    "requires_source_disjoint_behavior_replay" if structural_pass else "blocked_by_structural_gate"
                ),
            }
            for name in ("target_gain", "boundary_non_regression", "retained_replay")
        ]
        decisions.append(
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "corruption_type": candidate.get("corruption_type"),
                "parent_pattern_id": parent_id,
                "candidate_signature_sha256": actual_signature,
                "structural_gates": gate_results,
                "behavioral_gates": behavioral_gates,
                "decision": ("pending_behavioral_replay" if structural_pass else "reject_structural"),
                "admitted": False,
                "decision_reason": (
                    "structural_checks_pass_but_behavioral_evidence_is_missing"
                    if structural_pass
                    else "one_or_more_structural_gates_failed"
                ),
            }
        )
    return decisions


def _decision_index(decision: Mapping[str, Any], fallback: int) -> int:
    value = decision.get("decision_index", decision.get("turn_index", fallback))
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("decision_index/turn_index must be an integer")
    return value


def _first_divergence(record: Mapping[str, Any], decisions: Sequence[Mapping[str, Any]]) -> int | None:
    explicit = record.get(
        "first_consequential_divergence_index",
        record.get("first_divergence_index"),
    )
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int):
            raise ValueError("first consequential divergence must be an integer")
        return explicit
    for fallback, decision in enumerate(decisions):
        if decision.get(
            "is_consequential_divergence",
            decision.get("consequential_divergence", False),
        ):
            return _decision_index(decision, fallback)
    actual_keys = ("executed_action", "observed_action", "policy_action", "actual_action")
    reference_keys = ("reference_action", "corrected_action", "oracle_action")
    for fallback, decision in enumerate(decisions):
        actual = next((decision[key] for key in actual_keys if decision.get(key) is not None), None)
        reference = next(
            (decision[key] for key in reference_keys if decision.get(key) is not None),
            None,
        )
        if actual is not None and reference is not None and actual != reference and decision.get("consequential", True):
            return _decision_index(decision, fallback)
    return None


def _normalize_strategy(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"local", "first_divergence", "local_first_divergence"}:
        return "local_first_divergence"
    if normalized in {"whole", "episode", "whole_episode", "whole_route"}:
        return "whole_route"
    return None


def _target_indices(candidate: Mapping[str, Any]) -> list[int]:
    values = candidate.get("target_decision_indices")
    if values is None:
        single = candidate.get(
            "target_decision_index",
            candidate.get("target_turn_index"),
        )
        values = [] if single is None else [single]
    if not isinstance(values, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("target_decision_indices must be an array of integers")
    return sorted(set(values))


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return fmean(materialized) if materialized else None


def audit_locality(
    trajectory_records: Sequence[Mapping[str, Any]] | None,
    candidate_records: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Audit local-first-divergence targeting against whole-route updates."""

    if trajectory_records is None or candidate_records is None:
        missing = []
        if trajectory_records is None:
            missing.append("trajectory_log")
        if candidate_records is None:
            missing.append("candidate_log")
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "status": "pending_missing_inputs",
            "missing_inputs": missing,
            "measurements": None,
            "claim_boundary": "no_locality_measurement_available",
        }

    trajectories: dict[str, dict[str, Any]] = {}
    for record in trajectory_records:
        trajectory_id = str(record.get("trajectory_id", record.get("pair_id", "")))
        decisions = record.get("decisions", record.get("route"))
        if not trajectory_id or not isinstance(decisions, list) or not decisions:
            raise ValueError("trajectory record must have trajectory_id and nonempty decisions/route")
        indexes = [_decision_index(item, offset) for offset, item in enumerate(decisions)]
        if len(indexes) != len(set(indexes)):
            raise ValueError(f"{trajectory_id} has duplicate decision indices")
        first = _first_divergence(record, decisions)
        trajectories[trajectory_id] = {
            "trajectory_id": trajectory_id,
            "source_group_id": record.get("source_group_id"),
            "decision_indices": sorted(indexes),
            "first_consequential_divergence_index": first,
        }

    rows: list[dict[str, Any]] = []
    for candidate in candidate_records:
        candidate_id = str(candidate.get("candidate_id", ""))
        trajectory_id = str(candidate.get("trajectory_id", candidate.get("pair_id", "")))
        if not candidate_id or trajectory_id not in trajectories:
            raise ValueError(f"candidate {candidate_id or '<missing>'} cannot be joined to a trajectory")
        trajectory = trajectories[trajectory_id]
        route_indices = trajectory["decision_indices"]
        first = trajectory["first_consequential_divergence_index"]
        strategy = _normalize_strategy(candidate.get("update_strategy", candidate.get("credit_assignment")))
        actual_targets = _target_indices(candidate)
        if first is None:
            status = "no_consequential_divergence"
            expected_targets: list[int] = []
            compliant: bool | None = None
        elif strategy is None:
            status = "unknown_update_strategy"
            expected_targets = []
            compliant = False
        else:
            expected_targets = [first] if strategy == "local_first_divergence" else route_indices
            compliant = actual_targets == expected_targets
            status = "compliant" if compliant else "noncompliant"
        off_divergence_count = None if first is None else sum(index != first for index in actual_targets)
        rows.append(
            {
                "candidate_id": candidate_id,
                "trajectory_id": trajectory_id,
                "source_group_id": candidate.get(
                    "source_group_id",
                    trajectory.get("source_group_id"),
                ),
                "update_strategy": strategy,
                "route_decision_count": len(route_indices),
                "first_consequential_divergence_index": first,
                "actual_target_decision_indices": actual_targets,
                "expected_target_decision_indices": expected_targets,
                "actual_target_count": len(actual_targets),
                "actual_blast_radius": len(actual_targets) / len(route_indices),
                "off_divergence_target_count": off_divergence_count,
                "local_expected_target_count": 1 if first is not None else None,
                "whole_route_expected_target_count": (len(route_indices) if first is not None else None),
                "counterfactual_target_reduction": (len(route_indices) - 1 if first is not None else None),
                "status": status,
                "compliant": compliant,
            }
        )

    by_strategy: dict[str, dict[str, Any]] = {}
    for strategy in ("local_first_divergence", "whole_route", "unknown"):
        selected = [row for row in rows if (row["update_strategy"] or "unknown") == strategy]
        if not selected:
            continue
        comparable = [row for row in selected if row["compliant"] is not None]
        by_strategy[strategy] = {
            "candidate_count": len(selected),
            "comparable_count": len(comparable),
            "compliance_rate": _mean(float(row["compliant"]) for row in comparable),
            "mean_actual_target_count": _mean(float(row["actual_target_count"]) for row in selected),
            "mean_actual_blast_radius": _mean(float(row["actual_blast_radius"]) for row in selected),
            "mean_off_divergence_target_count": _mean(
                float(row["off_divergence_target_count"])
                for row in selected
                if row["off_divergence_target_count"] is not None
            ),
        }
    divergent_routes = [
        item for item in trajectories.values() if item["first_consequential_divergence_index"] is not None
    ]
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "measured_from_supplied_logs",
        "input_counts": {
            "trajectory_count": len(trajectories),
            "candidate_count": len(candidate_records),
            "trajectory_with_consequential_divergence_count": len(divergent_routes),
        },
        "counterfactual_structure": {
            "mean_local_expected_target_count": _mean(1.0 for _ in divergent_routes),
            "mean_whole_route_expected_target_count": _mean(
                float(len(item["decision_indices"])) for item in divergent_routes
            ),
            "mean_target_count_reduction": _mean(float(len(item["decision_indices"]) - 1) for item in divergent_routes),
        },
        "by_strategy": by_strategy,
        "rows": rows,
        "claim_boundary": (
            "structural_credit_assignment_locality_only;behavioral_gain_requires_source_disjoint_paired_replay"
        ),
    }


def _checkpoint_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in checkpoint.items() if key != "library"}


def _jsonl_text(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(record) + "\n" for record in records)


def build_pattern_evolution_evidence(
    *,
    pattern_artifact_dir: Path,
    output_dir: Path,
    trajectory_log: Path | None = None,
    candidate_log: Path | None = None,
    family_runtime_mapping_path: Path | None = None,
    runtime_library_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build staged repositories, corruption gates, locality audit, and lineage."""

    source = load_pattern_source(pattern_artifact_dir)
    project_root = Path(__file__).resolve().parents[3]
    family_runtime_mapping_path = family_runtime_mapping_path or (
        project_root / "configs/pattern_family_runtime_mapping_v1.json"
    )
    runtime_library_path = runtime_library_path or (
        project_root / "src/taira_mt/mt_agent/thought_pattern_skills_v1.json"
    )
    family_runtime_mapping = load_family_runtime_mapping(
        mapping_path=family_runtime_mapping_path,
        paper_library_path=source.artifact_dir / "pattern_library.public.json",
        runtime_library_path=runtime_library_path,
    )
    if progress is not None:
        progress(
            "Step 1/4 complete: validated "
            f"{len(source.library['patterns'])} paper procedure families and "
            f"{len(family_runtime_mapping.runtime_to_family)} internal runtime entries; "
            "these form a many-to-many implementation mapping and must not be added together. external_calls=False throughout."
        )
        progress("Step 2/4: freezing four levels of machine-readable lineage: seed, +success, +correction, and full-gated.")
    checkpoints = build_staged_checkpoints(source)
    if progress is not None:
        progress(
            "Step 2/4 complete: Pattern counts in order: "
            + " / ".join(str(item["stats"]["pattern_count"]) for item in checkpoints)
            + "; full-gated has no actual replay evidence and is therefore marked as identical to +correction."
        )
        progress("Step 3/4: injecting scope-widening, action-flip, private-field, and duplicate controls.")
    corruptions = generate_corruption_candidates(source.library)
    gate_decisions = evaluate_candidate_gates(source.library, corruptions)
    if progress is not None:
        progress(
            "Step 3/4 complete: generated "
            f"{len(corruptions)} deterministic candidates; "
            f"gate results={dict(sorted(Counter(item['decision'] for item in gate_decisions).items()))}."
        )
        progress("Step 4/4: summarizing structural locality for first-divergence local updates and whole-route updates.")
    trajectories = _read_json_records(trajectory_log) if trajectory_log is not None else None
    candidates = _read_json_records(candidate_log) if candidate_log is not None else None
    locality = audit_locality(trajectories, candidates)

    output_dir = output_dir.resolve()
    files: dict[str, str] = {}
    stage_dirs = ("00_seed", "01_success", "02_correction", "03_full_gated")
    for stage_dir, checkpoint in zip(stage_dirs, checkpoints, strict=True):
        files[f"checkpoints/{stage_dir}/pattern_library.public.json"] = _json_text(checkpoint["library"])
        files[f"checkpoints/{stage_dir}/checkpoint.json"] = _json_text(_checkpoint_metadata(checkpoint))
    lineage = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "source": {
            "artifact_dir": str(source.artifact_dir),
            "induction_id": source.manifest.get("induction_id"),
            "library_id": source.library.get("library_id"),
            "manifest_file_sha256": source.manifest_file_sha256,
            "library_file_sha256": source.library_file_sha256,
            "library_content_sha256": source.library_content_sha256,
            "family_runtime_mapping_sha256": (family_runtime_mapping.mapping_content_sha256),
            "runtime_entry_library_sha256": (family_runtime_mapping.runtime_content_sha256),
        },
        "checkpoint_order": [
            {
                "checkpoint_id": item["checkpoint_id"],
                "stage": item["stage"],
                "parent_checkpoint_id": item["parent_checkpoint_id"],
                "library_payload_sha256": item["library_payload_sha256"],
                "identical_to_parent": item["identical_to_parent"],
                "evidence_status": item["evidence_status"],
            }
            for item in checkpoints
        ],
        "test_isolation_contract": {
            "repository_writeback_during_test": False,
            "required_runtime_behavior": "pin_checkpoint_id_and_library_payload_sha256",
            "admission_sources": ["train", "development"],
            "forbidden_admission_sources": [
                "held_out_test_episode",
                "held_out_test_metric",
                "held_out_test_human_judgment",
            ],
        },
    }
    gate_counts = Counter(item["decision"] for item in gate_decisions)
    gate_counts_by_corruption: dict[str, Counter[str]] = defaultdict(Counter)
    for item in gate_decisions:
        gate_counts_by_corruption[str(item["corruption_type"])][item["decision"]] += 1
    summary = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "source_pattern_count": len(source.library["patterns"]),
        "checkpoint_stats": {item["stage"]: item["stats"] for item in checkpoints},
        "checkpoint_evidence_status": {item["stage"]: item["evidence_status"] for item in checkpoints},
        "family_runtime_granularity": {
            "mapping_id": family_runtime_mapping.payload["mapping_id"],
            **family_runtime_mapping.payload["count_contract"],
            "cardinality": family_runtime_mapping.payload["granularity_contract"]["cardinality"],
            "counts_are_not_additive": family_runtime_mapping.payload["granularity_contract"][
                "counts_are_not_additive"
            ],
            "relation_counts": family_runtime_mapping.relation_counts,
            "grounded_explanation_runtime_status": next(
                item["runtime_realization_status"]
                for item in family_runtime_mapping.payload["paper_procedural_families"]
                if item["family_id"] == "grounded_explanation"
            ),
            "claim_boundary": family_runtime_mapping.payload["granularity_contract"]["claim_boundary"],
        },
        "full_gated_identical_to_correction": (
            checkpoints[3]["library_payload_sha256"] == checkpoints[2]["library_payload_sha256"]
        ),
        "corruption_candidate_count": len(corruptions),
        "corruption_type_counts": dict(sorted(Counter(item["corruption_type"] for item in corruptions).items())),
        "gate_decision_counts": dict(sorted(gate_counts.items())),
        "gate_decisions_by_corruption": {
            key: dict(sorted(value.items())) for key, value in sorted(gate_counts_by_corruption.items())
        },
        "locality_status": locality["status"],
        "external_calls": False,
        "claim_boundaries": [
            "seed_and_success_are_retrospective_diagnostic_projections_not_historical_checkpoints",
            "full_gated_is_identical_to_correction_until_real_replay_evidence_admits_a_candidate",
            "pending_behavioral_gates_are_not_passes",
            "locality_statistics_do_not_establish_behavioral_gain",
        ],
    }
    files["lineage.json"] = _json_text(lineage)
    files["family_runtime_mapping.json"] = _json_text(family_runtime_mapping.payload)
    files["family_runtime_views.json"] = _json_text(
        {
            "schema_version": family_runtime_mapping.payload["schema_version"],
            "mapping_id": family_runtime_mapping.payload["mapping_id"],
            "family_to_runtime": family_runtime_mapping.family_to_runtime,
            "runtime_to_family": family_runtime_mapping.runtime_to_family,
            "relation_counts": family_runtime_mapping.relation_counts,
        }
    )
    files["corruption_candidates.jsonl"] = _jsonl_text(corruptions)
    files["gate_decisions.jsonl"] = _jsonl_text(gate_decisions)
    files["locality_audit.json"] = _json_text(locality)
    files["summary.json"] = _json_text(summary)

    for relative_path, text in sorted(files.items()):
        _write_idempotently(output_dir / relative_path, text)
    manifest = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "artifact_role": "offline_pattern_repository_evolution_evidence",
        "source_manifest_sha256": source.manifest_file_sha256,
        "source_library_sha256": source.library_content_sha256,
        "source_family_runtime_mapping_sha256": (family_runtime_mapping.mapping_content_sha256),
        "source_runtime_entry_library_sha256": (family_runtime_mapping.runtime_content_sha256),
        "external_calls": False,
        "outputs": [
            {
                "relative_path": relative_path,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "byte_size": len(text.encode("utf-8")),
                "line_count": text.count("\n") if relative_path.endswith(".jsonl") else None,
            }
            for relative_path, text in sorted(files.items())
        ],
    }
    _write_idempotently(output_dir / "manifest.json", _json_text(manifest))
    if progress is not None:
        if locality["status"] == "pending_missing_inputs":
            progress("Step 4/4 complete: trajectory/candidate logs are incomplete; locality is explicitly pending, with no fabricated measurements.")
        else:
            counts = locality["input_counts"]
            progress(
                "Step 4/4 complete: joined "
                f"{counts['trajectory_count']} trajectories and "
                f"{counts['candidate_count']} candidates."
            )
    return {
        "output_dir": output_dir,
        "summary": summary,
        "lineage": lineage,
        "locality": locality,
    }
