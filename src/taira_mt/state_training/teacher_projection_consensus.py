"""Strict offline consensus for three full-state DSUS teacher projections.

This module never calls a model and never reads private gold.  It replays raw
teacher JSON against the public prompt, permits only a quote-preserving
``catalog``/``system`` provenance alias normalization, and audits whether
cross-teacher differences can change the downstream state-machine action.
Terra remains the primary snapshot; exact signature equality is retained only
as a high-confidence diagnostic because open semantic slot names vary.

For prefix trajectories, a second pass checks that constraints and rejected
items persist.  A later snapshot that silently deletes either family without a
public revision signal is routed to disagreement instead of being accepted.
The accepted snapshot is intentionally still a full state: deterministic
adjacent-state differencing belongs to the next pipeline stage.
"""

from __future__ import annotations

import itertools
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, JsonValue, model_validator

from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json, content_sha256
from taira_mt.state_compiler import PublicDSUSState
from taira_mt.state_training.dataset_assembly import PublicRLPromptRecord
from taira_mt.state_training.teacher_projection import (
    TeacherCurrentStateSnapshot,
    TeacherProjectionRawRecord,
    TeacherProjectionValidationRecord,
    parse_teacher_projection,
    teacher_projection_user_payload,
)

TEACHER_CONSENSUS_VERSION = "dsus_teacher_snapshot_consensus_v1"
TEACHER_CONSENSUS_SUMMARY_VERSION = "dsus_teacher_snapshot_consensus_summary_v1"

ConsensusStatus = Literal["accepted", "disagreement", "rejected"]
EffectiveTeacherStatus = Literal[
    "valid",
    "valid_after_source_alias_normalization",
    "valid_after_unreferenced_evidence_pruning",
    "valid_after_safe_normalization",
    "invalid",
    "transport_error",
    "artifact_integrity_error",
]


def _strict_json_object(content: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    payload = json.loads(content, object_pairs_hook=no_duplicates)
    if not isinstance(payload, dict):
        raise ValueError("teacher output must be one JSON object")
    return payload


def _public_observations(public: PublicRLPromptRecord) -> list[dict[str, JsonValue]]:
    payload = teacher_projection_user_payload(public.public_prompt)
    observations = payload.get("public_history")
    if observations is None:
        observations = payload.get("latest_observations")
    if not isinstance(observations, list):  # pragma: no cover - public prompt schema guarantees it
        raise ValueError("public prompt has no observations")
    return [dict(item) for item in observations if isinstance(item, dict)]


def normalize_catalog_system_provenance(
    content: str,
    *,
    public: PublicRLPromptRecord,
) -> tuple[str, bool]:
    """Normalize only exact public quote aliases between catalog and system.

    ``user`` provenance is immutable.  A reference is changed only when its
    exact quote does not occur at the declared source/turn and does occur at
    the opposite public source.  Resolution prefers the same turn, then a
    unique occurrence, then the nearest occurrence no later than the cited
    turn.  No semantic field, quote text, or evidence ID is edited.
    """

    try:
        payload = _strict_json_object(content)
    except Exception:
        return content, False
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        return content, False

    observations: list[tuple[int, str, str]] = []
    for observation in _public_observations(public):
        turn_id = observation.get("turn_id")
        role = observation.get("role")
        text = observation.get("text")
        if isinstance(turn_id, int) and isinstance(role, str) and isinstance(text, str):
            observations.append((turn_id, role, text))

    role_groups = {
        "system": {"system"},
        "catalog": {"catalog", "tool"},
    }
    changed = False
    for reference in evidence:
        if not isinstance(reference, dict):
            continue
        source = reference.get("source")
        turn_id = reference.get("turn_id")
        quote = reference.get("quote")
        if source not in role_groups or not isinstance(turn_id, int):
            continue
        if not isinstance(quote, str) or not quote:
            continue
        declared_roles = role_groups[source]
        if any(
            candidate_turn == turn_id and role in declared_roles and quote in text
            for candidate_turn, role, text in observations
        ):
            continue
        opposite = "catalog" if source == "system" else "system"
        candidates = [
            candidate_turn
            for candidate_turn, role, text in observations
            if role in role_groups[opposite] and quote in text
        ]
        resolved_turn: int | None = None
        if turn_id in candidates:
            resolved_turn = turn_id
        elif len(candidates) == 1:
            resolved_turn = candidates[0]
        else:
            prior = [candidate for candidate in candidates if candidate <= turn_id]
            if prior:
                resolved_turn = max(prior)
        if resolved_turn is None:
            continue
        reference["source"] = opposite
        reference["turn_id"] = resolved_turn
        changed = True
    return (canonical_json(payload) if changed else content), changed


def prune_unreferenced_evidence(content: str) -> tuple[str, int]:
    """Delete only evidence objects whose IDs are referenced by no state fact.

    This is a structural no-op: every removed object is unreachable from
    control, constraints, axes, rejected items, and termination.  The function
    never edits a quote or any semantic field, and malformed evidence entries
    are retained so strict schema validation still catches them.
    """

    try:
        payload = _strict_json_object(content)
    except Exception:
        return content, 0
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        return content, 0

    referenced: set[str] = set()

    def collect(values: object) -> None:
        if isinstance(values, list) and all(isinstance(value, str) for value in values):
            referenced.update(values)

    control_evidence = payload.get("control_evidence")
    if isinstance(control_evidence, dict):
        for values in control_evidence.values():
            collect(values)
    for key in ("constraints", "unresolved_axes", "rejected_items"):
        collection = payload.get(key)
        if isinstance(collection, list):
            for item in collection:
                if isinstance(item, dict):
                    collect(item.get("evidence_ids"))
    collect(payload.get("termination_evidence_ids"))

    kept: list[object] = []
    pruned = 0
    seen_ids: set[str] = set()
    for item in evidence:
        evidence_id = item.get("evidence_id") if isinstance(item, dict) else None
        # Keep malformed or duplicate entries for strict validation to reject.
        if not isinstance(evidence_id, str) or not evidence_id or evidence_id in seen_ids:
            kept.append(item)
            continue
        seen_ids.add(evidence_id)
        if evidence_id in referenced:
            kept.append(item)
        else:
            pruned += 1
    if not pruned:
        return content, 0
    payload["evidence"] = kept
    return canonical_json(payload), pruned


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _normalized_json(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return _normalized_text(value)
    if isinstance(value, list):
        # Array order may carry meaning in open values/options, so preserve it.
        return [_normalized_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalized_json(value[key]) for key in sorted(value)}
    return value


class CanonicalSnapshotSignature(StrictModel):
    """Lexical/structural signature; IDs and evidence-reference IDs are excluded."""

    control: str
    key_control: str
    constraints: list[str]
    unresolved_axes: list[str]
    rejected_items: list[str]
    rejected_item_ids: list[str]
    termination: str
    full_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def canonical_snapshot_signature(
    snapshot: TeacherCurrentStateSnapshot,
) -> CanonicalSnapshotSignature:
    control = canonical_json(snapshot.control.model_dump(mode="json"))
    key_control = canonical_json(
        {
            "interaction_phase": snapshot.control.interaction_phase.value,
            "information_status": snapshot.control.information_status.value,
            "feedback_type": snapshot.control.feedback_type.value,
            "explanation_status": snapshot.control.explanation_status.value,
        }
    )
    constraints = sorted(
        canonical_json(
            _normalized_json(
                {
                    "name": item.name,
                    "value": item.value,
                    "polarity": item.polarity.value,
                    "importance": item.importance.value,
                    "status": item.status.value,
                }
            )
        )
        for item in snapshot.constraints
    )
    unresolved_axes = sorted(
        canonical_json(
            _normalized_json(
                {
                    "axis_type": item.axis_type,
                    "description": item.description,
                    "options": item.options,
                }
            )
        )
        for item in snapshot.unresolved_axes
    )
    rejected_items = sorted(
        canonical_json(
            _normalized_json(
                {
                    # Public item identifiers are opaque and remain exact.
                    "item_id": item.item_id,
                    "reason": item.reason,
                }
            )
        )
        for item in snapshot.rejected_items
    )
    rejected_item_ids = sorted(item.item_id for item in snapshot.rejected_items)
    termination = canonical_json(
        {"reason": (None if snapshot.termination_reason is None else snapshot.termination_reason.value)}
    )
    semantic_payload = {
        "control": control,
        "key_control": key_control,
        "constraints": constraints,
        "unresolved_axes": unresolved_axes,
        "rejected_items": rejected_items,
        "rejected_item_ids": rejected_item_ids,
        "termination": termination,
    }
    return CanonicalSnapshotSignature(
        **semantic_payload,
        full_sha256=content_sha256(semantic_payload),
    )


class TeacherProjectionAudit(StrictModel):
    teacher_id: str = Field(min_length=1)
    declared_status: Literal["valid", "invalid", "transport_error"]
    effective_status: EffectiveTeacherStatus
    source_alias_normalized: bool = False
    unreferenced_evidence_pruned_count: int = Field(default=0, ge=0)
    failure_code: str | None = None
    signature: CanonicalSnapshotSignature | None = None

    @model_validator(mode="after")
    def validate_signature_presence(self) -> TeacherProjectionAudit:
        valid = self.effective_status.startswith("valid")
        if valid != (self.signature is not None):
            raise ValueError("only effective-valid teachers may carry signatures")
        return self


class ConsensusReason(StrictModel):
    code: str = Field(min_length=1)
    teacher_ids: list[str] = Field(default_factory=list)
    detail: dict[str, JsonValue] = Field(default_factory=dict)


class SignatureGroup(StrictModel):
    full_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    teacher_ids: list[str] = Field(min_length=1)


class ConstraintFacetAgreement(StrictModel):
    matched_count: int = Field(ge=0)
    left_count: int = Field(ge=0)
    right_count: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    material_contradictions: list[str] = Field(default_factory=list)


class TeacherPairFacetAgreement(StrictModel):
    teacher_ids: list[str] = Field(min_length=2, max_length=2)
    key_control_exact: bool
    # Optional defaults preserve read compatibility with the early real12
    # diagnostic artifacts written before decision materiality was added.
    decision_action_left: Literal["ask", "recommend", "repair", "explain", "stop"] | None = None
    decision_action_right: Literal["ask", "recommend", "repair", "explain", "stop"] | None = None
    decision_action_exact: bool | None = None
    patience_alert_exact: bool | None = None
    termination_exact: bool
    rejected_item_ids_exact: bool
    constraints: ConstraintFacetAgreement
    material_conflicts: list[str] = Field(default_factory=list)
    soft_disagreements: list[str] = Field(default_factory=list)
    compatible: bool


class TeacherConsensusRecord(StrictModel):
    schema_version: Literal[TEACHER_CONSENSUS_VERSION] = TEACHER_CONSENSUS_VERSION
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trajectory_id: str = Field(min_length=1)
    source_turn_index: int = Field(ge=0)
    status: ConsensusStatus
    teacher_audits: list[TeacherProjectionAudit] = Field(min_length=2, max_length=4)
    signature_groups: list[SignatureGroup] = Field(default_factory=list)
    pair_agreements: list[TeacherPairFacetAgreement] = Field(default_factory=list)
    selected_teacher_id: str | None = None
    supporting_teacher_ids: list[str] = Field(default_factory=list)
    acceptance_tier: (
        Literal[
            "primary_valid_no_material_conflict",
            "cross_teacher_material_agreement",
            "third_teacher_arbitrated",
        ]
        | None
    ) = None
    consensus_snapshot: TeacherCurrentStateSnapshot | None = None
    reasons: list[ConsensusReason] = Field(min_length=1)
    persistence_checked: bool = False
    sol_arbitration_required: bool = False

    @model_validator(mode="after")
    def validate_outcome_contract(self) -> TeacherConsensusRecord:
        if self.status == "accepted":
            if self.consensus_snapshot is None or self.selected_teacher_id is None:
                raise ValueError("accepted consensus requires one selected snapshot")
            # Backward-compatible inference for exact-consensus artifacts made
            # before confidence tiers were materialized explicitly.
            if self.acceptance_tier is None and self.supporting_teacher_ids:
                self.acceptance_tier = (
                    "cross_teacher_material_agreement"
                    if len(self.supporting_teacher_ids) >= 2
                    else "primary_valid_no_material_conflict"
                )
            if not self.supporting_teacher_ids or self.acceptance_tier is None:
                raise ValueError("accepted consensus requires a confidence tier and support ledger")
            if self.sol_arbitration_required:
                raise ValueError("accepted consensus cannot require Sol arbitration")
        elif (
            self.consensus_snapshot is not None
            or self.selected_teacher_id is not None
            or self.acceptance_tier is not None
        ):
            raise ValueError("non-accepted consensus cannot expose a selected snapshot")
        if self.status == "disagreement" and not self.sol_arbitration_required:
            raise ValueError("teacher disagreement must be routed to Sol")
        return self


class TeacherConsensusSummary(StrictModel):
    schema_version: Literal[TEACHER_CONSENSUS_SUMMARY_VERSION] = TEACHER_CONSENSUS_SUMMARY_VERSION
    teacher_ids: list[str] = Field(min_length=2, max_length=4)
    selected_sample_count: int = Field(ge=1)
    outcome_counts: dict[str, int]
    acceptance_tier_counts: dict[str, int]
    effective_teacher_status_counts: dict[str, dict[str, int]]
    source_alias_normalization_count: int = Field(ge=0)
    unreferenced_evidence_pruned_count: int = Field(ge=0)
    exact_signature_quorum_count: int = Field(ge=0)
    persistence_pair_count: int = Field(ge=0)
    persistence_failure_count: int = Field(ge=0)
    sol_arbitration_count: int = Field(ge=0)
    private_gold_opened: Literal[False] = False
    external_model_calls: Literal[False] = False


class ThirdTeacherSelectionRecord(StrictModel):
    schema_version: Literal["dsus_third_teacher_selection_v1"] = "dsus_third_teacher_selection_v1"
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected: bool
    selection_reasons: list[str] = Field(default_factory=list)
    soft_disagreement_reasons: list[str] = Field(default_factory=list)
    teacher_audits: list[TeacherProjectionAudit] = Field(min_length=2, max_length=2)
    pair_agreement: TeacherPairFacetAgreement | None = None

    @model_validator(mode="after")
    def validate_selection_reasons(self) -> ThirdTeacherSelectionRecord:
        if self.selected and not self.selection_reasons:
            raise ValueError("selected third-teacher sample requires a reason")
        return self


def _effective_projection(
    *,
    teacher_id: str,
    public: PublicRLPromptRecord,
    raw: TeacherProjectionRawRecord,
    validation: TeacherProjectionValidationRecord,
) -> tuple[TeacherProjectionAudit, TeacherCurrentStateSnapshot | None]:
    declared = validation.status
    if declared == "transport_error":
        failure = "transport_error"
        if raw.model_output:
            failure = "transport_status_with_nonempty_raw"
            status: EffectiveTeacherStatus = "artifact_integrity_error"
        else:
            status = "transport_error"
        return (
            TeacherProjectionAudit(
                teacher_id=teacher_id,
                declared_status=declared,
                effective_status=status,
                failure_code=failure,
            ),
            None,
        )

    try:
        snapshot = parse_teacher_projection(
            raw.model_output,
            public_prompt=public.public_prompt,
        )
    except Exception as original_error:
        normalized_content, source_alias_normalized = normalize_catalog_system_provenance(
            raw.model_output,
            public=public,
        )
        normalized_content, pruned_count = prune_unreferenced_evidence(normalized_content)
        snapshot = None
        if source_alias_normalized or pruned_count:
            try:
                snapshot = parse_teacher_projection(
                    normalized_content,
                    public_prompt=public.public_prompt,
                )
            except Exception:
                snapshot = None
        if snapshot is None:
            status = "artifact_integrity_error" if declared == "valid" else "invalid"
            return (
                TeacherProjectionAudit(
                    teacher_id=teacher_id,
                    declared_status=declared,
                    effective_status=status,
                    source_alias_normalized=source_alias_normalized,
                    unreferenced_evidence_pruned_count=pruned_count,
                    failure_code=(
                        "declared_valid_revalidation_failed" if declared == "valid" else type(original_error).__name__
                    ),
                ),
                None,
            )
        # A valid declaration should already have passed strict public evidence
        # and reachability checks; needing repair indicates stale metadata.
        if declared == "valid":
            return (
                TeacherProjectionAudit(
                    teacher_id=teacher_id,
                    declared_status=declared,
                    effective_status="artifact_integrity_error",
                    source_alias_normalized=source_alias_normalized,
                    unreferenced_evidence_pruned_count=pruned_count,
                    failure_code="declared_valid_required_posthoc_normalization",
                ),
                None,
            )
        signature = canonical_snapshot_signature(snapshot)
        if source_alias_normalized and pruned_count:
            normalized_status: EffectiveTeacherStatus = "valid_after_safe_normalization"
        elif source_alias_normalized:
            normalized_status = "valid_after_source_alias_normalization"
        else:
            normalized_status = "valid_after_unreferenced_evidence_pruning"
        return (
            TeacherProjectionAudit(
                teacher_id=teacher_id,
                declared_status=declared,
                effective_status=normalized_status,
                source_alias_normalized=source_alias_normalized,
                unreferenced_evidence_pruned_count=pruned_count,
                signature=signature,
            ),
            snapshot,
        )

    if declared != "valid":
        # Fail closed on stale validation metadata rather than silently upgrading
        # arbitrary invalid output.  The sole permitted upgrade is the explicit
        # catalog/system alias path above.
        return (
            TeacherProjectionAudit(
                teacher_id=teacher_id,
                declared_status=declared,
                effective_status="artifact_integrity_error",
                failure_code="declared_invalid_but_strict_revalidation_passed",
            ),
            None,
        )
    signature = canonical_snapshot_signature(snapshot)
    return (
        TeacherProjectionAudit(
            teacher_id=teacher_id,
            declared_status=declared,
            effective_status="valid",
            signature=signature,
        ),
        snapshot,
    )


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_LEXICAL_STOPWORDS = frozenset({"a", "an", "and", "are", "for", "in", "is", "of", "or", "the", "to", "with"})


def _tokens(value: JsonValue) -> set[str]:
    text = canonical_json(_normalized_json(value))
    return {token for token in _TOKEN_PATTERN.findall(text) if token not in _LEXICAL_STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


def _constraint_similarity(left: Any, right: Any) -> tuple[float, float, float]:
    left_name = _tokens(left.name)
    right_name = _tokens(right.name)
    left_value = _tokens(left.value)
    right_value = _tokens(right.value)
    name_similarity = _jaccard(left_name, right_name)
    value_similarity = _jaccard(left_value, right_value)
    combined_similarity = _jaccard(left_name | left_value, right_name | right_value)
    # Open slot names vary (category/product_type, use/occasion).  Exact value
    # overlap is therefore a stronger lexical anchor, while combined overlap
    # preserves partial multi-token matches.  This is not semantic rewriting.
    score = max(combined_similarity, 0.35 * name_similarity + 0.65 * value_similarity)
    return score, name_similarity, value_similarity


def compare_constraint_facets(
    left: TeacherCurrentStateSnapshot | PublicDSUSState,
    right: TeacherCurrentStateSnapshot | PublicDSUSState,
    *,
    match_threshold: float = 0.38,
) -> ConstraintFacetAgreement:
    """Greedy one-to-one lexical matching plus explicit polarity conflicts."""

    if not 0.0 <= match_threshold <= 1.0:
        raise ValueError("constraint match threshold must lie in [0,1]")
    candidates: list[tuple[float, int, int, float, float]] = []
    contradictions: set[str] = set()
    for left_index, left_item in enumerate(left.constraints):
        for right_index, right_item in enumerate(right.constraints):
            score, name_similarity, value_similarity = _constraint_similarity(
                left_item,
                right_item,
            )
            candidates.append((score, left_index, right_index, name_similarity, value_similarity))
            polarity_conflict = (left_item.polarity.value == "avoid") != (right_item.polarity.value == "avoid")
            # Same slot with different values is not contradictory by itself
            # (for example prefer wool and avoid polyester). Require direct
            # lexical value overlap before calling opposite polarity material.
            if polarity_conflict and value_similarity >= match_threshold:
                contradictions.add(
                    "avoid_polarity_conflict:"
                    + canonical_json(
                        sorted(
                            [
                                _normalized_text(left_item.name),
                                _normalized_text(right_item.name),
                            ]
                        )
                    )
                )
    matched_left: set[int] = set()
    matched_right: set[int] = set()
    matched = 0
    for score, left_index, right_index, _, _ in sorted(
        candidates,
        key=lambda item: (-item[0], item[1], item[2]),
    ):
        if score < match_threshold:
            break
        if left_index in matched_left or right_index in matched_right:
            continue
        matched_left.add(left_index)
        matched_right.add(right_index)
        matched += 1
    left_count = len(left.constraints)
    right_count = len(right.constraints)
    precision = 1.0 if right_count == 0 and left_count == 0 else (0.0 if right_count == 0 else matched / right_count)
    recall = 1.0 if left_count == 0 and right_count == 0 else (0.0 if left_count == 0 else matched / left_count)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return ConstraintFacetAgreement(
        matched_count=matched,
        left_count=left_count,
        right_count=right_count,
        precision=precision,
        recall=recall,
        f1=f1,
        material_contradictions=sorted(contradictions),
    )


def compare_snapshot_facets(
    left_teacher: str,
    left: TeacherCurrentStateSnapshot,
    right_teacher: str,
    right: TeacherCurrentStateSnapshot,
    *,
    constraint_overlap_threshold: float = 0.45,
) -> TeacherPairFacetAgreement:
    constraints = compare_constraint_facets(left, right)
    left_signature = canonical_snapshot_signature(left)
    right_signature = canonical_snapshot_signature(right)
    key_control_exact = left_signature.key_control == right_signature.key_control
    left_action = decision_action_signature(left)
    right_action = decision_action_signature(right)
    decision_action_exact = left_action == right_action
    left_patience_alert = left.control.patience_risk.value in {"high", "exhausted"}
    right_patience_alert = right.control.patience_risk.value in {"high", "exhausted"}
    patience_alert_exact = left_patience_alert == right_patience_alert
    termination_exact = left_signature.termination == right_signature.termination
    rejected_exact = left_signature.rejected_item_ids == right_signature.rejected_item_ids
    material_conflicts: list[str] = []
    if not decision_action_exact:
        material_conflicts.append("next_action_conflict")
    if not patience_alert_exact:
        material_conflicts.append("patience_alert_conflict")
    if not termination_exact:
        material_conflicts.append("termination_conflict")
    if not rejected_exact:
        material_conflicts.append("rejected_item_ids_conflict")
    if constraints.material_contradictions:
        material_conflicts.append("constraint_polarity_conflict")
    soft_disagreements: list[str] = []
    if not key_control_exact and decision_action_exact:
        soft_disagreements.append("non_action_control_boundary_difference")
    if constraints.f1 < constraint_overlap_threshold:
        soft_disagreements.append("constraint_lexical_granularity_difference")
    if left_signature.full_sha256 != right_signature.full_sha256:
        soft_disagreements.append("full_signature_difference")
    return TeacherPairFacetAgreement(
        teacher_ids=sorted([left_teacher, right_teacher]),
        key_control_exact=key_control_exact,
        decision_action_left=left_action,
        decision_action_right=right_action,
        decision_action_exact=decision_action_exact,
        patience_alert_exact=patience_alert_exact,
        termination_exact=termination_exact,
        rejected_item_ids_exact=rejected_exact,
        constraints=constraints,
        material_conflicts=material_conflicts,
        soft_disagreements=soft_disagreements,
        compatible=not material_conflicts,
    )


def decision_action_signature(
    snapshot: TeacherCurrentStateSnapshot | PublicDSUSState,
) -> Literal["ask", "recommend", "repair", "explain", "stop"]:
    """Project controls to the next action class used by the TAIRAS adapter."""

    if snapshot.termination_reason is not None or snapshot.control.interaction_phase.value == "terminal":
        return "stop"
    if snapshot.control.feedback_type.value == "explanation_request" or snapshot.control.explanation_status.value in {
        "requested",
        "pending",
    }:
        return "explain"
    if snapshot.control.feedback_type.value in {"correction", "rejection"}:
        return "repair"
    if snapshot.control.information_status.value in {"missing", "ambiguous", "conflicting"}:
        return "ask"
    return "recommend"


def build_sample_consensus(
    *,
    public: PublicRLPromptRecord,
    trajectory_id: str,
    source_turn_index: int,
    raw_by_teacher: Mapping[str, TeacherProjectionRawRecord],
    validation_by_teacher: Mapping[str, TeacherProjectionValidationRecord],
    preferred_teacher: str | None = None,
    constraint_overlap_threshold: float = 0.45,
) -> TeacherConsensusRecord:
    """Select one valid snapshot under auditable cross-teacher facet support."""

    teacher_ids = sorted(raw_by_teacher)
    if not 2 <= len(teacher_ids) <= 4 or set(teacher_ids) != set(validation_by_teacher):
        raise ValueError("sample consensus requires the same two to four teachers")
    if preferred_teacher is not None and preferred_teacher not in teacher_ids:
        raise ValueError("preferred teacher is outside the three-teacher panel")

    audits: list[TeacherProjectionAudit] = []
    snapshots: dict[str, TeacherCurrentStateSnapshot] = {}
    signature_teachers: dict[str, list[str]] = defaultdict(list)
    reasons: list[ConsensusReason] = []
    for teacher_id in teacher_ids:
        raw = raw_by_teacher[teacher_id]
        validation = validation_by_teacher[teacher_id]
        if raw.sample_id != public.sample_id or validation.sample_id != public.sample_id:
            raise ValueError("teacher record sample ID does not match public record")
        if (
            raw.public_prompt_sha256 != public.public_prompt_sha256
            or validation.public_prompt_sha256 != public.public_prompt_sha256
        ):
            raise ValueError("teacher/public prompt hash mismatch")
        audit, snapshot = _effective_projection(
            teacher_id=teacher_id,
            public=public,
            raw=raw,
            validation=validation,
        )
        audits.append(audit)
        if snapshot is not None:
            snapshots[teacher_id] = snapshot
            assert audit.signature is not None
            signature_teachers[audit.signature.full_sha256].append(teacher_id)
        else:
            reasons.append(
                ConsensusReason(
                    code=f"teacher_{audit.effective_status}",
                    teacher_ids=[teacher_id],
                    detail={"failure_code": audit.failure_code},
                )
            )
        if audit.source_alias_normalized:
            reasons.append(
                ConsensusReason(
                    code="catalog_system_source_alias_normalized",
                    teacher_ids=[teacher_id],
                )
            )
        if audit.unreferenced_evidence_pruned_count:
            reasons.append(
                ConsensusReason(
                    code="unreferenced_evidence_pruned",
                    teacher_ids=[teacher_id],
                    detail={"count": audit.unreferenced_evidence_pruned_count},
                )
            )

    groups = sorted(
        (
            SignatureGroup(full_sha256=signature, teacher_ids=sorted(supporters))
            for signature, supporters in signature_teachers.items()
        ),
        key=lambda group: (-len(group.teacher_ids), group.full_sha256),
    )
    pair_agreements = [
        compare_snapshot_facets(
            left_teacher,
            snapshots[left_teacher],
            right_teacher,
            snapshots[right_teacher],
            constraint_overlap_threshold=constraint_overlap_threshold,
        )
        for left_teacher, right_teacher in itertools.combinations(sorted(snapshots), 2)
    ]
    agreement_by_pair = {tuple(agreement.teacher_ids): agreement for agreement in pair_agreements}

    def pair(left: str, right: str) -> TeacherPairFacetAgreement:
        return agreement_by_pair[tuple(sorted((left, right)))]

    def unique_plurality_support(
        candidate: str,
        values: Mapping[str, str],
    ) -> list[str]:
        candidate_value = values[candidate]
        counts = Counter(values.values())
        support = sorted(teacher for teacher, value in values.items() if value == candidate_value)
        other_counts = [count for value, count in counts.items() if value != candidate_value]
        if len(support) < 2 or (other_counts and len(support) <= max(other_counts)):
            return []
        return support

    action_values = {teacher: decision_action_signature(snapshot) for teacher, snapshot in snapshots.items()}
    patience_values = {
        teacher: ("alert" if snapshot.control.patience_risk.value in {"high", "exhausted"} else "normal")
        for teacher, snapshot in snapshots.items()
    }
    term_values = {
        teacher: canonical_snapshot_signature(snapshot).termination for teacher, snapshot in snapshots.items()
    }
    rejected_values = {
        teacher: canonical_json(canonical_snapshot_signature(snapshot).rejected_item_ids)
        for teacher, snapshot in snapshots.items()
    }
    candidate_support: dict[str, set[str]] = {}
    candidate_details: dict[str, dict[str, JsonValue]] = {}
    for candidate in sorted(snapshots):
        supporters: set[str] = {candidate}
        failed_facets: list[str] = []
        facet_support: dict[str, list[str]] = {}
        other_teachers = [teacher for teacher in snapshots if teacher != candidate]
        pair_material = {other: pair(candidate, other).material_conflicts for other in other_teachers}
        if not other_teachers:
            if candidate != preferred_teacher:
                failed_facets.append("unverified_non_primary_snapshot")
        elif all(not conflicts for conflicts in pair_material.values()):
            supporters.update(other_teachers)
            facet_support["all_material_facets"] = sorted({candidate, *other_teachers})
        elif len(snapshots) >= 3:
            for facet, values in (
                ("next_action", action_values),
                ("patience_alert", patience_values),
                ("termination", term_values),
                ("rejected_item_ids", rejected_values),
            ):
                support = unique_plurality_support(candidate, values)
                facet_support[facet] = support
                if support:
                    supporters.update(support)
                else:
                    failed_facets.append(facet)
            constraint_support = sorted(
                other for other in other_teachers if not pair(candidate, other).constraints.material_contradictions
            )
            facet_support["constraint_polarity"] = [candidate, *constraint_support]
            if constraint_support:
                supporters.update(constraint_support)
            else:
                failed_facets.append("constraint_polarity")
        else:
            failed_facets.extend(sorted({conflict for conflicts in pair_material.values() for conflict in conflicts}))
        candidate_details[candidate] = {
            "failed_facets": failed_facets,
            "facet_support": facet_support,
            "pair_material_conflicts": pair_material,
        }
        if not failed_facets:
            candidate_support[candidate] = supporters

    eligible = sorted(candidate_support)
    selected: str | None = None
    if preferred_teacher is not None and preferred_teacher in eligible:
        selected = preferred_teacher
    if selected is None and eligible:
        ranked = sorted(
            eligible,
            key=lambda teacher: (-len(candidate_support[teacher]), teacher),
        )
        selected = ranked[0]

    if selected is not None:
        supporters = sorted(candidate_support[selected])
        had_material_conflict = any(
            agreement.material_conflicts for agreement in pair_agreements if selected in agreement.teacher_ids
        )
        if len(snapshots) == 1:
            acceptance_tier = "primary_valid_no_material_conflict"
        elif had_material_conflict:
            acceptance_tier = "third_teacher_arbitrated"
        else:
            acceptance_tier = "cross_teacher_material_agreement"
        selected_signature = canonical_snapshot_signature(snapshots[selected])
        exact_supporters = next(
            (group.teacher_ids for group in groups if group.full_sha256 == selected_signature.full_sha256),
            [selected],
        )
        reasons.append(
            ConsensusReason(
                code="decision_materiality_supported_primary_snapshot",
                teacher_ids=supporters,
                detail={
                    "constraint_overlap_threshold": constraint_overlap_threshold,
                    "exact_signature_support": len(exact_supporters),
                    "facet_support": candidate_details[selected],
                },
            )
        )
        return TeacherConsensusRecord(
            sample_id=public.sample_id,
            public_prompt_sha256=public.public_prompt_sha256,
            trajectory_id=trajectory_id,
            source_turn_index=source_turn_index,
            status="accepted",
            teacher_audits=audits,
            signature_groups=groups,
            pair_agreements=pair_agreements,
            selected_teacher_id=selected,
            supporting_teacher_ids=supporters,
            acceptance_tier=acceptance_tier,
            consensus_snapshot=snapshots[selected],
            reasons=reasons,
        )

    # A missing/invalid primary may only be replaced by an actual agreement
    # between alternative teachers.  In particular, do not silently accept
    # the sole surviving secondary/third teacher: that is exactly the case
    # for which Sol (or later manual review) is required.
    primary_invalid_without_alternative_consensus = (
        preferred_teacher is not None and preferred_teacher not in snapshots
    )
    if len(snapshots) >= 2 or primary_invalid_without_alternative_consensus:
        reason_code = (
            "primary_invalid_without_alternative_consensus"
            if primary_invalid_without_alternative_consensus and len(snapshots) < 2
            else "material_facet_disagreement"
        )
        reasons.append(
            ConsensusReason(
                code=reason_code,
                teacher_ids=sorted(snapshots),
                detail={
                    "distinct_signature_count": len(signature_teachers),
                    "constraint_overlap_threshold": constraint_overlap_threshold,
                    "candidate_failures": candidate_details,
                    "valid_teacher_count": len(snapshots),
                },
            )
        )
        return TeacherConsensusRecord(
            sample_id=public.sample_id,
            public_prompt_sha256=public.public_prompt_sha256,
            trajectory_id=trajectory_id,
            source_turn_index=source_turn_index,
            status="disagreement",
            teacher_audits=audits,
            signature_groups=groups,
            pair_agreements=pair_agreements,
            reasons=reasons,
            sol_arbitration_required=True,
        )

    reasons.append(
        ConsensusReason(
            code="insufficient_valid_teacher_quorum",
            teacher_ids=sorted(snapshots),
            detail={"valid_teacher_count": len(snapshots)},
        )
    )
    return TeacherConsensusRecord(
        sample_id=public.sample_id,
        public_prompt_sha256=public.public_prompt_sha256,
        trajectory_id=trajectory_id,
        source_turn_index=source_turn_index,
        status="rejected",
        teacher_audits=audits,
        signature_groups=groups,
        pair_agreements=pair_agreements,
        reasons=reasons,
    )


def select_third_teacher_samples(
    *,
    public_by_id: Mapping[str, PublicRLPromptRecord],
    raw_by_teacher: Mapping[str, Mapping[str, TeacherProjectionRawRecord]],
    validation_by_teacher: Mapping[str, Mapping[str, TeacherProjectionValidationRecord]],
    preferred_teacher: str | None = None,
    hard_slice_ids: set[str] | None = None,
    constraint_overlap_threshold: float = 0.45,
    minimum_count: int = 0,
    maximum_count: int | None = None,
) -> tuple[list[ThirdTeacherSelectionRecord], list[str]]:
    """Route uncertain two-teacher cases to a selectively called third teacher.

    A primary-teacher failure or a disagreement that changes the downstream
    state-machine action is mandatory.  Non-action control boundaries,
    lexical constraint granularity, and a failed secondary audit remain in the
    soft ledger but do not by themselves spend a third-teacher call.
    """

    teachers = sorted(raw_by_teacher)
    if len(teachers) != 2 or set(teachers) != set(validation_by_teacher):
        raise ValueError("third-teacher selection requires exactly two teachers")
    primary_teacher = preferred_teacher or teachers[0]
    if primary_teacher not in teachers:
        raise ValueError("preferred teacher is outside the two-teacher panel")
    secondary_teacher = next(teacher for teacher in teachers if teacher != primary_teacher)
    if minimum_count < 0:
        raise ValueError("minimum_count must be nonnegative")
    if maximum_count is not None and maximum_count < 1:
        raise ValueError("maximum_count must be positive")
    if maximum_count is not None and minimum_count > maximum_count:
        raise ValueError("minimum_count cannot exceed maximum_count")
    hard = set() if hard_slice_ids is None else set(hard_slice_ids)
    unknown_hard = hard.difference(public_by_id)
    if unknown_hard:
        raise ValueError(f"hard-slice IDs are outside public panel: {sorted(unknown_hard)[:3]}")

    provisional: list[ThirdTeacherSelectionRecord] = []
    for sample_id, public in public_by_id.items():
        audits: list[TeacherProjectionAudit] = []
        snapshots: dict[str, TeacherCurrentStateSnapshot] = {}
        reasons: list[str] = []
        soft_reasons: list[str] = []
        for teacher_id in teachers:
            raw = raw_by_teacher[teacher_id][sample_id]
            validation = validation_by_teacher[teacher_id][sample_id]
            audit, snapshot = _effective_projection(
                teacher_id=teacher_id,
                public=public,
                raw=raw,
                validation=validation,
            )
            audits.append(audit)
            if snapshot is not None:
                snapshots[teacher_id] = snapshot
        if primary_teacher not in snapshots:
            primary_audit = next(audit for audit in audits if audit.teacher_id == primary_teacher)
            reasons.append(f"primary_invalid:{primary_audit.effective_status}")
        if secondary_teacher not in snapshots:
            secondary_audit = next(audit for audit in audits if audit.teacher_id == secondary_teacher)
            soft_reasons.append(f"secondary_invalid:{secondary_audit.effective_status}")
        pair: TeacherPairFacetAgreement | None = None
        if len(snapshots) == 2:
            pair = compare_snapshot_facets(
                teachers[0],
                snapshots[teachers[0]],
                teachers[1],
                snapshots[teachers[1]],
                constraint_overlap_threshold=constraint_overlap_threshold,
            )
            reasons.extend(pair.material_conflicts)
            soft_reasons.extend(pair.soft_disagreements)
        if sample_id in hard:
            reasons.append("predefined_hard_slice")
        provisional.append(
            ThirdTeacherSelectionRecord(
                sample_id=sample_id,
                public_prompt_sha256=public.public_prompt_sha256,
                selected=bool(reasons),
                selection_reasons=sorted(set(reasons)),
                soft_disagreement_reasons=sorted(set(soft_reasons)),
                teacher_audits=audits,
                pair_agreement=pair,
            )
        )

    mandatory = [record for record in provisional if record.selected]
    if maximum_count is not None and len(mandatory) > maximum_count:
        raise ValueError(f"mandatory third-teacher cases ({len(mandatory)}) exceed maximum_count ({maximum_count})")
    target = max(len(mandatory), minimum_count)
    if maximum_count is not None:
        target = min(target, maximum_count)
    if target > len(provisional):
        target = len(provisional)
    selected_ids = {record.sample_id for record in mandatory}
    if len(selected_ids) < target:
        candidates = sorted(
            (record for record in provisional if not record.selected),
            key=lambda record: content_sha256(
                {
                    "sample_id": record.sample_id,
                    "purpose": "third_teacher_minimum_coverage_v1",
                }
            ),
        )
        for record in candidates[: target - len(selected_ids)]:
            record.selected = True
            record.selection_reasons = ["deterministic_minimum_coverage"]
            selected_ids.add(record.sample_id)
    ordered_ids = [record.sample_id for record in provisional if record.sample_id in selected_ids]
    return provisional, ordered_ids


_REVISION_SIGNAL = re.compile(
    r"\b(?:actually|instead|rather|no longer|not anymore|changed? my mind|"
    r"scratch that|never mind|nevermind|forget (?:that|it)|drop|remove|"
    r"do not (?:want|need|care)|don['’]t (?:want|need|care)|without|"
    r"but not|except|reconsider|works? after all|okay with|fine with)\b",
    flags=re.IGNORECASE,
)


def _prompt_history(public: PublicRLPromptRecord) -> list[tuple[str, str]]:
    prompt = public.public_prompt
    observations = prompt.public_history
    if observations is None:
        observations = prompt.latest_observations
    assert observations is not None
    return [(item.role, item.text) for item in observations]


def _revision_evidence(
    previous: PublicRLPromptRecord,
    current: PublicRLPromptRecord,
) -> tuple[bool, list[str], bool]:
    previous_history = _prompt_history(previous)
    current_history = _prompt_history(current)
    prefix_ok = current_history[: len(previous_history)] == previous_history
    if not prefix_ok:
        return False, [], False
    new_user_text = [text for role, text in current_history[len(previous_history) :] if role == "user"]
    signals: list[str] = []
    for text in new_user_text:
        match = _REVISION_SIGNAL.search(text)
        if match is not None:
            signals.append(match.group(0))
    return bool(signals), signals, True


def _exact_avoid_persistence_anchors(
    snapshot: TeacherCurrentStateSnapshot,
) -> set[str]:
    """Return only deterministic negative-constraint persistence anchors.

    Open slot taxonomies are not stable across independently projected
    snapshots: a teacher may legitimately rewrite ``target_gender=women`` as
    part of ``product_type=women's boots``, or change status/importance while
    preserving the preference.  Treating those surface changes as deletions
    creates false hard failures.  An exact normalized name/value/AVOID tuple,
    however, is a safe narrow invariant.  IDs, evidence, status and importance
    are deliberately excluded.
    """

    return {
        canonical_json(
            {
                "name": _normalized_text(item.name),
                "value": _normalized_json(item.value),
                "polarity": item.polarity.value,
            }
        )
        for item in snapshot.constraints
        if item.polarity.value == "avoid"
    }


def audit_trajectory_persistence(
    records: Sequence[TeacherConsensusRecord],
    *,
    public_by_id: Mapping[str, PublicRLPromptRecord],
) -> tuple[list[TeacherConsensusRecord], int, int]:
    """Audit narrow hard persistence and ledger open-semantic drift.

    Only rejected public item IDs and exact normalized AVOID
    name/value/polarity anchors are hard monotonic invariants.  Other
    constraint surface changes are recorded as soft evidence because open
    names, values, status and importance can be paraphrased or consolidated
    without changing the underlying user intent.
    """

    by_trajectory: dict[str, list[TeacherConsensusRecord]] = defaultdict(list)
    for record in records:
        by_trajectory[record.trajectory_id].append(record.model_copy(deep=True))
    output: list[TeacherConsensusRecord] = []
    pair_count = 0
    failure_count = 0
    for trajectory_id in sorted(by_trajectory):
        trajectory = sorted(
            by_trajectory[trajectory_id],
            key=lambda item: (item.source_turn_index, item.sample_id),
        )
        indices = [item.source_turn_index for item in trajectory]
        if len(indices) != len(set(indices)):
            raise ValueError(f"trajectory has duplicate source turns: {trajectory_id}")
        for index, current in enumerate(trajectory):
            if index == 0:
                output.append(current)
                continue
            previous = trajectory[index - 1]
            if current.source_turn_index != previous.source_turn_index + 1:
                output.append(current)
                continue
            if previous.status != "accepted" or current.status != "accepted":
                output.append(current)
                continue
            pair_count += 1
            assert previous.consensus_snapshot is not None
            assert current.consensus_snapshot is not None
            previous_signature = canonical_snapshot_signature(previous.consensus_snapshot)
            current_signature = canonical_snapshot_signature(current.consensus_snapshot)
            removed_constraint_surfaces = sorted(
                set(previous_signature.constraints).difference(current_signature.constraints)
            )
            previous_avoid_anchors = _exact_avoid_persistence_anchors(previous.consensus_snapshot)
            current_avoid_anchors = _exact_avoid_persistence_anchors(current.consensus_snapshot)
            removed_avoid_anchors = sorted(previous_avoid_anchors.difference(current_avoid_anchors))
            previous_rejected = {item.item_id for item in previous.consensus_snapshot.rejected_items}
            current_rejected = {item.item_id for item in current.consensus_snapshot.rejected_items}
            removed_rejected = sorted(previous_rejected.difference(current_rejected))
            revision, revision_signals, prefix_ok = _revision_evidence(
                public_by_id[previous.sample_id],
                public_by_id[current.sample_id],
            )
            current.persistence_checked = True
            persistence_reasons: list[ConsensusReason] = []
            if removed_constraint_surfaces:
                current.reasons.append(
                    ConsensusReason(
                        code="soft_open_constraint_persistence_difference",
                        detail={
                            "removed_surface_count": len(removed_constraint_surfaces),
                            "exact_avoid_anchor_removed_count": len(removed_avoid_anchors),
                            "policy": "open_name_value_status_importance_changes_are_soft",
                        },
                    )
                )
            if not prefix_ok:
                persistence_reasons.append(ConsensusReason(code="non_monotonic_public_prefix"))
            elif (removed_avoid_anchors or removed_rejected) and not revision:
                if removed_avoid_anchors:
                    persistence_reasons.append(
                        ConsensusReason(
                            code="exact_avoid_constraint_deleted_without_public_revision_evidence",
                            detail={"removed_anchor_count": len(removed_avoid_anchors)},
                        )
                    )
                if removed_rejected:
                    persistence_reasons.append(
                        ConsensusReason(
                            code="rejected_item_deleted_without_public_revision_evidence",
                            detail={"removed_item_ids": removed_rejected},
                        )
                    )
            elif removed_avoid_anchors or removed_rejected:
                current.reasons.append(
                    ConsensusReason(
                        code="public_revision_signal_observed",
                        detail={"lexical_signals": revision_signals},
                    )
                )
            if persistence_reasons:
                failure_count += 1
                current.status = "disagreement"
                current.sol_arbitration_required = True
                current.selected_teacher_id = None
                current.supporting_teacher_ids = []
                current.acceptance_tier = None
                current.consensus_snapshot = None
                current.reasons.extend(persistence_reasons)
            output.append(current)
    return (
        sorted(output, key=lambda item: (item.trajectory_id, item.source_turn_index, item.sample_id)),
        pair_count,
        failure_count,
    )


def read_public_records(path: Path, sample_ids: Sequence[str]) -> dict[str, PublicRLPromptRecord]:
    requested = list(sample_ids)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("selected sample IDs must be unique and nonempty")
    requested_set = set(requested)
    output: dict[str, PublicRLPromptRecord] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = PublicRLPromptRecord.model_validate(json.loads(line))
        except Exception as exc:
            raise ValueError(f"invalid public record at {path}:{line_number}") from exc
        if record.sample_id in requested_set:
            if record.sample_id in output:
                raise ValueError(f"duplicate public sample: {record.sample_id}")
            output[record.sample_id] = record
    missing = sorted(requested_set.difference(output))
    if missing:
        raise ValueError(f"selected samples missing from public prompts: {missing[:3]}")
    return {sample_id: output[sample_id] for sample_id in requested}


def read_teacher_run(
    run_dir: Path,
    *,
    public_by_id: Mapping[str, PublicRLPromptRecord],
    require_exact: bool = True,
) -> tuple[dict[str, TeacherProjectionRawRecord], dict[str, TeacherProjectionValidationRecord]]:
    def read(path: Path, model: type[StrictModel]) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = model.model_validate(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid teacher record at {path}:{line_number}") from exc
            if row.sample_id in rows:
                raise ValueError(f"duplicate teacher sample at {path}: {row.sample_id}")
            rows[row.sample_id] = row
        return rows

    raw = read(run_dir / "projection.raw.jsonl", TeacherProjectionRawRecord)
    validation = read(
        run_dir / "projection.validation.jsonl",
        TeacherProjectionValidationRecord,
    )
    expected = set(public_by_id)
    if set(raw) != set(validation):
        raise ValueError(f"teacher raw/validation sample sets differ: {run_dir}")
    if require_exact and set(raw) != expected:
        raise ValueError(f"teacher run does not exactly cover selected samples: {run_dir}")
    if not require_exact and not set(raw).issubset(expected):
        raise ValueError(f"teacher run contains samples outside selected panel: {run_dir}")
    for sample_id in raw:
        expected_hash = public_by_id[sample_id].public_prompt_sha256
        if (
            raw[sample_id].public_prompt_sha256 != expected_hash
            or validation[sample_id].public_prompt_sha256 != expected_hash
        ):
            raise ValueError(f"teacher/public prompt hash mismatch: {run_dir}/{sample_id}")
    return raw, validation


def read_trajectory_index(
    path: Path | None,
    *,
    sample_ids: Sequence[str],
) -> dict[str, tuple[str, int]]:
    """Read public/source lineage without interpreting any gold label."""

    if path is None:
        return {sample_id: (sample_id, 0) for sample_id in sample_ids}
    requested = set(sample_ids)
    output: dict[str, tuple[str, int]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise ValueError(f"invalid trajectory index at {path}:{line_number}") from exc
        if not isinstance(row, dict) or row.get("sample_id") not in requested:
            continue
        sample_id = row["sample_id"]
        trajectory_id = row.get("trajectory_id", row.get("source_conversation_id"))
        turn_index = row.get("source_turn_index", row.get("turn_index"))
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError(f"trajectory index lacks trajectory ID at {path}:{line_number}")
        if not isinstance(turn_index, int) or turn_index < 0:
            raise ValueError(f"trajectory index lacks nonnegative turn at {path}:{line_number}")
        if sample_id in output:
            raise ValueError(f"duplicate trajectory index sample: {sample_id}")
        output[sample_id] = (trajectory_id, turn_index)
    missing = sorted(requested.difference(output))
    if missing:
        raise ValueError(f"trajectory index misses selected samples: {missing[:3]}")
    if len(set(output.values())) != len(output):
        raise ValueError("trajectory index repeats a trajectory/source-turn pair")
    return output


__all__ = [
    "CanonicalSnapshotSignature",
    "ConstraintFacetAgreement",
    "ConsensusReason",
    "ThirdTeacherSelectionRecord",
    "TeacherConsensusRecord",
    "TeacherConsensusSummary",
    "TeacherPairFacetAgreement",
    "TeacherProjectionAudit",
    "audit_trajectory_persistence",
    "build_sample_consensus",
    "canonical_snapshot_signature",
    "compare_constraint_facets",
    "compare_snapshot_facets",
    "normalize_catalog_system_provenance",
    "prune_unreferenced_evidence",
    "read_public_records",
    "read_teacher_run",
    "read_trajectory_index",
    "select_third_teacher_samples",
]
