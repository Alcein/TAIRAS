"""Offline full-state projection protocol for strong DSUS teachers.

The production compiler predicts a typed :class:`~taira_mt.state_compiler.StatePatch`.
That contract is intentionally left unchanged.  This module defines a simpler
*offline labeling* observation point: a strong teacher reads public dialogue
history and emits one evidence-grounded snapshot of the final current state.

Generation and validation are separated deliberately.  The raw runner calls an
OpenAI-compatible transport exactly once and preserves its text even when the
text is not valid JSON.  A later offline step may parse the text with
``parse_teacher_projection`` and, if desired, derive training deltas without
changing the student runtime.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Protocol

from pydantic import Field, JsonValue, field_validator, model_validator

from taira_mt.models import StrictModel
from taira_mt.provenance import content_sha256
from taira_mt.providers.openai_compatible import (
    JsonCompletion,
    JsonCompletionTransport,
    OpenAICompatibleTransportProfile,
    build_openai_compatible_json_transport,
)
from taira_mt.state_compiler import (
    ConstraintStatus,
    ControlField,
    ControlState,
    EvidenceReference,
    PublicDSUSState,
    RejectedItem,
    SemanticConstraint,
    TerminationReason,
    UnresolvedAxis,
)
from taira_mt.state_compiler.runtime import (
    APICompilerBackendProfile,
    StateCompilerRuntimeProfile,
    validate_public_evidence,
)
from taira_mt.state_training.dataset_assembly import PublicRLPromptRecord
from taira_mt.state_training.training_views import DSUSPublicPrompt, PublicTextObservation

TEACHER_PROJECTION_SCHEMA_VERSION = "dsus_teacher_current_state_snapshot_v1"
TEACHER_PROJECTION_RAW_VERSION = "dsus_teacher_projection_raw_v1"
TEACHER_PROJECTION_TELEMETRY_VERSION = "dsus_teacher_projection_telemetry_v1"
TEACHER_PROJECTION_VALIDATED_VERSION = "dsus_teacher_projection_validated_v1"
TEACHER_PROJECTION_VALIDATION_VERSION = "dsus_teacher_projection_validation_v1"


TeacherProjectionPromptVersion = Literal["v1", "semantic_guard_v2", "external_noop_v3"]


TEACHER_PROJECTION_SYSTEM_PROMPT_V1 = """You are an offline DSUS current-state annotation teacher.
Read only the supplied public dialogue and return exactly one JSON object. The dialogue may be a
long, natural, real-world shopping conversation. Project ONLY the final CURRENT user state after
the last visible observation. Do not output an event log, a chronology, prior states, future states,
operations, patches, deltas, actions, Thought Patterns, rewards, or private benchmark fields.

The JSON object must contain exactly these top-level keys:
schema_version, evidence, control, control_evidence, constraints, unresolved_axes, rejected_items,
termination_reason, termination_evidence_ids. additionalProperties are forbidden.

Use schema_version "dsus_teacher_current_state_snapshot_v1".
Each evidence object has exactly:
{"evidence_id":"ev_1","turn_id":0,"source":"user|system|catalog",
 "quote":"an exact nonempty substring of that public observation"}.
Input observations carry explicit public turn_id values; copy them exactly. A tool observation maps
to source "catalog". Text headed `Displayed recommendations` that is embedded inside an assistant
or system observation still has source "system"; use "catalog" only when the input observation role
itself is `catalog` or `tool`. Prefer short exact quotes. Evidence IDs must be unique, every stored fact must
cite evidence_ids, and every evidence object must be referenced by at least one stored fact.

control must contain exactly these seven fields and one allowed value per field:
interaction_phase = opening|elicitation|grounding|recommendation|feedback|repair|terminal;
information_status = missing|sufficient|ambiguous|conflicting;
intent_stability = stable|strengthened|relaxed|revised;
feedback_type = none|answer|acceptance|rejection|correction|explanation_request;
explanation_status = none|requested|pending|resolved;
commitment_status = unknown|exploring|comparing|ready|accepted|abandoned;
patience_risk = unknown|low|medium|high|exhausted.
control_evidence maps control-field names to evidence-ID arrays. Cite every non-default control value;
the defaults are opening/missing/stable/none/none/unknown/unknown. Auxiliary commitment and patience
values other than unknown require explicit user evidence, never speculation.

Each constraint has exactly constraint_id, name, value, polarity, importance, status, evidence_ids.
polarity is must|prefer|avoid; importance is hard|soft; status is active|uncertain. Keep only user
requirements that are current at the end. A later correction/revision overrides older statements;
omit superseded or removed constraints instead of narrating their history. System or catalog claims
are context, not user preferences, unless the user explicitly adopts them.

Each unresolved axis has exactly axis_id, axis_type, description, options, evidence_ids. Include only
questions that remain unresolved at the end; omit already answered axes. Each rejected item has
exactly item_id, reason, evidence_ids, and item_id must come from a publicly displayed recommendation.
Do not invent an item ID. Preserve a still-relevant earlier rejection, but do not replay feedback
events. feedback_type describes the final current feedback condition, not a list of past acts.

termination_reason is null unless the user explicitly accepts, abandons, ends the task, or no public
solution remains; otherwise use accepted|abandoned|no_solution|user_ended. A non-null termination
reason requires interaction_phase terminal and nonempty termination_evidence_ids. If not terminated,
termination_evidence_ids must be []. Return compact JSON only, with no markdown or explanation.
"""

# Backwards-compatible public name.  Its bytes must remain identical to the original V1 prompt so
# existing generation commands and their model behavior are reproducible by default.
TEACHER_PROJECTION_SYSTEM_PROMPT = TEACHER_PROJECTION_SYSTEM_PROMPT_V1

TEACHER_PROJECTION_SYSTEM_PROMPT_SEMANTIC_GUARD_V2 = (
    TEACHER_PROJECTION_SYSTEM_PROMPT_V1
    + """

SEMANTIC-GUARD V2 — apply all rules below in addition to the base protocol. These rules constrain
the final current-state semantics; the JSON schema_version remains
"dsus_teacher_current_state_snapshot_v1".

1. Accepted-item binding. If termination_reason is "accepted", constraints MUST contain exactly
one accepted selection constraint with all of these exact fields:
name="selected_item", value="<exact item ID from public history>", polarity="must",
importance="hard", status="active". The value must be the exact, unique item ID of the accepted
item as it appeared in a publicly displayed recommendation; never use a title, description, or
invented ID. Cite both the public system recommendation that exposes that ID and the user's
acceptance. Resolve references such as "the second one" only when they identify one displayed item
unambiguously. If the accepted item cannot be bound to exactly one public item ID, do not guess and
do not claim accepted termination; keep the selection unresolved/nonterminal instead.

2. Positive-valued avoid constraints. For every constraint whose polarity is "avoid", value MUST
be a positive attribute noun phrase naming the undesired property. It must not contain negation
tokens or phrases such as no, not, without, never, don't, doesn't, or cannot. Examples: "no rips"
becomes value="ripped detailing" with polarity="avoid"; "no fake leather" becomes
value="synthetic leather" with polarity="avoid"; "not too baggy" becomes
value="excessively baggy fit" with polarity="avoid".

3. Termination is an overlay, not a state reset. When the final observation terminates the task,
preserve every current constraint, unresolved axis, and rejected item that still existed immediately
before termination. Setting accepted, abandoned, no_solution, or user_ended must never by itself
clear current-state content. Only an explicit user withdrawal or replacement may remove it.

4. Withdrawal and replacement remove obsolete state. When the user explicitly withdraws a
requirement or replaces it with another one, omit the old constraint entirely. For example,
"it doesn't have to have a design" removes the design requirement; it does not create an avoid
constraint for design. A changed product category removes the old category constraint. Do not keep
withdrawn values as active, uncertain, or avoid unless the user explicitly states a new current
avoidance.
"""
)

TEACHER_PROJECTION_SYSTEM_PROMPT_EXTERNAL_NOOP_V3 = (
    TEACHER_PROJECTION_SYSTEM_PROMPT_SEMANTIC_GUARD_V2
    + """

EXTERNAL NO-OP V3 — use this rule only for source-disjoint external evaluation. A causal prefix
may contain only greetings, acknowledgements, or social text and may therefore carry no shopping
state. When, and only when, the final current state is exactly the default empty DSUS state, return
evidence=[], default control, control_evidence={}, constraints=[], unresolved_axes=[],
rejected_items=[], termination_reason=null, and termination_evidence_ids=[]. Never invent a
preference, unresolved shopping axis, feedback event, commitment, patience signal, or termination
merely to make evidence nonempty. If any non-default field or semantic state is present, the normal
evidence-grounding rules remain mandatory.
"""
)


def teacher_projection_system_prompt(
    prompt_version: TeacherProjectionPromptVersion,
) -> str:
    """Resolve one explicitly versioned system prompt without implicit upgrades."""

    if prompt_version == "v1":
        return TEACHER_PROJECTION_SYSTEM_PROMPT_V1
    if prompt_version == "semantic_guard_v2":
        return TEACHER_PROJECTION_SYSTEM_PROMPT_SEMANTIC_GUARD_V2
    if prompt_version == "external_noop_v3":
        return TEACHER_PROJECTION_SYSTEM_PROMPT_EXTERNAL_NOOP_V3
    raise ValueError(f"unsupported teacher projection prompt version: {prompt_version}")


def teacher_projection_system_prompt_sha256(
    prompt_version: TeacherProjectionPromptVersion,
) -> str:
    """Return the raw UTF-8 SHA-256 identity of the selected system prompt."""

    prompt = teacher_projection_system_prompt(prompt_version)
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


TEACHER_PROJECTION_SYSTEM_PROMPT_SHA256_V1 = teacher_projection_system_prompt_sha256("v1")


def _unique_evidence_ids(values: list[str], *, owner: str) -> list[str]:
    normalized = [value.strip() for value in values]
    if any(not value for value in normalized):
        raise ValueError(f"{owner} evidence IDs cannot be blank")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{owner} evidence IDs must be unique")
    return sorted(normalized)


class TeacherCurrentStateSnapshot(StrictModel):
    """Strong-teacher output: one final state, never a transition program."""

    schema_version: Literal[TEACHER_PROJECTION_SCHEMA_VERSION] = TEACHER_PROJECTION_SCHEMA_VERSION
    # Source-disjoint dialogue can contain a genuine greeting-only/no-op prefix.  An empty
    # evidence list is valid only for the exact default empty state: every non-default control or
    # semantic fact below still requires references and therefore fails validation without them.
    evidence: list[EvidenceReference] = Field(default_factory=list)
    control: ControlState
    control_evidence: dict[ControlField, list[str]] = Field(default_factory=dict)
    constraints: list[SemanticConstraint] = Field(default_factory=list)
    unresolved_axes: list[UnresolvedAxis] = Field(default_factory=list)
    rejected_items: list[RejectedItem] = Field(default_factory=list)
    termination_reason: TerminationReason | None = None
    termination_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("control_evidence")
    @classmethod
    def validate_control_evidence(
        cls,
        value: dict[ControlField, list[str]],
    ) -> dict[ControlField, list[str]]:
        return {
            field: _unique_evidence_ids(refs, owner=f"control field {field.value}")
            for field, refs in value.items()
        }

    @field_validator("termination_evidence_ids")
    @classmethod
    def validate_termination_evidence_ids(cls, value: list[str]) -> list[str]:
        if not value:
            return []
        return _unique_evidence_ids(value, owner="termination")

    @model_validator(mode="after")
    def validate_current_evidence_grounding(self) -> TeacherCurrentStateSnapshot:
        if any(item.status == ConstraintStatus.SUPERSEDED for item in self.constraints):
            raise ValueError("a current-state snapshot cannot retain superseded constraints")

        # Reuse the canonical public-state invariants (unique IDs, registered
        # evidence, terminal consistency, and auxiliary-control grounding).
        state = self.to_public_state(revision=1)
        registered = {item.evidence_id for item in state.evidence}
        referenced: set[str] = set(state.termination_evidence_ids)
        for refs in state.control_evidence.values():
            referenced.update(refs)
        for collection in (state.constraints, state.unresolved_axes, state.rejected_items):
            for item in collection:
                referenced.update(item.evidence_ids)

        default_control = ControlState()
        for field in ControlField:
            if getattr(state.control, field.value) != getattr(default_control, field.value):
                if not state.control_evidence.get(field):
                    raise ValueError(
                        f"non-default control {field.value} requires explicit control_evidence"
                    )

        dangling = registered.difference(referenced)
        if dangling:
            raise ValueError(f"snapshot contains unreferenced evidence IDs: {sorted(dangling)}")
        return self

    def to_public_state(self, *, revision: int) -> PublicDSUSState:
        """Materialize the snapshot as a state without deriving a ``StatePatch``.

        Revision remains caller-owned because a cold full-history annotation has
        no intrinsic transition count.  This method is an offline projection; it
        does not alter or bypass the production reducer.
        """

        return PublicDSUSState(
            revision=revision,
            initialized=True,
            control=self.control.model_copy(deep=True),
            constraints=[item.model_copy(deep=True) for item in self.constraints],
            unresolved_axes=[item.model_copy(deep=True) for item in self.unresolved_axes],
            rejected_items=[item.model_copy(deep=True) for item in self.rejected_items],
            evidence=[item.model_copy(deep=True) for item in self.evidence],
            control_evidence={field: list(refs) for field, refs in self.control_evidence.items()},
            termination_reason=self.termination_reason,
            termination_evidence_ids=list(self.termination_evidence_ids),
        )


def _strict_json_object(content: str) -> Mapping[str, JsonValue]:
    def reject_duplicate_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        output: dict[str, JsonValue] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"teacher projection JSON repeats key: {key}")
            output[key] = value
        return output

    payload = json.loads(content, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("teacher projection output must be one JSON object")
    return payload


def parse_teacher_projection(
    content: str,
    *,
    public_prompt: DSUSPublicPrompt,
) -> TeacherCurrentStateSnapshot:
    """Strictly parse one raw teacher output and ground its public quotes."""

    snapshot = TeacherCurrentStateSnapshot.model_validate(_strict_json_object(content))
    validate_public_evidence(snapshot.evidence, prompt=public_prompt)
    return snapshot


def _observation_turns(prompt: DSUSPublicPrompt) -> list[tuple[int, PublicTextObservation]]:
    """Mirror the immutable production turn-ID convention for teacher evidence."""

    observations = prompt.public_history if prompt.public_history is not None else prompt.latest_observations
    assert observations is not None
    turn = max((item.turn_id for item in prompt.previous_public_state.evidence), default=0)
    if prompt.public_history is not None:
        turn = 0
    output: list[tuple[int, PublicTextObservation]] = []
    seen_opening_user = False
    for observation in observations:
        if observation.role == "user":
            if seen_opening_user:
                if not output or output[-1][1].role != "system":
                    turn += 1
            else:
                seen_opening_user = True
        elif observation.role == "system":
            turn += 1
        output.append((turn, observation))
    return output


def teacher_projection_user_payload(public_prompt: DSUSPublicPrompt) -> dict[str, JsonValue]:
    """Return a public-only snapshot task, omitting the delta-oriented student schema."""

    observation_key = (
        "public_history" if public_prompt.public_history is not None else "latest_observations"
    )
    observations: list[dict[str, JsonValue]] = []
    for turn_id, observation in _observation_turns(public_prompt):
        row = observation.model_dump(mode="json")
        row["turn_id"] = turn_id
        observations.append(row)
    return {
        "task": "project_final_current_dsus_state",
        "previous_public_state": public_prompt.previous_public_state.canonical_payload(),
        observation_key: observations,
    }


class TeacherProjectionBackend(Protocol):
    backend_id: str
    model_id: str

    def generate(self, *, public_prompt: DSUSPublicPrompt) -> JsonCompletion: ...


class OpenAICompatibleTeacherProjectionBackend:
    """One-call API backend with the snapshot prompt, independent of student runtime."""

    def __init__(
        self,
        *,
        transport: JsonCompletionTransport,
        backend_id: str,
        model_id: str,
        prompt_version: TeacherProjectionPromptVersion = "v1",
    ) -> None:
        self.transport = transport
        self.backend_id = backend_id
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.system_prompt = teacher_projection_system_prompt(prompt_version)
        self.system_prompt_sha256 = teacher_projection_system_prompt_sha256(prompt_version)

    def generate(self, *, public_prompt: DSUSPublicPrompt) -> JsonCompletion:
        return self.transport.complete_json(
            system_prompt=self.system_prompt,
            user_payload=teacher_projection_user_payload(public_prompt),
            retry_feedback=None,
        )


def build_teacher_projection_backend(
    profile: StateCompilerRuntimeProfile,
    *,
    max_output_tokens: int | None = None,
    prompt_version: TeacherProjectionPromptVersion = "v1",
    environ: Mapping[str, str] | None = None,
    transport_factory: Callable[..., JsonCompletionTransport] | None = None,
) -> tuple[OpenAICompatibleTeacherProjectionBackend, OpenAICompatibleTransportProfile]:
    """Reuse one API compiler profile, optionally overriding only teacher output length."""

    if not isinstance(profile.backend, APICompilerBackendProfile):
        raise ValueError("offline teacher projection requires an openai_compatible backend profile")
    transport_profile = profile.backend.transport
    if max_output_tokens is not None:
        if not 1 <= max_output_tokens <= 32768:
            raise ValueError("max_output_tokens must lie in [1,32768]")
        transport_profile = transport_profile.model_copy(
            update={"max_completion_tokens": max_output_tokens}
        )
    # The factory is typed loosely at this public seam to avoid exposing SDK
    # constructor details as part of the projection protocol.
    transport = build_openai_compatible_json_transport(
        transport_profile,
        environ=environ,
        transport_factory=transport_factory,
    )
    backend_suffix = (
        ":teacher_projection"
        if prompt_version == "v1"
        else f":teacher_projection:{prompt_version}"
    )
    return (
        OpenAICompatibleTeacherProjectionBackend(
            transport=transport,
            backend_id=f"{profile.backend.backend_id}{backend_suffix}",
            model_id=transport_profile.model,
            prompt_version=prompt_version,
        ),
        transport_profile,
    )


class TeacherProjectionRawRecord(StrictModel):
    schema_version: Literal[TEACHER_PROJECTION_RAW_VERSION] = TEACHER_PROJECTION_RAW_VERSION
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_output: str


class TeacherProjectionTelemetryRecord(StrictModel):
    schema_version: Literal[TEACHER_PROJECTION_TELEMETRY_VERSION] = (
        TEACHER_PROJECTION_TELEMETRY_VERSION
    )
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_index: Literal[1] = 1
    status: Literal["completed", "transport_error"]
    backend_id: str = Field(min_length=1)
    requested_model: str | None = None
    returned_model: str | None = None
    finish_reason: str | None = None
    raw_content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0.0)
    transport_error_kind: str | None = None


class TeacherProjectionValidatedRecord(StrictModel):
    schema_version: Literal[TEACHER_PROJECTION_VALIDATED_VERSION] = (
        TEACHER_PROJECTION_VALIDATED_VERSION
    )
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot: TeacherCurrentStateSnapshot


class TeacherProjectionValidationRecord(StrictModel):
    """Raw-output-free validation status; error messages are deliberately excluded."""

    schema_version: Literal[TEACHER_PROJECTION_VALIDATION_VERSION] = (
        TEACHER_PROJECTION_VALIDATION_VERSION
    )
    sample_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["valid", "invalid", "transport_error"]
    failure_kind: str | None = None


def generate_one_teacher_projection(
    record: PublicRLPromptRecord,
    *,
    backend: TeacherProjectionBackend,
) -> tuple[TeacherProjectionRawRecord, TeacherProjectionTelemetryRecord]:
    """Call the teacher once and retain raw output without parse, repair, or retry."""

    started = time.perf_counter()
    try:
        completion = backend.generate(public_prompt=record.public_prompt)
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return (
            TeacherProjectionRawRecord(
                sample_id=record.sample_id,
                public_prompt_sha256=record.public_prompt_sha256,
                model_output="",
            ),
            TeacherProjectionTelemetryRecord(
                sample_id=record.sample_id,
                public_prompt_sha256=record.public_prompt_sha256,
                status="transport_error",
                backend_id=backend.backend_id,
                requested_model=backend.model_id,
                latency_ms=latency_ms,
                transport_error_kind=type(exc).__name__,
            ),
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    return (
        TeacherProjectionRawRecord(
            sample_id=record.sample_id,
            public_prompt_sha256=record.public_prompt_sha256,
            model_output=completion.content,
        ),
        TeacherProjectionTelemetryRecord(
            sample_id=record.sample_id,
            public_prompt_sha256=record.public_prompt_sha256,
            status="completed",
            backend_id=backend.backend_id,
            requested_model=completion.requested_model,
            returned_model=completion.returned_model,
            finish_reason=completion.finish_reason,
            raw_content_sha256=content_sha256(completion.content),
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            total_tokens=completion.usage.total_tokens,
            latency_ms=latency_ms,
        ),
    )


def generate_teacher_projection_records(
    records: Sequence[PublicRLPromptRecord],
    *,
    backend: TeacherProjectionBackend,
    workers: int = 1,
) -> tuple[list[TeacherProjectionRawRecord], list[TeacherProjectionTelemetryRecord]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        pairs = [generate_one_teacher_projection(record, backend=backend) for record in records]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pairs = list(
                executor.map(
                    lambda record: generate_one_teacher_projection(record, backend=backend),
                    records,
                )
            )
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def validate_teacher_projection_records(
    raw_records: Sequence[TeacherProjectionRawRecord],
    public_records: Sequence[PublicRLPromptRecord],
) -> tuple[list[TeacherProjectionValidatedRecord], list[TeacherProjectionValidationRecord]]:
    """Validate raw outputs offline without repair, retry, or another model call."""

    public_by_id = {record.sample_id: record for record in public_records}
    if len(public_by_id) != len(public_records):
        raise ValueError("public records contain duplicate sample IDs")
    validated: list[TeacherProjectionValidatedRecord] = []
    audit: list[TeacherProjectionValidationRecord] = []
    for raw in raw_records:
        public = public_by_id.get(raw.sample_id)
        if public is None:
            raise ValueError(f"raw projection references an unknown public sample: {raw.sample_id}")
        if raw.public_prompt_sha256 != public.public_prompt_sha256:
            raise ValueError(f"raw/public prompt hash mismatch: {raw.sample_id}")
        if not raw.model_output:
            audit.append(
                TeacherProjectionValidationRecord(
                    sample_id=raw.sample_id,
                    public_prompt_sha256=raw.public_prompt_sha256,
                    status="transport_error",
                )
            )
            continue
        try:
            snapshot = parse_teacher_projection(raw.model_output, public_prompt=public.public_prompt)
        except Exception as exc:
            audit.append(
                TeacherProjectionValidationRecord(
                    sample_id=raw.sample_id,
                    public_prompt_sha256=raw.public_prompt_sha256,
                    status="invalid",
                    failure_kind=type(exc).__name__,
                )
            )
            continue
        validated.append(
            TeacherProjectionValidatedRecord(
                sample_id=raw.sample_id,
                public_prompt_sha256=raw.public_prompt_sha256,
                snapshot=snapshot,
            )
        )
        audit.append(
            TeacherProjectionValidationRecord(
                sample_id=raw.sample_id,
                public_prompt_sha256=raw.public_prompt_sha256,
                status="valid",
            )
        )
    return validated, audit


__all__ = [
    "TEACHER_PROJECTION_SCHEMA_VERSION",
    "TEACHER_PROJECTION_SYSTEM_PROMPT",
    "TEACHER_PROJECTION_SYSTEM_PROMPT_SEMANTIC_GUARD_V2",
    "TEACHER_PROJECTION_SYSTEM_PROMPT_SHA256_V1",
    "TEACHER_PROJECTION_SYSTEM_PROMPT_V1",
    "OpenAICompatibleTeacherProjectionBackend",
    "TeacherCurrentStateSnapshot",
    "TeacherProjectionBackend",
    "TeacherProjectionRawRecord",
    "TeacherProjectionTelemetryRecord",
    "TeacherProjectionPromptVersion",
    "TeacherProjectionValidatedRecord",
    "TeacherProjectionValidationRecord",
    "build_teacher_projection_backend",
    "generate_one_teacher_projection",
    "generate_teacher_projection_records",
    "parse_teacher_projection",
    "teacher_projection_user_payload",
    "teacher_projection_system_prompt",
    "teacher_projection_system_prompt_sha256",
    "validate_teacher_projection_records",
]
