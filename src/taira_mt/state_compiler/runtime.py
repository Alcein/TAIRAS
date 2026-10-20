"""Runtime contracts for the trainable DSUS v2 state compiler.

The state compiler is deliberately split into two layers:

* a backend produces one JSON candidate from a public :class:`DSUSPublicPrompt`;
* :class:`DSUSStateCompilerRuntime` validates the candidate, grounds every
  evidence quote, and applies the typed delta with the deterministic reducer.

Both an OpenAI-compatible API model and a local LoRA model implement the same
backend protocol.  Consequently model replacement changes only the first
generation step; schema validation, state transition semantics, telemetry and
the downstream TAIRAS adapter remain identical.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from pydantic import Field, JsonValue, model_validator

from taira_mt.models import StrictModel
from taira_mt.provenance import content_sha256
from taira_mt.providers.openai_compatible import (
    JsonCompletionTransport,
    OpenAICompatibleTransportProfile,
    build_openai_compatible_json_transport,
)
from taira_mt.state_compiler import (
    PATCH_SCHEMA_VERSION,
    EvidenceReference,
    EvidenceSource,
    PublicDSUSState,
    StateOperation,
    StatePatch,
    reduce_state,
)
from taira_mt.state_training.training_views import DSUSPublicPrompt, PublicTextObservation

STATE_COMPILER_RUNTIME_VERSION = "dsus_state_compiler_runtime_v1"
STATE_COMPILER_PROFILE_VERSION = "dsus_state_compiler_runtime_profile_v1"

_STATE_COMPILER_SYSTEM_PROMPT = """You are a DSUS State Compiler.
Read only the supplied public prompt and return exactly one JSON object with exactly these keys:
schema_version, base_revision, evidence, delta. Never emit markdown, explanations, an action,
a Thought Pattern identifier, or a private benchmark field. additionalProperties are forbidden.

The exact top-level form is:
{"schema_version":"dsus_state_patch_v2","base_revision":<previous state revision>,
 "evidence":[...],"delta":[...]}

Each evidence object has exactly:
{"evidence_id":"e1","turn_id":<nonnegative integer>,"source":"user|system|catalog",
 "quote":"an exact nonempty substring of that public observation"}.
Do not use id, dialogue_turn, public_history, or tool as field/source names. Input observations carry
an explicit public `turn_id`; copy it exactly. Map a tool observation to evidence source `catalog`.
Every operation cites one or more IDs from this patch's evidence list.

Every delta object uses the discriminator field `op` (never `operation`) and must be exactly one of:
{"op":"initialize_state","evidence_ids":[...]}
{"op":"set_control","field":"<one control field>","value":"<one allowed value>","evidence_ids":[...]}
{"op":"upsert_constraint","constraint_id":"...","name":"...","value":<JSON value>,
 "polarity":"must|prefer|avoid","importance":"hard|soft","status":"active|uncertain",
 "evidence_ids":[...]}
{"op":"supersede_constraint","constraint_id":"existing id","evidence_ids":[...]}
{"op":"remove_constraint","constraint_id":"existing id","evidence_ids":[...]}
{"op":"add_unresolved_axis","axis_id":"...","axis_type":"...","description":"...",
 "options":[],"evidence_ids":[...]}
{"op":"resolve_axis","axis_id":"existing id","evidence_ids":[...]}
{"op":"record_feedback","feedback_type":"answer|acceptance|rejection|correction|explanation_request",
 "item_id":null,"reason":null,"evidence_ids":[...]}
{"op":"terminate","reason":"accepted|abandoned|no_solution|user_ended","evidence_ids":[...]}
{"op":"noop","reason":"...","evidence_ids":[...]}

Control fields and allowed values are:
interaction_phase = opening|elicitation|grounding|recommendation|feedback|repair|terminal;
information_status = missing|sufficient|ambiguous|conflicting;
intent_stability = stable|strengthened|relaxed|revised;
feedback_type = none|answer|acceptance|rejection|correction|explanation_request;
explanation_status = none|requested|pending|resolved;
commitment_status = unknown|exploring|comparing|ready|accepted|abandoned;
patience_risk = unknown|low|medium|high|exhausted.

Emit only the minimal state change relative to previous_public_state. If it is uninitialized, include
initialize_state. Preserve facts not contradicted by new public evidence. Use separate set_control
operations, not a fields object. Use stable IDs so later patches can supersede/resolve prior entries.

Apply these authoring rules exactly:
1. `initialize_state` already creates the default controls opening/missing/stable/none/none/unknown/
   unknown. Do not emit `set_control` for a value already present after initialization or in the
   previous state.
2. Never update every control field mechanically. Change a control only when the new user evidence
   makes its value different. Auxiliary commitment/patience fields require explicit evidence; do not
   infer low patience or readiness from an ordinary opening.
3. A generic acknowledgment such as "okay", "got it", or "thank you" is `noop` unless it explicitly
   accepts/rejects an item, answers a clarification, changes a preference, requests an explanation,
   or ends the shopping task. Do not equate politeness with product acceptance.
4. `record_feedback` itself updates feedback_type; never also `set_control` feedback_type in the same
   patch. `terminate` itself updates interaction_phase and accepted/abandoned commitment; never touch
   those generated fields separately in the same patch. `noop` must be the sole delta operation.
5. Treat previous_public_state as authoritative. Resolve/remove/supersede only IDs that occur there,
   copy those IDs exactly, and do not silently overwrite an existing constraint with a new meaning.
6. System/catalog claims are context, not user preferences. Add a constraint only from a user-stated,
   accepted, or corrected requirement. Reuse one distinct evidence object across multiple operations;
   do not duplicate the same quote merely to give it another evidence ID.
7. Evidence IDs must be fresh relative to previous_public_state. Prefer deterministic IDs such as
   `ev_r<base_revision>_1`, `ev_r<base_revision>_2` in observation order.
8. When `public_history` contains an entire multi-turn dialogue and the previous state is empty,
   reconstruct only the CURRENT state after the final visible user utterance. Do not replay every
   historical feedback event as a sequence of delta operations. A patch may mutate each control
   field, constraint ID, unresolved-axis ID, or rejected-item ID at most once; keep only the final
   current value for that entity.
9. `initialize_state` is evidence-grounded too: its `evidence_ids` must contain at least one ID,
   normally the opening user's shopping-request evidence. Every evidence object must be cited by at
   least one delta operation; omit evidence that would otherwise be unused.
10. In `record_feedback`, `reason` MUST be null for answer, acceptance, correction, and
    explanation_request. A non-null reason is permitted only for rejection with a non-null concrete
    item_id copied from a publicly displayed recommendation. If the rejected item cannot be resolved
    to one displayed item_id, use both `item_id:null` and `reason:null`; encode transferable negative
    product requirements as `avoid` constraints instead.
11. If the final user explicitly ends or accepts the task, emit `terminate` last. Because terminate
    already writes interaction_phase and commitment_status, do not emit set_control for either field.
    Keep the JSON compact and complete; prefer the shortest exact evidence spans that justify the
    current state.
"""


class DeterministicGenerationConfig(StrictModel):
    """Frozen greedy decoding contract shared by API and local backends."""

    temperature: Literal[0.0] = 0.0
    do_sample: Literal[False] = False
    num_beams: Literal[1] = 1
    enable_thinking: Literal[False] = False
    max_new_tokens: int = Field(default=1024, ge=64, le=8192)


class APICompilerBackendProfile(StrictModel):
    kind: Literal["openai_compatible"] = "openai_compatible"
    backend_id: str = Field(min_length=1, max_length=200)
    transport: OpenAICompatibleTransportProfile
    generation: DeterministicGenerationConfig = Field(default_factory=DeterministicGenerationConfig)

    @model_validator(mode="after")
    def validate_transport_generation_alignment(self) -> APICompilerBackendProfile:
        if self.transport.temperature != self.generation.temperature:
            raise ValueError("API transport temperature must match deterministic generation config")
        if self.transport.max_completion_tokens != self.generation.max_new_tokens:
            raise ValueError("API token limit must match deterministic generation config")
        if self.transport.enable_thinking not in (False, None):
            raise ValueError("DSUS API compiler cannot enable thinking")
        return self


class LocalCompilerBackendProfile(StrictModel):
    kind: Literal["local_lora"] = "local_lora"
    backend_id: str = Field(min_length=1, max_length=200)
    base_model_name_or_path: str = Field(min_length=1, max_length=500)
    base_model_revision: str | None = Field(default=None, max_length=200)
    adapter_path: str = Field(min_length=1, max_length=1000)
    load_in_4bit: bool = False
    generation: DeterministicGenerationConfig = Field(default_factory=DeterministicGenerationConfig)


CompilerBackendProfile = Annotated[
    APICompilerBackendProfile | LocalCompilerBackendProfile,
    Field(discriminator="kind"),
]


class StateCompilerRuntimeProfile(StrictModel):
    """Serializable, secret-free configuration for one swappable compiler."""

    schema_version: Literal[STATE_COMPILER_PROFILE_VERSION] = STATE_COMPILER_PROFILE_VERSION
    compiler_id: str = Field(min_length=1, max_length=240)
    backend: CompilerBackendProfile
    max_attempts: int = Field(default=2, ge=1, le=4)
    fallback_to_rule: bool = True

    def secret_free_telemetry(self) -> dict[str, JsonValue]:
        backend: dict[str, JsonValue]
        if isinstance(self.backend, APICompilerBackendProfile):
            backend = {
                "kind": self.backend.kind,
                "backend_id": self.backend.backend_id,
                "transport": self.backend.transport.secret_free_telemetry(),
                "generation": self.backend.generation.model_dump(mode="json"),
            }
        else:
            backend = {
                "kind": self.backend.kind,
                "backend_id": self.backend.backend_id,
                "base_model_name_or_path": self.backend.base_model_name_or_path,
                "base_model_revision": self.backend.base_model_revision,
                "adapter_path": self.backend.adapter_path,
                "generation": self.backend.generation.model_dump(mode="json"),
            }
        payload: dict[str, JsonValue] = {
            "schema_version": self.schema_version,
            "compiler_id": self.compiler_id,
            "backend": backend,
            "max_attempts": self.max_attempts,
            "fallback_to_rule": self.fallback_to_rule,
        }
        return {**payload, "profile_sha256": content_sha256(payload)}


class CompilerPatchOutput(StrictModel):
    """Exact JSON contract learned by SFT/RLVR and used at runtime."""

    schema_version: Literal[PATCH_SCHEMA_VERSION] = PATCH_SCHEMA_VERSION
    base_revision: int = Field(ge=0)
    evidence: list[EvidenceReference] = Field(min_length=1)
    delta: list[StateOperation] = Field(min_length=1)

    def as_patch(self) -> StatePatch:
        return StatePatch(
            schema_version=self.schema_version,
            base_revision=self.base_revision,
            evidence=self.evidence,
            operations=self.delta,
        )


class CompilerBackendCompletion(StrictModel):
    """One raw completion plus secret-free measured runtime metadata."""

    content: str = Field(min_length=1)
    backend_id: str = Field(min_length=1)
    requested_model: str | None = None
    returned_model: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0.0)


class CompilerAttemptTelemetry(StrictModel):
    """Raw-output-free audit record for one bounded JSON attempt."""

    attempt_index: int = Field(ge=1)
    status: Literal["committed", "rejected"]
    backend_id: str = Field(min_length=1)
    requested_model: str | None = None
    returned_model: str | None = None
    finish_reason: str | None = None
    raw_content_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0.0)
    failure_kind: str | None = None


class StateCompilerResult(StrictModel):
    """A committed, replayed transition shared by every backend."""

    schema_version: Literal[STATE_COMPILER_RUNTIME_VERSION] = STATE_COMPILER_RUNTIME_VERSION
    compiler_id: str = Field(min_length=1)
    backend_id: str = Field(min_length=1)
    public_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_state: PublicDSUSState
    patch: StatePatch
    next_state: PublicDSUSState
    attempts: list[CompilerAttemptTelemetry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_committed_transition(self) -> StateCompilerResult:
        if self.public_prompt_sha256 == "":  # pragma: no cover - regex already enforces this
            raise ValueError("public prompt hash cannot be blank")
        if self.attempts[-1].status != "committed":
            raise ValueError("a successful compiler result must end with a committed attempt")
        replayed = reduce_state(self.previous_state, self.patch)
        if replayed.canonical_json() != self.next_state.canonical_json():
            raise ValueError("compiler result next_state does not match reducer replay")
        return self

    @property
    def retry_count(self) -> int:
        return len(self.attempts) - 1

    @property
    def latency_ms(self) -> float:
        return sum(item.latency_ms for item in self.attempts)

    @property
    def total_tokens(self) -> int | None:
        values = [item.total_tokens for item in self.attempts]
        return None if any(value is None for value in values) else sum(int(value) for value in values)


class StateCompilerBackend(Protocol):
    backend_id: str
    model_id: str | None
    generation: DeterministicGenerationConfig

    def generate(
        self,
        *,
        public_prompt: DSUSPublicPrompt,
        retry_feedback: str | None = None,
    ) -> CompilerBackendCompletion: ...


class LocalJSONGenerator(Protocol):
    """Small seam used by tests and by the future loaded Qwen LoRA runtime."""

    model_id: str

    def generate_json(
        self,
        *,
        model_input: str,
        retry_feedback: str | None,
        generation: DeterministicGenerationConfig,
    ) -> CompilerBackendCompletion: ...


class OpenAICompatibleStateCompilerBackend:
    """API backend; transport construction remains secret-safe and provider-agnostic."""

    def __init__(
        self,
        *,
        transport: JsonCompletionTransport,
        backend_id: str,
        model_id: str,
        generation: DeterministicGenerationConfig,
    ) -> None:
        self.transport = transport
        self.backend_id = backend_id
        self.model_id = model_id
        self.generation = generation

    def generate(
        self,
        *,
        public_prompt: DSUSPublicPrompt,
        retry_feedback: str | None = None,
    ) -> CompilerBackendCompletion:
        started = time.perf_counter()
        completion = self.transport.complete_json(
            system_prompt=_STATE_COMPILER_SYSTEM_PROMPT,
            user_payload=_api_model_payload(public_prompt),
            retry_feedback=retry_feedback,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return CompilerBackendCompletion(
            content=completion.content,
            backend_id=self.backend_id,
            requested_model=completion.requested_model,
            returned_model=completion.returned_model,
            finish_reason=completion.finish_reason,
            input_tokens=completion.usage.input_tokens,
            output_tokens=completion.usage.output_tokens,
            total_tokens=completion.usage.total_tokens,
            latency_ms=latency_ms,
        )


def _api_model_payload(public_prompt: DSUSPublicPrompt) -> dict[str, JsonValue]:
    """Attach public turn IDs without changing the frozen prompt identity.

    Turn numbering is deterministic protocol structure, not a private label.  API teachers receive
    the same text as local compilers, but no longer need to reconstruct turn indices from prose.
    The original :class:`DSUSPublicPrompt` remains the object hashed, verified, and stored.
    """

    payload = public_prompt.canonical_payload()
    observation_key = (
        "public_history" if public_prompt.public_history is not None else "latest_observations"
    )
    annotated = []
    for turn_id, observation in _observation_turns(public_prompt):
        row = observation.model_dump(mode="json")
        row["turn_id"] = turn_id
        annotated.append(row)
    payload[observation_key] = annotated
    return payload


class LocalLoRAStateCompilerBackend:
    """Local backend over an injected deterministic JSON generator."""

    def __init__(
        self,
        *,
        generator: LocalJSONGenerator,
        backend_id: str,
        generation: DeterministicGenerationConfig,
    ) -> None:
        self.generator = generator
        self.backend_id = backend_id
        self.model_id = generator.model_id
        self.generation = generation

    def generate(
        self,
        *,
        public_prompt: DSUSPublicPrompt,
        retry_feedback: str | None = None,
    ) -> CompilerBackendCompletion:
        completion = self.generator.generate_json(
            model_input=public_prompt.canonical_json(),
            retry_feedback=retry_feedback,
            generation=self.generation,
        )
        if completion.backend_id != self.backend_id:
            raise ValueError("local generator returned telemetry for a different backend")
        return completion


class HuggingFaceGreedyJSONGenerator:
    """Lazy-compatible local generator over an already loaded model/tokenizer.

    Constructing this class does not import torch, move a tensor or load a
    checkpoint.  The profile factory below performs those operations only when
    an actual local evaluation is explicitly launched.
    """

    def __init__(self, *, model: Any, tokenizer: Any, model_id: str, backend_id: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.backend_id = backend_id

    def generate_json(
        self,
        *,
        model_input: str,
        retry_feedback: str | None,
        generation: DeterministicGenerationConfig,
    ) -> CompilerBackendCompletion:
        import torch

        content = model_input
        if retry_feedback is not None:
            content = f"{model_input}\n\nVALIDATION RETRY:\n{retry_feedback}"
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=generation.enable_thinking,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        try:
            device = next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_tokens = int(encoded["attention_mask"].sum().item())
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=generation.do_sample,
                num_beams=generation.num_beams,
                max_new_tokens=generation.max_new_tokens,
                # Qwen3.5 exposes a cache-backed auto-compile path in recent
                # Transformers releases.  This workload generates one short
                # JSON patch at a time; explicit eager decoding avoids a
                # multi-minute inductor build on every fresh worker while
                # preserving the same greedy decoding contract.
                disable_compile=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        prompt_width = int(encoded["input_ids"].shape[1])
        generated = output[:, prompt_width:]
        output_tokens = int(generated.shape[1])
        decoded = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(decoded) != 1 or not decoded[0].strip():
            raise RuntimeError("local compiler returned an empty or non-single completion")
        return CompilerBackendCompletion(
            content=decoded[0].strip(),
            backend_id=self.backend_id,
            requested_model=self.model_id,
            returned_model=self.model_id,
            finish_reason="stop" if output_tokens < generation.max_new_tokens else "length",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms,
        )


class StateCompilerFailure(RuntimeError):
    """Bounded failure that carries telemetry but never raw model text."""

    def __init__(
        self,
        *,
        compiler_id: str,
        attempts: Sequence[CompilerAttemptTelemetry],
        previous_state: PublicDSUSState,
    ) -> None:
        self.compiler_id = compiler_id
        self.attempts = list(attempts)
        # This is the typed public state supplied to the failed transition,
        # never raw model output.  It lets the Agent preserve a committed
        # dialogue prefix when only the newest incremental patch is invalid.
        self.previous_state = previous_state.model_copy(deep=True)
        failure = self.attempts[-1].failure_kind if self.attempts else "unknown"
        super().__init__(
            f"state compiler {compiler_id!r} failed after {len(self.attempts)} attempts ({failure})"
        )


def _strict_json_object(content: str) -> Mapping[str, JsonValue]:
    def reject_duplicate_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        output: dict[str, JsonValue] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"compiler JSON repeats key: {key}")
            output[key] = value
        return output

    payload = json.loads(content, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise ValueError("compiler output must be one JSON object")
    return payload


def _observation_turns(prompt: DSUSPublicPrompt) -> list[tuple[int, PublicTextObservation]]:
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
                # A system/user exchange shares one dialogue transition id.
                if not output or output[-1][1].role != "system":
                    turn += 1
            else:
                seen_opening_user = True
        elif observation.role == "system":
            turn += 1
        output.append((turn, observation))
    return output


def validate_public_evidence(
    evidence: Sequence[EvidenceReference],
    *,
    prompt: DSUSPublicPrompt,
) -> None:
    """Require every cited quote to occur at the cited public source/turn."""

    by_source = {
        EvidenceSource.USER: {"user"},
        EvidenceSource.SYSTEM: {"system"},
        EvidenceSource.CATALOG: {"catalog", "tool"},
    }
    observations = _observation_turns(prompt)
    for item in evidence:
        candidates = [
            observation.text
            for turn_id, observation in observations
            if turn_id == item.turn_id and observation.role in by_source[item.source]
        ]
        if not any(item.quote in text for text in candidates):
            raise ValueError(
                f"evidence {item.evidence_id!r} is not an exact quote at public "
                f"{item.source.value} turn {item.turn_id}"
            )


def bind_public_evidence(
    evidence: Sequence[EvidenceReference],
    *,
    prompt: DSUSPublicPrompt,
) -> list[EvidenceReference]:
    """Resolve exact public quotes to their deterministic source and turn address.

    Real chat logs frequently contain consecutive messages from the same role, so a small semantic
    compiler should not have to reproduce the runtime's bookkeeping convention.  The compiler still
    has to emit a verbatim public quote.  This linker changes only ``source`` and ``turn_id`` when that
    quote can be found in the public observations; evidence text and state semantics remain untouched.

    If a quote occurs more than once, the declared turn is preferred, then the declared source, then
    the most recent public occurrence.  Missing quotes are left unchanged and rejected by the normal
    evidence validator.
    """

    observations = _observation_turns(prompt)
    source_by_role = {
        "user": EvidenceSource.USER,
        "system": EvidenceSource.SYSTEM,
        "catalog": EvidenceSource.CATALOG,
        "tool": EvidenceSource.CATALOG,
    }
    rebound: list[EvidenceReference] = []
    for item in evidence:
        candidates = [
            (index, turn_id, source_by_role[observation.role])
            for index, (turn_id, observation) in enumerate(observations)
            if observation.role in source_by_role and item.quote in observation.text
        ]
        if not candidates:
            rebound.append(item.model_copy(deep=True))
            continue
        exact_address = [
            candidate
            for candidate in candidates
            if candidate[1] == item.turn_id and candidate[2] == item.source
        ]
        same_turn = [candidate for candidate in candidates if candidate[1] == item.turn_id]
        same_source = [candidate for candidate in candidates if candidate[2] == item.source]
        chosen = (exact_address or same_turn or same_source or candidates)[-1]
        rebound.append(
            item.model_copy(update={"turn_id": chosen[1], "source": chosen[2]}, deep=True)
        )
    return rebound


def build_public_compiler_prompt(
    *,
    previous_state: PublicDSUSState,
    observations: Sequence[PublicTextObservation],
) -> DSUSPublicPrompt:
    if not observations:
        raise ValueError("state compilation requires at least one public observation")
    return DSUSPublicPrompt(
        previous_public_state=previous_state,
        public_history=list(observations),
    )


def _attempt(
    *,
    attempt_index: int,
    backend: StateCompilerBackend,
    elapsed_ms: float,
    completion: CompilerBackendCompletion | None,
    failure: Exception | None,
) -> CompilerAttemptTelemetry:
    return CompilerAttemptTelemetry(
        attempt_index=attempt_index,
        status="rejected" if failure is not None else "committed",
        backend_id=backend.backend_id,
        requested_model=(backend.model_id if completion is None else completion.requested_model),
        returned_model=None if completion is None else completion.returned_model,
        finish_reason=None if completion is None else completion.finish_reason,
        raw_content_sha256=None if completion is None else content_sha256(completion.content),
        input_tokens=None if completion is None else completion.input_tokens,
        output_tokens=None if completion is None else completion.output_tokens,
        total_tokens=None if completion is None else completion.total_tokens,
        latency_ms=elapsed_ms if completion is None else completion.latency_ms,
        failure_kind=None if failure is None else type(failure).__name__,
    )


class DSUSStateCompilerRuntime:
    """Backend-neutral bounded JSON compiler with executable transition checks."""

    def __init__(self, *, compiler_id: str, backend: StateCompilerBackend, max_attempts: int = 2) -> None:
        if not compiler_id.strip():
            raise ValueError("compiler_id cannot be blank")
        if not 1 <= max_attempts <= 4:
            raise ValueError("max_attempts must lie in [1,4]")
        self.compiler_id = compiler_id
        self.backend = backend
        self.max_attempts = max_attempts

    def compile(self, *, prompt: DSUSPublicPrompt) -> StateCompilerResult:
        attempts: list[CompilerAttemptTelemetry] = []
        retry_feedback: str | None = None
        for attempt_index in range(1, self.max_attempts + 1):
            completion: CompilerBackendCompletion | None = None
            started = time.perf_counter()
            try:
                completion = self.backend.generate(
                    public_prompt=prompt,
                    retry_feedback=retry_feedback,
                )
                payload = CompilerPatchOutput.model_validate(_strict_json_object(completion.content))
                patch = payload.as_patch()
                if patch.base_revision != prompt.previous_public_state.revision:
                    raise ValueError("compiler patch base_revision does not match previous state")
                patch = patch.model_copy(
                    update={"evidence": bind_public_evidence(patch.evidence, prompt=prompt)},
                    deep=True,
                )
                validate_public_evidence(patch.evidence, prompt=prompt)
                next_state = reduce_state(prompt.previous_public_state, patch)
                attempts.append(
                    _attempt(
                        attempt_index=attempt_index,
                        backend=self.backend,
                        elapsed_ms=(time.perf_counter() - started) * 1000.0,
                        completion=completion,
                        failure=None,
                    )
                )
                return StateCompilerResult(
                    compiler_id=self.compiler_id,
                    backend_id=self.backend.backend_id,
                    public_prompt_sha256=prompt.content_hash(),
                    previous_state=prompt.previous_public_state,
                    patch=patch,
                    next_state=next_state,
                    attempts=attempts,
                )
            except Exception as exc:
                attempts.append(
                    _attempt(
                        attempt_index=attempt_index,
                        backend=self.backend,
                        elapsed_ms=(time.perf_counter() - started) * 1000.0,
                        completion=completion,
                        failure=exc,
                    )
                )
                retry_feedback = (
                    f"The prior candidate failed {type(exc).__name__}. Return only a fresh JSON object "
                    "with exactly schema_version, base_revision, evidence, delta; cite exact public quotes "
                    "and obey the typed transition schema."
                )
        raise StateCompilerFailure(
            compiler_id=self.compiler_id,
            attempts=attempts,
            previous_state=prompt.previous_public_state,
        )


def load_state_compiler_profile(path: Path) -> tuple[StateCompilerRuntimeProfile, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return StateCompilerRuntimeProfile.model_validate(payload), path.parent


def build_state_compiler_backend(
    profile: StateCompilerRuntimeProfile,
    *,
    profile_base_dir: Path,
    environ: Mapping[str, str] | None = None,
    api_transport_factory: Callable[..., JsonCompletionTransport] | None = None,
) -> StateCompilerBackend:
    """Build the selected backend only when an explicit runtime launch occurs."""

    backend = profile.backend
    if isinstance(backend, APICompilerBackendProfile):
        transport = build_openai_compatible_json_transport(
            backend.transport,
            environ=environ,
            transport_factory=api_transport_factory,
        )
        return OpenAICompatibleStateCompilerBackend(
            transport=transport,
            backend_id=backend.backend_id,
            model_id=backend.transport.model,
            generation=backend.generation,
        )

    from taira_mt.state_training.sft_harness import load_model_and_adapter_for_generation

    adapter_path = Path(backend.adapter_path)
    if not adapter_path.is_absolute():
        adapter_path = (profile_base_dir / adapter_path).resolve()
    model, tokenizer = load_model_and_adapter_for_generation(
        base_model_name_or_path=backend.base_model_name_or_path,
        adapter_path=adapter_path,
        base_model_revision=backend.base_model_revision,
        load_in_4bit=backend.load_in_4bit,
    )
    generator = HuggingFaceGreedyJSONGenerator(
        model=model,
        tokenizer=tokenizer,
        model_id=backend.base_model_name_or_path,
        backend_id=backend.backend_id,
    )
    return LocalLoRAStateCompilerBackend(
        generator=generator,
        backend_id=backend.backend_id,
        generation=backend.generation,
    )


__all__ = [
    "APICompilerBackendProfile",
    "CompilerAttemptTelemetry",
    "CompilerBackendCompletion",
    "CompilerPatchOutput",
    "DSUSStateCompilerRuntime",
    "DeterministicGenerationConfig",
    "HuggingFaceGreedyJSONGenerator",
    "LocalCompilerBackendProfile",
    "LocalJSONGenerator",
    "LocalLoRAStateCompilerBackend",
    "OpenAICompatibleStateCompilerBackend",
    "STATE_COMPILER_PROFILE_VERSION",
    "STATE_COMPILER_RUNTIME_VERSION",
    "StateCompilerBackend",
    "StateCompilerFailure",
    "StateCompilerResult",
    "StateCompilerRuntimeProfile",
    "bind_public_evidence",
    "build_public_compiler_prompt",
    "build_state_compiler_backend",
    "load_state_compiler_profile",
    "validate_public_evidence",
]
