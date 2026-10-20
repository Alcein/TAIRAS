"""Minimal OpenAI-compatible JSON transport with secret-free records.

The transport itself is deliberately small: it owns an in-memory SDK client
and returns only typed completion metadata.  The environment-profile helper in
this module is the supported way to construct it for a real external run.  It
keeps credentials and endpoint values out of Pydantic models, manifests, and
normal ``repr``/telemetry paths.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from taira_mt.models import StrictModel
from taira_mt.provenance import canonical_json, content_sha256

OPENAI_COMPATIBLE_TRANSPORT_PROFILE_VERSION = "openai_compatible_transport_profile_v1"

_ENVIRONMENT_VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class OpenAICompatibleTransportProfile(StrictModel):
    """Secret-free, serializable profile for one external model route.

    The profile intentionally stores *environment variable names*, never their
    values.  In particular, a private endpoint and API key are only read when
    :func:`build_openai_compatible_json_transport` is explicitly invoked.  The
    returned ``secret_free_telemetry`` can safely be copied into a run manifest.
    """

    schema_version: Literal[OPENAI_COMPATIBLE_TRANSPORT_PROFILE_VERSION] = (
        OPENAI_COMPATIBLE_TRANSPORT_PROFILE_VERSION
    )
    provider_profile_id: str = Field(min_length=1, max_length=160)
    model: str = Field(min_length=1, max_length=200)
    api_key_env: str = Field(default="TAIRA_MT_API_KEY", min_length=1, max_length=160)
    base_url_env: str = Field(default="TAIRA_MT_BASE_URL", min_length=1, max_length=160)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_completion_tokens: int = Field(default=768, ge=1, le=32768)
    completion_token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    json_response_format: bool = True
    reasoning_effort: str | None = Field(default=None, max_length=64)
    enable_thinking: bool | None = None
    timeout_seconds: float = Field(default=120.0, gt=0.0, le=1800.0)

    @field_validator("provider_profile_id", "model", "api_key_env", "base_url_env", mode="before")
    @classmethod
    def normalize_required_text(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("transport profile text fields cannot be blank")
        return value.strip()

    @field_validator("api_key_env", "base_url_env")
    @classmethod
    def validate_environment_variable_name(cls, value: str) -> str:
        if _ENVIRONMENT_VARIABLE_NAME.fullmatch(value) is None:
            raise ValueError("transport profile environment variable names must be shell-safe identifiers")
        return value

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("reasoning_effort must be a non-blank string or null")
        return value.strip()

    @model_validator(mode="after")
    def validate_distinct_environment_variable_names(self) -> OpenAICompatibleTransportProfile:
        if self.api_key_env == self.base_url_env:
            raise ValueError("API-key and base-URL environment variable names must differ")
        return self

    def secret_free_telemetry(self) -> dict[str, object]:
        """Return provenance fields that intentionally exclude endpoint/key values."""

        payload = {
            "schema_version": self.schema_version,
            "provider_profile_id": self.provider_profile_id,
            "requested_model": self.model,
            "api_key_env": self.api_key_env,
            "base_url_env": self.base_url_env,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "completion_token_parameter": self.completion_token_parameter,
            "json_response_format": self.json_response_format,
            "reasoning_effort": self.reasoning_effort,
            "enable_thinking": self.enable_thinking,
            "timeout_seconds": self.timeout_seconds,
        }
        return {
            **payload,
            "profile_sha256": content_sha256(payload),
        }


class JsonTokenUsage(StrictModel):
    """Provider-reported usage; completion already includes reasoning tokens."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class JsonCompletion(StrictModel):
    content: str = Field(min_length=1)
    provider_request_id: str | None = None
    requested_model: str = Field(min_length=1)
    returned_model: str = Field(min_length=1)
    finish_reason: str | None = None
    usage: JsonTokenUsage


class JsonCompletionTransport(Protocol):
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: Mapping[str, object],
        retry_feedback: str | None = None,
    ) -> JsonCompletion: ...


class JsonResponseError(RuntimeError):
    """Malformed provider response with safe metadata, including billed tokens."""

    def __init__(self, code: str, *, response: Any, requested_model: str) -> None:
        super().__init__(code)
        choices = response.choices
        choice = choices[0] if len(choices) == 1 else None
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        self.audit_metadata = {
            "failure_code": code,
            "requested_model": requested_model,
            "returned_model": response.model,
            "finish_reason": getattr(choice, "finish_reason", None),
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        }


class OpenAICompatibleJsonTransport:
    """One-model transport; it never falls back to another endpoint or model."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_completion_tokens: int = 768,
        completion_token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens",
        json_response_format: bool = True,
        reasoning_effort: str | None = None,
        enable_thinking: bool | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("API key cannot be blank")
        if not model.strip():
            raise ValueError("model cannot be blank")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install the OpenAI Python client for external calibration") from exc
        # The corpus builders already own bounded, root-local retries.  Disable
        # the SDK's hidden exponential retries so one transient provider error
        # cannot hold a shard for several full timeout windows before our
        # explicit retry/fallback logic gets control again.
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.completion_token_parameter = completion_token_parameter
        self.json_response_format = json_response_format
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: Mapping[str, object],
        retry_feedback: str | None = None,
    ) -> JsonCompletion:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": canonical_json(user_payload)},
        ]
        if retry_feedback is not None:
            messages.append({"role": "user", "content": retry_feedback})
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.json_response_format:
            request["response_format"] = {"type": "json_object"}
        request[self.completion_token_parameter] = self.max_completion_tokens
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        if self.enable_thinking is not None:
            request["extra_body"] = {"enable_thinking": self.enable_thinking}
        response = self._client.chat.completions.create(**request)
        if len(response.choices) != 1:
            raise JsonResponseError("invalid_choice_count", response=response, requested_model=self.model)
        choice = response.choices[0]
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise JsonResponseError("empty_or_nontext_content", response=response, requested_model=self.model)
        usage = response.usage
        if usage is None:
            raise JsonResponseError("missing_usage", response=response, requested_model=self.model)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
        return JsonCompletion(
            content=content.strip(),
            provider_request_id=response.id,
            requested_model=self.model,
            returned_model=response.model,
            finish_reason=choice.finish_reason,
            usage=JsonTokenUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                reasoning_tokens=reasoning_tokens,
            ),
        )


def build_openai_compatible_json_transport(
    profile: OpenAICompatibleTransportProfile,
    *,
    environ: Mapping[str, str] | None = None,
    transport_factory: Callable[..., JsonCompletionTransport] | None = None,
) -> JsonCompletionTransport:
    """Build one transport from transient environment values only.

    ``profile`` and the returned completion records are safe to serialize.  The
    API key and endpoint are deliberately never returned, logged, or embedded
    in an exception.  ``transport_factory`` exists for offline contract tests;
    it receives the same constructor arguments as
    :class:`OpenAICompatibleJsonTransport` but must not persist credentials.
    """

    environment = os.environ if environ is None else environ
    api_key = _required_environment_value(environment, profile.api_key_env, "API key")
    base_url = _validate_https_base_url(
        _required_environment_value(environment, profile.base_url_env, "base URL"),
        environment_name=profile.base_url_env,
    )
    factory = transport_factory or OpenAICompatibleJsonTransport
    return factory(
        base_url=base_url,
        api_key=api_key,
        model=profile.model,
        temperature=profile.temperature,
        max_completion_tokens=profile.max_completion_tokens,
        completion_token_parameter=profile.completion_token_parameter,
        json_response_format=profile.json_response_format,
        reasoning_effort=profile.reasoning_effort,
        enable_thinking=profile.enable_thinking,
        timeout_seconds=profile.timeout_seconds,
    )


def _required_environment_value(
    environment: Mapping[str, str],
    environment_name: str,
    value_kind: str,
) -> str:
    value = environment.get(environment_name)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"required {value_kind} environment variable is missing or blank: {environment_name}")
    return value.strip()


def _validate_https_base_url(value: str, *, environment_name: str) -> str:
    """Accept one ordinary HTTPS endpoint without exposing it on failure."""

    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise RuntimeError(f"base URL from {environment_name} must be an absolute HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError(f"base URL from {environment_name} cannot embed credentials")
    if parsed.query or parsed.fragment:
        raise RuntimeError(f"base URL from {environment_name} cannot include query or fragment data")
    return value.rstrip("/")
