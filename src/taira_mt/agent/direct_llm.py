"""Safe real-provider adapter for the equal-tool Direct Agent.

``DirectAgentRunner`` deliberately depends only on the small
``JsonCompletionTransport`` protocol.  This module provides the explicit
environment-only construction path for a live Direct planner without making
the runner itself aware of an API key, base URL, or provider SDK.

It is intentionally an adapter rather than another policy: all canonical JSON
validation, retry, public action guards, and fail-closed STOP behavior remain
in :mod:`taira_mt.agent.direct`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from taira_mt.providers.openai_compatible import (
    JsonCompletion,
    JsonCompletionTransport,
    OpenAICompatibleTransportProfile,
    build_openai_compatible_json_transport,
)

DIRECT_OPENAI_COMPATIBLE_PLANNER_VERSION = "direct_openai_compatible_planner_v1"


class DirectOpenAICompatiblePlanner:
    """A secret-free Direct-planner facade over one configured transport.

    The only persistent state is a protocol-compatible transport.  Its key and
    endpoint are read by :meth:`from_environment` at construction time and are
    never surfaced by ``planner_id``, ``public_telemetry``, or ``repr``.
    """

    def __init__(
        self,
        *,
        transport: JsonCompletionTransport,
        profile: OpenAICompatibleTransportProfile,
    ) -> None:
        self._transport = transport
        self._profile = profile
        profile_digest = str(profile.secret_free_telemetry()["profile_sha256"])
        self.planner_id = (
            f"{DIRECT_OPENAI_COMPATIBLE_PLANNER_VERSION}:"
            f"{profile.provider_profile_id}:{profile_digest[:16]}"
        )

    @classmethod
    def from_environment(
        cls,
        profile: OpenAICompatibleTransportProfile,
        *,
        environ: Mapping[str, str] | None = None,
        transport_factory: Callable[..., JsonCompletionTransport] | None = None,
    ) -> DirectOpenAICompatiblePlanner:
        """Create one planner from transient environment variables.

        ``environ`` and ``transport_factory`` are dependency-injection seams
        for offline tests.  Production code should omit both, which reads the
        profile's named variables from ``os.environ`` and uses the normal
        OpenAI-compatible SDK transport.
        """

        return cls(
            transport=build_openai_compatible_json_transport(
                profile,
                environ=environ,
                transport_factory=transport_factory,
            ),
            profile=profile,
        )

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: Mapping[str, object],
        retry_feedback: str | None = None,
    ) -> JsonCompletion:
        """Delegate a single no-fallback completion to the configured route."""

        return self._transport.complete_json(
            system_prompt=system_prompt,
            user_payload=user_payload,
            retry_feedback=retry_feedback,
        )

    def public_telemetry(self) -> dict[str, object]:
        """Return safe manifest fields for this planner profile."""

        return {
            "planner_adapter_version": DIRECT_OPENAI_COMPATIBLE_PLANNER_VERSION,
            "planner_id": self.planner_id,
            "provider_profile": self._profile.secret_free_telemetry(),
        }

    def __repr__(self) -> str:
        """Avoid default object graphs that could reveal SDK implementation state."""

        return f"DirectOpenAICompatiblePlanner(planner_id={self.planner_id!r})"
