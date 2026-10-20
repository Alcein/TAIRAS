"""Explicit adapters for optional external model calibration."""

from taira_mt.providers.openai_compatible import (
    JsonCompletion,
    JsonCompletionTransport,
    JsonTokenUsage,
    OpenAICompatibleJsonTransport,
    OpenAICompatibleTransportProfile,
    build_openai_compatible_json_transport,
)

__all__ = [
    "JsonCompletion",
    "JsonCompletionTransport",
    "JsonTokenUsage",
    "OpenAICompatibleJsonTransport",
    "OpenAICompatibleTransportProfile",
    "build_openai_compatible_json_transport",
]
