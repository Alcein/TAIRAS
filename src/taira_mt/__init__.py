"""TAIRAS research package."""

__version__ = "0.1.0"
PUBLIC_METHOD_NAME = "TAIRAS"
PUBLIC_METHOD_EXPANSION = (
    "Thought-Augmented Interactive Recommender Agent with decision-sufficient user States"
)
# Compatibility alias for historical registries and saved evaluation commands.
# It must never be emitted as the paper-facing method name.
LEGACY_PUBLIC_METHOD_NAMES = ("TAIRA-MT",)


def normalize_public_method_name(name: str) -> str:
    """Map a legacy display label to TAIRAS without changing machine IDs."""

    for legacy_name in LEGACY_PUBLIC_METHOD_NAMES:
        if name == legacy_name or name.startswith(f"{legacy_name} "):
            return f"{PUBLIC_METHOD_NAME}{name[len(legacy_name):]}"
    return name


__all__ = [
    "LEGACY_PUBLIC_METHOD_NAMES",
    "PUBLIC_METHOD_EXPANSION",
    "PUBLIC_METHOD_NAME",
    "__version__",
    "normalize_public_method_name",
]
