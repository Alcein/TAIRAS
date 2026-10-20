"""Small, shared lexical predicates over public dialogue text.

These helpers intentionally operate on only the visible user utterance.  They
are not a user-state parser: the bounded rejection predicate is shared by the
deterministic interpreter, Direct baseline, and Pattern safety envelope solely
to keep their structural no-repeat guards comparable on the frozen public
language bank.
"""

from __future__ import annotations

import re

# These phrases are deliberately finite and are covered verbatim by
# ``tests/test_rule_based_frozen_language_coverage.py`` against the frozen D0
# public language bank.  Keep this list public-text-only: no simulator frame,
# evaluator label, or latent preference is consulted here.
PUBLIC_DIRECT_REJECTION_MARKERS: tuple[str, ...] = (
    "does not fit",
    "do not fit",
    "don't fit",
    "doesn't fit",
    "not suitable",
    "aren't suitable",
    "unsuitable",
    "not appropriate",
    "aren't right",
    "not right for me",
    "aren't quite what i'm looking for",
    "don't match",
    "doesn't match",
    "do not match",
    "don't work",
    "doesn't work",
    "do not work",
    "don't think those",
    "none of those",
    "none of these",
    "aren't the right",
    "can't use",
    "cannot use",
    "don't need those",
    "fail to align",
    "fail to match",
    "misses my needs",
    "reject",
)

_PUBLIC_UNVERIFIABLE_REJECTION = re.compile(
    r"\b(?:cannot|can['’]t|unable\s+to|impossible\s+to|no\s+way\s+to|"
    r"not\s+possible(?:\s+for\s+me)?\s+to)\s+"
    r"(?:verify|confirm|determine|check|ascertain|validate)\b",
    flags=re.IGNORECASE,
)
_PUBLIC_UNVERIFIABLE_LITERAL_MARKERS: tuple[str, ...] = (
    "does not allow me to verify",
)


def has_direct_public_rejection_marker(utterance: str) -> bool:
    """Return whether text explicitly rejects a recommendation by wording.

    This excludes unresolved-verifiability language because that wording only
    means rejection in the context of a recommendation or explanation.  The
    caller is responsible for checking that public action context.
    """

    folded = utterance.casefold()
    return any(marker in folded for marker in PUBLIC_DIRECT_REJECTION_MARKERS)


def has_public_unverifiable_rejection(utterance: str) -> bool:
    """Return whether public text says the prior recommendation is unverifiable."""

    folded = utterance.casefold()
    return (
        _PUBLIC_UNVERIFIABLE_REJECTION.search(utterance) is not None
        or any(marker in folded for marker in _PUBLIC_UNVERIFIABLE_LITERAL_MARKERS)
    )


def is_explicit_public_rejection(utterance: str) -> bool:
    """Recognize the public rejection surface after a recommendation.

    This is the shared no-repeat predicate.  It is intentionally lexical and
    should only be invoked where the immediately preceding public action makes
    a rejection interpretation meaningful.
    """

    return has_direct_public_rejection_marker(utterance) or has_public_unverifiable_rejection(utterance)
