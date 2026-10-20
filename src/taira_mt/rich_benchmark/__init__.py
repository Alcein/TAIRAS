"""Paper-first rich-intent benchmark for TAIRAS.

This package deliberately keeps the benchmark object small and inspectable:
scenario needs are converted to attribute cards, candidates are retrieved from
the full local Amazon catalog, and a deterministic dialogue contract exposes
why state-conditioned Pattern planning matters.
"""

from taira_mt.rich_benchmark.core import (
    RichBenchmarkConfig,
    build_rich_benchmark,
    inspect_rich_benchmark,
    write_rich_benchmark,
)

__all__ = [
    "RichBenchmarkConfig",
    "build_rich_benchmark",
    "inspect_rich_benchmark",
    "write_rich_benchmark",
]
