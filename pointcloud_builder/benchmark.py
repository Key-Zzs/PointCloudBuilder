"""Simple benchmark helpers for offline scripts."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any


def benchmark_callable(
    fn: Callable[[], Any],
    *,
    warmup: int = 5,
    iterations: int = 50,
) -> dict[str, float]:
    """Benchmark a zero-argument callable."""

    for _ in range(warmup):
        fn()
    start = perf_counter()
    for _ in range(iterations):
        fn()
    total = perf_counter() - start
    return {
        "iterations": float(iterations),
        "total_seconds": total,
        "mean_seconds": total / max(iterations, 1),
    }
