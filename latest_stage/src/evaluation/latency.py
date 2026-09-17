"""Latency percentile and RTF helpers."""
from __future__ import annotations

import numpy as np


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "p99": percentile(values, 99.0),
        "max": float(arr.max()),
        "count": int(arr.size),
    }


def rtf(processing_wall: float, audio_duration: float) -> float:
    if audio_duration <= 0.0:
        return 0.0
    return float(processing_wall / audio_duration)