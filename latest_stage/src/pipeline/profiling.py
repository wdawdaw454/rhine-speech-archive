"""Timing helpers for backend calls in realtime pipelines."""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Iterator

from ..evaluation.latency import latency_summary

VAD = "vad"
SPEAKER_EMBEDDING = "speaker_embedding"
ASR = "asr"
PUNCTUATION = "punctuation"


class BackendProfiler:
    """Record per-call wall time for synchronous model backends.

    These measurements include Python dispatch, model runtime, and any implicit GPU
    queueing observed by the calling stream. They do not claim to isolate pure GPU
    kernel time.
    """

    def __init__(self) -> None:
        self._durations: dict[str, list[float]] = {}
        self._errors: dict[str, int] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started_at = time.perf_counter()
        try:
            yield
        except Exception:
            self._errors[stage] = self._errors.get(stage, 0) + 1
            raise
        else:
            self._durations.setdefault(stage, []).append(time.perf_counter() - started_at)

    def summary(self) -> dict[str, dict[str, float | int]]:
        stages = set(self._durations) | set(self._errors)
        return {
            stage: {
                **latency_summary(self._durations.get(stage, [])),
                "error_count": self._errors.get(stage, 0),
            }
            for stage in sorted(stages)
        }

    def durations(self) -> dict[str, list[float]]:
        return {stage: list(values) for stage, values in self._durations.items()}
