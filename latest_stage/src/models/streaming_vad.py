"""Chunk-driven FunASR FSMN VAD adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VadBoundary:
    start: float | None
    end: float | None


class FunasrStreamingVad:
    """Expose FunASR's streaming VAD as a mutable chunk consumer.

    FunASR reports unknown boundaries as ``-1``. Timestamps are normally in
    milliseconds. ``reset(offset_seconds)`` is used after an energy-based early
    closure: the model state restarts, while pipeline timestamps stay absolute.
    """

    def __init__(self, model_dir: str | Path, device: str = "cpu", *,
                 end_silence_ms: int | None = None,
                 max_segment_ms: int | None = None) -> None:
        from funasr import AutoModel

        options = {}
        if end_silence_ms is not None:
            options["max_end_silence_time"] = int(end_silence_ms)
        if max_segment_ms is not None:
            options["max_single_segment_time"] = int(max_segment_ms)
        self.model = AutoModel(
            model=str(model_dir),
            device=device,
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
            **options,
        )
        self.sample_rate = 16000
        self.reset()

    def reset(self, offset_seconds: float = 0.0) -> None:
        self.cache: dict[str, Any] = {}
        self.offset_seconds = max(0.0, float(offset_seconds))
        self.relative_duration = 0.0

    @staticmethod
    def _bounds(item: Any) -> tuple[float, float]:
        if isinstance(item, dict):
            return float(item["start"]), float(item["end"])
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            return float(item[0]), float(item[1])
        raise TypeError(f"unsupported streaming VAD boundary: {item!r}")

    def accept(self, chunk: np.ndarray, *, is_final: bool) -> list[VadBoundary]:
        wav = np.asarray(chunk, dtype=np.float32)
        if wav.ndim != 1:
            raise ValueError("streaming VAD expects a mono chunk")

        chunk_ms = max(1, int(round(len(wav) * 1000.0 / self.sample_rate)))
        results = self.model.generate(
            input=wav,
            cache=self.cache,
            is_final=is_final,
            chunk_size=chunk_ms,
            fs=self.sample_rate,
            data_type="sound",
        )
        result = results[0] if isinstance(results, list) and results else results
        raw_boundaries = result.get("value", []) if isinstance(result, dict) else []

        converted: list[tuple[float, float]] = []
        max_bound = 0.0
        for item in raw_boundaries:
            start, end = self._bounds(item)
            max_bound = max(max_bound, start, end)
            converted.append((start, end))

        self.relative_duration += len(wav) / self.sample_rate
        divide_by = 1000.0
        if (
            self.relative_duration > 0
            and max_bound / 1000.0 > self.relative_duration * 1.5
            and max_bound <= len(wav) * 1.5
        ):
            divide_by = 1.0

        return [
            VadBoundary(
                start=None if start < 0 else self.offset_seconds + start / divide_by,
                end=None if end < 0 else self.offset_seconds + end / divide_by,
            )
            for start, end in converted
        ]
