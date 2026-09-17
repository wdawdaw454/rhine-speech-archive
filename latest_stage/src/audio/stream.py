"""FileStream: chunked iterator that simulates a realtime audio source."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class FileStream:
    """Yield (audio_time, chunk) tuples for a fixed waveform."""

    waveform: np.ndarray
    sample_rate: int
    chunk_size: float = 0.20
    realtime_sleep: bool = False
    phase_period: float | None = None
    phase_offset: float = 0.0

    def __post_init__(self) -> None:
        self._waveform = np.asarray(self.waveform, dtype=np.float32)
        if self.phase_period is not None and self.phase_period <= 0:
            raise ValueError("phase_period must be positive")
        if self.phase_period is not None and not 0 <= self.phase_offset < self.phase_period:
            raise ValueError("phase_offset must be in [0, phase_period)")
        if self._waveform.ndim != 1:
            raise ValueError("FileStream expects mono waveform")
        self._samples_per_chunk = max(1, int(round(self.chunk_size * self.sample_rate)))

    def total_duration(self) -> float:
        return len(self._waveform) / self.sample_rate

    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        idx = 0
        n = len(self._waveform)
        spc = self._samples_per_chunk
        consumed = 0
        first = True
        next_yield_at: float | None = None
        while consumed < n or idx == 0:
            chunk = self._waveform[idx : idx + spc]
            if chunk.size == 0:
                break
            # audio_time = end-of-chunk wall time. The very first yield for a
            # short file (< one full chunk) is reported as 0.0 so consumers can
            # treat it as "stream start".
            if first and chunk.size < spc:
                audio_time = 0.0
            else:
                audio_time = (consumed + chunk.size) / self.sample_rate
            first = False
            if self.realtime_sleep:
                now = time.monotonic()
                if next_yield_at is None:
                    if self.phase_period is None:
                        next_yield_at = now + self.chunk_size
                    else:
                        phase = self.phase_offset % self.phase_period
                        wait = (phase - now) % self.phase_period
                        next_yield_at = now + wait
                delay = next_yield_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            yield audio_time, chunk.astype(np.float32, copy=False)
            idx += spc
            consumed += chunk.size
            if next_yield_at is not None:
                next_yield_at += self.chunk_size
