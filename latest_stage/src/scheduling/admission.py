"""Admission control and deterministic phase assignment for realtime streams."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseLease:
    stream_id: str
    phase_offset: float
    slot_id: int


class PhaseAdmissionController:
    """Limit active streams and assign each one a deterministic chunk phase.

    A lease is both a capacity reservation and a phase reservation. Releasing a
    lease makes its phase slot available to the next queued stream.
    """

    def __init__(self, *, max_active_streams: int, phase_period: float) -> None:
        if max_active_streams < 1:
            raise ValueError("max_active_streams must be positive")
        if phase_period <= 0:
            raise ValueError("phase_period must be positive")
        self.max_active_streams = int(max_active_streams)
        self.phase_period = float(phase_period)
        self._slots = [
            PhaseLease(
                stream_id="",
                phase_offset=(slot * phase_period) / self.max_active_streams,
                slot_id=slot,
            )
            for slot in range(self.max_active_streams)
        ]
        self._active: dict[str, PhaseLease] = {}
        self.peak_active_streams = 0

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def available_count(self) -> int:
        return len(self._slots)

    def reserve(self, stream_id: str) -> PhaseLease | None:
        if not stream_id:
            raise ValueError("stream_id must be non-empty")
        if stream_id in self._active:
            raise ValueError(f"stream is already active: {stream_id}")
        if not self._slots:
            return None
        slot = self._slots.pop(0)
        lease = PhaseLease(
            stream_id=stream_id,
            phase_offset=slot.phase_offset,
            slot_id=slot.slot_id,
        )
        self._active[stream_id] = lease
        self.peak_active_streams = max(self.peak_active_streams, self.active_count)
        return lease

    def release(self, stream_id: str) -> PhaseLease:
        lease = self._active.pop(stream_id, None)
        if lease is None:
            raise ValueError(f"stream is not active: {stream_id}")
        self._slots.append(
            PhaseLease(
                stream_id="",
                phase_offset=lease.phase_offset,
                slot_id=lease.slot_id,
            )
        )
        self._slots.sort(key=lambda item: item.slot_id)
        return lease

    def snapshot(self) -> dict[str, int | list[float]]:
        return {
            "active_count": self.active_count,
            "available_count": self.available_count,
            "peak_active_streams": self.peak_active_streams,
            "active_stream_ids": sorted(self._active),
            "available_phase_offsets": sorted(item.phase_offset for item in self._slots),
        }
