"""Resident lifecycle management for realtime stream workers."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import subprocess
from typing import Any, Callable

from .admission import PhaseAdmissionController, PhaseLease

QUEUEING = "queueing"
ACTIVE = "active"
CANCELLING = "cancelling"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"


@dataclass(frozen=True)
class WorkerHandle:
    """Process-facing details returned by the embedding application."""

    process: subprocess.Popen[Any]
    command: list[str]
    details: dict[str, Any] = field(default_factory=dict)


Launcher = Callable[[str, Any, PhaseLease], WorkerHandle]


@dataclass
class WorkerRecord:
    stream_id: str
    payload: Any
    state: str = QUEUEING
    submitted_at: float = 0.0
    launched_at: float | None = None
    ended_at: float | None = None
    lease: PhaseLease | None = None
    handle: WorkerHandle | None = None
    returncode: int | None = None
    error: str | None = None
    cancellation_requested: bool = False


class StreamWorkerService:
    """Keep stream admission state alive across separate submit batches.

    The service intentionally does not own model objects. An embedding application
    supplies a launcher callback, which keeps the scheduler independent of the ASR
    backend and makes the same service usable by a CLI runner or a network server.
    """

    def __init__(
        self,
        *,
        max_active_streams: int,
        phase_period: float,
        launcher: Launcher,
        estimated_service_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        poll_process: Callable[[WorkerHandle], int | None] | None = None,
    ) -> None:
        if estimated_service_seconds <= 0:
            raise ValueError("estimated_service_seconds must be positive")

        self.admission = PhaseAdmissionController(
            max_active_streams=max_active_streams,
            phase_period=phase_period,
        )
        self.launcher = launcher
        self.initial_service_estimate = float(estimated_service_seconds)
        self._clock = clock
        self._poll_process = poll_process or self._default_poll_process
        self._records: dict[str, WorkerRecord] = {}
        self._queue: list[str] = []
        self._active: dict[str, WorkerRecord] = {}
        self._completed_service_seconds: list[float] = []

    @property
    def clock(self) -> Callable[[], float]:
        return self._clock

    @staticmethod
    def _default_poll_process(handle: WorkerHandle) -> int | None:
        return handle.process.poll()

    @property
    def max_active_streams(self) -> int:
        return self.admission.max_active_streams

    @property
    def phase_period(self) -> float:
        return self.admission.phase_period

    @property
    def queued_stream_ids(self) -> list[str]:
        return list(self._queue)

    @property
    def records(self) -> tuple[WorkerRecord, ...]:
        return tuple(self._records.values())

    def submit(self, stream_id: str, payload: Any = None) -> dict[str, Any]:
        if not stream_id:
            raise ValueError("stream_id must be non-empty")
        if stream_id in self._records:
            raise ValueError(f"stream_id has already been submitted: {stream_id}")

        now = self._clock()
        self._records[stream_id] = WorkerRecord(
            stream_id=stream_id,
            payload=payload,
            submitted_at=now,
        )
        self._queue.append(stream_id)
        return self.status(stream_id)

    def cancel(self, stream_id: str) -> dict[str, Any]:
        record = self._records.get(stream_id)
        if record is None:
            raise KeyError(f"unknown stream_id: {stream_id}")

        now = self._clock()
        if record.state in {SUCCEEDED, FAILED, CANCELLED}:
            return self.status(stream_id)

        if record.state == CANCELLING:
            return self.status(stream_id)

        if stream_id in self._queue:
            self._queue.remove(stream_id)
            record.state = CANCELLED
            record.ended_at = now
            return self.status(stream_id)

        record.cancellation_requested = True
        record.state = CANCELLING
        if record.handle is not None:
            try:
                record.handle.process.terminate()
            except OSError:
                pass
        return self.status(stream_id)

    def status(self, stream_id: str) -> dict[str, Any]:
        record = self._records.get(stream_id)
        if record is None:
            raise KeyError(f"unknown stream_id: {stream_id}")

        now = self._clock()
        position = self._queue.index(stream_id) if stream_id in self._queue else None
        response: dict[str, Any] = {
            "stream_id": stream_id,
            "state": record.state,
            "submitted_age_seconds": max(0.0, now - record.submitted_at),
            "queue_position": position,
            "estimated_wait_seconds": self._estimate_wait(position),
            "active_count": self.admission.active_count,
            "max_active_streams": self.max_active_streams,
        }
        if record.lease is not None:
            response["phase_slot_id"] = record.lease.slot_id
            response["phase_offset"] = record.lease.phase_offset
        if record.launched_at is not None:
            response["queue_seconds"] = max(0.0, record.launched_at - record.submitted_at)
        if record.ended_at is not None and record.launched_at is not None:
            response["service_seconds"] = max(0.0, record.ended_at - record.launched_at)
        if record.returncode is not None:
            response["returncode"] = record.returncode
        if record.error is not None:
            response["error"] = record.error
        return response

    def _estimate_wait(self, position: int | None) -> float | None:
        if position is None:
            return 0.0
        if self.admission.active_count < self.max_active_streams and position == 0:
            return 0.0
        parallelism = self.max_active_streams
        return (position + 1) * self._estimated_service_seconds / parallelism

    @property
    def _estimated_service_seconds(self) -> float:
        if not self._completed_service_seconds:
            return self.initial_service_estimate
        observations = [self.initial_service_estimate, *self._completed_service_seconds]
        return sum(observations) / len(observations)

    def poll_once(self, *, now: float | None = None, launch_pending: bool = True) -> None:
        current = self._clock() if now is None else now

        while launch_pending and self._queue and self.admission.available_count:
            stream_id = self._queue.pop(0)
            record = self._records[stream_id]
            lease = self.admission.reserve(stream_id)
            if lease is None:
                self._queue.insert(0, stream_id)
                break
            record.lease = lease
            try:
                handle = self.launcher(stream_id, record.payload, lease)
            except Exception as exc:
                self.admission.release(stream_id)
                record.lease = None
                record.state = FAILED
                record.ended_at = current
                record.error = f"worker launch failed: {exc}"
                continue

            record.handle = handle
            record.state = ACTIVE
            record.launched_at = current
            self._active[stream_id] = record

        for stream_id, record in list(self._active.items()):
            returncode = self._poll_process(record.handle)
            if returncode is None:
                continue
            self._finish(stream_id, record, returncode, current)

    def _finish(
        self,
        stream_id: str,
        record: WorkerRecord,
        returncode: int,
        now: float,
    ) -> None:
        record.returncode = returncode
        record.state = (
            CANCELLED if record.cancellation_requested else SUCCEEDED if returncode == 0 else FAILED
        )
        record.ended_at = now
        if record.launched_at is not None:
            if not record.cancellation_requested:
                self._completed_service_seconds.append(max(0.0, now - record.launched_at))
        self.admission.release(stream_id)
        self._active.pop(stream_id, None)

    def run_until_idle(
        self,
        *,
        timeout: float,
        poll_interval: float = 0.1,
        termination_grace_seconds: float = 10.0,
    ) -> bool:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if poll_interval < 0:
            raise ValueError("poll_interval must be non-negative")
        if termination_grace_seconds < 0:
            raise ValueError("termination_grace_seconds must be non-negative")

        started_at = self._clock()
        while self._queue or self._active:
            self.poll_once()
            if not self._queue and not self._active:
                return True
            if self._clock() - started_at >= timeout:
                self.shutdown(timeout=termination_grace_seconds)
                return False
            if poll_interval:
                time.sleep(poll_interval)
        return True

    def shutdown(self, *, timeout: float = 10.0) -> None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")

        for record in self._active.values():
            if record.handle is not None:
                try:
                    record.handle.process.terminate()
                except OSError:
                    pass

        deadline = time.monotonic() + timeout
        while self._active and time.monotonic() < deadline:
            self.poll_once(launch_pending=False)
            if self._active:
                time.sleep(0.05)

        for record in list(self._active.values()):
            if record.handle is not None:
                try:
                    record.handle.process.kill()
                except OSError:
                    pass
        self.poll_once(now=self._clock(), launch_pending=False)

        for stream_id, record in list(self._active.items()):
            if record.returncode is None:
                self._finish(stream_id, record, -9, self._clock())

        now = self._clock()
        for stream_id in self._queue:
            record = self._records[stream_id]
            record.state = CANCELLED
            record.ended_at = now
        self._queue.clear()

    def snapshot(self) -> dict[str, Any]:
        states = [record.state for record in self._records.values()]
        return {
            **self.admission.snapshot(),
            "phase_period_seconds": self.phase_period,
            "queued_count": len(self._queue),
            "queued_stream_ids": self.queued_stream_ids,
            "estimated_service_seconds": self._estimated_service_seconds,
            "request_count": len(states),
            "succeeded_count": states.count(SUCCEEDED),
            "failed_count": states.count(FAILED),
            "cancelled_count": states.count(CANCELLED),
            "cancelling_count": states.count(CANCELLING),
            "requests": [self.status(stream_id) for stream_id in self._records],
        }
