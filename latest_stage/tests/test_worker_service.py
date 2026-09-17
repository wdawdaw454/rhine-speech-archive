from __future__ import annotations

import pytest

from src.scheduling.service import (
    ACTIVE,
    CANCELLED,
    CANCELLING,
    FAILED,
    SUCCEEDED,
    QUEUEING,
    StreamWorkerService,
    WorkerHandle,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def make_service(clock: FakeClock, *, max_active: int = 2):
    processes: list[FakeProcess] = []

    def launch(stream_id: str, payload, lease) -> WorkerHandle:
        process = FakeProcess()
        processes.append(process)
        return WorkerHandle(
            process=process,
            command=[stream_id],
            details={"phase_slot_id": lease.slot_id},
        )

    service = StreamWorkerService(
        max_active_streams=max_active,
        phase_period=0.2,
        launcher=launch,
        estimated_service_seconds=60.0,
        clock=clock,
    )
    return service, processes


def test_service_queues_and_reuses_phase_slot_across_requests():
    clock = FakeClock()
    service, processes = make_service(clock)

    for stream_id in ["stream-0", "stream-1", "stream-2"]:
        service.submit(stream_id)

    assert service.status("stream-0")["queue_position"] == 0
    assert service.status("stream-2")["estimated_wait_seconds"] == 90.0

    service.poll_once(now=0.0)
    assert service.status("stream-0")["state"] == ACTIVE
    assert service.status("stream-2")["queue_position"] == 0
    assert service.status("stream-2")["estimated_wait_seconds"] == 30.0
    assert [service.status(f"stream-{index}")["phase_slot_id"] for index in range(2)] == [0, 1]

    processes[0].returncode = 0
    service.poll_once(now=30.0)
    assert service.status("stream-0")["state"] == SUCCEEDED
    assert service.status("stream-0")["service_seconds"] == 30.0

    service.poll_once(now=30.0)
    assert service.status("stream-2")["state"] == ACTIVE
    assert service.status("stream-2")["phase_slot_id"] == 0
    assert service.status("stream-2")["queue_seconds"] == 30.0

    processes[1].returncode = 0
    processes[2].returncode = 0
    service.poll_once(now=60.0)

    snapshot = service.snapshot()
    assert snapshot["active_count"] == 0
    assert snapshot["succeeded_count"] == 3
    assert snapshot["failed_count"] == 0
    assert snapshot["estimated_service_seconds"] == pytest.approx(45.0)

    resubmitted = service.submit("stream-3")
    assert resubmitted["state"] == QUEUEING
    assert resubmitted["estimated_wait_seconds"] == 0.0
    service.poll_once(now=60.0)
    assert service.status("stream-3")["state"] == ACTIVE
    processes[3].returncode = 0
    service.poll_once(now=61.0)
    assert service.status("stream-3")["state"] == SUCCEEDED
    assert service.snapshot()["request_count"] == 4


def test_launch_failure_releases_capacity_and_reports_error():
    clock = FakeClock()

    def launch(stream_id: str, payload, lease) -> WorkerHandle:
        raise RuntimeError("model unavailable")

    service = StreamWorkerService(
        max_active_streams=1,
        phase_period=0.2,
        launcher=launch,
        clock=clock,
    )
    service.submit("stream-0")
    service.poll_once(now=0.0)

    status = service.status("stream-0")
    assert status["state"] == FAILED
    assert "model unavailable" in status["error"]
    assert service.admission.available_count == 1

    with pytest.raises(ValueError, match="already been submitted"):
        service.submit("stream-0")


def test_cancel_queued_and_active_requests():
    clock = FakeClock()
    service, processes = make_service(clock, max_active=2)
    for stream_id in ["stream-0", "stream-1", "stream-2"]:
        service.submit(stream_id)
    service.poll_once(now=0.0)

    queued_cancel = service.cancel("stream-2")
    active_cancel = service.cancel("stream-1")

    assert queued_cancel["state"] == CANCELLED
    assert active_cancel["state"] == CANCELLING
    assert processes[1].returncode == -15
    assert service.queued_stream_ids == []

    service.poll_once(now=1.0)
    assert service.status("stream-1")["state"] == CANCELLED
    assert service.status("stream-1")["service_seconds"] == 1.0
    assert service.admission.available_count == 1
    assert service.snapshot()["cancelled_count"] == 2
    assert service.cancel("stream-1")["state"] == CANCELLED

    processes[0].returncode = 0
    service.poll_once(now=2.0)
    service.poll_once(now=2.0)
    # Only the successful 2-second stream contributes; the cancelled stream is excluded.
    assert service.snapshot()["estimated_service_seconds"] == 31.0


def test_timeout_terminates_active_and_cancels_queued():

    class IncrementingClock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            current = self.now
            self.now += 0.25
            return current

    service, processes = make_service(IncrementingClock(), max_active=2)
    for stream_id in ["stream-0", "stream-1", "stream-2"]:
        service.submit(stream_id)

    completed = service.run_until_idle(timeout=2.0, poll_interval=0.0)

    assert completed is False
    assert service.status("stream-0")["state"] == FAILED
    assert service.status("stream-1")["state"] == FAILED
    assert service.status("stream-2")["state"] == CANCELLED
    assert [process.returncode for process in processes] == [-15, -15]
    assert service.queued_stream_ids == []
