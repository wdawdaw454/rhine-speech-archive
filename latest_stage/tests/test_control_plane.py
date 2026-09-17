from __future__ import annotations

from src.scheduling.control_plane import StreamControlPlane
from src.scheduling.service import (
    ACTIVE,
    CANCELLING,
    CANCELLED,
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


def make_control_plane(clock: FakeClock):
    def launch(stream_id: str, payload, lease) -> WorkerHandle:
        return WorkerHandle(process=FakeProcess(), command=[stream_id])

    service = StreamWorkerService(
        max_active_streams=1,
        phase_period=0.2,
        launcher=launch,
        clock=clock,
    )
    return StreamControlPlane(service, max_service_seconds=10.0)


def test_control_plane_wraps_submit_and_cancel():
    clock = FakeClock()
    control = make_control_plane(clock)
    control.submit("queued")
    control.submit("active")

    control.poll_once()
    assert control.status("queued")["state"] == ACTIVE
    assert control.status("active")["state"] == QUEUEING

    assert control.cancel("queued")["state"] == CANCELLING
    assert control.cancel("active")["state"] == CANCELLED
    control.poll_once()
    assert control.status("active")["state"] == CANCELLED
    # The freed slot is immediately reusable by the next queued request.
    control.submit("next")
    control.poll_once()
    assert control.status("next")["state"] == ACTIVE
    assert control.snapshot()["cancelled_count"] == 2


def test_control_plane_cancels_streams_that_exceed_service_limit():
    clock = FakeClock()
    control = make_control_plane(clock)
    control.submit("stream")
    control.poll_once()

    clock.now = 9.9
    control.poll_once()
    assert control.status("stream")["state"] == ACTIVE

    clock.now = 10.0
    control.poll_once()
    assert control.status("stream")["state"] == CANCELLED
    assert control.snapshot()["cancelled_count"] == 1
