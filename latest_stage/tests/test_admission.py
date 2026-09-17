import pytest
import math

from src.scheduling.admission import PhaseAdmissionController


def test_admission_assigns_even_phase_slots():
    controller = PhaseAdmissionController(max_active_streams=4, phase_period=0.2)

    leases = [controller.reserve(f"stream-{index}") for index in range(4)]

    assert all(
        math.isclose(lease.phase_offset, expected, abs_tol=1e-9)
        for lease, expected in zip(leases, [0.0, 0.05, 0.1, 0.15], strict=True)
    )
    assert [lease.slot_id for lease in leases] == [0, 1, 2, 3]
    assert controller.active_count == 4
    assert controller.available_count == 0
    assert controller.reserve("stream-4") is None
    assert controller.peak_active_streams == 4


def test_release_reuses_deterministic_phase_slot():
    controller = PhaseAdmissionController(max_active_streams=4, phase_period=0.2)
    first = controller.reserve("stream-0")
    assert first is not None
    assert controller.available_count == 3

    released = controller.release("stream-0")
    assert controller.available_count == 4

    assert released.slot_id == first.slot_id
    assert controller.available_count == 4
    reused = controller.reserve("stream-4")
    assert reused is not None
    assert reused.slot_id == first.slot_id
    assert reused.phase_offset == first.phase_offset
    assert controller.snapshot()["active_stream_ids"] == ["stream-4"]


def test_controller_rejects_invalid_state():
    controller = PhaseAdmissionController(max_active_streams=2, phase_period=0.2)

    with pytest.raises(ValueError, match="max_active_streams"):
        PhaseAdmissionController(max_active_streams=0, phase_period=0.2)
    with pytest.raises(ValueError, match="phase_period"):
        PhaseAdmissionController(max_active_streams=1, phase_period=0)
    with pytest.raises(ValueError, match="non-empty"):
        controller.reserve("")
    with pytest.raises(ValueError, match="not active"):
        controller.release("missing")
