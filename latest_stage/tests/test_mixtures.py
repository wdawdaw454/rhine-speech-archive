from __future__ import annotations

import numpy as np

from src.data.mixtures import (
    MixtureSource,
    render_mixture,
    schedule_alternating,
)


def _source(source_id: str, speaker: str, duration: float) -> MixtureSource:
    return MixtureSource(
        source_id=source_id,
        speaker=speaker,
        text=f"text-{source_id}",
        duration=duration,
    )


def test_alternating_schedule_applies_gap_and_overlap():
    placements = schedule_alternating(
        target_sources=[_source("t1", "target", 2.0)],
        non_target_sources=[_source("n1", "other", 1.0)],
        num_utterances=4,
        gap=0.5,
        overlap=0.2,
    )

    assert [(p.role, p.start, p.end) for p in placements] == [
        ("target", 0.0, 2.0),
        ("non_target", 2.3, 3.3),
        ("target", 3.6, 5.6),
        ("non_target", 5.9, 6.9),
    ]


def test_render_mixture_preserves_length_and_limits_peak():
    target = np.tile(np.sin(np.linspace(0.0, 2 * np.pi, 160)), 10).astype(np.float32)
    interferer = np.tile(np.sin(np.linspace(0.0, 4 * np.pi, 160)), 5).astype(np.float32)
    placements = schedule_alternating(
        target_sources=[_source("target", "target", 0.1)],
        non_target_sources=[_source("other", "other", 0.05)],
        num_utterances=2,
        gap=0.0,
        overlap=0.0,
    )
    mixture, normalization = render_mixture(
        placements,
        {"target": target, "other": interferer},
        sample_rate=16000,
        target_snr_db=8.0,
        tail_duration=0.0,
    )

    assert 0.0 < normalization < 1.0
    assert len(mixture) == 2400
    assert float(np.max(np.abs(mixture))) <= 0.99 + 1e-6
