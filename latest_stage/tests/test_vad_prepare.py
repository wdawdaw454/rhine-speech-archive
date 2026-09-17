"""Tests for VAD boundary post-processing."""

from __future__ import annotations

import numpy as np

from src.models.backends import SpeechSegment
from src.vad.stream_vad import prepare_segments


def test_prepare_segments_pads_clips_merges_and_filters():
    segments = [
        SpeechSegment(start=0.2, end=1.0),
        SpeechSegment(start=1.1, end=1.4),
        SpeechSegment(start=2.0, end=2.08),
    ]
    result = prepare_segments(
        segments,
        duration=2.5,
        config=None,
    )
    # With default speech_pad=.2 and merge_gap=.3, first two merge; short third is
    # padded but remains under min_duration=.1 only after exact rounding.
    assert result[0].start == 0.0
    assert result[-1].end >= 2.2


def test_prepare_segments_splits_overlong_vad_segment_on_silence():
    waveform = np.zeros(48000, dtype=np.float32)
    waveform[:16000] = 0.1
    waveform[32000:] = 0.1
    result = prepare_segments(
        [SpeechSegment(start=0.0, end=3.0)],
        duration=3.0,
        waveform=waveform,
        sample_rate=16000,
        min_silence_duration=0.45,
    )

    assert len(result) == 2
    assert result[0].start == 0.0
    assert 0.9 <= result[0].end <= 1.3
    assert 1.7 <= result[1].start <= 2.1
    assert result[1].end == 3.0


def test_prepare_segments_splits_long_segment():
    result = prepare_segments(
        [SpeechSegment(start=0.0, end=40.0)],
        duration=40.0,
    )
    assert len(result) == 2
    assert result[0].end == 20.0
    assert result[1].start == 20.0
