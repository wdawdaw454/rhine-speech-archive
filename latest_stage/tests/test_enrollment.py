"""Tests for speaker enrollment profiles."""

from __future__ import annotations

import numpy as np
import pytest

from src.speaker.enrollment import (
    EnrollmentError,
    build_speaker_profile,
    load_profile,
    profile_embedding,
    save_profile,
)


class ConstantEncoder:
    def __init__(self, value: float):
        self.value = value

    def extract(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        arr = np.asarray([self.value, self.value, self.value, self.value], dtype=np.float32)
        return arr / np.linalg.norm(arr)


def _audio(path, seconds=3.1, amplitude=0.2):
    import soundfile as sf

    sr = 16000
    samples = int(sr * seconds)
    waveform = np.full(samples, amplitude, dtype=np.float32)
    sf.write(path, waveform, sr)


def test_profile_roundtrip(tmp_path):
    audio = tmp_path / "enroll.wav"
    _audio(audio)
    profile = build_speaker_profile(
        "target", [audio], ConstantEncoder(0.2), model_id="fake-speaker"
    )
    output = save_profile(profile, tmp_path / "profile.json")
    loaded = load_profile(output)
    embedding = profile_embedding(loaded)
    assert embedding.shape == (4,)
    assert np.isclose(np.linalg.norm(embedding), 1.0)


def test_short_audio_rejected(tmp_path):
    audio = tmp_path / "short.wav"
    _audio(audio, seconds=1.0)
    with pytest.raises(EnrollmentError):
        build_speaker_profile("target", [audio], ConstantEncoder(0.2), model_id="fake-speaker")
