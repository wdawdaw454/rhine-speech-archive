import numpy as np
import pytest

from src.audio.stream import FileStream


def _sine(sr: int, duration: float, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_filestream_yields_chunks_with_monotonic_audio_time():
    sr = 16000
    wav = _sine(sr, 1.0)
    stream = FileStream(wav, sr, chunk_size=0.20)
    times = [t for t, _ in stream]
    assert len(times) == 5
    assert np.allclose(np.diff(times), 0.20)
    assert abs(times[-1] - 1.0) < 1e-6
    assert abs(stream.total_duration() - 1.0) < 1e-6


def test_filestream_handles_short_file():
    sr = 16000
    wav = _sine(sr, 0.10)
    stream = FileStream(wav, sr, chunk_size=0.20)
    chunks = list(stream)
    assert len(chunks) == 1
    audio_time, chunk = chunks[0]
    assert audio_time == 0.0
    assert len(chunk) == int(sr * 0.10)


def test_filestream_chunk_lengths_consistent():
    sr = 16000
    wav = _sine(sr, 2.0)
    stream = FileStream(wav, sr, chunk_size=0.25)
    chunks = list(stream)
    for _, chunk in chunks[:-1]:
        assert len(chunk) == int(sr * 0.25)


def test_filestream_rejects_invalid_phase():
    sr = 16000
    wav = _sine(sr, 0.1)
    with pytest.raises(ValueError, match="phase_offset"):
        FileStream(wav, sr, phase_period=0.2, phase_offset=0.2)


def test_filestream_aligns_first_chunk_to_phase(monkeypatch):
    sr = 16000
    wav = _sine(sr, 0.1)
    current = [100.0]
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return current[0]

    def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        current[0] += delay

    monkeypatch.setattr("src.audio.stream.time.monotonic", fake_monotonic)
    monkeypatch.setattr("src.audio.stream.time.sleep", fake_sleep)
    stream = FileStream(
        wav,
        sr,
        chunk_size=0.2,
        realtime_sleep=True,
        phase_period=0.2,
        phase_offset=0.15,
    )
    iterator = iter(stream)
    next(iterator)

    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(0.15)
