from __future__ import annotations

import numpy as np
import pytest

from scripts.run_fast_dictation import _device, build_parser
from src.audio.microphone import MicrophoneStream
from src.pipeline.events import EventType
from src.pipeline.fast_dictation import FastDictationPipeline


class EventCollector:
    def __init__(self) -> None:
        self.events = []

    def write(self, event) -> None:
        self.events.append(event)

    @property
    def types(self):
        return [event.type for event in self.events]


class FakePrefixBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []
        self.buffer = np.empty(0, dtype=np.float32)
        self.resets = 0

    def reset(self) -> None:
        self.buffer = np.empty(0, dtype=np.float32)
        self.resets += 1

    def accept(self, chunk: np.ndarray, *, is_final: bool) -> str:
        self.buffer = np.concatenate((self.buffer, chunk)).astype(np.float32, copy=False)
        self.calls.append((len(chunk), is_final))
        duration = len(self.buffer) / 16000
        if is_final:
            return "你好"
        if duration >= 1.0:
            return "你好"
        if duration >= 0.5:
            return "你"
        return ""


def _chunk(seconds: float, *, speech: bool) -> np.ndarray:
    value = 0.08 if speech else 0.0
    return np.full(int(round(seconds * 16000)), value, dtype=np.float32)


def _pipeline(backend: FakePrefixBackend, collector: EventCollector, **overrides):
    values = {
        "run_id": "fast-test",
        "backend": backend,
        "event_sink": collector,
        "sample_rate": 16000,
        "input_chunk": 0.10,
        "decode_interval": 0.50,
        "preroll": 0.30,
        "speech_rms_threshold": 0.005,
        "min_speech_duration": 0.20,
        "silence_duration": 0.60,
        "max_segment_duration": 12.0,
    }
    values.update(overrides)
    return FastDictationPipeline(**values)


def test_fast_dictation_commits_after_silence_and_bounds_decode_calls():
    backend = FakePrefixBackend()
    collector = EventCollector()
    pipeline = _pipeline(backend, collector)

    for _ in range(10):
        pipeline.push_chunk(_chunk(0.10, speech=False))
    for _ in range(7):
        pipeline.push_chunk(_chunk(0.10, speech=True))
    for _ in range(6):
        pipeline.push_chunk(_chunk(0.10, speech=False))
    stats = pipeline.finish()

    assert stats.committed_segments == 1
    assert stats.discarded_segments == 0
    assert EventType.VAD_START in collector.types
    assert EventType.ASR_PARTIAL in collector.types
    assert EventType.CAPTION_COMMIT in collector.types
    assert collector.events[0].audio_time == 1.1
    partials = [event for event in collector.events if event.type == EventType.ASR_PARTIAL]
    assert all(event.text in ("你", "你好") for event in partials)
    # Re-decodes occur every 0.5s of segment audio, then one final call.
    assert len(backend.calls) == 4
    assert backend.calls[-1][1] is True


def test_fast_dictation_finishes_active_residual_segment():
    backend = FakePrefixBackend()
    collector = EventCollector()
    pipeline = _pipeline(backend, collector, preroll=0.0)

    for _ in range(3):
        pipeline.push_chunk(_chunk(0.10, speech=True), is_final=False)
    stats = pipeline.finish()

    assert stats.committed_segments == 1
    assert backend.calls[-1] == (4800, True)
    commits = [event for event in collector.events if event.type == EventType.CAPTION_COMMIT]
    assert commits[0].text == "你好"


def test_fast_dictation_discards_short_noise_blip():
    backend = FakePrefixBackend()
    collector = EventCollector()
    pipeline = _pipeline(backend, collector, preroll=0.0)

    pipeline.push_chunk(_chunk(0.10, speech=True))
    for _ in range(6):
        pipeline.push_chunk(_chunk(0.10, speech=False))
    stats = pipeline.finish()

    assert stats.committed_segments == 0
    assert stats.discarded_segments == 1
    assert EventType.CAPTION_COMMIT not in collector.types


def test_fast_dictation_forces_max_segment_length():
    backend = FakePrefixBackend()
    collector = EventCollector()
    pipeline = _pipeline(
        backend,
        collector,
        preroll=0.0,
        min_speech_duration=0.1,
        max_segment_duration=0.4,
    )

    for _ in range(4):
        pipeline.push_chunk(_chunk(0.10, speech=True))
    stats = pipeline.finish()

    assert stats.committed_segments == 1
    assert EventType.CAPTION_COMMIT in collector.types


def test_fast_dictation_ignores_idle_final_stream():
    backend = FakePrefixBackend()
    collector = EventCollector()
    pipeline = _pipeline(backend, collector)

    for _ in range(5):
        pipeline.push_chunk(_chunk(0.10, speech=False))
    stats = pipeline.finish()

    assert stats.committed_segments == 0
    assert stats.discarded_segments == 0
    assert collector.events == []


@pytest.mark.parametrize("values", [{"sample_rate": 0}, {"input_chunk": 0}, {"decode_interval": 0}])
def test_fast_dictation_rejects_invalid_rates_and_chunks(values):
    with pytest.raises(ValueError):
        _pipeline(FakePrefixBackend(), EventCollector(), **values)


def test_microphone_stream_validates_arguments():
    with pytest.raises(ValueError):
        MicrophoneStream(duration=0)


def test_parser_accepts_microphone_and_file_modes():
    parser = build_parser()

    mic = parser.parse_args(["--microphone", "--duration", "10"])
    assert mic.microphone is True
    assert mic.duration == 10

    file = parser.parse_args(["--input", "example.wav", "--realtime"])
    assert file.input is not None
    assert file.realtime is True

    onnx_mode = parser.parse_args(["--input", "example.wav", "--backend", "onnx"])
    assert onnx_mode.backend == "onnx"


def test_microphone_device_parser():
    assert _device("auto") is None
    assert _device("1") == 1
    assert _device("USB Microphone") == "USB Microphone"
