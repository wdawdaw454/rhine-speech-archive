"""Pipeline behavior tests with deterministic fake backends."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.models.backends import SpeechSegment
from src.pipeline.events import EventType
from src.pipeline.event_io import EventSink
from src.pipeline.realtime_pipeline import RealtimePipeline


class FakeVad:
    def detect(self, waveform: np.ndarray, sample_rate: int):
        return [SpeechSegment(start=0.1, end=min(2.1, len(waveform) / sample_rate))]


class FakeSpeaker:
    def __init__(self, similarity: float):
        self.similarity = similarity

    def extract(self, waveform: np.ndarray, sample_rate: int):
        value = np.sqrt(max(0.0, self.similarity))
        return np.asarray([value, value, value, value], dtype=np.float32)


class FakeAsr:
    def __init__(self):
        self.last_text = ""

    def reset(self):
        self.last_text = ""

    def accept(self, chunk, *, is_final: bool):
        self.last_text = "你好世界"
        return self.last_text


@dataclass
class Collector(EventSink):
    events: list = field(default_factory=list)

    def write(self, event):
        self.events.append(event)


def _pipeline(similarity: float, collector: Collector) -> RealtimePipeline:
    enrolled = np.asarray([1, 1, 1, 1], dtype=np.float32)
    enrolled /= np.linalg.norm(enrolled)
    return RealtimePipeline(
        run_id="test",
        audio_path="test.wav",
        vad=FakeVad(),
        speaker_encoder=FakeSpeaker(similarity),
        asr=FakeAsr(),
        enrolled_embedding=enrolled,
        accept_threshold=0.90,
        reject_threshold=0.50,
        event_sink=collector,
        sample_rate=16000,
        asr_chunk_size=0.6,
        embedding_window=1.5,
    )


def test_target_speech_is_committed():
    collector = Collector()
    stats = _pipeline(1.0, collector).process(np.ones(32000, dtype=np.float32) * 0.1)
    assert stats.target_committed == 1
    assert stats.non_target_suppressed == 0
    assert any(e.type == EventType.ASR_FINAL for e in collector.events)
    assert any(e.type == EventType.CAPTION_COMMIT for e in collector.events)
    assert stats.backend_timing["vad"]["count"] == 1
    assert stats.backend_timing["asr"]["count"] == 4
    assert stats.backend_timing["speaker_embedding"]["count"] >= 2
    summary = next(e for e in collector.events if e.type == EventType.RUN_SUMMARY)
    assert summary.extras["backend_timing"] == stats.backend_timing


def test_non_target_speech_is_suppressed():
    collector = Collector()
    stats = _pipeline(0.0, collector).process(np.ones(32000, dtype=np.float32) * 0.1)
    assert stats.target_committed == 0
    assert stats.non_target_suppressed == 1
    assert any(e.type == EventType.CAPTION_SUPPRESS for e in collector.events)
