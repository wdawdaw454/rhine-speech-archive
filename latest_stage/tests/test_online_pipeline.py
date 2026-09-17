from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.models.streaming_vad import VadBoundary
from src.pipeline.event_io import EventSink
from src.pipeline.events import EventType
from src.pipeline.online_pipeline import OnlineRealtimePipeline


class FakeOnlineVad:
    def __init__(self):
        self.offset = 0.0
        self.samples = 0
        self.active = False
        self.reset_count = 0

    def reset(self, offset_seconds: float = 0.0) -> None:
        self.offset = float(offset_seconds)
        self.samples = 0
        self.active = False
        self.reset_count += 1

    def accept(self, chunk: np.ndarray, *, is_final: bool):
        start_sample = self.samples
        self.samples += len(chunk)
        boundaries = []
        if not self.active and float(np.max(np.abs(chunk))) > 0.01:
            boundaries.append(VadBoundary(start=self.offset + start_sample / 16000, end=None))
            self.active = True
        if is_final and self.active:
            boundaries.append(VadBoundary(start=None, end=self.offset + self.samples / 16000))
            self.active = False
        return boundaries


class FakeSpeaker:
    def __init__(self, similarity: float):
        self.similarity = similarity

    def extract(self, waveform: np.ndarray, sample_rate: int):
        value = np.sqrt(max(0.0, self.similarity))
        return np.asarray([value, value, value, value], dtype=np.float32)


class FakeAsr:
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = []

    def accept(self, chunk, *, is_final: bool):
        self.calls.append((len(chunk), is_final))
        return "你好世界"


@dataclass
class Collector(EventSink):
    events: list = field(default_factory=list)

    def write(self, event):
        self.events.append(event)


def _pipeline(
    similarity: float,
    collector: Collector,
    vad: FakeOnlineVad | None = None,
) -> OnlineRealtimePipeline:
    enrolled = np.asarray([1, 1, 1, 1], dtype=np.float32)
    enrolled /= np.linalg.norm(enrolled)
    return OnlineRealtimePipeline(
        run_id="online-test",
        audio_path="test.wav",
        vad=vad or FakeOnlineVad(),
        speaker_encoder=FakeSpeaker(similarity),
        asr=FakeAsr(),
        enrolled_embedding=enrolled,
        accept_threshold=0.90,
        reject_threshold=0.50,
        event_sink=collector,
        sample_rate=16000,
        asr_chunk_size=0.60,
        embedding_window=1.50,
        embedding_hop=0.50,
        declared_duration=2.0,
    )


def _push_two_seconds(pipeline: OnlineRealtimePipeline):
    samples_per_chunk = 3200
    for index in range(10):
        chunk = np.ones(samples_per_chunk, dtype=np.float32) * 0.1
        pipeline.push_chunk(chunk, is_final=index == 9)


def test_online_target_generates_partial_before_final():
    collector = Collector()
    pipeline = _pipeline(1.0, collector)
    samples_per_chunk = 3200
    for index in range(9):
        pipeline.push_chunk(np.ones(samples_per_chunk, dtype=np.float32) * 0.1)

    partials = [e for e in collector.events if e.type == EventType.ASR_PARTIAL]
    assert partials
    assert all(e.monotonic_time >= 0 for e in partials)
    assert partials[0].speaker_decision == "pending"
    assert any(e.speaker_decision == "target" for e in partials)

    pipeline.push_chunk(np.ones(samples_per_chunk, dtype=np.float32) * 0.1, is_final=True)
    stats = pipeline.finish()

    finals = [e for e in collector.events if e.type == EventType.ASR_FINAL]
    commits = [e for e in collector.events if e.type == EventType.CAPTION_COMMIT]
    assert len(finals) == 1
    assert len(commits) == 1
    assert collector.events.index(partials[0]) < collector.events.index(finals[0])
    assert stats.backend_timing["vad"]["count"] == 10
    assert stats.backend_timing["asr"]["count"] >= 3
    assert stats.backend_timing["speaker_embedding"]["count"] >= 2
    assert len(stats.backend_durations["vad"]) == 10


def test_online_non_target_is_suppressed():
    collector = Collector()
    pipeline = _pipeline(0.0, collector)
    _push_two_seconds(pipeline)
    partials = [e for e in collector.events if e.type == EventType.ASR_PARTIAL]
    assert partials
    assert all(e.speaker_decision == "pending" for e in partials)
    stats = pipeline.finish()

    assert stats.vad_segments == 1
    assert stats.target_committed == 0
    assert stats.non_target_suppressed == 1
    assert any(e.type == EventType.CAPTION_SUPPRESS for e in collector.events)


def test_sustained_silence_closes_segment_and_resets_vad():
    collector = Collector()
    vad = FakeOnlineVad()
    pipeline = _pipeline(1.0, collector, vad=vad)
    speech = np.ones(3200, dtype=np.float32) * 0.1
    silence = np.zeros(3200, dtype=np.float32)

    for _ in range(5):
        pipeline.push_chunk(speech)
    for _ in range(3):
        pipeline.push_chunk(silence)

    assert vad.reset_count == 1
    assert vad.offset == 1.6
    assert any(e.type == EventType.CAPTION_COMMIT for e in collector.events)

    pipeline.push_chunk(speech, is_final=True)
    stats = pipeline.finish()
    assert stats.vad_segments == 2
    assert stats.target_committed == 2
