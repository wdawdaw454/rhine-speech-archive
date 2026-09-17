"""VAD-segmented, single-flight prefix decoding for non-streaming Nano.

The capture thread only runs CPU VAD and submits snapshots. One ASR worker
coalesces previews, prioritizes finals and drains on stop. No audio is dropped
to hide overload: a bounded final backlog fails explicitly instead.
"""
from collections import deque
from dataclasses import dataclass
from threading import Condition, Thread
import time

import numpy as np

from .event_io import NullSink
from .events import EventType, PipelineEvent


@dataclass
class NanoRealtimeStats:
    accepted_chunks: int = 0
    accepted_seconds: float = 0.0
    decode_calls: int = 0
    partial_events: int = 0
    committed_segments: int = 0
    discarded_segments: int = 0
    coalesced_previews: int = 0
    reused_finals: int = 0
    backend_seconds: float = 0.0
    vad_seconds: float = 0.0
    max_segment_seconds: float = 0.0
    max_final_backlog_seconds: float = 0.0
    first_text_seconds: float | None = None
    last_final_delay_seconds: float | None = None


@dataclass
class _Segment:
    number: int
    start: int
    end: int | None = None
    result_end: int = -1
    result: str = ""


@dataclass
class _Job:
    segment: _Segment
    end: int
    audio: np.ndarray
    final: bool = False


class NanoRealtimePipeline:
    def __init__(self, *, run_id, backend, vad, event_sink=None, sample_rate=16000,
                 initial_audio=1.2, decode_interval=1.2, max_segment_duration=8.0,
                 max_backlog_seconds=30.0, clock=time.monotonic):
        if sample_rate != 16000:
            raise ValueError("Nano realtime requires 16 kHz audio")
        if not 0 < initial_audio <= max_segment_duration or decode_interval <= 0:
            raise ValueError("invalid Nano realtime timing")
        self.run_id, self.backend, self.vad = run_id, backend, vad
        self.sink = event_sink or NullSink()
        self.sample_rate, self.clock = sample_rate, clock
        self.initial_samples = round(initial_audio * sample_rate)
        self.interval_samples = round(decode_interval * sample_rate)
        self.max_samples = round(max_segment_duration * sample_rate)
        self.max_backlog_seconds = max_backlog_seconds
        self.stats = NanoRealtimeStats()
        self._cv = Condition()
        self._audio = np.empty(0, np.float32)
        self._base = self._total = self._floor = self._number = 0
        self._active = None
        self._preview = None
        self._finals = deque()
        self._inflight = None
        self._next_preview = 0
        self._origin = None
        self._closing = self._aborted = False
        self._error = None
        self.vad.reset()
        self._worker = Thread(target=self._decode_loop, name="nano-asr", daemon=True)
        self._worker.start()

    def _check(self):
        if self._error is not None:
            raise RuntimeError(f"Nano 实时识别失败：{self._error}") from self._error

    def _emit(self, kind, segment, *, end=None, text=None, **extras):
        # All emitters hold _cv, including the inference worker. Event files
        # and the UI therefore see the same serialized sequence.
        self.sink.write(PipelineEvent(
            type=kind, run_id=self.run_id, segment_id=f"nano-{segment.number}",
            start=segment.start / self.sample_rate,
            end=None if end is None else end / self.sample_rate, text=text,
            audio_time=self._total / self.sample_rate,
            monotonic_time=self.clock(), extras=extras,
        ))

    def _start(self, start):
        start = max(self._floor, self._base, min(self._total, start))
        self._number += 1
        self._active = _Segment(self._number, start)
        self._next_preview = start + self.initial_samples
        self._emit(EventType.VAD_START, self._active)

    def _job(self, segment, end, final=False):
        return _Job(segment, end,
                    self._audio[segment.start - self._base:end - self._base].copy(), final)

    def _end(self, end):
        segment = self._active
        if segment is None:
            return
        end = max(segment.start, min(end, self._total))
        segment.end = end
        self._active = None
        self._floor = max(self._floor, end)
        if self._preview is not None and self._preview.segment is segment:
            self._preview = None
            self.stats.coalesced_previews += 1
        self._emit(EventType.VAD_END, segment, end=end, awaiting_final=True)
        job = self._job(segment, end, final=True)
        # VAD, not raw energy, decides speech. Reject only tiny/empty/digital
        # silence segments to limit hallucinations without excluding quiet speech.
        if len(job.audio) < self.sample_rate * 0.2 or not np.any(job.audio):
            self.stats.discarded_segments += 1
            self._emit(EventType.CAPTION_SUPPRESS, segment, end=end)
            return
        self._finals.append(job)
        backlog = sum(len(j.audio) for j in self._finals) / self.sample_rate
        self.stats.max_final_backlog_seconds = max(self.stats.max_final_backlog_seconds, backlog)
        if backlog > self.max_backlog_seconds:
            raise RuntimeError(f"Nano 句尾任务积压超过 {self.max_backlog_seconds:g} 秒，已停止；请降低系统负载后重试")
        self._cv.notify_all()

    def _boundaries(self, boundaries):
        for bound in boundaries:
            if bound.start is not None:
                start = round(bound.start * self.sample_rate)
                if self._active is None:
                    self._start(start)
            if bound.end is not None:
                end = min(self._total, round(bound.end * self.sample_rate))
                # Defensive hard bound even if a VAD implementation emits a
                # whole long segment at once. Adjacent slices do not overlap.
                while self._active is not None and end - self._active.start > self.max_samples:
                    split = self._active.start + self.max_samples
                    self._end(split)
                    self._start(split)
                self._end(end)

    def push_chunk(self, chunk):
        samples = np.asarray(chunk, np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("Nano expects finite mono audio")
        if not len(samples):
            return
        with self._cv:
            self._check()
            if self._closing:
                raise RuntimeError("Nano session already closed")
            if self._origin is None:
                self._origin = self.clock() - len(samples) / self.sample_rate
            self._audio = np.concatenate((self._audio, samples))
            self._total += len(samples)
            self.stats.accepted_chunks += 1
            self.stats.accepted_seconds = self._total / self.sample_rate
            started = self.clock()
            boundaries = self.vad.accept(samples, is_final=False)
            self.stats.vad_seconds += self.clock() - started
            self._boundaries(boundaries)
            while self._active is not None and self._total - self._active.start >= self.max_samples:
                split = self._active.start + self.max_samples
                self._end(split)
                self._start(split)
            if self._active is not None and self._total >= self._next_preview:
                if self._preview is not None:
                    self.stats.coalesced_previews += 1
                self._preview = self._job(self._active, self._total)
                self._next_preview = self._total + self.interval_samples
                self._cv.notify_all()
            # Retain lookback for VAD onset, but never the entire recording.
            keep = self._active.start if self._active is not None else max(0, self._total - self.sample_rate)
            keep = max(self._base, keep)
            self._audio = self._audio[keep - self._base:].copy()
            self._base = keep

    def _decode_loop(self):
        try:
            while True:
                with self._cv:
                    self._cv.wait_for(lambda: self._closing or self._finals or self._preview is not None)
                    if self._aborted:
                        return
                    if self._finals:
                        job = self._finals.popleft()
                    elif self._preview is not None:
                        job, self._preview = self._preview, None
                    elif self._closing:
                        return
                    else:
                        continue
                    self._inflight = job
                    reuse = job.final and job.segment.result_end == job.end
                    text = job.segment.result if reuse else ""
                started = self.clock()
                if not reuse:
                    text = self.backend.transcribe(job.audio).strip()
                duration = self.clock() - started
                with self._cv:
                    self._inflight = None
                    if self._aborted:
                        return
                    if reuse:
                        self.stats.reused_finals += 1
                    else:
                        self.stats.decode_calls += 1
                        self.stats.backend_seconds += duration
                    self.stats.max_segment_seconds = max(
                        self.stats.max_segment_seconds, len(job.audio) / self.sample_rate)
                    segment = job.segment
                    segment.result_end, segment.result = job.end, text
                    if job.final:
                        delay = max(0.0, self.clock() - self._origin - job.end / self.sample_rate)
                        self.stats.last_final_delay_seconds = delay
                        if text and any(c.isalnum() for c in text):
                            self.stats.committed_segments += 1
                            self._first_text(segment)
                            self._emit(EventType.CAPTION_COMMIT, segment, end=job.end, text=text,
                                       backend_seconds=duration, final_delay_seconds=delay,
                                       reused_preview=reuse)
                        else:
                            self.stats.discarded_segments += 1
                            self._emit(EventType.CAPTION_SUPPRESS, segment, end=job.end)
                    elif segment is self._active and text and any(c.isalnum() for c in text):
                        # Never let an old utterance's preview overwrite a newer
                        # utterance or reappear after its final boundary.
                        self.stats.partial_events += 1
                        self._first_text(segment)
                        self._emit(EventType.ASR_PARTIAL, segment, end=job.end, text=text,
                                   backend_seconds=duration, provisional=True)
        except Exception as exc:
            with self._cv:
                self._error = exc
                self._cv.notify_all()

    def _first_text(self, segment):
        if self.stats.first_text_seconds is None:
            self.stats.first_text_seconds = max(
                0.0, self.clock() - self._origin - segment.start / self.sample_rate)

    def finish(self):
        try:
            with self._cv:
                self._check()
                if not self._closing:
                    if self._total:
                        started = self.clock()
                        self._boundaries(self.vad.accept(np.empty(0, np.float32), is_final=True))
                        self.stats.vad_seconds += self.clock() - started
                    self._end(self._total)
                    self._preview = None
                    self._closing = True
                    self._cv.notify_all()
            self._worker.join()
            with self._cv:
                self._check()
            return self.stats
        except Exception:
            self.close()
            raise

    def close(self):
        """Error cleanup: stop queued work and wait before releasing the sink/model."""
        with self._cv:
            self._aborted = self._closing = True
            self._finals.clear()
            self._preview = None
            self._cv.notify_all()
        self._worker.join()
