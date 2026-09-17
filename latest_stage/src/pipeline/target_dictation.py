"""Fail-closed, bounded-window speaker gating for live, mostly alternating speech.

Only verified audio reaches ASR. Every window is checked independently: a past
match never grants the rest of the turn permission. This is verification, not
source separation; mixed-speaker windows can still fool the embedding model.
"""
from dataclasses import dataclass
import time

import numpy as np

from ..speaker.verification import cosine_similarity
from .events import EventType, PipelineEvent


@dataclass
class TargetStats:
    accepted_seconds: float = 0.0
    verified_windows: int = 0
    suppressed_windows: int = 0
    committed_segments: int = 0
    decode_calls: int = 0
    backend_seconds: float = 0.0
    first_text_audio_seconds: float | None = None


class TargetDictationPipeline:
    def __init__(self, *, run_id, backend, vad, encoder, embedding, event_sink,
                 threshold=0.45, sample_rate=16000, window_seconds=1.5,
                 min_window_seconds=0.6):
        self.run_id, self.backend, self.vad = run_id, backend, vad
        self.encoder, self.embedding, self.sink = encoder, embedding, event_sink
        self.threshold, self.sr = float(threshold), sample_rate
        if not 0.1 <= self.threshold <= 0.95:
            raise ValueError("声纹阈值需在 0.10–0.95 之间")
        self.window = int(window_seconds * sample_rate)
        self.minimum = int(min_window_seconds * sample_rate)
        self.history = np.empty(0, np.float32)
        self.history_start = self.total = 0
        self.cursor = None
        self.pending = np.empty(0, np.float32)
        self.pending_start = 0
        self.asr_pending = np.empty(0, np.float32)
        self.asr_active = False
        self.asr_seconds = 0.0
        self.stats = TargetStats()
        self.backend.reset()
        self.vad.reset()

    def emit(self, kind, **kwargs):
        if kind in {EventType.ASR_PARTIAL, EventType.CAPTION_COMMIT} and kwargs.get("text"):
            if self.stats.first_text_audio_seconds is None:
                self.stats.first_text_audio_seconds = self.total / self.sr
        self.sink.write(PipelineEvent(type=kind, run_id=self.run_id,
                                     audio_time=self.total / self.sr, **kwargs))

    def _decode(self, audio, *, is_final):
        started = time.perf_counter()
        text = self.backend.accept(audio, is_final=is_final) or ""
        self.stats.backend_seconds += time.perf_counter() - started
        self.stats.decode_calls += 1
        return text.strip()

    def _flush(self):
        if self.asr_active:
            text = self._decode(self.asr_pending, is_final=True)
            # Prefix re-decoding may revise a preview to empty; never turn an
            # obsolete SenseVoice preview into a final transcript.
            if text.strip():
                self.emit(EventType.CAPTION_COMMIT, text=text.strip(), speaker_decision="target")
                self.stats.committed_segments += 1
            else:
                self.emit(EventType.CAPTION_SUPPRESS, speaker_decision="target",
                          extras={"reason": "empty_final"})
        self.backend.reset()
        self.asr_pending = np.empty(0, np.float32)
        self.asr_active = False
        self.asr_seconds = 0.0

    def _verify(self, audio, start):
        score = None
        decision = "pending"
        if len(audio) >= self.minimum and float(np.sqrt(np.mean(audio ** 2))) >= 0.001:
            score = cosine_similarity(self.encoder.extract(audio, self.sr), self.embedding)
            if not np.isfinite(score):
                raise ValueError("声纹模型返回了无效分数")
            decision = "target" if score >= self.threshold else "non_target"
        # Finish only previously verified samples, BEFORE reporting rejection.
        if decision != "target":
            self._flush()
        self.emit(EventType.SPEAKER_DECISION, similarity=score, speaker_decision=decision,
                  start=start / self.sr, end=(start + len(audio)) / self.sr)
        if decision != "target":
            self.stats.suppressed_windows += 1
            self.emit(EventType.CAPTION_SUPPRESS, speaker_decision=decision,
                      extras={"reason": "short_or_quiet" if score is None else "speaker_mismatch"})
            return
        self.stats.verified_windows += 1
        self.asr_active = True
        self.asr_seconds += len(audio) / self.sr
        self.asr_pending = np.concatenate((self.asr_pending, audio))
        # One prefix decode per approved window. Keep the continuous approved
        # context in the backend; rejection always resets it via _flush.
        # Short approved tails wait for the final decode, without padding.
        if self.asr_seconds >= 12:
            self._flush()
        elif len(self.asr_pending) >= self.window:
            text = self._decode(self.asr_pending, is_final=False)
            self.asr_pending = np.empty(0, np.float32)
            self.emit(EventType.ASR_PARTIAL, text=text, speaker_decision="target")

    def _consume(self, end):
        if self.cursor is None:
            return
        end = max(self.cursor, min(end, self.total))
        if self.cursor < self.history_start:
            raise RuntimeError("VAD 回溯超出缓冲范围，已停止以避免错误转写")
        if end > self.cursor:
            if not len(self.pending):
                self.pending_start = self.cursor
            self.pending = np.concatenate((self.pending,
                self.history[self.cursor - self.history_start:end - self.history_start]))
            self.cursor = end
        while len(self.pending) >= self.window:
            self._verify(self.pending[:self.window], self.pending_start)
            self.pending = self.pending[self.window:]
            self.pending_start += self.window

    def _end(self, end):
        self._consume(end)
        if len(self.pending):
            self._verify(self.pending, self.pending_start)
        self.pending = np.empty(0, np.float32)
        self._flush()
        self.cursor = None
        self.emit(EventType.VAD_END)

    def push_chunk(self, chunk, *, is_final=False):
        audio = np.asarray(chunk, np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("目标转写需要有效的单声道音频")
        self.history = np.concatenate((self.history, audio))
        self.total += len(audio)
        self.stats.accepted_seconds = self.total / self.sr
        boundaries = self.vad.accept(audio, is_final=is_final)
        self._boundaries(boundaries)
        # Keep a little lookahead for retrospective VAD end boundaries.
        self._consume(max(0, self.total - int(0.3 * self.sr)))
        keep_from = max(0, self.total - 4 * self.sr)
        if self.cursor is not None:
            keep_from = min(keep_from, self.cursor)
        self.history = self.history[keep_from - self.history_start:]
        self.history_start = keep_from

    def _boundaries(self, boundaries):
        for boundary in boundaries:
            if boundary.start is not None and self.cursor is None:
                self.cursor = min(self.total, max(self.history_start, int(round(boundary.start * self.sr))))
                self.emit(EventType.VAD_START)
            if boundary.end is not None and self.cursor is not None:
                self._end(int(round(boundary.end * self.sr)))

    def finish(self):
        if self.total:
            # Flush VAD only. Synthetic silence must never enter speaker
            # verification or ASR, even if VAD ends inside the padding.
            boundaries = self.vad.accept(np.zeros(int(0.2 * self.sr), np.float32), is_final=True)
            self._boundaries(boundaries)
        if self.cursor is not None:
            self._end(self.total)
        return self.stats
