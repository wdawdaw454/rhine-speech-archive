"""Chunk-driven target-speaker realtime pipeline."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from ..models.backends import (
    PunctuationBackend,
    SpeakerEncoderBackend,
    StreamingAsrBackend,
)
from ..models.streaming_vad import FunasrStreamingVad, VadBoundary
from ..speaker.verification import cosine_similarity, decide
from .event_io import EventSink, NullSink
from .events import CommitState, EventType, PipelineEvent
from .realtime_pipeline import PipelineStats
from .profiling import ASR, PUNCTUATION, SPEAKER_EMBEDDING, VAD, BackendProfiler


@dataclass
class _SegmentRuntime:
    segment_id: str
    start: float
    start_sample: int
    wave_parts: list[np.ndarray]
    asr_buffer: np.ndarray
    deferred_silence: list[np.ndarray]
    silence_start_sample: int | None
    peak_rms: float
    provisional_target: bool
    finalizing: bool
    next_embedding_samples: int
    asr_seen_samples: int
    final_text: str


class OnlineRealtimePipeline:
    """Consume audio chunks and emit events as segments are finalized."""

    def __init__(
        self,
        *,
        run_id: str,
        audio_path: str,
        vad: FunasrStreamingVad,
        speaker_encoder: SpeakerEncoderBackend,
        asr: StreamingAsrBackend,
        enrolled_embedding: np.ndarray,
        accept_threshold: float,
        reject_threshold: float,
        event_sink: EventSink | None = None,
        punctuator: PunctuationBackend | None = None,
        sample_rate: int = 16000,
        asr_chunk_size: float = 0.60,
        embedding_window: float = 1.50,
        embedding_hop: float = 0.50,
        first_embedding_window: float = 0.90,
        min_vad_duration: float = 0.10,
        max_vad_duration: float = 20.0,
        vad_min_silence_duration: float = 0.45,
        energy_silence_rms: float = 1e-4,
        emit_unconfirmed_partials: bool = True,
        declared_duration: float | None = None,
        stream_start_offset: float = 0.0,
    ) -> None:
        self.run_id = run_id
        self.audio_path = audio_path
        self.vad = vad
        self.speaker_encoder = speaker_encoder
        self.asr = asr
        self.punctuator = punctuator
        self.enrolled_embedding = np.asarray(enrolled_embedding, dtype=np.float32).ravel()
        self.accept_threshold = float(accept_threshold)
        self.reject_threshold = float(reject_threshold)
        self.event_sink: EventSink = event_sink or NullSink()
        self.sample_rate = int(sample_rate)
        self.asr_chunk_size = float(asr_chunk_size)
        self.embedding_window = float(embedding_window)
        self.embedding_hop = float(embedding_hop)
        self.first_embedding_window = float(first_embedding_window)
        self.min_vad_duration = float(min_vad_duration)
        self.max_vad_duration = float(max_vad_duration)
        self.vad_min_silence_duration = float(vad_min_silence_duration)
        self.energy_silence_rms = float(energy_silence_rms)
        self.emit_unconfirmed_partials = bool(emit_unconfirmed_partials)
        self.declared_duration = None if declared_duration is None else float(declared_duration)
        self.stream_start_offset = float(stream_start_offset)
        if self.stream_start_offset < 0:
            raise ValueError("stream_start_offset must be non-negative")

        self._started_at: float | None = None
        self._total_samples = 0
        self._audio_time = 0.0
        self._segment_index = 0
        self._segment: _SegmentRuntime | None = None
        self._residual = np.empty(0, dtype=np.float32)
        self._residual_start_sample = 0
        self._target_committed = 0
        self._non_target_suppressed = 0
        self._pending_suppressed = 0
        self._audio_info_emitted = False
        self.profiler = BackendProfiler()

    def _now(self) -> float:
        if self._started_at is None:
            return 0.0
        return round(time.monotonic() - self._started_at, 6)

    def _emit(self, event: PipelineEvent) -> None:
        self.event_sink.write(event)

    def _begin_if_needed(self, first_chunk_duration: float) -> None:
        if self._started_at is not None:
            return
        self._started_at = time.monotonic() - self.stream_start_offset
        duration = self.declared_duration or first_chunk_duration
        self._emit(
            PipelineEvent(
                type=EventType.AUDIO_INFO,
                run_id=self.run_id,
                audio_path=self.audio_path,
                audio_time=0.0,
                monotonic_time=0.0,
                extras={
                    "sample_rate": self.sample_rate,
                    "duration": round(duration, 3),
                    "mode": "online",
                },
            )
        )
        self._audio_info_emitted = True

    def _start_segment(self, start: float) -> None:
        segment = _SegmentRuntime(
            segment_id=f"seg_{self._segment_index:04d}",
            start=max(0.0, float(start)),
            start_sample=int(round(max(0.0, float(start)) * self.sample_rate)),
            wave_parts=[],
            asr_buffer=np.empty(0, dtype=np.float32),
            deferred_silence=[],
            silence_start_sample=None,
            peak_rms=0.0,
            provisional_target=False,
            finalizing=False,
            next_embedding_samples=max(
                1, int(round(self.first_embedding_window * self.sample_rate))
            ),
            asr_seen_samples=0,
            final_text="",
        )
        self._segment = segment
        self.asr.reset()
        self._emit(
            PipelineEvent(
                type=EventType.VAD_START,
                run_id=self.run_id,
                audio_path=self.audio_path,
                segment_id=segment.segment_id,
                start=segment.start,
                end=None,
                audio_time=segment.start,
                monotonic_time=self._now(),
            )
        )

    def _segment_wave(self) -> np.ndarray:
        if self._segment is None or not self._segment.wave_parts:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._segment.wave_parts).astype(np.float32, copy=False)

    def _decide(
        self,
        waveform: np.ndarray,
        segment: _SegmentRuntime,
        end: float,
        *,
        rolling: bool = True,
    ) -> tuple[str, float]:
        if segment.finalizing:
            rolling = False
        if rolling:
            window_samples = max(1, int(round(self.embedding_window * self.sample_rate)))
            waveform = waveform[-window_samples:]
        with self.profiler.measure(SPEAKER_EMBEDDING):
            embedding = self.speaker_encoder.extract(waveform, self.sample_rate)
        self._emit(
            PipelineEvent(
                type=EventType.SPEAKER_EMBEDDING,
                run_id=self.run_id,
                audio_path=self.audio_path,
                segment_id=segment.segment_id,
                start=segment.start,
                end=end,
                audio_time=end,
                monotonic_time=self._now(),
                extras={"embedding_dim": int(embedding.size)},
            )
        )
        similarity = cosine_similarity(embedding, self.enrolled_embedding)
        decision = decide(similarity, None, self.accept_threshold, self.reject_threshold)
        self._emit(
            PipelineEvent(
                type=EventType.SPEAKER_DECISION,
                run_id=self.run_id,
                audio_path=self.audio_path,
                segment_id=segment.segment_id,
                start=segment.start,
                end=end,
                similarity=round(similarity, 6),
                speaker_decision=decision.decision,
                audio_time=end,
                monotonic_time=self._now(),
            )
        )
        return decision.decision, similarity

    def _accept_asr(
        self,
        chunk: np.ndarray,
        *,
        is_final: bool,
        end: float,
    ) -> None:
        segment = self._segment
        if segment is None:
            return
        with self.profiler.measure(ASR):
            text = self.asr.accept(np.asarray(chunk, dtype=np.float32), is_final=is_final)
        if text:
            segment.final_text = text
        if text and (segment.provisional_target or self.emit_unconfirmed_partials):
            self._emit(
                PipelineEvent(
                    type=EventType.ASR_PARTIAL,
                    run_id=self.run_id,
                    audio_path=self.audio_path,
                    segment_id=segment.segment_id,
                    start=segment.start,
                    end=end,
                    text=text,
                    speaker_decision=("target" if segment.provisional_target else "pending"),
                    commit_state=CommitState.PROVISIONAL,
                    audio_time=end,
                    monotonic_time=self._now(),
                )
            )

    def _maybe_embed(self, sample_count: int) -> None:
        segment = self._segment
        if segment is None or sample_count < segment.next_embedding_samples:
            return
        end = segment.start + sample_count / self.sample_rate
        decision, _ = self._decide(self._segment_wave(), segment, end)
        if decision == "target":
            segment.provisional_target = True
        segment.next_embedding_samples = sample_count + max(
            1, int(round(self.embedding_hop * self.sample_rate))
        )

    def _flush_asr_chunks(self) -> bool:
        segment = self._segment
        if segment is None:
            return False
        chunk_samples = max(1, int(round(self.asr_chunk_size * self.sample_rate)))
        while len(segment.asr_buffer) >= chunk_samples:
            chunk = segment.asr_buffer[:chunk_samples]
            segment.asr_buffer = segment.asr_buffer[chunk_samples:]
            segment.asr_seen_samples += len(chunk)
            end = segment.start + segment.asr_seen_samples / self.sample_rate
            self._accept_asr(chunk, is_final=False, end=end)
            if self._segment is None:
                return False
            segment = self._segment
        return True

    def _append_speech(self, samples: np.ndarray) -> bool:
        segment = self._segment
        if segment is None or samples.size == 0:
            return self._segment is not None
        segment.wave_parts.append(np.asarray(samples, dtype=np.float32))
        segment.asr_buffer = np.concatenate(
            (segment.asr_buffer, np.asarray(samples, dtype=np.float32))
        )
        sample_count = sum(len(part) for part in segment.wave_parts)
        self._maybe_embed(sample_count)
        if self._segment is None:
            return False
        if not self._flush_asr_chunks():
            return False
        segment = self._segment
        max_samples = max(1, int(round(self.max_vad_duration * self.sample_rate)))
        if sample_count >= max_samples:
            self._close_segment("max_duration")
            return False
        return True

    def _feed_audio(self, samples: np.ndarray, absolute_start_sample: int) -> None:
        if self._segment is None or samples.size == 0:
            return
        if self._residual.size:
            # Residual and new samples are always contiguous in a live stream.
            combined = np.concatenate((self._residual, samples.astype(np.float32)))
            feed_start = self._residual_start_sample
        else:
            combined = np.asarray(samples, dtype=np.float32)
            feed_start = absolute_start_sample

        frame_size = max(1, int(round(0.03 * self.sample_rate)))
        frame_count = len(combined) // frame_size
        min_silence_samples = int(round(self.vad_min_silence_duration * self.sample_rate))

        for index in range(frame_count):
            segment = self._segment
            if segment is None:
                return
            frame = combined[index * frame_size : (index + 1) * frame_size]
            frame_start = feed_start + index * frame_size
            rms = float(np.sqrt(np.mean(np.square(frame))))
            threshold = max(
                self.energy_silence_rms,
                segment.peak_rms * 0.0025,
                1e-7,
            )
            is_silent = rms <= threshold

            if is_silent:
                if segment.silence_start_sample is None:
                    segment.silence_start_sample = frame_start
                segment.deferred_silence.append(frame)
                silence_run = frame_start - segment.silence_start_sample
                if (
                    self.vad_min_silence_duration > 0
                    and silence_run + frame_size >= min_silence_samples
                ):
                    silence_start = segment.silence_start_sample / self.sample_rate
                    self._close_segment("energy_silence", logical_end=silence_start)
                    return
                continue

            segment.peak_rms = max(segment.peak_rms, rms)
            if segment.deferred_silence:
                segment.wave_parts.extend(segment.deferred_silence)
                segment.deferred_silence.clear()
            if not self._append_speech(frame):
                return
            segment.silence_start_sample = None

        self._residual = combined[frame_count * frame_size :].astype(np.float32, copy=False)
        self._residual_start_sample = feed_start + frame_count * frame_size

    def _clear_segment_audio_state(self) -> None:
        self._segment = None
        self._residual = np.empty(0, dtype=np.float32)
        self._deferred_clear()

    def _deferred_clear(self) -> None:
        if self._segment is not None:
            self._segment.deferred_silence.clear()

    def _close_segment(
        self,
        reason: str,
        *,
        logical_end: float | None = None,
    ) -> None:
        segment = self._segment
        if segment is None:
            return

        waveform = self._segment_wave()
        sample_count = len(waveform)
        actual_end = segment.start + sample_count / self.sample_rate
        end = actual_end if logical_end is None else min(actual_end, float(logical_end))
        duration = end - segment.start

        if duration < self.min_vad_duration:
            self._emit(
                PipelineEvent(
                    type=EventType.VAD_END,
                    run_id=self.run_id,
                    audio_path=self.audio_path,
                    segment_id=segment.segment_id,
                    start=segment.start,
                    end=end,
                    audio_time=end,
                    monotonic_time=self._now(),
                    extras={"reason": "short_segment"},
                )
            )
            self._clear_segment_audio_state()
            self._segment_index += 1
            if reason == "energy_silence":
                self.vad.reset(self._audio_time)
            return

        final_chunk = segment.asr_buffer
        segment.asr_buffer = np.empty(0, dtype=np.float32)
        segment.finalizing = True
        segment.asr_seen_samples += len(final_chunk)
        final_chunk_end = segment.start + segment.asr_seen_samples / self.sample_rate
        self._accept_asr(final_chunk, is_final=True, end=final_chunk_end)

        # A closure may be triggered while flushing the final ASR chunk.
        segment = self._segment or segment
        final_decision, similarity = self._decide(waveform, segment, end)
        text = segment.final_text.strip()
        self._emit(
            PipelineEvent(
                type=EventType.ASR_FINAL,
                run_id=self.run_id,
                audio_path=self.audio_path,
                segment_id=segment.segment_id,
                start=segment.start,
                end=end,
                text=text,
                similarity=round(similarity, 6),
                speaker_decision=final_decision,
                audio_time=end,
                monotonic_time=self._now(),
            )
        )

        if final_decision == "target":
            if self.punctuator is not None and text:
                with self.profiler.measure(PUNCTUATION):
                    text = self.punctuator.restore(text)
            self._target_committed += 1
            self._emit(
                PipelineEvent(
                    type=EventType.CAPTION_COMMIT,
                    run_id=self.run_id,
                    audio_path=self.audio_path,
                    segment_id=segment.segment_id,
                    start=segment.start,
                    end=end,
                    text=text,
                    similarity=round(similarity, 6),
                    speaker_decision="target",
                    commit_state=CommitState.COMMITTED,
                    audio_time=end,
                    monotonic_time=self._now(),
                    extras={
                        "segment_duration": round(duration, 3),
                        "closure_reason": reason,
                    },
                )
            )
        else:
            closure_reason = "pending_similarity" if final_decision == "pending" else "non_target"
            if final_decision == "pending":
                self._pending_suppressed += 1
            else:
                self._non_target_suppressed += 1
            self._emit(
                PipelineEvent(
                    type=EventType.CAPTION_SUPPRESS,
                    run_id=self.run_id,
                    audio_path=self.audio_path,
                    segment_id=segment.segment_id,
                    start=segment.start,
                    end=end,
                    text=text,
                    similarity=round(similarity, 6),
                    speaker_decision=final_decision,
                    commit_state=CommitState.SUPPRESSED,
                    audio_time=end,
                    monotonic_time=self._now(),
                    extras={"reason": closure_reason, "closure_reason": reason},
                )
            )

        self._emit(
            PipelineEvent(
                type=EventType.VAD_END,
                run_id=self.run_id,
                audio_path=self.audio_path,
                segment_id=segment.segment_id,
                start=segment.start,
                end=end,
                audio_time=end,
                monotonic_time=self._now(),
            )
        )
        self._clear_segment_audio_state()
        self._segment_index += 1
        if reason == "energy_silence":
            self.vad.reset(self._audio_time)

    def _clamp_sample(self, value: float, chunk_start: int, chunk_end: int) -> int:
        return max(chunk_start, min(chunk_end, int(round(value * self.sample_rate))))

    def push_chunk(self, chunk: np.ndarray, *, is_final: bool = False) -> None:
        wav = np.asarray(chunk, dtype=np.float32)
        if wav.ndim != 1:
            raise ValueError("OnlineRealtimePipeline expects a mono chunk")
        if wav.size == 0:
            return
        self._begin_if_needed(len(wav) / self.sample_rate)

        chunk_start = self._total_samples
        self._total_samples += len(wav)
        chunk_end = self._total_samples
        self._audio_time = chunk_end / self.sample_rate
        with self.profiler.measure(VAD):
            boundaries = self.vad.accept(wav, is_final=is_final)
        cursor = chunk_start

        for boundary in boundaries:
            if boundary.start is not None and self._segment is None:
                start_sample = self._clamp_sample(boundary.start, chunk_start, chunk_end)
                self._start_segment(start_sample / self.sample_rate)
                cursor = max(cursor, start_sample)

            if boundary.end is not None and self._segment is not None:
                end_sample = self._clamp_sample(boundary.end, max(cursor, chunk_start), chunk_end)
                if end_sample > cursor:
                    self._feed_audio(wav[cursor - chunk_start : end_sample - chunk_start], cursor)
                if self._segment is not None:
                    self._close_segment("model_end", logical_end=boundary.end)
                cursor = max(cursor, end_sample)

        if self._segment is not None:
            self._feed_audio(wav[cursor - chunk_start :], cursor)
        elif not any(boundary.start is not None for boundary in boundaries):
            self._residual = np.empty(0, dtype=np.float32)

        if is_final and self._segment is not None:
            self._close_segment("stream_final")

    def finish(self) -> PipelineStats:
        if self._started_at is None:
            raise ValueError("cannot finish an online pipeline before any audio chunk")
        if self._segment is not None:
            self._close_segment("stream_final")

        duration = self._total_samples / self.sample_rate
        processing_wall = time.monotonic() - self._started_at
        stats = PipelineStats(
            audio_duration=round(duration, 6),
            processing_wall=round(processing_wall, 6),
            rtf=round(processing_wall / duration, 6) if duration > 0 else 0.0,
            vad_segments=self._segment_index,
            target_committed=self._target_committed,
            non_target_suppressed=self._non_target_suppressed,
            pending_suppressed=self._pending_suppressed,
            backend_timing=self.profiler.summary(),
            backend_durations=self.profiler.durations(),
        )
        self._emit(
            PipelineEvent(
                type=EventType.RUN_SUMMARY,
                run_id=self.run_id,
                audio_path=self.audio_path,
                audio_time=duration,
                monotonic_time=self._now(),
                extras=asdict(stats),
            )
        )
        return stats
