"""Target-speaker file-streaming pipeline for Phase 1."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..models.backends import (
    PunctuationBackend,
    SpeakerEncoderBackend,
    SpeechSegment,
    StreamingAsrBackend,
    VadBackend,
)
from ..speaker.verification import cosine_similarity, decide
from ..vad.stream_vad import VadConfigValues, prepare_segments, waveform_slice
from .event_io import EventSink, NullSink
from .events import CommitState, EventType, PipelineEvent
from .profiling import ASR, PUNCTUATION, SPEAKER_EMBEDDING, VAD, BackendProfiler


@dataclass
class PipelineStats:
    audio_duration: float
    processing_wall: float
    rtf: float
    vad_segments: int
    target_committed: int
    non_target_suppressed: int
    pending_suppressed: int
    backend_timing: dict[str, dict[str, float | int]] = field(default_factory=dict)
    backend_durations: dict[str, list[float]] = field(default_factory=dict)


class RealtimePipeline:
    """Run VAD, speaker gating, streaming ASR, and final caption commit.

    Phase 1 deliberately keeps model calls inside injected backends. The input is
    consumed as a simulated stream by callers; this class keeps event semantics
    independent of whether chunks arrive from a file, microphone, or socket.
    """

    def __init__(
        self,
        *,
        run_id: str,
        audio_path: str,
        vad: VadBackend,
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
        speech_pad: float = 0.20,
        min_vad_duration: float = 0.10,
        max_vad_duration: float = 20.0,
        vad_merge_gap: float = 0.30,
        vad_min_silence_duration: float = 0.45,
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
        self.speech_pad = float(speech_pad)
        self.min_vad_duration = float(min_vad_duration)
        self.max_vad_duration = float(max_vad_duration)
        self.vad_merge_gap = float(vad_merge_gap)
        self.vad_min_silence_duration = float(vad_min_silence_duration)
        self.profiler = BackendProfiler()

    def _emit(self, event: PipelineEvent) -> None:
        self.event_sink.write(event)

    def _relative_monotonic(self, started_at: float) -> float:
        return round(time.monotonic() - started_at, 6)

    def _decide(
        self,
        waveform: np.ndarray,
        segment_id: str,
        start: float,
        end: float,
        audio_time: float,
        elapsed: float,
    ) -> tuple[str, float]:
        with self.profiler.measure(SPEAKER_EMBEDDING):
            embedding = self.speaker_encoder.extract(waveform, self.sample_rate)
        self._emit(
            PipelineEvent(
                type=EventType.SPEAKER_EMBEDDING,
                run_id=self.run_id,
                audio_path=self.audio_path,
                segment_id=segment_id,
                start=start,
                end=end,
                audio_time=audio_time,
                monotonic_time=elapsed,
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
                segment_id=segment_id,
                start=start,
                end=end,
                similarity=round(similarity, 6),
                speaker_decision=decision.decision,
                audio_time=audio_time,
                monotonic_time=elapsed,
            )
        )
        return decision.decision, similarity

    def _process_segment(
        self,
        segment: tuple[float, float],
        segment_index: int,
        waveform: np.ndarray,
        started_at: float,
        stats_target: list[int],
        stats_non_target: list[int],
        stats_pending: list[int],
    ) -> None:
        start, end = segment
        segment_id = f"seg_{segment_index:04d}"
        duration = end - start
        segment_wave = waveform_slice(
            waveform, self.sample_rate, SpeechSegment(start=start, end=end)
        )
        samples = len(segment_wave)
        chunk_samples = max(1, int(round(self.asr_chunk_size * self.sample_rate)))
        embedding_samples = max(1, int(round(self.embedding_window * self.sample_rate)))
        embedding_hop_samples = max(1, int(round(self.embedding_hop * self.sample_rate)))

        self._emit(
            PipelineEvent(
                type=EventType.VAD_START,
                run_id=self.run_id,
                audio_path=self.audio_path,
                segment_id=segment_id,
                start=start,
                end=end,
                audio_time=start,
                monotonic_time=self._relative_monotonic(started_at),
            )
        )

        self.asr.reset()
        provisional_target = False
        next_embedding_offset = embedding_samples
        offset = 0
        final_text = ""

        while offset < samples:
            chunk_end = min(samples, offset + chunk_samples)
            chunk = segment_wave[offset:chunk_end]

            if offset >= next_embedding_offset or chunk_end == samples:
                current_end = start + chunk_end / self.sample_rate
                early_decision, _ = self._decide(
                    segment_wave[:chunk_end],
                    segment_id,
                    start,
                    current_end,
                    current_end,
                    self._relative_monotonic(started_at),
                )
                if early_decision == "target":
                    provisional_target = True
                next_embedding_offset = chunk_end + embedding_hop_samples

            is_final = chunk_end == samples
            with self.profiler.measure(ASR):
                text = self.asr.accept(chunk, is_final=is_final)
            if text:
                final_text = text
            if provisional_target and text:
                self._emit(
                    PipelineEvent(
                        type=EventType.ASR_PARTIAL,
                        run_id=self.run_id,
                        audio_path=self.audio_path,
                        segment_id=segment_id,
                        start=start,
                        end=start + chunk_end / self.sample_rate,
                        text=text,
                        speaker_decision="target",
                        commit_state=CommitState.PROVISIONAL,
                        audio_time=start + chunk_end / self.sample_rate,
                        monotonic_time=self._relative_monotonic(started_at),
                    )
                )
            offset = chunk_end

        final_decision, similarity = self._decide(
            segment_wave,
            segment_id,
            start,
            end,
            end,
            self._relative_monotonic(started_at),
        )
        text = final_text.strip()
        self._emit(
            PipelineEvent(
                type=EventType.ASR_FINAL,
                run_id=self.run_id,
                audio_path=self.audio_path,
                segment_id=segment_id,
                start=start,
                end=end,
                text=text,
                similarity=round(similarity, 6),
                speaker_decision=final_decision,
                audio_time=end,
                monotonic_time=self._relative_monotonic(started_at),
            )
        )
        if final_decision == "target":
            if self.punctuator is not None and text:
                with self.profiler.measure(PUNCTUATION):
                    text = self.punctuator.restore(text)
            stats_target.append(1)
            self._emit(
                PipelineEvent(
                    type=EventType.CAPTION_COMMIT,
                    run_id=self.run_id,
                    audio_path=self.audio_path,
                    segment_id=segment_id,
                    start=start,
                    end=end,
                    text=text,
                    similarity=round(similarity, 6),
                    speaker_decision="target",
                    commit_state=CommitState.COMMITTED,
                    audio_time=end,
                    monotonic_time=self._relative_monotonic(started_at),
                    extras={"segment_duration": round(duration, 3)},
                )
            )
        else:
            reason = "pending_similarity" if final_decision == "pending" else "non_target"
            if final_decision == "pending":
                stats_pending.append(1)
            else:
                stats_non_target.append(1)
            self._emit(
                PipelineEvent(
                    type=EventType.CAPTION_SUPPRESS,
                    run_id=self.run_id,
                    audio_path=self.audio_path,
                    segment_id=segment_id,
                    start=start,
                    end=end,
                    text=text,
                    similarity=round(similarity, 6),
                    speaker_decision=final_decision,
                    commit_state=CommitState.SUPPRESSED,
                    audio_time=end,
                    monotonic_time=self._relative_monotonic(started_at),
                    extras={"reason": reason},
                )
            )
        self._emit(
            PipelineEvent(
                type=EventType.VAD_END,
                run_id=self.run_id,
                audio_path=self.audio_path,
                segment_id=segment_id,
                start=start,
                end=end,
                audio_time=end,
                monotonic_time=self._relative_monotonic(started_at),
            )
        )

    def process(self, waveform: np.ndarray) -> PipelineStats:
        wav = np.asarray(waveform, dtype=np.float32)
        if wav.ndim != 1:
            raise ValueError("RealtimePipeline expects a mono waveform")
        started_at = time.monotonic()
        duration = len(wav) / self.sample_rate
        self._emit(
            PipelineEvent(
                type=EventType.AUDIO_INFO,
                run_id=self.run_id,
                audio_path=self.audio_path,
                audio_time=0.0,
                monotonic_time=0.0,
                extras={"sample_rate": self.sample_rate, "duration": round(duration, 3)},
            )
        )

        try:
            with self.profiler.measure(VAD):
                raw_segments = self.vad.detect(wav, self.sample_rate)
            segments = prepare_segments(
                raw_segments,
                duration=duration,
                config=VadConfigValues(
                    speech_pad=self.speech_pad,
                    min_duration=self.min_vad_duration,
                    max_duration=self.max_vad_duration,
                    merge_gap=self.vad_merge_gap,
                ),
                waveform=wav,
                sample_rate=self.sample_rate,
                min_silence_duration=self.vad_min_silence_duration,
            )
        except Exception as exc:
            self._emit(
                PipelineEvent(
                    type=EventType.PIPELINE_ERROR,
                    run_id=self.run_id,
                    audio_path=self.audio_path,
                    audio_time=duration,
                    monotonic_time=self._relative_monotonic(started_at),
                    extras={"stage": "vad", "error": str(exc)},
                )
            )
            raise

        target: list[int] = []
        non_target: list[int] = []
        pending: list[int] = []
        for index, segment in enumerate(segments):
            self._process_segment(
                (segment.start, segment.end), index, wav, started_at, target, non_target, pending
            )

        processing_wall = time.monotonic() - started_at
        stats = PipelineStats(
            audio_duration=round(duration, 6),
            processing_wall=round(processing_wall, 6),
            rtf=round(processing_wall / duration, 6) if duration > 0 else 0.0,
            vad_segments=len(segments),
            target_committed=len(target),
            non_target_suppressed=len(non_target),
            pending_suppressed=len(pending),
            backend_timing=self.profiler.summary(),
            backend_durations=self.profiler.durations(),
        )
        self._emit(
            PipelineEvent(
                type=EventType.RUN_SUMMARY,
                run_id=self.run_id,
                audio_path=self.audio_path,
                audio_time=duration,
                monotonic_time=self._relative_monotonic(started_at),
                extras=asdict(stats),
            )
        )
        return stats
