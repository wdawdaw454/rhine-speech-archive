"""Fast local dictation pipeline with bounded SenseVoice segments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from ..models.backends import StreamingAsrBackend
from .event_io import EventSink, NullSink
from .events import CommitState, EventType, PipelineEvent


@dataclass
class FastDictationStats:
    accepted_chunks: int = 0
    accepted_seconds: float = 0.0
    decode_calls: int = 0
    partial_events: int = 0
    committed_segments: int = 0
    discarded_segments: int = 0
    backend_seconds: float = 0.0
    max_segment_seconds: float = 0.0


class FastDictationPipeline:
    """Turn a continuous microphone stream into bounded rapid partials.

    The prefix ASR backend is intentionally called at a coarse cadence, while
    this pipeline can consume smaller microphone callbacks. Silence and a
    maximum segment length ensure that prefix re-decoding cost stays bounded.
    """

    def __init__(
        self,
        *,
        run_id: str,
        backend: StreamingAsrBackend,
        event_sink: EventSink | None = None,
        audio_path: str | None = None,
        sample_rate: int = 16000,
        input_chunk: float = 0.10,
        decode_interval: float = 0.50,
        preroll: float = 0.30,
        speech_rms_threshold: float = 0.005,
        min_speech_duration: float = 0.20,
        silence_duration: float = 0.60,
        max_segment_duration: float = 12.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if input_chunk <= 0 or decode_interval <= 0:
            raise ValueError("input_chunk and decode_interval must be positive")
        if not 0 <= preroll < decode_interval:
            raise ValueError("preroll must be in [0, decode_interval)")
        if speech_rms_threshold < 0:
            raise ValueError("speech_rms_threshold must be non-negative")
        if min_speech_duration <= 0 or silence_duration <= 0:
            raise ValueError("endpoint durations must be positive")
        if max_segment_duration < min_speech_duration:
            raise ValueError("max_segment_duration is shorter than min_speech_duration")

        self.run_id = run_id
        self.backend = backend
        self.event_sink: EventSink = event_sink or NullSink()
        self.audio_path = audio_path
        self.sample_rate = int(sample_rate)
        self.input_chunk = float(input_chunk)
        self.decode_interval = float(decode_interval)
        self.preroll = float(preroll)
        self.speech_rms_threshold = float(speech_rms_threshold)
        self.min_speech_duration = float(min_speech_duration)
        self.silence_duration = float(silence_duration)
        self.max_segment_duration = float(max_segment_duration)

        self.stats = FastDictationStats()
        self._started_at: float | None = None
        self._total_samples = 0
        self._active = False
        self._preroll_buffer = np.empty(0, dtype=np.float32)
        self._segment_wave = np.empty(0, dtype=np.float32)
        self._decode_buffer = np.empty(0, dtype=np.float32)
        self._segment_start = 0.0
        self._speech_samples = 0
        self._silence_samples = 0
        self._last_text = ""
        self._segment_index = 0
        self._decode_samples = max(1, int(round(self.decode_interval * self.sample_rate)))

    def _now(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._started_at)

    def _emit(self, event: PipelineEvent) -> None:
        self.event_sink.write(event)

    @staticmethod
    def _rms(chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))

    def _reset_waiting(self) -> None:
        self._active = False
        self._preroll_buffer = np.empty(0, dtype=np.float32)
        self._segment_wave = np.empty(0, dtype=np.float32)
        self._decode_buffer = np.empty(0, dtype=np.float32)
        self._speech_samples = 0
        self._silence_samples = 0
        self._last_text = ""
        self.backend.reset()

    def _start_segment(self, chunk: np.ndarray, audio_time: float) -> None:
        segment = np.concatenate((self._preroll_buffer, chunk)).astype(np.float32, copy=False)
        self._active = True
        self._segment_wave = segment
        self._decode_buffer = segment
        self._speech_samples = len(chunk)
        self._silence_samples = 0
        self._segment_start = max(0.0, audio_time - len(segment) / self.sample_rate)
        segment_id = f"fast_{self._segment_index:04d}"
        self._emit(
            PipelineEvent(
                type=EventType.VAD_START,
                run_id=self.run_id,
                audio_path=self.audio_path,
                segment_id=segment_id,
                start=self._segment_start,
                end=audio_time,
                audio_time=audio_time,
                monotonic_time=self._now(),
            )
        )

    def _decode(self, chunk: np.ndarray, *, is_final: bool, audio_time: float) -> str:
        started = time.perf_counter()
        text = self.backend.accept(np.asarray(chunk, dtype=np.float32), is_final=is_final)
        self.stats.backend_seconds += time.perf_counter() - started
        self.stats.decode_calls += 1
        text = str(text).strip()
        if text and text != self._last_text:
            self._last_text = text
            self._emit(
                PipelineEvent(
                    type=EventType.ASR_PARTIAL,
                    run_id=self.run_id,
                    audio_path=self.audio_path,
                    segment_id=f"fast_{self._segment_index:04d}",
                    start=self._segment_start,
                    end=audio_time,
                    text=text,
                    commit_state=CommitState.PROVISIONAL,
                    audio_time=audio_time,
                    monotonic_time=self._now(),
                )
            )
            self.stats.partial_events += 1
        return text

    def _maybe_decode(self, audio_time: float) -> None:
        while len(self._decode_buffer) >= self._decode_samples:
            chunk = self._decode_buffer[: self._decode_samples]
            self._decode_buffer = self._decode_buffer[self._decode_samples :]
            self._decode(chunk, is_final=False, audio_time=audio_time)

    def _finish_segment(self, audio_time: float, *, discarded: bool = False) -> None:
        segment_id = f"fast_{self._segment_index:04d}"
        segment_duration = len(self._segment_wave) / self.sample_rate
        text = ""
        if discarded:
            self.stats.discarded_segments += 1
        else:
            text = self._decode(self._decode_buffer, is_final=True, audio_time=audio_time)
            self.stats.committed_segments += 1
            self.stats.max_segment_seconds = max(self.stats.max_segment_seconds, segment_duration)

        end_event = PipelineEvent(
            type=EventType.VAD_END,
            run_id=self.run_id,
            audio_path=self.audio_path,
            segment_id=segment_id,
            start=self._segment_start,
            end=audio_time,
            audio_time=audio_time,
            monotonic_time=self._now(),
            extras={"discarded": discarded, "segment_duration": round(segment_duration, 6)},
        )
        if discarded:
            end_event.commit_state = CommitState.REJECTED_BY_VAD
        self._emit(end_event)

        if not discarded:
            final_event = PipelineEvent(
                type=EventType.ASR_FINAL,
                run_id=self.run_id,
                audio_path=self.audio_path,
                segment_id=segment_id,
                start=self._segment_start,
                end=audio_time,
                text=text,
                commit_state=CommitState.COMMITTED,
                audio_time=audio_time,
                monotonic_time=self._now(),
            )
            self._emit(final_event)
            if text:
                self._emit(
                    PipelineEvent(
                        type=EventType.CAPTION_COMMIT,
                        run_id=self.run_id,
                        audio_path=self.audio_path,
                        segment_id=segment_id,
                        start=self._segment_start,
                        end=audio_time,
                        text=text,
                        commit_state=CommitState.COMMITTED,
                        audio_time=audio_time,
                        monotonic_time=self._now(),
                    )
                )

        self._segment_index += 1
        self._reset_waiting()

    def push_chunk(self, chunk: np.ndarray, *, is_final: bool = False) -> None:
        wav = np.asarray(chunk, dtype=np.float32)
        if wav.ndim != 1 or wav.size == 0:
            raise ValueError("fast dictation expects a non-empty mono chunk")
        if self._started_at is None:
            self._started_at = time.monotonic()
        self._total_samples += len(wav)
        audio_time = self._total_samples / self.sample_rate
        self.stats.accepted_chunks += 1
        self.stats.accepted_seconds = self._total_samples / self.sample_rate
        rms = self._rms(wav)

        if not self._active:
            if rms < self.speech_rms_threshold:
                preroll_samples = max(1, int(round(self.preroll * self.sample_rate)))
                self._preroll_buffer = np.concatenate((self._preroll_buffer, wav))[
                    -preroll_samples:
                ].astype(np.float32, copy=False)
                if is_final:
                    return
                return
            self._start_segment(wav, audio_time)
            self._maybe_decode(audio_time)
            if is_final:
                self._finish_segment(audio_time)
            return

        self._segment_wave = np.concatenate((self._segment_wave, wav)).astype(
            np.float32, copy=False
        )
        self._decode_buffer = np.concatenate((self._decode_buffer, wav)).astype(
            np.float32, copy=False
        )
        if rms >= self.speech_rms_threshold:
            self._speech_samples += len(wav)
            self._silence_samples = 0
        else:
            self._silence_samples += len(wav)

        speech_duration = self._speech_samples / self.sample_rate
        silence_reached = (
            speech_duration >= self.min_speech_duration
            and self._silence_samples / self.sample_rate >= self.silence_duration
        )
        max_reached = len(self._segment_wave) / self.sample_rate >= self.max_segment_duration
        if is_final or silence_reached or max_reached:
            self._finish_segment(audio_time, discarded=speech_duration < self.min_speech_duration)
            return
        self._maybe_decode(audio_time)

    def finish(self) -> FastDictationStats:
        if self._active:
            speech_duration = self._speech_samples / self.sample_rate
            self._finish_segment(
                self._total_samples / self.sample_rate,
                discarded=speech_duration < self.min_speech_duration,
            )
        return self.stats

    def stats_dict(self) -> dict:
        return asdict(self.stats)
