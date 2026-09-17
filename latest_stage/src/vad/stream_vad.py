"""Pure helpers around VAD backend output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models.backends import FunasrVad, SpeechSegment, VadBackend


@dataclass
class VadConfigValues:
    speech_pad: float = 0.20
    min_duration: float = 0.10
    max_duration: float = 20.0
    merge_gap: float = 0.30


def build_funasr_vad(model_dir: str | Path, device: str = "cpu") -> FunasrVad:
    return FunasrVad(model_dir, device=device)


def split_segments_on_silence(
    segments: list[SpeechSegment],
    *,
    waveform: np.ndarray,
    sample_rate: int,
    min_silence_duration: float,
) -> list[SpeechSegment]:
    """Split overlong VAD segments at sustained low-energy regions.

    FunASR occasionally marks an alternating synthetic mixture as one
    continuous segment even when the waveform contains exact silence. This
    fallback is conservative: the low-energy region must be long and far below
    the segment's 95th-percentile frame energy.
    """

    if min_silence_duration <= 0 or len(waveform) == 0:
        return segments

    frame_size = max(1, int(round(0.03 * sample_rate)))
    frame_duration = frame_size / sample_rate
    min_frames = int(np.ceil(min_silence_duration / frame_duration))
    result: list[SpeechSegment] = []

    for segment in segments:
        start_sample = max(0, int(round(segment.start * sample_rate)))
        end_sample = min(len(waveform), int(round(segment.end * sample_rate)))
        wav = np.asarray(waveform[start_sample:end_sample], dtype=np.float32)
        frame_count = len(wav) // frame_size
        if frame_count <= min_frames:
            result.append(segment)
            continue

        frames = wav[: frame_count * frame_size].reshape(frame_count, frame_size)
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        peak = float(np.percentile(rms, 95))
        floor = float(np.percentile(rms, 5))
        floor_threshold = min(floor * 4.0, peak * 0.1)
        threshold = max(peak * 0.0025, floor_threshold, 1e-5)
        silent = rms <= threshold

        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for index, is_silent in enumerate(silent):
            if is_silent and run_start is None:
                run_start = index
            elif not is_silent and run_start is not None:
                if index - run_start >= min_frames:
                    runs.append((run_start, index))
                run_start = None
        if run_start is not None and len(silent) - run_start >= min_frames:
            runs.append((run_start, len(silent)))

        start_offset = 0.0
        end_offset = frame_count * frame_duration
        internal_gaps: list[tuple[float, float]] = []
        for run_start_index, run_end_index in runs:
            run_start_time = run_start_index * frame_duration
            run_end_time = run_end_index * frame_duration
            if run_start_time <= 0.15:
                start_offset = run_end_time
            elif run_end_time >= end_offset - 0.15:
                end_offset = run_start_time
            else:
                internal_gaps.append((run_start_time, run_end_time))

        cursor = start_offset
        for gap_start, gap_end in internal_gaps:
            if gap_start - cursor > 0.0:
                result.append(
                    SpeechSegment(start=segment.start + cursor, end=segment.start + gap_start)
                )
            cursor = gap_end
        if end_offset - cursor > 0.0:
            result.append(
                SpeechSegment(
                    start=segment.start + cursor,
                    end=segment.start + end_offset,
                )
            )

    return sorted(result, key=lambda item: item.start)


def prepare_segments(
    segments: list[SpeechSegment],
    *,
    duration: float,
    config: VadConfigValues | None = None,
    waveform: np.ndarray | None = None,
    sample_rate: int = 16000,
    min_silence_duration: float = 0.0,
) -> list[SpeechSegment]:
    """Pad, clamp, merge, split, and filter VAD segments.

    Splitting long segments keeps later speaker and ASR processing bounded. This
    is also useful when a VAD backend fails to split long continuous speech.
    """

    cfg = config or VadConfigValues()
    normalized = list(segments)
    if waveform is not None and min_silence_duration > 0:
        normalized = split_segments_on_silence(
            normalized,
            waveform=waveform,
            sample_rate=sample_rate,
            min_silence_duration=min_silence_duration,
        )

    padded: list[tuple[float, float]] = []
    for seg in sorted(normalized, key=lambda item: item.start):
        start = max(0.0, seg.start - cfg.speech_pad)
        end = min(duration, seg.end + cfg.speech_pad)
        if end <= start:
            continue
        if padded and start - padded[-1][1] <= cfg.merge_gap:
            prev_start, prev_end = padded[-1]
            padded[-1] = (prev_start, max(prev_end, end))
        else:
            padded.append((start, end))

    prepared: list[SpeechSegment] = []
    for start, end in padded:
        span = end - start
        if span < cfg.min_duration:
            continue
        if span <= cfg.max_duration:
            prepared.append(SpeechSegment(start=start, end=end))
            continue
        pieces = int(np.ceil(span / cfg.max_duration))
        step = span / pieces
        for i in range(pieces):
            seg_start = start + i * step
            seg_end = min(end, seg_start + step)
            prepared.append(SpeechSegment(start=seg_start, end=seg_end))
    return prepared


def waveform_slice(
    waveform: np.ndarray,
    sample_rate: int,
    segment: SpeechSegment,
) -> np.ndarray:
    start = max(0, int(round(segment.start * sample_rate)))
    end = min(len(waveform), int(round(segment.end * sample_rate)))
    return np.asarray(waveform[start:end], dtype=np.float32)
