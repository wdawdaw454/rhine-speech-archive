"""Deterministic two-speaker mixture construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SpeakerRole = Literal["target", "non_target"]


@dataclass(frozen=True)
class MixtureSource:
    source_id: str
    speaker: str
    text: str
    duration: float


@dataclass(frozen=True)
class MixturePlacement:
    utterance_id: str
    source_id: str
    speaker: str
    role: SpeakerRole
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def schedule_alternating(
    *,
    target_sources: list[MixtureSource],
    non_target_sources: list[MixtureSource],
    num_utterances: int,
    gap: float,
    overlap: float,
) -> list[MixturePlacement]:
    """Schedule target and interfering utterances in alternating turns.

    ``overlap`` is subtracted from the inter-utterance gap. A positive value
    makes adjacent turns overlap; a negative value is rejected because it is
    indistinguishable from a larger gap in this lightweight format.
    """

    if not target_sources or not non_target_sources:
        raise ValueError("both target and non-target source lists are required")
    if num_utterances <= 0:
        raise ValueError("num_utterances must be positive")
    if gap < 0:
        raise ValueError("gap must be non-negative")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    placements: list[MixturePlacement] = []
    cursor = 0.0
    for index in range(num_utterances):
        is_target = index % 2 == 0
        sources = target_sources if is_target else non_target_sources
        source = sources[(index // 2) % len(sources)]
        if source.duration <= 0:
            raise ValueError(f"source has non-positive duration: {source.source_id}")

        start = 0.0 if index == 0 else max(0.0, cursor + gap - overlap)
        end = start + source.duration
        role: SpeakerRole = "target" if is_target else "non_target"
        placements.append(
            MixturePlacement(
                utterance_id=f"utt_{index:04d}_{source.source_id}",
                source_id=source.source_id,
                speaker=source.speaker,
                role=role,
                start=round(start, 6),
                end=round(end, 6),
                text=source.text,
            )
        )
        cursor = end
    return placements


def _rms(waveform: np.ndarray) -> float:
    wav = np.asarray(waveform, dtype=np.float32)
    if wav.size == 0:
        raise ValueError("cannot compute RMS for an empty waveform")
    value = float(np.sqrt(np.mean(np.square(wav), dtype=np.float64)))
    if value < 1e-8:
        raise ValueError("mixture source is silent or nearly silent")
    return value


def render_mixture(
    placements: list[MixturePlacement],
    waveforms: dict[str, np.ndarray],
    *,
    sample_rate: int,
    target_snr_db: float,
    tail_duration: float = 0.20,
) -> tuple[np.ndarray, float]:
    """Mix normalized sources and return the mixture plus a peak-normalization factor.

    Each source is RMS-normalized before mixing. The interferer is scaled from
    the requested target-to-interferer SNR. If the mixture would clip, all
    sources are scaled by the same factor so relative level is preserved.
    """

    if not placements:
        raise ValueError("at least one placement is required")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if tail_duration < 0:
        raise ValueError("tail_duration must be non-negative")

    total_samples = max(
        int(round((placement.end + tail_duration) * sample_rate))
        for placement in placements
    )
    mixture = np.zeros(total_samples, dtype=np.float32)
    interferer_gain = 10.0 ** (-float(target_snr_db) / 20.0)

    for placement in placements:
        if placement.source_id not in waveforms:
            raise KeyError(f"missing waveform for source: {placement.source_id}")
        wav = np.asarray(waveforms[placement.source_id], dtype=np.float32)
        if wav.ndim != 1:
            raise ValueError(f"source waveform must be mono: {placement.source_id}")
        normalized = wav / _rms(wav)
        gain = 1.0 if placement.role == "target" else interferer_gain
        start_sample = int(round(placement.start * sample_rate))
        end_sample = min(total_samples, start_sample + len(normalized))
        if end_sample <= start_sample:
            continue
        mixture[start_sample:end_sample] += normalized[: end_sample - start_sample] * gain

    peak = float(np.max(np.abs(mixture)))
    normalization = 1.0
    if peak > 0.99:
        normalization = 0.99 / peak
        mixture *= normalization
    return mixture, normalization
