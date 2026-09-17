"""Speaker enrollment profile creation and persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..audio.io import load_audio
from ..models.backends import SpeakerEncoderBackend
from ..speaker.verification import enroll_embedding


class EnrollmentError(Exception):
    """Raised when enrollment audio is missing, silent, or too short."""


@dataclass
class SpeakerProfile:
    speaker_id: str
    model_id: str
    embedding_dim: int
    embedding: list[float]
    embedding_norm: str
    num_enroll_utterances: int
    enroll_duration: float
    created_at: str


def _check_waveform(waveform: np.ndarray, path: Path) -> tuple[float, float]:
    if waveform.size == 0:
        raise EnrollmentError(f"empty enrollment audio: {path}")
    peak = float(np.max(np.abs(waveform)))
    rms = float(np.sqrt(np.mean(np.square(waveform))))
    if peak >= 0.999:
        raise EnrollmentError(f"clipped enrollment audio: {path}")
    if rms < 1e-4:
        raise EnrollmentError(f"silent or extremely low-level enrollment audio: {path}")
    return peak, rms


def build_speaker_profile(
    speaker_id: str,
    audio_paths: list[str | Path],
    encoder: SpeakerEncoderBackend,
    *,
    model_id: str,
    sample_rate: int = 16000,
    min_total_duration: float = 3.0,
) -> SpeakerProfile:
    if not audio_paths:
        raise EnrollmentError("at least one enrollment audio file is required")

    embeddings: list[np.ndarray] = []
    durations: list[float] = []
    quality: list[dict[str, float]] = []
    for raw_path in audio_paths:
        path = Path(raw_path)
        if not path.exists():
            raise EnrollmentError(f"enrollment audio not found: {path}")
        waveform, sr = load_audio(path, target_sr=sample_rate)
        peak, rms = _check_waveform(waveform, path)
        embeddings.append(encoder.extract(waveform, sr))
        duration = len(waveform) / sr
        durations.append(duration)
        quality.append({"peak": peak, "rms": rms})

    total_duration = float(sum(durations))
    if total_duration < min_total_duration:
        raise EnrollmentError(
            f"total enrollment duration {total_duration:.2f}s is below "
            f"{min_total_duration:.2f}s"
        )
    embedding = enroll_embedding(embeddings, norm="l2")
    return SpeakerProfile(
        speaker_id=speaker_id,
        model_id=model_id,
        embedding_dim=int(embedding.size),
        embedding=[float(x) for x in embedding],
        embedding_norm="l2",
        num_enroll_utterances=len(embeddings),
        enroll_duration=round(total_duration, 3),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def save_profile(profile: SpeakerProfile, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(profile), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def load_profile(path: str | Path) -> SpeakerProfile:
    p = Path(path)
    if not p.exists():
        raise EnrollmentError(f"speaker profile not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return SpeakerProfile(**data)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise EnrollmentError(f"invalid speaker profile {p}: {exc}") from exc


def profile_embedding(profile: SpeakerProfile) -> np.ndarray:
    embedding = np.asarray(profile.embedding, dtype=np.float32)
    if embedding.ndim != 1 or embedding.size != profile.embedding_dim:
        raise EnrollmentError("profile embedding dimension mismatch")
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm == 0.0:
        raise EnrollmentError("profile embedding is empty or non-finite")
    return embedding / norm
