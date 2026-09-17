"""Audio file loading with 16k mono float32 normalization."""

from __future__ import annotations

from pathlib import Path
import subprocess
from io import BytesIO

import numpy as np
import soundfile as sf

from .resample import resample


class AudioLoadError(Exception):
    """Raised when an audio file cannot be loaded."""


def load_audio(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    p = Path(path)
    if not p.exists():
        raise AudioLoadError(f"audio file not found: {p}")
    try:
        wav, sr = sf.read(str(p), always_2d=False, dtype="float32")
    except Exception:
        # libsndfile usually handles WAV/FLAC/MP3, but MP4/M4A and some codecs
        # are better handled by ffmpeg. Keep this as a no-shell fallback.
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(p),
            "-map",
            "a:0",
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-f",
            "wav",
            "-",
        ]
        try:
            proc = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
            )
        except FileNotFoundError as exc:
            raise AudioLoadError("audio decoding failed and ffmpeg was not found") from exc
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", errors="replace").strip()
            raise AudioLoadError(f"failed to read audio file {p}: {detail}")
        wav, sr = sf.read(BytesIO(proc.stdout), always_2d=False, dtype="float32")

    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32, copy=False)

    if sr != target_sr:
        wav = resample(wav, sr, target_sr)
        sr = target_sr

    return wav, sr
