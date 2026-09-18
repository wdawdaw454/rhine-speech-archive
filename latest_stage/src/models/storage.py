"""Project-local model storage layout shared by services and installers."""
from __future__ import annotations

from pathlib import Path


MODEL_DIRECTORIES = {
    "sensevoice-small": "sensevoice-small",
    "sensevoice-onnx": "sensevoice-onnx",
    "sensevoice-onnx-int8": "sensevoice-onnx-int8",
    "fsmn-vad": "fsmn-vad",
    "cam-plus": "cam-plus",
    "fun-asr-nano": "fun-asr-nano",
    "qwen3-asr": "qwen3-asr",
    "moss-transcribe-diarize": "moss-transcribe-diarize",
    "sample-x": "sample-x",
    "silero-vad": "silero-vad",
    "punctuation": "punctuation",
}


def model_storage_root(project_root: str | Path) -> Path:
    """Return the repository-level model root for a latest_stage project root."""
    return Path(project_root).resolve().parent / "models"


def model_directory(project_root: str | Path, name: str) -> Path:
    try:
        directory = MODEL_DIRECTORIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown model directory: {name}") from exc
    return model_storage_root(project_root) / directory
