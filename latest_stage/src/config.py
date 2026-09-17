"""Configuration schema and loader (pydantic v2)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value: str | Path, base: Path) -> Path:
    """Resolve a path string against a base directory if not absolute."""
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_rate: int = 16000
    mono: bool = True
    seed: int = 42
    device: Literal["auto", "cpu", "cuda", "cuda:0", "cuda:1"] = "auto"

    def resolved_device(self) -> str:
        if self.device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    # Prefer the GPU with the most free memory. Shared servers often
                    # have GPU 0 occupied even when another GPU is idle.
                    free = [
                        torch.cuda.mem_get_info(index)[0]
                        for index in range(torch.cuda.device_count())
                    ]
                    return f"cuda:{free.index(max(free))}" if free else "cuda:0"
            except ImportError:
                pass
            return "cpu"
        return self.device


class AudioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_size: float = 0.20
    min_duration: float = 0.10
    max_duration: float = 3600.0
    realtime_sleep: bool = False


class VadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    model_dir: str
    threshold: float = 0.5
    min_silence_duration: float = 0.45
    max_segment_length: float = 20.0
    merge_gap: float = 0.30
    speech_pad: float = 0.20
    chunk_size: float = 0.20


class SpeakerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    model_dir: str
    min_segment_duration: float = 0.60
    embedding_window: float = 1.50
    embedding_hop: float = 0.50
    first_embedding_window: float = 0.90
    accept_threshold: float | None = None
    reject_threshold: float | None = None
    calibration_target_far: float = 0.01
    embedding_norm: Literal["l2"] = "l2"
    emit_unconfirmed_partials: bool = True


class PunctuationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    model_id: str
    model_dir: str


class AsrConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["sensevoice_onnx"] = "sensevoice_onnx"
    model_id: str
    model_dir: str
    chunk_size: float = 0.60
    hotwords_file: str | None = None
    min_avg_confidence: float | None = None


class PostprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    punctuation: PunctuationConfig
    remove_fillers: bool = False
    normalize_width: bool = True
    normalize_number: bool = True


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_dir: str = "outputs/runs"
    save_jsonl: bool = True
    save_transcript: bool = True
    save_audio_debug: bool = False


class CerNormalizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_whitespace: bool = True
    to_simplified_chinese: bool = True
    normalize_numbers: bool = True


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cer_normalization: CerNormalizationConfig


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VadConfig = Field(
        default_factory=lambda: VadConfig(
            model_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", model_dir="models/vad"
        )
    )
    speaker: SpeakerConfig = Field(
        default_factory=lambda: SpeakerConfig(
            model_id="iic/speech_campplus_sv_zh-cn_16k-common", model_dir="models/speaker"
        )
    )
    asr: AsrConfig = Field(
        default_factory=lambda: AsrConfig(
            model_id="iic/SenseVoiceSmall",
            model_dir="models/sensevoice_small_int8_bundle",
        )
    )
    postprocess: PostprocessConfig = Field(
        default_factory=lambda: PostprocessConfig(
            punctuation=PunctuationConfig(
                model_id="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
                model_dir="models/punc",
            )
        )
    )
    output: OutputConfig = Field(default_factory=OutputConfig)
    evaluation: EvaluationConfig = Field(
        default_factory=lambda: EvaluationConfig(cer_normalization=CerNormalizationConfig())
    )


def load_config(path: str | Path, *, base: Path | None = None) -> AppConfig:
    """Load and validate a YAML config file.

    Relative model_dir / hotwords_file paths are resolved against the config file's parent directory.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    base_dir = base or p.parent
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    for section in ("vad", "speaker", "asr"):
        sec = raw.get(section)
        if isinstance(sec, dict) and "model_dir" in sec:
            sec["model_dir"] = str(resolve_path(sec["model_dir"], base_dir))
    if "asr" in raw and isinstance(raw["asr"], dict) and raw["asr"].get("hotwords_file"):
        raw["asr"]["hotwords_file"] = str(resolve_path(raw["asr"]["hotwords_file"], base_dir))
    if "postprocess" in raw and isinstance(raw["postprocess"], dict):
        punc = raw["postprocess"].get("punctuation")
        if isinstance(punc, dict) and "model_dir" in punc:
            punc["model_dir"] = str(resolve_path(punc["model_dir"], base_dir))

    cfg = AppConfig(**raw)
    return cfg
