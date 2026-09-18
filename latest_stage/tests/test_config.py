from pathlib import Path

import pytest

from src.config import (
    AppConfig,
    AudioConfig,
    VadConfig,
    SpeakerConfig,
    AsrConfig,
    PostprocessConfig,
    OutputConfig,
    EvaluationConfig,
    ProjectConfig,
    load_config,
)


def test_default_config_validates():
    cfg = AppConfig()
    assert cfg.project.sample_rate == 16000
    assert cfg.audio.chunk_size == 0.20


def test_load_config_unknown_field_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("project:\n  unknown_field: 1\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_config(p)


def test_load_config_resolves_relative_path(tmp_path: Path):
    cfg_text = """
project:
  sample_rate: 16000
  mono: true
  seed: 1
  device: cpu
audio:
  chunk_size: 0.20
vad:
  model_id: x
  model_dir: models/vad
  threshold: 0.5
  min_silence_duration: 0.4
  max_segment_length: 20
  speech_pad: 0.2
  chunk_size: 0.2
speaker:
  model_id: x
  model_dir: models/speaker
  min_segment_duration: 0.6
  embedding_window: 1.5
  embedding_hop: 0.5
  accept_threshold: null
  reject_threshold: null
  calibration_target_far: 0.01
  embedding_norm: l2
asr:
  backend: sensevoice_small
  model_id: x
  model_dir: models/sensevoice_small_int8_bundle
  chunk_size: 0.6
  hotwords_file: null
  min_avg_confidence: null
postprocess:
  punctuation:
    enabled: true
    model_id: x
    model_dir: models/punc
  remove_fillers: false
  normalize_width: true
  normalize_number: true
output:
  run_dir: outputs/runs
  save_jsonl: true
  save_transcript: true
  save_audio_debug: false
evaluation:
  cer_normalization:
    lowercase: true
    remove_punctuation: true
    remove_whitespace: true
    to_simplified_chinese: true
    normalize_numbers: true
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(cfg_text, encoding="utf-8")
    cfg = load_config(p)
    # relative paths are resolved against config file's parent dir
    assert Path(cfg.vad.model_dir).is_absolute()


def test_device_alias_auto_resolves():
    cfg = AppConfig()
    cfg.project.device = "auto"
    resolved = cfg.project.resolved_device()
    assert resolved == "cpu" or resolved.startswith("cuda:")
