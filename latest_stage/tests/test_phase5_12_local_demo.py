from __future__ import annotations

import numpy as np
import pytest

from scripts.run_local_dictation_demo import _speed_metrics, _text_metrics, build_parser
from src.models.sensevoice_onnx import numpy_kaldi_fbank


def _partial(index: int, text: str) -> dict:
    return {
        "type": "asr_partial",
        "text": text,
        "monotonic_time": 0.5 + index * 0.5,
        "audio_time": 0.7 + index * 0.5,
    }


def test_demo_text_metrics_normalizes_and_computes_cer():
    metrics = _text_metrics("房地产，测试！", "房地产测试")
    assert metrics["cer"] == 0.0
    assert metrics["reference"] == "房地产测试"
    assert metrics["hypothesis"] == "房地产测试"

    metrics = _text_metrics("一二三四五", "一三四五")
    assert metrics["errors"] == 1
    assert metrics["cer"] == pytest.approx(0.2)


def test_demo_speed_metrics_reports_first_partial_and_intervals():
    events = [
        _partial(0, "你"),
        _partial(1, "你好"),
        _partial(3, "你好吗"),
    ]
    stats = {"backend_seconds": 0.5, "accepted_seconds": 4.0}
    metrics = _speed_metrics(events, stats)
    assert metrics["first_partial_wall"] == 0.5
    assert metrics["partial_count"] == 3
    assert metrics["partial_interval_p50"] == 0.5
    assert metrics["partial_interval_p95"] == 1.0
    assert metrics["backend_rtf"] == 0.125


def test_numpy_kaldi_fbank_has_expected_shape_and_range():
    waveform = 0.1 * np.sin(np.linspace(0.0, 2 * np.pi * 100.0, 16000, dtype=np.float32))
    features = numpy_kaldi_fbank(waveform)
    assert features.shape == (98, 80)
    assert np.isfinite(features).all()
    assert features.max() < 0.0


def test_local_demo_enables_itn_by_default_and_allows_opt_out():
    parser = build_parser()

    assert parser.parse_args(["--microphone"]).itn is True
    assert parser.parse_args(["--microphone", "--no-itn"]).itn is False
