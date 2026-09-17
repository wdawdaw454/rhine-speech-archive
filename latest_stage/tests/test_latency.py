import numpy as np

from src.evaluation.latency import percentile, latency_summary, rtf


def test_percentile_basic():
    vals = list(range(1, 101))
    p50 = percentile(vals, 50.0)
    p95 = percentile(vals, 95.0)
    p99 = percentile(vals, 99.0)
    assert abs(p50 - 50.5) < 1.0
    assert abs(p95 - 95.05) < 1.0
    assert abs(p99 - 99.01) < 1.0


def test_percentile_empty():
    assert percentile([], 50.0) == 0.0


def test_percentile_single_value():
    assert percentile([0.42], 50.0) == 0.42
    assert percentile([0.42], 99.0) == 0.42


def test_latency_summary_keys():
    s = latency_summary([0.1, 0.2, 0.3, 0.4])
    assert set(s.keys()) == {"mean", "p50", "p95", "p99", "max", "count"}
    assert s["count"] == 4
    assert s["max"] == 0.4


def test_latency_summary_empty_returns_zero():
    s = latency_summary([])
    assert s["count"] == 0
    assert s["mean"] == 0.0


def test_rtf_basic():
    assert rtf(processing_wall=2.0, audio_duration=4.0) == 0.5
    assert rtf(processing_wall=5.0, audio_duration=2.0) == 2.5