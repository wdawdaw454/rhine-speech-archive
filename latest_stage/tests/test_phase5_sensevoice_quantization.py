from __future__ import annotations

import pytest

from scripts.benchmark_sensevoice_onnx import _summarize
from scripts.quantize_sensevoice_onnx import _excluded_nodes, _validate


def _bundle(tmp_path, name="source"):
    root = tmp_path / name
    root.mkdir()
    for filename in ("model.onnx", "bundle.json", "tokens.json", "am.mvn"):
        (root / filename).write_text(filename, encoding="utf-8")
    return root


def test_quantizer_selects_funasr_excluded_nodes():
    names = ["encoder_output", "bias_encoder_1", "safe_matmul", "bias_decoder_2"]
    assert _excluded_nodes(names) == ["encoder_output", "bias_encoder_1", "bias_decoder_2"]


def test_quantizer_validates_bundle_paths(tmp_path):
    source = _bundle(tmp_path)
    output = tmp_path / "output"
    _validate(source, output)
    output.mkdir()
    with pytest.raises(SystemExit):
        _validate(source, output)

    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(SystemExit):
        _validate(missing, output)


def _row(index: int, *, errors: int = 0, chars: int = 10, audio=2.0, decode=0.1):
    return {
        "id": f"utt-{index}",
        "bundle_dir": "bundle",
        "reference": "r" * chars,
        "hypothesis": "h" * chars,
        "errors": errors,
        "reference_chars": chars,
        "cer": errors / max(1, chars),
        "audio_seconds": audio,
        "decode_seconds": decode,
    }


def test_benchmark_summary_aggregates_corpus_metrics():
    summary = _summarize(
        [_row(0), _row(1, errors=2, chars=20, audio=4.0, decode=0.3)],
        load_seconds=3.5,
    )
    assert summary["examples"] == 2
    assert summary["errors"] == 2
    assert summary["chars"] == 30
    assert summary["corpus_cer"] == pytest.approx(2 / 30)
    assert summary["exact"] == 1
    assert summary["audio_seconds"] == pytest.approx(6.0)
    assert summary["decode_seconds"] == pytest.approx(0.4)
    assert summary["compute_rtf"] == pytest.approx(0.4 / 6.0)
    assert summary["decode_p50"] == pytest.approx(0.1)
    assert summary["decode_p95"] == pytest.approx(0.3)
