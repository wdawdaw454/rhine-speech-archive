from __future__ import annotations

import numpy as np
import pytest

from scripts.export_sensevoice_onnx import _validate
from scripts.validate_sensevoice_onnx import _visible_text
from src.models.sensevoice_onnx import (
    KaldiLfrFeatures,
    OnnxSenseVoiceAsr,
    apply_cmvn,
    apply_lfr,
    ctc_collapse,
    decode_token_ids,
    load_cmvn,
)


class FakeFeatures:
    def __init__(self) -> None:
        self.waveforms = []

    def extract(self, waveform):
        self.waveforms.append(np.asarray(waveform))
        return np.zeros((2, 3), dtype=np.float32), 2


class FakeSession:
    def __init__(self):
        self.inputs = []

    def run(self, _output_names, feed):
        self.inputs.append(feed)
        logits = np.full((1, 2, 8), -10.0, dtype=np.float32)
        logits[0, 0, 1] = 10.0
        logits[0, 1, 2] = 10.0
        return logits, np.asarray([2], dtype=np.int64)


def test_ctc_collapse_removes_repeats_and_blank():
    assert ctc_collapse([0, 1, 1, 0, 2, 2, 1]) == [1, 2, 1]


def test_decode_token_ids_strips_model_tags_and_handles_word_marker():
    tokens = [
        "<blank>",
        "<|zh|>",
        "<|NEUTRAL|>",
        "<|Speech|>",
        "<|woitn|>",
        "你",
        "▁Cod",
        "ex",
    ]
    assert decode_token_ids([1, 2, 3, 4, 5, 6, 7], tokens) == "你 Codex"


def test_apply_lfr_matches_funasr_padding_behavior():
    features = np.arange(9, dtype=np.float32).reshape(9, 1)
    output = apply_lfr(features, window=5, hop=2)
    assert output.shape == (5, 5)
    np.testing.assert_allclose(output[0], [0, 0, 0, 1, 2])
    np.testing.assert_allclose(output[1], [0, 1, 2, 3, 4])
    np.testing.assert_allclose(output[2], [2, 3, 4, 5, 6])
    np.testing.assert_allclose(output[3], [4, 5, 6, 7, 8])
    np.testing.assert_allclose(output[4], [6, 7, 8, 8, 8])


def test_apply_lfr_handles_short_prefix_boundaries():
    features = np.arange(150, dtype=np.float32).reshape(50, 3)
    output = apply_lfr(features)
    assert output.shape == (9, 21)
    np.testing.assert_allclose(output[-1, -6:], np.tile([147, 148, 149], 2))


def test_load_and_apply_cmvn(tmp_path):

    path = tmp_path / "am.mvn"
    path.write_text(
        "<Nnet>\n<AddShift> 2 2\n<LearnRateCoef> 0 [ -1 2 ]\n"
        "<Rescale> 2 2\n<LearnRateCoef> 0 [ 2 3 ]\n</Nnet>\n",
        encoding="utf-8",
    )
    cmvn = load_cmvn(path)
    np.testing.assert_allclose(cmvn, [[-1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_allclose(apply_cmvn(np.array([[1.0, 2.0]]), cmvn), [[0.0, 12.0]])


def test_onnx_backend_uses_decode_cadence_and_prefix_buffer():
    features = FakeFeatures()
    session = FakeSession()
    backend = OnnxSenseVoiceAsr(
        session=session,
        feature_extractor=features,
        tokens=["<blank>", "你", "好"],
        decode_chunk=0.50,
    )
    chunk = np.ones(1600, dtype=np.float32)
    for _ in range(4):
        assert backend.accept(chunk, is_final=False) == ""
    assert len(features.waveforms) == 0
    assert backend.accept(chunk, is_final=False) == "你好"
    assert len(features.waveforms) == 1
    assert features.waveforms[0].shape == (1600 * 5,)
    np.testing.assert_allclose(
        session.inputs[-1]["speech_lengths"], np.asarray([2], dtype=np.int32)
    )


def test_onnx_backend_passes_enabled_itn_query():
    session = FakeSession()
    backend = OnnxSenseVoiceAsr(
        session=session,
        feature_extractor=FakeFeatures(),
        tokens=["<blank>", "你", "好"],
        textnorm_id=14,
    )

    backend.accept(np.ones(1600, dtype=np.float32), is_final=True)

    np.testing.assert_array_equal(session.inputs[-1]["textnorm"], np.asarray([14]))


def test_onnx_backend_warmup_runs_once_and_resets_buffer():
    features = FakeFeatures()
    backend = OnnxSenseVoiceAsr(
        session=FakeSession(),
        feature_extractor=features,
        tokens=["<blank>", "你", "好"],
    )
    backend.warmup(0.10)
    assert len(features.waveforms) == 1
    assert backend.text == ""
    assert backend.buffer.size == 0


def test_kaldi_feature_constructor_validates_cmvn(tmp_path):

    path = tmp_path / "am.mvn"
    path.write_text("<Nnet>\n</Nnet>\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing AddShift"):
        KaldiLfrFeatures(path)


def test_export_validation_rejects_missing_and_existing_paths(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.yaml").write_text("x", encoding="utf-8")
    (source / "model.pt").write_text("x", encoding="utf-8")
    (source / "tokens.json").write_text("[]", encoding="utf-8")
    (source / "am.mvn").write_text("x", encoding="utf-8")
    output = tmp_path / "output"
    _validate(source, output)
    output.mkdir()
    with pytest.raises(SystemExit):
        _validate(source, output)


def test_visible_text_strips_prefix_tags():
    assert _visible_text("<|zh|><|NEUTRAL|><|Speech|><|woitn|>你好") == "你好"
    assert _visible_text("你好") == "你好"
