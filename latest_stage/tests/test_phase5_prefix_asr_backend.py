from __future__ import annotations

import numpy as np
import pytest

import src.models.backends as backends
from src.models.backends import FunasrPrefixAsr


class FakeSenseVoiceModel:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] = {}

    def generate(self, input: np.ndarray, **kwargs: object) -> list[dict[str, str]]:
        self.last_kwargs = kwargs
        samples = len(input)
        text = ""
        if samples >= 19200:
            text = "你好"
        elif samples >= 9600:
            text = "你"
        return [{"text": f"<|zh|><|NEUTRAL|><|Speech|>{text}"}]


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> FunasrPrefixAsr:
    monkeypatch.setattr(
        backends, "_load_funasr_model", lambda model_dir, device: FakeSenseVoiceModel()
    )
    return FunasrPrefixAsr("/tmp/model", decode_chunk=0.60)


def test_prefix_asr_replaces_text_and_strips_model_tags(backend: FunasrPrefixAsr):
    chunk = np.zeros(3200, dtype=np.float32)

    assert backend.accept(chunk, is_final=False) == ""
    assert backend.accept(chunk, is_final=False) == ""
    assert backend.accept(chunk, is_final=False) == "你"
    backend.accept(chunk, is_final=False)
    backend.accept(chunk, is_final=False)
    assert backend.accept(chunk, is_final=False) == "你好"


def test_prefix_asr_decodes_short_final(backend: FunasrPrefixAsr):
    backend.accept(np.zeros(3200, dtype=np.float32), is_final=False)

    assert backend.accept(np.zeros(3200, dtype=np.float32), is_final=True) == ""


def test_prefix_asr_reset_clears_buffer(backend: FunasrPrefixAsr):
    backend.accept(np.zeros(9600, dtype=np.float32), is_final=False)
    assert backend.text == "你"

    backend.reset()

    assert backend.text == ""
    assert len(backend.buffer) == 0


def test_prefix_asr_can_enable_itn(monkeypatch: pytest.MonkeyPatch):
    model = FakeSenseVoiceModel()
    monkeypatch.setattr(backends, "_load_funasr_model", lambda model_dir, device: model)
    backend = FunasrPrefixAsr("/tmp/model", decode_chunk=0.10, use_itn=True)

    backend.accept(np.zeros(1600, dtype=np.float32), is_final=True)

    assert model.last_kwargs["use_itn"] is True
