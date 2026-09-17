"""Model catalog and downloader input validation tests."""

from __future__ import annotations

import pytest

from src.models.catalog import MODEL_SPECS
from src.models.downloader import download_models


def test_catalog_contains_phase1_models():
    assert set(MODEL_SPECS) == {
        "asr-sensevoice",
        "vad",
        "speaker",
        "punc",
    }


def test_download_models_rejects_unknown():
    with pytest.raises(ValueError):
        download_models(["not-a-model"], "/tmp/unused")
