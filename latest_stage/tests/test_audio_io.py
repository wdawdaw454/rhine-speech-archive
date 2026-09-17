import numpy as np
import pytest
import soundfile as sf

from src.audio.io import load_audio, AudioLoadError


def test_load_wav_returns_mono_16k(tmp_path):
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    wav = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, np.stack([wav, wav], axis=1), sr, subtype="FLOAT")
    audio, out_sr = load_audio(str(path), target_sr=16000)
    assert out_sr == 16000
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert len(audio) == int(round(1.0 * 16000))


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(AudioLoadError):
        load_audio(str(tmp_path / "nope.wav"))