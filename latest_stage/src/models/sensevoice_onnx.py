"""Standalone ONNX SenseVoiceSmall frontend and greedy CTC decoder."""

from __future__ import annotations

from itertools import groupby
import json
from pathlib import Path
import re
from typing import Any, Protocol

import numpy as np


class OnnxFeatureExtractor(Protocol):
    def extract(self, waveform: np.ndarray) -> tuple[np.ndarray, int]: ...


def load_tokens(path: str | Path) -> list[str]:
    tokens = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"invalid SenseVoice token table: {path}")
    return [str(token) for token in tokens]


def load_cmvn(path: str | Path) -> np.ndarray:
    """Parse the Kaldi-style AddShift/Rescale matrix used by FunASR."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    shifts: list[str] | None = None
    scales: list[str] | None = None
    for index, line in enumerate(lines):
        fields = line.split()
        if fields[:1] == ["<AddShift>"] and index + 1 < len(lines):
            next_fields = lines[index + 1].split()
            if next_fields[:1] == ["<LearnRateCoef>"]:
                shifts = next_fields[3:-1]
        elif fields[:1] == ["<Rescale>"] and index + 1 < len(lines):
            next_fields = lines[index + 1].split()
            if next_fields[:1] == ["<LearnRateCoef>"]:
                scales = next_fields[3:-1]
    if shifts is None or scales is None:
        raise ValueError(f"missing AddShift or Rescale values in {path}")
    return np.asarray([shifts, scales], dtype=np.float32)


def apply_lfr(features: np.ndarray, *, window: int = 7, hop: int = 6) -> np.ndarray:
    """Reproduce WavFrontend's low-frame-rate stacking without FunASR imports."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("LFR input must be a frame x dimension matrix")
    original_frames = len(features)
    output_frames = int(np.ceil(original_frames / hop))
    left_padding = int((window - 1) // 2)
    padded = np.concatenate((np.repeat(features[:1], left_padding, axis=0), features), axis=0)
    required_frames = (output_frames - 1) * hop + window
    if len(padded) < required_frames:
        # The exact FunASR formula can divide by zero for short prefixes;
        # making the final sliding window complete is behavior-equivalent.
        extra = required_frames - len(padded)
        padded = np.concatenate((padded, np.repeat(padded[-1:], extra, axis=0)), axis=0)

    stride_bytes = padded.astype(np.float32, copy=False).strides
    view = np.lib.stride_tricks.as_strided(
        padded,
        shape=(output_frames, window * padded.shape[1]),
        strides=(hop * stride_bytes[0], stride_bytes[1]),
        writeable=False,
    )
    return np.ascontiguousarray(view, dtype=np.float32)


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length() if value > 1 else 1


def _mel_filterbanks(
    *,
    num_bins: int,
    fft_length: int,
    sample_rate: int,
    low_freq: float = 20.0,
    high_freq: float = 8000.0,
) -> np.ndarray:
    nyquist = sample_rate / 2.0
    high_freq = high_freq if high_freq > 0.0 else high_freq + nyquist
    mel = lambda freq: 1127.0 * np.log1p(freq / 700.0)
    inverse_mel = lambda value: 700.0 * np.expm1(value / 1127.0)
    fft_bin_width = sample_rate / fft_length
    mel_low = mel(low_freq)
    mel_high = mel(high_freq)
    delta = (mel_high - mel_low) / (num_bins + 1)
    bins = np.arange(num_bins, dtype=np.float32)[:, None]
    left = mel_low + bins * delta
    center = mel_low + (bins + 1.0) * delta
    right = mel_low + (bins + 2.0) * delta
    fft_mels = mel(fft_bin_width * np.arange(fft_length // 2, dtype=np.float32))[None, :]
    up = (fft_mels - left) / (center - left)
    down = (right - fft_mels) / (right - center)
    return np.maximum(0.0, np.minimum(up, down)).astype(np.float32)


def numpy_kaldi_fbank(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    dither: float = 0.0,
) -> np.ndarray:
    """NumPy implementation of the Kaldi fbank settings used by SenseVoice."""
    wave = np.asarray(waveform, dtype=np.float32)
    if wave.ndim != 1 or wave.size == 0:
        raise ValueError("waveform must be non-empty and mono")
    window_shift = max(1, int(sample_rate * frame_shift_ms / 1000.0))
    frame_length = min(frame_length_ms, len(wave) / sample_rate * 1000.0)
    window_size = max(2, int(sample_rate * frame_length / 1000.0))
    if len(wave) < window_size:
        return np.empty((0, num_mel_bins), dtype=np.float32)
    padded_size = _next_power_of_two(window_size)
    frames = 1 + (len(wave) - window_size) // window_shift
    strided = np.lib.stride_tricks.as_strided(
        wave,
        shape=(frames, window_size),
        strides=(wave.strides[0] * window_shift, wave.strides[0]),
        writeable=False,
    ).astype(np.float32, copy=True)
    if dither:
        rng = np.random.default_rng()
        strided += rng.normal(0.0, dither, strided.shape).astype(np.float32)
    strided -= np.mean(strided, axis=1, keepdims=True)
    previous = np.pad(strided, ((0, 0), (1, 0)), mode="edge")[:, :-1]
    strided -= 0.97 * previous
    positions = np.arange(window_size, dtype=np.float32)
    window = 0.54 - 0.46 * np.cos(2.0 * np.pi * positions / (window_size - 1))
    strided *= window.astype(np.float32)
    if padded_size != window_size:
        strided = np.pad(strided, ((0, 0), (0, padded_size - window_size)))
    spectrum = np.fft.rfft(strided, axis=1)
    power = np.square(np.abs(spectrum)).astype(np.float32)
    filters = _mel_filterbanks(
        num_bins=num_mel_bins, fft_length=padded_size, sample_rate=sample_rate
    )
    filters = np.pad(filters, ((0, 0), (0, 1)))
    mel = power @ filters.T
    epsilon = np.asarray(1.1920929e-7, dtype=np.float32)
    return np.log(np.maximum(mel, epsilon)).astype(np.float32)


def apply_cmvn(features: np.ndarray, cmvn: np.ndarray) -> np.ndarray:
    output = np.asarray(features, dtype=np.float32).copy()
    output += cmvn[0:1, : output.shape[1]]
    output *= cmvn[1:2, : output.shape[1]]
    return output.astype(np.float32, copy=False)


class KaldiLfrFeatures:
    """Compute the 560-dimensional features expected by the exported graph."""

    def __init__(
        self,
        cmvn_path: str | Path,
        *,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        lfr_window: int = 7,
        lfr_hop: int = 6,
        dither: float = 0.0,
    ) -> None:
        self.cmvn = load_cmvn(cmvn_path)
        self.sample_rate = int(sample_rate)
        self.num_mel_bins = int(num_mel_bins)
        self.frame_length_ms = float(frame_length_ms)
        self.frame_shift_ms = float(frame_shift_ms)
        self.lfr_window = int(lfr_window)
        self.lfr_hop = int(lfr_hop)
        self.dither = float(dither)

    def extract(self, waveform: np.ndarray) -> tuple[np.ndarray, int]:

        wave = np.asarray(waveform, dtype=np.float32)
        if wave.ndim != 1 or wave.size == 0:
            raise ValueError("waveform must be non-empty and mono")
        # FunASR's WavFrontend converts [-1, 1] float samples back to PCM scale.
        frame_length = min(self.frame_length_ms, len(wave) / self.sample_rate * 1000.0)
        fbank = numpy_kaldi_fbank(
            wave * (1 << 15),
            sample_rate=self.sample_rate,
            num_mel_bins=self.num_mel_bins,
            frame_length_ms=frame_length,
            frame_shift_ms=self.frame_shift_ms,
            dither=self.dither,
        )
        features = apply_lfr(fbank, window=self.lfr_window, hop=self.lfr_hop)
        features = apply_cmvn(features, self.cmvn)
        return features, len(features)


def ctc_collapse(token_ids: list[int] | np.ndarray, *, blank_id: int = 0) -> list[int]:
    return [
        int(token)
        for token, _group in groupby(int(value) for value in token_ids)
        if token != blank_id
    ]


def decode_token_ids(token_ids: list[int] | np.ndarray, tokens: list[str]) -> str:
    text = "".join(tokens[int(token)] for token in token_ids)
    text = text.replace("▁", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # The first four model tokens are language/event/task/textnorm tags.
    if text.startswith("<|"):
        text = text.rsplit(">", 1)[-1].strip()
    return text


class OnnxSenseVoiceAsr:
    """Prefix re-decoding backend implemented with ONNX Runtime."""

    def __init__(
        self,
        *,
        session: Any,
        feature_extractor: OnnxFeatureExtractor,
        tokens: list[str],
        decode_chunk: float = 0.50,
        sample_rate: int = 16000,
        language_id: int = 3,
        textnorm_id: int = 15,
    ) -> None:
        if decode_chunk <= 0:
            raise ValueError("decode_chunk must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.session = session
        self.feature_extractor = feature_extractor
        self.tokens = tokens
        self.decode_chunk = float(decode_chunk)
        self.sample_rate = int(sample_rate)
        self.language_id = int(language_id)
        self.textnorm_id = int(textnorm_id)
        self.reset()

    @classmethod
    def from_bundle(
        cls,
        model_dir: str | Path,
        *,
        decode_chunk: float = 0.50,
        sample_rate: int = 16000,
        intra_op_threads: int | None = None,
        warmup_seconds: float = 0.50,
        language_id: int = 3,
        textnorm_id: int = 15,
    ) -> "OnnxSenseVoiceAsr":
        import onnxruntime as ort

        root = Path(model_dir)
        options = ort.SessionOptions()
        if intra_op_threads is not None:
            if intra_op_threads <= 0:
                raise ValueError("intra_op_threads must be positive")
            options.intra_op_num_threads = intra_op_threads
        session = ort.InferenceSession(
            str(root / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        backend = cls(
            session=session,
            feature_extractor=KaldiLfrFeatures(root / "am.mvn", sample_rate=sample_rate),
            tokens=load_tokens(root / "tokens.json"),
            decode_chunk=decode_chunk,
            sample_rate=sample_rate,
            language_id=language_id,
            textnorm_id=textnorm_id,
        )
        backend.warmup(warmup_seconds)
        return backend

    def warmup(self, seconds: float = 0.50) -> None:
        if seconds <= 0:
            raise ValueError("warmup seconds must be positive")
        self.reset()
        self.buffer = np.zeros(max(1, int(round(seconds * self.sample_rate))), dtype=np.float32)
        self._decode()
        self.reset()

    def reset(self) -> None:
        self.buffer = np.empty(0, dtype=np.float32)
        self.text = ""

    def _decode(self) -> str:
        features, length = self.feature_extractor.extract(self.buffer)
        logits, output_lengths = self.session.run(
            None,
            {
                "speech": features[None, ...],
                "speech_lengths": np.asarray([length], dtype=np.int32),
                "language": np.asarray([self.language_id], dtype=np.int32),
                "textnorm": np.asarray([self.textnorm_id], dtype=np.int32),
            },
        )
        token_ids = ctc_collapse(
            np.asarray(logits[0, : int(output_lengths[0])]).argmax(axis=-1).tolist()
        )
        self.text = decode_token_ids(token_ids, self.tokens)
        return self.text

    def accept(self, chunk: np.ndarray, *, is_final: bool) -> str:
        self.buffer = np.concatenate((self.buffer, np.asarray(chunk, dtype=np.float32)))
        decode_samples = max(1, int(round(self.decode_chunk * self.sample_rate)))
        if is_final or len(self.buffer) >= decode_samples:
            return self._decode()
        return self.text
