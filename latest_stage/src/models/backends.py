"""FunASR runtime adapters for VAD, speaker embedding, ASR, and punctuation.

The rest of the pipeline depends only on the lightweight protocols defined here.
Keeping FunASR calls in this module makes the streaming logic testable without
downloading models and leaves a narrow place to handle FunASR API changes.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float


class VadBackend(Protocol):
    def detect(self, waveform: np.ndarray, sample_rate: int) -> list[SpeechSegment]: ...


class SpeakerEncoderBackend(Protocol):
    def extract(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray: ...


class StreamingAsrBackend(Protocol):
    def reset(self) -> None: ...

    def accept(self, chunk: np.ndarray, *, is_final: bool) -> str: ...


class PunctuationBackend(Protocol):
    def restore(self, text: str) -> str: ...


def _load_funasr_model(model_dir: str | Path, device: str = "cpu") -> Any:
    from funasr import AutoModel

    return AutoModel(model=str(model_dir), device=device, disable_update=True)


def _first_result(results: Any) -> Any:
    if isinstance(results, list) and results:
        return results[0]
    return results


def _as_float(value: Any) -> float:
    return float(value)


def _segment_bounds(item: Any) -> tuple[float, float]:
    if isinstance(item, dict):
        return _as_float(item["start"]), _as_float(item["end"])
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return _as_float(item[0]), _as_float(item[1])
    raise TypeError(f"unsupported VAD segment format: {item!r}")


class FunasrVad:
    """FSMN VAD adapter.

    FunASR commonly reports FSMN VAD boundaries in milliseconds. The adapter also
    accepts sample indices for compatibility with local model variants.
    """

    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        cache_dir = Path(model_dir).resolve().parent.parent / ".cache" / "jieba"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["JIEBA_CACHE_DIR"] = str(cache_dir)
        self.model = _load_funasr_model(model_dir, device)

    def detect(self, waveform: np.ndarray, sample_rate: int) -> list[SpeechSegment]:
        wav = np.asarray(waveform, dtype=np.float32)
        result = _first_result(self.model.generate(input=wav))
        raw_segments = result.get("value", []) if isinstance(result, dict) else []
        duration = len(wav) / sample_rate
        converted: list[tuple[float, float]] = []
        max_end = 0.0
        for item in raw_segments:
            start, end = _segment_bounds(item)
            max_end = max(max_end, start, end)
            converted.append((start, end))

        # Millisecond output is common. Sample output is accepted when treating
        # values as milliseconds would make the timestamps far exceed the audio.
        divide_by = 1000.0
        if duration > 0 and max_end / 1000.0 > duration * 1.5 and max_end <= len(wav) * 1.5:
            divide_by = 1.0

        segments = [
            SpeechSegment(start=max(0.0, s / divide_by), end=min(duration, e / divide_by))
            for s, e in converted
            if e > s
        ]
        return sorted(segments, key=lambda x: x.start)


class FunasrSpeakerEncoder:
    """CAM++ speaker embedding adapter."""

    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        self.model = _load_funasr_model(model_dir, device)

    def extract(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        del sample_rate  # FunASR's model expects 16 kHz; the pipeline enforces this.
        result = _first_result(self.model.generate(input=np.asarray(waveform, dtype=np.float32)))
        embedding: Any = None
        if isinstance(result, dict):
            embedding = result.get("embedding", result.get("spk_embedding"))
        if embedding is None and isinstance(result, list) and result:
            candidate = result[0]
            if isinstance(candidate, dict):
                embedding = candidate.get("embedding", candidate.get("spk_embedding"))
            else:
                embedding = candidate
        if embedding is None:
            raise RuntimeError(f"speaker model did not return an embedding: {result!r}")
        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().numpy()
        elif hasattr(embedding, "cpu") and not isinstance(embedding, np.ndarray):
            embedding = embedding.cpu()
        arr = np.asarray(embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            raise RuntimeError("speaker model returned an all-zero embedding")
        return arr / norm


class FunasrUniAsr:
    """UniASR 2-pass streaming adapter using the model's native chunk config."""

    def __init__(self, model_dir: str | Path, *, device: str = "cpu") -> None:
        self.model = _load_funasr_model(model_dir, device)
        self.reset()

    def reset(self) -> None:
        self.cache: dict[str, Any] = {}
        self.last_text = ""
        self.text = ""

    def accept(self, chunk: np.ndarray, *, is_final: bool) -> str:
        result = _first_result(
            self.model.generate(
                input=np.asarray(chunk, dtype=np.float32),
                cache=self.cache,
                is_final=is_final,
            )
        )
        text = ""
        if isinstance(result, dict):
            text = str(result.get("text", ""))
        elif result is not None:
            text = str(result)
        self.last_text = text.strip()
        self.text += self.last_text
        return self.text


class FunasrPrefixAsr:
    """Offline ASR adapter that re-decodes a growing short prefix."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        decode_chunk: float = 0.60,
        device: str = "cpu",
        sample_rate: int = 16000,
        use_itn: bool = False,
    ) -> None:
        if decode_chunk <= 0:
            raise ValueError("decode_chunk must be positive")
        self.model = _load_funasr_model(model_dir, device)
        self.sample_rate = int(sample_rate)
        self.decode_chunk = float(decode_chunk)
        self.use_itn = bool(use_itn)
        self.reset()

    def reset(self) -> None:
        self.buffer = np.empty(0, dtype=np.float32)
        self.text = ""

    def _decode(self) -> str:
        result = _first_result(
            self.model.generate(input=self.buffer, cache={}, language="zh", use_itn=self.use_itn)
        )
        text = ""
        if isinstance(result, dict):
            text = str(result.get("text", ""))
        elif result is not None:
            text = str(result)
        text = text.strip()
        if text.startswith("<|"):
            text = text.rsplit(">", 1)[-1].strip()
        self.text = text
        return self.text

    def accept(self, chunk: np.ndarray, *, is_final: bool) -> str:
        self.buffer = np.concatenate((self.buffer, np.asarray(chunk, dtype=np.float32)))
        decode_samples = max(1, int(round(self.decode_chunk * self.sample_rate)))
        if is_final or len(self.buffer) >= decode_samples:
            return self._decode()
        return self.text


class FunasrPunctuator:
    """CT-Transformer punctuation adapter."""

    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        self.model = _load_funasr_model(model_dir, device)

    def restore(self, text: str) -> str:
        if not text.strip():
            return text
        result = _first_result(self.model.generate(input=text))
        if isinstance(result, dict):
            return str(result.get("text", text)).strip()
        return text
