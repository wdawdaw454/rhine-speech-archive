"""Local target-mode models and VAD-trimmed enrollment, without ASR loading."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..models.backends import FunasrSpeakerEncoder
from ..models.streaming_vad import FunasrStreamingVad
from ..models.storage import model_storage_root
from .enrollment import SpeakerProfile
from .verification import enroll_embedding


MODEL_ID = "iic/speech_campplus_sv_zh-cn_16k-common"


class TargetModels:
    def __init__(self, model_root: Path | None = None):
        root = model_root or model_storage_root(Path(__file__).resolve().parents[2])
        spk = root / "cam-plus"
        vad = root / "fsmn-vad"
        if not (spk / "campplus_cn_common.bin").is_file() or not (vad / "model.pt").is_file():
            raise FileNotFoundError("缺少 CAM++ / FSMN-VAD 本地模型，请先运行 scripts/prepare_target_models.py")
        self.encoder = FunasrSpeakerEncoder(spk, device="cpu")
        self.vad = FunasrStreamingVad(vad, device="cpu")

    def enroll(self, audio, name, sample_rate=16000):
        if not 3 <= len(audio) / sample_rate <= 30:
            raise ValueError("注册音频需为 3–30 秒，建议单人连续说话 10–20 秒")
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) >= 0.999:
            raise ValueError("注册音频无效或削波失真，请降低录音音量后重试")
        self.vad.reset()
        ranges, start = [], None
        block = int(0.2 * sample_rate)
        for offset in range(0, len(audio), block):
            for bound in self.vad.accept(audio[offset:offset + block], is_final=offset + block >= len(audio)):
                if bound.start is not None:
                    start = max(0, int(bound.start * sample_rate))
                if bound.end is not None and start is not None:
                    ranges.append(audio[start:min(len(audio), int(bound.end * sample_rate))])
                    start = None
        if start is not None:
            ranges.append(audio[start:])
        speech = np.concatenate(ranges) if ranges else np.empty(0, np.float32)
        if len(speech) < 3 * sample_rate or float(np.sqrt(np.mean(speech ** 2))) < 0.001:
            raise ValueError("有效语音不足 3 秒，请在安静环境中连续说话 10–20 秒")
        embeddings = [self.encoder.extract(speech[i:i + 3 * sample_rate], sample_rate)
                      for i in range(0, len(speech) - sample_rate + 1, 3 * sample_rate)]
        embedding = enroll_embedding(embeddings, norm="l2")
        if not np.isfinite(embedding).all():
            raise ValueError("无效声纹，请重新注册")
        return SpeakerProfile(name, MODEL_ID, len(embedding), embedding.tolist(), "l2",
                              len(embeddings), round(len(speech) / sample_rate, 2),
                              datetime.now(timezone.utc).isoformat())
