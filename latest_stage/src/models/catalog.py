"""ModelScope model catalog used by this project."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    directory: str
    description: str


MODEL_SPECS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            key="asr-sensevoice",
            model_id="iic/SenseVoiceSmall",
            directory="models/sensevoice_small",
            description="SenseVoiceSmall prefix-redecode fast ASR",
        ),
        ModelSpec(
            key="vad",
            model_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            directory="models/vad",
            description="FSMN streaming VAD",
        ),
        ModelSpec(
            key="speaker",
            model_id="iic/speech_campplus_sv_zh-cn_16k-common",
            directory="models/speaker",
            description="CAM++ speaker verification / embedding",
        ),
        ModelSpec(
            key="punc",
            model_id="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
            directory="models/punc",
            description="Chinese-English punctuation restoration",
        ),
    )
}
