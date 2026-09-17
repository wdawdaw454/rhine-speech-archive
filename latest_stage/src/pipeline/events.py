"""PipelineEvent dataclass and EventType enum."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    AUDIO_INFO = "audio_info"
    VAD_START = "vad_start"
    VAD_END = "vad_end"
    SPEAKER_EMBEDDING = "speaker_embedding"
    SPEAKER_DECISION = "speaker_decision"
    ASR_PARTIAL = "asr_partial"
    ASR_FINAL = "asr_final"
    CAPTION_COMMIT = "caption_commit"
    CAPTION_SUPPRESS = "caption_suppress"
    PIPELINE_ERROR = "pipeline_error"
    RUN_SUMMARY = "run_summary"


class CommitState(str, Enum):
    PROVISIONAL = "provisional"
    COMMITTED = "committed"
    SUPPRESSED = "suppressed"
    REJECTED_BY_VAD = "rejected_by_vad"
    LOW_CONFIDENCE = "low_confidence"


@dataclass
class PipelineEvent:
    type: EventType
    run_id: str
    audio_path: str | None = None
    segment_id: str | None = None
    start: float | None = None
    end: float | None = None
    text: str | None = None
    similarity: float | None = None
    speaker_decision: str | None = None
    commit_state: CommitState | None = None
    audio_time: float | None = None
    monotonic_time: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        d["type"] = self.type.value
        if d["commit_state"] is not None:
            cs = self.commit_state
            d["commit_state"] = cs.value if hasattr(cs, "value") else cs
        return json.dumps(d, ensure_ascii=False)