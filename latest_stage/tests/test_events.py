import json
import time

from src.pipeline.events import PipelineEvent, EventType


def test_event_to_json_roundtrip():
    ev = PipelineEvent(
        type=EventType.ASR_FINAL,
        run_id="run_test",
        audio_path="data/x.wav",
        segment_id="seg_001",
        start=1.0,
        end=2.0,
        text="hello",
        similarity=0.4,
        speaker_decision="target",
        commit_state="committed",
        audio_time=2.0,
        monotonic_time=time.time(),
        extras={"foo": "bar"},
    )
    raw = ev.to_json()
    parsed = json.loads(raw)
    assert parsed["type"] == "asr_final"
    assert parsed["text"] == "hello"
    assert parsed["extras"]["foo"] == "bar"


def test_event_type_values():
    assert EventType.AUDIO_INFO == "audio_info"
    assert EventType.VAD_START == "vad_start"
    assert EventType.VAD_END == "vad_end"
    assert EventType.SPEAKER_EMBEDDING == "speaker_embedding"
    assert EventType.SPEAKER_DECISION == "speaker_decision"
    assert EventType.ASR_PARTIAL == "asr_partial"
    assert EventType.ASR_FINAL == "asr_final"
    assert EventType.CAPTION_COMMIT == "caption_commit"
    assert EventType.CAPTION_SUPPRESS == "caption_suppress"
    assert EventType.PIPELINE_ERROR == "pipeline_error"
    assert EventType.RUN_SUMMARY == "run_summary"