from __future__ import annotations

from src.evaluation.report import (
    ReferenceUtterance,
    build_evaluation_report,
)
from src.pipeline.events import CommitState, EventType, PipelineEvent


def _commit(
    segment_id: str,
    start: float,
    end: float,
    text: str,
    monotonic_time: float,
) -> PipelineEvent:
    return PipelineEvent(
        type=EventType.CAPTION_COMMIT,
        run_id="run",
        segment_id=segment_id,
        start=start,
        end=end,
        text=text,
        similarity=0.8,
        speaker_decision="target",
        commit_state=CommitState.COMMITTED,
        audio_time=end,
        monotonic_time=monotonic_time,
    )


def test_report_computes_target_gate_and_cer():
    references = [
        ReferenceUtterance("target-ref", "speaker1", 0.0, 2.0, "你好世界"),
        ReferenceUtterance("other-ref", "speaker2", 3.0, 4.5, "不要转写"),
    ]
    final = PipelineEvent(
        type=EventType.ASR_FINAL,
        run_id="run",
        segment_id="seg_0000",
        start=0.1,
        end=2.1,
        text="你好视界",
        audio_time=2.1,
        monotonic_time=2.2,
    )
    commit = _commit("seg_0000", 0.1, 2.1, "你好视界", 2.3)
    pending_partial = PipelineEvent(
        type=EventType.ASR_PARTIAL,
        run_id="run",
        segment_id="seg_0000",
        start=0.1,
        end=2.1,
        text="你",
        speaker_decision="pending",
        commit_state=CommitState.PROVISIONAL,
        audio_time=2.1,
        monotonic_time=2.15,
    )
    partial = PipelineEvent(
        type=EventType.ASR_PARTIAL,
        run_id="run",
        segment_id="seg_0000",
        start=0.1,
        end=2.1,
        text="你好",
        speaker_decision="target",
        commit_state=CommitState.PROVISIONAL,
        audio_time=2.1,
        monotonic_time=2.25,
    )
    summary = PipelineEvent(
        type=EventType.RUN_SUMMARY,
        run_id="run",
        audio_time=4.5,
        monotonic_time=2.8,
        extras={"rtf": 0.2},
    )
    report = build_evaluation_report(
        [pending_partial, partial, final, commit, summary],
        references,
        target_speaker="speaker1",
    )

    metrics = report["target_metrics"]
    assert metrics["target_recall"] == 1.0
    assert metrics["target_precision"] == 1.0
    assert metrics["non_target_leakage_rate"] == 0.0
    assert metrics["target_cer"] == 0.25
    assert report["counts"]["covered_target_references"] == 1
    assert report["counts"]["clean_target_caption_commits"] == 1
    assert report["counts"]["committed_non_target_references"] == 0
    assert report["latency"]["caption_end_to_end"]["count"] == 1
    assert report["latency"]["first_partial"]["count"] == 1
    assert abs(report["latency"]["first_partial"]["p50"] - 0.05) < 1e-9
    assert report["latency"]["first_target_partial"]["count"] == 1
    assert abs(report["latency"]["first_target_partial"]["p50"] - 0.15) < 1e-9
    assert report["latency"]["asr_final"]["count"] == 1
    assert abs(report["latency"]["caption_end_to_end"]["p50"] - 0.2) < 1e-9
    assert report["matching"] == {"seg_0000": "target-ref"}


def test_report_detects_non_target_text_inside_merged_caption():
    references = [
        ReferenceUtterance("target-ref", "speaker1", 0.0, 2.0, "目标语音"),
        ReferenceUtterance("other-ref", "speaker2", 2.0, 4.0, "干扰语音"),
    ]
    merged = _commit("seg_merged", 0.0, 4.0, "目标语音干扰语音", 4.1)
    report = build_evaluation_report([merged], references, target_speaker="speaker1")
    metrics = report["target_metrics"]
    assert metrics["target_recall"] == 1.0
    assert metrics["target_precision"] == 0.0
    assert metrics["non_target_leakage_rate"] == 1.0


def test_report_counts_missed_target_as_cer_and_recall_error():
    references = [
        ReferenceUtterance("target-ref", "speaker1", 0.0, 2.0, "目标语音"),
        ReferenceUtterance("other-ref", "speaker2", 2.2, 3.2, "干扰语音"),
    ]
    leaked = _commit("seg_0001", 2.1, 3.3, "干扰语音", 3.4)
    report = build_evaluation_report([leaked], references, target_speaker="speaker1")

    metrics = report["target_metrics"]
    assert metrics["target_recall"] == 0.0
    assert metrics["target_precision"] == 0.0
    assert metrics["non_target_leakage_rate"] == 1.0
    assert metrics["target_cer"] == 1.0
    assert metrics["unmatched_caption_commits"] == 0
