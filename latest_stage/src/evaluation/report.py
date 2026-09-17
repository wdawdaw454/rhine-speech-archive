"""Evaluation report built from pipeline events and utterance references."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data.normalize import normalize_chinese_text
from ..evaluation.cer import cer, char_edit_distance
from ..evaluation.latency import latency_summary
from ..pipeline.events import EventType, PipelineEvent


@dataclass(frozen=True)
class ReferenceUtterance:
    utterance_id: str
    speaker: str
    start: float
    end: float
    text: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceUtterance":
        required = ("speaker", "start", "end", "text")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"reference is missing fields: {', '.join(missing)}")
        start, end = float(data["start"]), float(data["end"])
        if end <= start:
            raise ValueError(f"reference has non-positive duration: {data.get('id')}")
        return cls(
            utterance_id=str(data.get("id", data.get("utterance_id", ""))),
            speaker=str(data["speaker"]),
            start=start,
            end=end,
            text=str(data["text"]),
        )


def read_references(path: str | Path) -> list[ReferenceUtterance]:
    references: list[ReferenceUtterance] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                references.append(ReferenceUtterance.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid reference at {path}:{line_number}: {exc}") from exc
    if not references:
        raise ValueError(f"reference file is empty: {path}")
    return references


def _normalized_text(text: str) -> str:
    return normalize_chinese_text(text)


def _pair_score(
    event: PipelineEvent,
    reference: ReferenceUtterance,
    *,
    min_overlap: float,
) -> float | None:
    if event.start is None or event.end is None:
        return None
    overlap = min(event.end, reference.end) - max(event.start, reference.start)
    if overlap < min_overlap:
        return None
    ref_text = _normalized_text(reference.text)
    hyp_text = _normalized_text(event.text or "")
    text_similarity = 1.0 - cer(ref_text, hyp_text) if ref_text else 0.0
    return overlap + text_similarity


def match_events_to_references(
    events: list[PipelineEvent],
    references: list[ReferenceUtterance],
    *,
    min_overlap: float = 0.05,
) -> dict[str, str]:
    """Greedily match caption commits to references by time and text overlap."""

    commits = [event for event in events if event.type == EventType.CAPTION_COMMIT]
    candidates: list[tuple[float, int, int]] = []
    for event_index, event in enumerate(commits):
        for reference_index, reference in enumerate(references):
            score = _pair_score(event, reference, min_overlap=min_overlap)
            if score is not None:
                candidates.append((score, event_index, reference_index))

    matched_events: set[int] = set()
    matched_references: set[int] = set()
    event_to_reference: dict[str, str] = {}
    for _, event_index, reference_index in sorted(candidates, reverse=True):
        if event_index in matched_events or reference_index in matched_references:
            continue
        event = commits[event_index]
        reference = references[reference_index]
        if not event.segment_id:
            raise ValueError("caption_commit event is missing segment_id")
        event_to_reference[event.segment_id] = reference.utterance_id
        matched_events.add(event_index)
        matched_references.add(reference_index)
    return event_to_reference


def _similarity_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _target_cer(
    *,
    commits: list[PipelineEvent],
    references: list[ReferenceUtterance],
    event_to_reference: dict[str, str],
    target_speaker: str,
) -> tuple[float, int, int]:
    target_refs = [reference for reference in references if reference.speaker == target_speaker]
    ref_by_id = {reference.utterance_id: reference for reference in target_refs}
    matched_ref_ids = {
        reference_id for reference_id in event_to_reference.values() if reference_id in ref_by_id
    }

    errors = 0
    denominator = 0
    for reference in target_refs:
        ref_text = _normalized_text(reference.text)
        denominator += len(ref_text)
        if reference.utterance_id not in matched_ref_ids:
            errors += len(ref_text)

    for event in commits:
        reference_id = event_to_reference.get(event.segment_id or "")
        if reference_id not in matched_ref_ids:
            continue
        reference = ref_by_id[reference_id]
        ref_text = _normalized_text(reference.text)
        hyp_text = _normalized_text(event.text or "")
        errors += char_edit_distance(ref_text, hyp_text)

    # Insertions from unmatched commits also count against corpus CER. Length is
    # used because there is no reference to align against.
    matched_segment_ids = set(event_to_reference)
    for event in commits:
        if event.segment_id not in matched_segment_ids:
            errors += len(_normalized_text(event.text or ""))

    if denominator == 0:
        value = 1.0 if errors else 0.0
    else:
        value = min(1.0, errors / denominator)
    return value, errors, denominator


def _latency_report(events: list[PipelineEvent]) -> dict[str, Any]:
    commits = [event for event in events if event.type == EventType.CAPTION_COMMIT]
    partials_by_segment: dict[str, PipelineEvent] = {}
    target_partials_by_segment: dict[str, PipelineEvent] = {}
    finals = [event for event in events if event.type == EventType.ASR_FINAL]
    for event in events:
        if event.type != EventType.ASR_PARTIAL or not event.segment_id:
            continue
        current = partials_by_segment.get(event.segment_id)
        if current is None or (event.monotonic_time or 0) < (current.monotonic_time or 0):
            partials_by_segment[event.segment_id] = event
        if event.speaker_decision == "target":
            current_target = target_partials_by_segment.get(event.segment_id)
            if current_target is None or (
                (event.monotonic_time or 0) < (current_target.monotonic_time or 0)
            ):
                target_partials_by_segment[event.segment_id] = event
    final_by_segment = {
        event.segment_id: event
        for event in events
        if event.type == EventType.ASR_FINAL and event.segment_id is not None
    }

    end_to_end: list[float] = []
    punctuation: list[float] = []
    for event in commits:
        if event.monotonic_time is None or event.audio_time is None:
            continue
        end_to_end.append(float(event.monotonic_time - event.audio_time))
        final = final_by_segment.get(event.segment_id or "")
        if final is not None and final.monotonic_time is not None:
            punctuation.append(float(event.monotonic_time - final.monotonic_time))

    realtime_valid = bool(end_to_end) and all(value >= -1e-3 for value in end_to_end)
    report: dict[str, Any] = {
        "caption_end_to_end": latency_summary(end_to_end),
        "first_partial": latency_summary(
            [
                float(event.monotonic_time - event.audio_time)
                for event in partials_by_segment.values()
                if event.monotonic_time is not None and event.audio_time is not None
            ]
        ),
        "first_target_partial": latency_summary(
            [
                float(event.monotonic_time - event.audio_time)
                for event in target_partials_by_segment.values()
                if event.monotonic_time is not None and event.audio_time is not None
            ]
        ),
        "asr_final": latency_summary(
            [
                float(event.monotonic_time - event.audio_time)
                for event in finals
                if event.monotonic_time is not None and event.audio_time is not None
            ]
        ),
        "caption_after_asr_final": latency_summary(punctuation),
    }
    if not realtime_valid:
        report["note"] = (
            "Negative or empty end-to-end values mean this run was offline-paced; "
            "use --realtime with an event-carrying online pipeline for user-facing latency."
        )
    return report


def build_evaluation_report(
    events: list[PipelineEvent],
    references: list[ReferenceUtterance],
    *,
    target_speaker: str,
    min_overlap: float = 0.05,
) -> dict[str, Any]:
    if min_overlap < 0:
        raise ValueError("min_overlap must be non-negative")
    if not any(reference.speaker == target_speaker for reference in references):
        raise ValueError(f"reference set has no target speaker: {target_speaker}")

    event_to_reference = match_events_to_references(
        events,
        references,
        min_overlap=min_overlap,
    )
    commits = [event for event in events if event.type == EventType.CAPTION_COMMIT]
    suppressions = [event for event in events if event.type == EventType.CAPTION_SUPPRESS]
    ref_by_id = {reference.utterance_id: reference for reference in references}

    target_reference_count = sum(reference.speaker == target_speaker for reference in references)
    non_target_reference_count = len(references) - target_reference_count
    overlaps_by_segment: dict[str, set[str]] = {}
    for event in commits:
        overlaps_by_segment[event.segment_id or ""] = {
            reference.utterance_id
            for reference in references
            if _pair_score(event, reference, min_overlap=min_overlap) is not None
        }
    covered_target_ids = {
        reference_id
        for reference_ids in overlaps_by_segment.values()
        for reference_id in reference_ids
        if ref_by_id[reference_id].speaker == target_speaker
    }
    committed_non_target_ids = {
        reference_id
        for reference_ids in overlaps_by_segment.values()
        for reference_id in reference_ids
        if ref_by_id[reference_id].speaker != target_speaker
    }
    clean_target_commits = [
        event
        for event in commits
        if any(
            ref_by_id[reference_id].speaker == target_speaker
            for reference_id in overlaps_by_segment[event.segment_id or ""]
        )
        and not any(
            ref_by_id[reference_id].speaker != target_speaker
            for reference_id in overlaps_by_segment[event.segment_id or ""]
        )
    ]
    unmatched_commits = [
        event for event in commits if not overlaps_by_segment[event.segment_id or ""]
    ]

    target_recall = (
        len(covered_target_ids) / target_reference_count if target_reference_count else 0.0
    )
    target_precision = len(clean_target_commits) / len(commits) if commits else 0.0
    non_target_leakage = (
        len(committed_non_target_ids) / non_target_reference_count
        if non_target_reference_count
        else 0.0
    )
    target_cer, target_errors, target_chars = _target_cer(
        commits=commits,
        references=references,
        event_to_reference=event_to_reference,
        target_speaker=target_speaker,
    )

    run_summary = next(
        (event.extras for event in reversed(events) if event.type == EventType.RUN_SUMMARY),
        {},
    )
    similarities = [
        float(event.similarity) for event in commits + suppressions if event.similarity is not None
    ]
    if not all(math.isfinite(value) for value in similarities):
        raise ValueError("events contain a non-finite speaker similarity")

    decision_counts: dict[str, int] = {}
    for event in suppressions:
        decision = event.speaker_decision or "unknown"
        decision_counts[decision] = decision_counts.get(decision, 0) + 1

    return {
        "target_speaker": target_speaker,
        "counts": {
            "references": len(references),
            "target_references": target_reference_count,
            "non_target_references": non_target_reference_count,
            "caption_commits": len(commits),
            "caption_suppressions": len(suppressions),
            "matched_caption_commits": len(event_to_reference),
            "covered_target_references": len(covered_target_ids),
            "clean_target_caption_commits": len(clean_target_commits),
            "committed_non_target_references": len(committed_non_target_ids),
        },
        "target_metrics": {
            "target_cer": target_cer,
            "target_character_errors": target_errors,
            "target_reference_characters": target_chars,
            "target_recall": target_recall,
            "target_precision": target_precision,
            "non_target_leakage_rate": non_target_leakage,
            "unmatched_caption_commits": len(unmatched_commits),
        },
        "latency": _latency_report(events),
        "rtf": float(run_summary.get("rtf", 0.0)),
        "similarity": _similarity_summary(similarities),
        "suppressed_decisions": decision_counts,
        "matching": event_to_reference,
    }
