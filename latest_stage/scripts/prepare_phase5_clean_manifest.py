"""Prepare clean AISHELL manifests for Phase 5 tuning feasibility checks."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.extract_aishell_speakers import load_tier_speakers
from src.data.aishell import AishellUtterance, read_aishell_transcript
from src.data.manifest import ManifestEntry, write_jsonl

SPEAKER_ID_RE = re.compile(r"^S\d{4}$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build speaker-disjoint clean manifests from a Phase 5 tier"
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "data/raw/aishell_phase5_debug",
    )
    parser.add_argument(
        "--speaker-plan",
        type=Path,
        default=PROJECT_ROOT / "data/processed/phase5_tuning/speaker_plan.json",
    )
    parser.add_argument("--tier", default="debug", choices=("debug", "full"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data/processed/phase5_tuning/clean_debug",
    )
    parser.add_argument("--validation-speakers", type=int, default=4)
    parser.add_argument("--smoke-speakers", type=int, default=8)
    parser.add_argument("--smoke-utterances-per-speaker", type=int, default=8)
    parser.add_argument("--seed", type=int, default=52)
    return parser


def _reserved_speakers(plan: dict[str, Any]) -> set[str]:
    reserved = plan.get("reserved_speakers", {})
    if not isinstance(reserved, dict):
        raise ValueError("speaker plan reserved_speakers must be an object")
    speakers = {speaker for values in reserved.values() for speaker in values}
    if any(not isinstance(item, str) or not SPEAKER_ID_RE.fullmatch(item) for item in speakers):
        raise ValueError("speaker plan contains an invalid reserved speaker ID")
    return speakers


def _split_speakers(speakers: list[str], validation_count: int) -> tuple[list[str], list[str]]:
    if validation_count <= 0 or validation_count >= len(speakers):
        raise ValueError("validation speaker count must leave at least one training speaker")
    return speakers[:-validation_count], speakers[-validation_count:]


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getnchannels() != 1:
                raise ValueError(f"AISHELL WAV is not mono: {path}")
            if audio.getframerate() != 16000:
                raise ValueError(f"AISHELL WAV is not 16kHz: {path}")
            return audio.getnframes() / audio.getframerate()
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"cannot read AISHELL WAV: {path}") from exc


def _entry(utterance: AishellUtterance, duration: float) -> ManifestEntry:
    return ManifestEntry(
        id=utterance.utterance_id,
        audio_path=str(utterance.audio_path.resolve()),
        sample_rate=16000,
        duration=round(duration, 6),
        extras={
            "speaker": utterance.speaker,
            "text": utterance.text,
            "condition": "clean",
        },
    )


def _write_manifest(
    path: Path,
    utterances: list[AishellUtterance],
    durations: dict[str, float],
) -> int:
    entries = [_entry(item, durations[item.utterance_id]) for item in utterances]
    write_jsonl(path, entries)
    return len(entries)


def _summary_split(
    utterances: list[AishellUtterance],
    durations: dict[str, float],
) -> dict[str, Any]:
    speaker_counts = Counter(item.speaker for item in utterances)
    total_duration = sum(durations[item.utterance_id] for item in utterances)
    return {
        "utterances": len(utterances),
        "speaker_count": len(speaker_counts),
        "speaker_counts": dict(sorted(speaker_counts.items())),
        "duration_hours": round(total_duration / 3600, 4),
    }


def _smoke_subset(
    by_speaker: dict[str, list[AishellUtterance]],
    *,
    speakers: list[str],
    utterances_per_speaker: int,
) -> list[AishellUtterance]:
    subset: list[AishellUtterance] = []
    for speaker in speakers:
        candidates = by_speaker[speaker]
        if len(candidates) < utterances_per_speaker:
            raise ValueError(
                f"smoke speaker has too few utterances: {speaker} "
                f"({len(candidates)} < {utterances_per_speaker})"
            )
        subset.extend(candidates[:utterances_per_speaker])
    return sorted(subset, key=lambda item: item.utterance_id)


def _validate_source(raw_root: Path, selected: list[str]) -> None:
    source_path = raw_root / "source.json"
    if not source_path.is_file():
        raise SystemExit(f"extraction metadata not found: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    extracted = [str(item) for item in source.get("selected_speakers", [])]
    if set(extracted) != set(selected):
        raise SystemExit(
            "extracted speakers do not match the requested Phase 5 tier; "
            "rerun extraction with the same speaker plan"
        )

    audio_root = raw_root / "data_aishell" / "wav"
    actual_speakers = {path.name for path in audio_root.iterdir() if path.is_dir()}
    if actual_speakers != set(selected):
        missing = sorted(set(selected) - actual_speakers)
        extra = sorted(actual_speakers - set(selected))
        raise SystemExit(f"speaker directory mismatch: missing={missing} extra={extra}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.smoke_speakers <= 0 or args.smoke_utterances_per_speaker <= 0:
        raise SystemExit("smoke speaker and utterance counts must be positive")

    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    speaker_plan_path = args.speaker_plan.resolve()
    if not speaker_plan_path.is_file():
        raise SystemExit(f"speaker plan not found: {speaker_plan_path}")
    if not raw_root.is_dir():
        raise SystemExit(f"extracted tier not found: {raw_root}")
    if output_root.exists():
        raise SystemExit(f"output root already exists: {output_root}")
    staging = output_root.with_name(f"{output_root.name}.partial")
    if staging.exists():
        raise SystemExit(f"incomplete staging directory exists: {staging}")

    plan = json.loads(speaker_plan_path.read_text(encoding="utf-8"))
    selected = load_tier_speakers(speaker_plan_path, args.tier)
    reserved = _reserved_speakers(plan)
    reserved_overlap = sorted(reserved & set(selected))
    if reserved_overlap:
        raise SystemExit(f"reserved evaluation speakers found in tuning tier: {reserved_overlap}")
    _validate_source(raw_root, selected)

    train_speakers, validation_speakers = _split_speakers(selected, args.validation_speakers)
    if args.smoke_speakers > len(train_speakers):
        raise SystemExit("smoke speaker count cannot exceed training speakers")

    transcript = raw_root / "data_aishell" / "transcript" / "aishell_transcript_v0.8.txt"
    audio_root = raw_root / "data_aishell" / "wav"
    utterances = read_aishell_transcript(transcript, audio_root)
    selected_set = set(selected)
    utterances = [item for item in utterances if item.speaker in selected_set]
    by_speaker: dict[str, list[AishellUtterance]] = {speaker: [] for speaker in selected}
    for utterance in utterances:
        by_speaker[utterance.speaker].append(utterance)

    empty = [speaker for speaker, items in by_speaker.items() if not items]
    if empty:
        raise SystemExit(f"tier speakers have no transcribed WAV files: {empty}")

    wav_paths = sorted(audio_root.rglob("*.wav"))
    matched_ids = {item.utterance_id for item in utterances}
    unmatched_wavs = [path for path in wav_paths if path.stem.upper() not in matched_ids]
    durations = {item.utterance_id: _wav_duration(item.audio_path) for item in utterances}

    train_utterances = sorted(
        (item for speaker in train_speakers for item in by_speaker[speaker]),
        key=lambda item: item.utterance_id,
    )
    validation_utterances = sorted(
        (item for speaker in validation_speakers for item in by_speaker[speaker]),
        key=lambda item: item.utterance_id,
    )
    smoke_utterances = _smoke_subset(
        by_speaker,
        speakers=train_speakers[: args.smoke_speakers],
        utterances_per_speaker=args.smoke_utterances_per_speaker,
    )

    train_count = _write_manifest(staging / "train.jsonl", train_utterances, durations)
    validation_count = _write_manifest(
        staging / "validation.jsonl", validation_utterances, durations
    )
    smoke_count = _write_manifest(staging / "smoke_train.jsonl", smoke_utterances, durations)

    reserved_plan_speakers = sorted(reserved)
    summary = {
        "schema_version": 1,
        "phase": "5.1",
        "tier": args.tier,
        "condition": "clean",
        "seed": args.seed,
        "speaker_plan": str(speaker_plan_path),
        "raw_root": str(raw_root),
        "selected_speakers": selected,
        "reserved_speakers": reserved_plan_speakers,
        "reserved_overlap": reserved_overlap,
        "splits": {
            "train": {
                "speakers": train_speakers,
                **_summary_split(train_utterances, durations),
            },
            "validation": {
                "speakers": validation_speakers,
                **_summary_split(validation_utterances, durations),
            },
            "smoke_train": {
                "speakers": train_speakers[: args.smoke_speakers],
                **_summary_split(smoke_utterances, durations),
            },
        },
        "wav_without_transcript": len(unmatched_wavs),
        "manifests": {
            "train": str((output_root / "train.jsonl").resolve()),
            "validation": str((output_root / "validation.jsonl").resolve()),
            "smoke_train": str((output_root / "smoke_train.jsonl").resolve()),
        },
    }
    (staging / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (staging / "recipe.yaml").write_text(
        yaml.safe_dump(
            {
                "phase": "5.1",
                "tier": args.tier,
                "condition": "clean",
                "train_manifest": "train.jsonl",
                "validation_manifest": "validation.jsonl",
                "smoke_manifest": "smoke_train.jsonl",
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(output_root))

    print(f"prepared={output_root / 'summary.json'}")
    print(
        f"tier={args.tier} train={train_count} validation={validation_count} "
        f"smoke={smoke_count}"
    )
    print(f"reserved_overlap={len(reserved_overlap)}")


if __name__ == "__main__":
    main()
