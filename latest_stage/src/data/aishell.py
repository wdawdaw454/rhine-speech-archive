"""Helpers for parsing AISHELL-1 subset and full-release metadata."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AishellUtterance:
    utterance_id: str
    speaker: str
    split: str
    audio_path: Path
    text: str


def _audio_index(raw_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in raw_root.rglob("*.wav"):
        key = path.relative_to(raw_root).as_posix().lower()
        if key in index:
            raise ValueError(f"duplicate AISHELL audio key: {key}")
        index[key] = path
    return index


def read_aishell_csv(csv_path: Path, raw_root: Path) -> list[AishellUtterance]:
    audio_index = _audio_index(raw_root)
    utterances: list[AishellUtterance] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_path = row.get("Audio:FILE", "")
            text = "".join(row.get("Text:LABEL", "").split())
            match = re.search(r"/(s\d{4})/", raw_path + "/", re.IGNORECASE)
            split_match = re.search(r"/wav/([^/]+)/", raw_path + "/", re.IGNORECASE)
            name_match = re.search(r"(bac009s\d{4}w\d{4})\.wav$", raw_path, re.IGNORECASE)
            if not raw_path or not text or not match or not split_match or not name_match:
                raise ValueError(f"invalid AISHELL CSV row in {csv_path}: {row}")
            relative = raw_path.split("/", 1)[1] if "/" in raw_path else raw_path
            audio_path = audio_index.get(relative.lower())
            if audio_path is None:
                raise FileNotFoundError(f"AISHELL audio not found for CSV row: {raw_path}")
            utterance_id = name_match.group(1).upper()
            utterances.append(
                AishellUtterance(
                    utterance_id=utterance_id,
                    speaker=match.group(1).upper(),
                    split=split_match.group(1).lower(),
                    audio_path=audio_path,
                    text=text,
                )
            )
    if not utterances:
        raise ValueError(f"AISHELL CSV has no utterances: {csv_path}")
    return sorted(utterances, key=lambda item: item.utterance_id)


def read_aishell_transcript(transcript_path: Path, audio_root: Path) -> list[AishellUtterance]:
    """Read full AISHELL-1 transcripts and match extracted WAV files.

    The public release contains a small number of WAV files without transcript lines.
    Such files are skipped; callers should validate per-speaker counts before use.
    """
    texts: dict[str, str] = {}
    with transcript_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.strip().split(None, 1)
            if not fields:
                continue
            utterance_id = fields[0].upper()
            if not re.fullmatch(r"BAC009S\d{4}W\d{4}", utterance_id):
                raise ValueError(
                    f"invalid AISHELL transcript ID at {transcript_path}:{line_number}: "
                    f"{fields[0]}"
                )
            if utterance_id in texts:
                raise ValueError(f"duplicate AISHELL transcript ID: {utterance_id}")
            text = "".join(fields[1].split()) if len(fields) == 2 else ""
            if not text:
                raise ValueError(f"empty AISHELL transcript text: {utterance_id}")
            texts[utterance_id] = text

    utterances: list[AishellUtterance] = []
    for wav_path in sorted(audio_root.rglob("*.wav")):
        utterance_id = wav_path.stem.upper()
        if utterance_id not in texts:
            continue
        speaker_match = re.search(r"S\d{4}", utterance_id)
        if speaker_match is None:
            raise ValueError(f"cannot infer speaker from utterance ID: {utterance_id}")
        utterances.append(
            AishellUtterance(
                utterance_id=utterance_id,
                speaker=speaker_match.group(0),
                split="full",
                audio_path=wav_path,
                text=texts[utterance_id],
            )
        )
    if not utterances:
        raise ValueError(f"no AISHELL WAV files found under: {audio_root}")
    return sorted(utterances, key=lambda item: item.utterance_id)
