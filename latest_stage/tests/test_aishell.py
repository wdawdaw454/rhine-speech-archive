from __future__ import annotations

from pathlib import Path

import json

from scripts.extract_aishell_speakers import load_tier_speakers, select_speakers
from src.data.aishell import read_aishell_csv, read_aishell_transcript


def test_read_aishell_csv_normalizes_text_and_path_case(tmp_path: Path):
    audio = tmp_path / "speech_asr_aishell_subset" / "wav" / "validation" / "S0724"
    audio.mkdir(parents=True)
    wav = audio / "BAC009S0724W0121.wav"
    wav.write_bytes(b"RIFF")
    csv_path = tmp_path / "validation.csv"
    csv_path.write_text(
        "Audio:FILE,Text:LABEL\n"
        "speech_asr_aishell_subset/wav/validation/S0724/BAC009S0724W0121.wav,"
        "广 州 市\n",
        encoding="utf-8",
    )

    utterances = read_aishell_csv(
        csv_path,
        tmp_path / "speech_asr_aishell_subset",
    )

    assert len(utterances) == 1
    assert utterances[0].utterance_id == "BAC009S0724W0121"
    assert utterances[0].speaker == "S0724"
    assert utterances[0].split == "validation"
    assert utterances[0].text == "广州市"
    assert utterances[0].audio_path == wav


def test_read_aishell_csv_rejects_missing_audio(tmp_path: Path):
    csv_path = tmp_path / "validation.csv"
    csv_path.write_text(
        "Audio:FILE,Text:LABEL\n"
        "speech_asr_aishell_subset/wav/validation/S0724/BAC009S0724W0121.wav,text\n",
        encoding="utf-8",
    )
    try:
        read_aishell_csv(csv_path, tmp_path / "speech_asr_aishell_subset")
    except FileNotFoundError as exc:
        assert "BAC009S0724W0121.wav" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_read_aishell_transcript_matches_extracted_audio(tmp_path: Path):
    audio = tmp_path / "wav" / "S0100"
    audio.mkdir(parents=True)
    wav = audio / "BAC009S0100W0122.wav"
    wav.write_bytes(b"RIFF")
    transcript = tmp_path / "aishell_transcript_v0.8.txt"
    transcript.write_text("BAC009S0100W0122\t广 州 市\n", encoding="utf-8")

    utterances = read_aishell_transcript(transcript, tmp_path / "wav")

    assert len(utterances) == 1
    assert utterances[0].speaker == "S0100"
    assert utterances[0].split == "full"
    assert utterances[0].text == "广州市"
    assert utterances[0].audio_path == wav


def test_load_tier_speakers_validates_phase5_plan(tmp_path: Path):
    plan = {
        "tiers": {
            "debug": {
                "speaker_count": 2,
                "speakers": ["S0100", "S0101"],
            }
        }
    }
    plan_path = tmp_path / "speaker_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    assert load_tier_speakers(plan_path, "debug") == ["S0100", "S0101"]

    plan["tiers"]["debug"]["speakers"] = ["S0100", "S0100"]
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    try:
        load_tier_speakers(plan_path, "debug")
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("expected duplicate speaker failure")


def test_select_speakers_is_deterministic_and_excludes_pilot():
    members = {f"S{index:04d}": object() for index in range(20)}

    selected = select_speakers(members, count=6, seed="phase3-v2")

    assert selected == select_speakers(members, count=6, seed="phase3-v2")
    assert len(selected) == len(set(selected)) == 6
    assert not set(selected) & {"S0002", "S0003", "S0724", "S0764"}
