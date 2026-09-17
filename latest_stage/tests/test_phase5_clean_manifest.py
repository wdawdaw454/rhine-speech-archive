from __future__ import annotations

import json
import wave
from pathlib import Path

from scripts.prepare_phase5_clean_manifest import main


def _write_wav(path: Path, text_length: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0" * (2 * text_length))


def _write_plan(path: Path) -> None:
    plan = {
        "reserved_speakers": {
            "phase3_pilot": ["S0002"],
            "phase3_dev": ["S0003"],
        },
        "tiers": {
            "debug": {
                "speaker_count": 3,
                "speakers": ["S0100", "S0101", "S0102"],
            }
        },
    }
    path.write_text(json.dumps(plan), encoding="utf-8")


def _write_raw_root(root: Path) -> None:
    transcript_lines = []
    for speaker_index, speaker in enumerate(("S0100", "S0101", "S0102")):
        for utterance_index in range(2):
            utterance_id = f"BAC009{speaker}W{utterance_index:04d}"
            audio = root / "data_aishell" / "wav" / speaker / f"{utterance_id}.wav"
            _write_wav(audio)
            transcript_lines.append(f"{utterance_id} 测 试 {utterance_index}")

    transcript_root = root / "data_aishell" / "transcript"
    transcript_root.mkdir(parents=True)
    (transcript_root / "aishell_transcript_v0.8.txt").write_text(
        "\n".join(transcript_lines) + "\n", encoding="utf-8"
    )
    (root / "source.json").write_text(
        json.dumps(
            {
                "selected_speakers": ["S0100", "S0101", "S0102"],
            }
        ),
        encoding="utf-8",
    )


def test_prepare_phase5_clean_manifest_splits_validation_speakers(tmp_path: Path):
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "processed" / "clean_debug"
    plan_path = tmp_path / "speaker_plan.json"
    _write_plan(plan_path)
    _write_raw_root(raw_root)

    main(
        [
            "--raw-root",
            str(raw_root),
            "--speaker-plan",
            str(plan_path),
            "--output-root",
            str(output_root),
            "--validation-speakers",
            "1",
            "--smoke-speakers",
            "2",
            "--smoke-utterances-per-speaker",
            "1",
        ]
    )

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["splits"]["train"]["speakers"] == ["S0100", "S0101"]
    assert summary["splits"]["validation"]["speakers"] == ["S0102"]
    assert summary["reserved_overlap"] == []
    assert summary["splits"]["train"]["utterances"] == 4
    assert summary["splits"]["validation"]["utterances"] == 2
    assert summary["splits"]["smoke_train"]["utterances"] == 2

    train_ids = {
        json.loads(line)["id"]
        for line in (output_root / "train.jsonl").read_text(encoding="utf-8").splitlines()
    }
    validation_ids = {
        json.loads(line)["id"]
        for line in (output_root / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert train_ids.isdisjoint(validation_ids)

    smoke = [
        json.loads(line)
        for line in (output_root / "smoke_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(smoke) == 2
    assert all(item["duration"] == 1.0 for item in smoke)
    assert all(item["extras"]["condition"] == "clean" for item in smoke)
