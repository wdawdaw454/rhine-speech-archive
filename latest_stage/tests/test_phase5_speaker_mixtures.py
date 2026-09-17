from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.prepare_phase5_speaker_mixtures import main
from src.audio.io import load_audio as read_mono_pcm16
from src.data.manifest import ManifestEntry, read_jsonl


def _write_source_manifest(path: Path, speakers: list[str], utterances: int = 2) -> None:
    entries: list[ManifestEntry] = []
    for speaker in speakers:
        for index in range(utterances):
            wav_path = path.parent / f"{speaker}_{index}.wav"
            samples = int(16000 * (0.12 + 0.01 * index))
            frequency = 180.0 if index % 2 == 0 else 220.0
            time = np.arange(samples, dtype=np.float32) / 16000.0
            waveform = 0.2 * np.sin(2.0 * np.pi * frequency * time).astype(np.float32)
            sf.write(wav_path, waveform, 16000, subtype="FLOAT")
            entries.append(
                ManifestEntry(
                    id=f"{speaker}_u{index}",
                    audio_path=str(wav_path.resolve()),
                    sample_rate=16000,
                    duration=round(samples / 16000.0, 6),
                    extras={
                        "speaker": speaker,
                        "text": f"{speaker}文本{index}",
                        "condition": "clean",
                    },
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")


def _speaker_plan(path: Path, speakers: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "tiers": {"debug": {"speaker_count": len(speakers), "speakers": speakers}},
                "reserved_speakers": {"phase3": ["S0090"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_builds_speaker_disjoint_mixture_manifests(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    train_manifest = source_root / "train.jsonl"
    validation_manifest = source_root / "validation.jsonl"
    _write_source_manifest(train_manifest, ["S0001", "S0002"])
    _write_source_manifest(validation_manifest, ["S0003", "S0004"])
    speaker_plan = tmp_path / "speaker_plan.json"
    _speaker_plan(speaker_plan, ["S0001", "S0002", "S0003", "S0004"])
    output_root = tmp_path / "mixtures"

    result = main(
        [
            "--train-manifest",
            str(train_manifest),
            "--validation-manifest",
            str(validation_manifest),
            "--speaker-plan",
            str(speaker_plan),
            "--tier",
            "debug",
            "--output-root",
            str(output_root),
            "--max-train-pairs",
            "1",
            "--max-validation-pairs",
            "1",
            "--utterances-per-mixture",
            "2",
        ]
    )

    assert result == 0
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["phase"] == "5.6"
    assert summary["condition"] == "speaker_mixture"
    assert summary["reserved_overlap"] == []
    assert summary["splits"]["train"]["mixtures"] == 3
    assert summary["splits"]["validation"]["mixtures"] == 3
    assert summary["splits"]["train"]["manifest"] == str((output_root / "train.jsonl").resolve())

    train_entries = list(read_jsonl(output_root / "train.jsonl"))
    validation_entries = list(read_jsonl(output_root / "validation.jsonl"))
    train_speakers = {entry.extras["speaker"] for entry in train_entries}
    validation_speakers = {entry.extras["speaker"] for entry in validation_entries}
    assert train_speakers <= {"S0001", "S0002"}
    assert validation_speakers <= {"S0003", "S0004"}
    assert not train_speakers & validation_speakers

    for entry in train_entries:
        assert Path(entry.audio_path).is_file()
        waveform, sample_rate = read_mono_pcm16(Path(entry.audio_path))
        assert sample_rate == 16000 and waveform.size > 0
        references_path = Path(entry.extras["references"])
        references = [
            json.loads(line)
            for line in references_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        assert len(references) == 2
        assert {item["role"] for item in references} == {"target", "non_target"}
        expected_text = "".join(
            item["text"] for item in references if item["role"] == "target"
        )
        assert entry.extras["text"] == expected_text
        assert entry.extras["interferer_speaker"] != entry.extras["speaker"]
        assert entry.extras["condition"] == "speaker_mixture"
