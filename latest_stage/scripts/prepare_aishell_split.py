"""Prepare speaker-disjoint manifests from the ModelScope AISHELL-1 subset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.data.aishell import AishellUtterance, read_aishell_csv
from src.data.manifest import ManifestEntry, write_jsonl

SPLITS = {
    "dev": {
        "csv": "speech_asr_aishell_subset_validation.csv",
        "target": "S0724",
        "interferer": "S0764",
        "utterances": {
            "S0724": "speech_asr_aishell_subset_validation.csv",
            "S0764": "speech_asr_aishell_subset_testsets.csv",
        },
    },
    "test": {
        "csv": "speech_asr_aishell_subset_trainsets.csv",
        "target": "S0002",
        "interferer": "S0003",
        "utterances": {
            "S0002": "speech_asr_aishell_subset_trainsets.csv",
            "S0003": "speech_asr_aishell_subset_trainsets.csv",
        },
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare AISHELL speaker-disjoint manifests")
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data/raw/aishell_subset")
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "data/processed/aishell_subset"
    )
    parser.add_argument("--enroll-utterances", type=int, default=3)
    parser.add_argument("--eval-utterances-per-speaker", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _entry(utterance: AishellUtterance) -> ManifestEntry:
    return ManifestEntry(
        id=utterance.utterance_id,
        audio_path=str(
            Path("../../..") / utterance.audio_path.resolve().relative_to(PROJECT_ROOT / "data")
        ),
        sample_rate=16000,
        duration=0.0,
        extras={"speaker": utterance.speaker, "text": utterance.text},
    )


def _write_manifest(path: Path, utterances: list[AishellUtterance]) -> None:
    entries = [_entry(item) for item in utterances]
    if path.exists():
        path.unlink()
    write_jsonl(path, entries)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.enroll_utterances <= 0 or args.eval_utterances_per_speaker < 4:
        raise SystemExit("enroll count must be positive and eval count must be at least 4")
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"output root already exists: {output_root}")
    staging = output_root.with_name(f"{output_root.name}.partial")
    if staging.exists():
        raise SystemExit(f"incomplete staging directory exists: {staging}")

    all_csvs = {
        name: read_aishell_csv(raw_root / name, raw_root / "speech_asr_aishell_subset")
        for name in {
            "speech_asr_aishell_subset_trainsets.csv",
            "speech_asr_aishell_subset_validation.csv",
            "speech_asr_aishell_subset_testsets.csv",
        }
    }
    summary: dict[str, object] = {
        "seed": args.seed,
        "enroll_utterances": args.enroll_utterances,
        "eval_utterances_per_speaker": args.eval_utterances_per_speaker,
        "splits": {},
    }
    for split_name, split in SPLITS.items():
        speakers = (split["target"], split["interferer"])
        if len(set(speakers)) != 2:
            raise SystemExit(f"{split_name} target and interferer must differ")
        selected: dict[str, list[AishellUtterance]] = {}
        enroll: dict[str, list[AishellUtterance]] = {}
        for speaker in speakers:
            utterances = list(all_csvs[split["utterances"][speaker]])
            by_speaker = [item for item in utterances if item.speaker == speaker]
            if len(by_speaker) < args.enroll_utterances + args.eval_utterances_per_speaker:
                raise SystemExit(
                    f"not enough utterances for {speaker}: {len(by_speaker)} available, "
                    f"{args.enroll_utterances + args.eval_utterances_per_speaker} required"
                )
            rng = random.Random(args.seed + int(speaker[1:]))
            shuffled = list(by_speaker)
            rng.shuffle(shuffled)
            enroll[speaker] = shuffled[: args.enroll_utterances]
            selected[speaker] = shuffled[
                args.enroll_utterances : args.enroll_utterances + args.eval_utterances_per_speaker
            ]

        split_root = staging / split_name
        split_root.mkdir(parents=True)
        _write_manifest(split_root / "enroll_target.jsonl", enroll[split["target"]])
        _write_manifest(
            split_root / "sources.jsonl",
            sorted(
                selected[split["target"]] + selected[split["interferer"]],
                key=lambda item: item.utterance_id,
            ),
        )
        summary["splits"][split_name] = {
            "target": split["target"],
            "interferer": split["interferer"],
            "target_enroll": len(enroll[split["target"]]),
            "target_eval": len(selected[split["target"]]),
            "interferer_eval": len(selected[split["interferer"]]),
            "enroll_manifest": str((output_root / split_name / "enroll_target.jsonl").resolve()),
            "source_manifest": str((output_root / split_name / "sources.jsonl").resolve()),
        }

    (staging / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(output_root))
    print(f"prepared={output_root / 'split_summary.json'}")
    for split_name, info in summary["splits"].items():
        print(
            f"{split_name}: target={info['target']} interferer={info['interferer']} "
            f"eval={info['target_eval']}+{info['interferer_eval']}"
        )


if __name__ == "__main__":
    main()
