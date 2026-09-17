"""Prepare a multi-pair speaker-disjoint split from extracted AISHELL-1 audio."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.data.aishell import AishellUtterance, read_aishell_transcript
from src.data.manifest import ManifestEntry, write_jsonl

SCENARIOS = (
    {"name": "alternate_snr20", "gap": 1.0, "overlap": 0.0, "target_snr_db": 20.0},
    {"name": "alternate_snr8", "gap": 1.0, "overlap": 0.0, "target_snr_db": 8.0},
    {"name": "overlap02_snr8", "gap": 1.0, "overlap": 0.2, "target_snr_db": 8.0},
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare multi-pair AISHELL-1 manifests")
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data/raw/aishell_full")
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "data/processed/aishell_full"
    )
    parser.add_argument("--pairs-per-split", type=int, default=4)
    parser.add_argument("--enroll-utterances", type=int, default=3)
    parser.add_argument("--eval-utterances-per-speaker", type=int, default=8)
    parser.add_argument("--num-utterances", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _manifest_entry(utterance: AishellUtterance, manifest_root: Path) -> ManifestEntry:
    relative = Path(os.path.relpath(utterance.audio_path, manifest_root))
    return ManifestEntry(
        id=utterance.utterance_id,
        audio_path=relative.as_posix(),
        sample_rate=16000,
        duration=0.0,
        extras={"speaker": utterance.speaker, "text": utterance.text},
    )


def _write_manifest(path: Path, utterances: list[AishellUtterance], manifest_root: Path) -> None:
    entries = [_manifest_entry(item, manifest_root) for item in utterances]
    write_jsonl(path, entries)


def _pair_config(target: str, interferer: str, seed: int, num_utterances: int) -> dict[str, Any]:
    return {
        "version": 1,
        "target_speaker": target,
        "interferer_speaker": interferer,
        "num_utterances": num_utterances,
        "sample_rate": 16000,
        "seeds": [seed],
        "scenarios": [dict(item) for item in SCENARIOS],
    }


def _select_utterances(
    utterances: list[AishellUtterance],
    *,
    enroll_count: int,
    eval_count: int,
    seed: int,
) -> tuple[list[AishellUtterance], list[AishellUtterance]]:
    shuffled = list(utterances)
    random.Random(seed).shuffle(shuffled)
    required = enroll_count + eval_count
    if len(shuffled) < required:
        raise ValueError(f"need {required} utterances, found {len(shuffled)}")
    return shuffled[:enroll_count], shuffled[enroll_count : enroll_count + eval_count]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.pairs_per_split <= 0:
        raise SystemExit("pairs-per-split must be positive")
    if args.enroll_utterances <= 0 or args.eval_utterances_per_speaker < 4:
        raise SystemExit("enroll count must be positive and eval count must be at least 4")
    if args.num_utterances <= 0 or args.num_utterances % 2:
        raise SystemExit("num-utterances must be a positive even number")

    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    source_path = raw_root / "source.json"
    if not source_path.is_file():
        raise SystemExit(f"extraction metadata not found: {source_path}")
    if output_root.exists():
        raise SystemExit(f"output root already exists: {output_root}")
    staging = output_root.with_name(f"{output_root.name}.partial")
    if staging.exists():
        raise SystemExit(f"incomplete staging directory exists: {staging}")

    source = json.loads(source_path.read_text(encoding="utf-8"))
    selected = [str(item) for item in source["selected_speakers"]]
    needed = args.pairs_per_split * 4
    if len(selected) < needed:
        raise SystemExit(f"need {needed} selected speakers, found {len(selected)}")
    if len(set(selected)) != len(selected):
        raise SystemExit("selected speakers contain duplicates")

    transcript = raw_root / "data_aishell" / "transcript" / "aishell_transcript_v0.8.txt"
    audio_root = raw_root / "data_aishell" / "wav"
    all_utterances = read_aishell_transcript(transcript, audio_root)
    by_speaker: dict[str, list[AishellUtterance]] = {speaker: [] for speaker in selected}
    for utterance in all_utterances:
        if utterance.speaker in by_speaker:
            by_speaker[utterance.speaker].append(utterance)

    selected_data: dict[str, dict[str, list[AishellUtterance]]] = {}
    for speaker in selected:
        enroll, evaluation = _select_utterances(
            by_speaker[speaker],
            enroll_count=args.enroll_utterances,
            eval_count=args.eval_utterances_per_speaker,
            seed=args.seed + int(speaker[1:]),
        )
        selected_data[speaker] = {"enroll": enroll, "eval": evaluation}

    split_summary: dict[str, Any] = {"splits": {}}
    for split_index, split in enumerate(("dev", "test")):
        speakers = selected[
            split_index * args.pairs_per_split * 2 : (split_index + 1) * args.pairs_per_split * 2
        ]
        split_root = staging / split
        config_root = split_root / "configs"
        config_root.mkdir(parents=True)
        split_sources: list[AishellUtterance] = []
        pairs: list[dict[str, Any]] = []
        for pair_index in range(args.pairs_per_split):
            target = speakers[pair_index * 2]
            interferer = speakers[pair_index * 2 + 1]
            pair_name = f"{target}_{interferer}"
            enroll_path = split_root / "targets" / target / "enroll_target.jsonl"
            _write_manifest(enroll_path, selected_data[target]["enroll"], enroll_path.parent)
            config_path = config_root / f"{pair_name}.yaml"
            seed = 42 if split == "dev" else 52
            config = _pair_config(target, interferer, seed, args.num_utterances)
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            pairs.append(
                {
                    "split": split,
                    "target": target,
                    "interferer": interferer,
                    "config": str((output_root / split / "configs" / config_path.name).resolve()),
                    "enroll_manifest": str(
                        (output_root / split / "targets" / target / "enroll_target.jsonl").resolve()
                    ),
                    "profile": str(
                        (PROJECT_ROOT / "profiles/aishell_full" / f"{target}.json").resolve()
                    ),
                }
            )
            split_sources.extend(selected_data[target]["eval"])
            split_sources.extend(selected_data[interferer]["eval"])

        _write_manifest(
            split_root / "sources.jsonl",
            sorted(split_sources, key=lambda item: item.utterance_id),
            split_root,
        )
        split_summary["splits"][split] = {
            "pairs": pairs,
            "source_manifest": str((output_root / split / "sources.jsonl").resolve()),
        }

    summary = {
        "seed": args.seed,
        "selection_seed": source["selection_seed"],
        "pairs_per_split": args.pairs_per_split,
        "enroll_utterances": args.enroll_utterances,
        "eval_utterances_per_speaker": args.eval_utterances_per_speaker,
        "num_utterances": args.num_utterances,
        "selected_speakers": selected,
        **split_summary,
    }
    (staging / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(output_root))
    print(f"prepared={output_root / 'split_summary.json'}")
    for split, info in split_summary["splits"].items():
        print(
            f"{split}: pairs={len(info['pairs'])} "
            f"speakers={len({p['target'] for p in info['pairs']} | {p['interferer'] for p in info['pairs']})}"
        )


if __name__ == "__main__":
    main()
