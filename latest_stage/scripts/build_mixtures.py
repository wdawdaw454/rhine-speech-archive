"""Build a deterministic alternating or overlapping two-speaker mixture."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.audio.io import load_audio
from src.data.manifest import read_jsonl
from src.data.mixtures import (
    MixtureSource,
    render_mixture,
    schedule_alternating,
)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a two-speaker test mixture")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--target-speaker", required=True)
    parser.add_argument("--interferer-speaker", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-utterances", type=int, default=4)
    parser.add_argument("--gap", type=float, default=0.40)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--target-snr-db", type=float, default=8.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _source(entry_id: str, speaker: str, text: str, duration: float) -> MixtureSource:
    return MixtureSource(
        source_id=entry_id,
        speaker=speaker,
        text=text,
        duration=duration,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.target_speaker == args.interferer_speaker:
        raise SystemExit("target and interferer speakers must be different")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    by_speaker: dict[str, list[MixtureSource]] = defaultdict(list)
    waveforms: dict[str, np.ndarray] = {}
    manifest_path = args.manifest.resolve()

    for entry in read_jsonl(manifest_path):
        speaker = str(entry.extras.get("speaker", ""))
        text = str(entry.extras.get("text", ""))
        if not speaker:
            raise SystemExit(f"manifest entry is missing extras.speaker: {entry.id}")
        if not text:
            raise SystemExit(f"manifest entry is missing extras.text: {entry.id}")
        raw_path = Path(entry.audio_path)
        audio_path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
        waveform, sample_rate = load_audio(audio_path, target_sr=args.sample_rate)
        if sample_rate != args.sample_rate:
            raise SystemExit(f"unexpected sample rate for {audio_path}: {sample_rate}")
        source = _source(
            entry.id,
            speaker,
            text,
            len(waveform) / args.sample_rate,
        )
        by_speaker[speaker].append(source)
        waveforms[source.source_id] = waveform

    target_sources = list(by_speaker[args.target_speaker])
    interferer_sources = list(by_speaker[args.interferer_speaker])
    if not target_sources or not interferer_sources:
        missing = args.target_speaker if not target_sources else args.interferer_speaker
        raise SystemExit(f"manifest has no utterances for speaker: {missing}")

    rng.shuffle(target_sources)
    rng.shuffle(interferer_sources)
    placements = schedule_alternating(
        target_sources=target_sources,
        non_target_sources=interferer_sources,
        num_utterances=args.num_utterances,
        gap=args.gap,
        overlap=args.overlap,
    )
    mixture, peak_normalization = render_mixture(
        placements,
        waveforms,
        sample_rate=args.sample_rate,
        target_snr_db=args.target_snr_db,
    )

    mixture_path = args.output_dir / "mixture.wav"
    reference_path = args.output_dir / "references.jsonl"
    summary_path = args.output_dir / "mixture_summary.json"
    sf.write(str(mixture_path), mixture, args.sample_rate, subtype="FLOAT")
    with reference_path.open("w", encoding="utf-8", newline="\n") as handle:
        for placement in placements:
            record = asdict(placement)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "manifest": str(manifest_path),
        "target_speaker": args.target_speaker,
        "interferer_speaker": args.interferer_speaker,
        "num_utterances": len(placements),
        "gap": args.gap,
        "overlap": args.overlap,
        "target_snr_db": args.target_snr_db,
        "sample_rate": args.sample_rate,
        "seed": args.seed,
        "duration": len(mixture) / args.sample_rate,
        "peak_normalization": peak_normalization,
        "mixture": str(mixture_path),
        "references": str(reference_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"mixture={mixture_path}")
    print(f"references={reference_path}")
    print(f"summary={summary_path}")
    print(f"duration={summary['duration']:.3f}s utterances={len(placements)}")


if __name__ == "__main__":
    main()
