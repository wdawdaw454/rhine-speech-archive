"""Build a deterministic multi-scenario Phase 3 mixture dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.audio.io import load_audio
from src.data.manifest import read_jsonl
from src.data.mixtures import MixtureSource, render_mixture, schedule_alternating
from src.experiments.datasets import ExperimentDatasetConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a threshold-sweep mixture dataset")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    with args.config.open("r", encoding="utf-8") as handle:
        spec = ExperimentDatasetConfig.from_dict(yaml.safe_load(handle))

    output_root = args.output_root.resolve()
    staging_root = output_root.with_name(f"{output_root.name}.partial")
    if output_root.exists():
        raise SystemExit(f"output root already exists: {output_root}")
    if staging_root.exists():
        raise SystemExit(f"incomplete staging directory already exists: {staging_root}")
    staging_root.mkdir(parents=True)

    manifest_path = args.manifest.resolve()
    by_speaker: dict[str, list[MixtureSource]] = defaultdict(list)
    waveforms: dict[str, np.ndarray] = {}
    for entry in read_jsonl(manifest_path):
        speaker = str(entry.extras.get("speaker", ""))
        text = str(entry.extras.get("text", ""))
        if not speaker or not text:
            raise SystemExit(f"manifest entry missing extras.speaker/text: {entry.id}")
        raw_path = Path(entry.audio_path)
        audio_path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
        waveform, sample_rate = load_audio(audio_path, target_sr=spec.sample_rate)
        if sample_rate != spec.sample_rate:
            raise SystemExit(
                f"unexpected sample rate for {audio_path}: {sample_rate} != {spec.sample_rate}"
            )
        source = MixtureSource(
            source_id=entry.id,
            speaker=speaker,
            text=text,
            duration=len(waveform) / spec.sample_rate,
        )
        if source.source_id in waveforms:
            raise SystemExit(f"duplicate manifest id: {source.source_id}")
        by_speaker[speaker].append(source)
        waveforms[source.source_id] = waveform

    if not by_speaker[spec.target_speaker]:
        raise SystemExit(f"manifest has no target speaker: {spec.target_speaker}")
    if not by_speaker[spec.interferer_speaker]:
        raise SystemExit(f"manifest has no interferer speaker: {spec.interferer_speaker}")

    items: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(spec.scenarios):
        for seed in spec.seeds:
            rng = np.random.default_rng(seed + scenario_index)
            target_sources = list(by_speaker[spec.target_speaker])
            interferer_sources = list(by_speaker[spec.interferer_speaker])
            rng.shuffle(target_sources)
            rng.shuffle(interferer_sources)
            placements = schedule_alternating(
                target_sources=target_sources,
                non_target_sources=interferer_sources,
                num_utterances=spec.num_utterances,
                gap=scenario.gap,
                overlap=scenario.overlap,
            )
            mixture, normalization = render_mixture(
                placements,
                waveforms,
                sample_rate=spec.sample_rate,
                target_snr_db=scenario.target_snr_db,
            )

            scenario_root = staging_root / scenario.name / f"seed_{seed}"
            scenario_root.mkdir(parents=True)
            mixture_path = scenario_root / "mixture.wav"
            reference_path = scenario_root / "references.jsonl"
            sf.write(str(mixture_path), mixture, spec.sample_rate, subtype="FLOAT")
            with reference_path.open("w", encoding="utf-8", newline="\n") as handle:
                for placement in placements:
                    handle.write(json.dumps(asdict(placement), ensure_ascii=False) + "\n")

            items.append(
                {
                    "scenario": scenario.name,
                    "seed": seed,
                    "gap": scenario.gap,
                    "overlap": scenario.overlap,
                    "target_snr_db": scenario.target_snr_db,
                    "mixture": mixture_path.relative_to(staging_root).as_posix(),
                    "references": reference_path.relative_to(staging_root).as_posix(),
                    "duration": round(len(mixture) / spec.sample_rate, 6),
                    "num_utterances": len(placements),
                }
            )

    index = {
        "version": 1,
        "config": str(args.config.resolve()),
        "manifest": str(manifest_path),
        "target_speaker": spec.target_speaker,
        "interferer_speaker": spec.interferer_speaker,
        "sample_rate": spec.sample_rate,
        "num_utterances": spec.num_utterances,
        "items": items,
    }
    (staging_root / "dataset_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging_root), str(output_root))
    print(f"dataset={output_root / 'dataset_index.json'}")
    print(f"mixtures={len(items)} duration={sum(float(item['duration']) for item in items):.3f}s")


if __name__ == "__main__":
    main()
