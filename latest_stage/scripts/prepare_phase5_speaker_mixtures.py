"""Build deterministic Phase 5 two-speaker mixture tuning manifests."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.extract_aishell_speakers import load_tier_speakers
from scripts.prepare_phase5_clean_manifest import _reserved_speakers
from src.audio.io import load_audio
from src.data.manifest import ManifestEntry, read_jsonl, write_jsonl
from src.data.mixtures import MixtureSource, render_mixture, schedule_alternating

STANDARD_SCENARIOS = (
    {"name": "alternate_snr20", "gap": 1.0, "overlap": 0.0, "target_snr_db": 20.0},
    {"name": "alternate_snr8", "gap": 1.0, "overlap": 0.0, "target_snr_db": 8.0},
    {"name": "overlap02_snr8", "gap": 1.0, "overlap": 0.2, "target_snr_db": 8.0},
)
EXTENDED_SCENARIOS = STANDARD_SCENARIOS + (
    {"name": "alternate_snr0", "gap": 1.0, "overlap": 0.0, "target_snr_db": 0.0},
    {"name": "overlap05_snr0", "gap": 1.0, "overlap": 0.5, "target_snr_db": 0.0},
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build speaker-disjoint Phase 5 mixture tuning manifests"
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/processed/phase5_tuning/clean_debug/train.jsonl",
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/processed/phase5_tuning/clean_debug/validation.jsonl",
    )
    parser.add_argument(
        "--speaker-plan",
        type=Path,
        default=PROJECT_ROOT / "data/processed/phase5_tuning/speaker_plan.json",
    )
    parser.add_argument("--tier", choices=("debug", "full"), default="debug")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data/processed/phase5_tuning/speaker_mixture_debug",
    )
    parser.add_argument("--scenario-set", choices=("standard", "extended"), default="standard")
    parser.add_argument("--max-train-pairs", type=int, default=8)
    parser.add_argument("--max-validation-pairs", type=int, default=2)
    parser.add_argument("--variants-per-scenario", type=int, default=1)
    parser.add_argument("--utterances-per-mixture", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=56)
    return parser


def _scenarios(name: str) -> tuple[dict[str, Any], ...]:
    return EXTENDED_SCENARIOS if name == "extended" else STANDARD_SCENARIOS


def _load_sources(
    manifest_path: Path,
    *,
    sample_rate: int,
) -> tuple[dict[str, list[MixtureSource]], dict[str, np.ndarray], set[str]]:
    by_speaker: dict[str, list[MixtureSource]] = defaultdict(list)
    waveforms: dict[str, np.ndarray] = {}
    seen_ids: set[str] = set()
    for entry in read_jsonl(manifest_path):
        speaker = str(entry.extras.get("speaker", ""))
        text = str(entry.extras.get("text", ""))
        if not speaker or not text:
            raise SystemExit(f"manifest entry missing extras.speaker/text: {entry.id}")
        if entry.sample_rate != sample_rate:
            raise SystemExit(
                f"manifest sample-rate mismatch for {entry.id}: "
                f"{entry.sample_rate} != {sample_rate}"
            )
        if entry.id in seen_ids:
            raise SystemExit(f"duplicate manifest id: {entry.id}")
        seen_ids.add(entry.id)
        raw_path = Path(entry.audio_path)
        audio_path = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
        waveform, loaded_rate = load_audio(audio_path, target_sr=sample_rate)
        if loaded_rate != sample_rate:
            raise SystemExit(f"unexpected sample rate for {audio_path}: {loaded_rate}")
        by_speaker[speaker].append(
            MixtureSource(
                source_id=entry.id,
                speaker=speaker,
                text=text,
                duration=len(waveform) / sample_rate,
            )
        )
        waveforms[entry.id] = waveform
    return dict(by_speaker), waveforms, set(by_speaker)


def _speaker_pairs(speakers: Iterable[str], limit: int, *, seed: int) -> list[tuple[str, str]]:
    sorted_speakers = sorted(speakers)
    rng = np.random.default_rng(seed)
    ordered = [sorted_speakers[index] for index in rng.permutation(len(sorted_speakers))]
    pairs = [(ordered[i], ordered[i + 1]) for i in range(0, len(ordered) - 1, 2)]
    return pairs[: max(0, limit)]


def _rotated_slice(sources: list[MixtureSource], start: int, count: int) -> list[MixtureSource]:
    if count <= 0:
        raise ValueError("count must be positive")
    if len(sources) < count:
        raise ValueError(f"only {len(sources)} sources available; {count} required")
    offset = start % len(sources)
    return [sources[(offset + index) % len(sources)] for index in range(count)]


def _manifest_entry(
    *,
    mixture_id: str,
    mixture_path: Path,
    duration: float,
    placements: list[Any],
    target_speaker: str,
    interferer_speaker: str,
    scenario: dict[str, Any],
    references_path: Path,
    sample_rate: int,
) -> ManifestEntry:
    target_text = "".join(str(item.text) for item in placements if item.role == "target")
    if not target_text:
        raise ValueError(f"mixture has no target transcript: {mixture_id}")
    return ManifestEntry(
        id=mixture_id,
        audio_path=str(mixture_path.resolve()),
        sample_rate=sample_rate,
        duration=round(duration, 6),
        extras={
            "speaker": target_speaker,
            "text": target_text,
            "condition": "speaker_mixture",
            "scenario": str(scenario["name"]),
            "target_speaker": target_speaker,
            "interferer_speaker": interferer_speaker,
            "gap": float(scenario["gap"]),
            "overlap": float(scenario["overlap"]),
            "target_snr_db": float(scenario["target_snr_db"]),
            "target_source_ids": [item.source_id for item in placements if item.role == "target"],
            "interferer_source_ids": [
                item.source_id for item in placements if item.role == "non_target"
            ],
            "references": str(references_path.resolve()),
        },
    )


def _build_split(
    *,
    split_name: str,
    pairs: list[tuple[str, str]],
    by_speaker: dict[str, list[MixtureSource]],
    waveforms: dict[str, np.ndarray],
    scenarios: tuple[dict[str, Any], ...],
    variants_per_scenario: int,
    utterances_per_mixture: int,
    sample_rate: int,
    staging_root: Path,
    seed: int,
) -> tuple[list[ManifestEntry], dict[str, Any]]:
    if utterances_per_mixture <= 0:
        raise SystemExit("utterances-per-mixture must be positive")
    if variants_per_scenario <= 0:
        raise SystemExit("variants-per-scenario must be positive")
    role_turns = (utterances_per_mixture + 1) // 2
    non_target_turns = utterances_per_mixture // 2
    if non_target_turns == 0:
        raise SystemExit("utterances-per-mixture must include at least one interferer turn")

    entries: list[ManifestEntry] = []
    scenario_counts: Counter[str] = Counter()
    durations: list[float] = []
    for pair_index, (target_speaker, interferer_speaker) in enumerate(pairs):
        target_sources = sorted(
            by_speaker[target_speaker], key=lambda item: item.source_id
        )
        interferer_sources = sorted(
            by_speaker[interferer_speaker], key=lambda item: item.source_id
        )
        if len(target_sources) < role_turns or len(interferer_sources) < non_target_turns:
            raise SystemExit(
                f"insufficient source utterances for pair: {target_speaker}/{interferer_speaker}"
            )

        for scenario_index, scenario in enumerate(scenarios):
            for variant in range(variants_per_scenario):
                target_subset = _rotated_slice(
                    target_sources,
                    start=(scenario_index * role_turns) + (variant * role_turns),
                    count=role_turns,
                )
                interferer_subset = _rotated_slice(
                    interferer_sources,
                    start=(scenario_index * non_target_turns) + (variant * non_target_turns),
                    count=non_target_turns,
                )
                placements = schedule_alternating(
                    target_sources=target_subset,
                    non_target_sources=interferer_subset,
                    num_utterances=utterances_per_mixture,
                    gap=float(scenario["gap"]),
                    overlap=float(scenario["overlap"]),
                )
                mixture, _ = render_mixture(
                    placements,
                    waveforms,
                    sample_rate=sample_rate,
                    target_snr_db=float(scenario["target_snr_db"]),
                )
                mixture_id = (
                    f"{target_speaker}_{interferer_speaker}_{scenario['name']}_v{variant + 1:02d}"
                )
                item_root = staging_root / "audio" / split_name / mixture_id
                item_root.mkdir(parents=True)
                mixture_path = item_root / "mixture.wav"
                references_path = item_root / "references.jsonl"
                sf.write(str(mixture_path), mixture, sample_rate, subtype="PCM_16")
                with references_path.open("w", encoding="utf-8", newline="\n") as handle:
                    for placement in placements:
                        handle.write(json.dumps(asdict(placement), ensure_ascii=False) + "\n")
                duration = len(mixture) / sample_rate
                entries.append(
                    _manifest_entry(
                        mixture_id=mixture_id,
                        mixture_path=mixture_path,
                        duration=duration,
                        placements=placements,
                        target_speaker=target_speaker,
                        interferer_speaker=interferer_speaker,
                        scenario=scenario,
                        references_path=references_path,
                        sample_rate=sample_rate,
                    )
                )
                scenario_counts[str(scenario["name"])] += 1
                durations.append(duration)

    rng = np.random.default_rng(seed)
    rng.shuffle(entries)
    speaker_counts = Counter(entry.extras["target_speaker"] for entry in entries)
    return entries, {
        "mixtures": len(entries),
        "speaker_count": len(speaker_counts),
        "target_speaker_counts": dict(sorted(speaker_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "duration_hours": round(sum(durations) / 3600.0, 4),
        "pairs": [{"target": target, "interferer": interferer} for target, interferer in pairs],
    }

def _rebase_manifest_paths(
    entries: list[ManifestEntry], staging_root: Path, output_root: Path
) -> None:
    """Point manifest paths at the final directory after transactional rename."""
    for entry in entries:
        relative_audio = Path(entry.audio_path).relative_to(staging_root)
        entry.audio_path = str((output_root / relative_audio).resolve())
        relative_references = Path(entry.extras["references"]).relative_to(staging_root)
        entry.extras["references"] = str((output_root / relative_references).resolve())

def _write_split(
    staging: Path,
    split_name: str,
    entries: list[ManifestEntry],
) -> Path:
    path = staging / f"{split_name}.jsonl"
    write_jsonl(path, entries)
    return path


def _validate_counts(
    *,
    max_train_pairs: int,
    max_validation_pairs: int,
    train_pairs: list[tuple[str, str]],
    validation_pairs: list[tuple[str, str]],
) -> None:
    if max_train_pairs <= 0 or max_validation_pairs <= 0:
        raise SystemExit("train and validation pair limits must be positive")
    if not train_pairs or not validation_pairs:
        raise SystemExit("at least one training and validation speaker pair is required")
    train_speakers = {speaker for pair in train_pairs for speaker in pair}
    validation_speakers = {speaker for pair in validation_pairs for speaker in pair}
    overlap = sorted(train_speakers & validation_speakers)
    if overlap:
        raise SystemExit(f"train/validation speaker overlap: {overlap}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_path = args.train_manifest.resolve()
    validation_path = args.validation_manifest.resolve()
    speaker_plan_path = args.speaker_plan.resolve()
    output_root = args.output_root.resolve()
    if not train_path.is_file() or not validation_path.is_file():
        raise SystemExit("clean train or validation manifest not found")
    if not speaker_plan_path.is_file():
        raise SystemExit(f"speaker plan not found: {speaker_plan_path}")
    if output_root.exists():
        raise SystemExit(f"output root already exists: {output_root}")
    staging = output_root.with_name(f"{output_root.name}.partial")
    if staging.exists():
        raise SystemExit(f"incomplete staging directory already exists: {staging}")

    tier_speakers = set(load_tier_speakers(speaker_plan_path, args.tier))
    reserved = _reserved_speakers(json.loads(speaker_plan_path.read_text(encoding="utf-8")))
    train_sources, train_waveforms, train_speakers = _load_sources(
        train_path, sample_rate=args.sample_rate
    )
    validation_sources, validation_waveforms, validation_speakers = _load_sources(
        validation_path, sample_rate=args.sample_rate
    )
    manifest_speakers = train_speakers | validation_speakers
    outside_tier = sorted(manifest_speakers - tier_speakers)
    reserved_overlap = sorted(manifest_speakers & reserved)
    if outside_tier:
        raise SystemExit(f"manifest speakers outside requested tier: {outside_tier}")
    if reserved_overlap:
        raise SystemExit(f"reserved evaluation speakers found: {reserved_overlap}")
    if train_speakers & validation_speakers:
        raise SystemExit("clean manifests already overlap across train and validation")

    train_pairs = _speaker_pairs(train_speakers, args.max_train_pairs, seed=args.seed)
    validation_pairs = _speaker_pairs(
        validation_speakers, args.max_validation_pairs, seed=args.seed + 1
    )
    _validate_counts(
        max_train_pairs=args.max_train_pairs,
        max_validation_pairs=args.max_validation_pairs,
        train_pairs=train_pairs,
        validation_pairs=validation_pairs,
    )

    scenarios = _scenarios(args.scenario_set)
    staging.mkdir(parents=True)
    train_entries, train_summary = _build_split(
        split_name="train",
        pairs=train_pairs,
        by_speaker=train_sources,
        waveforms=train_waveforms,
        scenarios=scenarios,
        variants_per_scenario=args.variants_per_scenario,
        utterances_per_mixture=args.utterances_per_mixture,
        sample_rate=args.sample_rate,
        staging_root=staging,
        seed=args.seed,
    )
    validation_entries, validation_summary = _build_split(
        split_name="validation",
        pairs=validation_pairs,
        by_speaker=validation_sources,
        waveforms=validation_waveforms,
        scenarios=scenarios,
        variants_per_scenario=args.variants_per_scenario,
        utterances_per_mixture=args.utterances_per_mixture,
        sample_rate=args.sample_rate,
        staging_root=staging,
        seed=args.seed + 1,
    )
    if not train_entries or not validation_entries:
        raise SystemExit("generated train or validation mixture split is empty")
    _rebase_manifest_paths(train_entries, staging, output_root)
    _rebase_manifest_paths(validation_entries, staging, output_root)

    train_manifest = _write_split(staging, "train", train_entries)
    validation_manifest = _write_split(staging, "validation", validation_entries)
    smoke_manifest = _write_split(staging, "smoke_train", train_entries[: min(8, len(train_entries))])

    summary = {
        "schema_version": 1,
        "phase": "5.6",
        "tier": args.tier,
        "condition": "speaker_mixture",
        "scenario_set": args.scenario_set,
        "scenarios": [dict(item) for item in scenarios],
        "seed": args.seed,
        "sample_rate": args.sample_rate,
        "utterances_per_mixture": args.utterances_per_mixture,
        "variants_per_scenario": args.variants_per_scenario,
        "speaker_plan": str(speaker_plan_path),
        "source_manifests": {
            "train": str(train_path),
            "validation": str(validation_path),
        },
        "reserved_speakers": sorted(reserved),
        "reserved_overlap": reserved_overlap,
        "splits": {
            "train": {
                **train_summary,
                "manifest": str((output_root / "train.jsonl").resolve()),
            },
            "validation": {
                **validation_summary,
                "manifest": str((output_root / "validation.jsonl").resolve()),
            },
            "smoke_train": {
                "mixtures": min(8, len(train_entries)),
                "manifest": str((output_root / "smoke_train.jsonl").resolve()),
            },
        },
    }
    (staging / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (staging / "recipe.yaml").write_text(
        yaml.safe_dump(
            {
                "phase": "5.6",
                "tier": args.tier,
                "condition": "speaker_mixture",
                "scenario_set": args.scenario_set,
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
        f"train={len(train_entries)} validation={len(validation_entries)} "
        f"smoke={min(8, len(train_entries))}"
    )
    print(f"train_hours={train_summary['duration_hours']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
