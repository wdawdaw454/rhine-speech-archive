"""Selectively extract deterministic AISHELL-1 speaker samples."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PILOT_SPEAKERS = {"S0002", "S0003", "S0724", "S0764"}
SPEAKER_RE = re.compile(r"^data_aishell/wav/(S\d{4})\.tar\.gz$")
TRANSCRIPT_MEMBER = "data_aishell/transcript/aishell_transcript_v0.8.txt"
SPEAKER_ID_RE = re.compile(r"^S\d{4}$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract selected AISHELL-1 speakers")
    parser.add_argument(
        "--archive", type=Path, default=PROJECT_ROOT / "data/raw/aishell/data_aishell.tgz"
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data/raw/aishell_full")
    parser.add_argument("--num-speakers", type=int, default=16)
    parser.add_argument("--selection-seed", default="phase3-v2")
    parser.add_argument(
        "--speaker-plan",
        type=Path,
        default=None,
        help="JSON speaker plan produced by plan_phase5_training_speakers.py",
    )
    parser.add_argument("--tier", default="debug", choices=("debug", "full"))
    return parser


def select_speakers(
    speaker_members: dict[str, tarfile.TarInfo],
    *,
    count: int,
    seed: str,
    excluded: set[str] = PILOT_SPEAKERS,
) -> list[str]:
    if count <= 0 or count % 2:
        raise ValueError("speaker count must be a positive even number")
    candidates = sorted(set(speaker_members) - excluded)
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} candidate speakers available for count={count}")
    ranked = sorted(
        candidates, key=lambda speaker: hashlib.sha256(f"{seed}:{speaker}".encode()).hexdigest()
    )
    return ranked[:count]


def load_tier_speakers(plan_path: Path, tier: str) -> list[str]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tiers = plan.get("tiers")
    if not isinstance(tiers, dict) or tier not in tiers:
        raise ValueError(f"speaker plan does not contain tier: {tier}")

    tier_data = tiers[tier]
    speakers = tier_data.get("speakers", [])
    expected_count = tier_data.get("speaker_count")
    if not isinstance(speakers, list) or not speakers:
        raise ValueError(f"speaker plan tier is empty: {tier}")
    if not all(isinstance(item, str) and SPEAKER_ID_RE.fullmatch(item) for item in speakers):
        raise ValueError(f"speaker plan tier contains an invalid speaker ID: {tier}")
    if len(set(speakers)) != len(speakers):
        raise ValueError(f"speaker plan tier contains duplicate speakers: {tier}")
    if expected_count != len(speakers):
        raise ValueError(
            f"speaker plan count mismatch for {tier}: "
            f"expected {expected_count}, found {len(speakers)}"
        )
    return list(speakers)


def _extract_wav_member(
    member: tarfile.TarInfo, source: io.BufferedReader, destination: Path
) -> None:
    relative = Path(member.name)
    if relative.is_absolute() or ".." in relative.parts or not member.isfile():
        raise ValueError(f"unsafe AISHELL speaker archive member: {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source, destination.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)


def _extract_speaker(member: tarfile.TarInfo, outer: tarfile.TarFile, wav_root: Path) -> None:
    speaker_match = SPEAKER_RE.match(member.name)
    if speaker_match is None:
        raise ValueError(f"invalid selected speaker member: {member.name}")
    speaker = speaker_match.group(1)
    extracted = outer.extractfile(member)
    if extracted is None:
        raise ValueError(f"cannot read selected speaker archive: {member.name}")
    speaker_root = wav_root / speaker
    with tarfile.open(fileobj=extracted, mode="r:gz") as speaker_archive:
        for inner_member in speaker_archive:
            if inner_member.name.endswith(".wav"):
                source = speaker_archive.extractfile(inner_member)
                if source is None:
                    raise ValueError(f"cannot read AISHELL WAV: {speaker}/{inner_member.name}")
                _extract_wav_member(
                    inner_member, source, speaker_root / Path(inner_member.name).name
                )
    extracted.close()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    archive = args.archive.resolve()
    output_root = args.output_root.resolve()
    if not archive.is_file():
        raise SystemExit(f"AISHELL archive not found: {archive}")
    if output_root.exists():
        raise SystemExit(f"output root already exists: {output_root}")
    staging = output_root.with_name(f"{output_root.name}.partial")
    if staging.exists():
        raise SystemExit(f"incomplete extraction already exists: {staging}")

    data_root = staging / "data_aishell"
    wav_root = data_root / "wav"
    transcript_root = data_root / "transcript"
    wav_root.mkdir(parents=True)
    transcript_root.mkdir(parents=True)

    with tarfile.open(archive, mode="r:gz") as outer:
        speaker_members: dict[str, tarfile.TarInfo] = {}
        transcript_member: tarfile.TarInfo | None = None
        selected: list[str]
        for member in outer:
            speaker_match = SPEAKER_RE.match(member.name)
            if speaker_match:
                speaker_members[speaker_match.group(1)] = member
            elif member.name == TRANSCRIPT_MEMBER:
                transcript_member = member
        if transcript_member is None:
            raise SystemExit("transcript member not found in AISHELL archive")

        if args.speaker_plan is None:
            selected = select_speakers(
                speaker_members,
                count=args.num_speakers,
                seed=args.selection_seed,
            )
            selection_mode = "seeded"
        else:
            selected = load_tier_speakers(args.speaker_plan.resolve(), args.tier)
            missing = sorted(set(selected) - set(speaker_members))
            if missing:
                raise SystemExit(
                    f"speaker plan members missing from archive ({len(missing)}): "
                    f"{', '.join(missing[:8])}"
                )
            selection_mode = f"phase5-{args.tier}"

        source = outer.extractfile(transcript_member)
        if source is None:
            raise SystemExit("cannot read AISHELL transcript")
        _extract_wav_member(
            transcript_member,
            source,
            transcript_root / Path(TRANSCRIPT_MEMBER).name,
        )
        total_wavs = 0
        for speaker in selected:
            print(f"extracting {speaker}", flush=True)
            _extract_speaker(speaker_members[speaker], outer, wav_root)
            total_wavs += len(list((wav_root / speaker).glob("*.wav")))

    metadata = {
        "archive": str(archive),
        "selection_mode": selection_mode,
        "selection_seed": args.selection_seed,
        "selected_speakers": selected,
        "excluded_pilot_speakers": sorted(PILOT_SPEAKERS),
        "speaker_count": len(selected),
        "wav_count": total_wavs,
        "transcript": str((output_root / TRANSCRIPT_MEMBER).resolve()),
        "audio_root": str((output_root / "data_aishell" / "wav").resolve()),
    }
    if args.speaker_plan is not None:
        metadata["speaker_plan"] = str(args.speaker_plan.resolve())
        metadata["speaker_plan_tier"] = args.tier
    (staging / "source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(output_root))
    print(f"extracted_root={data_root.resolve()}")
    print(f"speakers={len(selected)} wavs={total_wavs}")


if __name__ == "__main__":
    main()
