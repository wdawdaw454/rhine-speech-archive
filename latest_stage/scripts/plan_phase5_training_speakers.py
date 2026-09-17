"""Plan AISHELL-1 training speakers without extracting audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PILOT_SPEAKERS = {"S0002", "S0003", "S0724", "S0764"}
SPEAKER_MEMBER_RE = re.compile(r"^data_aishell/wav/(S\d{4})\.tar\.gz$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan Phase 5 AISHELL tuning speakers")
    parser.add_argument(
        "--archive",
        type=Path,
        default=PROJECT_ROOT / "data/raw/aishell/data_aishell.tgz",
    )
    parser.add_argument(
        "--reserved-summary",
        type=Path,
        default=PROJECT_ROOT / "data/processed/aishell_full/split_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/phase5_tuning/speaker_plan.json",
    )
    parser.add_argument("--selection-seed", default="phase5-v1")
    parser.add_argument("--debug-speakers", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser


def read_archive_speakers(archive: Path) -> list[str]:
    if not archive.is_file():
        raise FileNotFoundError(f"AISHELL archive is missing: {archive}")
    speakers: set[str] = set()
    with tarfile.open(archive, mode="r:gz") as outer:
        for member in outer:
            match = SPEAKER_MEMBER_RE.fullmatch(member.name)
            if match:
                speakers.add(match.group(1))
    return sorted(speakers)


def read_reserved_speakers(summary_path: Path) -> dict[str, list[str]]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"reserved split summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reserved: dict[str, list[str]] = {
        "phase3_pilot": sorted(PILOT_SPEAKERS),
        "phase3_full_dev": [],
        "phase3_full_test": [],
    }
    for split, key in (("dev", "phase3_full_dev"), ("test", "phase3_full_test")):
        speakers: set[str] = set()
        for pair in summary.get("splits", {}).get(split, {}).get("pairs", []):
            speakers.add(str(pair["target"]))
            speakers.add(str(pair["interferer"]))
        reserved[key] = sorted(speakers)
    if not reserved["phase3_full_dev"] or not reserved["phase3_full_test"]:
        raise ValueError("reserved summary does not contain both dev and test pairs")
    return reserved


def build_speaker_plan(
    archive_speakers: list[str],
    *,
    reserved: dict[str, list[str]],
    debug_count: int,
    selection_seed: str,
) -> dict[str, Any]:
    if debug_count <= 0:
        raise ValueError("debug-speakers must be positive")
    all_reserved = {
        speaker
        for speakers in reserved.values()
        for speaker in speakers
        if isinstance(speaker, str)
    }
    duplicates = {
        speaker
        for speakers in reserved.values()
        if isinstance(speakers, list)
        for speaker in speakers
        if sum(speaker in group for group in reserved.values() if isinstance(group, list)) > 1
    }
    if duplicates:
        raise ValueError(f"reserved speaker groups overlap: {sorted(duplicates)}")

    unique_archive = sorted(set(archive_speakers))
    if len(unique_archive) != len(archive_speakers):
        raise ValueError("archive speaker list contains duplicates")
    candidates = sorted(set(unique_archive) - all_reserved)
    if len(candidates) < debug_count:
        raise ValueError(f"only {len(candidates)} tuning candidates available; need {debug_count}")

    ranked = sorted(
        candidates,
        key=lambda speaker: hashlib.sha256(f"{selection_seed}:{speaker}".encode()).hexdigest(),
    )
    debug = ranked[:debug_count]
    full = ranked[debug_count:]
    return {
        "schema_version": 1,
        "phase": "5.0",
        "selection_seed": selection_seed,
        "selection_method": "sha256(selection_seed:speaker), ascending",
        "archive_speaker_count": len(unique_archive),
        "reserved_speakers": reserved,
        "reserved_speaker_count": len(all_reserved),
        "tuning_candidate_count": len(candidates),
        "tiers": {
            "debug": {
                "speaker_count": len(debug),
                "speakers": debug,
                "purpose": "training pipeline and recipe verification",
            },
            "full": {
                "speaker_count": len(full),
                "speakers": full,
                "purpose": "complete tuning after debug tier passes",
            },
        },
        "constraints": {
            "reserved_speakers_excluded": True,
            "evaluation_split": "phase3_full_dev",
            "heldout_split": "phase3_full_test",
            "speaker_disjoint": True,
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"output already exists: {output} (use --force to replace)")

    reserved = read_reserved_speakers(args.reserved_summary.resolve())
    archive_speakers = read_archive_speakers(args.archive.resolve())
    plan = build_speaker_plan(
        archive_speakers,
        reserved=reserved,
        debug_count=args.debug_speakers,
        selection_seed=args.selection_seed,
    )
    plan["archive"] = str(args.archive.resolve())
    plan["reserved_summary"] = str(args.reserved_summary.resolve())
    plan["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        f"plan={output} archive_speakers={plan['archive_speaker_count']} "
        f"reserved={plan['reserved_speaker_count']} "
        f"debug={plan['tiers']['debug']['speaker_count']} "
        f"full={plan['tiers']['full']['speaker_count']}"
    )


if __name__ == "__main__":
    main()
