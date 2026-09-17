"""Create speaker profiles for every target in an AISHELL full split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.config import load_config
from src.data.manifest import read_jsonl
from src.models.backends import FunasrSpeakerEncoder
from src.speaker.enrollment import build_speaker_profile, save_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enroll AISHELL full-split targets")
    parser.add_argument(
        "--input-root", type=Path, default=PROJECT_ROOT / "data/processed/aishell_full"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "profiles/aishell_full")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"])
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary_path = args.input_root.resolve() / "split_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"split summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pairs = [pair for split in summary["splits"].values() for pair in split["pairs"]]

    cfg = load_config(args.config)
    if args.device:
        cfg.project.device = args.device
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = FunasrSpeakerEncoder(cfg.speaker.model_dir, device=cfg.project.resolved_device())

    created: list[dict[str, str]] = []
    for pair in pairs:
        speaker = str(pair["target"])
        profile_path = output_dir / f"{speaker}.json"
        if profile_path.exists() and not args.overwrite:
            print(f"skip existing profile={profile_path}")
            created.append({"speaker": speaker, "profile": str(profile_path)})
            continue
        manifest_path = Path(pair["enroll_manifest"])
        audio_paths = [
            (manifest_path.parent / str(entry.audio_path)).resolve()
            for entry in read_jsonl(manifest_path)
        ]
        profile = build_speaker_profile(
            speaker,
            audio_paths,
            encoder,
            model_id=cfg.speaker.model_id,
            sample_rate=cfg.project.sample_rate,
        )
        save_profile(profile, profile_path)
        print(
            f"enrolled speaker={speaker} duration={profile.enroll_duration:.2f}s "
            f"profile={profile_path}"
        )
        created.append({"speaker": speaker, "profile": str(profile_path)})

    index_path = output_dir / "profiles.json"
    index_path.write_text(
        json.dumps({"profiles": created}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"profile_index={index_path}")


if __name__ == "__main__":
    main()
