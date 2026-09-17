"""Evaluate a pipeline run from events.jsonl and a reference JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.evaluation.report import build_evaluation_report, read_references
from src.pipeline.event_io import read_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate target-speaker ASR events")
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--target-speaker", required=True)
    parser.add_argument("--min-overlap", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    events = read_events(args.events)
    references = read_references(args.reference)
    report = build_evaluation_report(
        events,
        references,
        target_speaker=args.target_speaker,
        min_overlap=args.min_overlap,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
