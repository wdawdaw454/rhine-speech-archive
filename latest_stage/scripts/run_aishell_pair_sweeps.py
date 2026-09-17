"""Build and sweep a multi-pair AISHELL-1 experiment in one command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.experiments.aggregate import aggregate_by_threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AISHELL full-split pair sweeps")
    parser.add_argument(
        "--input-root", type=Path, default=PROJECT_ROOT / "data/processed/aishell_full"
    )
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/phase3_latency.yaml")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"], default="cuda:1"
    )
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--reject-threshold", type=float, default=-1.0)
    parser.add_argument(
        "--experiment-root", type=Path, default=PROJECT_ROOT / "data/experiments/aishell_full"
    )
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "outputs/experiments/aishell_full"
    )
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run(command: list[str]) -> None:
    printable = " ".join(str(item) for item in command)
    print(f"RUN {printable}", flush=True)
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.thresholds:
        raise SystemExit("at least one threshold is required")
    if any(threshold <= args.reject_threshold for threshold in args.thresholds):
        raise SystemExit("every accept threshold must be greater than reject threshold")

    summary = _read_json(args.input_root.resolve() / "split_summary.json")
    split_names = ("dev", "test") if args.split == "all" else (args.split,)
    pairs = [pair for name in split_names for pair in summary["splits"][name]["pairs"]]
    if not pairs:
        raise SystemExit(f"no pairs found for split: {args.split}")

    python = sys.executable
    mode = "realtime" if args.realtime else "offline"
    all_reports: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    for pair in pairs:
        split = str(pair["split"])
        target = str(pair["target"])
        interferer = str(pair["interferer"])
        pair_name = f"{target}_{interferer}"
        dataset_root = args.experiment_root.resolve() / split / pair_name
        sweep_root = args.output_root.resolve() / mode / split / pair_name

        if not dataset_root.exists():
            _run(
                [
                    python,
                    str(PROJECT_ROOT / "scripts/build_experiment_mixtures.py"),
                    "--config",
                    str(pair["config"]),
                    "--manifest",
                    str(summary["splits"][split]["source_manifest"]),
                    "--output-root",
                    str(dataset_root),
                ]
            )
        else:
            print(f"skip existing dataset={dataset_root}")

        command = [
            python,
            str(PROJECT_ROOT / "scripts/sweep_thresholds.py"),
            "--dataset-index",
            str(dataset_root / "dataset_index.json"),
            "--profile",
            str(pair["profile"]),
            "--config",
            str(args.config),
            "--device",
            args.device,
            "--accept-thresholds",
            *[str(value) for value in args.thresholds],
            "--reject-threshold",
            str(args.reject_threshold),
            "--output-dir",
            str(sweep_root),
        ]
        if args.realtime:
            command.append("--realtime")
        if args.resume:
            command.append("--resume")
        _run(command)

        reports = _read_json(sweep_root / "reports.json")
        for report in reports:
            report.setdefault("experiment", {})["interferer_speaker"] = interferer
        all_reports.extend(reports)
        pair_summary = _read_json(sweep_root / "summary.json")
        pair_results.append(
            {
                "split": split,
                "target": target,
                "interferer": interferer,
                "dataset_index": str(dataset_root / "dataset_index.json"),
                "sweep_root": str(sweep_root),
                "threshold_summaries": pair_summary["threshold_summaries"],
            }
        )

    global_root = (
        args.output_root.resolve() / mode / split_names[0]
        if len(split_names) == 1
        else args.output_root.resolve() / mode / "all"
    )
    _write_json(global_root / "reports.json", all_reports)
    _write_json(
        global_root / "summary.json",
        {
            "split": args.split,
            "mode": mode,
            "config": str(args.config.resolve()),
            "realtime": args.realtime,
            "runs": len(all_reports),
            "pairs": pair_results,
            "threshold_summaries": aggregate_by_threshold(all_reports),
        },
    )
    print(f"aggregate_reports={global_root / 'reports.json'}")
    print(f"aggregate_summary={global_root / 'summary.json'}")
    for result in aggregate_by_threshold(all_reports):
        metrics = result["target_metrics"]
        print(
            f"accept={result['thresholds']['accept']:.2f} "
            f"cer={metrics['target_cer']:.3f} "
            f"recall={metrics['target_recall']:.3f} "
            f"precision={metrics['target_precision']:.3f} "
            f"leak={metrics['non_target_leakage_rate']:.3f} "
            f"runs={result['runs']}"
        )


if __name__ == "__main__":
    main()
