"""Run a Phase 3 threshold sweep over an experiment dataset index."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.io import load_audio
from src.audio.stream import FileStream
from src.config import load_config
from src.evaluation.report import build_evaluation_report, read_references
from src.experiments.aggregate import aggregate_by_threshold
from src.models.backends import FunasrSpeakerEncoder
from src.models.sensevoice_onnx import OnnxSenseVoiceAsr
from src.models.streaming_vad import FunasrStreamingVad
from src.pipeline.event_io import (
    CompositeEventSink,
    JsonlEventWriter,
    TranscriptWriter,
    read_events,
)
from src.pipeline.online_pipeline import OnlineRealtimePipeline
from src.speaker.enrollment import load_profile, profile_embedding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep speaker thresholds across mixtures")
    parser.add_argument("--dataset-index", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"])
    parser.add_argument(
        "--accept-thresholds",
        nargs="+",
        type=float,
        default=[0.30, 0.40, 0.50, 0.55, 0.58, 0.60],
    )
    parser.add_argument("--reject-threshold", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _require_model_dir(path: str | Path, name: str) -> Path:
    model_dir = Path(path)
    if not model_dir.exists() or not any(model_dir.iterdir()):
        raise SystemExit(f"{name} model directory is missing or empty: {model_dir}")
    return model_dir


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _decorate(
    report: dict[str, Any],
    *,
    item: dict[str, Any],
    accept: float,
    reject: float,
) -> dict[str, Any]:
    decorated = dict(report)
    decorated["experiment"] = {
        "scenario": item["scenario"],
        "seed": item["seed"],
        "gap": item["gap"],
        "overlap": item["overlap"],
        "target_snr_db": item["target_snr_db"],
    }
    decorated["thresholds"] = {"accept": accept, "reject": reject}
    return decorated


def _run_one(
    *,
    pipeline_args: dict[str, Any],
    item: dict[str, Any],
    accept: float,
    reject: float,
    run_dir: Path,
    realtime: bool,
) -> tuple[Any, dict[str, Any]]:
    waveform = pipeline_args.pop("waveform")
    sample_rate = pipeline_args.pop("sample_rate")
    mixture_path = pipeline_args.pop("mixture_path")
    reference_path = pipeline_args.pop("reference_path")
    target_speaker = pipeline_args.pop("target_speaker")
    input_chunk_size = pipeline_args.pop("input_chunk_size")
    stream = FileStream(
        waveform=waveform,
        sample_rate=sample_rate,
        chunk_size=input_chunk_size,
        realtime_sleep=realtime,
    )
    events_path = run_dir / "events.jsonl"
    transcript_path = run_dir / "transcript.txt"
    sink = CompositeEventSink(JsonlEventWriter(events_path), TranscriptWriter(transcript_path))
    run_id = f"phase3_{item['scenario']}_seed{item['seed']}_a{accept:.2f}"
    pipeline_args["vad"].reset()
    pipeline = OnlineRealtimePipeline(
        run_id=run_id,
        audio_path=str(mixture_path),
        accept_threshold=accept,
        reject_threshold=reject,
        event_sink=sink,
        declared_duration=len(waveform) / sample_rate,
        **pipeline_args,
        stream_start_offset=input_chunk_size if realtime else 0.0,
    )
    try:
        consumed = 0
        for _, chunk in stream:
            consumed += len(chunk)
            pipeline.push_chunk(chunk, is_final=consumed >= len(waveform))
        stats = pipeline.finish()
    finally:
        sink.close()

    _write_json(run_dir / "summary.json", asdict(stats))
    report = build_evaluation_report(
        read_events(events_path),
        read_references(reference_path),
        target_speaker=target_speaker,
    )
    report = _decorate(report, item=item, accept=accept, reject=reject)
    _write_json(run_dir / "evaluation.json", report)
    return stats, report


def _report_csv_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        rows.append(
            {
                "scenario": report["experiment"]["scenario"],
                "seed": report["experiment"]["seed"],
                "accept": report["thresholds"]["accept"],
                "reject": report["thresholds"]["reject"],
                "target_cer": report["target_metrics"]["target_cer"],
                "target_recall": report["target_metrics"]["target_recall"],
                "target_precision": report["target_metrics"]["target_precision"],
                "non_target_leakage": report["target_metrics"]["non_target_leakage_rate"],
                "rtf": report["rtf"],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.accept_thresholds:
        raise SystemExit("at least one accept threshold is required")
    thresholds = sorted(set(args.accept_thresholds))
    if any(threshold <= args.reject_threshold for threshold in thresholds):
        raise SystemExit("every accept threshold must be greater than reject threshold")

    dataset_index_path = args.dataset_index.resolve()
    dataset_root = dataset_index_path.parent
    dataset = _read_json(dataset_index_path)
    if int(dataset.get("version", 0)) != 1:
        raise SystemExit(f"unsupported dataset index version: {dataset_index_path}")
    items = dataset.get("items", [])
    if not items:
        raise SystemExit(f"dataset index has no items: {dataset_index_path}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, output_dir / "config.snapshot.yaml")

    cfg = load_config(args.config)
    if args.device:
        cfg.project.device = args.device
    device = cfg.project.resolved_device()
    _require_model_dir(cfg.speaker.model_dir, "speaker")
    _require_model_dir(cfg.asr.model_dir, "ASR")
    _require_model_dir(cfg.vad.model_dir, "VAD")

    profile = load_profile(args.profile)
    if profile.speaker_id != dataset["target_speaker"]:
        raise SystemExit(
            f"profile speaker {profile.speaker_id} does not match dataset target "
            f"{dataset['target_speaker']}"
        )
    enrolled = profile_embedding(profile)
    speaker_encoder = FunasrSpeakerEncoder(cfg.speaker.model_dir, device=device)
    asr = OnnxSenseVoiceAsr.from_bundle(
        cfg.asr.model_dir,
        decode_chunk=cfg.asr.chunk_size,
        sample_rate=cfg.project.sample_rate,
    )
    vad = FunasrStreamingVad(cfg.vad.model_dir, device=device)

    reports: list[dict[str, Any]] = []
    completed = 0
    total_runs = len(items) * len(thresholds)
    for item in items:
        mixture_path = dataset_root / str(item["mixture"])
        reference_path = dataset_root / str(item["references"])
        waveform, sample_rate = load_audio(mixture_path, target_sr=cfg.project.sample_rate)
        if sample_rate != cfg.project.sample_rate:
            raise SystemExit(
                f"unexpected sample rate for {mixture_path}: {sample_rate} != "
                f"{cfg.project.sample_rate}"
            )
        for accept in thresholds:
            run_dir = output_dir / f"accept_{accept:.2f}" / f"{item['scenario']}_seed{item['seed']}"
            evaluation_path = run_dir / "evaluation.json"
            if evaluation_path.exists():
                report = _decorate(
                    _read_json(evaluation_path),
                    item=item,
                    accept=accept,
                    reject=args.reject_threshold,
                )
                reports.append(report)
                completed += 1
                print(f"skip run={run_dir}")
                continue
            if run_dir.exists():
                raise SystemExit(f"incomplete run already exists: {run_dir}")
            staging_dir = run_dir.with_name(f"{run_dir.name}.partial")
            if staging_dir.exists():
                raise SystemExit(f"incomplete run staging already exists: {staging_dir}")
            staging_dir.mkdir(parents=True)

            pipeline_args: dict[str, Any] = {
                "vad": vad,
                "speaker_encoder": speaker_encoder,
                "asr": asr,
                "enrolled_embedding": enrolled,
                "sample_rate": sample_rate,
                "asr_chunk_size": cfg.asr.chunk_size,
                "embedding_window": cfg.speaker.embedding_window,
                "embedding_hop": cfg.speaker.embedding_hop,
                "first_embedding_window": cfg.speaker.first_embedding_window,
                "min_vad_duration": cfg.speaker.min_segment_duration,
                "max_vad_duration": cfg.vad.max_segment_length,
                "vad_min_silence_duration": cfg.vad.min_silence_duration,
                "input_chunk_size": cfg.audio.chunk_size,
                "emit_unconfirmed_partials": cfg.speaker.emit_unconfirmed_partials,
                "waveform": waveform,
                "mixture_path": mixture_path,
                "reference_path": reference_path,
                "target_speaker": dataset["target_speaker"],
            }
            stats, report = _run_one(
                pipeline_args=pipeline_args,
                item=item,
                accept=accept,
                reject=args.reject_threshold,
                run_dir=staging_dir,
                realtime=args.realtime,
            )
            shutil.move(str(staging_dir), str(run_dir))
            reports.append(report)
            completed += 1
            metrics = report["target_metrics"]
            print(
                f"run={completed}/{total_runs} scenario={item['scenario']} "
                f"seed={item['seed']} accept={accept:.2f} cer={metrics['target_cer']:.3f} "
                f"recall={metrics['target_recall']:.3f} "
                f"leak={metrics['non_target_leakage_rate']:.3f} rtf={stats.rtf:.3f}"
            )

    summaries = aggregate_by_threshold(reports)
    _write_json(output_dir / "reports.json", reports)
    _write_json(
        output_dir / "summary.json",
        {
            "dataset_index": str(dataset_index_path),
            "profile": str(args.profile.resolve()),
            "realtime": args.realtime,
            "runs": len(reports),
            "threshold_summaries": summaries,
        },
    )
    rows = _report_csv_rows(reports)
    with (output_dir / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"reports={output_dir / 'reports.json'}")
    print(f"summary={output_dir / 'summary.json'}")
    for summary in summaries:
        metrics = summary["target_metrics"]
        print(
            f"accept={summary['thresholds']['accept']:.2f} "
            f"cer={metrics['target_cer']:.3f} recall={metrics['target_recall']:.3f} "
            f"precision={metrics['target_precision']:.3f} "
            f"leak={metrics['non_target_leakage_rate']:.3f} runs={summary['runs']}"
        )


if __name__ == "__main__":
    main()
