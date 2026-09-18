"""Command-line entry points for Phase 1 enrollment and file-stream inference."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .audio.io import load_audio
from .audio.stream import FileStream
from .config import PROJECT_ROOT, load_config
from .logging_utils import setup_logging
from .models.backends import (
    FunasrPrefixAsr,
    FunasrPunctuator,
    FunasrSpeakerEncoder,
    FunasrVad,
)
from .models.streaming_vad import FunasrStreamingVad
from .pipeline.event_io import CompositeEventSink, JsonlEventWriter, TranscriptWriter
from .pipeline.online_pipeline import OnlineRealtimePipeline
from .pipeline.realtime_pipeline import RealtimePipeline
from .speaker.enrollment import (
    build_speaker_profile,
    load_profile,
    profile_embedding,
    save_profile,
)
from .vad.stream_vad import VadConfigValues


def _device(args: argparse.Namespace, cfg) -> str:
    if getattr(args, "device", None):
        cfg.project.device = args.device
    return cfg.project.resolved_device()


def _require_model_dir(path: str | Path, model_key: str) -> Path:
    p = Path(path)
    if not p.exists() or not any(p.iterdir()):
        raise FileNotFoundError(
            f"{model_key} model directory is empty or missing: {p}. "
            "Run: python scripts/download_models.py --model all"
        )
    return p


def _thresholds(args: argparse.Namespace, cfg) -> tuple[float, float]:
    accept = (
        args.accept_threshold if args.accept_threshold is not None else cfg.speaker.accept_threshold
    )
    reject = (
        args.reject_threshold if args.reject_threshold is not None else cfg.speaker.reject_threshold
    )
    if accept is None or reject is None:
        raise ValueError(
            "speaker thresholds are required. Set speaker.accept_threshold and "
            "speaker.reject_threshold in YAML, or pass --accept-threshold and "
            "--reject-threshold on the command line."
        )
    if reject >= accept:
        raise ValueError(
            f"reject threshold ({reject}) must be lower than accept threshold ({accept})"
        )
    return float(accept), float(reject)


def _make_run_id() -> str:
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def cmd_enroll(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = _device(args, cfg)
    setup_logging(args.log_file, level=args.log_level)
    _require_model_dir(cfg.speaker.model_dir, "speaker")
    encoder = FunasrSpeakerEncoder(cfg.speaker.model_dir, device=device)
    profile = build_speaker_profile(
        args.speaker_id,
        args.audio,
        encoder,
        model_id=cfg.speaker.model_id,
        sample_rate=cfg.project.sample_rate,
    )
    output = save_profile(profile, args.output)
    print(
        f"enrolled speaker={profile.speaker_id} "
        f"utterances={profile.num_enroll_utterances} "
        f"duration={profile.enroll_duration:.2f}s profile={output}"
    )


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.chunk_phase is not None and not 0 <= args.chunk_phase < cfg.audio.chunk_size:
        raise ValueError(f"chunk-phase must be in [0, {cfg.audio.chunk_size}) seconds")
    device = _device(args, cfg)
    setup_logging(args.log_file, level=args.log_level)
    accept, reject = _thresholds(args, cfg)

    _require_model_dir(cfg.vad.model_dir, "VAD")
    _require_model_dir(cfg.speaker.model_dir, "speaker")
    _require_model_dir(cfg.asr.model_dir, "ASR")
    punctuation = None
    if cfg.postprocess.punctuation.enabled and not args.no_punctuation:
        _require_model_dir(cfg.postprocess.punctuation.model_dir, "punctuation")
        punctuation = FunasrPunctuator(cfg.postprocess.punctuation.model_dir, device=device)

    waveform, sample_rate = load_audio(args.input, target_sr=cfg.project.sample_rate)
    if waveform.size == 0:
        raise ValueError(f"audio file is empty: {args.input}")
    stream = FileStream(
        waveform=waveform,
        sample_rate=sample_rate,
        chunk_size=cfg.audio.chunk_size,
        realtime_sleep=args.realtime,
        phase_period=cfg.audio.chunk_size if args.chunk_phase is not None else None,
        phase_offset=args.chunk_phase or 0.0,
    )

    run_id = _make_run_id()
    output_root = (
        Path(args.output_dir) if args.output_dir else PROJECT_ROOT / cfg.output.run_dir
    ).resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    shutil.copyfile(args.config, run_dir / "config.snapshot.yaml")

    events_path = run_dir / "events.jsonl"
    transcript_path = run_dir / "transcript.txt"
    sink = CompositeEventSink(JsonlEventWriter(events_path), TranscriptWriter(transcript_path))
    profile = load_profile(args.profile)
    enrolled = profile_embedding(profile)
    speaker_encoder = FunasrSpeakerEncoder(cfg.speaker.model_dir, device=device)
    asr_backend = FunasrPrefixAsr(
        cfg.asr.model_dir,
        decode_chunk=cfg.asr.chunk_size,
        device=device,
        sample_rate=sample_rate,
        use_itn=True,
    )

    if args.batch:
        pipeline = RealtimePipeline(
            run_id=run_id,
            audio_path=str(args.input),
            vad=FunasrVad(cfg.vad.model_dir, device=device),
            speaker_encoder=speaker_encoder,
            asr=asr_backend,
            enrolled_embedding=enrolled,
            accept_threshold=accept,
            reject_threshold=reject,
            event_sink=sink,
            punctuator=punctuation,
            sample_rate=sample_rate,
            asr_chunk_size=cfg.asr.chunk_size,
            embedding_window=cfg.speaker.embedding_window,
            embedding_hop=cfg.speaker.embedding_hop,
            first_embedding_window=cfg.speaker.first_embedding_window,
            speech_pad=cfg.vad.speech_pad,
            min_vad_duration=cfg.speaker.min_segment_duration,
            max_vad_duration=cfg.vad.max_segment_length,
            vad_merge_gap=cfg.vad.merge_gap,
            vad_min_silence_duration=cfg.vad.min_silence_duration,
        )
    else:
        pipeline = OnlineRealtimePipeline(
            run_id=run_id,
            audio_path=str(args.input),
            vad=FunasrStreamingVad(cfg.vad.model_dir, device=device),
            speaker_encoder=speaker_encoder,
            asr=asr_backend,
            enrolled_embedding=enrolled,
            accept_threshold=accept,
            reject_threshold=reject,
            event_sink=sink,
            punctuator=punctuation,
            sample_rate=sample_rate,
            asr_chunk_size=cfg.asr.chunk_size,
            embedding_window=cfg.speaker.embedding_window,
            embedding_hop=cfg.speaker.embedding_hop,
            min_vad_duration=cfg.speaker.min_segment_duration,
            max_vad_duration=cfg.vad.max_segment_length,
            vad_min_silence_duration=cfg.vad.min_silence_duration,
            declared_duration=len(waveform) / sample_rate,
            emit_unconfirmed_partials=cfg.speaker.emit_unconfirmed_partials,
            stream_start_offset=cfg.audio.chunk_size if args.realtime else 0.0,
        )

    try:
        if args.start_delay > 0:
            time.sleep(args.start_delay)
        if args.batch:
            chunks = [chunk for _, chunk in stream]
            stats = pipeline.process(np.concatenate(chunks, axis=0))
        else:
            consumed_samples = 0
            for _, chunk in stream:
                consumed_samples += len(chunk)
                pipeline.push_chunk(chunk, is_final=consumed_samples >= len(waveform))
            stats = pipeline.finish()
    finally:
        sink.close()

    (run_dir / "summary.json").write_text(
        json.dumps(asdict(stats), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"run={run_id} segments={stats.vad_segments} "
        f"target={stats.target_committed} suppressed={stats.non_target_suppressed} "
        f"pending={stats.pending_suppressed} rtf={stats.rtf:.3f}"
    )
    print(f"events={events_path}")
    print(f"transcript={transcript_path}")
    print(f"summary={run_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Target-speaker streaming ASR")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enroll = subparsers.add_parser("enroll", help="create a speaker profile")
    enroll.add_argument("--speaker-id", required=True)
    enroll.add_argument("--audio", nargs="+", required=True, type=Path)
    enroll.add_argument("--output", required=True, type=Path)
    enroll.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    enroll.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"])
    enroll.add_argument("--log-level", default="INFO")
    enroll.add_argument("--log-file", type=Path)
    enroll.set_defaults(func=cmd_enroll)

    run = subparsers.add_parser("run", help="run file-simulated realtime inference")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument(
        "--batch",
        action="store_true",
        help="use the Phase 1 batch pipeline instead of online processing",
    )
    run.add_argument("--profile", required=True, type=Path)
    run.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    run.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"])
    run.add_argument("--accept-threshold", type=float)
    run.add_argument("--reject-threshold", type=float)
    run.add_argument("--realtime", action="store_true")
    run.add_argument(
        "--start-delay",
        type=float,
        default=0.0,
        help="wait after model setup before feeding the first audio chunk",
    )
    run.add_argument(
        "--chunk-phase",
        type=float,
        help="align the first realtime chunk to this offset within the chunk period",
    )
    run.add_argument("--no-punctuation", action="store_true")
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--log-level", default="INFO")
    run.add_argument("--log-file", type=Path)
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
