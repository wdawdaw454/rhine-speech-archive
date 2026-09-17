"""Run a Windows-local fast dictation demo and report CER plus speed."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.io import load_audio
from src.audio.microphone import MicrophoneStream, write_status_file
from src.audio.stream import FileStream
from src.data.normalize import normalize_chinese_text
from src.evaluation.cer import char_edit_distance
from src.models.backends import FunasrPrefixAsr
from src.models.sensevoice_onnx import OnnxSenseVoiceAsr
from src.pipeline.event_io import CompositeEventSink, EventSink, JsonlEventWriter
from src.pipeline.events import EventType, PipelineEvent
from src.pipeline.fast_dictation import FastDictationPipeline

DEFAULT_REFERENCE = "房地产行业如今的景气度与此前相比已经不可同日而语"


class ConsoleDemoSink:
    def __init__(self) -> None:
        self._active = False

    def write(self, event: PipelineEvent) -> None:
        if event.type == EventType.ASR_PARTIAL and event.text:
            print(f"partial: {event.text}", flush=True)
            self._active = True
        elif event.type == EventType.CAPTION_COMMIT and event.text:
            print(f"commit  : {event.text}", flush=True)
            self._active = False

    def close(self) -> None:
        if self._active:
            print()


class DemoTranscriptSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("w", encoding="utf-8", newline="\n")

    def write(self, event: PipelineEvent) -> None:
        if event.type != EventType.CAPTION_COMMIT or not event.text:
            return
        self.handle.write(f"[{event.start or 0:.3f} -> {event.end or 0:.3f}] {event.text}\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Windows local fast dictation demo")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="WAV file for deterministic validation")
    source.add_argument("--microphone", action="store_true", help="read the default microphone")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / "models/sensevoice_small_int8_bundle"
    )
    parser.add_argument(
        "--backend",
        choices=("onnx", "funasr"),
        default="onnx",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--input-chunk", type=float, default=0.10)
    parser.add_argument("--decode-interval", type=float, default=0.50)
    parser.add_argument("--preroll", type=float, default=0.30)
    parser.add_argument("--speech-rms", type=float, default=0.005)
    parser.add_argument("--min-speech", type=float, default=0.20)
    parser.add_argument("--silence", type=float, default=0.60)
    parser.add_argument("--max-segment", type=float, default=12.0)
    parser.add_argument("--realtime", action="store_true", help="use file audio clock")
    parser.add_argument("--intra-op-threads", type=int, default=4)
    parser.add_argument(
        "--itn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable inverse text normalization (default: enabled)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs/phase5_12/local_demo"
    )
    return parser


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


def _backend(args: argparse.Namespace):
    if args.backend == "onnx":
        if args.device.lower() not in ("cpu", "default", "auto"):
            raise SystemExit("the local demo uses ONNX Runtime CPU execution")
        return OnnxSenseVoiceAsr.from_bundle(
            args.model_dir,
            decode_chunk=args.decode_interval,
            sample_rate=args.sample_rate,
            intra_op_threads=args.intra_op_threads,
            textnorm_id=14 if args.itn else 15,
        )
    return FunasrPrefixAsr(
        args.model_dir,
        decode_chunk=args.decode_interval,
        device=args.device,
        sample_rate=args.sample_rate,
        use_itn=args.itn,
    )


def _load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _text_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    normalized_reference = normalize_chinese_text(reference)
    normalized_hypothesis = normalize_chinese_text(hypothesis)
    errors = char_edit_distance(normalized_reference, normalized_hypothesis)
    return {
        "reference": normalized_reference,
        "hypothesis": normalized_hypothesis,
        "cer": errors / len(normalized_reference) if normalized_reference else 0.0,
        "errors": errors,
        "reference_chars": len(normalized_reference),
    }


def _speed_metrics(events: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, Any]:
    partials = [event for event in events if event["type"] == "asr_partial" and event.get("text")]
    partial_times = [float(event["monotonic_time"]) for event in partials]
    intervals = [right - left for left, right in zip(partial_times, partial_times[1:])]
    commits = [event for event in events if event["type"] == "caption_commit" and event.get("text")]
    return {
        "first_partial_wall": partial_times[0] if partial_times else None,
        "first_partial_audio_time": partials[0]["audio_time"] if partials else None,
        "partial_count": len(partials),
        "partial_interval_p50": _percentile(intervals, 0.50),
        "partial_interval_p95": _percentile(intervals, 0.95),
        "commit_count": len(commits),
        "backend_seconds": stats["backend_seconds"],
        "audio_seconds": stats["accepted_seconds"],
        "backend_rtf": (
            stats["backend_seconds"] / stats["accepted_seconds"]
            if stats["accepted_seconds"]
            else 0.0
        ),
    }


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    if args.duration <= 0:
        raise SystemExit("duration must be positive")
    if args.countdown < 0:
        raise SystemExit("countdown must be non-negative")
    if not args.model_dir.is_dir():
        raise SystemExit(f"model directory not found: {args.model_dir}")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir.resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    transcript_path = output_dir / "transcript.txt"
    recording_path = output_dir / "recording.wav"
    status_path = output_dir / "microphone_status.txt"

    if args.input is not None:
        waveform, sample_rate = load_audio(args.input, target_sr=args.sample_rate)
        stream = FileStream(
            waveform=waveform,
            sample_rate=sample_rate,
            chunk_size=args.input_chunk,
            realtime_sleep=args.realtime,
        )
        source_mode = "file"
        audio_path = str(args.input.resolve())
        duration = len(waveform) / sample_rate
    else:
        stream = MicrophoneStream(
            sample_rate=args.sample_rate,
            blocksize=max(1, int(round(args.input_chunk * args.sample_rate))),
            duration=args.duration,
        )
        source_mode = "microphone"
        audio_path = "microphone"
        duration = args.duration

    event_sink: EventSink = CompositeEventSink(
        ConsoleDemoSink(), JsonlEventWriter(events_path), DemoTranscriptSink(transcript_path)
    )
    torch = None
    cuda_index = 0
    if str(args.device).startswith("cuda"):
        try:
            import torch as torch_module

            torch = torch_module
            if ":" in str(args.device):
                cuda_index = int(str(args.device).split(":", 1)[1])
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(cuda_index)
        except ImportError:
            pass

    load_started = time.perf_counter()
    backend = _backend(args)
    load_seconds = time.perf_counter() - load_started
    pipeline = FastDictationPipeline(
        run_id=run_id,
        backend=backend,
        event_sink=event_sink,
        audio_path=audio_path,
        sample_rate=args.sample_rate,
        input_chunk=args.input_chunk,
        decode_interval=args.decode_interval,
        preroll=args.preroll,
        speech_rms_threshold=args.speech_rms,
        min_speech_duration=args.min_speech,
        silence_duration=args.silence,
        max_segment_duration=args.max_segment,
    )

    recorded_chunks: list[np.ndarray] = []
    print("=" * 48, flush=True)
    print(f"请朗读：{args.reference}", flush=True)
    print("请在倒计时结束后开始说话，停顿约一秒会提交文本。", flush=True)
    print("=" * 48, flush=True)
    remaining = args.countdown
    while remaining > 0:
        print(f"{ceil_seconds(remaining)}...", flush=True)
        delay = min(1.0, remaining)
        time.sleep(delay)
        remaining -= delay

    started = time.perf_counter()
    try:
        for _, chunk in stream:
            if source_mode == "microphone":
                recorded_chunks.append(np.asarray(chunk, dtype=np.float32).copy())
            pipeline.push_chunk(chunk)
        stats = pipeline.finish()
    except KeyboardInterrupt:
        stats = pipeline.finish()
    finally:
        closer = getattr(event_sink, "close", None)
        if callable(closer):
            closer()
        if source_mode == "microphone":
            write_status_file(status_path, stream.statuses)
            if recorded_chunks:
                import soundfile as sf

                recording = np.concatenate(recorded_chunks)
                sf.write(recording_path, recording, args.sample_rate, subtype="PCM_16")

    wall_seconds = time.perf_counter() - started
    events = _load_events(events_path)
    commits = [event for event in events if event["type"] == "caption_commit" and event.get("text")]
    hypothesis = "".join(str(event.get("text", "")) for event in commits)
    text_metrics = _text_metrics(args.reference, hypothesis)
    speed_metrics = _speed_metrics(events, asdict(stats))

    summary = {
        "schema_version": 1,
        "phase": "5.14",
        "run_id": run_id,
        "source_mode": source_mode,
        "backend": args.backend,
        "model_dir": str(args.model_dir.resolve()),
        "device": args.device,
        "itn_enabled": args.itn,
        "duration_seconds": round(duration, 6),
        "wall_seconds": round(wall_seconds, 6),
        "model_load_seconds": round(load_seconds, 6),
        "peak_gpu_memory_gb": (
            round(torch.cuda.max_memory_reserved(cuda_index) / 1024**3, 4)
            if torch is not None
            else None
        ),
        "parameters": {
            "sample_rate": args.sample_rate,
            "input_chunk": args.input_chunk,
            "decode_interval": args.decode_interval,
            "speech_rms_threshold": args.speech_rms,
            "silence_duration": args.silence,
            "max_segment_duration": args.max_segment,
            "intra_op_threads": args.intra_op_threads,
        },
        "text": text_metrics,
        "speed": speed_metrics,
        "stats": asdict(stats),
        "events_path": str(events_path),
        "transcript_path": str(transcript_path),
        "recording_path": str(recording_path) if source_mode == "microphone" else None,
    }
    summary_path = output_dir / "demo_report.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n===== DEMO REPORT =====", flush=True)
    print(f"CER                : {text_metrics['cer']:.4f}", flush=True)
    print(f"reference          : {text_metrics['reference']}", flush=True)
    print(f"hypothesis         : {text_metrics['hypothesis']}", flush=True)
    first_partial = speed_metrics["first_partial_wall"]
    first_partial_text = f"{first_partial:.3f}s" if first_partial is not None else "N/A"
    print(f"first partial      : {first_partial_text}", flush=True)
    print(
        "partial interval   : "
        f"P50 {speed_metrics['partial_interval_p50']:.3f}s / "
        f"P95 {speed_metrics['partial_interval_p95']:.3f}s",
        flush=True,
    )
    print(f"backend RTF        : {speed_metrics['backend_rtf']:.3f}", flush=True)
    print(f"model load         : {summary['model_load_seconds']:.3f}s", flush=True)
    if summary["peak_gpu_memory_gb"] is not None:
        print(f"peak GPU memory    : {summary['peak_gpu_memory_gb']:.3f} GB", flush=True)
    print(f"report             : {summary_path}", flush=True)
    return summary


def ceil_seconds(value: float) -> int:
    return int(np.ceil(value))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_demo(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
