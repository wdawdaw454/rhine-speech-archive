"""Run the Phase 5.9 fast local dictation loop from a file or microphone."""

from __future__ import annotations

from dataclasses import asdict
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.io import load_audio
from src.audio.microphone import MicrophoneStream, write_status_file
from src.audio.stream import FileStream
from src.models.backends import FunasrPrefixAsr
from src.models.sensevoice_onnx import OnnxSenseVoiceAsr
from src.pipeline.event_io import CompositeEventSink, EventSink, JsonlEventWriter
from src.pipeline.events import EventType, PipelineEvent
from src.pipeline.fast_dictation import FastDictationPipeline


class ConsoleFastSink:
    def __init__(self) -> None:
        self._active_line = False

    def write(self, event: PipelineEvent) -> None:
        if event.type == EventType.ASR_PARTIAL and event.text:
            print(f"\rpartial: {event.text}", end="", flush=True)
            self._active_line = True
        elif event.type == EventType.CAPTION_COMMIT and event.text:
            if self._active_line:
                print("\r", end="", flush=True)
                self._active_line = False
            print(f"commit : {event.text}", flush=True)


class FastTranscriptSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8", newline="\n")

    def write(self, event: PipelineEvent) -> None:
        if event.type != EventType.CAPTION_COMMIT or not event.text:
            return
        self.handle.write(f"[{event.start or 0:.3f} -> {event.end or 0:.3f}] {event.text}\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _device(value: str) -> int | str | None:
    if value.lower() in ("auto", "default"):
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _build_backend(args: argparse.Namespace):
    if args.backend == "onnx":
        if args.device.lower() not in ("cpu", "default", "auto"):
            raise SystemExit("the ONNX backend currently supports CPU execution only")
        return OnnxSenseVoiceAsr.from_bundle(
            args.model_dir,
            decode_chunk=args.decode_interval,
            sample_rate=args.sample_rate,
        )
    return FunasrPrefixAsr(
        args.model_dir,
        decode_chunk=args.decode_interval,
        device=args.device,
        sample_rate=args.sample_rate,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rapid-partial local Chinese dictation")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="WAV/audio file for deterministic validation")
    source.add_argument(
        "--microphone", action="store_true", help="read from the default microphone"
    )
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/sensevoice_small")
    parser.add_argument("--backend", choices=("funasr", "onnx"), default="funasr")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--duration", type=float, help="microphone capture duration in seconds")
    parser.add_argument(
        "--microphone-device", type=_device, help="sounddevice device index or name"
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--input-chunk", type=float, default=0.10)
    parser.add_argument("--decode-interval", type=float, default=0.50)
    parser.add_argument("--preroll", type=float, default=0.30)
    parser.add_argument("--speech-rms", type=float, default=0.005)
    parser.add_argument("--min-speech", type=float, default=0.20)
    parser.add_argument("--silence", type=float, default=0.60)
    parser.add_argument("--max-segment", type=float, default=12.0)
    parser.add_argument("--realtime", action="store_true", help="sleep to audio time in file mode")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/phase5/fast_dictation",
    )
    return parser


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.input is not None:
        waveform, sample_rate = load_audio(args.input, target_sr=args.sample_rate)
        stream = FileStream(
            waveform=waveform,
            sample_rate=sample_rate,
            chunk_size=args.input_chunk,
            realtime_sleep=args.realtime,
        )
        audio_path = str(args.input.resolve())
        source_mode = "file"
    else:
        stream = MicrophoneStream(
            sample_rate=args.sample_rate,
            blocksize=max(1, int(round(args.input_chunk * args.sample_rate))),
            device=args.microphone_device,
            duration=args.duration,
        )
        audio_path = "microphone"
        source_mode = "microphone"

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir.resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    transcript_path = output_dir / "transcript.txt"
    status_path = output_dir / "microphone_status.txt"
    event_sink: EventSink = CompositeEventSink(
        ConsoleFastSink(),
        JsonlEventWriter(events_path),
        FastTranscriptSink(transcript_path),
    )

    backend = _build_backend(args)
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

    started = time.perf_counter()
    try:
        for _, chunk in stream:
            pipeline.push_chunk(chunk)
        stats = pipeline.finish()
    except KeyboardInterrupt:
        stats = pipeline.finish()
    finally:
        closer = getattr(event_sink, "close", None)
        if callable(closer):
            closer()
        if args.microphone:
            write_status_file(status_path, stream.statuses)

    summary = {
        "schema_version": 1,
        "phase": "5.9",
        "run_id": run_id,
        "source_mode": source_mode,
        "backend": args.backend,
        "audio_path": audio_path,
        "model_dir": str(args.model_dir.resolve()),
        "device": args.device,
        "parameters": {
            "sample_rate": args.sample_rate,
            "input_chunk": args.input_chunk,
            "decode_interval": args.decode_interval,
            "preroll": args.preroll,
            "speech_rms_threshold": args.speech_rms,
            "min_speech_duration": args.min_speech,
            "silence_duration": args.silence,
            "max_segment_duration": args.max_segment,
        },
        "wall_seconds": round(time.perf_counter() - started, 6),
        "stats": asdict(stats),
        "events_path": str(events_path),
        "transcript_path": str(transcript_path),
    }
    _write_json(output_dir / "summary.json", summary)
    print(
        f"segments={stats.committed_segments} partials={stats.partial_events} "
        f"decode_calls={stats.decode_calls} "
        f"backend_seconds={stats.backend_seconds:.3f} "
        f"audio_seconds={stats.accepted_seconds:.3f}"
    )
    print(f"events={events_path}")
    print(f"transcript={transcript_path}")
    print(f"summary={output_dir / 'summary.json'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    main()
