"""Measure ASR-only fast-mode partial latency and quality.

This benchmark deliberately bypasses VAD, speaker embedding, and target gating.
It answers the product question: how quickly can streaming ASR show the first
partial and continue refreshing text on a local-style microphone stream?
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable, Protocol

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.io import load_audio
from src.data.manifest import read_jsonl
from src.data.normalize import normalize_chinese_text
from src.evaluation.cer import char_edit_distance, cer
from src.evaluation.latency import latency_summary


@dataclass(frozen=True)
class FastExample:
    id: str
    audio_path: str
    waveform: np.ndarray
    sample_rate: int
    duration: float
    reference: str
    speaker: str


@dataclass
class FastEvent:
    audio_time: float
    monotonic_time: float
    new_text: str
    cumulative_text: str
    backend_seconds: float
    is_final: bool


@dataclass
class FastUtteranceMetrics:
    example_id: str
    speaker: str
    duration_seconds: float
    reference_text: str
    final_text: str
    normalized_reference: str
    normalized_final: str
    char_errors: int
    reference_chars: int
    cer: float
    partial_count: int
    first_partial_audio_time: float | None
    first_partial_wall_latency: float | None
    first_partial_backend_seconds: float | None
    partial_audio_intervals: list[float]
    partial_wall_intervals: list[float]
    stabilized_wall_latency: float | None
    final_commit_wall_latency: float
    backend_seconds: float
    compute_rtf: float
    events: list[FastEvent]


class StreamingBackendForBenchmark(Protocol):
    def reset(self) -> None: ...

    def accept(self, chunk: np.ndarray, *, is_final: bool) -> str: ...


BackendFactory = Callable[[float, float], StreamingBackendForBenchmark]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark ASR-only fast-mode streaming behavior")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/processed/aishell_full/dev/sources.jsonl",
    )
    parser.add_argument("--model-dir", type=Path,
        default=PROJECT_ROOT.parent / "models/sensevoice-small")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--backend", choices=["uniasr", "sensevoice"], default="sensevoice"
    )
    parser.add_argument("--input-chunk", type=float, default=0.20)
    parser.add_argument("--asr-chunks", type=float, nargs="+", default=[0.20, 0.30, 0.40, 0.60])
    parser.add_argument("--lookaheads", type=float, nargs="+", default=[0.20])
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="sleep to the audio clock; wall latency then models user-visible latency",
    )
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-events", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs/phase5/fast_asr_baseline"
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.input_chunk <= 0:
        raise ValueError("input chunk must be positive")
    if not args.asr_chunks or any(value <= 0 for value in args.asr_chunks):
        raise ValueError("ASR chunks must be positive")
    if len(set(args.asr_chunks)) != len(args.asr_chunks):
        raise ValueError("ASR chunks must be unique")
    if not args.lookaheads or any(value <= 0 for value in args.lookaheads):
        raise ValueError("lookaheads must be positive")
    if len(set(args.lookaheads)) != len(args.lookaheads):
        raise ValueError("lookaheads must be unique")
    if args.max_examples <= 0:
        raise ValueError("max examples must be positive")


def _resolve_audio_path(audio_path: str, manifest_path: Path) -> Path:
    path = Path(audio_path)
    if path.is_absolute() or path.exists():
        return path
    return (manifest_path.parent / path).resolve()


def load_examples(
    manifest_path: Path, max_examples: int, *, require_text: bool = True
) -> list[FastExample]:
    examples: list[FastExample] = []
    for entry in read_jsonl(manifest_path):
        if len(examples) >= max_examples:
            break
        text = str(entry.extras.get("text", "")).strip()
        if require_text and not normalize_chinese_text(text):
            raise ValueError(f"manifest entry has no usable reference text: {entry.id}")
        audio_path = _resolve_audio_path(entry.audio_path, manifest_path.resolve())
        waveform, sample_rate = load_audio(audio_path, target_sr=entry.sample_rate)
        duration = len(waveform) / sample_rate
        if duration <= 0:
            raise ValueError(f"empty waveform: {entry.id}")
        examples.append(
            FastExample(
                id=entry.id,
                audio_path=str(audio_path),
                waveform=waveform,
                sample_rate=sample_rate,
                duration=duration,
                reference=text,
                speaker=str(entry.extras.get("speaker", "unknown")),
            )
        )
    if not examples:
        raise ValueError(f"no benchmark examples found in {manifest_path}")
    return examples


def _iter_input_chunks(
    waveform: np.ndarray, sample_rate: int, chunk_seconds: float
) -> Iterable[tuple[bool, float, np.ndarray]]:
    samples_per_chunk = max(1, int(round(chunk_seconds * sample_rate)))
    total = len(waveform)
    for start in range(0, total, samples_per_chunk):
        chunk = waveform[start : start + samples_per_chunk]
        is_last = start + samples_per_chunk >= total
        audio_time = (start + len(chunk)) / sample_rate
        yield is_last, audio_time, chunk.astype(np.float32, copy=False)


def _visible_delta(previous: str, cumulative: str) -> str:
    if cumulative.startswith(previous):
        return cumulative[len(previous) :]
    return "" if cumulative == previous else cumulative


def run_fast_stream(
    example: FastExample,
    backend: StreamingBackendForBenchmark,
    *,
    asr_chunk: float,
    input_chunk: float,
    realtime_sleep: bool,
) -> FastUtteranceMetrics:
    if input_chunk <= 0 or asr_chunk <= 0:
        raise ValueError("input and ASR chunk sizes must be positive")

    backend.reset()
    stream_started = time.perf_counter()
    asr_samples = max(1, int(round(asr_chunk * example.sample_rate)))
    buffer = np.empty(0, dtype=np.float32)
    cumulative_text = ""
    events: list[FastEvent] = []
    backend_seconds = 0.0
    final_text = ""
    final_wall_time = stream_started

    def call_backend(chunk: np.ndarray, audio_time: float, is_final: bool) -> None:
        nonlocal cumulative_text, backend_seconds, final_text, final_wall_time
        call_started = time.perf_counter()
        result = backend.accept(np.asarray(chunk, dtype=np.float32), is_final=is_final)
        call_elapsed = time.perf_counter() - call_started
        backend_seconds += call_elapsed
        previous = cumulative_text
        cumulative_text = str(result).strip()
        final_text = cumulative_text
        final_wall_time = time.perf_counter()
        new_text = _visible_delta(previous, cumulative_text)
        if new_text.strip():
            events.append(
                FastEvent(
                    audio_time=audio_time,
                    monotonic_time=final_wall_time,
                    new_text=new_text,
                    cumulative_text=cumulative_text,
                    backend_seconds=call_elapsed,
                    is_final=is_final,
                )
            )

    for is_last_input, input_audio_time, input_wave in _iter_input_chunks(
        example.waveform, example.sample_rate, input_chunk
    ):
        if realtime_sleep:
            target_time = stream_started + input_audio_time
            remaining = target_time - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

        buffer = np.concatenate((buffer, input_wave))
        while len(buffer) >= asr_samples:
            asr_wave = buffer[:asr_samples]
            buffer = buffer[asr_samples:]
            call_backend(asr_wave, input_audio_time, is_final=is_last_input and len(buffer) == 0)
        if is_last_input and len(buffer):
            call_backend(buffer, input_audio_time, is_final=True)
            buffer = np.empty(0, dtype=np.float32)

    final_wall_latency = final_wall_time - stream_started
    normalized_reference = normalize_chinese_text(example.reference)
    normalized_final = normalize_chinese_text(final_text)
    errors = char_edit_distance(normalized_reference, normalized_final)
    first_event = events[0] if events else None
    stable_index = next(
        (
            index
            for index, event in enumerate(events)
            if normalize_chinese_text(event.cumulative_text) == normalized_final
        ),
        None,
    )
    audio_times = [event.audio_time for event in events]
    wall_times = [event.monotonic_time for event in events]
    relative_wall_times = [value - stream_started for value in wall_times]

    return FastUtteranceMetrics(
        example_id=example.id,
        speaker=example.speaker,
        duration_seconds=example.duration,
        reference_text=example.reference,
        final_text=final_text,
        normalized_reference=normalized_reference,
        normalized_final=normalized_final,
        char_errors=errors,
        reference_chars=len(normalized_reference),
        cer=cer(normalized_reference, normalized_final),
        partial_count=len(events),
        first_partial_audio_time=first_event.audio_time if first_event else None,
        first_partial_wall_latency=relative_wall_times[0] if relative_wall_times else None,
        first_partial_backend_seconds=first_event.backend_seconds if first_event else None,
        partial_audio_intervals=[
            round(second - first, 6) for first, second in zip(audio_times, audio_times[1:])
        ],
        partial_wall_intervals=[
            round(second - first, 6) for first, second in zip(wall_times, wall_times[1:])
        ],
        stabilized_wall_latency=(
            relative_wall_times[stable_index]
            if stable_index is not None and relative_wall_times
            else None
        ),
        final_commit_wall_latency=final_wall_latency,
        backend_seconds=backend_seconds,
        compute_rtf=backend_seconds / example.duration,
        events=events,
    )


def _optional_summary(values: list[float | None]) -> dict[str, float]:
    return latency_summary([float(value) for value in values if value is not None])


def aggregate_run(
    metrics: list[FastUtteranceMetrics],
    *,
    asr_chunk: float,
    lookahead: float,
    input_chunk: float,
    realtime: bool,
) -> dict[str, Any]:
    if not metrics:
        raise ValueError("cannot aggregate an empty run")
    char_errors = sum(item.char_errors for item in metrics)
    reference_chars = sum(item.reference_chars for item in metrics)
    audio_intervals = [
        value for item in metrics for value in item.partial_audio_intervals if value > 0
    ]
    wall_intervals = [
        value for item in metrics for value in item.partial_wall_intervals if value > 0
    ]
    return {
        "asr_chunk_seconds": asr_chunk,
        "lookahead_seconds": lookahead,
        "input_chunk_seconds": input_chunk,
        "timing_mode": "realtime" if realtime else "compute",
        "examples": len(metrics),
        "duration_seconds": round(sum(item.duration_seconds for item in metrics), 6),
        "corpus_cer": round(char_errors / reference_chars, 6) if reference_chars else 0.0,
        "mean_cer": round(float(np.mean([item.cer for item in metrics])), 6),
        "char_errors": char_errors,
        "reference_chars": reference_chars,
        "first_partial_audio_time": _optional_summary(
            [item.first_partial_audio_time for item in metrics]
        ),
        "first_partial_wall_latency": _optional_summary(
            [item.first_partial_wall_latency for item in metrics]
        ),
        "first_partial_backend_seconds": _optional_summary(
            [item.first_partial_backend_seconds for item in metrics]
        ),
        "partial_update_interval_audio": latency_summary(audio_intervals),
        "partial_update_interval_wall": latency_summary(wall_intervals),
        "stabilized_wall_latency": _optional_summary(
            [item.stabilized_wall_latency for item in metrics]
        ),
        "final_commit_wall_latency": latency_summary(
            [item.final_commit_wall_latency for item in metrics]
        ),
        "compute_rtf": latency_summary([item.compute_rtf for item in metrics]),
        "partial_count": latency_summary([float(item.partial_count) for item in metrics]),
        "empty_output_examples": sum(not item.final_text.strip() for item in metrics),
        "empty_partial_examples": sum(item.partial_count == 0 for item in metrics),
    }


def metrics_to_dict(item: FastUtteranceMetrics, *, include_events: bool = False) -> dict[str, Any]:
    value = asdict(item)
    value.pop("events", None)
    if include_events:
        value["events"] = [asdict(event) for event in item.events]
    return value


class NativeBackendFactory:
    """Reuse a backend that applies its own native streaming configuration."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def __call__(self, asr_chunk: float, lookahead: float) -> StreamingBackendForBenchmark:
        del asr_chunk, lookahead
        self.backend.reset()
        return self.backend


class PrefixBackendFactory:
    """Reuse an offline model while changing its prefix re-decode cadence."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def __call__(self, asr_chunk: float, lookahead: float) -> StreamingBackendForBenchmark:
        del lookahead
        self.backend.decode_chunk = float(asr_chunk)
        self.backend.reset()
        return self.backend


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_benchmark(
    args: argparse.Namespace,
    examples: list[FastExample],
    backend_factory: BackendFactory,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for asr_chunk in args.asr_chunks:
        for lookahead in args.lookaheads:
            if args.warmup:
                run_fast_stream(
                    examples[0],
                    backend_factory(asr_chunk, lookahead),
                    asr_chunk=asr_chunk,
                    input_chunk=args.input_chunk,
                    realtime_sleep=False,
                )
            backend = backend_factory(asr_chunk, lookahead)
            metrics = [
                run_fast_stream(
                    example,
                    backend,
                    asr_chunk=asr_chunk,
                    input_chunk=args.input_chunk,
                    realtime_sleep=args.realtime,
                )
                for example in examples
            ]
            run_summary = aggregate_run(
                metrics,
                asr_chunk=asr_chunk,
                lookahead=lookahead,
                input_chunk=args.input_chunk,
                realtime=args.realtime,
            )
            runs.append(run_summary)
            for item in metrics:
                config = {
                    "asr_chunk_seconds": asr_chunk,
                    "lookahead_seconds": lookahead,
                    "input_chunk_seconds": args.input_chunk,
                    "timing_mode": "realtime" if args.realtime else "compute",
                }
                example_rows.append({"config": config, **metrics_to_dict(item)})
                if args.write_events:
                    event_rows.extend(
                        {"config": config, "example_id": item.example_id, **asdict(event)}
                        for event in item.events
                    )
            print(
                f"chunk={asr_chunk:.2f} lookahead={lookahead:.2f} "
                f"cer={run_summary['corpus_cer']:.4f} "
                f"first_audio={run_summary['first_partial_audio_time']['p50']:.3f}s "
                f"compute_rtf={run_summary['compute_rtf']['p50']:.4f}",
                flush=True,
            )

    summary = {
        "schema_version": 1,
        "phase": "5.7",
        "probe": "fast-asr-only-baseline",
        "manifest": str(args.manifest.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "device": args.device,
        "backend": args.backend,
        "realtime": args.realtime,
        "warmup": args.warmup,
        "max_examples": args.max_examples,
        "runs": runs,
    }
    return summary, example_rows, event_rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        examples = load_examples(args.manifest, args.max_examples)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    from src.models.backends import (
        FunasrPrefixAsr,
        FunasrUniAsr,
    )

    load_started = time.perf_counter()
    if args.backend == "sensevoice":
        backend = FunasrPrefixAsr(
            args.model_dir, decode_chunk=args.asr_chunks[0], device=args.device
        )
        backend_factory: BackendFactory = PrefixBackendFactory(backend)
    elif args.backend == "uniasr":
        backend = FunasrUniAsr(args.model_dir, device=args.device)
        backend_factory: BackendFactory = NativeBackendFactory(backend)
    model_load_seconds = time.perf_counter() - load_started
    summary, example_rows, event_rows = run_benchmark(args, examples, backend_factory)
    summary["model_load_seconds"] = round(model_load_seconds, 6)
    summary["examples"] = [
        {
            "id": item.id,
            "speaker": item.speaker,
            "duration_seconds": round(item.duration, 6),
        }
        for item in examples
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "summary.json", summary)
    _write_jsonl(args.output_dir / "runs.jsonl", summary["runs"])
    _write_jsonl(args.output_dir / "examples.jsonl", example_rows)
    if args.write_events:
        _write_jsonl(args.output_dir / "events.jsonl", event_rows)
    print(f"wrote benchmark output to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    main()
