from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark_phase5_fast_asr import (
    FastExample,
    aggregate_run,
    build_parser,
    run_benchmark,
    run_fast_stream,
    validate_args,
)


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []
        self.outputs: list[str] = []
        self.text = ""

    def reset(self) -> None:
        self.calls = []
        self.outputs = []
        self.text = ""

    def accept(self, chunk: np.ndarray, *, is_final: bool) -> str:
        self.calls.append((len(chunk), is_final))
        self.outputs.append("你" if len(self.calls) == 1 else "好")
        self.text += self.outputs[-1]
        return self.text


def _example(duration: float = 0.8, reference: str = "你好") -> FastExample:
    samples = int(round(duration * 16000))
    return FastExample(
        id="example-1",
        audio_path="/tmp/example-1.wav",
        waveform=np.zeros(samples, dtype=np.float32),
        sample_rate=16000,
        duration=samples / 16000,
        reference=reference,
        speaker="S0001",
    )


def test_fast_stream_aggregates_input_and_marks_exact_final():
    backend = FakeBackend()
    metrics = run_fast_stream(
        _example(),
        backend,
        asr_chunk=0.40,
        input_chunk=0.20,
        realtime_sleep=False,
    )

    assert backend.calls == [(6400, False), (6400, True)]
    assert metrics.final_text == "你好"
    assert metrics.normalized_final == "你好"
    assert metrics.cer == 0.0
    assert metrics.partial_count == 2
    assert metrics.first_partial_audio_time == 0.4
    assert metrics.partial_audio_intervals == [0.4]
    assert metrics.stabilized_wall_latency is not None
    assert metrics.final_commit_wall_latency >= metrics.stabilized_wall_latency


def test_fast_stream_flushes_residual_as_final():
    backend = FakeBackend()
    metrics = run_fast_stream(
        _example(duration=0.7),
        backend,
        asr_chunk=0.40,
        input_chunk=0.20,
        realtime_sleep=False,
    )

    assert backend.calls == [(6400, False), (4800, True)]
    assert metrics.final_text == "你好"
    assert metrics.char_errors == 0


def test_fast_stream_cer_normalizes_punctuation():
    class PunctuationBackend(FakeBackend):
        def accept(self, chunk: np.ndarray, *, is_final: bool) -> str:
            super().accept(chunk, is_final=is_final)
            return "你好。"

    metrics = run_fast_stream(
        _example(reference="你好"),
        PunctuationBackend(),
        asr_chunk=0.80,
        input_chunk=0.20,
        realtime_sleep=False,
    )

    assert metrics.final_text == "你好。"
    assert metrics.cer == 0.0


def test_realtime_mode_preserves_text():
    example = _example(duration=0.06)
    compute_backend = FakeBackend()
    realtime_backend = FakeBackend()

    compute = run_fast_stream(
        example,
        compute_backend,
        asr_chunk=0.02,
        input_chunk=0.02,
        realtime_sleep=False,
    )
    realtime = run_fast_stream(
        example,
        realtime_backend,
        asr_chunk=0.02,
        input_chunk=0.02,
        realtime_sleep=True,
    )

    assert compute.final_text == realtime.final_text == "你好好"
    assert realtime.final_commit_wall_latency >= 0.06


def test_aggregate_run_uses_corpus_cer_and_optional_latencies():
    first = run_fast_stream(
        _example(reference="你好"),
        FakeBackend(),
        asr_chunk=0.4,
        input_chunk=0.2,
        realtime_sleep=False,
    )
    second = run_fast_stream(
        _example(reference="好吗"),
        FakeBackend(),
        asr_chunk=0.4,
        input_chunk=0.2,
        realtime_sleep=False,
    )

    summary = aggregate_run(
        [first, second],
        asr_chunk=0.4,
        lookahead=0.2,
        input_chunk=0.2,
        realtime=False,
    )

    assert summary["examples"] == 2
    assert summary["char_errors"] == 2
    assert summary["reference_chars"] == 4
    assert summary["corpus_cer"] == 0.5
    assert summary["first_partial_audio_time"]["count"] == 2
    assert summary["empty_output_examples"] == 0


def test_run_benchmark_writes_run_example_and_event_rows():
    args = argparse.Namespace(
        backend="sensevoice",
        input_chunk=0.2,
        asr_chunks=[0.4],
        lookaheads=[0.2],
        realtime=False,
        warmup=False,
        write_events=True,
        manifest=Path("/tmp/manifest.jsonl"),
        model_dir=Path("/tmp/model"),
        device="cpu",
        max_examples=1,
    )

    summary, example_rows, event_rows = run_benchmark(
        args, [_example()], lambda chunk, lookahead: FakeBackend()
    )

    assert len(summary["runs"]) == 1
    assert summary["runs"][0]["timing_mode"] == "compute"
    assert len(example_rows) == 1
    assert len(event_rows) == 2


def _args(**overrides):
    values = {
        "backend": "sensevoice",
        "input_chunk": 0.2,
        "asr_chunks": [0.2, 0.4],
        "lookaheads": [0.2],
        "max_examples": 8,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_chunk": 0},
        {"asr_chunks": []},
        {"asr_chunks": [0.2, 0.2]},
        {"lookaheads": [0]},
        {"max_examples": 0},
    ],
)
def test_validate_args_rejects_invalid_sweeps(overrides):
    with pytest.raises(ValueError):
        validate_args(_args(**overrides))


def test_parser_accepts_fast_mode_defaults():
    args = build_parser().parse_args([])

    assert args.backend == "sensevoice"
    assert args.input_chunk == 0.2
    assert args.asr_chunks == [0.2, 0.3, 0.4, 0.6]
    assert args.lookaheads == [0.2]
    assert args.realtime is False
    validate_args(args)
