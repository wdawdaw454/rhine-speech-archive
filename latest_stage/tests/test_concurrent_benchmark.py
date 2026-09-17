from __future__ import annotations

import json
import inspect
from pathlib import Path

import pytest

from scripts.benchmark_concurrent_streams import (
    StreamSpec,
    build_worker_command,
    load_stream_specs,
    control_plane_event,
    summarize_backend_timing,
    run_service_requests,
)


def test_load_stream_specs_filters_and_round_robins(tmp_path: Path):
    mixture = tmp_path / "alternate_snr20" / "seed_52" / "mixture.wav"
    references = tmp_path / "alternate_snr20" / "seed_52" / "references.jsonl"
    mixture.parent.mkdir(parents=True)
    mixture.write_bytes(b"RIFF")
    references.write_text(
        json.dumps({"id": "utt", "speaker": "S001", "start": 0.0, "end": 1.0, "text": "测试"})
        + "\n",
        encoding="utf-8",
    )
    profile_root = tmp_path / "profiles"
    profile_root.mkdir()
    profile = profile_root / "S001.json"
    profile.write_text("{}", encoding="utf-8")
    pair_index = tmp_path / "dataset_index.json"
    pair_index.write_text(
        json.dumps(
            {
                "target_speaker": "S001",
                "interferer_speaker": "S002",
                "items": [
                    {
                        "scenario": "alternate_snr20",
                        "seed": 52,
                        "mixture": "alternate_snr20/seed_52/mixture.wav",
                        "references": "alternate_snr20/seed_52/references.jsonl",
                    },
                    {
                        "scenario": "overlap02_snr8",
                        "seed": 52,
                        "mixture": "missing.wav",
                        "references": "missing.jsonl",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    specs = load_stream_specs(
        [pair_index],
        profile_root=profile_root,
        scenarios=["alternate_snr20"],
        concurrency=3,
    )

    assert len(specs) == 3
    assert specs[0].name == "S001_S002-alternate_snr20-seed52"
    assert specs[0].mixture == mixture
    assert specs[0].references == references
    assert specs[0].profile == profile
    assert specs[1] == specs[0]


def test_worker_command_uses_project_paths_and_disables_punctuation(tmp_path: Path):
    config, output_dir = tmp_path / "config", tmp_path / "output"
    spec = StreamSpec(
        name="stream",
        mixture=tmp_path / "mixture.wav",
        references=tmp_path / "references.jsonl",
        profile=tmp_path / "profile.json",
        target_speaker="S001",
    )

    command = build_worker_command(
        python="/custom/python",
        spec=spec,
        config=config,
        device="cuda:1",
        accept_threshold=0.5,
        reject_threshold=-1.0,
        realtime=True,
        punctuation=False,
        output_dir=output_dir,
        start_delay=0.2,
        chunk_phase=0.0125,
    )

    assert command[:2] == [
        "/custom/python",
        str(Path(__file__).parents[1] / "scripts/run_file_stream.py"),
    ]
    assert "--realtime" in command
    assert "--no-punctuation" in command
    assert command[command.index("--device") + 1] == "cuda:1"
    assert command[command.index("--accept-threshold") + 1] == "0.5"
    assert command[command.index("--reject-threshold") + 1] == "-1"
    assert command[command.index("--start-delay") + 1] == "0.2"
    assert command[command.index("--chunk-phase") + 1] == "0.0125"


def test_run_service_requests_parameters_are_keyword_only():
    parameters = inspect.signature(run_service_requests).parameters

    assert list(parameters)[2:] == [
        "arrival_interval",
        "timeout",
        "poll_interval",
        "trace_path",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in list(parameters.values())[2:]
    )


def test_control_plane_event_reports_admission_state():
    event = control_plane_event(
        {
            "active_count": 16,
            "queued_count": 7,
            "peak_active_streams": 16,
            "succeeded_count": 3,
            "failed_count": 1,
            "cancelled_count": 2,
            "queued_stream_ids": ["19", "20"],
            "requests": ["omitted"],
        },
        event="poll",
        elapsed_seconds=1.25,
    )

    assert event == {
        "event": "poll",
        "elapsed_seconds": 1.25,
        "active_count": 16,
        "queued_count": 7,
        "peak_active_streams": 16,
        "succeeded_count": 3,
        "failed_count": 1,
        "cancelled_count": 2,
        "queued_stream_ids": ["19", "20"],
    }


def test_summarize_backend_timing_pools_worker_durations():
    summary = summarize_backend_timing(
        [
            {"backend_durations": {"asr": [0.1, 0.2]}},
            {"backend_durations": {"asr": [0.3], "vad": [0.05]}},
        ]
    )

    assert summary["asr"]["count"] == 3
    assert summary["asr"]["mean"] == pytest.approx(0.2)
    assert summary["vad"]["count"] == 1
