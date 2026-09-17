"""Benchmark concurrent file-simulated realtime streams."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.latency import latency_summary
from src.evaluation.report import build_evaluation_report, read_references
from src.pipeline.event_io import read_events
from src.scheduling.admission import PhaseLease
from src.scheduling.service import StreamWorkerService, WorkerHandle


@dataclass(frozen=True)
class StreamSpec:
    name: str
    mixture: Path
    references: Path
    profile: Path
    target_speaker: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", required=True, type=Path)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--profile-root", type=Path, default=Path("profiles/aishell_full"))
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--max-active-streams", type=int)
    parser.add_argument("--admission-phase", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_latency.yaml"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--accept-threshold", type=float, default=0.50)
    parser.add_argument("--reject-threshold", type=float, default=-1.00)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--start-stagger", type=float, default=0.0)
    parser.add_argument("--arrival-interval", type=float, default=0.0)
    parser.add_argument("--phase-stagger", type=float, default=0.0)
    parser.add_argument("--phase-period", type=float, default=0.20)
    parser.add_argument("--punctuation", action="store_true")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--poll-interval", type=float, default=0.50)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--estimated-service-seconds", type=float, default=60.0)
    parser.add_argument("--python", default=sys.executable)
    return parser


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_stream_specs(
    pair_indexes: list[Path],
    *,
    profile_root: Path,
    scenarios: list[str],
    concurrency: int,
) -> list[StreamSpec]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    candidates: list[StreamSpec] = []
    for pair_index_path in pair_indexes:
        pair_index = pair_index_path.resolve()
        dataset = _read_json(pair_index)
        items = dataset.get("items", [])
        if not items:
            raise ValueError(f"dataset index has no items: {pair_index}")
        if scenarios:
            items = [item for item in items if item.get("scenario") in scenarios]
        if not items:
            raise ValueError(f"no requested scenarios found in: {pair_index}")

        target_speaker = str(dataset["target_speaker"])
        profile = (profile_root / f"{target_speaker}.json").resolve()
        if not profile.is_file():
            raise FileNotFoundError(f"target profile is missing: {profile}")

        for item in items:
            mixture = pair_index.parent / str(item["mixture"])
            references = pair_index.parent / str(item["references"])
            if not mixture.is_file():
                raise FileNotFoundError(f"mixture is missing: {mixture}")
            if not references.is_file():
                raise FileNotFoundError(f"references are missing: {references}")
            interferer = dataset.get("interferer_speaker", "unknown")
            name = f"{target_speaker}_{interferer}-{item['scenario']}-seed{item['seed']}"
            candidates.append(
                StreamSpec(
                    name=name,
                    mixture=mixture,
                    references=references,
                    profile=profile,
                    target_speaker=target_speaker,
                )
            )

    if not candidates:
        raise ValueError("no stream specs selected")
    return [candidates[index % len(candidates)] for index in range(concurrency)]


def build_worker_command(
    *,
    python: str,
    spec: StreamSpec,
    config: Path,
    device: str,
    accept_threshold: float,
    reject_threshold: float,
    realtime: bool,
    punctuation: bool,
    output_dir: Path,
    start_delay: float = 0.0,
    chunk_phase: float | None = None,
) -> list[str]:
    command = [
        python,
        str(PROJECT_ROOT / "scripts/run_file_stream.py"),
        "--input",
        str(spec.mixture),
        "--profile",
        str(spec.profile),
        "--config",
        str(config.resolve()),
        "--device",
        device,
        "--accept-threshold",
        f"{accept_threshold:.6g}",
        "--reject-threshold",
        f"{reject_threshold:.6g}",
        "--output-dir",
        str(output_dir.resolve()),
    ]
    if realtime:
        command.append("--realtime")
    if start_delay > 0:
        command.extend(("--start-delay", f"{start_delay:.9g}"))
    if chunk_phase is not None:
        command.extend(("--chunk-phase", f"{chunk_phase:.9g}"))
    if not punctuation:
        command.append("--no-punctuation")
    return command


def _device_index(device: str) -> int:
    value = device.lower()
    if value.startswith("cuda:"):
        try:
            return int(value.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"cannot sample non-numeric CUDA device: {device}") from exc
    raise ValueError(f"GPU sampling requires an explicit cuda:N device, got: {device}")


class GpuSampler:
    def __init__(self, device: str, interval: float) -> None:
        self.index = _device_index(device)
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples: list[dict[str, float]] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if self.interval <= 0:
            return
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        query = [
            "nvidia-smi",
            f"--id={self.index}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self.stop_event.wait(self.interval):
            try:
                output = subprocess.check_output(query, text=True, stderr=subprocess.DEVNULL)
                memory_mb, utilization = output.strip().split(",", 1)
                self.samples.append(
                    {
                        "memory_mb": float(memory_mb),
                        "utilization_percent": float(utilization),
                    }
                )
            except (OSError, ValueError, subprocess.SubprocessError):
                continue

    @property
    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"count": 0}
        memory = [sample["memory_mb"] for sample in self.samples]
        utilization = [sample["utilization_percent"] for sample in self.samples]
        return {
            "count": len(self.samples),
            "memory_mb": {
                "mean": float(np.mean(memory)),
                "max": float(np.max(memory)),
            },
            "utilization_percent": {
                "mean": float(np.mean(utilization)),
                "max": float(np.max(utilization)),
            },
        }


def _latencies(events: list[Any], event_type_name: str) -> list[float]:
    values: list[float] = []
    for event in events:
        if event.type.value != event_type_name:
            continue
        if event.monotonic_time is None or event.audio_time is None:
            continue
        values.append(float(event.monotonic_time - event.audio_time))
    return values


def _first_latency_by_segment(events: list[Any]) -> list[float]:
    earliest: dict[str, float] = {}
    for event in events:
        if event.type.value != "asr_partial" or not event.segment_id:
            continue
        if event.monotonic_time is None or event.audio_time is None:
            continue
        value = float(event.monotonic_time - event.audio_time)
        if event.segment_id not in earliest or value < earliest[event.segment_id]:
            earliest[event.segment_id] = value
    return list(earliest.values())


def summarize_backend_timing(workers: list[dict[str, Any]]) -> dict[str, Any]:
    durations: dict[str, list[float]] = {}
    for worker in workers:
        for stage, values in worker.get("backend_durations", {}).items():
            durations.setdefault(stage, []).extend(float(value) for value in values)
    return {stage: latency_summary(values) for stage, values in durations.items()}


def summarize_workers(workers: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [worker for worker in workers if worker["returncode"] == 0]
    if not successful:
        return {"successful_workers": 0, "failed_workers": len(workers)}
    return {
        "successful_workers": len(successful),
        "failed_workers": len(workers) - len(successful),
        "launch_to_exit_seconds": latency_summary(
            [worker["launch_to_exit_seconds"] for worker in successful]
        ),
        "pipeline_rtf": latency_summary([worker["evaluation"]["rtf"] for worker in successful]),
        "caption_commit_latency": latency_summary(
            [
                value
                for worker in successful
                for value in _latencies(worker["events"], "caption_commit")
            ]
        ),
        "first_partial_latency": latency_summary(
            [
                value
                for worker in successful
                for value in _first_latency_by_segment(worker["events"])
            ]
        ),
        "target_cer": latency_summary(
            [worker["evaluation"]["target_metrics"]["target_cer"] for worker in successful]
        ),
    }


def control_plane_event(
    snapshot: dict[str, Any], *, event: str, elapsed_seconds: float
) -> dict[str, Any]:
    fields = (
        "active_count",
        "queued_count",
        "peak_active_streams",
        "succeeded_count",
        "failed_count",
        "cancelled_count",
    )
    return {
        "event": event,
        "elapsed_seconds": round(elapsed_seconds, 6),
        **{field: snapshot.get(field, 0) for field in fields},
        "queued_stream_ids": snapshot.get("queued_stream_ids", []),
    }


def run_service_requests(
    service: StreamWorkerService,
    specs: list[StreamSpec],
    *,
    arrival_interval: float,
    timeout: float,
    poll_interval: float,
    trace_path: Path,
) -> bool:
    started_at = time.monotonic()
    with trace_path.open("w", encoding="utf-8", newline="\n") as trace:

        def write(event: str) -> None:
            record = control_plane_event(
                service.snapshot(),
                event=event,
                elapsed_seconds=time.monotonic() - started_at,
            )
            trace.write(json.dumps(record, ensure_ascii=False) + "\n")

        for index, spec in enumerate(specs):
            service.submit(str(index), spec)
            write("request_submitted")
            next_arrival_at = started_at + (index + 1) * arrival_interval
            while time.monotonic() < next_arrival_at:
                if time.monotonic() - started_at >= timeout:
                    service.shutdown(timeout=10.0)
                    write("timeout")
                    return False
                service.poll_once()
                write("poll")
                remaining = next_arrival_at - time.monotonic()
                if remaining > 0:
                    time.sleep(min(poll_interval, remaining))

        remaining_timeout = timeout - (time.monotonic() - started_at)
        if remaining_timeout <= 0:
            service.shutdown(timeout=10.0)
            write("timeout")
            return False
        completed = service.run_until_idle(
            timeout=remaining_timeout,
            poll_interval=poll_interval,
            termination_grace_seconds=10.0,
        )
        write("timeout" if not completed else "completed")
        return completed


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("timeout must be positive")
    if args.poll_interval < 0:
        raise SystemExit("poll-interval must be non-negative")
    if args.start_stagger < 0:
        raise SystemExit("start-stagger must be non-negative")
    if args.arrival_interval < 0:
        raise SystemExit("arrival-interval must be non-negative")
    if args.phase_stagger < 0:
        raise SystemExit("phase-stagger must be non-negative")
    if args.phase_period <= 0:
        raise SystemExit("phase-period must be positive")
    if args.phase_stagger >= args.phase_period:
        raise SystemExit("phase-stagger must be smaller than phase-period")
    if args.estimated_service_seconds <= 0:
        raise SystemExit("estimated-service-seconds must be positive")
    max_active_streams = args.max_active_streams or args.concurrency
    if max_active_streams < 1:
        raise SystemExit("max-active-streams must be positive")
    if max_active_streams > args.concurrency:
        raise SystemExit("max-active-streams cannot exceed concurrency")

    specs = load_stream_specs(
        args.pair,
        profile_root=args.profile_root,
        scenarios=args.scenario,
        concurrency=args.concurrency,
    )
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"output directory already exists: {output_root}")
    output_root.mkdir(parents=True)
    metadata = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "concurrency": args.concurrency,
        "max_active_streams": max_active_streams,
        "admission_phase_enabled": args.admission_phase,
        "realtime": args.realtime,
        "punctuation": args.punctuation,
        "start_stagger_seconds": args.start_stagger,
        "arrival_interval_seconds": args.arrival_interval,
        "phase_stagger_seconds": args.phase_stagger,
        "phase_period_seconds": args.phase_period,
        "device": args.device,
        "estimated_service_seconds": args.estimated_service_seconds,
        "thresholds": {
            "accept": args.accept_threshold,
            "reject": args.reject_threshold,
        },
        "config": str(args.config.resolve()),
        "pairs": [str(path.resolve()) for path in args.pair],
        "scenarios": args.scenario,
    }
    _write_json(output_root / "benchmark.json", metadata)

    sampler = GpuSampler(args.device, args.poll_interval)

    def launch(stream_id: str, payload: StreamSpec, lease: PhaseLease) -> WorkerHandle:
        index = int(stream_id)
        spec = payload
        worker_root = output_root / f"worker_{index:03d}"
        worker_root.mkdir()
        log_path = worker_root / "process.log"

        log = log_path.open("w", encoding="utf-8", newline="\n")
        command = build_worker_command(
            python=args.python,
            spec=spec,
            config=args.config,
            device=args.device,
            accept_threshold=args.accept_threshold,
            reject_threshold=args.reject_threshold,
            realtime=args.realtime,
            punctuation=args.punctuation,
            output_dir=worker_root,
            start_delay=index * args.start_stagger,
            chunk_phase=(
                lease.phase_offset
                if args.admission_phase
                else (
                    (index * args.phase_stagger) % args.phase_period if args.phase_stagger else None
                )
            ),
        )
        _write_json(
            worker_root / "command.json",
            {
                "command": command,
                "phase_slot_id": lease.slot_id,
                "phase_offset": lease.phase_offset,
            },
        )

        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.close()
        return WorkerHandle(
            process=process,
            command=command,
            details={"worker_root": worker_root, "process_log": log_path},
        )

    service = StreamWorkerService(
        max_active_streams=max_active_streams,
        phase_period=args.phase_period,
        launcher=launch,
        estimated_service_seconds=args.estimated_service_seconds,
    )
    sampler.start()
    completed = run_service_requests(
        service,
        specs,
        arrival_interval=args.arrival_interval,
        timeout=args.timeout,
        poll_interval=min(0.2, max(0.01, args.poll_interval)),
        trace_path=output_root / "service_events.jsonl",
    )
    failed = not completed
    sampler.stop()
    workers: list[dict[str, Any]] = []
    for record in service.records:
        index = int(record.stream_id)
        spec = record.payload
        worker_root = (
            record.handle.details.get("worker_root") if record.handle is not None else None
        )
        log_path = record.handle.details.get("process_log") if record.handle is not None else None
        returncode = record.returncode
        if returncode is None and record.handle is not None:
            returncode = record.handle.process.wait()

        worker: dict[str, Any] = {
            "index": index,
            "state": record.state,
            "stream": spec.name,
            "target_speaker": spec.target_speaker,
            "mixture": str(spec.mixture),
            "profile": str(spec.profile),
            "returncode": returncode,
            "launch_to_exit_seconds": (
                round(record.ended_at - record.launched_at, 6)
                if record.launched_at is not None and record.ended_at is not None
                else None
            ),
            "admission_queue_seconds": (
                round(record.launched_at - record.submitted_at, 6)
                if record.launched_at is not None
                else None
            ),
            "phase_slot_id": record.lease.slot_id if record.lease else None,
            "phase_offset": record.lease.phase_offset if record.lease else None,
            "worker_root": str(worker_root) if worker_root is not None else None,
            "process_log": str(log_path) if log_path is not None else None,
        }
        if worker_root is None:
            worker["evaluation_error"] = record.error or "worker was not launched"
        else:
            run_dirs = [path for path in worker_root.iterdir() if path.is_dir()]
            if returncode == 0 and len(run_dirs) == 1:
                events_path = run_dirs[0] / "events.jsonl"
                try:
                    pipeline_summary = _read_json(run_dirs[0] / "summary.json")
                    worker["pipeline_summary"] = pipeline_summary
                    worker["backend_durations"] = pipeline_summary.get("backend_durations", {})
                    events = read_events(events_path)
                    evaluation = build_evaluation_report(
                        events,
                        read_references(spec.references),
                        target_speaker=spec.target_speaker,
                    )
                    worker["events"] = events
                    worker["evaluation"] = evaluation
                    worker["run_dir"] = str(run_dirs[0])
                except (OSError, ValueError, KeyError) as exc:
                    worker["evaluation_error"] = str(exc)
        workers.append(worker)

    visible_workers = []
    for worker in workers:
        copied = dict(worker)
        copied.pop("events", None)
        visible_workers.append(copied)

    result = {
        **metadata,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "timeout": args.timeout,
        "timed_out": failed,
        "gpu": sampler.summary,
        "summary": summarize_workers(workers),
        "backend_timing": summarize_backend_timing(workers),
        "scheduler": {
            **service.snapshot(),
            "pending_stream_ids": service.queued_stream_ids,
            "queue_summary": latency_summary(
                [
                    worker["admission_queue_seconds"]
                    for worker in workers
                    if worker["admission_queue_seconds"] is not None
                ]
            ),
        },
        "workers": visible_workers,
    }
    _write_json(output_root / "summary.json", result)
    for worker in workers:
        if "evaluation" in worker:
            evaluation_path = Path(worker["worker_root"]) / "evaluation.json"
            _write_json(evaluation_path, worker["evaluation"])

    summary = result["summary"]
    print(f"benchmark={output_root / 'summary.json'}")
    print(
        f"workers={summary.get('successful_workers', 0)}/{args.concurrency} "
        f"failed={summary.get('failed_workers', args.concurrency)} "
        f"timed_out={failed}"
    )
    if summary.get("successful_workers"):
        commit = summary["caption_commit_latency"]
        rtf = summary["pipeline_rtf"]
        queue = result["scheduler"]["queue_summary"]
        print(
            f"commit_p50_p95={commit['p50']:.3f}s/{commit['p95']:.3f}s "
            f"rtf_mean_max={rtf['mean']:.3f}/{rtf['max']:.3f}"
            f" queue_p50_p95={queue['p50']:.3f}s/{queue['p95']:.3f}s"
        )
    raise SystemExit(1 if failed or summary.get("failed_workers") else 0)


if __name__ == "__main__":
    main()
