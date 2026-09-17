"""Run a long-lived HTTP control plane for file-simulated realtime streams."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_concurrent_streams import StreamSpec, build_worker_command
from src.scheduling.control_plane import StreamControlPlane
from src.scheduling.service import StreamWorkerService, WorkerHandle

STREAM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
MAX_REQUEST_BYTES = 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--max-active-streams", required=True, type=int)
    parser.add_argument("--phase-period", type=float, default=0.20)
    parser.add_argument("--admission-phase", action="store_true")
    parser.add_argument("--max-service-seconds", type=float, default=300.0)
    parser.add_argument("--estimated-service-seconds", type=float, default=60.0)
    parser.add_argument("--poll-interval", type=float, default=0.10)
    parser.add_argument("--status-interval", type=float, default=1.0)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_latency.yaml"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--accept-threshold", type=float, default=0.50)
    parser.add_argument("--reject-threshold", type=float, default=-1.00)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--punctuation", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class ControlPlaneJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def write(self, event: str, **values: Any) -> None:
        record = {"event": event, "wall_time": time.time(), **values}
        with self.lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_stream_request(body: dict[str, Any]) -> tuple[str, StreamSpec]:
    stream_id = str(body.get("stream_id", ""))
    if not STREAM_ID_PATTERN.fullmatch(stream_id):
        raise ValueError("stream_id must match [A-Za-z0-9_.-]+")

    required = ("mixture", "references", "profile", "target_speaker")
    missing = [field for field in required if not body.get(field)]
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

    mixture = Path(str(body["mixture"])).resolve()
    references = Path(str(body["references"])).resolve()
    profile = Path(str(body["profile"])).resolve()
    if not mixture.is_file():
        raise FileNotFoundError(f"mixture is missing: {mixture}")
    if not references.is_file():
        raise FileNotFoundError(f"references are missing: {references}")
    if not profile.is_file():
        raise FileNotFoundError(f"profile is missing: {profile}")

    spec = StreamSpec(
        name=str(body.get("name") or stream_id),
        mixture=mixture,
        references=references,
        profile=profile,
        target_speaker=str(body["target_speaker"]),
    )
    return stream_id, spec


def build_handler(
    control: StreamControlPlane,
    *,
    journal: ControlPlaneJournal,
    stop_event: threading.Event,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            journal.write(
                "http",
                method=self.command,
                path=urlparse(self.path).path,
                message=format % args,
            )

        def _send(self, status: HTTPStatus, value: dict[str, Any]) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("request body must be between 1 byte and 1 MiB")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/health"):
                self._send(
                    HTTPStatus.OK,
                    {"state": "running", "max_service_seconds": control.max_service_seconds},
                )
                return
            if path == "/status":
                self._send(HTTPStatus.OK, control.snapshot())
                return
            if path.startswith("/requests/"):
                stream_id = unquote(path.removeprefix("/requests/"))
                try:
                    self._send(HTTPStatus.OK, control.status(stream_id))
                except KeyError:
                    self._send(HTTPStatus.NOT_FOUND, {"error": f"unknown stream: {stream_id}"})
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/shutdown":
                stop_event.set()
                journal.write("shutdown_requested")
                self._send(HTTPStatus.ACCEPTED, {"state": "stopping"})
                return
            try:
                body = self._read_json()
                if path == "/requests":
                    stream_id, spec = parse_stream_request(body)
                    status = control.submit(stream_id, spec)
                    journal.write("request_submitted", stream_id=stream_id, status=status)
                    self._send(HTTPStatus.CREATED, status)
                    return
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except FileNotFoundError as exc:
                self._send(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

        def do_DELETE(self) -> None:
            path = urlparse(self.path).path
            if not path.startswith("/requests/"):
                self._send(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
            stream_id = unquote(path.removeprefix("/requests/"))
            try:
                status = control.cancel(stream_id)
                journal.write("request_cancelled", stream_id=stream_id, status=status)
                self._send(HTTPStatus.OK, status)
            except KeyError:
                self._send(HTTPStatus.NOT_FOUND, {"error": f"unknown stream: {stream_id}"})

    return Handler


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("port must be in [1, 65535]")
    if args.max_active_streams < 1:
        raise SystemExit("max-active-streams must be positive")
    if args.phase_period <= 0:
        raise SystemExit("phase-period must be positive")
    if args.max_service_seconds <= 0:
        raise SystemExit("max-service-seconds must be positive")
    if args.estimated_service_seconds <= 0:
        raise SystemExit("estimated-service-seconds must be positive")
    if args.poll_interval < 0:
        raise SystemExit("poll-interval must be non-negative")
    if args.status_interval < 0:
        raise SystemExit("status-interval must be non-negative")

    state_root = args.state_root.resolve()
    if state_root.exists():
        raise SystemExit(f"state directory already exists: {state_root}")
    state_root.mkdir(parents=True)
    workers_root = state_root / "workers"
    workers_root.mkdir()
    journal = ControlPlaneJournal(state_root / "control_events.jsonl")

    def launch(stream_id: str, payload: StreamSpec, lease) -> WorkerHandle:
        output_dir = workers_root / stream_id
        output_dir.mkdir()
        start_delay = 0.0
        command = build_worker_command(
            python=args.python,
            spec=payload,
            config=args.config,
            device=args.device,
            accept_threshold=args.accept_threshold,
            reject_threshold=args.reject_threshold,
            realtime=args.realtime,
            punctuation=args.punctuation,
            output_dir=output_dir,
            start_delay=start_delay,
            chunk_phase=lease.phase_offset if args.admission_phase else None,
        )
        log_path = output_dir / "process.log"
        log = log_path.open("w", encoding="utf-8", newline="\n")
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        finally:
            log.close()
        return WorkerHandle(
            process=process,
            command=command,
            details={"worker_root": output_dir, "process_log": log_path},
        )

    service = StreamWorkerService(
        max_active_streams=args.max_active_streams,
        phase_period=args.phase_period,
        launcher=launch,
        estimated_service_seconds=args.estimated_service_seconds,
    )
    control = StreamControlPlane(service, max_service_seconds=args.max_service_seconds)
    stop_event = threading.Event()

    metadata = {
        "started_at": time.time(),
        "host": args.host,
        "port": args.port,
        "max_active_streams": args.max_active_streams,
        "phase_period_seconds": args.phase_period,
        "admission_phase_enabled": args.admission_phase,
        "max_service_seconds": args.max_service_seconds,
        "config": str(args.config.resolve()),
        "device": args.device,
    }
    _write_json(state_root / "service.json", metadata)

    handler = build_handler(control, journal=journal, stop_event=stop_event)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, name="control-plane-http")
    server_thread.start()
    journal.write("started", **metadata)

    def request_stop(signum: int, frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    next_status_at = 0.0
    try:
        while not stop_event.is_set():
            control.poll_once()
            now = time.monotonic()
            if now >= next_status_at:
                _write_json(state_root / "status.json", control.snapshot())
                next_status_at = now + args.status_interval
            if args.poll_interval:
                time.sleep(args.poll_interval)
    finally:
        journal.write("stopping")
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5.0)
        control.stop()
        _write_json(state_root / "status.json", control.snapshot())
        journal.write("stopped")


if __name__ == "__main__":
    main()
