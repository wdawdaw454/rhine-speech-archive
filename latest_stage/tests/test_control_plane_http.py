from __future__ import annotations

import json
from pathlib import Path
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts.run_stream_control_plane import build_handler


class FakeControl:
    max_service_seconds = 10.0

    def __init__(self):
        self.requests = {}

    def submit(self, stream_id, payload):
        status = {"stream_id": stream_id, "state": "queueing"}
        self.requests[stream_id] = status
        return status

    def cancel(self, stream_id):
        if stream_id not in self.requests:
            raise KeyError(stream_id)
        self.requests[stream_id]["state"] = "cancelled"
        return self.requests[stream_id]

    def status(self, stream_id):
        if stream_id not in self.requests:
            raise KeyError(stream_id)
        return self.requests[stream_id]

    def snapshot(self):
        return {"requests": list(self.requests.values())}


def test_control_plane_http_endpoints(tmp_path: Path):
    control = FakeControl()
    stop_event = threading.Event()
    journal_path = tmp_path / "events.jsonl"
    handler = build_handler(control, journal=FakeJournal(journal_path), stop_event=stop_event)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    try:
        with urlopen(f"{base}/health", timeout=2) as response:
            assert json.load(response)["state"] == "running"

        mixture = tmp_path / "mixture.wav"
        references = tmp_path / "references.jsonl"
        profile = tmp_path / "profile.json"
        for path in (mixture, references, profile):
            path.write_text("x", encoding="utf-8")
        request = Request(
            f"{base}/requests",
            data=json.dumps(
                {
                    "stream_id": "stream-1",
                    "mixture": str(mixture),
                    "references": str(references),
                    "profile": str(profile),
                    "target_speaker": "S001",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            assert json.load(response)["stream_id"] == "stream-1"

        request = Request(f"{base}/requests/stream-1", method="DELETE")
        with urlopen(request, timeout=2) as response:
            assert json.load(response)["state"] == "cancelled"

        try:
            urlopen(f"{base}/requests/missing", timeout=2)
        except HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("missing stream should return 404")

        request = Request(f"{base}/shutdown", data=b"", method="POST")
        with urlopen(request, timeout=2) as response:
            assert json.load(response)["state"] == "stopping"
        assert stop_event.is_set()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class FakeJournal:
    def __init__(self, path: Path):
        self.path = path

    def write(self, event: str, **values):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, **values}) + "\n")
