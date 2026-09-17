"""Thread-safe control facade around the resident stream worker service."""

from __future__ import annotations

import threading
from typing import Any

from .service import ACTIVE, StreamWorkerService


class StreamControlPlane:
    """Serialize control-plane mutations around a long-running worker service.

    The scheduler remains single-threaded by design. HTTP or command handlers may
    run on other threads, but they enter through this lock instead of mutating
    admission state concurrently.
    """

    def __init__(
        self,
        service: StreamWorkerService,
        *,
        max_service_seconds: float = 300.0,
    ) -> None:
        if max_service_seconds <= 0:
            raise ValueError("max_service_seconds must be positive")
        self.service = service
        self.max_service_seconds = float(max_service_seconds)
        self._lock = threading.RLock()

    def submit(self, stream_id: str, payload: Any = None) -> dict[str, Any]:
        with self._lock:
            return self.service.submit(stream_id, payload)

    def cancel(self, stream_id: str) -> dict[str, Any]:
        with self._lock:
            return self.service.cancel(stream_id)

    def status(self, stream_id: str) -> dict[str, Any]:
        with self._lock:
            return self.service.status(stream_id)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self.service.snapshot()

    def poll_once(self) -> None:
        with self._lock:
            now = self.service.clock()
            for record in list(self.service.records):
                if (
                    record.state == ACTIVE
                    and record.launched_at is not None
                    and now - record.launched_at >= self.max_service_seconds
                ):
                    self.service.cancel(record.stream_id)
            self.service.poll_once()

    def stop(self, *, timeout: float = 10.0) -> None:
        with self._lock:
            self.service.shutdown(timeout=timeout)
