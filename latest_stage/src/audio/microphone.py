"""Callback-backed microphone input stream for local Windows dictation."""

from __future__ import annotations

from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event
from typing import Iterator

import numpy as np


class MicrophoneStream:
    """Yield `(audio_time, chunk)` tuples from a mono 16 kHz microphone."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        blocksize: int = 1600,
        device: int | str | None = None,
        duration: float | None = None,
        stop_event: Event | None = None,
        max_queue_seconds: float = 30.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if blocksize <= 0:
            raise ValueError("blocksize must be positive")
        if duration is not None and duration <= 0:
            raise ValueError("duration must be positive")
        if max_queue_seconds <= 0:
            raise ValueError("max_queue_seconds must be positive")
        self.sample_rate = int(sample_rate)
        self.blocksize = int(blocksize)
        self.device = device
        self.duration = None if duration is None else float(duration)
        self.stop_event = stop_event
        self.max_queue_seconds = float(max_queue_seconds)
        self.statuses: list[str] = []

    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - exercised only on host installs
            raise RuntimeError(
                "sounddevice is required for microphone input; "
                "install it in the Windows environment"
            ) from exc

        total_samples = None
        if self.duration is not None:
            total_samples = int(round(self.duration * self.sample_rate))

        max_blocks = max(
            1,
            int(np.ceil(self.max_queue_seconds * self.sample_rate / self.blocksize)),
        )
        audio_queue: Queue[np.ndarray] = Queue(maxsize=max_blocks)

        def callback(indata, _frames, _time_info, status) -> None:
            if self.stop_event is not None and self.stop_event.is_set():
                raise sd.CallbackStop
            if status:
                self.statuses.append(f"portaudio-{status}")
            chunk = np.asarray(indata, dtype=np.float32)
            if chunk.ndim == 2:
                chunk = chunk.mean(axis=1)
            chunk = chunk.ravel().copy()
            if chunk.size == 0:
                return
            try:
                audio_queue.put_nowait(chunk)
            except Full:
                # Never block PortAudio's callback. A 30-second queue means this
                # is a sustained backend overload, not a transient cold start.
                self.statuses.append("capture-queue-overflow")
                try:
                    audio_queue.get_nowait()
                except Empty:
                    pass
                audio_queue.put_nowait(chunk)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=callback,
        ) as stream:
            consumed = 0
            while total_samples is None or consumed < total_samples:
                # Preserve audio already captured while a streaming decode was
                # running. The callback stops adding blocks after this event.
                if self.stop_event is not None and self.stop_event.is_set() and audio_queue.empty():
                    break
                try:
                    chunk = audio_queue.get(timeout=0.10)
                except Empty:
                    if not stream.active:
                        break
                    continue
                if total_samples is not None:
                    chunk = chunk[: total_samples - consumed]
                yield consumed / self.sample_rate, chunk
                consumed += chunk.size


def write_status_file(path: str | Path, statuses: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(statuses) + ("\n" if statuses else ""), encoding="utf-8")
