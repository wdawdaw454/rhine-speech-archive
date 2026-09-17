from __future__ import annotations

import sys
from threading import Event, Thread
import time
from types import SimpleNamespace

import numpy as np

from src.audio.microphone import MicrophoneStream


class FakeCallbackStop(Exception):
    pass


class FakeInputStream:
    def __init__(self, *, callback, blocksize, **_kwargs) -> None:
        self.callback = callback
        self.blocksize = blocksize
        self.active = False
        self.worker: Thread | None = None

    def __enter__(self):
        self.active = True

        def produce() -> None:
            try:
                for index in range(10):
                    chunk = np.full((self.blocksize, 1), index, dtype=np.float32)
                    self.callback(chunk, self.blocksize, None, None)
                    time.sleep(0.002)
            except FakeCallbackStop:
                pass
            finally:
                self.active = False

        self.worker = Thread(target=produce, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, *_args) -> None:
        if self.worker is not None:
            self.worker.join(1.0)


def test_callback_capture_survives_slow_consumer(monkeypatch):
    fake_module = SimpleNamespace(InputStream=FakeInputStream, CallbackStop=FakeCallbackStop)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_module)
    stream = MicrophoneStream(sample_rate=100, blocksize=10, duration=1.0)

    chunks = []
    for _, chunk in stream:
        chunks.append(chunk)
        time.sleep(0.01)

    assert len(chunks) == 10
    assert sum(len(chunk) for chunk in chunks) == 100
    assert stream.statuses == []


def test_callback_capture_stops_only_after_stop_event(monkeypatch):
    fake_module = SimpleNamespace(InputStream=FakeInputStream, CallbackStop=FakeCallbackStop)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_module)
    stopped = Event()
    stream = MicrophoneStream(sample_rate=100, blocksize=10, stop_event=stopped)

    chunks = []
    for _, chunk in stream:
        chunks.append(chunk)
        if len(chunks) == 3:
            stopped.set()

    assert len(chunks) == 3
    assert "driver-clock-stall" not in stream.statuses


def test_stop_preserves_already_queued_audio(monkeypatch):
    class PreloadedStream(FakeInputStream):
        def __enter__(self):
            for index in range(4):
                self.callback(np.full((10, 1), index, dtype=np.float32), 10, None, None)
            return self

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(
        InputStream=PreloadedStream, CallbackStop=FakeCallbackStop))
    stopped = Event()
    chunks = []
    for _, chunk in MicrophoneStream(sample_rate=100, blocksize=10, stop_event=stopped):
        chunks.append(chunk)
        stopped.set()
    assert len(chunks) == 4
