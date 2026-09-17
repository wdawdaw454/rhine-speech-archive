import json
from pathlib import Path
import sys
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import soxr

from src.audio.system_audio import SystemAudioStream, system_audio_device
from src.web.dictation_server import DictationController, DictationModel


def fake_audio(monkeypatch, *, rate=48000, channels=2, blocks=10, failure=0, available=True):
    state = SimpleNamespace(opened=[], closed=0, terminated=0, frames=[], silent_output=[])
    device = {"name": "Test output [Loopback]", "index": 42, "isLoopbackDevice": True,
              "maxInputChannels": channels, "defaultSampleRate": rate}
    output = {"name": "Test output", "index": 41, "maxOutputChannels": channels, "defaultSampleRate": rate}
    class Stream:
        def __init__(self, **kwargs):
            state.opened.append(kwargs)
            self.active = True
            n = kwargs["frames_per_buffer"]
            if kwargs.get("output"):
                rendered, _ = kwargs["stream_callback"](None, n, {}, 0)
                state.silent_output.append(rendered)
                return
            # Preload callbacks so slow consumers exercise queue draining.
            for i in range(blocks):
                t = (np.arange(n) + i * n) / rate
                mono = (.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
                wav = np.repeat(mono[:, None], channels, axis=1)
                state.frames.append(mono)
                _, flag = kwargs["stream_callback"](wav.tobytes(), n,
                    {"input_buffer_adc_time": 10 + i * n / rate}, failure)
                if flag != 0:
                    self.active = False
                    break
        def is_active(self): return self.active
        def stop_stream(self): self.active = False
        def close(self): state.closed += 1
    class Manager:
        def __enter__(self): return self
        def __exit__(self, *_): state.terminated += 1
        def get_default_wasapi_loopback(self):
            if not available:
                raise OSError("no output device")
            return device
        def get_default_wasapi_device(self, *, d_out):
            assert d_out
            if not available: raise OSError("no output device")
            return output
        def get_wasapi_loopback_analogue_by_dict(self, selected):
            assert selected is output
            return device
        def open(self, **kwargs): return Stream(**kwargs)
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", SimpleNamespace(
        PyAudio=Manager, paFloat32=1, paContinue=0, paComplete=1, paAbort=2))
    return state


@pytest.mark.parametrize("rate", [44100, 48000, 96000])
def test_loopback_resamples_continuously_and_flushes_tail(monkeypatch, rate):
    state = fake_audio(monkeypatch, rate=rate)
    stream = SystemAudioStream(duration=1)
    chunks = list(stream)
    actual = np.concatenate([x for _, x in chunks])
    expected = soxr.resample(np.concatenate(state.frames), rate, 16000, quality="HQ")
    assert len(actual) == 16000
    assert actual.dtype == np.float32
    assert np.allclose(actual, expected, atol=1e-6)
    assert [stamp for stamp, _ in chunks] == [i / 10 for i in range(10)]
    assert state.opened[1]["input_device_index"] == 42
    assert state.opened[1]["input"] is True
    assert state.opened[0]["output_device_index"] == 41
    assert not any(state.silent_output[0])
    assert state.closed == 2 and state.terminated == 1


def test_loopback_stop_drains_queued_native_samples_and_filter_tail(monkeypatch):
    state = fake_audio(monkeypatch, blocks=4)
    stop = Event()
    stream = SystemAudioStream(stop_event=stop)
    chunks = []
    for _, chunk in stream:
        chunks.append(chunk)
        stop.set()
    assert sum(map(len, chunks)) == 6400
    assert state.closed == 2


def test_short_native_tail_is_not_lost(monkeypatch):
    state = fake_audio(monkeypatch)
    stream = SystemAudioStream(duration=.235)
    chunks = list(stream)
    assert sum(len(x) for _, x in chunks) == 3760
    assert len(chunks[-1][1]) == 560


def test_loopback_overflow_fails_instead_of_dropping_frames(monkeypatch):
    state = fake_audio(monkeypatch)
    with pytest.raises(RuntimeError, match="积压"):
        list(SystemAudioStream(max_queue_seconds=.1))
    assert state.closed == 2 and state.terminated == 1


def test_loopback_driver_error_closes_devices(monkeypatch):
    state = fake_audio(monkeypatch, failure=2)
    with pytest.raises(RuntimeError, match="异常"):
        list(SystemAudioStream())
    assert state.closed == 2


def test_loopback_explicit_close_releases_native_device(monkeypatch):
    state = fake_audio(monkeypatch, blocks=4)
    stream = SystemAudioStream()
    it = iter(stream)
    next(it)
    stream.close()
    assert state.closed == 2


def test_loopback_stalled_device_fails_and_releases_device(monkeypatch):
    state = fake_audio(monkeypatch, blocks=0)
    with pytest.raises(RuntimeError, match="长时间未返回"):
        list(SystemAudioStream(stall_timeout=.01))
    assert state.closed == 2 and state.terminated == 1


def test_loopback_open_failure_also_closes_silent_output(monkeypatch):
    state = fake_audio(monkeypatch)
    manager = sys.modules["pyaudiowpatch"].PyAudio
    original = manager.open
    def fail_capture(self, **kwargs):
        if kwargs.get("input"):
            raise OSError("capture open failed")
        return original(self, **kwargs)
    monkeypatch.setattr(manager, "open", fail_capture)
    with pytest.raises(RuntimeError, match="capture open failed"):
        list(SystemAudioStream())
    assert state.closed == state.terminated == 1
    assert not any(state.silent_output[0])


def test_cancel_before_start_does_not_open_device(monkeypatch):
    state = fake_audio(monkeypatch)
    stop = Event(); stop.set()
    assert list(SystemAudioStream(stop_event=stop)) == []
    assert not state.opened


def test_device_query_does_not_capture_and_reports_missing_endpoint(monkeypatch):
    state = fake_audio(monkeypatch)
    assert system_audio_device()["name"] == "Test output [Loopback]"
    assert not state.opened
    state = fake_audio(monkeypatch, available=False)
    assert system_audio_device()["available"] is False
    with pytest.raises(RuntimeError, match="默认输出"):
        list(SystemAudioStream())
    assert not state.opened


def test_missing_library_is_not_replaced_with_microphone(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", None)
    with pytest.raises(RuntimeError, match="PyAudioWPatch"):
        list(SystemAudioStream())


class Backend:
    def reset(self): pass
    def accept(self, wav, *, is_final): return "电脑音频结果" if is_final else "电脑音频"
    def transcribe(self, wav): return "非实时电脑音频结果"


def make_controller(tmp_path, **kwargs):
    (tmp_path / "model.onnx").write_bytes(b"fake")
    spec = DictationModel("fake", "Fake", tmp_path, "test", modes=("streaming", "offline"),
                         recognition_types=("normal", "target"))
    calls = []
    class SystemStream:
        device_name = "Test output [Loopback]"
        native_sample_rate, native_channels = 48000, 2
        statuses = []
        def __iter__(self):
            for _ in range(15): yield 0, np.full(1600, .1, np.float32)
            for _ in range(6): yield 0, np.zeros(1600, np.float32)
        def close(self): pass
    def system_factory(stop):
        calls.append("system")
        return SystemStream()
    c = DictationController(project_root=tmp_path, models=[spec], backend_factory=lambda _: Backend(),
        stream_factory=lambda _: pytest.fail("microphone must not open"),
        system_stream_factory=kwargs.pop("system_stream_factory", system_factory), **kwargs)
    # Controller cache cleanup imports torch once, even for a fake backend;
    # allow a cold Windows import without racing the first test's start().
    c.load_model("fake"); c.wait(60)
    assert c.snapshot()["loaded_model_id"] == "fake", c.snapshot()
    return c, calls


@pytest.mark.parametrize("mode", ["streaming", "offline"])
def test_system_source_routes_to_all_modes_and_saves_audio(tmp_path, mode):
    c, calls = make_controller(tmp_path)
    c.start("fake", mode, source="system")
    c.wait(5)
    status = c.snapshot()
    assert status["state"] == "idle", status
    assert status["source"] == "system"
    assert status["capture_device"] == "Test output [Loopback]"
    assert "电脑音频结果" in status["committed_text"]
    assert calls == ["system"]
    assert c.recording_path().is_file()
    summary = json.loads((Path(status["output_dir"]) / "summary.json").read_text(encoding="utf-8"))
    assert summary["source"] == "system"
    assert summary["capture_native_sample_rate"] == 48000


@pytest.mark.parametrize("source", ["unknown", "wav"])
def test_invalid_live_source_rejected(tmp_path, source):
    c, _ = make_controller(tmp_path)
    with pytest.raises(ValueError): c.start("fake", source=source)


def test_system_file_conflict_rejected(tmp_path):
    c, _ = make_controller(tmp_path)
    with pytest.raises(ValueError):
        c.start("fake", source="system", file_audio=np.ones(1600))


def test_system_device_failure_exposed_without_mic_fallback(tmp_path):
    def fail(_): raise RuntimeError("output disconnected")
    c, _ = make_controller(tmp_path, system_stream_factory=fail)
    c.start("fake", source="system"); c.wait(5)
    assert c.snapshot()["state"] == "error"
    assert "output disconnected" in c.snapshot()["error"]


def test_existing_target_mode_accepts_system_audio(tmp_path):
    from src.models.streaming_vad import VadBoundary
    from src.speaker.enrollment import SpeakerProfile
    from src.speaker.web_target import MODEL_ID
    class Vad:
        def reset(self): self.started = False
        def accept(self, chunk, *, is_final):
            if not self.started:
                self.started = True
                return [VadBoundary(0, None)]
            return []
    c, calls = make_controller(tmp_path)
    c._target_models = SimpleNamespace(vad=Vad(), encoder=SimpleNamespace(extract=lambda *_: np.array([1., 0.])))
    c._profile = SpeakerProfile("test", MODEL_ID, 2, [1., 0.], "l2", 1, 3, "now")
    c.start("fake", source="system", target_only=True); c.wait(5)
    status = c.snapshot()
    assert status["state"] == "idle", status
    assert status["target_only"] and status["source"] == "system"
    assert status["committed_text"] and calls == ["system"]
