import json
from pathlib import Path
from threading import Event
import time

import numpy as np
import pytest

from src.models.streaming_vad import VadBoundary
from src.pipeline.events import EventType
from src.pipeline.nano_realtime import NanoRealtimePipeline
from src.web.dictation_server import DictationController, DictationModel


class Vad:
    def __init__(self, ends=(), silent=False):
        self.ends, self.silent = ends, silent
        self.reset()

    def reset(self):
        self.n = 0
        self.active = False

    def accept(self, chunk, *, is_final):
        if self.silent:
            return []
        if is_final:
            return [VadBoundary(None, self.n / 16000)] if self.active else []
        result = []
        if not self.active:
            result.append(VadBoundary(self.n / 16000, None))
            self.active = True
        self.n += len(chunk)
        if self.n in self.ends:
            result.append(VadBoundary(None, self.n / 16000))
            self.active = False
        return result


class Sink:
    def __init__(self):
        self.events = []
        self.partial = Event()

    def write(self, event):
        self.events.append(event)
        if event.type == EventType.ASR_PARTIAL:
            self.partial.set()

    @property
    def commits(self):
        return [e for e in self.events if e.type == EventType.CAPTION_COMMIT]


class Backend:
    def __init__(self, vad=None):
        self.calls = []
        self.realtime_vad = vad or Vad()

    def reset(self):
        pass

    def transcribe(self, audio):
        self.calls.append(audio.copy())
        return f"识别{len(audio)}"


def audio(seconds):
    return np.full(round(seconds * 16000), 0.08, np.float32)


def pipeline(backend=None, vad=None, **kwargs):
    sink = Sink()
    backend = backend or Backend()
    return NanoRealtimePipeline(run_id="test", backend=backend, vad=vad or Vad(),
                                event_sink=sink, **kwargs), sink, backend


def test_preview_starts_only_after_initial_audio_and_final_reuses_exact_input():
    p, sink, backend = pipeline()
    try:
        p.push_chunk(audio(1.1))
        assert not backend.calls
        p.push_chunk(audio(0.1))
        assert sink.partial.wait(3)
        stats = p.finish()
        assert len(backend.calls) == 1
        assert len(sink.commits) == 1
        assert stats.reused_finals == 1
        assert sink.commits[0].extras["reused_preview"]
    finally:
        p.close()


def test_short_utterance_still_gets_final_on_stop_without_a_preview():
    p, sink, backend = pipeline()
    p.push_chunk(audio(0.4))
    stats = p.finish()
    assert stats.partial_events == 0
    assert len(backend.calls) == 1
    assert sink.commits[0].text == "识别6400"
    assert not p._worker.is_alive()


def test_empty_session_closes_without_model_calls():
    p, sink, backend = pipeline()
    assert p.finish().accepted_seconds == 0
    assert not backend.calls
    assert not p._worker.is_alive()


def test_silence_never_calls_nano():
    p, sink, backend = pipeline(vad=Vad(silent=True))
    for _ in range(200):
        p.push_chunk(np.zeros(1600, np.float32))
    assert len(p._audio) <= 16000
    p.finish()
    assert not backend.calls
    assert not sink.commits


def test_long_continuous_audio_is_bounded_and_no_samples_are_lost():
    p, sink, backend = pipeline(decode_interval=100)
    for _ in range(175):
        p.push_chunk(audio(0.1))
    p.finish()
    assert [(e.start, e.end) for e in sink.commits] == [(0, 8), (8, 16), (16, 17.5)]
    assert max(len(x) for x in backend.calls) <= 8 * 16000
    assert sum(e.end - e.start for e in sink.commits) == 17.5


def test_slow_inference_does_not_block_capture_and_coalesces_pending_previews():
    entered, release = Event(), Event()

    class Slow(Backend):
        def transcribe(self, wav):
            self.calls.append(wav.copy())
            if len(self.calls) == 1:
                entered.set()
                assert release.wait(5)
            return "完整文本"

    backend = Slow()
    p, sink, _ = pipeline(backend=backend)
    try:
        p.push_chunk(audio(1.2))
        assert entered.wait(3)
        for _ in range(4):
            p.push_chunk(audio(1.2))
        # VAD/capture progressed while the only ASR call remained blocked.
        assert p.stats.accepted_seconds == 6
        assert len(backend.calls) == 1
        assert p.stats.coalesced_previews == 3
        release.set()
        p.finish()
        assert len(sink.commits) == 1
        assert len(backend.calls) <= 3
        assert len(backend.calls[-1]) == 6 * 16000
    finally:
        release.set()
        p.close()


def test_stale_preview_cannot_overwrite_new_segment_and_finals_stay_ordered():
    entered, release = Event(), Event()

    class Slow(Backend):
        def transcribe(self, wav):
            self.calls.append(wav.copy())
            if len(self.calls) == 1:
                entered.set()
                assert release.wait(5)
                return "旧预览"
            return "定稿"

    p, sink, _ = pipeline(backend=Slow(), vad=Vad(ends=(32000,)))
    try:
        p.push_chunk(audio(1.2))
        assert entered.wait(3)
        p.push_chunk(audio(0.8))
        p.push_chunk(audio(0.5))
        release.set()
        p.finish()
        assert not any(e.type == EventType.ASR_PARTIAL and e.text == "旧预览" for e in sink.events)
        assert [(e.start, e.end) for e in sink.commits] == [(0, 2), (2, 2.5)]
    finally:
        release.set()
        p.close()


def test_final_trim_does_not_reuse_preview_with_extra_silence():
    class TrimVad(Vad):
        def accept(self, chunk, *, is_final):
            if is_final:
                return [VadBoundary(None, 1.0)]
            return super().accept(chunk, is_final=False)

    p, sink, backend = pipeline(vad=TrimVad())
    p.push_chunk(audio(1.2))
    assert sink.partial.wait(3)
    stats = p.finish()
    assert [len(x) for x in backend.calls] == [19200, 16000]
    assert stats.reused_finals == 0
    assert sink.commits[0].end == 1


def test_inference_failure_is_propagated_and_worker_is_closed():
    class Broken(Backend):
        def transcribe(self, wav):
            raise ValueError("GPU failure")
    p, _, _ = pipeline(backend=Broken())
    p.push_chunk(audio(0.4))
    with pytest.raises(RuntimeError, match="GPU failure"):
        p.finish()
    assert not p._worker.is_alive()


def test_backlog_is_bounded_without_silent_audio_drops():
    entered, release = Event(), Event()
    class Slow(Backend):
        def transcribe(self, wav):
            entered.set()
            assert release.wait(5)
            return "测试"
    p, _, _ = pipeline(backend=Slow(), max_backlog_seconds=1)
    try:
        p.push_chunk(audio(1.2))
        assert entered.wait(3)
        with pytest.raises(RuntimeError, match="积压"):
            p.push_chunk(audio(6.8))
    finally:
        release.set()
        p.close()


def controller(tmp_path, stream_factory=None, vad=None, system_stream_factory=None):
    (tmp_path / "model.pt").write_bytes(b"test")
    spec = DictationModel("fun-asr-nano", "Fun-ASR-Nano", tmp_path, "test",
                         backend="funasr-offline", required_file="model.pt",
                         modes=("streaming", "offline"), decode_interval=1.2)
    backend = Backend(vad=vad)
    c = DictationController(project_root=tmp_path, models=[spec],
                            backend_factory=lambda _: backend, stream_factory=stream_factory,
                            system_stream_factory=system_stream_factory)
    c.load_model(spec.id)
    c.wait(3)
    assert c.snapshot()["state"] == "idle"
    return c, backend


def test_wav_runs_in_real_time_without_opening_mic_and_saves_final(tmp_path):
    def forbidden(_):
        raise AssertionError("must not open microphone")
    c, backend = controller(tmp_path, stream_factory=forbidden)
    started = time.monotonic()
    c.start("fun-asr-nano", file_audio=audio(0.4), filename="file.wav")
    c.wait(5)
    status = c.snapshot()
    assert status["state"] == "idle", status
    assert time.monotonic() - started >= 0.38
    assert status["committed_text"] == "识别6400"
    assert status["recording_seconds"] == 0.4
    assert len(backend.calls) == 1
    report = json.loads((Path(status["output_dir"]) / "summary.json").read_text(encoding="utf-8"))
    assert report["parameters"]["file_realtime_replay"]
    assert report["parameters"]["vad_end_silence_ms"] == 600
    assert report["stats"]["committed_segments"] == 1


def test_stop_file_replay_drains_only_audio_already_delivered(tmp_path):
    c, backend = controller(tmp_path)
    c.start("fun-asr-nano", file_audio=audio(10), filename="long.wav")
    deadline = time.monotonic() + 3
    while c.snapshot()["recording_seconds"] < 0.3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert c.snapshot()["state"] == "listening"
    c.stop()
    c.wait(5)
    status = c.snapshot()
    assert status["state"] == "idle", status
    assert 0.3 <= status["recording_seconds"] < 1
    assert status["committed_text"]
    assert len(backend.calls[-1]) < 16000


def test_mic_input_uses_same_vad_and_final_pipeline(tmp_path):
    def stream(_):
        for i in range(4):
            yield i * .1, audio(.1)
    c, backend = controller(tmp_path, stream_factory=stream)
    c.start("fun-asr-nano")
    c.wait(5)
    status = c.snapshot()
    assert status["state"] == "idle", status
    assert status["source"] == "microphone"
    assert status["committed_text"] == "识别6400"
    assert backend.realtime_vad.n == 6400
    assert c.recording_path().is_file()


def test_nano_target_mode_is_not_accidentally_enabled(tmp_path):
    c, _ = controller(tmp_path)
    with pytest.raises(ValueError, match="仅支持"):
        c.start("fun-asr-nano", target_only=True)


def test_system_input_uses_nano_vad_without_opening_microphone(tmp_path):
    def forbidden(_):
        raise AssertionError("must not open microphone")
    def stream(_):
        for i in range(4):
            yield i * .1, audio(.1)
    c, backend = controller(tmp_path, stream_factory=forbidden, system_stream_factory=stream)
    c.start("fun-asr-nano", source="system")
    c.wait(5)
    status = c.snapshot()
    assert status["state"] == "idle", status
    assert status["source"] == "system"
    assert status["committed_text"] == "识别6400"
    assert backend.realtime_vad.n == 6400
    assert c.recording_path().is_file()


def test_nano_offline_remains_unpaced(tmp_path):
    c, backend = controller(tmp_path)
    c.start("fun-asr-nano", "offline", file_audio=audio(4), filename="file.wav")
    c.wait(3)
    status = c.snapshot()
    assert status["state"] == "idle", status
    assert status["committed_text"] == "识别64000"
    assert backend.realtime_vad.n == 0


def test_controller_preserves_recording_and_error_on_nano_failure(tmp_path):
    def stream(_):
        yield 0, audio(.4)
    c, backend = controller(tmp_path, stream_factory=stream)
    def fail(_):
        raise ValueError("synthetic decode failure")
    backend.transcribe = fail
    c.start("fun-asr-nano")
    c.wait(5)
    status = c.snapshot()
    assert status["state"] == "error", status
    assert "synthetic decode failure" in status["error"]
    assert status["partial_text"] == ""
    assert c.recording_path().is_file()
    # A subsequent attempt reuses the model only after the failed worker exits.
    backend.transcribe = lambda _: "恢复后的结果"
    c.start("fun-asr-nano")
    c.wait(5)
    assert c.snapshot()["state"] == "idle"
    assert c.snapshot()["committed_text"] == "恢复后的结果"


def test_vad_failure_stops_cleanly_and_preserves_accepted_audio(tmp_path):
    class BrokenVad(Vad):
        def accept(self, chunk, *, is_final):
            raise ValueError("VAD failure")
    def stream(_):
        yield 0, audio(.4)
    c, _ = controller(tmp_path, stream_factory=stream, vad=BrokenVad())
    c.start("fun-asr-nano")
    c.wait(5)
    status = c.snapshot()
    assert status["state"] == "error"
    assert "VAD failure" in status["error"]
    assert c.recording_path().is_file()
