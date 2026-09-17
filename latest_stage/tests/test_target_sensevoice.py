"""SenseVoice verified-prefix scheduling and controller integration."""
import json
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from src.pipeline.events import EventType, PipelineEvent
from src.pipeline.target_dictation import TargetDictationPipeline
from src.models.streaming_vad import VadBoundary
from src.speaker.enrollment import SpeakerProfile
from src.speaker.web_target import MODEL_ID
from src.web.dictation_server import DictationController, DictationModel, default_models


def audio(seconds, value=.1):
    return np.full(round(seconds * 16000), value, np.float32)


class Vad:
    def reset(self): self.total = 0
    def accept(self, chunk, *, is_final):
        old = self.total
        self.total += len(chunk)
        return [VadBoundary(0, None)] if old == 0 else []


class Encoder:
    def extract(self, chunk, sr):
        return np.array([1., 0.]) if np.mean(chunk) > 0 else np.array([0., 1.])


class PrefixBackend:
    def __init__(self):
        self.calls = []
        self.reset()
    def reset(self): self.prefix = np.empty(0, np.float32)
    def accept(self, chunk, *, is_final):
        self.prefix = np.concatenate([self.prefix, chunk])
        self.calls.append((chunk.copy(), self.prefix.copy(), is_final))
        return f"字数{len(self.prefix)}"


class Sink:
    def __init__(self): self.events = []
    def write(self, event): self.events.append(event)


def pipeline(backend=None, vad=None):
    backend, sink = backend or PrefixBackend(), Sink()
    p = TargetDictationPipeline(run_id="prefix", backend=backend, vad=vad or Vad(),
        encoder=Encoder(), embedding=np.array([1., 0.]), event_sink=sink)
    return p, backend, sink


def feed(p, data):
    for offset in range(0, len(data), 1600):
        p.push_chunk(data[offset:offset + 1600])
    return p.finish()


def test_one_decode_per_verified_window_and_one_final_for_short_tail():
    p, backend, sink = pipeline()
    stats = feed(p, audio(5.2))
    assert [len(c[0]) for c in backend.calls] == [24000, 24000, 24000, 11200]
    assert [c[2] for c in backend.calls] == [False, False, False, True]
    assert len(backend.calls[-1][1]) == 83200
    assert stats.decode_calls == 4
    assert stats.first_text_audio_seconds == 1.8
    assert stats.accepted_seconds == 5.2
    assert stats.backend_seconds >= 0
    assert [e.text for e in sink.events if e.type == EventType.CAPTION_COMMIT] == ["字数83200"]


def test_rejected_window_never_enters_prefix_and_next_target_starts_fresh():
    p, backend, sink = pipeline()
    feed(p, np.concatenate([audio(1.5), audio(1.5, -.1), audio(1.5, .2)]))
    final_prefixes = [c[1] for c in backend.calls if c[2]]
    assert [len(c) for c in final_prefixes] == [24000, 24000]
    assert np.allclose(final_prefixes[0], .1)
    assert np.allclose(final_prefixes[1], .2)
    assert all(np.all(c[1] > 0) for c in backend.calls)
    kinds = [e.type for e in sink.events]
    assert kinds.index(EventType.CAPTION_COMMIT) < kinds.index(EventType.CAPTION_SUPPRESS)


def test_long_prefix_is_bounded_and_last_window_is_final_not_preview_twice():
    p, backend, _ = pipeline()
    stats = feed(p, audio(13.5))
    assert max(len(c[1]) for c in backend.calls) == 12 * 16000
    assert [len(c[1]) for c in backend.calls if c[2]] == [12 * 16000, 24000]
    assert sum(len(c[1]) == 12 * 16000 for c in backend.calls) == 1
    assert stats.committed_segments == 2
    assert sum(len(c[0]) for c in backend.calls) == 13.5 * 16000


@pytest.mark.parametrize("seconds,expected", [(.5, 0), (.6, 9600), (1., 16000)])
def test_vad_stop_padding_is_not_speaker_or_asr_audio(seconds, expected):
    class EndInPadding(Vad):
        def accept(self, chunk, *, is_final):
            bounds = super().accept(chunk, is_final=is_final)
            return bounds + ([VadBoundary(None, self.total / 16000)] if is_final else [])
    p, backend, _ = pipeline(vad=EndInPadding())
    stats = feed(p, audio(seconds))
    assert sum(len(c[0]) for c in backend.calls) == expected
    assert all(np.all(c[0] > 0) for c in backend.calls)
    assert stats.accepted_seconds == seconds


def test_final_empty_does_not_commit_old_preview():
    class EmptyFinal(PrefixBackend):
        def accept(self, chunk, *, is_final):
            text = super().accept(chunk, is_final=is_final)
            return "" if is_final else text
    p, _, sink = pipeline(backend=EmptyFinal())
    stats = feed(p, audio(2.1))
    assert any(e.type == EventType.ASR_PARTIAL and e.text for e in sink.events)
    assert not any(e.type == EventType.CAPTION_COMMIT for e in sink.events)
    assert stats.committed_segments == 0
    assert any(e.extras.get("reason") == "empty_final" for e in sink.events)


def test_preview_can_be_revised_to_empty():
    class Revision(PrefixBackend):
        def accept(self, chunk, *, is_final):
            text = super().accept(chunk, is_final=is_final)
            return text if len(self.calls) == 1 else ""
    p, _, sink = pipeline(backend=Revision())
    feed(p, audio(3.3))
    assert [e.text for e in sink.events if e.type == EventType.ASR_PARTIAL] == ["字数24000", ""]


def controller(tmp_path, **kwargs):
    (tmp_path / "model.onnx").write_bytes(b"fake")
    model = DictationModel("sensevoice-realtime", "SenseVoice", tmp_path, "test",
        decode_interval=.8, recognition_types=("normal", "target"))
    backend = kwargs.pop("backend", None) or PrefixBackend()
    c = DictationController(project_root=tmp_path, models=[model], backend_factory=lambda _: backend, **kwargs)
    c._target_models = type("Models", (), {"vad": Vad(), "encoder": Encoder()})()
    c._profile = SpeakerProfile("test", MODEL_ID, 2, [1., 0.], "l2", 1, 3., "now")
    c.load_model(model.id); c.wait(60)
    assert c.snapshot()["loaded_model_id"] == model.id, c.snapshot()
    return c, backend


@pytest.mark.parametrize("source", ["wav", "microphone", "system"])
def test_controller_all_inputs_use_prefix_strategy_and_keep_full_recording(tmp_path, source):
    data = np.concatenate([audio(1.5), audio(1.5, -.1), audio(1.5)])
    selected = []
    def stream(kind):
        def factory(stop):
            selected.append(kind)
            for i in range(0, len(data), 1600): yield i / 16000, data[i:i + 1600]
        return factory
    c, backend = controller(tmp_path, stream_factory=stream("microphone"), system_stream_factory=stream("system"))
    options = {"file_audio": data, "filename": "sample.wav"} if source == "wav" else {"source": source}
    c.start("sensevoice-realtime", target_only=True, **options); c.wait(5)
    s = c.snapshot()
    assert s["state"] == "idle", s
    assert s["source"] == source and s["target_only"]
    assert s["committed_text"] == "字数24000\n字数24000"
    assert selected == ([] if source == "wav" else [source])
    assert [len(call[0]) for call in backend.calls] == [24000, 0, 24000, 0]
    summary = json.loads((Path(s["output_dir"]) / "summary.json").read_text(encoding="utf-8"))
    assert summary["parameters"]["target_decode_strategy"] == "verified_prefix"
    assert summary["stats"]["decode_calls"] == 4
    import soundfile as sf
    saved, _ = sf.read(c.recording_path())
    assert len(saved) == len(data) and np.min(saved) < 0  # Whole input is saved, only text is filtered.


def test_stop_flushes_approved_short_tail_without_reopening_input(tmp_path):
    delivered = Event()
    def stream(stop):
        for i in range(10): yield i / 10, audio(.1)
        delivered.set()
        assert stop.wait(5)
    c, backend = controller(tmp_path, stream_factory=stream)
    c.start("sensevoice-realtime", target_only=True)
    assert delivered.wait(5)
    assert not backend.calls
    c.stop(); c.wait(5)
    assert c.snapshot()["state"] == "idle"
    assert c.snapshot()["committed_text"] == "字数16000"
    assert c.snapshot()["partial_text"] == ""
    assert len(backend.calls) == 1 and backend.calls[0][2]


def test_decode_failure_closes_input_saves_audio_and_allows_clean_retry(tmp_path):
    closed = []
    class Failure(PrefixBackend):
        fail = True
        def accept(self, chunk, *, is_final):
            if self.fail: raise RuntimeError("synthetic prefix failure")
            return super().accept(chunk, is_final=is_final)
    def stream(stop):
        try:
            for i in range(24): yield i / 10, audio(.1)
        finally: closed.append(True)
    backend = Failure()
    c, _ = controller(tmp_path, backend=backend, stream_factory=stream)
    c.start("sensevoice-realtime", target_only=True); c.wait(5)
    assert c.snapshot()["state"] == "error"
    assert "synthetic prefix failure" in c.snapshot()["error"]
    assert c.recording_path().is_file() and closed == [True]
    backend.fail = False
    c.start("sensevoice-realtime", target_only=True); c.wait(5)
    assert c.snapshot()["state"] == "idle"
    assert c.snapshot()["committed_text"] == "字数38400"
    assert closed == [True, True]


def test_empty_partial_event_clears_old_prefix(tmp_path):
    c, _ = controller(tmp_path)
    c.handle_event(PipelineEvent(type=EventType.ASR_PARTIAL, run_id="x", text="old"))
    c.handle_event(PipelineEvent(type=EventType.ASR_PARTIAL, run_id="x", text=""))
    assert c.snapshot()["partial_text"] == ""


def test_registry_exposes_only_realtime_sensevoice_for_target(tmp_path):
    models = default_models(tmp_path)
    target = [m.public_dict() for m in models if "target" in m.recognition_types]
    assert [m["target_name"] for m in target] == ["FSMN-VAD + CAM++ + SenseVoice"]
    assert [m["id"] for m in target] == ["sensevoice-realtime"]
