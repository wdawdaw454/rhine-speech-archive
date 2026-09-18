from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from src.models.streaming_vad import VadBoundary
from src.pipeline.events import EventType
from src.pipeline.target_dictation import TargetDictationPipeline
from src.speaker.enrollment import SpeakerProfile
from src.speaker.web_target import MODEL_ID, TargetModels
from src.web.dictation_server import DictationController, DictationModel


def _model(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_bytes(b"fake")
    return DictationModel(id="fake", name="fake", description="test", model_dir=model_dir,
                          recognition_types=("normal", "target"))


class FakeBackend:
    def reset(self): pass
    def accept(self, audio, *, is_final): return "十二号" if is_final else "十二"


class Sink:
    def __init__(self): self.events = []
    def write(self, event): self.events.append(event)


class Vad:
    def reset(self): self.count = 0
    def accept(self, audio, *, is_final):
        self.count += len(audio)
        # Delayed start must replay buffered samples, not lose the onset.
        return [VadBoundary(0, None)] if self.count == 4800 else []


class Encoder:
    def extract(self, audio, sr):
        return np.array([1., 0.]) if np.mean(audio) > 0 else np.array([0., 1.])


class Backend(FakeBackend):
    def __init__(self): self.samples = []
    def accept(self, audio, *, is_final):
        self.samples.extend(audio.tolist())
        return super().accept(audio, is_final=is_final)


def pipeline(**kwargs):
    sink, backend = Sink(), Backend()
    p = TargetDictationPipeline(run_id="test", backend=backend, vad=Vad(), encoder=Encoder(),
        embedding=np.array([1., 0.]), event_sink=sink, **kwargs)
    return p, sink, backend


def feed(p, audio):
    for offset in range(0, len(audio), 1600): p.push_chunk(audio[offset:offset + 1600])
    return p.finish()


def test_gates_each_window_and_recovers_after_speaker_change():
    p, sink, backend = pipeline()
    audio = np.concatenate([np.full(24000, 0.1), np.full(24000, -0.1), np.full(24000, 0.1)])
    stats = feed(p, audio)
    assert stats.verified_windows == 2
    assert stats.suppressed_windows >= 1
    assert stats.committed_segments == 2
    assert min(backend.samples) > 0  # No rejected audio ever reaches ASR.
    assert len(backend.samples) == 48000  # Includes the delayed VAD onset.
    decisions = [e.speaker_decision for e in sink.events if e.type == EventType.SPEAKER_DECISION]
    assert decisions[:3] == ["target", "non_target", "target"]


def test_no_partial_before_identity_and_short_speech_suppressed():
    p, sink, backend = pipeline()
    for _ in range(5): p.push_chunk(np.full(1600, 0.1))
    assert not backend.samples
    p.finish()
    assert not [e for e in sink.events if e.type in {EventType.ASR_PARTIAL, EventType.CAPTION_COMMIT}]


def test_empty_finish_and_non_target_never_decoded():
    p, sink, backend = pipeline()
    assert p.finish().accepted_seconds == 0
    p, sink, backend = pipeline()
    feed(p, np.full(48000, -0.1))
    assert not backend.samples
    assert not [e for e in sink.events if e.text]


def test_invalid_embedding_fails_closed():
    p, sink, backend = pipeline()
    p.encoder.extract = lambda *_: np.array([float("nan"), 0.])
    with pytest.raises(ValueError): feed(p, np.full(48000, 0.1))
    assert not backend.samples


class Models:
    def __init__(self): self.vad, self.encoder = Vad(), Encoder()
    def enroll(self, audio, name, sr):
        if not len(audio) or np.mean(audio) <= 0: raise ValueError("无有效语音")
        return SpeakerProfile(name, MODEL_ID, 2, [1., 0.], "l2", 1, len(audio)/sr, "now")


def controller(tmp_path, **kwargs):
    model = _model(tmp_path)
    c = DictationController(project_root=tmp_path, models=[model], backend_factory=lambda _: Backend(),
        target_factory=Models, **kwargs)
    return c, model


def prepare(c, model):
    c.load_target(); c.wait(5)
    c.enroll("目标", np.full(48000, 0.1)); c.wait(5)
    c.load_model(model.id); c.wait(5)


def test_enrollment_is_independent_persistent_and_atomic(tmp_path):
    c, model = controller(tmp_path, stream_factory=lambda _: pytest.fail("microphone opened"))
    c.load_target(); c.wait(5)
    assert c.snapshot()["loaded_model_id"] is None
    c.enroll("目标", np.full(48000, 0.1)); c.wait(5)
    assert c.snapshot()["speaker_profile"]["name"] == "目标"
    assert not list(tmp_path.rglob("*.wav"))
    c.enroll("失败", np.full(48000, -0.1)); c.wait(5)
    assert c.snapshot()["state"] == "error"
    assert c.snapshot()["speaker_profile"]["name"] == "目标"
    restored = DictationController(project_root=tmp_path, models=[model])
    assert restored.snapshot()["speaker_profile"]["name"] == "目标"
    restored.forget_speaker()
    assert not restored._profile_path.exists()
    assert restored.snapshot()["speaker_profile"] is None


def test_target_file_only_commits_approved_audio(tmp_path):
    c, model = controller(tmp_path, stream_factory=lambda _: pytest.fail("microphone opened"))
    prepare(c, model)
    audio = np.concatenate([np.full(24000, 0.1), np.full(24000, -0.1)])
    c.start(model.id, file_audio=audio, filename="two.wav", target_only=True); c.wait(5)
    status = c.snapshot()
    assert status["state"] == "idle", status
    assert status["committed_text"] == "十二号"
    assert status["partial_text"] == ""
    assert status["target_only"] is True
    assert c.recording_path().exists()
    assert "embedding" not in str(status)


def test_target_requires_registration_and_valid_threshold(tmp_path):
    c, model = controller(tmp_path)
    c.load_model(model.id); c.wait(5)
    with pytest.raises(RuntimeError, match="注册"): c.start(model.id, target_only=True)
    prepare(c, model)
    for value in (0, 1, float("nan")):
        with pytest.raises(ValueError): c.start(model.id, target_only=True, speaker_threshold=value)
    c._models[model.id] = replace(model, recognition_types=("normal",))
    with pytest.raises(ValueError, match="SenseVoice"): c.start(model.id, target_only=True)
    c._models[model.id] = replace(model, modes=("offline",))
    with pytest.raises(ValueError, match="流式"): c.start(model.id, "offline", target_only=True)


@pytest.mark.parametrize("source", ["microphone", "system"])
def test_enrollment_selected_source_and_max_duration(tmp_path, source):
    def stream(stop):
        for _ in range(400): yield 0, np.full(1600, 0.1)
    def forbidden(stop):
        pytest.fail("unselected recording device opened")
    c, model = controller(tmp_path,
        stream_factory=stream if source == "microphone" else forbidden,
        system_stream_factory=stream if source == "system" else forbidden)
    c.load_target(); c.wait(5)
    c.enroll("目标", source=source); c.wait(5)
    assert c.snapshot()["source"] == source
    assert c.snapshot()["speaker_profile"]["speech_seconds"] == 30
    assert c._stop_event.is_set()


@pytest.mark.parametrize("previous_mode,previous_target", [("streaming", False), ("offline", False), ("streaming", True)])
@pytest.mark.parametrize("source", ["microphone", "system"])
def test_stopping_enrollment_never_reports_transcription_state(tmp_path, previous_mode, previous_target, source):
    from threading import Event
    captured, release = Event(), Event()

    def stream(stop):
        yield 0, np.full(48000, 0.1)
        captured.set()
        assert stop.wait(5)
        assert release.wait(5)

    c, _ = controller(tmp_path, stream_factory=stream, system_stream_factory=stream)
    c.load_target(); c.wait(5)
    # These fields belong to an earlier transcription, not the registration.
    c._mode, c._target_only = previous_mode, previous_target
    c.enroll("目标", source=source)
    try:
        assert captured.wait(5)
        assert c.snapshot()["state"] == "enroll_recording"
        assert c.snapshot()["source"] == source
        c.stop()
        status = c.snapshot()
        # The UI synchronizes model/type/source on transcription states,
        # including "stopping". Enrollment must never take that branch.
        assert status["state"] == "enrolling"
        assert "注册" in status["message"]
        assert status["mode"] == previous_mode
        assert status["target_only"] is previous_target
        c.stop()  # Repeated stop must not turn this into a transcription.
        assert c.snapshot()["state"] == "enrolling"
    finally:
        release.set(); c.stop(); c.wait(5)
    assert c.snapshot()["state"] == "idle"
    assert c.snapshot()["speaker_profile"]["name"] == "目标"


def test_enrollment_system_device_and_failure_preserve_profile(tmp_path):
    class Stream:
        device_name = "Test headphones (loopback)"
        closed = False
        def __iter__(self):
            yield 0, np.full(48000, 0.1)
            raise RuntimeError("output device disconnected")
        def close(self): self.closed = True
    stream = Stream()
    c, model = controller(tmp_path,
        stream_factory=lambda _: pytest.fail("microphone opened"),
        system_stream_factory=lambda _: stream)
    prepare(c, model)
    original = c._profile_path.read_bytes()
    c.enroll("新目标", source="system"); c.wait(5)
    status = c.snapshot()
    assert status["state"] == "error"
    assert "disconnected" in status["error"]
    assert status["capture_device"] == stream.device_name
    assert status["speaker_profile"]["name"] == "目标"
    assert c._profile_path.read_bytes() == original
    assert stream.closed


@pytest.mark.parametrize("source", ["wav", "invalid", "", None])
def test_invalid_enrollment_source_does_not_open_device(tmp_path, source):
    c, _ = controller(tmp_path,
        stream_factory=lambda _: pytest.fail("microphone opened"),
        system_stream_factory=lambda _: pytest.fail("system capture opened"))
    c.load_target(); c.wait(5)
    with pytest.raises(ValueError): c.enroll("目标", source=source)
    assert c.snapshot()["state"] == "idle"
    assert c.snapshot()["speaker_profile"] is None


def test_enrollment_does_not_show_previous_session_playback(tmp_path):
    c, model = controller(tmp_path)
    c._recording_path = tmp_path / "old.wav"
    c._committed = ["old text"]
    c.load_target(); c.wait(5)
    c.enroll("target", np.full(48000, 0.1)); c.wait(5)
    assert c.snapshot()["recording_url"] is None
    assert c.snapshot()["committed_text"] == ""


@pytest.mark.parametrize("length", [0, 2, 31])
def test_reject_short_long_enrollment(tmp_path, length):
    c, model = controller(tmp_path)
    with pytest.raises(ValueError): c.enroll("x", np.ones(16000 * length))


def test_real_enrollment_validation_without_models():
    m = TargetModels.__new__(TargetModels)
    m.vad, m.encoder = Vad(), Encoder()
    for audio in (np.ones(48000), np.full(48000, np.nan), np.zeros(48000)):
        with pytest.raises(ValueError): m.enroll(audio, "test")


def test_busy_registration_blocks_model_switch_delete_and_start(tmp_path):
    from threading import Event
    gate = Event()
    c, model = controller(tmp_path)
    prepare(c, model)
    original = c._target_models.enroll
    def waiting(*args):
        gate.wait(5)
        return original(*args)
    c._target_models.enroll = waiting
    c.enroll("new", np.full(48000, 0.1))
    try:
        for operation in (c.forget_speaker, c.load_target, lambda: c.load_model(model.id), lambda: c.start(model.id)):
            with pytest.raises(RuntimeError): operation()
    finally:
        gate.set(); c.wait(5)


def test_target_http_endpoints_and_cross_origin(tmp_path):
    import json
    from io import BytesIO
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    import soundfile as sf
    from src.web.dictation_server import make_handler
    opened = []
    def stream(source):
        def factory(stop):
            opened.append(source)
            return iter([(0, np.full(48000, 0.1))])
        return factory
    c, model = controller(tmp_path, stream_factory=stream("microphone"), system_stream_factory=stream("system"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(c, tmp_path))
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    def post(path, data=b"{}", **headers):
        with urlopen(Request(base + path, data=data, headers=headers), timeout=5) as response:
            return json.load(response)
    try:
        with pytest.raises(HTTPError) as denied:
            post("/api/target/load", Origin="http://other.example")
        assert denied.value.code == 403
        post("/api/target/load"); c.wait(5)
        wav = BytesIO(); sf.write(wav, np.full(48000, 0.1), 16000, format="WAV")
        post("/api/target/enroll-file?name=demo", wav.getvalue()); c.wait(5)
        assert c.snapshot()["speaker_profile"]["name"] == "demo"
        assert c.snapshot()["source"] == "wav"
        assert opened == []
        for source in ("system", "microphone"):
            post("/api/target/enroll-recording", json.dumps({"name": "demo", "source": source}).encode()); c.wait(5)
            assert c.snapshot()["source"] == source
            assert c.snapshot()["speaker_profile"]["name"] == "demo"
        assert opened == ["system", "microphone"]
        for source in ("wav", "invalid"):
            with pytest.raises(HTTPError) as invalid:
                post("/api/target/enroll-recording", json.dumps({"name": "demo", "source": source}).encode())
            assert invalid.value.code == 400
        assert opened == ["system", "microphone"]
        post("/api/target/enroll-microphone", b'{"name":"legacy"}'); c.wait(5)
        assert opened[-1] == "microphone"
        assert c.snapshot()["speaker_profile"]["name"] == "legacy"
        c.load_model(model.id); c.wait(5)
        post(f"/api/transcribe-file?model_id={model.id}&mode=streaming&filename=test.wav&target_only=true", wav.getvalue()); c.wait(5)
        assert c.snapshot()["target_only"]
        assert c.snapshot()["committed_text"]
        post("/api/target/forget")
        assert c.snapshot()["speaker_profile"] is None
    finally:
        server.shutdown(); server.server_close(); thread.join(5)
