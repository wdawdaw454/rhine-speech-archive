from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.web.dictation_server import (
    DictationController,
    DictationModel,
    LocalOfflineAsr,
    SenseVoiceSmallAsr,
    decode_wav,
    default_models,
    make_handler,
)


def _load_and_start(controller, model_id, mode="streaming"):
    controller.load_model(model_id, mode)
    controller.wait(5)
    assert controller.snapshot()["loaded_model_id"] == model_id
    controller.start(model_id, mode)


class FakeBackend:
    def reset(self) -> None:
        return None

    def accept(self, _chunk: np.ndarray, *, is_final: bool) -> str:
        return "十二号" if is_final else "十二"


def test_rhine_frontend_is_same_origin_and_confined(tmp_path):
    from http.server import ThreadingHTTPServer
    from http.client import HTTPConnection
    from threading import Thread

    legacy = tmp_path / 'legacy'
    rhine = tmp_path / 'rhine'
    legacy.mkdir()
    rhine.mkdir()
    (legacy / 'index.html').write_text('legacy', encoding='utf-8')
    (rhine / 'index.html').write_text('speech', encoding='utf-8')
    (rhine / 'app.js').write_text('export {};', encoding='utf-8')
    (tmp_path / 'private.txt').write_text('private', encoding='utf-8')
    controller = DictationController(project_root=tmp_path, models=[])
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(controller, legacy, rhine))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection(*server.server_address, timeout=5)
    try:
        for path, expected in [('/', b'legacy'), ('/rhine/', b'speech'), ('/rhine/app.js', b'export {};')]:
            connection.request('GET', path)
            response = connection.getresponse()
            assert response.status == 200
            assert response.read() == expected
        connection.request('GET', '/rhine')
        response = connection.getresponse()
        assert response.status == 308
        assert response.getheader('Location') == '/rhine/'
        response.read()
        for path in ['/rhine/%2e%2e/private.txt', '/rhine/..%5cprivate.txt', '/rhine/missing']:
            connection.request('GET', path)
            response = connection.getresponse()
            assert response.status == 404
            response.read()
        connection.request('POST', '/api/clear', '{}', {'Origin': 'https://untrusted.example'})
        response = connection.getresponse()
        assert response.status == 403
        response.read()
        origin = f'http://127.0.0.1:{server.server_port}'
        connection.request('POST', '/api/clear', '{}', {'Origin': origin})
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        connection.request('GET', '/api/model-manager')
        response = connection.getresponse()
        assert response.status == 200
        registry = json.load(response)
        assert registry["operation"] is None
        assert registry["models"]
        connection.request('POST', '/api/model-manager/install', json.dumps({'id': 'unknown'}),
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        assert response.status == 400
        assert '未知模型' in json.load(response)['error']
        response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        controller.shutdown()
        thread.join(3)


class FakeStream:
    statuses: list[str] = []

    def __iter__(self):
        for _ in range(3):
            yield 0.0, np.full(1600, 0.08, dtype=np.float32)
        for _ in range(6):
            yield 0.0, np.zeros(1600, dtype=np.float32)


def _model(tmp_path: Path) -> DictationModel:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pt").write_bytes(b"fake")
    return DictationModel(
        id="fake-model",
        name="Fake Model",
        model_dir=model_dir,
        description="test",
    )


def test_browser_controller_runs_session_and_writes_summary(tmp_path):
    model = _model(tmp_path)
    controller = DictationController(
        project_root=tmp_path,
        models=[model],
        backend_factory=lambda _model: FakeBackend(),
        stream_factory=lambda _stop: FakeStream(),
    )

    _load_and_start(controller, model.id)
    controller.wait(5.0)
    status = controller.snapshot()

    assert status["state"] == "idle"
    assert status["itn_enabled"] is True
    assert status["committed_text"] == "十二号"
    assert status["load_timings"]["cache_hit"] is False
    assert status["load_timings"]["backend_prepare_seconds"] >= 0
    output_dir = Path(status["output_dir"])
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["itn_enabled"] is True
    assert summary["load_timings"] == status["load_timings"]
    assert (output_dir / "transcript.txt").read_text(encoding="utf-8") == "十二号\n"


def test_browser_controller_reports_backend_cache_hit(tmp_path):
    model = _model(tmp_path)
    controller = DictationController(
        project_root=tmp_path,
        models=[model],
        backend_factory=lambda _model: FakeBackend(),
        stream_factory=lambda _stop: FakeStream(),
    )

    _load_and_start(controller, model.id)
    controller.wait(5.0)
    _load_and_start(controller, model.id)
    controller.wait(5.0)

    assert controller.snapshot()["load_timings"]["cache_hit"] is True


def test_loading_another_engine_preserves_transcript_identity(tmp_path):
    from dataclasses import replace
    model = _model(tmp_path)
    other = replace(model, id='other-engine')
    controller = DictationController(project_root=tmp_path, models=[model, other],
        backend_factory=lambda _: FakeBackend(), stream_factory=lambda _: FakeStream())
    try:
        _load_and_start(controller, model.id)
        controller.wait(5)
        before = controller.snapshot()
        controller.load_model(other.id)
        controller.wait(5)
        after = controller.snapshot()
        assert after['loaded_model_id'] == other.id
        assert after['run_id'] == before['run_id']
        assert after['run_context'] == before['run_context']
        assert after['run_context']['model_id'] == model.id
    finally:
        controller.shutdown()


def test_browser_controller_rejects_unknown_or_missing_model(tmp_path):
    model = _model(tmp_path)
    controller = DictationController(project_root=tmp_path, models=[model])

    try:
        controller.start("unknown")
    except ValueError as exc:
        assert "未知模型" in str(exc)
    else:
        raise AssertionError("unknown model should be rejected")

    (model.model_dir / "model.pt").unlink()
    try:
        controller.start(model.id)
    except FileNotFoundError as exc:
        assert "模型文件不存在" in str(exc)
    else:
        raise AssertionError("missing model should be rejected")


def test_default_registry_contains_local_streaming_and_offline_models(tmp_path):
    models = default_models(tmp_path)

    assert [model.id for model in models] == [
        "sensevoice-small", "fun-asr-nano", "qwen3-asr", "moss-transcribe-diarize",
    ]
    assert models[0].backend == "funasr-offline"
    assert models[0].modes == ("streaming", "offline")
    assert [m.id for m in models if "offline" in m.modes and "normal" in m.recognition_types] == [
        "sensevoice-small", "fun-asr-nano", "qwen3-asr",
    ]
    assert models[0].device == "cuda:0"
    assert models[0].required_file == "model.pt"
    assert models[0].decode_interval == 0.80
    assert models[0].recognition_types == ("normal", "target")
    assert models[0].public_dict()["target_name"] == "FSMN-VAD + CAM++ + SenseVoice Small"
    assert models[1].modes == ("streaming", "offline")
    assert models[1].decode_interval == 1.2
    assert models[1].recognition_types == ("normal",)
    assert all(m.modes == ("offline",) for m in (models[2], models[3]))
    assert all(tmp_path.parent / "models" in m.model_dir.parents for m in models[:3])
    assert models[2].model_key == "Qwen/Qwen3-ASR-1.7B"
    assert not any("paraformer" in model.id.lower() for model in models)
    assert models[-1].recognition_types == ("meeting",)
    assert models[-1].modes == ("offline",)
    assert models[-1].backend == "moss-meeting"
    assert all("meeting" not in model.recognition_types for model in models if "streaming" in model.modes)


def test_sensevoice_small_checkpoint_supports_prefix_and_offline_paths():
    asr = SenseVoiceSmallAsr.__new__(SenseVoiceSmallAsr)
    asr.sample_rate = 16_000
    asr.decode_interval = 0.50
    calls: list[int] = []
    asr.transcribe = lambda audio: calls.append(len(audio)) or f"{len(audio)} samples"  # type: ignore[method-assign]
    asr.reset()

    chunk = np.zeros(4_000, dtype=np.float32)
    assert asr.accept(chunk, is_final=False) == ""
    assert asr.accept(chunk, is_final=False) == "8000 samples"
    assert asr.accept(chunk[:1], is_final=True) == "8001 samples"
    assert calls == [8_000, 8_001]


@pytest.mark.parametrize("options", [{"target_only": True}, {"recognition_type": "target"}])
def test_normal_only_model_rejects_target_requests_but_keeps_transcription(tmp_path, options):
    from dataclasses import replace
    original = default_models(tmp_path)[0]
    marker = _model(tmp_path)
    model = replace(original, model_dir=marker.model_dir, required_file=marker.required_file,
                    recognition_types=("normal",))
    controller = DictationController(project_root=tmp_path, models=[model],
        backend_factory=lambda _: FakeBackend(), stream_factory=lambda _: FakeStream())
    controller.load_model(model.id); controller.wait(60)
    with pytest.raises(ValueError, match="SenseVoice"):
        controller.start(model.id, **options)
    assert controller.snapshot()["state"] == "idle"
    controller.start(model.id, recognition_type="normal"); controller.wait(60)
    assert controller.snapshot()["state"] == "idle"
    assert controller.snapshot()["committed_text"]


@pytest.mark.parametrize("action", ["load", "microphone", "system", "wav"])
@pytest.mark.parametrize("mode", ["streaming", "offline"])
def test_removed_paraformer_requests_are_rejected_before_loading_or_capture(tmp_path, action, mode):
    calls = []
    controller = DictationController(project_root=tmp_path, models=default_models(tmp_path),
        backend_factory=lambda _: calls.append("backend"),
        stream_factory=lambda _: calls.append("microphone"),
        system_stream_factory=lambda _: calls.append("system"))
    before = controller.snapshot()
    with pytest.raises(ValueError, match="未知模型"):
        if action == "load":
            controller.load_model("paraformer-zh-streaming", mode)
        else:
            controller.start("paraformer-zh-streaming", mode, source=action,
                file_audio=np.zeros(1600, dtype=np.float32) if action == "wav" else None)
    assert calls == []
    after = controller.snapshot()
    assert after["state"] == before["state"] == "idle"
    assert after["run_id"] == before["run_id"]
    assert after["loaded_model_id"] == before["loaded_model_id"]


def test_offline_records_before_transcribing_and_saves_audio(tmp_path):
    from dataclasses import replace

    model = replace(_model(tmp_path), modes=("offline",))
    captured = []
    calls = []

    class OfflineBackend:
        def reset(self):
            pass

        def transcribe(self, audio):
            assert captured == [True]
            calls.append(audio.copy())
            return "录音转写结果"

    def stream(_stop):
        yield 0, np.full(1600, 0.1, dtype=np.float32)
        assert not calls
        captured.append(True)

    controller = DictationController(project_root=tmp_path, models=[model],
        backend_factory=lambda _: OfflineBackend(), stream_factory=stream)
    _load_and_start(controller, model.id, "offline")
    controller.wait(5)
    status = controller.snapshot()
    assert status["state"] == "idle", status
    assert status["committed_text"] == "录音转写结果"
    assert status["recording_seconds"] == 0.1
    assert status["recording_url"]
    assert controller.recording_path().is_file()
    assert len(calls) == 1


def test_incompatible_mode_is_rejected(tmp_path):
    import pytest
    model = _model(tmp_path)
    controller = DictationController(project_root=tmp_path, models=[model])
    with pytest.raises(ValueError, match="模式"):
        controller.start(model.id, "offline")


def test_loading_alone_does_not_open_microphone(tmp_path):
    from threading import Event
    entered, release = Event(), Event()
    def factory(_model):
        entered.set()
        release.wait(3)
        return FakeBackend()
    def stream(_stop):
        raise AssertionError("loading must never open the microphone")
    model = _model(tmp_path)
    controller = DictationController(project_root=tmp_path, models=[model],
        backend_factory=factory, stream_factory=stream)
    controller.load_model(model.id)
    assert entered.wait(3)
    import pytest
    with pytest.raises(RuntimeError, match="任务"):
        controller.start(model.id)
    release.set()
    controller.wait(5)
    assert controller.snapshot()["state"] == "idle"
    assert controller.snapshot()["loaded_model_id"] == model.id
    assert controller.snapshot()["run_id"] is None
    assert controller.snapshot()["recording_seconds"] == 0


def test_recording_limit_and_silent_offline_session(tmp_path):
    from dataclasses import replace
    model = replace(_model(tmp_path), modes=("offline",))
    def stream(stop):
        while not stop.is_set():
            yield 0, np.zeros(1600, dtype=np.float32)
    controller = DictationController(project_root=tmp_path, models=[model],
        backend_factory=lambda _: FakeBackend(), stream_factory=stream)
    controller.max_recording_seconds = 0.2
    _load_and_start(controller, model.id, "offline")
    controller.wait(5)
    assert controller.snapshot()["state"] == "idle"
    assert controller.snapshot()["recording_seconds"] == 0.2
    assert controller.snapshot()["committed_text"] == ""


def test_model_availability_uses_backend_specific_marker(tmp_path):
    model_dir = tmp_path / "offline-model"
    model_dir.mkdir()
    model = DictationModel(
        id="offline-model",
        name="Offline Model",
        model_dir=model_dir,
        description="test",
        backend="funasr-offline",
        required_file="config.json",
    )

    assert model.public_dict()["available"] is False
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    assert model.public_dict()["available"] is True


def test_model_availability_can_reject_partial_checkpoint(tmp_path):
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    checkpoint = model_dir / "streaming.pt"
    checkpoint.write_bytes(b"partial")
    model = DictationModel(
        id="checkpoint",
        name="Checkpoint",
        model_dir=model_dir,
        description="test",
        required_file=checkpoint.name,
        required_bytes=8,
    )

    assert model.public_dict()["available"] is False
    checkpoint.write_bytes(b"complete")
    assert model.public_dict()["available"] is True


def test_controller_keeps_only_one_model_backend_resident(tmp_path):
    first = _model(tmp_path)
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    (second_dir / "model.pt").write_bytes(b"fake")
    second = DictationModel("second", "Second", second_dir, "test")
    built: list[str] = []
    controller = DictationController(
        project_root=tmp_path,
        models=[first, second],
        backend_factory=lambda model: built.append(model.id) or FakeBackend(),
        stream_factory=lambda _stop: FakeStream(),
    )

    _load_and_start(controller, first.id)
    controller.wait(5.0)
    _load_and_start(controller, second.id)
    controller.wait(5.0)

    assert built == [first.id, second.id]
    assert list(controller._backend_cache) == [second.id]
    with pytest.raises(RuntimeError, match="先加载"):
        controller.start(first.id)


def test_nano_receives_tensor_and_automatic_language():
    import torch
    from types import SimpleNamespace

    calls = []
    backend = LocalOfflineAsr.__new__(LocalOfflineAsr)
    backend.model_id = "fun-asr-nano"
    backend.model = SimpleNamespace(generate=lambda **kwargs: calls.append(kwargs) or [{"text": "测试"}])
    assert backend.transcribe(np.zeros(1600, dtype=np.float32)) == "测试"
    assert isinstance(calls[0]["input"], torch.Tensor)
    assert calls[0]["language"] is None
    assert calls[0]["llm_dtype"] == "fp32"


def test_start_requires_explicit_model_load(tmp_path):
    model = _model(tmp_path)
    controller = DictationController(project_root=tmp_path, models=[model])
    with pytest.raises(RuntimeError, match="先加载"):
        controller.start(model.id)
    assert controller.snapshot()["run_id"] is None


def _wav_bytes(audio=None, rate=32000, format="WAV"):
    from io import BytesIO
    import soundfile as sf
    output = BytesIO()
    if audio is None:
        audio = np.full((3200, 2), 0.08, dtype=np.float32)
    sf.write(output, audio, rate, format=format)
    return output.getvalue()


def test_wav_resamples_and_downmixes():
    audio = decode_wav(_wav_bytes())
    assert audio.shape == (1600,)
    assert audio.dtype == np.float32
    assert np.isclose(audio[100], 0.08, atol=0.001)


@pytest.mark.parametrize("payload", [b"", b"not wav", _wav_bytes(format="FLAC"),
    _wav_bytes(audio=np.empty(0)), _wav_bytes(rate=4000)])
def test_wav_rejects_invalid_input(payload):
    with pytest.raises(ValueError):
        decode_wav(payload)


def test_wav_rejects_nonfinite_float_samples():
    from io import BytesIO
    import soundfile as sf
    output = BytesIO()
    sf.write(output, np.full(100, np.nan), 16000, format="WAV", subtype="FLOAT")
    with pytest.raises(ValueError, match="无效"):
        decode_wav(output.getvalue())


def test_wav_duration_limit():
    with pytest.raises(ValueError, match="600"):
        decode_wav(_wav_bytes(np.zeros(8000 * 601, dtype=np.float32), rate=8000))


def test_load_failure_can_be_retried(tmp_path):
    model = _model(tmp_path)
    attempts = []
    def factory(_):
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("load failed")
        return FakeBackend()
    controller = DictationController(project_root=tmp_path, models=[model], backend_factory=factory)
    controller.load_model(model.id)
    controller.wait(5)
    assert controller.snapshot()["state"] == "error"
    assert controller.snapshot()["loaded_model_id"] is None
    controller.load_model(model.id)
    controller.wait(5)
    assert controller.snapshot()["loaded_model_id"] == model.id
    assert controller.snapshot()["error"] is None


@pytest.mark.parametrize("mode", ["streaming", "offline"])
def test_wav_transcription_never_opens_microphone(tmp_path, mode):
    from dataclasses import replace
    model = replace(_model(tmp_path), modes=(mode,))
    class Backend(FakeBackend):
        def transcribe(self, audio):
            assert len(audio) == 4800
            return "文件转写"
    def forbidden_stream(_stop):
        raise AssertionError("file input must not open microphone")
    controller = DictationController(project_root=tmp_path, models=[model],
        backend_factory=lambda _: Backend(), stream_factory=forbidden_stream)
    controller.load_model(model.id, mode)
    controller.wait(5)
    payload = _wav_bytes(np.full((9600, 2), 0.08, dtype=np.float32))
    controller.start(model.id, mode, file_audio=decode_wav(payload), filename="测试.wav")
    controller.wait(5)
    status = controller.snapshot()
    assert status["state"] == "idle", status
    assert status["source"] == "wav"
    assert status["filename"] == "测试.wav"
    assert status["recording_seconds"] == 0.3
    assert controller.recording_path().is_file()
    if mode == "offline":
        assert status["committed_text"] == "文件转写"
    else:
        assert status["committed_text"] == "十二号"


def test_wav_is_not_truncated_at_microphone_limit(tmp_path):
    from dataclasses import replace
    model = replace(_model(tmp_path), modes=("offline",))
    lengths = []
    class Backend(FakeBackend):
        def transcribe(self, audio):
            lengths.append(len(audio))
            return "文件分段"
    controller = DictationController(project_root=tmp_path, models=[model],
        backend_factory=lambda _: Backend())
    controller.load_model(model.id, "offline")
    controller.wait(5)
    audio = np.full(16000 * 121, 0.08, dtype=np.float32)
    controller.start(model.id, "offline", file_audio=audio, filename="long.wav")
    controller.wait(5)
    assert controller.snapshot()["state"] == "idle"
    assert controller.snapshot()["recording_seconds"] == 121
    assert lengths == [16000 * 30] * 4 + [16000]


def test_http_recording_and_cross_origin_protection(tmp_path):
    from http.server import ThreadingHTTPServer
    from http.client import HTTPConnection
    from threading import Thread

    model = _model(tmp_path)
    controller = DictationController(project_root=tmp_path, models=[model],
        backend_factory=lambda _: FakeBackend(), stream_factory=lambda _: FakeStream())
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(controller, tmp_path))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    connection = HTTPConnection(*server.server_address, timeout=5)
    try:
        connection.request("POST", "/api/start", json.dumps({"model_id": model.id}),
                           {"Origin": "https://untrusted.example", "Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 403
        response.read()
        assert controller.snapshot()["state"] == "idle"
        connection.request("POST", "/api/load-model", json.dumps({"model_id": model.id}),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 202
        response.read()
        controller.wait(5)
        assert controller.snapshot()["run_id"] is None
        connection.request("POST", "/api/start", json.dumps({"model_id": model.id}),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 202
        response.read()
        controller.wait(5)
        connection.request("GET", controller.snapshot()["recording_url"])
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "audio/wav"
        assert response.read() == controller.recording_path().read_bytes()
        connection.request("POST", f"/api/transcribe-file?model_id={model.id}&mode=streaming&filename=test.wav",
                           _wav_bytes(), {"Content-Type": "audio/wav"})
        response = connection.getresponse()
        assert response.status == 202
        response.read()
        controller.wait(5)
        assert controller.snapshot()["source"] == "wav"
        assert controller.snapshot()["error"] is None
        connection.request("POST", f"/api/transcribe-file?model_id={model.id}&filename=broken.wav",
                           b"invalid", {"Content-Type": "audio/wav"})
        response = connection.getresponse()
        assert response.status == 400
        response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        worker.join(5)
