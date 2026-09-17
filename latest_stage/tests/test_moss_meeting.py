"""Meeting routing, whole-file identity, exports, cancellation and worker lifecycle."""
import json
from pathlib import Path
import sys
from threading import Event

import numpy as np
import pytest
import soundfile as sf

from src.models.moss_meeting import (MeetingCancelled, MossMeetingAsr,
    segment_text, segments_srt, validate_segments)
from src.web.dictation_server import DictationController, DictationModel, make_handler


SEGMENTS = [
    {"start": 0.1, "end": 2.1, "speaker": "S01", "text": "我们开始会议。"},
    {"start": 1.3, "end": 3.0, "speaker": "S02", "text": "我补充一下。"},
    {"start": 3.0, "end": 4.0, "speaker": "S01", "text": "好的。"},
]


class Backend:
    closed = False
    runtime_info = {"whole_recording": True, "device": "fake"}

    def __init__(self):
        self.calls = []
        self.result = dict(segments=SEGMENTS, raw_text="[0.1][S01]我们开始会议。[2.1]",
                           backend_seconds=.1, generated_tokens=30, truncated=False)

    def reset(self):
        pass

    def transcribe_file(self, path, *, progress, cancel_event):
        self.calls.append(sf.read(path)[0])
        progress({"stage": "decoding", "generated_tokens": 10})
        return self.result

    def close(self):
        self.closed = True


def controller(tmp_path, backend=None, **kwargs):
    marker = tmp_path / "model.onnx"
    marker.write_bytes(b"test")
    model = DictationModel("moss-test", "MOSS", tmp_path, "test", modes=("offline",),
                          recognition_types=("meeting",))
    backend = backend or Backend()
    c = DictationController(project_root=tmp_path, models=[model],
        backend_factory=lambda _: backend,
        target_factory=lambda: pytest.fail("meeting must not load CAM++"), **kwargs)
    c.load_model(model.id, "offline"); c.wait(5)
    assert c.snapshot()["state"] == "idle"
    return c, backend, model


@pytest.mark.parametrize("source", ["wav", "microphone", "system"])
def test_whole_recording_preserves_overlap_and_speaker_ids(tmp_path, source):
    audio = np.full(31 * 16000, .1, np.float32)
    opened = []
    def factory(kind):
        def stream(stop):
            opened.append(kind)
            yield 0, audio
        return stream
    c, backend, model = controller(tmp_path,
        stream_factory=factory("microphone"), system_stream_factory=factory("system"))
    options = {"file_audio": audio, "filename": "meeting.wav"} if source == "wav" else {"source": source}
    c.start(model.id, "offline", recognition_type="meeting", **options); c.wait(5)
    s = c.snapshot()
    assert s["state"] == "idle", s
    assert s["recognition_type"] == "meeting" and not s["target_only"]
    assert not s["target_ready"] and s["speaker_profile"] is None
    assert [len(call) for call in backend.calls] == [len(audio)]  # NOT 30s chunks.
    assert opened == ([] if source == "wav" else [source])
    assert s["meeting_segments"] == SEGMENTS
    assert s["meeting_segments"][1]["start"] < s["meeting_segments"][0]["end"]
    assert "[S02] 我补充一下。" in s["committed_text"]
    output = Path(s["output_dir"])
    saved = json.loads((output / "meeting_segments.json").read_text(encoding="utf-8"))
    assert saved["segments"] == SEGMENTS
    assert "00:00:01,300 --> 00:00:03,000" in (output / "meeting.srt").read_text(encoding="utf-8")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["parameters"]["whole_recording"] is True
    assert summary["stats"]["speakers"] == 2
    assert c.meeting_export_path("json", s["run_id"]).is_file()
    assert c.meeting_export_path("json", "old-run") is None
    assert c.meeting_export_path("../../anything", s["run_id"]) is None
    c.clear()
    assert c.snapshot()["meeting_segments"] == [] and c.snapshot()["meeting_exports"] == {}
    assert (output / "meeting_segments.json").is_file()  # UI clear is not deletion.
    c.shutdown()
    assert backend.closed


@pytest.mark.parametrize("kind", ["normal", "target", "invalid"])
def test_meeting_rejects_incompatible_recognition_type(tmp_path, kind):
    c, _, model = controller(tmp_path)
    with pytest.raises(ValueError):
        c.start(model.id, "offline", recognition_type=kind, file_audio=np.ones(16000))
    assert c.snapshot()["state"] == "idle"


def test_legacy_start_infers_meeting_type(tmp_path):
    c, _, model = controller(tmp_path)
    c.start(model.id, "offline", file_audio=np.full(64000, .1)); c.wait(5)
    assert c.snapshot()["recognition_type"] == "meeting"
    assert c.snapshot()["state"] == "idle"


@pytest.mark.parametrize("result,error", [
    ({"raw_text": "bad format", "segments": []}, "可解析"),
    ({"raw_text": "raw", "segments": [{"start": 0, "end": 999, "speaker": "S01", "text": "x"}]}, "时间戳"),
    ({"raw_text": "raw", "segments": SEGMENTS, "truncated": True}, "不完整"),
])
def test_bad_or_truncated_output_is_not_advertised_as_complete(tmp_path, result, error):
    backend = Backend(); backend.result = result
    c, _, model = controller(tmp_path, backend)
    c.start(model.id, "offline", file_audio=np.full(64000, .1)); c.wait(5)
    s = c.snapshot()
    assert s["state"] == "error" and error in s["error"]
    assert c.recording_path().is_file()
    assert (Path(s["output_dir"]) / "moss_raw.txt").read_text(encoding="utf-8") == result["raw_text"]
    if result.get("truncated"):
        assert s["meeting_segments"] == SEGMENTS


def test_silence_is_a_successful_empty_meeting(tmp_path):
    backend = Backend(); backend.result = {"raw_text": "", "segments": [], "truncated": False}
    c, _, model = controller(tmp_path, backend)
    c.start(model.id, "offline", file_audio=np.zeros(64000)); c.wait(5)
    assert c.snapshot()["state"] == "idle"
    assert not c.snapshot()["committed_text"]


def test_cancel_during_inference_preserves_audio_and_requires_reload(tmp_path):
    entered = Event()
    class Slow(Backend):
        def transcribe_file(self, path, *, progress, cancel_event):
            entered.set()
            assert cancel_event.wait(5)
            self.close()
            raise MeetingCancelled("已取消会议转写")
    c, backend, model = controller(tmp_path, Slow())
    c.start(model.id, "offline", file_audio=np.ones(64000) * .1)
    assert entered.wait(5)
    c.stop(); c.wait(5)
    s = c.snapshot()
    assert s["state"] == "idle" and s["error"] is None
    assert s["loaded_model_id"] is None and backend.closed
    assert c.recording_path().is_file()


def test_stop_recording_finalizes_meeting_instead_of_cancelling(tmp_path):
    captured = Event()
    def stream(stop):
        yield 0, np.full(64000, .1)
        captured.set()
        assert stop.wait(5)
    c, backend, model = controller(tmp_path, stream_factory=stream)
    c.start(model.id, "offline", recognition_type="meeting")
    assert captured.wait(5)
    c.stop(); c.wait(5)
    assert len(backend.calls) == 1
    assert not c._cancel_event.is_set()
    assert c.snapshot()["state"] == "idle"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_timestamps_are_rejected(value):
    with pytest.raises(ValueError):
        validate_segments([{**SEGMENTS[0], "start": value}], 4)


def test_exports_keep_literal_text_not_html_and_do_not_merge_overlap():
    segments = validate_segments([{**s, "text": "<script>alert(1)</script>"} for s in SEGMENTS], 4)
    assert len(segments) == 3
    assert "<script>" in segment_text(segments[0])
    assert "[S02]" in segments_srt(segments)


def test_http_meeting_exports_and_explicit_recognition_type(tmp_path):
    from http.client import HTTPConnection
    from http.server import ThreadingHTTPServer
    from io import BytesIO
    from threading import Thread
    c, _, model = controller(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(c, tmp_path))
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    connection = HTTPConnection(*server.server_address, timeout=5)
    try:
        wav = BytesIO(); sf.write(wav, np.full(64000, .1), 16000, format="WAV")
        connection.request("POST", f"/api/transcribe-file?model_id={model.id}&mode=offline&recognition_type=meeting&filename=meeting.wav",
                           wav.getvalue(), {"Content-Type": "audio/wav"})
        response = connection.getresponse(); assert response.status == 202; response.read()
        c.wait(5)
        for kind, url in c.snapshot()["meeting_exports"].items():
            connection.request("GET", url)
            response = connection.getresponse()
            assert response.status == 200
            assert "attachment" in response.getheader("Content-Disposition")
            assert response.read()
        connection.request("GET", "/api/meeting-export?format=json&run=old")
        response = connection.getresponse(); assert response.status == 404; response.read()
    finally:
        connection.close(); server.shutdown(); server.server_close(); thread.join(5); c.shutdown()


def test_worker_bridge_reuses_and_closes_a_local_process(tmp_path):
    scripts = tmp_path / "scripts"; scripts.mkdir()
    (scripts / "moss_worker.py").write_text(
        "import sys,json\nprint(json.dumps({'type':'ready'}),flush=True)\n"
        "for line in sys.stdin:\n print(json.dumps({'type':'result','segments':[]}),flush=True)\n", encoding="utf-8")
    bridge = MossMeetingAsr(tmp_path, tmp_path, python_executable=sys.executable, startup_timeout=10)
    try:
        process_id = bridge.process.pid
        assert bridge.transcribe_file(tmp_path / "test.wav")["segments"] == []
        assert bridge.transcribe_file(tmp_path / "test.wav")["segments"] == []
        assert bridge.process.pid == process_id
    finally:
        bridge.close()
    assert bridge.process.poll() is not None
    bridge.close()


def test_switching_model_closes_meeting_worker(tmp_path):
    c, backend, _ = controller(tmp_path)
    second = DictationModel("second", "Second", tmp_path, "test", modes=("offline",))
    c._models[second.id] = second
    c._backend_factory = lambda _: Backend()
    c.load_model(second.id, "offline"); c.wait(5)
    assert c.snapshot()["state"] == "idle"
    assert backend.closed and list(c._backend_cache) == [second.id]


def test_empty_meeting_capture_does_not_call_worker(tmp_path):
    c, backend, model = controller(tmp_path, stream_factory=lambda _: iter(()))
    c.start(model.id, "offline", recognition_type="meeting"); c.wait(5)
    assert c.snapshot()["state"] == "error"
    assert "未收到" in c.snapshot()["error"] and not backend.calls


def test_meeting_capture_overflow_is_not_silently_transcribed(tmp_path):
    class Stream:
        statuses = ["input overflow"]
        def __iter__(self):
            yield 0, np.ones(64000, np.float32)
    c, backend, model = controller(tmp_path, stream_factory=lambda _: Stream())
    c.start(model.id, "offline", recognition_type="meeting"); c.wait(5)
    assert c.snapshot()["state"] == "error" and not backend.calls
    assert "丢帧" in c.snapshot()["error"]


def test_failed_meeting_capture_preserves_received_audio(tmp_path):
    def stream(stop):
        yield 0, np.full(64000, .1)
        raise RuntimeError("device disconnected")
    c, backend, model = controller(tmp_path, stream_factory=stream)
    c.start(model.id, "offline", recognition_type="meeting"); c.wait(5)
    assert c.snapshot()["state"] == "error" and not backend.calls
    assert c.recording_path().is_file()
    assert sf.info(c.recording_path()).frames == 64000
