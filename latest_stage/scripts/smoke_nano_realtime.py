"""Exercise the real local HTTP/Nano/VAD stack with existing WAVs, never the mic.

Requires an idle running dictation server. Loads Nano and leaves it resident.
Run explicitly; this is not a fast unit test and replays audio in real time.
"""
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.web.dictation_server import decode_wav


def request(path, body=None, content_type="application/json"):
    if isinstance(body, dict):
        body = json.dumps(body).encode()
    with urlopen(Request("http://127.0.0.1:8765" + path, data=body,
                         headers={"Content-Type": content_type}), timeout=15) as response:
        return json.load(response)


def wait_idle(timeout=180):
    deadline = time.monotonic() + timeout
    observations = []
    while time.monotonic() < deadline:
        status = request("/api/status")
        observations.append({"wall": time.monotonic(), "state": status["state"],
                             "audio_seconds": status["recording_seconds"],
                             "partial": status["partial_text"], "committed": status["committed_text"]})
        if status["state"] in {"idle", "error"}:
            assert status["state"] == "idle", status
            return status, observations
        time.sleep(.15)
    raise TimeoutError("Nano smoke timed out; inspect the server, do not start another session")


def main():
    status = request("/api/status")
    assert status["state"] in {"idle", "error"}, "Server is busy; not interrupting it"
    request("/api/load-model", {"model_id": "fun-asr-nano", "mode": "streaming"})
    status, _ = wait_idle()
    assert status["loaded_model_id"] == "fun-asr-nano"
    report_dir = ROOT / "outputs/benchmarks/nano_realtime_smoke" / datetime.now().strftime("%Y%m%d-%H%M%S")
    report_dir.mkdir(parents=True)
    print("REPORT", report_dir, flush=True)
    sample = decode_wav((Path.home() / ".cache/modelscope/models/iic--speech_campplus_sv_zh-cn_16k-common/snapshots/master/examples/speaker1_b_cn_16k.wav").read_bytes())
    long_sample = decode_wav((ROOT / "Funasr_tests/data/zh_audio1.wav").read_bytes())
    silence = np.zeros(16000 * 2, np.float32)
    cases = [
        ("silence", np.zeros(16000 * 3, np.float32), "streaming", False),
        ("two_utterances", np.concatenate((sample, silence, sample, silence)), "streaming", False),
        ("long_speech", np.concatenate((long_sample, silence)), "streaming", False),
        ("stop_replay", np.tile(sample, 3), "streaming", True),
        ("offline_regression", sample, "offline", False),
    ]
    report = {"load_timings": status["load_timings"], "cases": []}
    for name, audio, mode, stop in cases:
        wav = BytesIO()
        sf.write(wav, audio, 16000, format="WAV", subtype="PCM_16")
        started = time.monotonic()
        request(f"/api/transcribe-file?model_id=fun-asr-nano&mode={mode}&filename={name}.wav", wav.getvalue(), "audio/wav")
        if stop:
            time.sleep(1.5)
            request("/api/stop", {})
        status, observations = wait_idle()
        output = Path(status["output_dir"])
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if name == "silence":
            assert not status["committed_text"] and summary["stats"]["decode_calls"] == 0
        else:
            assert status["committed_text"], status
        if name == "two_utterances":
            assert summary["stats"]["committed_segments"] >= 2, summary
        if mode == "streaming":
            assert summary["stats"]["max_segment_seconds"] <= 8.01
            assert not summary["microphone_statuses"]
            if not stop:
                assert time.monotonic() - started >= len(audio) / 16000 - .05
            else:
                assert status["recording_seconds"] < len(audio) / 16000
        item = {"name": name, "wall_seconds": time.monotonic() - started,
                "source_seconds": len(audio) / 16000, "status": status,
                "summary": summary, "observations": observations}
        report["cases"].append(item)
        (report_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"case": name, "text": status["committed_text"],
                          "stats": summary["stats"], "wall_seconds": item["wall_seconds"]}, ensure_ascii=False), flush=True)
    # Leave the UI in the requested realtime mode, with a cache hit (no mic).
    request("/api/load-model", {"model_id": "fun-asr-nano", "mode": "streaming"})
    wait_idle()
    print("PASS: all real Nano/VAD HTTP cases", flush=True)


if __name__ == "__main__":
    main()
