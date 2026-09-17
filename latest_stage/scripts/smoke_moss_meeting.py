"""Real MOSS HTTP smoke on public demo voices, never mic/enrollment.

Start a separate UI on port 8766 first. Leaves MOSS loaded for browser QA.
"""
import argparse
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, build_opener, ProxyHandler

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.web.dictation_server import decode_wav


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    opener = build_opener(ProxyHandler({}))
    report_dir = ROOT / "outputs/benchmarks/moss_meeting_smoke" / datetime.now().strftime("%Y%m%d-%H%M%S")
    report_dir.mkdir(parents=True)

    def request(path, body=None, content_type="application/json"):
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        with opener.open(Request(base + path, data=body, headers={"Content-Type": content_type}), timeout=15) as response:
            raw = response.read()
            return json.loads(raw) if "json" in response.headers["Content-Type"] else raw.decode("utf-8")

    def wait_idle(timeout=600):
        deadline = time.monotonic() + timeout
        last_print = 0
        while time.monotonic() < deadline:
            status = request("/api/status")
            if status["state"] in {"idle", "error"}:
                assert status["state"] == "idle", status
                return status
            if time.monotonic() - last_print > 10:
                print(status["message"], flush=True)
                last_print = time.monotonic()
            time.sleep(.3)
        raise TimeoutError("Inspect MOSS worker; smoke timed out")

    def load():
        request("/api/load-model", {"model_id": "moss-transcribe-diarize", "mode": "offline"})
        status = wait_idle()
        assert status["loaded_model_id"] == "moss-transcribe-diarize", status
        return status

    def submit(name, audio):
        wav = BytesIO()
        sf.write(wav, audio, 16000, format="WAV", subtype="PCM_16")
        (report_dir / (name + ".wav")).write_bytes(wav.getvalue())
        return request(f"/api/transcribe-file?model_id=moss-transcribe-diarize&mode=offline&recognition_type=meeting&filename={name}.wav", wav.getvalue(), "audio/wav")

    assert request("/api/status")["state"] in {"idle", "error"}, "Server is busy"
    status = load()
    report = {"load_timings": status["load_timings"], "cases": []}
    examples = Path.home() / ".cache/modelscope/models/iic--speech_campplus_sv_zh-cn_16k-common/snapshots/master/examples"
    a = decode_wav((examples / "speaker1_b_cn_16k.wav").read_bytes())
    b = decode_wav((examples / "speaker2_a_cn_16k.wav").read_bytes())
    gap = np.zeros(16000, np.float32)
    alternating = np.concatenate([a, gap, b, gap, a])
    overlap = np.zeros(max(len(a), len(b) + 16000), np.float32)
    overlap[:len(a)] += a * .55
    overlap[16000:16000 + len(b)] += b * .55
    cases = [("single", a), ("alternating", alternating), ("overlap", overlap),
             ("longer_than_30s", np.tile(alternating, 3)), ("silence", np.zeros(48000, np.float32))]
    print("REPORT " + str(report_dir), flush=True)
    for name, audio in cases:
        start = time.monotonic()
        submit(name, audio)
        status = wait_idle()
        segments = status["meeting_segments"]
        if name == "silence":
            assert not segments and not status["committed_text"], status
        else:
            assert segments and status["committed_text"], status
            exported = request(status["meeting_exports"]["json"])
            assert exported["segments"] == segments and not exported["truncated"]
            assert "-->" in request(status["meeting_exports"]["srt"])
        item = {"case": name, "source_seconds": len(audio) / 16000,
                "wall_seconds": round(time.monotonic() - start, 3), "status": status}
        report["cases"].append(item)
        (report_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in item.items() if key != "status"}, ensure_ascii=False), flush=True)
        print(status["committed_text"], flush=True)
    # Cancellation must release the worker but preserve the original recording.
    submit("cancel", np.tile(alternating, 6))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = request("/api/status")
        if status["meeting_metrics"].get("stage") in {"encoding", "decoding"}:
            break
        time.sleep(.1)
    assert status["state"] == "transcribing", status
    request("/api/stop", {})
    status = wait_idle(30)
    assert status["loaded_model_id"] is None and status["recording_url"], status
    assert Path(status["output_dir"], "recording.wav").is_file()
    report["cancellation"] = status
    load()
    (report_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS MOSS HTTP, silence, exports, cancellation and reload. Speaker accuracy is not asserted.", flush=True)


if __name__ == "__main__":
    main()
