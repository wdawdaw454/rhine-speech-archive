"""Real CAM++/FSMN/ASR smoke, separate enrollment and held-out speakers.

Uses official model demo files and a temporary profile, never the user's voice.
"""
from pathlib import Path
from dataclasses import replace
import argparse
from http.server import ThreadingHTTPServer
import json
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.web.dictation_server import DictationController, default_models, decode_wav, make_handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["sensevoice-realtime"], default="sensevoice-realtime")
    parser.add_argument("--serve", action="store_true", help="Keep an isolated UI on a random local port for manual QA")
    args = parser.parse_args()
    examples = ROOT.parent / "models/cam-plus/examples"
    def audio(name): return decode_wav((examples / name).read_bytes())
    with tempfile.TemporaryDirectory(prefix="target-smoke-") as directory:
        models = [replace(m, device="cpu") for m in default_models(ROOT) if "target" in m.recognition_types]
        model = next(m for m in models if m.id == args.model)
        c = DictationController(project_root=ROOT, models=models,
            stream_factory=lambda _: (_ for _ in ()).throw(AssertionError("microphone opened")))
        c.load_target(); c.wait(120)
        assert c.snapshot()["target_ready"], c.snapshot()
        c.enroll("official-demo-speaker1", audio("speaker1_a_cn_16k.wav")); c.wait(60)
        assert c.snapshot()["speaker_profile"], c.snapshot()
        c.load_model(model.id); c.wait(120)
        assert c.snapshot()["loaded_model_id"], c.snapshot()
        samples = [audio("speaker1_b_cn_16k.wav"), audio("speaker2_a_cn_16k.wav")]
        cases = [("same-speaker-held-out", samples[0]), ("different-speaker", samples[1]),
                 ("alternating", np.concatenate([samples[0], np.zeros(16000), samples[1], np.zeros(16000), samples[0]]))]
        cases.extend([("silence", np.zeros(3 * 16000, np.float32)),
                      ("long-target", np.tile(samples[0], 6))])
        for label, wav in cases:
            start = time.perf_counter()
            c.start(model.id, file_audio=wav, filename=f"{label}.wav", target_only=True); c.wait(120)
            s = c.snapshot()
            assert s["state"] == "idle", s
            summary = json.loads((Path(s["output_dir"]) / "summary.json").read_text(encoding="utf-8"))
            events = [json.loads(line) for line in (Path(s["output_dir"]) / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            scores = [round(e["similarity"], 3) for e in events if e["type"] == "speaker_decision" and e["similarity"] is not None]
            print(json.dumps({"case":label,"seconds":round(time.perf_counter()-start,2),"scores":scores,"text":s["committed_text"],"stats":summary["stats"]}, ensure_ascii=False), flush=True)
            if label in {"different-speaker", "silence"}: assert not s["committed_text"]
            else: assert s["committed_text"]
        print("TARGET SMOKE PASSED", flush=True)
        if args.serve:
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(c, ROOT / "web/dictation"))
            print(f"ISOLATED_UI=http://127.0.0.1:{server.server_port}/", flush=True)
            print("Public test voice only; user profile unchanged. Close via UI shutdown button.", flush=True)
            try:
                server.serve_forever(poll_interval=.25)
            finally:
                c.stop(); c.wait()
                server.server_close()


if __name__ == "__main__": main()
