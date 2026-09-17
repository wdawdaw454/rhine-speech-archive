"""Exercise browser backends against a local WAV without opening the microphone."""
from pathlib import Path
import sys
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.web.dictation_server import DictationController, default_models


def main():
    audio, sr = sf.read(ROOT / "Funasr_tests/data/en_audio1.wav", dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = resample_poly(audio, 16000, sr).astype(np.float32)

    def stream(_stop):
        for offset in range(0, len(audio), 1600):
            yield offset / 16000, audio[offset:offset + 1600]

    with tempfile.TemporaryDirectory(prefix="dictation-smoke-") as work:
        controller = DictationController(project_root=Path(work), models=default_models(ROOT),
                                         stream_factory=stream)
        for model in controller.model_list():
            if len(sys.argv) > 1 and model["id"] not in sys.argv[1:]:
                continue
            print("TEST", model["id"], "available=", model["available"], flush=True)
            controller.load_model(model["id"], model["modes"][0])
            controller.wait(240)
            assert controller.snapshot()["loaded_model_id"] == model["id"], controller.snapshot()
            controller.start(model["id"], model["modes"][0])
            controller.wait(240)
            status = controller.snapshot()
            print(status["state"], status["error"], status["committed_text"], flush=True)
            assert status["state"] == "idle", status
            assert status["committed_text"].strip(), status
            assert controller.recording_path().is_file()
    print("ALL LOCAL MODELS PASSED", flush=True)


if __name__ == "__main__":
    main()
