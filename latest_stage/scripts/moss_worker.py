"""Private JSON-lines worker. All model logs go to stderr, never the protocol."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    protocol = sys.stdout
    sys.stdout = sys.stderr

    def send(kind, **data):
        protocol.write(json.dumps({"type": kind, **data}, ensure_ascii=False) + "\n")
        protocol.flush()

    try:
        started = time.perf_counter()
        import numpy as np
        import soundfile as sf
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from moss_transcribe_diarize import parse_transcript
        from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription

        torch.set_num_threads(4)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, trust_remote_code=True, local_files_only=True,
            dtype=dtype, attn_implementation="sdpa").to(device).eval()
        processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True)
        send("ready", timings={"moss_load_seconds": round(time.perf_counter() - started, 3)},
             runtime={"device": str(device), "dtype": str(dtype), "attention": "sdpa",
                      "torch": torch.__version__, "whole_recording": True})
        for line in sys.stdin:
            request = json.loads(line)
            path = request["audio_path"]
            info = sf.info(path)
            duration = info.frames / info.samplerate
            if not 0 < duration <= 600 or info.samplerate != 16000 or info.channels != 1:
                raise ValueError("会议输入需为不超过 10 分钟的 16 kHz 单声道 WAV")
            audio, _ = sf.read(path, dtype="float32")
            if not np.isfinite(audio).all():
                raise ValueError("会议录音包含无效音频")
            # Skip digital silence only; do not use VAD to discard overlaps.
            if float(np.sqrt(np.mean(audio ** 2))) < .0001:
                send("result", raw_text="", segments=[], duration=duration,
                     generated_tokens=0, backend_seconds=0, truncated=False)
                continue
            token_limit = max(1024, math.ceil(duration * 14) + 256)
            started = time.perf_counter()
            last_progress = 0.0
            send("progress", stage="encoding", generated_tokens=0)

            def on_tokens(count):
                nonlocal last_progress
                now = time.monotonic()
                if now - last_progress >= .5:
                    send("progress", stage="decoding", generated_tokens=count)
                    last_progress = now

            result = generate_transcription(
                model, processor, build_transcription_messages(path),
                max_new_tokens=token_limit, do_sample=False, device=device, dtype=dtype,
                token_callback=on_tokens)
            segments = [asdict(segment) for segment in parse_transcript(result["text"])]
            send("result", raw_text=result["text"], segments=segments, duration=duration,
                 generated_tokens=result["generated_tokens"], token_limit=token_limit,
                 prompt_tokens=result["prompt_len"],
                 backend_seconds=round(time.perf_counter() - started, 3),
                 truncated=result["generated_tokens"] >= token_limit)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        send("error", error=f"MOSS: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
