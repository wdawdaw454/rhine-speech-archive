"""Single-request Nano/PyTorch latency benchmark; never modifies the web service.

Real-clock replay of existing WAVs, cumulative prefix decoding, coalescing late
updates (no infinite queue), and an explicitly ASSUMED 600 ms endpoint timeout.
First text means the first completed nonempty transcription, not an LLM token.
"""
from pathlib import Path
from datetime import datetime
from threading import Thread
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from urllib.request import urlopen

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.web.dictation_server import LocalOfflineAsr, decode_wav, default_models


class GpuMonitor:
    def __init__(self):
        self.samples = []
        self.phase = "initialization"
        self.process = subprocess.Popen([
            "nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu",
            "--format=csv,noheader,nounits", "-lms", "200"], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.thread = Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        for line in self.process.stdout:
            try:
                values = [float(x.strip()) for x in line.split(",")]
                self.samples.append(dict(phase=self.phase, wall=time.time(), memory_mib=values[0],
                    gpu_util=values[1], power_w=values[2], clock_mhz=values[3], temperature_c=values[4]))
            except (ValueError, IndexError):
                pass

    def close(self):
        self.process.terminate()
        self.process.wait(timeout=5)
        self.thread.join(5)


def service_idle():
    try:
        with urlopen("http://127.0.0.1:8765/api/status", timeout=2) as response:
            s = json.load(response)
    except OSError:
        return
    if s["state"] not in ("idle", "error"):
        raise RuntimeError("Web service became busy; benchmark stopped without changing its state")


def speech_trim(wav):
    # This is only an approximate speech-onset/end reference, NOT measured VAD.
    frame = 320
    levels = [np.sqrt(np.mean(wav[i:i+frame] ** 2)) for i in range(0, len(wav), frame)]
    threshold = max(0.001, max(levels) * 0.02)
    indices = [i for i, level in enumerate(levels) if level >= threshold]
    if not indices:
        raise ValueError("No energetic audio in test clip")
    a, b = indices[0] * frame, min(len(wav), (indices[-1] + 1) * frame)
    return wav[a:b], {"trim_start": a/16000, "trim_end": b/16000, "energy_threshold": threshold}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precisions", nargs="+", default=["default", "bf16"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--reuse-final", action="store_true",
                        help="Reuse a completed full-speech preview after the assumed endpoint fires")
    parser.add_argument("--initial-audio", type=float, nargs="+", default=[1.0, 2.0])
    args = parser.parse_args()
    torch.set_num_threads(4)
    run = ROOT / "outputs/benchmarks/nano_no_vllm" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    report = {"run_dir": str(run), "configuration": vars(args), "records": [],
              "assumed_endpoint_s": 0.6, "page_poll_s_not_included": 0.35,
              "scope": "single concurrent request; existing local WAV replay; no microphone or service mutation",
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "funasr", "transformers")},
              "python": platform.python_version(), "gpu": torch.cuda.get_device_name(0),
              "cuda": torch.version.cuda, "bf16_supported": torch.cuda.is_bf16_supported()}
    examples = Path.home() / ".cache/modelscope/models/iic--speech_campplus_sv_zh-cn_16k-common/snapshots/master/examples"
    paths = [examples / "speaker1_b_cn_16k.wav", ROOT / "Funasr_tests/data/zh_audio2.wav",
             ROOT / "Funasr_tests/data/zh_audio1.wav"]
    clips = []
    for p in paths:
        raw = decode_wav(p.read_bytes())
        wav, trim = speech_trim(raw)
        clips.append((p.stem, wav))
        report.setdefault("clips", []).append(dict(name=p.stem, path=str(p), duration_s=len(wav)/16000,
            source_duration_s=len(raw)/16000, **trim))
    events = (run / "events.jsonl").open("w", encoding="utf-8")
    def emit(record):
        report["records"].append(record)
        events.write(json.dumps(record, ensure_ascii=False) + "\n"); events.flush()
        print(json.dumps(record, ensure_ascii=False), flush=True)
    monitor = GpuMonitor()
    try:
        service_idle()
        time.sleep(0.5)
        print("RUN", run, flush=True)
        spec = next(m for m in default_models(ROOT) if m.id == "fun-asr-nano")
        report["model_dir"] = str(spec.model_dir)
        with (spec.model_dir / "model.pt").open("rb") as weights:
            report["checkpoint_sha256"] = hashlib.file_digest(weights, "sha256").hexdigest()
        started = time.perf_counter()
        adapter = LocalOfflineAsr(spec)
        torch.cuda.synchronize()
        report["load_seconds"] = time.perf_counter() - started
        report["after_load_allocated_mib"] = torch.cuda.memory_allocated()/2**20
        print("LOADED", report["load_seconds"], "seconds", flush=True)

        def decode(audio, precision):
            service_idle()
            torch.cuda.synchronize()
            t = time.perf_counter()
            if precision == "default":
                text = adapter.transcribe(audio)
            else:
                from funasr.utils.postprocess_utils import rich_transcription_postprocess
                result = adapter.model.generate(input=torch.from_numpy(audio.copy()), cache={},
                    batch_size=1, language=None, use_itn=True, itn=True, llm_dtype=precision,
                    max_length=512)
                text = "\n".join(rich_transcription_postprocess(str(r.get("text", ""))).strip() for r in result)
            torch.cuda.synchronize()
            return {"compute_s": time.perf_counter()-t, "text": text,
                    "allocated_peak_mib": torch.cuda.max_memory_allocated()/2**20,
                    "reserved_peak_mib": torch.cuda.max_memory_reserved()/2**20}

        for precision in args.precisions:
            monitor.phase = precision + ":cold_first_call"
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            emit(dict(kind="first_call", precision=precision, **decode(clips[0][1][:24000], precision)))
            # Settle kernels/dtype and free historical allocator blocks before measuring.
            decode(clips[0][1], precision)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            for name, wav in ([] if args.replay_only else clips):
                duration = len(wav)/16000
                monitor.phase = precision + ":whole"
                for repeat in range(args.repeats):
                    emit(dict(kind="whole_utterance", precision=precision, clip=name, repeat=repeat,
                        audio_s=duration, **decode(wav, precision)))
                # Prefix timing probes indicate what short initial audio actually produces.
                monitor.phase = precision + ":prefix"
                for seconds in (0.8, 1.0, 1.5, 2.0, 3.0):
                    emit(dict(kind="prefix_probe", precision=precision, clip=name, audio_s=seconds,
                        **decode(wav[:int(seconds*16000)], precision)))

            # Real-clock single-worker replay. New audio arrives while generate blocks.
            # Skip stale intermediate updates and consume the latest available prefix.
            for name, wav in clips:
                duration = len(wav)/16000
                for initial in args.initial_audio:
                    monitor.phase = precision + ":replay"
                    start = time.perf_counter()
                    next_due, first_text, calls, previews = initial, None, 0, []
                    final_due = duration + 0.6
                    while True:
                        elapsed = time.perf_counter() - start
                        due = min(next_due, final_due)
                        if elapsed < due:
                            time.sleep(min(0.02, due-elapsed))
                            continue
                        is_final = elapsed >= final_due
                        available = min(len(wav), int(elapsed*16000))
                        result = decode(wav[:available], precision)
                        delivered = time.perf_counter() - start
                        calls += 1
                        if first_text is None and any(c.isalnum() for c in result["text"]):
                            first_text = delivered
                        previews.append(dict(input_audio_s=available/16000, delivered_s=delivered,
                                             final=is_final, **result))
                        reused_final = args.reuse_final and not is_final and available == len(wav)
                        if reused_final:
                            while time.perf_counter() - start < final_due:
                                time.sleep(0.01)
                            delivered = time.perf_counter() - start
                        if is_final or reused_final:
                            emit(dict(kind="replay", precision=precision, clip=name, audio_s=duration,
                                initial_audio_s=initial, requested_hop_s=1.0, first_text_s=first_text,
                                final_tail_s=delivered-duration, calls=calls, reused_final=reused_final,
                                previews=previews))
                            break
                        next_due = max(next_due+1.0, (int(delivered)+1.0))
            emit(dict(kind="precision_peak", precision=precision,
                allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
                reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
                llm_dtype=str(next(adapter.model.model.llm.parameters()).dtype)))
    except Exception as exc:
        report["error"] = repr(exc)
        print("BENCHMARK ERROR", repr(exc), flush=True)
        raise
    finally:
        monitor.close()
        events.close()
        report["gpu_samples"] = monitor.samples
        (run / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("RESULTS", run / "results.json", flush=True)


if __name__ == "__main__":
    main()
