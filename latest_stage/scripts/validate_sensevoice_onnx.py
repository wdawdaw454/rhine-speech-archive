"""Validate a SenseVoice ONNX bundle against the source FunASR model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.io import load_audio
from src.models.sensevoice_onnx import OnnxSenseVoiceAsr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check SenseVoiceSmall ONNX inference")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--source-model-dir",
        type=Path,
        help="Optional source FunASR bundle used as the reference decoder",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0", "cuda:1"))
    parser.add_argument("--output", type=Path, help="Write a JSON summary to this path")
    return parser


def _visible_text(text: str) -> str:
    if text.startswith("<|"):
        return text.rsplit(">", 1)[-1].strip()
    return text.strip()


def _source_text(source_model_dir: Path, waveform, device: str) -> str:
    from funasr import AutoModel

    model = AutoModel(model=str(source_model_dir), device=device, disable_update=True)
    result = model.generate(input=waveform, language="zh", use_itn=False)[0]
    return _visible_text(str(result.get("text", "")))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    waveform, sample_rate = load_audio(args.input, target_sr=16000)
    started = time.perf_counter()
    backend = OnnxSenseVoiceAsr.from_bundle(args.bundle_dir, decode_chunk=0.50)
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    onnx_text = backend.accept(waveform, is_final=True)
    decode_seconds = time.perf_counter() - started

    summary: dict[str, Any] = {
        "schema_version": 1,
        "phase": "5.10",
        "input": str(args.input.resolve()),
        "bundle_dir": str(args.bundle_dir.resolve()),
        "sample_rate": sample_rate,
        "audio_seconds": round(len(waveform) / sample_rate, 6),
        "onnx_text": onnx_text,
        "load_seconds": round(load_seconds, 6),
        "decode_seconds": round(decode_seconds, 6),
    }
    if args.source_model_dir is not None:
        source_text = _source_text(args.source_model_dir, waveform, args.device)
        summary.update(
            {
                "source_text": source_text,
                "exact_match": source_text == onnx_text,
            }
        )

    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if args.source_model_dir is not None and not summary["exact_match"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
