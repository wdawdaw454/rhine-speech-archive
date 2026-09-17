"""Measure corpus CER and compute RTF for a SenseVoice ONNX bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.io import load_audio
from src.data.normalize import normalize_chinese_text
from src.evaluation.cer import char_edit_distance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark SenseVoice ONNX quality and speed")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/processed/aishell_full/dev/sources.jsonl",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--intra-op-threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _summarize(rows: list[dict[str, Any]], *, load_seconds: float) -> dict[str, Any]:
    audio_seconds = sum(row["audio_seconds"] for row in rows)
    decode_seconds = sum(row["decode_seconds"] for row in rows)
    errors = sum(row["errors"] for row in rows)
    chars = sum(row["reference_chars"] for row in rows)
    return {
        "schema_version": 1,
        "phase": "5.11",
        "bundle_dir": str(rows[0]["bundle_dir"]) if rows else "",
        "examples": len(rows),
        "corpus_cer": errors / chars if chars else 0.0,
        "errors": errors,
        "chars": chars,
        "exact": sum(row["cer"] == 0.0 for row in rows),
        "audio_seconds": audio_seconds,
        "decode_seconds": decode_seconds,
        "compute_rtf": decode_seconds / audio_seconds if audio_seconds else 0.0,
        "decode_p50": _percentile([row["decode_seconds"] for row in rows], 0.50),
        "decode_p95": _percentile([row["decode_seconds"] for row in rows], 0.95),
        "per_sample_cer_p50": statistics.median([row["cer"] for row in rows]) if rows else 0.0,
        "load_seconds": load_seconds,
    }


def benchmark_bundle(
    bundle_dir: Path,
    manifest_path: Path,
    *,
    limit: int = 8,
    offset: int = 0,
    intra_op_threads: int = 4,
) -> dict[str, Any]:
    if limit <= 0 or offset < 0:
        raise ValueError("limit must be positive and offset must be non-negative")
    from src.models.sensevoice_onnx import OnnxSenseVoiceAsr

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    selected = records[offset : offset + limit]
    if not selected:
        raise ValueError("manifest selection is empty")

    started = time.perf_counter()
    backend = OnnxSenseVoiceAsr.from_bundle(bundle_dir, intra_op_threads=intra_op_threads)
    load_seconds = time.perf_counter() - started

    rows: list[dict[str, Any]] = []
    root = manifest_path.parent
    for record in selected:
        waveform, sample_rate = load_audio(root / record["audio_path"], target_sr=16000)
        reference = normalize_chinese_text(record["extras"]["text"])
        decode_started = time.perf_counter()
        hypothesis = backend.accept(waveform, is_final=True)
        decode_seconds = time.perf_counter() - decode_started
        backend.reset()
        hypothesis = normalize_chinese_text(hypothesis)
        errors = char_edit_distance(reference, hypothesis)
        rows.append(
            {
                "id": record["id"],
                "bundle_dir": str(bundle_dir.resolve()),
                "reference": reference,
                "hypothesis": hypothesis,
                "errors": errors,
                "reference_chars": len(reference),
                "cer": errors / max(1, len(reference)),
                "audio_seconds": len(waveform) / sample_rate,
                "decode_seconds": decode_seconds,
            }
        )
    summary = _summarize(rows, load_seconds=load_seconds)
    summary["rows"] = rows
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = benchmark_bundle(
        args.bundle_dir,
        args.manifest,
        limit=args.limit,
        offset=args.offset,
        intra_op_threads=args.intra_op_threads,
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
