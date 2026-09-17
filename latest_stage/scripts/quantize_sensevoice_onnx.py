"""Quantize the SenseVoice ONNX bundle with MatMul dynamic INT8 weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ASSETS = ("tokens.json", "am.mvn")
EXCLUDE_PATTERNS = ("output", "bias_encoder", "bias_decoder")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a dynamic INT8 SenseVoice bundle")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-type", choices=("qint8", "quint8"), default="qint8")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate(bundle_dir: Path, output_dir: Path) -> None:
    required = ("model.onnx", "bundle.json", *ASSETS)
    missing = [name for name in required if not (bundle_dir / name).is_file()]
    if missing:
        raise SystemExit(f"source bundle missing files: {', '.join(missing)}")
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")


def _excluded_nodes(node_names: list[str]) -> list[str]:
    return [name for name in node_names if any(pattern in name for pattern in EXCLUDE_PATTERNS)]


def quantize_bundle(
    bundle_dir: Path,
    output_dir: Path,
    *,
    weight_type: str = "qint8",
) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    output_dir = output_dir.resolve()
    _validate(bundle_dir, output_dir)

    staging = output_dir.with_name(f".{output_dir.name}.partial-{uuid.uuid4().hex[:8]}")
    staging.mkdir(parents=True)
    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic

        source_model = onnx.load(str(bundle_dir / "model.onnx"), load_external_data=False)
        onnx.checker.check_model(source_model)
        excluded = _excluded_nodes([node.name for node in source_model.graph.node])
        quant_type = QuantType.QInt8 if weight_type == "qint8" else QuantType.QUInt8
        target_model = staging / "model.onnx"
        quantize_dynamic(
            bundle_dir / "model.onnx",
            target_model,
            op_types_to_quantize=["MatMul"],
            per_channel=True,
            reduce_range=False,
            weight_type=quant_type,
            nodes_to_exclude=excluded,
        )
        onnx.checker.check_model(onnx.load(str(target_model), load_external_data=False))

        for name in ASSETS:
            shutil.copy2(bundle_dir / name, staging / name)
        source_manifest = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
        manifest = {
            "schema_version": 1,
            "phase": "5.11",
            "model": source_manifest.get("model", "SenseVoiceSmall"),
            "source_bundle_dir": str(bundle_dir),
            "source_model_sha256": _sha256(bundle_dir / "model.onnx"),
            "quantization": {
                "method": "dynamic",
                "op_types": ["MatMul"],
                "per_channel": True,
                "reduce_range": False,
                "weight_type": weight_type,
                "excluded_nodes": len(excluded),
            },
            "onnx": {
                "path": target_model.name,
                "sha256": _sha256(target_model),
                "size_mb": round(target_model.stat().st_size / 1024**2, 2),
                "opset": source_manifest.get("onnx", {}).get("opset", 14),
                "io": source_manifest.get("onnx", {}).get("io", {}),
            },
            "assets": list(ASSETS),
        }
        (staging / "bundle.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.replace(output_dir)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = quantize_bundle(args.bundle_dir, args.output_dir, weight_type=args.weight_type)
    print(f"model_size_mb={manifest['onnx']['size_mb']}")
    print(f"weight_type={args.weight_type}")
    print(f"bundle={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
