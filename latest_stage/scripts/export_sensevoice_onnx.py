"""Export SenseVoiceSmall as a self-contained ONNX inference bundle."""

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

SOURCE_FILES = ("tokens.json", "am.mvn")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export SenseVoiceSmall to ONNX")
    parser.add_argument(
        "--source-model-dir",
        type=Path,
        default=PROJECT_ROOT.parent / "models/sensevoice-small",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT.parent / "models/sensevoice-onnx",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--opset", type=int, default=14)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate(source_model_dir: Path, output_dir: Path) -> None:
    if not source_model_dir.is_dir():
        raise SystemExit(f"source model directory not found: {source_model_dir}")
    missing = [
        name
        for name in ("config.yaml", "model.pt", *SOURCE_FILES)
        if not (source_model_dir / name).is_file()
    ]
    if missing:
        raise SystemExit(f"source bundle missing files: {', '.join(missing)}")
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")


def export_sensevoice_onnx(
    source_model_dir: Path,
    output_dir: Path,
    *,
    device: str = "cpu",
    opset: int = 14,
) -> dict[str, Any]:
    source_model_dir = source_model_dir.resolve()
    output_dir = output_dir.resolve()
    _validate(source_model_dir, output_dir)
    staging = output_dir.with_name(f".{output_dir.name}.partial-{uuid.uuid4().hex[:8]}")
    staging.mkdir(parents=True)

    try:
        # Imported lazily so unit-test environments do not need FunASR.
        from funasr import AutoModel
        import onnx

        model = AutoModel(model=str(source_model_dir), device=device, disable_update=True)
        exported_dir = Path(
            model.export(
                type="onnx",
                output_dir=str(staging),
                device=device,
                opset_version=opset,
            )
        )
        model_path = exported_dir / "model.onnx"
        if not model_path.is_file():
            raise RuntimeError(f"FunASR did not create {model_path}")

        for name in SOURCE_FILES:
            shutil.copy2(source_model_dir / name, staging / name)
        onnx.checker.check_model(onnx.load(str(model_path), load_external_data=False))

        manifest = {
            "schema_version": 1,
            "phase": "5.10",
            "model": "SenseVoiceSmall",
            "source_model_dir": str(source_model_dir),
            "source_model_sha256": _sha256(source_model_dir / "model.pt"),
            "onnx": {
                "path": model_path.name,
                "sha256": _sha256(model_path),
                "size_mb": round(model_path.stat().st_size / 1024**2, 2),
                "opset": opset,
                "io": {
                    "speech": ["batch_size", "feats_length", 560],
                    "speech_lengths": ["batch_size"],
                    "language": ["batch_size"],
                    "textnorm": ["batch_size"],
                    "ctc_logits": ["batch_size", "logits_length", 25055],
                    "encoder_out_lens": ["batch_size"],
                },
            },
            "assets": list(SOURCE_FILES),
            "decoder": {
                "type": "greedy_ctc",
                "blank_id": 0,
                "language_id_zh": 3,
                "textnorm_id_woitn": 15,
            },
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
    manifest = export_sensevoice_onnx(
        args.source_model_dir,
        args.output_dir,
        device=args.device,
        opset=args.opset,
    )
    print(f"model_size_mb={manifest['onnx']['size_mb']}")
    print(f"opset={manifest['onnx']['opset']}")
    print(f"bundle={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
