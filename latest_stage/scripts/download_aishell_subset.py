"""Download the ModelScope AISHELL-1 subset into the project data directory."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_ID = "speech_asr/speech_asr_aishell1_subset"
DATASET_NAME = "speech_asr_aishell1_subset"
NAMESPACE = "speech_asr"
ZIP_NAME = "speech_asr_aishell_subset.zip"
CSV_NAMES = (
    "speech_asr_aishell_subset_trainsets.csv",
    "speech_asr_aishell_subset_validation.csv",
    "speech_asr_aishell_subset_testsets.csv",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download AISHELL-1 subset from ModelScope")
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "data/raw/aishell_subset"
    )
    return parser


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f"{destination.name}.partial")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with staging.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                handle.write(chunk)
    shutil.move(str(staging), str(destination))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from modelscope.hub.api import HubApi

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    api = HubApi()
    objects = api.list_oss_dataset_objects(
        DATASET_NAME,
        NAMESPACE,
        max_limit=1,
        is_recursive=True,
        is_filter_dir=True,
        revision="master",
    )
    matches = [item for item in objects if item.get("Path") == ZIP_NAME]
    if len(matches) != 1:
        raise SystemExit(f"expected one AISHELL subset archive, found {len(matches)}")
    archive_object = matches[0]
    archive_path = output_root / ZIP_NAME
    if not archive_path.exists():
        _download(str(archive_object["Url"]), archive_path)
        if archive_path.stat().st_size != int(archive_object["Size"]):
            raise SystemExit(f"archive size mismatch: {archive_path}")

    extracted = output_root / "speech_asr_aishell_subset"
    if not extracted.exists():
        staging = output_root / "speech_asr_aishell_subset.partial"
        if staging.exists():
            raise SystemExit(f"incomplete extraction already exists: {staging}")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(staging)
        shutil.move(str(staging / "speech_asr_aishell_subset"), str(extracted))
        staging.rmdir()

    for name in CSV_NAMES:
        csv_path = output_root / name
        if csv_path.exists():
            continue
        url = api.get_dataset_file_url(name, DATASET_NAME, NAMESPACE)
        _download(url, csv_path)

    metadata = {
        "dataset_id": DATASET_ID,
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "audio_root": str(extracted),
        "csv_files": list(CSV_NAMES),
    }
    (output_root / "source.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"raw_root={output_root}")
    print(f"audio_root={extracted}")


if __name__ == "__main__":
    main()
