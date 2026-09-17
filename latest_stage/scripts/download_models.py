"""Download Phase 1 models from ModelScope."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models.catalog import MODEL_SPECS
from src.models.downloader import download_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=[*MODEL_SPECS, "all"],
        default="all",
        help="model key to download (default: all)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = download_models([args.model], args.project_root)
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
