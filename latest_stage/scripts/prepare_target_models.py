"""Download target-speaker dependencies into the repository model root."""
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.catalog import MODEL_SPECS
from src.models.downloader import download_model


if __name__ == "__main__":
    for key in ("speaker", "vad"):
        print(download_model(MODEL_SPECS[key], PROJECT_ROOT))
