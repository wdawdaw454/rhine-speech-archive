"""Run the Phase 1 file-simulated realtime pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.cli import main

if __name__ == "__main__":
    main(["run", *sys.argv[1:]])
