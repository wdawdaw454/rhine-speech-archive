"""Launch the local browser UI for microphone dictation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.web.dictation_server import serve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local browser microphone dictation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    return parser


def _already_running(url: str) -> bool:
    try:
        with urlopen(url + "api/status", timeout=0.8) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    url = f"http://{args.host}:{args.port}/"
    if _already_running(url):
        if args.open_browser:
            webbrowser.open(url)
        return 0
    serve(
        project_root=PROJECT_ROOT,
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
