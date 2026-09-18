"""Download a ModelScope snapshot and emit machine-readable progress events."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


class JsonProgressCallback:
    """Report ModelScope's per-file callbacks as one JSON object per line."""

    def __init__(self, filename: str, file_size: int) -> None:
        self.filename = filename
        self.file_size = max(0, int(file_size))
        self.downloaded = 0
        self.last_emit = 0.0
        self.emit("start", 0)

    def update(self, size: int) -> None:
        self.downloaded = min(self.file_size, self.downloaded + max(0, int(size)))
        now = time.monotonic()
        if now - self.last_emit >= 0.2 or self.downloaded == self.file_size:
            self.emit("update", self.downloaded)

    def end(self) -> None:
        if self.file_size:
            self.downloaded = self.file_size
        self.emit("complete", self.downloaded)

    def emit(self, event: str, downloaded: int) -> None:
        self.last_emit = time.monotonic()
        print(
            json.dumps(
                {
                    "type": "download",
                    "event": event,
                    "filename": self.filename,
                    "file_size": self.file_size,
                    "downloaded": downloaded,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()

    from modelscope import snapshot_download
    from modelscope.hub import ProgressCallback

    class Callback(JsonProgressCallback, ProgressCallback):
        pass

    destination = snapshot_download(
        args.model_id,
        local_dir=str(args.target),
        max_workers=1,
        progress_callbacks=[Callback],
    )
    print(json.dumps({"type": "done", "path": destination}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(json.dumps({"type": "error", "message": str(error)}, ensure_ascii=False), flush=True)
        print(f"error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
