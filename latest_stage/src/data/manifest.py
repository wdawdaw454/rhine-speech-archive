"""JSONL manifest read/write helpers."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class ManifestEntry:
    id: str
    audio_path: str
    sample_rate: int = 16000
    duration: float = 0.0
    extras: dict = field(default_factory=dict)


def write_jsonl(path: str | Path, entries: list[ManifestEntry]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[ManifestEntry]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield ManifestEntry(**d)