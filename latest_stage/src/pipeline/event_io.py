"""Event sink implementations for JSONL and human-readable transcripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .events import EventType, PipelineEvent


class EventSink(Protocol):
    def write(self, event: PipelineEvent) -> None: ...


class NullSink:
    def write(self, event: PipelineEvent) -> None:
        return None


class JsonlEventWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", newline="\n")

    def write(self, event: PipelineEvent) -> None:
        self._file.write(event.to_json() + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _timestamp(seconds: float | None) -> str:
    if seconds is None:
        seconds = 0.0
    total_centis = max(0, int(round(seconds * 100)))
    hours, rem = divmod(total_centis, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, centis = divmod(rem, 100)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centis:02d}"


class TranscriptWriter:
    """Write committed target captions in a compact subtitle format."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", newline="\n")

    def write(self, event: PipelineEvent) -> None:
        if event.type != EventType.CAPTION_COMMIT or not event.text:
            return
        start = _timestamp(event.start)
        end = _timestamp(event.end)
        self._file.write(f"[{start} -> {end}] [target] {event.text}\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class CompositeEventSink:
    def __init__(self, *sinks: EventSink) -> None:
        self.sinks = sinks

    def write(self, event: PipelineEvent) -> None:
        for sink in self.sinks:
            sink.write(event)

    def close(self) -> None:
        for sink in self.sinks:
            closer = getattr(sink, "close", None)
            if callable(closer):
                closer()


def read_events(path: str | Path) -> list[PipelineEvent]:
    """Read events written by JsonlEventWriter for evaluation utilities."""
    events: list[PipelineEvent] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            event_type = EventType(data.pop("type"))
            commit_state = data.pop("commit_state", None)
            extras = data.pop("extras", {})
            events.append(
                PipelineEvent(
                    type=event_type,
                    commit_state=commit_state,
                    extras=extras,
                    **data,
                )
            )
    return events
