from pathlib import Path

from src.data.manifest import write_jsonl, read_jsonl, ManifestEntry


def test_jsonl_roundtrip(tmp_path: Path):
    p = tmp_path / "manifest.jsonl"
    entries = [
        ManifestEntry(id="a", audio_path="/x/a.wav", sample_rate=16000, duration=1.0),
        ManifestEntry(id="b", audio_path="/x/b.wav", sample_rate=16000, duration=2.0),
    ]
    write_jsonl(p, entries)
    loaded = list(read_jsonl(p))
    assert [e.id for e in loaded] == ["a", "b"]
    assert loaded[1].duration == 2.0


def test_jsonl_appendable(tmp_path: Path):
    p = tmp_path / "manifest.jsonl"
    write_jsonl(p, [ManifestEntry(id="a", audio_path="/x/a.wav", sample_rate=16000, duration=1.0)])
    write_jsonl(p, [ManifestEntry(id="b", audio_path="/x/b.wav", sample_rate=16000, duration=2.0)])
    loaded = list(read_jsonl(p))
    assert [e.id for e in loaded] == ["a", "b"]