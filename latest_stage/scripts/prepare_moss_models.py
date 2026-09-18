"""Download the pinned official MOSS checkpoint; never upload user recordings."""
from pathlib import Path
import hashlib

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
REVISION = "704aa4a9c304e8520be88901e0d1960158ef5b15"
ROOT = Path(__file__).resolve().parents[1]


def main():
    from huggingface_hub import HfApi, hf_hub_url
    import requests
    directory = ROOT.parent / "models/moss-transcribe-diarize"
    directory.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    # Some proxies omit Hub HEAD metadata. GET the pinned files and verify
    # each against the API's Git-blob / LFS checksum instead of disabling TLS.
    for item in info.siblings:
        name = item.rfilename
        if "/" in name or not (name.endswith((".json", ".safetensors", ".py", ".jinja", ".txt")) or name == "README.md"):
            continue
        target = directory / name
        expected = item.lfs.sha256 if item.lfs else item.blob_id
        def digest(path):
            h = hashlib.sha256() if item.lfs else hashlib.sha1(f"blob {item.size}\0".encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        if target.is_file() and target.stat().st_size == item.size and digest(target) == expected:
            print(f"Verified cached {name}", flush=True)
            continue
        print(f"Downloading {name} ({item.size / 1024**2:.1f} MiB)", flush=True)
        temporary = target.with_name(target.name + ".download")
        with requests.get(hf_hub_url(MODEL_ID, name, revision=REVISION), stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    handle.write(chunk)
        if temporary.stat().st_size != item.size or digest(temporary) != expected:
            raise RuntimeError(f"Checksum mismatch: {name}; incomplete file left at {temporary}")
        temporary.replace(target)
    print(f"MOSS checkpoint ready: {directory}")


if __name__ == "__main__":
    main()
