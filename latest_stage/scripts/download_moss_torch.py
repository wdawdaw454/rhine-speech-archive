"""Resumable, hash-verified Windows CUDA wheel download from a domestic mirror."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
NAME = "torch-2.8.0+cu128-cp311-cp311-win_amd64.whl"
URL = "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/" + NAME.replace("+", "%2B")
SIZE = 3461420395
# Published by https://download.pytorch.org/whl/cu128/torch/
SHA256 = "34c55443aafd31046a7963b63d30bc3b628ee4a704f826796c865fdfd05bb596"
CHUNK = 16 * 1024 * 1024


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    directory = ROOT / ".cache/moss-wheels"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / NAME
    if output.exists() and output.stat().st_size == SIZE and digest(output) == SHA256:
        print(f"Already verified: {output}", flush=True)
        return
    parts = directory / (NAME + ".parts")
    parts.mkdir(exist_ok=True)

    def download(index):
        start, end = index * CHUNK, min(SIZE, (index + 1) * CHUNK) - 1
        part = parts / f"{index:04d}"
        for attempt in range(10):
            offset = part.stat().st_size if part.exists() else 0
            if offset == end - start + 1:
                return part
            if offset > end - start + 1:
                raise ValueError(f"Oversized part: {part}")
            try:
                request = urllib.request.Request(URL, headers={"Range": f"bytes={start + offset}-{end}"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    expected = f"bytes {start + offset}-{end}/{SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError(f"Invalid range response: {response.status} {response.headers}")
                    with part.open("ab") as handle:
                        shutil.copyfileobj(response, handle, length=1024 * 1024)
                if part.stat().st_size == end - start + 1:
                    return part
            except (OSError, ValueError) as exc:
                print(f"Part {index} retry {attempt + 1}: {exc}", flush=True)
                time.sleep(min(attempt + 1, 5))
        raise RuntimeError(f"Could not finish {part}")

    count = (SIZE + CHUNK - 1) // CHUNK
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, future in enumerate(as_completed([pool.submit(download, i) for i in range(count)]), 1):
            future.result()
            print(f"Downloaded {done}/{count} parts in {time.monotonic() - started:.0f}s", flush=True)
    assembled = output.with_suffix(".assembling")
    with assembled.open("wb") as handle:
        for i in range(count):
            with (parts / f"{i:04d}").open("rb") as source:
                shutil.copyfileobj(source, handle, length=1024 * 1024)
    if assembled.stat().st_size != SIZE or digest(assembled) != SHA256:
        raise ValueError("Wheel SHA-256 mismatch; not installing")
    assembled.replace(output)
    print(f"SHA-256 verified: {output}", flush=True)


if __name__ == "__main__":
    main()
