"""Local MOSS worker bridge, isolated from the existing Transformers 4 runtime."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import re
import subprocess
from threading import Thread
import time


MODEL_REVISION = "704aa4a9c304e8520be88901e0d1960158ef5b15"
WEIGHT_FILE = "model-00000-of-00001.safetensors"
WEIGHT_BYTES = 1817113576


class MeetingCancelled(RuntimeError):
    pass


def validate_segments(items, duration):
    """Keep overlapping segments and original speaker IDs; never merge by time."""
    segments = []
    for item in items:
        start, end = float(item["start"]), float(item["end"])
        speaker, text = str(item["speaker"]), str(item["text"]).strip()
        if (not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end < start or end > duration + 1
                or not re.fullmatch(r"S\d{1,4}", speaker)):
            raise ValueError("MOSS 返回了无效的说话人或时间戳，原始结果已保留供检查")
        if text:
            segments.append({"start": round(min(start, duration), 3),
                             "end": round(min(end, duration), 3),
                             "speaker": speaker, "text": text})
    # Sorting by start is stable and does not remove timestamp overlaps.
    return sorted(segments, key=lambda segment: segment["start"])


def timestamp(seconds, *, srt=False):
    milliseconds = max(0, round(seconds * 1000))
    hours, rest = divmod(milliseconds, 3600000)
    minutes, rest = divmod(rest, 60000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{',' if srt else '.'}{ms:03d}"


def segment_text(segment):
    return (f"[{timestamp(segment['start'])} → {timestamp(segment['end'])}] "
            f"[{segment['speaker']}] {segment['text']}")


def segments_srt(segments):
    return "\n\n".join(
        f"{index}\n{timestamp(s['start'], srt=True)} --> {timestamp(s['end'], srt=True)}\n"
        f"[{s['speaker']}] {s['text']}" for index, s in enumerate(segments, 1)) + ("\n" if segments else "")


class MossMeetingAsr:
    """One persistent worker, sequential requests, bounded waits and explicit close."""

    def __init__(self, model_dir, project_root, *, python_executable=None, startup_timeout=300):
        root = Path(project_root).resolve()
        self.python = Path(python_executable) if python_executable else root / ".venv-moss/Scripts/python.exe"
        if not self.python.is_file():
            raise RuntimeError("MOSS 独立环境未安装，请运行 scripts/setup_moss_windows.ps1")
        runtime_dir = root / "outputs/moss_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = runtime_dir / f"worker-{time.time_ns()}.log"
        self._log = self.log_path.open("w", encoding="utf-8")
        self._messages = Queue()
        self.process = None
        self.closed = False
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        try:
            self.process = subprocess.Popen(
                [str(self.python), "-u", str(root / "scripts/moss_worker.py"), "--model-dir", str(model_dir)],
                cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                text=True, encoding="utf-8", bufsize=1, env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self._reader = Thread(target=self._read_messages, daemon=True)
            self._reader.start()
            ready = self._wait(startup_timeout, expected="ready")
            self.load_timings = ready.get("timings", {})
            self.runtime_info = ready.get("runtime", {})
        except Exception:
            self.close()
            raise

    def _read_messages(self):
        try:
            for line in self.process.stdout:
                try:
                    self._messages.put(json.loads(line))
                except (ValueError, TypeError):
                    self._messages.put({"type": "error", "error": "MOSS 工作进程返回了无效消息"})
        finally:
            self._messages.put({"type": "exit"})

    def _wait(self, timeout, *, expected, progress=None, cancel_event=None):
        deadline = time.monotonic() + timeout
        while True:
            if cancel_event is not None and cancel_event.is_set():
                self.close()
                raise MeetingCancelled("已取消会议转写；录音已保存，重新加载模型后可重试")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise RuntimeError("MOSS 处理超时，已释放工作进程；请缩短录音后重新加载模型")
            try:
                message = self._messages.get(timeout=min(.2, remaining))
            except Empty:
                continue
            kind = message.get("type")
            if kind == expected:
                return message
            if kind == "progress":
                if progress:
                    progress(message)
            elif kind in {"error", "exit"}:
                self.close()
                raise RuntimeError(message.get("error") or f"MOSS 工作进程已退出，请查看 {self.log_path}")
            else:
                self.close()
                raise RuntimeError("MOSS 工作进程消息顺序异常，请重新加载模型")

    def reset(self):
        if self.closed or self.process.poll() is not None:
            raise RuntimeError("MOSS 工作进程不可用，请重新加载模型")

    def transcribe_file(self, path, *, progress=None, cancel_event=None):
        self.reset()
        try:
            self.process.stdin.write(json.dumps({"audio_path": str(Path(path).resolve())}) + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as exc:
            self.close()
            raise RuntimeError("MOSS 工作进程连接中断，请重新加载模型") from exc
        return self._wait(1800, expected="result", progress=progress, cancel_event=cancel_event)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process is not None:
            if self.process.stdin:
                try:
                    self.process.stdin.close()
                except OSError:
                    pass
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    # Windows venv python.exe can be a redirector process.
                    # Terminating only that PID would orphan the GPU worker.
                    subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=10, check=False,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                else:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            if self.process.stdout:
                self.process.stdout.close()
        self._log.close()
