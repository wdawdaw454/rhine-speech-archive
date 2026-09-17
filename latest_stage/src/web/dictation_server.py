"""Local-only HTTP server for browser-controlled microphone dictation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import gc
from io import BytesIO
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from threading import Event, Lock, Thread
import time
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, unquote, urlparse
import webbrowser
from .samplex_service import SampleXService

import numpy as np

from ..audio.microphone import MicrophoneStream, write_status_file
from ..audio.system_audio import SystemAudioStream, system_audio_device
from ..models.sensevoice_onnx import OnnxSenseVoiceAsr
from ..models.moss_meeting import (MossMeetingAsr, MeetingCancelled, MODEL_REVISION,
    WEIGHT_FILE, WEIGHT_BYTES, validate_segments, segment_text, segments_srt)
from ..pipeline.event_io import CompositeEventSink, JsonlEventWriter
from ..pipeline.events import EventType, PipelineEvent
from ..pipeline.fast_dictation import FastDictationPipeline
from ..pipeline.nano_realtime import NanoRealtimePipeline
from ..pipeline.target_dictation import TargetDictationPipeline
from ..speaker.web_target import TargetModels, MODEL_ID
from ..speaker.enrollment import load_profile, save_profile, profile_embedding


@dataclass(frozen=True)
class DictationModel:
    id: str
    name: str
    model_dir: Path
    description: str
    backend: str = "onnx"
    device: str = "cpu"
    required_file: str = "model.onnx"
    required_bytes: int | None = None
    decode_interval: float | None = None
    modes: tuple[str, ...] = ("streaming",)
    model_key: str | None = None
    recognition_types: tuple[str, ...] = ("normal",)

    def public_dict(self) -> dict[str, Any]:
        marker = self.model_dir / self.required_file
        available = marker.is_file()
        if available and self.required_bytes is not None:
            available = marker.stat().st_size == self.required_bytes
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "backend": self.backend,
            "device": self.device,
            "available": available,
            "modes": list(self.modes),
            "recognition_types": list(self.recognition_types),
            "target_name": f"FSMN-VAD + CAM++ + {self.name}" if "target" in self.recognition_types else None,
        }


def default_models(project_root: Path) -> list[DictationModel]:
    """Central registry used by the UI and future model additions."""
    cache = Path.home() / ".cache/modelscope/models"
    def snapshot(name: str) -> Path:
        return cache / name / "snapshots/master"

    return [
        DictationModel(
            id="sensevoice-realtime", name="SenseVoice",
            model_dir=project_root / "models/sensevoice_small_int8_bundle",
            description="实时更新 · ONNX INT8 / CPU · 分块重识别，非原生流式",
            required_bytes=241030219, decode_interval=0.80,
            recognition_types=("normal", "target"),
        ),
        DictationModel(
            id="sensevoice-small", name="SenseVoice Small",
            model_dir=snapshot("iic--SenseVoiceSmall"),
            description="非流式 · 中英等多语言 · 轻量快速，自动清理情感标签",
            backend="funasr-offline", device="cuda:0", required_file="model.pt",
            modes=("offline",),
        ),
        DictationModel(
            id="fun-asr-nano", name="Fun-ASR-Nano",
            model_dir=snapshot("FunAudioLLM--Fun-ASR-Nano-2512"),
            description="准实时 · PyTorch FP32 / GPU · FSMN-VAD 分段，预览可修订；也支持非实时",
            backend="funasr-offline", device="cuda:0", required_file="model.pt",
            modes=("streaming", "offline"), decode_interval=1.2,
        ),
        DictationModel(
            id="qwen3-asr", name="Qwen3-ASR 1.7B",
            model_dir=snapshot("Qwen--Qwen3-ASR-1.7B"),
            description="非流式 · 多语言自动检测 · 显存和加载耗时较高",
            backend="funasr-offline", device="cuda:0", required_file="model.safetensors.index.json",
            modes=("offline",), model_key="Qwen/Qwen3-ASR-1.7B",
        ),
        DictationModel(
            id="moss-transcribe-diarize", name="MOSS-Transcribe-Diarize",
            model_dir=project_root / "models/moss_transcribe_diarize",
            description="非实时多人会议 · 0.9B / GPU · 整段联合转写、说话人编号与时间戳，支持重叠讲话但不保证无误",
            backend="moss-meeting", device="cuda:0", required_file=WEIGHT_FILE,
            required_bytes=WEIGHT_BYTES, modes=("offline",), recognition_types=("meeting",),
        ),
    ]


class LocalOfflineAsr:
    """Local FunASR adapters; input is always mono float32 at 16 kHz."""

    def __init__(self, spec: DictationModel) -> None:
        from funasr import AutoModel
        import torch

        device = spec.device if torch.cuda.is_available() else "cpu"
        options = dict(device=device, disable_update=True, disable_pbar=True,
                       disable_log=True, trust_remote_code=False)
        if spec.model_key:
            options.update(model=spec.model_key, model_path=str(spec.model_dir),
                           max_inference_batch_size=1, max_new_tokens=512)
        else:
            options["model"] = str(spec.model_dir)
        self.model_id = spec.id
        self.model = AutoModel(**options)
        if self.model_id == "fun-asr-nano":
            from ..models.streaming_vad import FunasrStreamingVad

            vad_dir = Path.home() / ".cache/modelscope/models/iic--speech_fsmn_vad_zh-cn-16k-common-pytorch/snapshots/master"
            if not (vad_dir / "model.pt").is_file():
                raise FileNotFoundError("缺少 Nano 实时模式所需的 FSMN-VAD，请运行 scripts/prepare_target_models.py")
            # Separate CPU VAD: never share mutable VAD caches with target mode.
            self.realtime_vad = FunasrStreamingVad(
                vad_dir, device="cpu", end_silence_ms=600, max_segment_ms=8000)

    def reset(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray) -> str:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        samples = np.asarray(audio, dtype=np.float32)
        language = "auto"
        options = {}
        if self.model_id == "fun-asr-nano":
            # Nano's chat adapter accepts torch.Tensor or file paths, not numpy.
            import torch
            samples = torch.from_numpy(samples)
            language = None
            options = {"llm_dtype": "fp32", "max_new_tokens": 512}
        results = self.model.generate(input=samples,
                                      cache={}, batch_size=1, language=language,
                                      use_itn=True, itn=True, **options)
        return "\n".join(rich_transcription_postprocess(str(r.get("text", ""))).strip()
                         for r in results if r.get("text"))


class PlainTranscriptWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8", newline="\n")

    def write(self, event: PipelineEvent) -> None:
        if event.type == EventType.CAPTION_COMMIT and event.text:
            self._handle.write(event.text.strip() + "\n")
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()


BackendFactory = Callable[[DictationModel], Any]
StreamFactory = Callable[[Event], Iterable[tuple[float, np.ndarray]]]

BUSY_STATES = {"loading", "starting", "listening", "stopping", "transcribing", "enrolling", "enroll_recording"}
MAX_WAV_BYTES = 64 * 1024 * 1024
MAX_WAV_SECONDS = 600


def decode_wav(payload: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Validate a bounded WAV upload before allocating its decoded samples."""
    import soundfile as sf
    from scipy.signal import resample_poly

    if not payload or len(payload) > MAX_WAV_BYTES:
        raise ValueError("请选择非空 WAV 文件，大小不能超过 64 MB")
    try:
        with sf.SoundFile(BytesIO(payload)) as source:
            if source.format not in {"WAV", "WAVEX", "RF64"}:
                raise ValueError("仅支持 WAV 音频文件")
            if not (8000 <= source.samplerate <= 192000 and 1 <= source.channels <= 8):
                raise ValueError("WAV 需为 8–192 kHz、1–8 声道")
            if not 0 < source.frames <= source.samplerate * MAX_WAV_SECONDS:
                raise ValueError("WAV 时长必须在 0–600 秒之间")
            original_rate = source.samplerate
            audio = source.read(dtype="float32", always_2d=True).mean(axis=1)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise ValueError("无法读取 WAV 文件，文件可能已损坏") from exc
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("WAV 包含无效音频采样")
    if original_rate != sample_rate:
        audio = resample_poly(audio, sample_rate, original_rate)
    return np.asarray(audio, dtype=np.float32)


class DictationController:
    """Own one local microphone session and expose thread-safe UI state."""

    def __init__(
        self,
        *,
        project_root: Path,
        models: list[DictationModel] | None = None,
        backend_factory: BackendFactory | None = None,
        stream_factory: StreamFactory | None = None,
        system_stream_factory: StreamFactory | None = None,
        sample_rate: int = 16000,
        input_chunk: float = 0.10,
        decode_interval: float = 0.50,
        intra_op_threads: int = 4,
        target_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self._models = {model.id: model for model in (models or default_models(project_root))}
        self._backend_factory = backend_factory or self._build_backend
        self._stream_factory = stream_factory or self._build_stream
        self._system_stream_factory = system_stream_factory or self._build_system_stream
        self.sample_rate = int(sample_rate)
        self.input_chunk = float(input_chunk)
        self.decode_interval = float(decode_interval)
        self.intra_op_threads = int(intra_op_threads)
        self._lock = Lock()
        self._backend_cache: dict[str, Any] = {}
        self._loaded_model_id: str | None = None
        self._stop_event = Event()
        self._worker: Thread | None = None
        self._state = "idle"
        self._message = "准备就绪"
        self._error: str | None = None
        self._model_id = next(
            (model.id for model in self._models.values() if model.public_dict()["available"]),
            next(iter(self._models), ""),
        )
        self._committed: list[str] = []
        self._partial = ""
        self._started_at: float | None = None
        self._run_id: str | None = None
        self._output_dir: str | None = None
        self._load_timings: dict[str, Any] = {}
        self._revision = 0
        self._mode = "streaming"
        self._source = "microphone"
        self._capture_device = ""
        self._filename = ""
        self._recording_seconds = 0.0
        self._audio_level = 0.0
        self._recording_path: Path | None = None
        self._realtime_metrics: dict[str, Any] = {}
        self.max_recording_seconds = 120.0
        self._target_factory = target_factory or TargetModels
        self._target_models = None
        self._target_only = False
        self._recognition_type = "normal"
        self._run_context: dict[str, str] | None = None
        self._meeting_segments: list[dict[str, Any]] = []
        self._meeting_metrics: dict[str, Any] = {}
        self._cancel_event = Event()
        self._speaker_score = None
        self._speaker_decision = "pending"
        self._speaker_threshold = 0.45
        self._profile_path = self.project_root / "outputs/speaker_profiles/browser_target.json"
        self._profile = None
        if self._profile_path.is_file():
            try:
                profile = load_profile(self._profile_path)
                profile_embedding(profile)
                if profile.model_id != MODEL_ID:
                    raise ValueError("声纹模型不匹配")
                self._profile = profile
            except Exception as exc:
                self._error = f"已保存声纹无法读取，请重新注册：{exc}"

    def model_list(self) -> list[dict[str, Any]]:
        return [model.public_dict() for model in self._models.values()]

    def _touch(self) -> None:
        self._revision += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = time.monotonic() - self._started_at if self._started_at else 0.0
            return {
                "state": self._state,
                "message": self._message,
                "error": self._error,
                "model_id": self._model_id,
                "loaded_model_id": self._loaded_model_id,
                "source": self._source,
                "capture_device": self._capture_device,
                "filename": self._filename,
                "mode": self._mode,
                "recording_seconds": round(self._recording_seconds, 1),
                "max_recording_seconds": self.max_recording_seconds,
                "audio_level": self._audio_level,
                "recording_url": f"/api/recording?run={self._run_id}" if self._recording_path else None,
                "itn_enabled": True,
                "committed_text": "\n".join(self._committed),
                "partial_text": self._partial,
                "elapsed_seconds": round(elapsed, 1),
                "run_id": self._run_id,
                "output_dir": self._output_dir,
                "load_timings": dict(self._load_timings),
                "revision": self._revision,
                "realtime_metrics": dict(self._realtime_metrics),
                "target_ready": self._target_models is not None,
                "target_only": self._target_only,
                "recognition_type": self._recognition_type,
                "run_context": dict(self._run_context) if self._run_id and self._run_context else None,
                "meeting_segments": [dict(segment) for segment in self._meeting_segments],
                "meeting_metrics": dict(self._meeting_metrics),
                "meeting_exports": ({kind: f"/api/meeting-export?format={kind}&run={self._run_id}"
                    for kind in ("json", "srt")} if self._meeting_segments else {}),
                "speaker_score": self._speaker_score,
                "speaker_decision": self._speaker_decision,
                "speaker_threshold": self._speaker_threshold,
                "speaker_profile": ({"name": self._profile.speaker_id,
                    "speech_seconds": self._profile.enroll_duration,
                    "created_at": self._profile.created_at} if self._profile else None),
            }

    def load_target(self) -> None:
        with self._lock:
            if self._state in BUSY_STATES:
                raise RuntimeError("请先结束当前任务")
            self._state, self._message, self._error = "loading", "正在加载 CAM++ / FSMN-VAD（CPU）", None
            self._worker = Thread(target=self._load_target, daemon=True)
            self._worker.start()

    def _load_target(self):
        try:
            models = self._target_models or self._target_factory()
            with self._lock:
                self._target_models = models
                self._state, self._message = "idle", "声纹组件已就绪，可注册目标声音"
                self._touch()
        except Exception as exc:
            with self._lock:
                self._state, self._error = "error", str(exc)
                self._message = "声纹组件加载失败"

    def enroll(self, name: str, audio: np.ndarray | None = None, *, source: str = "microphone"):
        name = name.strip()
        if not 1 <= len(name) <= 40:
            raise ValueError("请输入 1–40 字的目标名称")
        if source not in {"microphone", "system", "wav"}:
            raise ValueError("未知音频输入来源")
        if audio is None and source == "wav":
            raise ValueError("WAV 输入请使用上传注册")
        if audio is not None and source == "system":
            raise ValueError("电脑音频不能同时上传文件")
        if audio is not None and not 3 <= len(audio) / self.sample_rate <= 30:
            raise ValueError("注册 WAV 需为 3–30 秒")
        with self._lock:
            if self._state in BUSY_STATES:
                raise RuntimeError("请先结束当前任务")
            if self._target_models is None:
                raise RuntimeError("请先加载声纹组件")
            self._stop_event = Event()
            self._source = "wav" if audio is not None else source
            self._capture_device = self._filename = ""
            self._state = "enrolling" if audio is not None else "enroll_recording"
            self._message = "正在提取声纹" if audio is not None else (
                "正在采集电脑音频注册，请仅播放目标说话人的声音（最多 30 秒）" if source == "system"
                else "正在录制麦克风注册音频，请仅由目标说话人讲话（最多 30 秒）")
            self._error = None
            self._recording_seconds = self._audio_level = 0.0
            self._recording_path = None
            self._run_id = self._output_dir = self._started_at = None
            self._committed, self._partial = [], ""
            self._meeting_segments, self._meeting_metrics = [], {}
            self._worker = Thread(target=self._enroll, args=(name, audio, source), daemon=True)
            self._worker.start()

    def _enroll(self, name, audio, source):
        stream = None
        try:
            if audio is None:
                if self._stop_event.is_set():
                    raise ValueError("已取消注册录音")
                chunks, count = [], 0
                stream = (self._system_stream_factory(self._stop_event) if source == "system"
                          else self._stream_factory(self._stop_event))
                for _, chunk in stream:
                    chunk = np.asarray(chunk, np.float32)[:30 * self.sample_rate - count]
                    chunks.append(chunk.copy())
                    count += len(chunk)
                    with self._lock:
                        if source == "system":
                            self._capture_device = getattr(stream, "device_name", "默认电脑音频输出")
                        self._recording_seconds = count / self.sample_rate
                        self._audio_level = min(1.0, float(np.sqrt(np.mean(chunk ** 2))) * 8) if len(chunk) else 0
                    if count >= 30 * self.sample_rate:
                        self._stop_event.set()
                        break
                if hasattr(stream, "close"):
                    stream.close()
                audio = np.concatenate(chunks) if chunks else np.empty(0, np.float32)
            with self._lock:
                self._state, self._message = "enrolling", "正在检测有效语音并提取声纹"
                self._audio_level = 0.0
            profile = self._target_models.enroll(audio, name, self.sample_rate)
            profile_embedding(profile)
            # Keep the previous profile intact if extraction or writing fails.
            staging = self._profile_path.with_suffix(".tmp")
            save_profile(profile, staging)
            staging.replace(self._profile_path)
            with self._lock:
                self._profile = profile
                self._state, self._message = "idle", f"已注册：{name}（有效语音 {profile.enroll_duration:.1f} 秒）"
                self._touch()
        except Exception as exc:
            with self._lock:
                self._state, self._message, self._error = "error", "注册失败，原声纹保持不变", str(exc)
        finally:
            if hasattr(stream, "close"):
                stream.close()

            with self._lock:
                self._audio_level = 0.0

    def forget_speaker(self):
        with self._lock:
            if self._state in BUSY_STATES:
                raise RuntimeError("请先结束当前任务")
            self._profile_path.unlink(missing_ok=True)
            self._profile = None
            self._target_only = False
            self._speaker_score, self._speaker_decision = None, "pending"
            self._message = "已删除本地目标声纹"
            self._touch()

    def _build_backend(self, model: DictationModel) -> Any:
        decode_interval = model.decode_interval or self.decode_interval
        if model.backend == "onnx":
            return OnnxSenseVoiceAsr.from_bundle(
                model.model_dir,
                decode_chunk=decode_interval,
                sample_rate=self.sample_rate,
                intra_op_threads=self.intra_op_threads,
                language_id=3,
                textnorm_id=14,
            )
        if model.backend == "funasr-offline":
            return LocalOfflineAsr(model)
        if model.backend == "moss-meeting":
            return MossMeetingAsr(model.model_dir, self.project_root)
        raise RuntimeError(f"unsupported browser backend: {model.backend}")

    def _release_backends(self):
        for backend in self._backend_cache.values():
            close = getattr(backend, "close", None)
            if callable(close):
                close()
        self._backend_cache.clear()

    def shutdown(self):
        self.stop()
        self.wait()
        self._release_backends()

    def _build_stream(self, stop_event: Event) -> MicrophoneStream:
        return MicrophoneStream(
            sample_rate=self.sample_rate,
            blocksize=max(1, int(round(self.input_chunk * self.sample_rate))),
            duration=None,
            stop_event=stop_event,
        )

    def _build_system_stream(self, stop_event: Event) -> SystemAudioStream:
        return SystemAudioStream(
            sample_rate=self.sample_rate,
            blocksize=max(1, round(self.input_chunk * self.sample_rate)),
            duration=self.max_recording_seconds,
            stop_event=stop_event,
        )

    def _backend(self, model: DictationModel) -> tuple[Any, dict[str, Any]]:
        prepare_started = time.perf_counter()
        backend = self._backend_cache.get(model.id)
        if getattr(backend, "closed", False):
            backend = None
        cache_hit = backend is not None
        cleanup_seconds = 0.0
        if backend is None:
            # An 8 GB laptop GPU cannot safely retain both LLM checkpoints.
            # Keep only the selected backend resident and release CUDA caches
            # before loading a different model.
            cleanup_started = time.perf_counter()
            self._release_backends()
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            cleanup_seconds = time.perf_counter() - cleanup_started
            backend = self._backend_factory(model)
            self._backend_cache[model.id] = backend
        backend.reset()
        timings: dict[str, Any] = {
            "cache_hit": cache_hit,
            "cache_cleanup_seconds": round(cleanup_seconds, 6),
            "backend_prepare_seconds": round(time.perf_counter() - prepare_started, 6),
        }
        if not cache_hit:
            timings.update(dict(getattr(backend, "load_timings", {})))
        return backend, timings

    def _validate_model(self, model_id: str, mode: str) -> DictationModel:
        if self._state in BUSY_STATES:
            raise RuntimeError("已有任务正在运行")
        model = self._models.get(model_id)
        if model is None:
            raise ValueError("未知模型")
        if mode not in model.modes:
            raise ValueError("该模型不支持所选识别模式")
        if not model.public_dict()["available"]:
            raise FileNotFoundError(f"模型文件不存在：{model.model_dir}")
        return model

    def load_model(self, model_id: str, mode: str = "streaming") -> None:
        """Prepare weights only. Never opens an input or starts a session."""
        with self._lock:
            model = self._validate_model(model_id, mode)
            self._state = "loading"
            self._message = "正在加载模型"
            self._error = None
            self._model_id = model_id
            self._mode = mode
            self._loaded_model_id = None
            self._load_timings = {}
            self._touch()
            self._worker = Thread(target=self._load_model, args=(model,), daemon=True)
            self._worker.start()

    def _load_model(self, model: DictationModel) -> None:
        try:
            _, timings = self._backend(model)
            with self._lock:
                self._load_timings = timings
                self._loaded_model_id = model.id
                self._state = "idle"
                self._message = "模型已就绪，请开始录音或转写 WAV"
                self._touch()
        except Exception as exc:
            with self._lock:
                self._state = "error"
                self._error = str(exc)
                self._message = "模型加载失败，请重试"
                self._touch()

    def start(self, model_id: str, mode: str = "streaming", *,
              file_audio: np.ndarray | None = None, filename: str = "",
              target_only: bool = False, speaker_threshold: float = 0.45,
              source: str = "microphone", recognition_type: str | None = None) -> None:
        with self._lock:
            model = self._validate_model(model_id, mode)
            if source not in {"microphone", "system", "wav"}:
                raise ValueError("未知音频输入来源")
            if source == "wav" and file_audio is None:
                raise ValueError("WAV 输入需要上传音频文件")
            if source == "system" and file_audio is not None:
                raise ValueError("电脑音频不能同时上传文件")
            if self._loaded_model_id != model_id:
                raise RuntimeError("请先加载当前选择的模型")
            selected_type = recognition_type or ("target" if target_only else
                "meeting" if model.recognition_types == ("meeting",) else "normal")
            if selected_type not in {"normal", "target", "meeting"}:
                raise ValueError("未知识别类型")
            if target_only and selected_type != "target":
                raise ValueError("目标过滤与所选识别类型不一致")
            target_only = selected_type == "target"
            if selected_type == "meeting" and mode != "offline":
                raise ValueError("多人会议转写仅支持非实时模式")
            if selected_type != "target" and selected_type not in model.recognition_types:
                raise ValueError("该模型不支持所选识别类型")
            if target_only:
                if mode != "streaming":
                    raise ValueError("目标说话人功能仅支持流式识别")
                if "target" not in model.recognition_types:
                    raise ValueError("目标说话人识别仅支持 FSMN-VAD + CAM++ + SenseVoice")
                if self._profile is None or self._target_models is None:
                    raise RuntimeError("请先加载声纹组件并注册目标声音")
                if not 0.1 <= speaker_threshold <= 0.95:
                    raise ValueError("声纹阈值需在 0.10–0.95 之间")
            self._target_only = bool(target_only)
            self._recognition_type = selected_type
            self._meeting_segments, self._meeting_metrics = [], {}
            self._cancel_event = Event()
            self._speaker_threshold = float(speaker_threshold)
            self._speaker_score, self._speaker_decision = None, "pending"
            self._stop_event = Event()
            self._state = "starting"
            self._message = "正在准备音频输入"
            self._error = None
            self._model_id = model_id
            self._mode = mode
            self._source = "wav" if file_audio is not None else source
            self._capture_device = ""
            self._filename = filename if file_audio is not None else ""
            self._recording_seconds = 0.0
            self._realtime_metrics = {}
            self._audio_level = 0.0
            self._recording_path = None
            self._committed = []
            self._partial = ""
            self._started_at = time.monotonic()
            self._run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            # Loading another engine must not relabel an existing transcript.
            self._run_context = {"model_id": model_id, "mode": mode,
                                 "recognition_type": selected_type, "source": self._source}
            self._output_dir = None
            self._touch()
            self._worker = Thread(target=self._run, args=(model, file_audio), daemon=True)
            self._worker.start()

    def stop(self) -> None:
        with self._lock:
            if self._recognition_type == "meeting" and self._mode == "offline" and self._state == "transcribing":
                self._cancel_event.set()
                self._state, self._message = "stopping", "正在取消会议转写，保留已保存的录音"
                self._touch()
                return
            if self._state not in {"starting", "listening", "enroll_recording"}:
                return
            if self._state == "enroll_recording":
                # "stopping" belongs to transcription: the UI would restore
                # its old mode/model/source and overwrite the user's choices.
                self._state = "enrolling"
                self._message = "正在结束注册录音并提取声纹"
            else:
                self._state = "stopping"
                self._message = "录音已停止，正在准备转写" if self._mode == "offline" else "正在停止并提交最后一段"
            self._stop_event.set()
            self._touch()

    def clear(self) -> None:
        with self._lock:
            if self._state in BUSY_STATES:
                raise RuntimeError("请先结束识别再清空文本")
            self._committed = []
            self._partial = ""
            self._meeting_segments, self._meeting_metrics = [], {}
            self._touch()

    def meeting_export_path(self, kind, run_id):
        with self._lock:
            if not self._meeting_segments or not self._output_dir or run_id != self._run_id:
                return None
            names = {"json": "meeting_segments.json", "srt": "meeting.srt"}
            return Path(self._output_dir) / names[kind] if kind in names else None

    def recording_path(self) -> Path | None:
        with self._lock:
            return self._recording_path

    def wait(self, timeout: float | None = None) -> None:
        worker = self._worker
        if worker is not None:
            worker.join(timeout)

    def handle_event(self, event: PipelineEvent) -> None:
        with self._lock:
            if event.type == EventType.VAD_START:
                self._message = "检测到语音，等待声纹验证" if self._target_only else "检测到语音"
                if self._target_only:
                    self._speaker_score, self._speaker_decision = None, "pending"
            elif event.type == EventType.ASR_PARTIAL:
                self._partial = (event.text or "").strip()
                self._message = "正在识别"
            elif event.type == EventType.CAPTION_COMMIT and event.text:
                self._committed.append(event.text.strip())
                if "meeting_segment" in event.extras:
                    self._meeting_segments.append(dict(event.extras["meeting_segment"]))
                self._partial = ""
                self._message = "正在转写 WAV 文件" if self._source == "wav" else (
                    "等待电脑播放语音" if self._source == "system" else "等待语音")
            elif event.type == EventType.SPEAKER_DECISION:
                self._speaker_score = event.similarity
                self._speaker_decision = event.speaker_decision
                self._message = {"target": "目标声音：正在转写", "non_target": "其他声音：已跳过",
                                 "pending": "证据不足：暂不转写"}.get(event.speaker_decision, "等待验证")
                if event.speaker_decision != "target":
                    self._partial = ""
            elif event.type == EventType.VAD_END and event.extras.get("awaiting_final"):
                self._message = "检测到句尾，Nano 正在定稿（请稍候）"
            elif event.type in {EventType.CAPTION_SUPPRESS, EventType.VAD_END}:
                self._partial = ""
            if event.segment_id and event.segment_id.startswith("nano-"):
                for key in ("backend_seconds", "final_delay_seconds"):
                    if key in event.extras:
                        self._realtime_metrics[key] = round(event.extras[key], 3)
                if self._state in {"stopping", "transcribing"}:
                    self._message = "音频输入已结束，正在等待 Nano 完成最后的定稿"
            self._touch()

    def _run(self, model: DictationModel, file_audio: np.ndarray | None = None) -> None:
        event_sink: CompositeEventSink | None = None
        pipeline: Any = None
        stream: Any = None
        recorded_chunks: list[np.ndarray] = []
        try:
            run_id = self._run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = (self.project_root / "outputs/phase5_16/browser_ui" / run_id).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            with self._lock:
                self._output_dir = str(output_dir)
                self._touch()

            backend = self._backend_cache[model.id]
            backend.reset()
            load_timings = dict(self._load_timings)
            decode_interval = model.decode_interval or self.decode_interval
            events_path = output_dir / "events.jsonl"
            transcript_path = output_dir / "transcript.txt"
            event_sink = CompositeEventSink(
                self,
                JsonlEventWriter(events_path),
                PlainTranscriptWriter(transcript_path),
            )
            nano_realtime = model.id == "fun-asr-nano" and self._mode == "streaming"
            pipeline = FastDictationPipeline(
                run_id=run_id,
                backend=backend,
                event_sink=event_sink,
                audio_path=self._filename or self._source,
                sample_rate=self.sample_rate,
                input_chunk=self.input_chunk,
                decode_interval=decode_interval,
                preroll=0.30,
                speech_rms_threshold=0.005,
                min_speech_duration=0.20,
                silence_duration=0.60,
                max_segment_duration=12.0,
            ) if self._mode == "streaming" and not self._target_only and not nano_realtime else None
            if nano_realtime:
                pipeline = NanoRealtimePipeline(
                    run_id=run_id, backend=backend, vad=backend.realtime_vad,
                    event_sink=event_sink, sample_rate=self.sample_rate,
                    initial_audio=1.2, decode_interval=decode_interval,
                    max_segment_duration=8.0,
                )
            if self._target_only:
                pipeline = TargetDictationPipeline(run_id=run_id, backend=backend,
                    vad=self._target_models.vad, encoder=self._target_models.encoder,
                    embedding=profile_embedding(self._profile), event_sink=event_sink,
                    threshold=self._speaker_threshold, sample_rate=self.sample_rate)
            if self._stop_event.is_set():
                with self._lock:
                    self._state = "idle"
                    self._message = "已取消录音"
                    self._started_at = None
                return
            if file_audio is None:
                stream = (self._system_stream_factory(self._stop_event) if self._source == "system"
                          else self._stream_factory(self._stop_event))
            else:
                blocksize = max(1, int(self.input_chunk * self.sample_rate))
                def file_chunks():
                    origin = time.monotonic()
                    for offset in range(0, len(file_audio), blocksize):
                        end = min(len(file_audio), offset + blocksize)
                        if nano_realtime:
                            # Match live capture, with a stop-responsive wait.
                            delay = max(0.0, origin + end / self.sample_rate - time.monotonic())
                            if self._stop_event.wait(delay):
                                break
                        yield offset / self.sample_rate, file_audio[offset:end]
                stream = file_chunks()
            with self._lock:
                if self._state != "stopping":
                    self._state = "transcribing" if file_audio is not None and not nano_realtime else "listening"
                    self._message = ("正在按实时速度输入 WAV，Nano 预览文字可修订" if nano_realtime and file_audio is not None else
                        "正在转写 WAV 文件") if file_audio is not None else (
                        ("正在采集电脑音频，停止后转写" if self._mode == "offline" else "正在采集电脑音频，请播放需要识别的声音")
                        if self._source == "system" else
                        "正在录音，停止后转写" if self._mode == "offline" else "正在聆听，实时转写中")
                    self._started_at = time.monotonic()
                    self._touch()

            for _, chunk in stream:
                if (self._target_only or nano_realtime or self._recognition_type == "meeting") and any(
                    "overflow" in item for item in getattr(stream, "statuses", [])):
                    raise RuntimeError("音频采集积压丢帧，转写已停止，请降低系统负载后重试")
                recorded_chunks.append(np.asarray(chunk, dtype=np.float32).copy())
                with self._lock:
                    self._recording_seconds += len(chunk) / self.sample_rate
                    self._audio_level = min(1.0, float(np.sqrt(np.mean(np.square(chunk)))) * 8)
                    if self._source == "system":
                        self._capture_device = getattr(stream, "device_name", "默认电脑音频输出")
                if pipeline is not None:
                    pipeline.push_chunk(chunk)
                if file_audio is None and self._recording_seconds >= self.max_recording_seconds:
                    self._stop_event.set()
                    break
            if hasattr(stream, "close"):
                stream.close()

            recording_path = output_dir / "recording.wav"
            if recorded_chunks:
                import soundfile as sf

                sf.write(
                    recording_path,
                    np.concatenate(recorded_chunks),
                    self.sample_rate,
                    subtype="PCM_16",
                )
                with self._lock:
                    self._recording_path = recording_path
                    self._audio_level = 0.0
            if pipeline is not None:
                if nano_realtime:
                    with self._lock:
                        self._state, self._message = "transcribing", "输入已结束，正在等待 Nano 句尾定稿，请勿关闭服务"
                        self._touch()
                stats_dict = asdict(pipeline.finish())
                if nano_realtime:
                    with self._lock:
                        self._realtime_metrics.update({key: stats_dict[key] for key in
                            ("first_text_seconds", "last_final_delay_seconds", "coalesced_previews", "reused_finals")})
            elif self._recognition_type == "meeting":
                if not recorded_chunks:
                    raise RuntimeError("未收到会议音频，请重新开始录音")
                with self._lock:
                    self._state, self._message = "transcribing", "正在整段分析会议录音并区分说话人…"
                    self._touch()
                def meeting_progress(message):
                    with self._lock:
                        self._meeting_metrics.update({key: message[key] for key in ("stage", "generated_tokens") if key in message})
                        if not self._cancel_event.is_set():
                            self._message = ("正在编码整段会议音频…" if message.get("stage") == "encoding"
                                else f"正在生成会议转写，已生成 {message.get('generated_tokens', 0)} 个文本 token（非完成百分比）")
                        self._touch()
                result = backend.transcribe_file(recording_path, progress=meeting_progress, cancel_event=self._cancel_event)
                (output_dir / "moss_raw.txt").write_text(result.get("raw_text", ""), encoding="utf-8")
                duration = sum(len(chunk) for chunk in recorded_chunks) / self.sample_rate
                meeting_segments = validate_segments(result["segments"], duration)
                if result.get("raw_text", "").strip() and not meeting_segments:
                    raise RuntimeError("MOSS 未返回可解析的会议分段；原始结果保存在 moss_raw.txt，请重试")
                (output_dir / "meeting_segments.json").write_text(json.dumps({
                    "model_id": model.id, "model_revision": MODEL_REVISION,
                    "duration_seconds": duration, "speaker_labels": "anonymous_per_recording",
                    "truncated": result.get("truncated", False), "segments": meeting_segments,
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                (output_dir / "meeting.srt").write_text(segments_srt(meeting_segments), encoding="utf-8")
                for index, segment in enumerate(meeting_segments):
                    event_sink.write(PipelineEvent(type=EventType.CAPTION_COMMIT, run_id=run_id,
                        segment_id=f"meeting-{index}", text=segment_text(segment),
                        start=segment["start"], end=segment["end"], extras={"meeting_segment": segment}))
                with self._lock:
                    self._meeting_metrics.update({key: result[key] for key in
                        ("backend_seconds", "generated_tokens", "prompt_tokens", "token_limit", "truncated") if key in result})
                    self._meeting_metrics["stage"] = "complete"
                stats_dict = {"accepted_seconds": duration, "committed_segments": len(meeting_segments),
                              "speakers": len({segment["speaker"] for segment in meeting_segments}),
                              **self._meeting_metrics}
                if result.get("truncated"):
                    raise RuntimeError("会议转写达到文本长度上限，结果可能不完整；已保存部分结果和原音频，请缩短录音后重试")
            else:
                with self._lock:
                    self._state = "transcribing"
                    self._message = "正在转写音频，请稍候"
                    self._touch()
                audio = np.concatenate(recorded_chunks) if recorded_chunks else np.empty(0)
                # Bound LLM memory usage and skip digital silence.
                window = self.sample_rate * 30
                segments = 0
                for offset in range(0, len(audio), window):
                    part = audio[offset:offset + window]
                    if float(np.sqrt(np.mean(part ** 2))) < 0.001:
                        continue
                    text = backend.transcribe(part)
                    if text and any(char.isalnum() for char in text):
                        event_sink.write(PipelineEvent(
                            type=EventType.CAPTION_COMMIT, run_id=run_id, text=text,
                            start=offset / self.sample_rate,
                            end=(offset + len(part)) / self.sample_rate,
                        ))
                        segments += 1
                stats_dict = {"accepted_seconds": len(audio) / self.sample_rate,
                              "committed_segments": segments}
            statuses = list(getattr(stream, "statuses", []))
            write_status_file(output_dir / "microphone_status.txt", statuses)
            summary = {
                "schema_version": 1,
                "phase": "5.17",
                "run_id": run_id,
                "model_id": model.id,
                "mode": self._mode,
                "source": self._source,
                "capture_device": self._capture_device,
                "capture_native_sample_rate": getattr(stream, "native_sample_rate", None),
                "capture_native_channels": getattr(stream, "native_channels", None),
                "filename": self._filename,
                "target_only": self._target_only,
                "recognition_type": self._recognition_type,
                "speaker_name": self._profile.speaker_id if self._target_only else None,
                "speaker_threshold": self._speaker_threshold if self._target_only else None,
                "model_dir": str(model.model_dir.resolve()),
                "itn_enabled": True,
                "parameters": {
                    "sample_rate": self.sample_rate,
                    "input_chunk": self.input_chunk,
                    "decode_interval": decode_interval,
                    "intra_op_threads": self.intra_op_threads,
                    **({"model_revision": MODEL_REVISION, "runtime": getattr(backend, "runtime_info", {}),
                        "whole_recording": True, "max_audio_seconds": MAX_WAV_SECONDS}
                       if self._recognition_type == "meeting" else {}),
                    **({"target_decode_strategy": "verified_prefix",
                        "speaker_window_seconds": 1.5, "speaker_min_window_seconds": 0.6,
                        "vad_tail_hold_seconds": 0.3, "max_verified_segment_seconds": 12.0}
                       if self._target_only else {}),
                    **({"initial_audio": 1.2, "vad_end_silence_ms": 600,
                        "max_segment_seconds": 8.0, "max_final_backlog_seconds": 30.0,
                        "llm_dtype": "fp32", "file_realtime_replay": file_audio is not None,
                        "ui_poll_interval_ms": 350} if nano_realtime else {}),
                },
                "stats": stats_dict,
                "load_timings": load_timings,
                "recording_path": str(recording_path) if recorded_chunks else None,
                "transcript_path": str(transcript_path),
                "events_path": str(events_path),
                "microphone_statuses": statuses,
            }
            (output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self._lock:
                self._state = "idle"
                self._message = "转写完成，音频与文本已保存" if self._committed else (
                    "音频处理结束，没有通过声纹验证的可转写语音" if self._target_only else "音频处理结束，未识别到语音")
                self._partial = ""
                self._started_at = None
                self._touch()
        except Exception as exc:
            # Do not advertise an idle/error state until the Nano worker has
            # released the backend; a retry must not race an old GPU decode.
            if isinstance(pipeline, NanoRealtimePipeline):
                self._stop_event.set()
                pipeline.close()
            if self._source == "system" or self._target_only or self._recognition_type == "meeting":
                self._stop_event.set()
                if hasattr(stream, "close"):
                    stream.close()
            if (isinstance(pipeline, NanoRealtimePipeline) or self._source == "system"
                    or self._target_only or self._recognition_type == "meeting"):
                # Preserve accepted audio for diagnosis/retry if a decode or
                # overload fails before the normal recording-save step.
                if recorded_chunks and self._recording_path is None:
                    try:
                        import soundfile as sf

                        recovery_path = output_dir / "recording.wav"
                        sf.write(recovery_path, np.concatenate(recorded_chunks), self.sample_rate, subtype="PCM_16")
                        with self._lock:
                            self._recording_path = recovery_path
                    except Exception:
                        pass  # Report the original recognition failure.
            with self._lock:
                cancelled = isinstance(exc, MeetingCancelled)
                self._state = "idle" if cancelled else "error"
                self._message = str(exc) if cancelled else "识别发生错误"
                self._error = None if cancelled else str(exc)
                if getattr(self._backend_cache.get(model.id), "closed", False):
                    self._loaded_model_id = None
                self._partial = ""
                self._audio_level = 0.0
                self._started_at = None
                self._touch()
        finally:
            if hasattr(stream, "close"):
                stream.close()
            if isinstance(pipeline, NanoRealtimePipeline):
                pipeline.close()
            if event_sink is not None:
                event_sink.close()

    def write(self, event: PipelineEvent) -> None:
        self.handle_event(event)


def microphone_name() -> str:
    try:
        import sounddevice as sd

        device = sd.query_devices(kind="input")
        return str(device.get("name", "默认麦克风"))
    except Exception:
        return "默认麦克风"


def make_handler(controller: DictationController, web_root: Path, rhine_root: Path | None = None) -> type[BaseHTTPRequestHandler]:
    root = web_root.resolve()
    samplex = SampleXService(root.parent.parent.parent, controller)
    speech_root = rhine_root.resolve() if rhine_root is not None else None

    class DictationHandler(BaseHTTPRequestHandler):
        server_version = "LocalDictation/1.0"

        def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _body(self) -> dict[str, Any]:
            length = min(int(self.headers.get("Content-Length", "0")), 65536)
            if not length:
                return {}
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("JSON body must be an object")
            return value

        def _static(self, request_path: str) -> None:
            selected_root = root
            if request_path.startswith('/rhine/'):
                if speech_root is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                selected_root = speech_root
                request_path = request_path[len('/rhine/'):]
            relative = unquote(request_path).lstrip("/") or "index.html"
            candidate = (selected_root / relative).resolve()
            if candidate != selected_root and selected_root not in candidate.parents:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not candidate.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = candidate.read_bytes()
            mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{mime}; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == '/rhine':
                self.send_response(HTTPStatus.PERMANENT_REDIRECT)
                self.send_header('Location', '/rhine/')
                self.send_header('Content-Length', '0')
                self.end_headers()
            elif path == "/api/status":
                self._json({**controller.snapshot(), 'external_busy': samplex.occupied()})
            elif path == "/api/sample-x/status":
                self._json(samplex.status())
            elif path == "/api/models":
                self._json({"models": controller.model_list()})
            elif path == "/api/device":
                source = parse_qs(urlparse(self.path).query).get("source", ["microphone"])[0]
                self._json(system_audio_device() if source == "system" else {"name": microphone_name(), "available": True})
            elif path == "/api/recording":
                recording = controller.recording_path()
                if recording is None or not recording.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                payload = recording.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            elif path == "/api/meeting-export":
                query = parse_qs(urlparse(self.path).query)
                kind = query.get("format", [""])[0]
                export = controller.meeting_export_path(kind, query.get("run", [""])[0])
                if export is None or not export.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                payload = export.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8" if kind == "json" else "application/x-subrip; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{export.name}"')
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            else:
                self._static(path)

        def do_POST(self) -> None:  # noqa: N802
            with samplex.lock:
                self._post()

        def _post(self) -> None:
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self._json({"error": "不允许跨站控制本地麦克风"}, HTTPStatus.FORBIDDEN)
                return
            path = urlparse(self.path).path
            try:
                if path == '/api/sample-x/claim':
                    samplex.claim(str(self._body().get('token', '')))
                    self._json({'ok': True})
                    return
                if path == '/api/sample-x/release':
                    samplex.release(str(self._body().get('token', '')))
                    self._json({'ok': True})
                    return
                if samplex.occupied():
                    raise RuntimeError('样品-X档案正在识别，请先结束并定稿')
                if path == '/api/sample-x/load':
                    self._body()
                    samplex.port = self.server.server_port
                    self._json(samplex.start(), HTTPStatus.ACCEPTED)
                elif path == "/api/load-model":
                    body = self._body()
                    controller.load_model(str(body.get("model_id", "")), str(body.get("mode", "streaming")))
                    self._json(controller.snapshot(), HTTPStatus.ACCEPTED)
                elif path == "/api/target/load":
                    self._body()
                    controller.load_target()
                    self._json(controller.snapshot(), HTTPStatus.ACCEPTED)
                elif path == "/api/target/enroll-microphone":
                    body = self._body()
                    controller.enroll(str(body.get("name", "")))
                    self._json(controller.snapshot(), HTTPStatus.ACCEPTED)
                elif path == "/api/target/enroll-recording":
                    body = self._body()
                    controller.enroll(str(body.get("name", "")), source=str(body.get("source", "microphone")))
                    self._json(controller.snapshot(), HTTPStatus.ACCEPTED)
                elif path == "/api/target/forget":
                    self._body()
                    controller.forget_speaker()
                    self._json(controller.snapshot())
                elif path == "/api/target/enroll-file":
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= MAX_WAV_BYTES:
                        raise ValueError("注册 WAV 文件大小无效")
                    query = parse_qs(urlparse(self.path).query)
                    controller.enroll(query.get("name", [""])[0],
                                      decode_wav(self.rfile.read(length), controller.sample_rate))
                    self._json(controller.snapshot(), HTTPStatus.ACCEPTED)
                elif path == "/api/transcribe-file":
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= MAX_WAV_BYTES:
                        raise ValueError("WAV 文件必须非空且不超过 64 MB")
                    query = parse_qs(urlparse(self.path).query)
                    filename = query.get("filename", ["audio.wav"])[0]
                    if not filename.lower().endswith(".wav"):
                        raise ValueError("仅支持单个 .wav 文件")
                    audio = decode_wav(self.rfile.read(length), controller.sample_rate)
                    controller.start(query.get("model_id", [""])[0],
                                     query.get("mode", ["offline"])[0],
                                     file_audio=audio, filename=filename,
                                     target_only=query.get("target_only", ["false"])[0] == "true",
                                     recognition_type=query.get("recognition_type", [None])[0],
                                     speaker_threshold=float(query.get("speaker_threshold", ["0.45"])[0]))
                    self._json(controller.snapshot(), HTTPStatus.ACCEPTED)
                elif path == "/api/start":
                    body = self._body()
                    controller.start(str(body.get("model_id", "")), str(body.get("mode", "streaming")),
                                     target_only=body.get("target_only") is True,
                                     recognition_type=body.get("recognition_type"),
                                     speaker_threshold=float(body.get("speaker_threshold", 0.45)),
                                     source=str(body.get("source", "microphone")))
                    self._json(controller.snapshot(), HTTPStatus.ACCEPTED)
                elif path == "/api/stop":
                    controller.stop()
                    self._json(controller.snapshot(), HTTPStatus.ACCEPTED)
                elif path == "/api/clear":
                    controller.clear()
                    self._json(controller.snapshot())
                elif path == "/api/shutdown":
                    controller.stop()
                    self._json({"ok": True})
                    Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, FileNotFoundError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return DictationHandler


def serve(
    *,
    project_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    controller = DictationController(project_root=project_root)
    web_root = project_root / "web/dictation"
    rhine_root = project_root.parent / 'RhineLabUI' / 'dist'
    server = ThreadingHTTPServer((host, port), make_handler(controller, web_root, rhine_root))
    url = f"http://{host}:{port}/"
    if open_browser:
        Thread(target=lambda: (time.sleep(0.35), webbrowser.open(url)), daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        controller.shutdown()
        server.server_close()
