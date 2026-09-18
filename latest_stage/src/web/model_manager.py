"""Model inventory and background install/uninstall operations."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import sys
import time
from threading import Lock, Thread
from typing import Callable, Sequence

from ..models.storage import model_directory, model_storage_root


BUSY_STATES = {"loading", "starting", "listening", "stopping", "transcribing", "enrolling", "enroll_recording"}


@dataclass(frozen=True)
class ManagedModel:
    id: str
    name: str
    description: str
    features: tuple[str, ...]
    source: str
    source_url: str
    license: str
    device: str
    estimated_size: str
    install_mode: str = "auto"
    modelscope_id: str | None = None
    local_directory: Path | None = None
    markers: tuple[Path, ...] = ()
    install_paths: tuple[Path, ...] = ()
    manual: str | None = None

    def public_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "features": list(self.features),
            "source": self.source,
            "source_url": self.source_url,
            "license": self.license,
            "device": self.device,
            "estimated_size": self.estimated_size,
            "install_mode": self.install_mode,
            "manual": self.manual,
        }


@dataclass
class ManagerOperation:
    id: str
    action: str
    state: str = "running"
    message: str = ""
    started_at: float = field(default_factory=time.time)
    log: list[str] = field(default_factory=list)
    error: str | None = None

    def public_dict(self) -> dict:
        return {
            "id": self.id,
            "action": self.action,
            "state": self.state,
            "message": self.message,
            "started_at": self.started_at,
            "log": list(self.log),
            "error": self.error,
        }


def default_managed_models(project_root: Path) -> list[ManagedModel]:
    def local(name: str) -> Path:
        return model_directory(project_root, name)

    return [
        ManagedModel(
            id="sensevoice-realtime",
            name="SenseVoice 实时引擎",
            description="从 SenseVoiceSmall 导出 ONNX，并生成 INT8 本地推理包；用于 CPU 实时普通识别与目标说话人识别。",
            features=("X-005 实时识别", "X-007 目标说话人识别"),
            source="ModelScope iic/SenseVoiceSmall，本地导出 ONNX INT8",
            source_url="https://modelscope.cn/models/iic/SenseVoiceSmall",
            license="FunASR 生态模型",
            device="CPU",
            estimated_size="约 230 MB 生成物；构建时另需约 900 MB 源模型缓存",
            markers=(local("sensevoice-onnx-int8") / "model.onnx",),
            install_paths=(
                local("sensevoice-onnx"),
                local("sensevoice-onnx-int8"),
            ),
        ),
        ManagedModel(
            id="sensevoice-small",
            name="SenseVoice Small",
            description="非实时中英等多语言识别模型；也是构建实时 ONNX 引擎的源模型。",
            features=("X-001 非实时识别", "X-005 实时引擎构建依赖"),
            source="ModelScope iic/SenseVoiceSmall",
            source_url="https://modelscope.cn/models/iic/SenseVoiceSmall",
            license="FunASR 生态模型",
            device="GPU 优先，可回退 CPU",
            estimated_size="约 900 MB",
            modelscope_id="iic/SenseVoiceSmall",
            local_directory=local("sensevoice-small"),
            markers=(local("sensevoice-small") / "model.pt",),
            install_paths=(local("sensevoice-small"),),
        ),
        ManagedModel(
            id="fsmn-vad",
            name="FSMN-VAD",
            description="语音活动检测模型，为实时分段、目标说话人窗口和 Fun-ASR-Nano 准实时模式划分语音边界。",
            features=("X-006 实时识别", "X-007 目标说话人识别", "X-009 声纹注册"),
            source="ModelScope iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            source_url="https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            license="FunASR 生态模型",
            device="CPU",
            estimated_size="轻量模型",
            modelscope_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            local_directory=local("fsmn-vad"),
            markers=(local("fsmn-vad") / "model.pt",),
            install_paths=(local("fsmn-vad"),),
        ),
        ManagedModel(
            id="cam-plus",
            name="CAM++ 声纹模型",
            description="从注册语音中提取说话人特征，并为目标识别提供相似度验证。",
            features=("X-007 目标说话人识别", "X-009 声纹注册"),
            source="ModelScope iic/speech_campplus_sv_zh-cn_16k-common",
            source_url="https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common",
            license="FunASR 生态模型",
            device="CPU",
            estimated_size="轻量模型",
            modelscope_id="iic/speech_campplus_sv_zh-cn_16k-common",
            local_directory=local("cam-plus"),
            markers=(local("cam-plus") / "campplus_cn_common.bin",),
            install_paths=(local("cam-plus"),),
        ),
        ManagedModel(
            id="fun-asr-nano",
            name="Fun-ASR-Nano",
            description="FunAudioLLM Nano 识别模型，支持中英日及中文方言；同一模型可用于整段识别和准实时识别。",
            features=("X-002 非实时识别", "X-006 实时识别"),
            source="ModelScope FunAudioLLM/Fun-ASR-Nano-2512",
            source_url="https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512",
            license="FunAudioLLM 模型",
            device="GPU 优先，可回退 CPU",
            estimated_size="约 2 GB 级",
            modelscope_id="FunAudioLLM/Fun-ASR-Nano-2512",
            local_directory=local("fun-asr-nano"),
            markers=(local("fun-asr-nano") / "model.pt",),
            install_paths=(local("fun-asr-nano"),),
        ),
        ManagedModel(
            id="qwen3-asr",
            name="Qwen3-ASR 1.7B",
            description="通义千问 1.7B 多语言 ASR 模型，整段识别并自动检测语言；首次加载和显存占用较高。",
            features=("X-003 非实时识别",),
            source="ModelScope Qwen/Qwen3-ASR-1.7B",
            source_url="https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B",
            license="通义千问 ASR 模型",
            device="GPU 优先，可回退 CPU",
            estimated_size="约 4 GB 级",
            modelscope_id="Qwen/Qwen3-ASR-1.7B",
            local_directory=local("qwen3-asr"),
            markers=(local("qwen3-asr") / "model.safetensors.index.json",),
            install_paths=(local("qwen3-asr"),),
        ),
        ManagedModel(
            id="moss-transcribe-diarize",
            name="MOSS-Transcribe-Diarize",
            description="多人会议联合转写模型，生成文字、匿名说话人编号和时间戳；使用独立运行环境避免依赖冲突。",
            features=("X-008 会议转写",),
            source="Hugging Face OpenMOSS-Team/MOSS-Transcribe-Diarize（固定 revision）",
            source_url="https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize",
            license="OpenMOSS 模型",
            device="NVIDIA GPU",
            estimated_size="模型约 2 GB；独立 CUDA 环境另需数 GB",
            markers=(
                local("moss-transcribe-diarize") / "model-00000-of-00001.safetensors",
                project_root / ".venv-moss/Scripts/python.exe",
            ),
            install_paths=(
                local("moss-transcribe-diarize"),
                project_root / ".venv-moss",
                project_root / ".cache/moss-wheels",
            ),
        ),
        ManagedModel(
            id="sample-x",
            name="样品-X",
            description="X-010 使用的本地准实时识别引擎。推理资产不随公开仓库分发，需要使用者自行准备后创建独立环境。",
            features=("X-010 实时识别",),
            source="使用者本地提供",
            source_url="https://github.com/wdawdaw454/rhine-speech-archive/tree/main/sample-x",
            license="按使用者本地模型授权",
            device="CUDA 优先，CPU 回退",
            estimated_size="由本地模型资产决定",
            install_mode="manual",
            markers=(
                local("sample-x") / "portable-models/no_stream/decoder_full.mnn",
                project_root.parent / "sample-x/.venv/Scripts/python.exe",
            ),
            install_paths=(
                local("sample-x") / "portable-models",
                local("sample-x") / "decoded/asr",
                local("sample-x") / "cuda-graphs",
                project_root.parent / "sample-x/.venv",
                project_root.parent / "sample-x/.venv-cuda",
            ),
            manual="请先按 sample-x/README.md 准备 models/sample-x/portable-models 与 models/sample-x/decoded/asr，再运行 sample-x/setup.ps1；本页可卸载已放入的资产和独立环境。",
        ),
        ManagedModel(
            id="silero-vad",
            name="Silero VAD",
            description="样品-X档案的本地端点检测模型，用于切分完整语音段并决定何时定稿。",
            features=("X-010 实时识别",),
            source="GitHub snakers4/silero-vad v6.2.1",
            source_url="https://github.com/snakers4/silero-vad",
            license="MIT",
            device="CPU",
            estimated_size="约 2.2 MB",
            install_mode="bundled",
            markers=(local("silero-vad") / "silero_vad.onnx",),
        ),
        ManagedModel(
            id="punctuation",
            name="标点恢复模型",
            description="旧版命令行目标识别流水线使用的中文/英文标点恢复模型；当前网页转写流水线不强制依赖。",
            features=("旧版 CLI 流水线",),
            source="ModelScope iic/punc_ct-transformer_cn-en-common-vocab471067-large",
            source_url="https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large",
            license="FunASR 生态模型",
            device="CPU",
            estimated_size="约 300 MB 级",
            modelscope_id="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
            local_directory=local("punctuation"),
            markers=(local("punctuation") / "model.pt",),
            install_paths=(local("punctuation"),),
        ),
    ]


class ModelManager:
    def __init__(
        self,
        *,
        project_root: Path,
        controller,
        specs: Sequence[ManagedModel] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        external_busy: Callable[[], bool] | None = None,
        external_lock=None,
    ) -> None:
        self.project_root = project_root.resolve()
        self._controller = controller
        self._specs = list(specs or default_managed_models(self.project_root))
        self._by_id = {spec.id: spec for spec in self._specs}
        self._lock = Lock()
        self._operation: ManagerOperation | None = None
        self._runner = runner or self._run
        self._external_busy = external_busy or (lambda: False)
        self._external_lock = external_lock

    def status(self) -> dict:
        with self._lock:
            operation = self._operation.public_dict() if self._operation else None
        models = []
        for spec in self._specs:
            installed = all(marker.is_file() for marker in spec.markers)
            models.append({
                **spec.public_dict(),
                "installed": installed,
            })
        return {"models": models, "operation": operation}

    def install(self, model_id: str) -> dict:
        return self._start(model_id, "install")

    def uninstall(self, model_id: str) -> dict:
        return self._start(model_id, "uninstall")

    def _start(self, model_id: str, action: str) -> dict:
        spec = self._by_id.get(model_id)
        if spec is None:
            raise ValueError("未知模型")
        if action not in {"install", "uninstall"}:
            raise ValueError("未知模型操作")
        if action == "install" and spec.install_mode != "auto":
            raise RuntimeError(spec.manual or "该模型无需在线安装")
        if action == "uninstall" and spec.install_mode == "bundled":
            raise RuntimeError("该模型随仓库分发，不能在网页中卸载")
        controller_state = self._controller.snapshot()
        if controller_state["state"] in BUSY_STATES:
            raise RuntimeError("请先结束当前识别任务")
        if self._external_busy():
            raise RuntimeError("样品-X档案正在识别，请先结束并定稿")
        with self._lock:
            if self._operation and self._operation.state == "running":
                raise RuntimeError("已有模型操作正在进行")
            self._operation = ManagerOperation(
                id=model_id,
                action=action,
                message="准备卸载模型…" if action == "uninstall" else "准备安装模型…",
            )
            operation = self._operation
        Thread(target=self._worker, args=(spec, operation), daemon=True).start()
        return self.status()

    def _worker(self, spec: ManagedModel, operation: ManagerOperation) -> None:
        if self._external_lock is None:
            self._run_operation(spec, operation)
            return
        with self._external_lock:
            self._run_operation(spec, operation)

    def _run_operation(self, spec: ManagedModel, operation: ManagerOperation) -> None:
        try:
            if self._external_busy():
                raise RuntimeError("样品-X档案正在识别，请先结束并定稿")
            self._controller.release_models()
            if operation.action == "uninstall":
                self._set(operation, message="正在删除模型文件…")
                for path in spec.install_paths:
                    self._remove(path)
                self._set(operation, state="complete", message=f"{spec.name} 已卸载")
                return
            self._set(operation, message=f"正在安装 {spec.name}…")
            if spec.id == "sensevoice-realtime":
                source = self._install_modelscope(
                    "iic/SenseVoiceSmall", model_directory(self.project_root, "sensevoice-small")
                )
                onnx = model_directory(self.project_root, "sensevoice-onnx")
                int8 = model_directory(self.project_root, "sensevoice-onnx-int8")
                if not (onnx / "model.onnx").is_file():
                    self._command(operation, [
                        sys.executable,
                        str(self.project_root / "scripts/export_sensevoice_onnx.py"),
                        "--source-model-dir", source,
                        "--output-dir", str(onnx),
                    ])
                if not (int8 / "model.onnx").is_file():
                    self._command(operation, [
                        sys.executable,
                        str(self.project_root / "scripts/quantize_sensevoice_onnx.py"),
                        "--bundle-dir", str(onnx),
                        "--output-dir", str(int8),
                    ])
            elif spec.id == "moss-transcribe-diarize":
                self._command(operation, [
                    "powershell",
                    "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", str(self.project_root / "scripts/setup_moss_windows.ps1"),
                ])
            else:
                self._install_modelscope(spec.modelscope_id, spec.local_directory)
            if not all(marker.is_file() for marker in spec.markers):
                raise RuntimeError("安装命令已结束，但未找到预期的模型文件")
            self._set(operation, state="complete", message=f"{spec.name} 已就绪")
        except Exception as exc:
            self._set(operation, state="error", error=str(exc), message=f"{spec.name} 操作失败")
        finally:
            self._controller.refresh_model_availability()

    def _install_modelscope(self, model_id: str | None, target: Path | None) -> str:
        if not model_id:
            raise ValueError("缺少 ModelScope 模型 ID")
        if target is None:
            raise ValueError("缺少 ModelScope 本地模型目录")
        code = (
            "from modelscope import snapshot_download; "
            f"print(snapshot_download({model_id!r}, local_dir={str(target)!r}))"
        )
        result = self._runner([sys.executable, "-c", code], str(self.project_root))
        if result.returncode != 0:
            raise RuntimeError(result.stdout.strip() or f"ModelScope 模型下载失败：{model_id}")
        return result.stdout.strip().splitlines()[-1]

    def _command(self, operation: ManagerOperation, command: list[str]) -> None:
        result = self._runner(command, str(self.project_root))
        if result.returncode != 0:
            raise RuntimeError(result.stdout.strip() or "模型安装命令失败")
        self._log(operation, result.stdout.strip())

    def _run(self, command: list[str], cwd: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def _remove(self, path: Path) -> None:
        resolved = path.resolve()
        allowed = [entry.resolve() for entry in (
            model_storage_root(self.project_root),
            self.project_root / ".venv-moss",
            self.project_root / ".cache/moss-wheels",
            model_directory(self.project_root, "sample-x") / "portable-models",
            model_directory(self.project_root, "sample-x") / "decoded/asr",
            model_directory(self.project_root, "sample-x") / "cuda-graphs",
            self.project_root.parent / "sample-x/.venv",
            self.project_root.parent / "sample-x/.venv-cuda",
        )]
        if not any(resolved == root or root in resolved.parents for root in allowed):
            raise ValueError(f"拒绝删除未声明的路径：{resolved}")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        elif resolved.exists():
            resolved.unlink()

    def _log(self, operation: ManagerOperation, text: str) -> None:
        if not text:
            return
        with self._lock:
            lines = [line for line in text.splitlines() if line.strip()]
            operation.log.extend(lines[-80:])

    def _set(self, operation: ManagerOperation, **values) -> None:
        with self._lock:
            for key, value in values.items():
                setattr(operation, key, value)
