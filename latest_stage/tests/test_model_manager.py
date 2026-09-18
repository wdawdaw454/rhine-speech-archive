from __future__ import annotations

from pathlib import Path
import subprocess
import time

import pytest

from src.web.model_manager import ManagedModel, ModelManager, default_managed_models


class FakeController:
    def __init__(self, state: str = "idle") -> None:
        self.state = state
        self.released = 0
        self.refreshed = 0

    def snapshot(self) -> dict:
        return {"state": self.state}

    def release_models(self) -> None:
        self.released += 1

    def refresh_model_availability(self) -> None:
        self.refreshed += 1


def _spec(tmp_path: Path, *, install_mode: str = "auto") -> ManagedModel:
    model_root = tmp_path / "models"
    marker = model_root / "model.bin"
    return ManagedModel(
        id="test-model",
        name="Test Model",
        description="test model",
        features=("X-000 测试",),
        source="test source",
        source_url="https://example.invalid/model",
        license="test license",
        device="CPU",
        estimated_size="small",
        install_mode=install_mode,
        manual="请手动准备该模型" if install_mode == "manual" else None,
        modelscope_id="example/model",
        local_directory=tmp_path.parent / "models/test-model",
        markers=(marker,),
        install_paths=(model_root,),
    )


def _wait(manager: ModelManager, state: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        operation = manager.status()["operation"]
        if operation and operation["state"] == state:
            return operation
        time.sleep(0.01)
    raise AssertionError(f"model operation did not reach {state}")


def test_registry_reports_models_when_cache_is_empty(tmp_path, monkeypatch):
    empty_home = tmp_path / "home"
    empty_home.mkdir()
    monkeypatch.setenv("HOME", str(empty_home))
    monkeypatch.setenv("USERPROFILE", str(empty_home))

    status = ModelManager(
        project_root=tmp_path,
        controller=FakeController(),
        specs=default_managed_models(tmp_path),
    ).status()

    assert [model["id"] for model in status["models"]] == [
        "sensevoice-realtime",
        "sensevoice-small",
        "fsmn-vad",
        "cam-plus",
        "fun-asr-nano",
        "qwen3-asr",
        "moss-transcribe-diarize",
        "sample-x",
        "silero-vad",
        "punctuation",
    ]
    assert all(model["installed"] is False for model in status["models"])
    assert status["operation"] is None
    assert all("markers" not in model and "install_paths" not in model for model in status["models"])
    assert next(model for model in status["models"] if model["id"] == "sample-x")["install_mode"] == "manual"
    assert next(model for model in status["models"] if model["id"] == "silero-vad")["install_mode"] == "bundled"


def test_modelscope_markers_use_repository_model_root(tmp_path, monkeypatch):
    empty_home = tmp_path / "home"
    empty_home.mkdir()
    monkeypatch.setenv("HOME", str(empty_home))
    monkeypatch.setenv("USERPROFILE", str(empty_home))

    expected_files = {
        "sensevoice-small": ("sensevoice-small", "model.pt"),
        "fsmn-vad": ("fsmn-vad", "model.pt"),
        "cam-plus": ("cam-plus", "campplus_cn_common.bin"),
        "fun-asr-nano": ("fun-asr-nano", "model.pt"),
        "qwen3-asr": ("qwen3-asr", "model.safetensors.index.json"),
        "punctuation": ("punctuation", "model.pt"),
    }
    specs = default_managed_models(tmp_path)

    for model_id, (repository, filename) in expected_files.items():
        spec = next(item for item in specs if item.id == model_id)
        expected = tmp_path.parent / "models" / repository / filename
        assert spec.markers == (expected,)


def test_legacy_modelscope_home_cache_is_not_reported_installed(tmp_path, monkeypatch):
    legacy_home = tmp_path / "home"
    legacy_home.mkdir()
    monkeypatch.setenv("HOME", str(legacy_home))
    monkeypatch.setenv("USERPROFILE", str(legacy_home))
    legacy = legacy_home / ".cache/modelscope/models/legacy/snapshots/master"
    legacy.mkdir(parents=True)
    (legacy / "model.pt").write_text("legacy cache", encoding="utf-8")

    specs = default_managed_models(tmp_path)
    assert all(not all(marker.is_file() for marker in spec.markers) for spec in specs)


def test_install_runs_in_background_and_refreshes_controller(tmp_path):
    controller = FakeController()
    spec = _spec(tmp_path)

    def runner(command, _cwd):
        spec.markers[0].parent.mkdir(parents=True, exist_ok=True)
        spec.markers[0].write_text("weights", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "download complete", "")

    manager = ModelManager(project_root=tmp_path, controller=controller, specs=[spec], runner=runner)
    manager.install(spec.id)
    operation = _wait(manager, "complete")

    assert operation["started_at"] > 0
    assert operation["error"] is None
    assert manager.status()["models"][0]["installed"] is True
    assert controller.released == 1
    assert controller.refreshed == 1


def test_modelscope_installer_targets_repository_model_root(tmp_path):
    controller = FakeController()
    spec = _spec(tmp_path)
    commands = []

    def runner(command, _cwd):
        commands.append(command)
        spec.markers[0].parent.mkdir(parents=True, exist_ok=True)
        spec.markers[0].write_text("weights", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "download complete", "")

    manager = ModelManager(project_root=tmp_path, controller=controller, specs=[spec], runner=runner)
    manager.install(spec.id)
    _wait(manager, "complete")

    installer = commands[0][commands[0].index("-c") + 1]
    assert f"local_dir={str(spec.local_directory)!r}" in installer


def test_manual_model_cannot_be_automatically_installed(tmp_path):
    controller = FakeController()
    spec = _spec(tmp_path, install_mode="manual")
    manager = ModelManager(project_root=tmp_path, controller=controller, specs=[spec])

    with pytest.raises(RuntimeError, match="手动"):
        manager.install(spec.id)
    assert manager.status()["operation"] is None


def test_controller_or_external_task_blocks_model_changes(tmp_path):
    spec = _spec(tmp_path)
    busy_controller = FakeController("listening")
    manager = ModelManager(project_root=tmp_path, controller=busy_controller, specs=[spec])
    with pytest.raises(RuntimeError, match="结束当前识别"):
        manager.install(spec.id)

    external_manager = ModelManager(
        project_root=tmp_path,
        controller=FakeController(),
        specs=[spec],
        external_busy=lambda: True,
    )
    with pytest.raises(RuntimeError, match="样品-X"):
        external_manager.uninstall(spec.id)
    assert external_manager.status()["operation"] is None


def test_second_model_operation_is_rejected_while_first_runs(tmp_path):
    from threading import Event

    entered = Event()
    release = Event()
    spec = _spec(tmp_path)

    def runner(_command, _cwd):
        entered.set()
        assert release.wait(5)
        spec.markers[0].parent.mkdir(parents=True, exist_ok=True)
        spec.markers[0].write_text("weights", encoding="utf-8")
        return subprocess.CompletedProcess([], 0, "done", "")

    manager = ModelManager(project_root=tmp_path, controller=FakeController(), specs=[spec], runner=runner)
    manager.install(spec.id)
    assert entered.wait(5)
    with pytest.raises(RuntimeError, match="已有模型操作"):
        manager.install(spec.id)
    release.set()
    _wait(manager, "complete")


def test_uninstall_only_removes_declared_model_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    spec = _spec(tmp_path)
    model_root = tmp_path.parent / "models/example--model"
    object.__setattr__(spec, "markers", (model_root / "snapshots/master/model.bin",))
    object.__setattr__(spec, "install_paths", (model_root,))
    spec.markers[0].parent.mkdir(parents=True)
    spec.markers[0].write_text("weights", encoding="utf-8")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "profile.json").write_text("private", encoding="utf-8")
    manager = ModelManager(project_root=tmp_path, controller=FakeController(), specs=[spec])

    manager.uninstall(spec.id)
    _wait(manager, "complete")

    assert not spec.markers[0].exists()
    assert (outputs / "profile.json").read_text(encoding="utf-8") == "private"
    with pytest.raises(ValueError, match="拒绝删除"):
        manager._remove(outputs)
