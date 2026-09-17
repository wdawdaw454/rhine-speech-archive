"""Download ModelScope models into the project-local models directory."""
from __future__ import annotations

import logging
from pathlib import Path

from .catalog import MODEL_SPECS, ModelSpec

logger = logging.getLogger(__name__)


def download_model(spec: ModelSpec, project_root: str | Path) -> Path:
    """Download one model into ``project_root / spec.directory``."""
    from modelscope import snapshot_download

    target = Path(project_root) / spec.directory
    target.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s (%s) to %s", spec.key, spec.model_id, target)
    snapshot_download(spec.model_id, local_dir=str(target))
    return target


def download_models(keys: list[str], project_root: str | Path) -> dict[str, Path]:
    if "all" in keys:
        keys = list(MODEL_SPECS)
    unknown = sorted(set(keys) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"unknown model keys: {', '.join(unknown)}")
    return {key: download_model(MODEL_SPECS[key], project_root) for key in keys}
