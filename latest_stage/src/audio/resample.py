"""Audio resampling helpers using soxr."""
from __future__ import annotations

import numpy as np


def resample(waveform: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample mono waveform from src_sr to dst_sr using soxr."""
    if src_sr == dst_sr:
        return waveform.astype(np.float32, copy=False)
    import soxr

    out = soxr.resample(waveform.astype(np.float32, copy=False), src_sr, dst_sr)
    return np.asarray(out, dtype=np.float32)