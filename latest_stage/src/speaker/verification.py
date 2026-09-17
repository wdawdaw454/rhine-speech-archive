"""Speaker cosine similarity and hysteresis decision logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


DecisionLiteral = Literal["target", "non_target", "pending"]


@dataclass
class SpeakerDecision:
    decision: DecisionLiteral
    similarity: float
    accept_threshold: float | None
    reject_threshold: float | None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def enroll_embedding(embeddings: list[np.ndarray], norm: str = "l2") -> np.ndarray:
    if not embeddings:
        raise ValueError("enroll_embedding requires at least one embedding")
    arr = np.mean(np.stack([np.asarray(e, dtype=np.float32) for e in embeddings], axis=0), axis=0)
    if norm == "l2":
        n = float(np.linalg.norm(arr))
        if n > 0:
            arr = arr / n
    return arr.astype(np.float32)


def decide(
    similarity: float,
    prev: SpeakerDecision | None,
    accept_th: float | None,
    reject_th: float | None,
) -> SpeakerDecision:
    if accept_th is None or reject_th is None:
        return SpeakerDecision("pending", similarity, accept_th, reject_th)
    if similarity >= accept_th:
        return SpeakerDecision("target", similarity, accept_th, reject_th)
    if similarity <= reject_th:
        return SpeakerDecision("non_target", similarity, accept_th, reject_th)
    return SpeakerDecision("pending", similarity, accept_th, reject_th)