import numpy as np

from src.speaker.verification import (
    cosine_similarity,
    enroll_embedding,
    decide,
    SpeakerDecision,
)


def test_cosine_similarity_identical_is_one():
    v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_enroll_embedding_averages_and_normalizes():
    e1 = np.array([1.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 1.0], dtype=np.float32)
    out = enroll_embedding([e1, e2])
    assert abs(np.linalg.norm(out) - 1.0) < 1e-6
    assert abs(out[0] - out[1]) < 1e-6


def test_decide_high_similarity_target():
    d = decide(0.6, prev=None, accept_th=0.5, reject_th=0.3)
    assert d.decision == "target"


def test_decide_low_similarity_non_target():
    d = decide(0.1, prev=None, accept_th=0.5, reject_th=0.3)
    assert d.decision == "non_target"


def test_decide_pending_when_no_thresholds():
    d = decide(0.4, prev=None, accept_th=None, reject_th=None)
    assert d.decision == "pending"


def test_decide_pending_when_hysteresis_band():
    d = decide(0.4, prev=None, accept_th=0.5, reject_th=0.3)
    assert d.decision == "pending"


def test_decide_pending_does_not_become_target_under_threshold():
    d = decide(0.4, prev=None, accept_th=0.45, reject_th=0.30)
    assert d.decision == "pending"