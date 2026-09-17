from src.evaluation.speaker_metrics import (
    eer,
    min_dcf,
    target_recall,
    target_precision,
    non_target_leakage_rate,
    non_target_leakage_time_rate,
)


def test_eer_perfect_separation():
    scores_t = [0.9] * 10
    scores_n = [0.1] * 10
    e = eer(scores_t, scores_n)
    assert 0.0 <= e <= 1.0
    assert e == 0.0


def test_min_dcf_perfect_separation_zero():
    scores_t = [0.9] * 10
    scores_n = [0.1] * 10
    d = min_dcf(scores_t, scores_n, p_target=0.5, c_miss=1.0, c_fa=1.0)
    assert d == 0.0


def test_target_recall_and_precision():
    committed_target_refs = {"a", "b", "c"}
    all_target_refs = {"a", "b", "c", "d"}
    all_committed_refs = {"a", "b", "c", "z"}
    r = target_recall(committed_target_refs, all_target_refs)
    p = target_precision(committed_target_refs, all_committed_refs)
    assert abs(r - 0.75) < 1e-9
    assert abs(p - 0.75) < 1e-9


def test_non_target_leakage_rate_utterance_level():
    committed_non_target = {"x", "y"}
    all_non_target = {"x", "y", "p", "q", "r"}
    rate = non_target_leakage_rate(committed_non_target, all_non_target)
    assert abs(rate - 0.4) < 1e-9


def test_non_target_leakage_rate_time_level():
    rate = non_target_leakage_time_rate(1.0, 4.0)
    assert abs(rate - 0.25) < 1e-9


def test_leakage_zero_when_no_non_target():
    assert non_target_leakage_rate(set(), set()) == 0.0
    assert non_target_leakage_time_rate(0.0, 0.0) == 0.0