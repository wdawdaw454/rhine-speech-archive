"""Speaker-related metrics: EER, minDCF, target recall/precision, leakage rates."""
from __future__ import annotations


def _sorted_scores(targets: list[float], nontargets: list[float]) -> tuple[list[float], list[float]]:
    return sorted(targets), sorted(nontargets)


def eer(target_scores: list[float], non_target_scores: list[float]) -> float:
    t, n = _sorted_scores(target_scores, non_target_scores)
    if not t or not n:
        return 0.0
    candidates = sorted(set(t + n))
    best_eer = 1.0
    for thr in candidates:
        frr = sum(1 for s in t if s < thr) / len(t)
        far = sum(1 for s in n if s >= thr) / len(n)
        diff = abs(frr - far)
        eer_candidate = (frr + far) / 2.0
        if diff < best_eer or (diff == best_eer and eer_candidate < best_eer):
            best_eer = eer_candidate
    return float(min(best_eer, 1.0))


def min_dcf(
    target_scores: list[float],
    non_target_scores: list[float],
    p_target: float = 0.5,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    t, n = _sorted_scores(target_scores, non_target_scores)
    if not t or not n:
        return 0.0
    candidates = sorted(set(t + n))
    best = float("inf")
    for thr in candidates:
        miss = sum(1 for s in t if s < thr) / len(t)
        fa = sum(1 for s in n if s >= thr) / len(n)
        dcf = p_target * c_miss * miss + (1 - p_target) * c_fa * fa
        if dcf < best:
            best = dcf
    return float(best)


def target_recall(committed_target_refs: set, all_target_refs: set) -> float:
    if not all_target_refs:
        return 0.0
    return len(committed_target_refs & all_target_refs) / len(all_target_refs)


def target_precision(committed_target_refs: set, all_committed_refs: set) -> float:
    if not all_committed_refs:
        return 0.0
    return len(committed_target_refs & all_committed_refs) / len(all_committed_refs)


def non_target_leakage_rate(committed_non_target: set, all_non_target: set) -> float:
    if not all_non_target:
        return 0.0
    return len(committed_non_target & all_non_target) / len(all_non_target)


def non_target_leakage_time_rate(committed_non_target_time: float, total_non_target_time: float) -> float:
    if total_non_target_time <= 0.0:
        return 0.0
    return float(committed_non_target_time / total_non_target_time)