from __future__ import annotations

from src.experiments.aggregate import aggregate_by_threshold


def _report(accept: float, **counts):
    target_refs = counts.get("target_refs", 2)
    non_target_refs = counts.get("non_target_refs", 2)
    commits = counts.get("commits", 2)
    return {
        "experiment": {"scenario": "test", "seed": 1},
        "thresholds": {"accept": accept, "reject": 0.0},
        "counts": {
            "target_references": target_refs,
            "non_target_references": non_target_refs,
            "caption_commits": commits,
        },
        "target_metrics": {
            "target_character_errors": counts.get("errors", 2),
            "target_reference_characters": counts.get("chars", 20),
            "target_recall": counts.get("recall", 1.0),
            "target_precision": counts.get("precision", 1.0),
            "non_target_leakage_rate": counts.get("leakage", 0.0),
        },
        "latency": {
            "first_partial": {
                "count": 2,
                "p50": counts.get("first_partial_p50", 0.9),
                "p95": counts.get("first_partial_p95", 1.5),
            },
            "caption_end_to_end": {"count": 2, "p50": 1.8, "p95": 2.4},
            "first_target_partial": {
                "count": 2,
                "p50": counts.get("first_target_partial_p50", 1.1),
                "p95": counts.get("first_target_partial_p95", 1.7),
            },
        },
        "rtf": 0.1,
    }


def test_aggregate_pools_counts_and_cer():
    summaries = aggregate_by_threshold(
        [
            _report(0.40, errors=2, chars=20, target_refs=2, non_target_refs=2),
            _report(
                0.40, errors=1, chars=30, target_refs=3, non_target_refs=1, first_partial_p50=1.1
            ),
        ]
    )
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["counts"]["target_references"] == 5
    assert summary["target_metrics"]["target_cer"] == 0.06
    assert summary["latency"]["first_partial_macro_p50"] == 1.1
    assert summary["latency"]["first_target_partial_macro_p50"] == 1.1
    assert summary["runs"] == 2


def test_aggregate_prefers_recall_and_lower_leakage():
    summaries = aggregate_by_threshold(
        [
            _report(0.30, recall=1.0, leakage=0.5),
            _report(0.50, recall=0.5, leakage=0.0),
        ]
    )
    assert summaries[0]["thresholds"]["accept"] == 0.3
    assert summaries[1]["thresholds"]["accept"] == 0.5


def test_aggregate_uses_exact_numerator_when_available():
    report = _report(0.40, target_refs=10, recall=2 / 3)
    report["counts"]["covered_target_references"] = 2
    summary = aggregate_by_threshold([report])[0]
    assert summary["counts"]["correct_target_commits"] == 2
    assert summary["target_metrics"]["target_recall"] == 0.2
