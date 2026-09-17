"""Aggregate threshold-sweep evaluation reports."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _metric(report: dict[str, Any], section: str, key: str, default: float = 0.0) -> float:
    value = report.get(section, {}).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pooled_count(
    report: dict[str, Any],
    count_key: str,
    metric_key: str,
    denominator_key: str,
) -> int:
    counts = report.get("counts", {})
    if count_key in counts:
        return int(counts[count_key])
    # Compatibility with reports generated before exact numerators were added.
    return round(
        _metric(report, "target_metrics", metric_key) * _metric(report, "counts", denominator_key)
    )


def aggregate_by_threshold(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pool counts and CER exactly, and report macro latency percentiles."""

    if not reports:
        return []

    groups: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        key = (
            _metric(report, "thresholds", "accept"),
            _metric(report, "thresholds", "reject"),
        )
        groups[key].append(report)

    summaries: list[dict[str, Any]] = []
    for (accept, reject), group in sorted(groups.items()):
        target_refs = sum(int(_metric(r, "counts", "target_references")) for r in group)
        non_target_refs = sum(int(_metric(r, "counts", "non_target_references")) for r in group)
        commits = sum(int(_metric(r, "counts", "caption_commits")) for r in group)
        target_errors = sum(
            int(_metric(r, "target_metrics", "target_character_errors")) for r in group
        )
        target_chars = sum(
            int(_metric(r, "target_metrics", "target_reference_characters")) for r in group
        )
        correct_targets = sum(
            _pooled_count(r, "covered_target_references", "target_recall", "target_references")
            for r in group
        )
        clean_commits = sum(
            _pooled_count(r, "clean_target_caption_commits", "target_precision", "caption_commits")
            for r in group
        )
        leaked_non_targets = sum(
            _pooled_count(
                r,
                "committed_non_target_references",
                "non_target_leakage_rate",
                "non_target_references",
            )
            for r in group
        )

        latency: dict[str, Any] = {}
        for metric_name in (
            "first_partial",
            "first_target_partial",
            "caption_end_to_end",
        ):
            p50_values = []
            p95_values = []
            for report in group:
                summary = report.get("latency", {}).get(metric_name, {})
                if not summary.get("count", 0):
                    continue
                latency_section = report.get("latency", {})
                if latency_section.get("note"):
                    continue
                p50_values.append(float(summary.get("p50", 0.0)))
                p95_values.append(float(summary.get("p95", 0.0)))
            if p50_values:
                latency[f"{metric_name}_macro_p50"] = round(
                    sorted(p50_values)[len(p50_values) // 2], 6
                )
            if p95_values:
                latency[f"{metric_name}_macro_p95"] = round(
                    sorted(p95_values)[len(p95_values) // 2], 6
                )

        summaries.append(
            {
                "thresholds": {"accept": accept, "reject": reject},
                "runs": len(group),
                "counts": {
                    "target_references": target_refs,
                    "non_target_references": non_target_refs,
                    "caption_commits": commits,
                    "correct_target_commits": int(correct_targets),
                    "clean_caption_commits": int(clean_commits),
                    "leaked_non_target_references": int(leaked_non_targets),
                },
                "target_metrics": {
                    "target_cer": (
                        round(min(1.0, target_errors / target_chars), 6)
                        if target_chars
                        else (1.0 if target_errors else 0.0)
                    ),
                    "target_recall": (
                        round(correct_targets / target_refs, 6) if target_refs else 0.0
                    ),
                    "target_precision": (round(clean_commits / commits, 6) if commits else 0.0),
                    "non_target_leakage_rate": (
                        round(leaked_non_targets / non_target_refs, 6) if non_target_refs else 0.0
                    ),
                },
                "latency": latency,
            }
        )

    def ranking(item: dict[str, Any]) -> tuple[float, float, float]:
        metrics = item["target_metrics"]
        return (
            -metrics["target_recall"],
            metrics["non_target_leakage_rate"],
            metrics["target_cer"],
        )

    return sorted(summaries, key=ranking)
