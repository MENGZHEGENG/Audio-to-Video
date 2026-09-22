"""Paired bootstrap confidence intervals for baseline-vs-candidate CIViC metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from a2v.metrics.civic import CivicMetricRow, aggregate_civic_rows, load_metric_rows


HIGHER_IS_BETTER = {
    "causal_selectivity": True,
    "non_target_leakage": False,
    "identity_stability": True,
    "listener_responsiveness": True,
    "turn_timing_error_ms": False,
    "long_form_drift": False,
    "instruction_following": True,
    "audio_video_sync": True,
    "latency": False,
    "fps": True,
    "civic_score": True,
}
DEFAULT_METRICS = (
    "causal_selectivity",
    "non_target_leakage",
    "identity_stability",
    "listener_responsiveness",
    "turn_timing_error_ms",
    "long_form_drift",
    "civic_score",
)


@dataclass(frozen=True)
class PairedMetricRow:
    key: tuple[str, str]
    baseline: CivicMetricRow
    candidate: CivicMetricRow


def _row_key(row: CivicMetricRow) -> tuple[str, str]:
    return row.clip_id, row.intervention_kind


def _key_example(key: tuple[str, str]) -> dict[str, str]:
    return {"clip_id": key[0], "intervention_kind": key[1]}


def _collapse_rows(rows: list[CivicMetricRow]) -> tuple[dict[tuple[str, str], CivicMetricRow], dict[tuple[str, str], int]]:
    grouped: dict[tuple[str, str], list[CivicMetricRow]] = {}
    for row in rows:
        key = _row_key(row)
        grouped.setdefault(key, []).append(row)

    collapsed: dict[tuple[str, str], CivicMetricRow] = {}
    duplicate_counts: dict[tuple[str, str], int] = {}
    for key, members in grouped.items():
        if len(members) == 1:
            collapsed[key] = members[0]
            continue
        duplicate_counts[key] = len(members)
        averaged = aggregate_civic_rows(members)
        exemplar = members[0]
        collapsed[key] = CivicMetricRow(
            clip_id=exemplar.clip_id,
            intervention_kind=exemplar.intervention_kind,
            causal_selectivity=float(averaged["causal_selectivity"]),
            non_target_leakage=float(averaged["non_target_leakage"]),
            identity_stability=float(averaged["identity_stability"]),
            listener_responsiveness=float(averaged["listener_responsiveness"]),
            turn_timing_error_ms=float(averaged["turn_timing_error_ms"]),
            long_form_drift=float(averaged["long_form_drift"]),
            instruction_following=float(averaged["instruction_following"]) if "instruction_following" in averaged else None,
            audio_video_sync=float(averaged["audio_video_sync"]) if "audio_video_sync" in averaged else None,
            latency=float(averaged["latency"]) if "latency" in averaged else None,
            fps=float(averaged["fps"]) if "fps" in averaged else None,
        )
    return collapsed, duplicate_counts


def _index_rows(rows: list[CivicMetricRow], *, label: str) -> dict[tuple[str, str], CivicMetricRow]:
    indexed, _ = _collapse_rows(rows)
    return indexed


def align_metric_rows(
    baseline_rows: list[CivicMetricRow],
    candidate_rows: list[CivicMetricRow],
) -> list[PairedMetricRow]:
    baseline = _index_rows(baseline_rows, label="baseline")
    candidate = _index_rows(candidate_rows, label="candidate")
    common_keys = sorted(set(baseline).intersection(candidate))
    if not common_keys:
        raise ValueError("baseline and candidate metric rows have no shared clip/intervention keys")
    return [PairedMetricRow(key, baseline[key], candidate[key]) for key in common_keys]


def _row_civic_score(row: CivicMetricRow) -> float:
    return (
        row.causal_selectivity
        - row.non_target_leakage
        + row.identity_stability
        + row.listener_responsiveness
        - row.turn_timing_error_ms / 1000.0
        - row.long_form_drift
    )


def _metric_value(row: CivicMetricRow, metric: str) -> float:
    if metric == "civic_score":
        return _row_civic_score(row)
    return float(getattr(row, metric))


def paired_deltas(pairs: list[PairedMetricRow], *, metric: str) -> np.ndarray:
    if metric not in HIGHER_IS_BETTER:
        raise ValueError(f"unknown metric: {metric}")
    sign = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
    return np.asarray(
        [sign * (_metric_value(pair.candidate, metric) - _metric_value(pair.baseline, metric)) for pair in pairs],
        dtype="float64",
    )


def paired_bootstrap_deltas(
    baseline_rows: list[CivicMetricRow],
    candidate_rows: list[CivicMetricRow],
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    resamples: int = 1000,
    seed: int = 1234,
    confidence: float = 0.95,
    max_examples: int = 20,
) -> dict[str, object]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    unknown = sorted(set(metrics).difference(HIGHER_IS_BETTER))
    if unknown:
        raise ValueError(f"unknown metrics: {unknown}")
    baseline, baseline_duplicate_counts = _collapse_rows(baseline_rows)
    candidate, candidate_duplicate_counts = _collapse_rows(candidate_rows)
    common_keys = sorted(set(baseline).intersection(candidate))
    if not common_keys:
        raise ValueError("baseline and candidate metric rows have no shared clip/intervention keys")
    pairs = [PairedMetricRow(key, baseline[key], candidate[key]) for key in common_keys]
    baseline_only = sorted(set(baseline).difference(candidate))
    candidate_only = sorted(set(candidate).difference(baseline))
    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    report: dict[str, object] = {
        "baseline_rows": len(baseline_rows),
        "candidate_rows": len(candidate_rows),
        "baseline_collapsed_rows": len(baseline),
        "candidate_collapsed_rows": len(candidate),
        "baseline_duplicate_rows": int(sum(count - 1 for count in baseline_duplicate_counts.values())),
        "candidate_duplicate_rows": int(sum(count - 1 for count in candidate_duplicate_counts.values())),
        "paired_rows": len(pairs),
        "unpaired_baseline_rows": len(baseline_only),
        "unpaired_candidate_rows": len(candidate_only),
        "baseline_duplicate_examples": [
            {**_key_example(key), "rows": count} for key, count in list(sorted(baseline_duplicate_counts.items()))[:max_examples]
        ],
        "candidate_duplicate_examples": [
            {**_key_example(key), "rows": count} for key, count in list(sorted(candidate_duplicate_counts.items()))[:max_examples]
        ],
        "unpaired_baseline_examples": [_key_example(key) for key in baseline_only[:max_examples]],
        "unpaired_candidate_examples": [_key_example(key) for key in candidate_only[:max_examples]],
        "unpaired_examples_truncated": len(baseline_only) > max_examples or len(candidate_only) > max_examples,
        "duplicate_examples_truncated": len(baseline_duplicate_counts) > max_examples or len(candidate_duplicate_counts) > max_examples,
        "confidence": confidence,
        "metrics": {},
    }
    metric_reports: dict[str, dict[str, float | str]] = {}
    for metric in metrics:
        deltas = paired_deltas(pairs, metric=metric)
        samples = []
        for _ in range(resamples):
            indices = rng.integers(0, len(deltas), size=len(deltas))
            samples.append(float(deltas[indices].mean()))
        sample_values = np.asarray(samples, dtype="float64")
        metric_reports[metric] = {
            "better_direction": "higher" if HIGHER_IS_BETTER[metric] else "lower",
            "mean_delta_positive_is_better": float(deltas.mean()),
            "ci_low": float(np.quantile(sample_values, alpha)),
            "ci_high": float(np.quantile(sample_values, 1.0 - alpha)),
            "bootstrap_std": float(sample_values.std(ddof=1)) if len(sample_values) > 1 else 0.0,
            "probability_delta_positive": float((sample_values > 0).mean()),
        }
    report["metrics"] = metric_reports
    return report


def render_paired_bootstrap_markdown(
    report: dict[str, object],
    *,
    title: str = "CIViC-A2V Paired Improvement Confidence Intervals",
) -> str:
    lines = [
        f"# {title}",
        "",
        f"Baseline rows: {report.get('baseline_rows', 'unknown')}",
        f"Candidate rows: {report.get('candidate_rows', 'unknown')}",
        f"Collapsed baseline rows: {report.get('baseline_collapsed_rows', report.get('baseline_rows', 'unknown'))}",
        f"Collapsed candidate rows: {report.get('candidate_collapsed_rows', report.get('candidate_rows', 'unknown'))}",
        f"Collapsed duplicate baseline rows: {report.get('baseline_duplicate_rows', 0)}",
        f"Collapsed duplicate candidate rows: {report.get('candidate_duplicate_rows', 0)}",
        f"Paired rows: {report['paired_rows']}",
        f"Baseline-only rows: {report.get('unpaired_baseline_rows', 0)}",
        f"Candidate-only rows: {report.get('unpaired_candidate_rows', 0)}",
        f"Confidence: {report['confidence']}",
        "",
        "Positive deltas mean the candidate is better after metric-direction normalization.",
        "",
        "| Metric | Better Direction | Mean Delta | CI Low | CI High | P(Delta > 0) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    for metric, values in metrics.items():
        assert isinstance(values, dict)
        lines.append(
            f"| {metric} | {values['better_direction']} | "
            f"{float(values['mean_delta_positive_is_better']):.4f} | "
            f"{float(values['ci_low']):.4f} | {float(values['ci_high']):.4f} | "
            f"{float(values['probability_delta_positive']):.3f} |"
        )
    duplicate_baseline_examples = report.get("baseline_duplicate_examples", [])
    duplicate_candidate_examples = report.get("candidate_duplicate_examples", [])
    if duplicate_baseline_examples or duplicate_candidate_examples:
        lines.extend(["", "## Collapsed Duplicate Key Examples"])
        for label, examples in (("Baseline duplicate", duplicate_baseline_examples), ("Candidate duplicate", duplicate_candidate_examples)):
            if not isinstance(examples, list) or not examples:
                continue
            lines.append(f"- {label}:")
            for row in examples:
                if isinstance(row, dict):
                    lines.append(
                        f"  - `{row.get('clip_id')}` / `{row.get('intervention_kind')}` ({row.get('rows')} rows averaged)"
                    )
        if report.get("duplicate_examples_truncated"):
            lines.append("- Additional duplicate keys are omitted; rerun with a larger `--max-examples` to list more.")
    baseline_examples = report.get("unpaired_baseline_examples", [])
    candidate_examples = report.get("unpaired_candidate_examples", [])
    if baseline_examples or candidate_examples:
        lines.extend(["", "## Unpaired Key Examples"])
        for label, examples in (("Baseline-only", baseline_examples), ("Candidate-only", candidate_examples)):
            if not isinstance(examples, list) or not examples:
                continue
            lines.append(f"- {label}:")
            for row in examples:
                if isinstance(row, dict):
                    lines.append(f"  - `{row.get('clip_id')}` / `{row.get('intervention_kind')}`")
        if report.get("unpaired_examples_truncated"):
            lines.append("- Additional unpaired keys are omitted; rerun with a larger `--max-examples` to list more.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap deltas for CIViC baseline-vs-candidate metrics.")
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/civic_paired_bootstrap.json"))
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--metric-names", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()
    report = paired_bootstrap_deltas(
        load_metric_rows(args.baseline_metrics),
        load_metric_rows(args.candidate_metrics),
        metrics=tuple(item.strip() for item in args.metric_names.split(",") if item.strip()),
        resamples=args.resamples,
        seed=args.seed,
        confidence=args.confidence,
        max_examples=args.max_examples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_paired_bootstrap_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "paired_rows": report["paired_rows"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
