"""Bootstrap confidence intervals for CIViC-A2V metric rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from a2v.metrics.civic import CivicMetricRow, aggregate_civic_rows, civic_score, load_metric_rows


METRIC_FUNCTIONS: dict[str, Callable[[list[CivicMetricRow]], float]] = {
    "causal_selectivity": lambda rows: aggregate_civic_rows(rows)["causal_selectivity"],
    "non_target_leakage": lambda rows: aggregate_civic_rows(rows)["non_target_leakage"],
    "identity_stability": lambda rows: aggregate_civic_rows(rows)["identity_stability"],
    "listener_responsiveness": lambda rows: aggregate_civic_rows(rows)["listener_responsiveness"],
    "turn_timing_error_ms": lambda rows: aggregate_civic_rows(rows)["turn_timing_error_ms"],
    "long_form_drift": lambda rows: aggregate_civic_rows(rows)["long_form_drift"],
    "instruction_following": lambda rows: aggregate_civic_rows(rows)["instruction_following"],
    "audio_video_sync": lambda rows: aggregate_civic_rows(rows)["audio_video_sync"],
    "latency": lambda rows: aggregate_civic_rows(rows)["latency"],
    "fps": lambda rows: aggregate_civic_rows(rows)["fps"],
    "civic_score": civic_score,
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


def bootstrap_confidence_intervals(
    rows: list[CivicMetricRow],
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    resamples: int = 1000,
    seed: int = 1234,
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("rows must be non-empty")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    unknown = sorted(set(metrics).difference(METRIC_FUNCTIONS))
    if unknown:
        raise ValueError(f"unknown metrics: {unknown}")
    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    row_count = len(rows)
    report: dict[str, dict[str, float]] = {}
    for metric in metrics:
        metric_fn = METRIC_FUNCTIONS[metric]
        observed = float(metric_fn(rows))
        samples = []
        for _ in range(resamples):
            indices = rng.integers(0, row_count, size=row_count)
            sampled_rows = [rows[int(index)] for index in indices]
            samples.append(float(metric_fn(sampled_rows)))
        values = np.asarray(samples, dtype="float64")
        report[metric] = {
            "mean": observed,
            "ci_low": float(np.quantile(values, alpha)),
            "ci_high": float(np.quantile(values, 1.0 - alpha)),
            "bootstrap_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return report


def render_bootstrap_markdown(report: dict[str, dict[str, float]], *, title: str = "CIViC-A2V Metric Confidence Intervals") -> str:
    lines = [
        f"# {title}",
        "",
        "| Metric | Mean | CI Low | CI High | Bootstrap Std |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, values in report.items():
        lines.append(
            f"| {metric} | {values['mean']:.4f} | {values['ci_low']:.4f} | "
            f"{values['ci_high']:.4f} | {values['bootstrap_std']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for CIViC metric rows.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/civic_bootstrap.json"))
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--metric-names", default=",".join(DEFAULT_METRICS))
    args = parser.parse_args()
    rows = load_metric_rows(args.metrics)
    report = bootstrap_confidence_intervals(
        rows,
        metrics=tuple(item.strip() for item in args.metric_names.split(",") if item.strip()),
        resamples=args.resamples,
        seed=args.seed,
        confidence=args.confidence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_bootstrap_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": len(report)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
