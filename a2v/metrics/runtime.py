"""Runtime and streaming-quality metric helpers for CIViC-A2V."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _metric(payload: dict[str, Any], *names: str, default: float | None = None) -> float:
    for name in names:
        if name in payload and payload[name] is not None:
            return float(payload[name])
    if default is None:
        raise ValueError(f"missing one of required metric fields: {names}")
    return default


def runtime_metrics(
    payload: dict[str, Any],
    *,
    target_fps: float = 25.0,
    max_acceptable_latency_ms: float = 200.0,
    max_acceptable_drift: float = 1.0,
) -> dict[str, float]:
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if max_acceptable_latency_ms <= 0:
        raise ValueError("max_acceptable_latency_ms must be positive")
    if max_acceptable_drift <= 0:
        raise ValueError("max_acceptable_drift must be positive")
    frames = _metric(payload, "generated_frames", "frames")
    elapsed_seconds = _metric(payload, "elapsed_seconds", "wall_time_seconds", "runtime_seconds")
    if frames < 0 or elapsed_seconds <= 0:
        raise ValueError("frames must be non-negative and elapsed_seconds must be positive")
    fps = frames / elapsed_seconds
    latency_ms = _metric(payload, "latency_ms", "mean_latency_ms", default=1000.0 / max(fps, 1e-8))
    long_form_drift = _metric(payload, "long_form_drift", default=0.0)
    reconstruction_rmse = _metric(payload, "motion_rmse", "reconstruction_rmse", default=0.0)
    visual_quality = _metric(payload, "visual_quality", default=1.0 / (1.0 + reconstruction_rmse + long_form_drift))
    return {
        "latency": latency_ms,
        "fps": fps,
        "long_form_drift": long_form_drift,
        "visual_quality": visual_quality,
        "latency_budget_score": max(0.0, min(1.0, max_acceptable_latency_ms / max(latency_ms, 1e-8))),
        "fps_budget_score": max(0.0, min(1.0, fps / target_fps)),
        "drift_budget_score": max(0.0, min(1.0, 1.0 - (long_form_drift / max_acceptable_drift))),
    }


def runtime_report(
    *,
    metrics_path: Path,
    target_fps: float = 25.0,
    max_acceptable_latency_ms: float = 200.0,
    max_acceptable_drift: float = 1.0,
) -> dict[str, Any]:
    metrics = runtime_metrics(
        _load_json(metrics_path),
        target_fps=target_fps,
        max_acceptable_latency_ms=max_acceptable_latency_ms,
        max_acceptable_drift=max_acceptable_drift,
    )
    return {
        "input": str(metrics_path),
        "target_fps": target_fps,
        "max_acceptable_latency_ms": max_acceptable_latency_ms,
        "max_acceptable_drift": max_acceptable_drift,
        "metrics": metrics,
    }


def render_runtime_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CIViC-A2V Runtime and Streaming Metrics",
        "",
        f"Input: `{report['input']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in report["metrics"].items():
        lines.append(f"| {key} | {float(value):.4f} |")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute runtime and streaming-quality metrics from a run metrics JSON file.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=25.0)
    parser.add_argument("--max-acceptable-latency-ms", type=float, default=200.0)
    parser.add_argument("--max-acceptable-drift", type=float, default=1.0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = runtime_report(
        metrics_path=args.metrics,
        target_fps=args.target_fps,
        max_acceptable_latency_ms=args.max_acceptable_latency_ms,
        max_acceptable_drift=args.max_acceptable_drift,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_runtime_markdown(report), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
