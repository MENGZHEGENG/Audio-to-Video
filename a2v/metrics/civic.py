"""CIViC-Bench aggregate metric helpers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class CivicMetricRow:
    clip_id: str
    intervention_kind: str
    causal_selectivity: float
    non_target_leakage: float
    identity_stability: float
    listener_responsiveness: float
    turn_timing_error_ms: float
    long_form_drift: float
    run_id: str | None = None
    instruction_following: float | None = None
    audio_video_sync: float | None = None
    latency: float | None = None
    fps: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CivicMetricRow":
        required = {
            "clip_id",
            "intervention_kind",
            "causal_selectivity",
            "non_target_leakage",
            "identity_stability",
            "listener_responsiveness",
            "turn_timing_error_ms",
            "long_form_drift",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"metric row missing fields: {sorted(missing)}")
        return cls(
            clip_id=str(payload["clip_id"]),
            intervention_kind=str(payload["intervention_kind"]),
            run_id=str(payload["run_id"]) if payload.get("run_id") is not None else None,
            causal_selectivity=float(payload["causal_selectivity"]),
            non_target_leakage=float(payload["non_target_leakage"]),
            identity_stability=float(payload["identity_stability"]),
            listener_responsiveness=float(payload["listener_responsiveness"]),
            turn_timing_error_ms=float(payload["turn_timing_error_ms"]),
            long_form_drift=float(payload["long_form_drift"]),
            instruction_following=float(payload["instruction_following"]) if payload.get("instruction_following") is not None else None,
            audio_video_sync=float(payload["audio_video_sync"]) if payload.get("audio_video_sync") is not None else None,
            latency=float(payload["latency"]) if payload.get("latency") is not None else None,
            fps=float(payload["fps"]) if payload.get("fps") is not None else None,
        )


def aggregate_civic_rows(rows: list[CivicMetricRow]) -> dict[str, float]:
    if not rows:
        raise ValueError("rows must be non-empty")
    summary = {
        "causal_selectivity": mean(row.causal_selectivity for row in rows),
        "non_target_leakage": mean(row.non_target_leakage for row in rows),
        "identity_stability": mean(row.identity_stability for row in rows),
        "listener_responsiveness": mean(row.listener_responsiveness for row in rows),
        "turn_timing_error_ms": mean(row.turn_timing_error_ms for row in rows),
        "long_form_drift": mean(row.long_form_drift for row in rows),
    }
    for key in ("instruction_following", "audio_video_sync", "latency", "fps"):
        values = [value for row in rows if (value := getattr(row, key)) is not None]
        if values:
            summary[key] = mean(values)
    return summary


def civic_score(rows: list[CivicMetricRow]) -> float:
    """Return a compact scalar for coarse model selection, not paper reporting."""

    metrics = aggregate_civic_rows(rows)
    return (
        metrics["causal_selectivity"]
        - metrics["non_target_leakage"]
        + metrics["identity_stability"]
        + metrics["listener_responsiveness"]
        - metrics["turn_timing_error_ms"] / 1000.0
        - metrics["long_form_drift"]
    )


def load_metric_rows(path: Path) -> list[CivicMetricRow]:
    rows: list[CivicMetricRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                rows.append(CivicMetricRow.from_dict(json.loads(stripped)))
            except Exception as exc:  # noqa: BLE001 - row context matters for metric files.
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def aggregate_file(path: Path) -> dict[str, float]:
    rows = load_metric_rows(path)
    metrics = aggregate_civic_rows(rows)
    metrics["civic_score"] = civic_score(rows)
    metrics["rows"] = float(len(rows))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CIViC-Bench JSONL metrics.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = aggregate_file(args.metrics)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
