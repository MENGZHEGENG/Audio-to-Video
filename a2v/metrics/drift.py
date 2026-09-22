"""Long-form motion drift diagnostics for audio-to-video generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_array(path: Path, key: str) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    with np.load(path) as payload:
        if key not in payload:
            raise ValueError(f"{path} does not contain key {key!r}; available={sorted(payload.files)}")
        return np.asarray(payload[key])


def _as_time_major(values: np.ndarray, time_axis: int) -> np.ndarray:
    axis = time_axis if time_axis >= 0 else values.ndim + time_axis
    if axis < 0 or axis >= values.ndim:
        raise ValueError(f"time_axis {time_axis} out of bounds for shape {values.shape}")
    return np.moveaxis(values.astype("float32"), axis, 0)


def _flatten_window_time(values: np.ndarray) -> np.ndarray:
    if values.ndim < 3:
        raise ValueError(f"flattened window-time arrays need at least 3 dimensions, got {values.shape}")
    return values.reshape(values.shape[0] * values.shape[1], *values.shape[2:])


def segment_motion_drift(
    values: np.ndarray,
    *,
    fps: float,
    segment_seconds: float,
    time_axis: int = 0,
    flatten_window_axis: bool = False,
) -> dict[str, object]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")
    if flatten_window_axis:
        values = _flatten_window_time(values)
        time_axis = 0
    axis = time_axis if time_axis >= 0 else values.ndim + time_axis
    time_major = _as_time_major(values, axis).reshape(values.shape[axis], -1)
    segment_frames = max(int(round(fps * segment_seconds)), 1)
    segment_means = []
    for start in range(0, time_major.shape[0], segment_frames):
        segment = time_major[start : start + segment_frames]
        if len(segment):
            segment_means.append(segment.mean(axis=0))
    if not segment_means:
        raise ValueError("motion array must contain at least one frame")
    reference = segment_means[0]
    per_segment = [float(np.linalg.norm(segment - reference) / np.sqrt(reference.size)) for segment in segment_means]
    later = per_segment[1:]
    return {
        "frames": int(time_major.shape[0]),
        "fps": float(fps),
        "segment_seconds": float(segment_seconds),
        "segment_frames": int(segment_frames),
        "segments": len(per_segment),
        "flatten_window_axis": flatten_window_axis,
        "per_segment_drift": per_segment,
        "long_form_drift": float(np.mean(later)) if later else 0.0,
        "max_segment_drift": float(max(per_segment)) if per_segment else 0.0,
    }


def drift_report(
    *,
    prediction_path: Path,
    key: str = "motion",
    fps: float,
    segment_seconds: float = 10.0,
    time_axis: int = 0,
    flatten_window_axis: bool = False,
) -> dict[str, object]:
    metrics = segment_motion_drift(
        _load_array(prediction_path, key),
        fps=fps,
        segment_seconds=segment_seconds,
        time_axis=time_axis,
        flatten_window_axis=flatten_window_axis,
    )
    return {"prediction": str(prediction_path), "key": key, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute segment-wise long-form drift for generated motion arrays.")
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--key", default="motion")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--time-axis", type=int, default=0)
    parser.add_argument("--flatten-window-axis", action="store_true", help="Treat [windows, frames, ...] as one time axis.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = drift_report(
        prediction_path=args.prediction,
        key=args.key,
        fps=args.fps,
        segment_seconds=args.segment_seconds,
        time_axis=args.time_axis,
        flatten_window_axis=args.flatten_window_axis,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
