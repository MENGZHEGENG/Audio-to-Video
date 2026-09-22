"""Lightweight audio/video alignment metrics for A2V prediction exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_array(path: Path, *, key: str | None = None) -> np.ndarray:
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        try:
            selected_key = key or next(iter(payload.files))
            return np.asarray(payload[selected_key], dtype=np.float32)
        finally:
            payload.close()
    return np.asarray(payload, dtype=np.float32)


def frame_energy(array: np.ndarray, *, time_axis: int = 0) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 0:
        raise ValueError("array must have a time axis")
    if time_axis < 0:
        time_axis += values.ndim
    if not 0 <= time_axis < values.ndim:
        raise ValueError(f"time_axis {time_axis} is out of range for shape {values.shape}")
    values = np.moveaxis(values, time_axis, 0)
    flattened = values.reshape(values.shape[0], -1)
    if flattened.shape[0] <= 1:
        return np.linalg.norm(flattened, axis=1)
    velocity = np.diff(flattened, axis=0, prepend=flattened[:1])
    return np.linalg.norm(velocity, axis=1)


def _flatten_time(array: np.ndarray, *, time_axis: int = 0) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 0:
        raise ValueError("array must have a time axis")
    if time_axis < 0:
        time_axis += values.ndim
    if not 0 <= time_axis < values.ndim:
        raise ValueError(f"time_axis {time_axis} is out of range for shape {values.shape}")
    values = np.moveaxis(values, time_axis, 0)
    return values.reshape(values.shape[0], -1)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    left_centered = left - float(left.mean())
    right_centered = right - float(right.mean())
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-8:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def lagged_correlation(audio_energy: np.ndarray, video_energy: np.ndarray, *, max_lag: int) -> tuple[int, float]:
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    frames = min(audio_energy.shape[0], video_energy.shape[0])
    if frames <= 0:
        raise ValueError("audio and video signals must contain at least one frame")
    audio = np.asarray(audio_energy[:frames], dtype=np.float32)
    video = np.asarray(video_energy[:frames], dtype=np.float32)
    best_lag = 0
    best_corr = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left = audio[-lag:]
            right = video[: frames + lag]
        elif lag > 0:
            left = audio[: frames - lag]
            right = video[lag:]
        else:
            left = audio
            right = video
        corr = _safe_corr(left, right)
        if corr > best_corr:
            best_lag = lag
            best_corr = corr
    return best_lag, best_corr


def generation_quality_score(motion: np.ndarray, target_motion: np.ndarray | None = None) -> float:
    motion_values = np.asarray(motion, dtype=np.float32)
    if target_motion is not None:
        target_values = np.asarray(target_motion, dtype=np.float32)
        frames = min(motion_values.shape[0], target_values.shape[0])
        if frames <= 0:
            raise ValueError("motion and target_motion must contain at least one frame")
        rmse = float(np.sqrt(np.mean((motion_values[:frames] - target_values[:frames]) ** 2)))
        return 1.0 / (1.0 + rmse)
    energy = frame_energy(motion_values)
    smoothness_penalty = float(np.mean(np.abs(np.diff(energy)))) if energy.size > 1 else 0.0
    return 1.0 / (1.0 + smoothness_penalty)


def temporally_aligned_rmse(
    motion: np.ndarray,
    target_motion: np.ndarray,
    *,
    time_axis: int = 0,
    max_warp_frames: int | None = None,
) -> float:
    return temporal_alignment_report(
        motion,
        target_motion,
        time_axis=time_axis,
        max_warp_frames=max_warp_frames,
    )["temporal_alignment_rmse"]


def temporal_alignment_report(
    motion: np.ndarray,
    target_motion: np.ndarray,
    *,
    time_axis: int = 0,
    max_warp_frames: int | None = None,
) -> dict[str, float | int]:
    predicted = _flatten_time(motion, time_axis=time_axis)
    target = _flatten_time(target_motion, time_axis=time_axis)
    if predicted.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("motion and target_motion must contain at least one frame")
    if predicted.shape[1] != target.shape[1]:
        raise ValueError(f"motion feature dims must match, got {predicted.shape[1]} and {target.shape[1]}")
    if max_warp_frames is not None and max_warp_frames < 0:
        raise ValueError("max_warp_frames must be non-negative")

    frames_pred, frames_target = predicted.shape[0], target.shape[0]
    band = max(frames_pred, frames_target) if max_warp_frames is None else int(max_warp_frames)
    band = max(band, abs(frames_pred - frames_target))
    costs = np.full((frames_pred + 1, frames_target + 1), np.inf, dtype=np.float64)
    steps = np.zeros((frames_pred + 1, frames_target + 1), dtype=np.int32)
    backpointers = np.zeros((frames_pred + 1, frames_target + 1), dtype=np.int8)
    costs[0, 0] = 0.0

    for pred_idx in range(1, frames_pred + 1):
        start = max(1, pred_idx - band)
        stop = min(frames_target, pred_idx + band) + 1
        for target_idx in range(start, stop):
            candidates = (
                (costs[pred_idx - 1, target_idx - 1], steps[pred_idx - 1, target_idx - 1], 1),
                (costs[pred_idx - 1, target_idx], steps[pred_idx - 1, target_idx], 2),
                (costs[pred_idx, target_idx - 1], steps[pred_idx, target_idx - 1], 3),
            )
            prev_cost, prev_steps, action = min(candidates, key=lambda item: item[0])
            local_cost = float(np.mean((predicted[pred_idx - 1] - target[target_idx - 1]) ** 2))
            costs[pred_idx, target_idx] = prev_cost + local_cost
            steps[pred_idx, target_idx] = prev_steps + 1
            backpointers[pred_idx, target_idx] = action

    if not np.isfinite(costs[frames_pred, frames_target]) or steps[frames_pred, frames_target] <= 0:
        raise ValueError("no temporal alignment path found; increase max_warp_frames")
    pred_idx, target_idx = frames_pred, frames_target
    diagonal_steps = 0
    predicted_only_steps = 0
    target_only_steps = 0
    max_abs_warp = 0
    total_abs_warp = 0
    while pred_idx > 0 and target_idx > 0:
        action = int(backpointers[pred_idx, target_idx])
        max_abs_warp = max(max_abs_warp, abs((pred_idx - 1) - (target_idx - 1)))
        total_abs_warp += abs((pred_idx - 1) - (target_idx - 1))
        if action == 1:
            diagonal_steps += 1
            pred_idx -= 1
            target_idx -= 1
        elif action == 2:
            predicted_only_steps += 1
            pred_idx -= 1
        elif action == 3:
            target_only_steps += 1
            target_idx -= 1
        else:  # pragma: no cover - defensive guard for corrupted paths.
            raise ValueError("invalid temporal alignment backpointer")
    path_steps = int(steps[frames_pred, frames_target])
    warped_steps = predicted_only_steps + target_only_steps
    temporal_rmse = float(np.sqrt(costs[frames_pred, frames_target] / path_steps))
    return {
        "temporal_alignment_rmse": temporal_rmse,
        "temporal_alignment_quality": 1.0 / (1.0 + temporal_rmse),
        "temporal_alignment_path_steps": path_steps,
        "temporal_alignment_diagonal_steps": diagonal_steps,
        "temporal_alignment_predicted_only_steps": predicted_only_steps,
        "temporal_alignment_target_only_steps": target_only_steps,
        "temporal_warp_fraction": float(warped_steps / path_steps),
        "temporal_path_length_ratio": float(path_steps / max(frames_pred, frames_target)),
        "temporal_mean_abs_warp_frames": float(total_abs_warp / path_steps),
        "temporal_max_abs_warp_frames": int(max_abs_warp),
    }


def temporally_aligned_quality_score(
    motion: np.ndarray,
    target_motion: np.ndarray,
    *,
    time_axis: int = 0,
    max_warp_frames: int | None = None,
) -> float:
    rmse = temporally_aligned_rmse(
        motion,
        target_motion,
        time_axis=time_axis,
        max_warp_frames=max_warp_frames,
    )
    return 1.0 / (1.0 + rmse)


def av_alignment_report(
    *,
    audio: np.ndarray,
    video: np.ndarray,
    fps: float,
    target_video: np.ndarray | None = None,
    max_lag_ms: float = 400.0,
    max_warp_ms: float = 1000.0,
    time_axis: int = 0,
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    max_lag = max(0, int(round(max_lag_ms * fps / 1000.0)))
    audio_energy = frame_energy(audio, time_axis=time_axis)
    video_energy = frame_energy(video, time_axis=time_axis)
    lag_frames, best_corr = lagged_correlation(audio_energy, video_energy, max_lag=max_lag)
    zero_lag_corr = _safe_corr(audio_energy[: min(audio_energy.size, video_energy.size)], video_energy[: min(audio_energy.size, video_energy.size)])
    generation_quality = generation_quality_score(video, target_video)
    report = {
        "audio_video_sync": max(0.0, min(1.0, (best_corr + 1.0) / 2.0)),
        "sync_lag_ms": float(lag_frames * 1000.0 / fps),
        "modality_consistency": max(0.0, min(1.0, (zero_lag_corr + 1.0) / 2.0)),
        "generation_quality": generation_quality,
        "max_lag_ms": float(max_lag_ms),
        "frames": int(min(audio_energy.size, video_energy.size)),
    }
    if target_video is not None:
        max_warp_frames = max(0, int(round(max_warp_ms * fps / 1000.0)))
        temporal_report = temporal_alignment_report(
            video,
            target_video,
            time_axis=time_axis,
            max_warp_frames=max_warp_frames,
        )
        report.update(temporal_report)
        report["max_warp_ms"] = float(max_warp_ms)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute lightweight audio/video alignment metrics.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--target-video", type=Path)
    parser.add_argument("--audio-key")
    parser.add_argument("--video-key")
    parser.add_argument("--target-key")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--max-lag-ms", type=float, default=400.0)
    parser.add_argument("--max-warp-ms", type=float, default=1000.0)
    parser.add_argument("--time-axis", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = load_array(args.target_video, key=args.target_key) if args.target_video else None
    report = av_alignment_report(
        audio=load_array(args.audio, key=args.audio_key),
        video=load_array(args.video, key=args.video_key),
        target_video=target,
        fps=args.fps,
        max_lag_ms=args.max_lag_ms,
        max_warp_ms=args.max_warp_ms,
        time_axis=args.time_axis,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
