"""Dialogue-aware interaction metrics for CIViC-Bench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _as_motion(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 2:
        return values[:, None, :]
    if values.ndim == 3:
        return values
    raise ValueError(f"motion must have shape [frames, dims] or [frames, speakers, dims], got {values.shape}")


def _align(dialogue: np.ndarray, motion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dialogue_values = np.asarray(dialogue, dtype=np.float32)
    motion_values = _as_motion(motion)
    if dialogue_values.ndim != 2:
        raise ValueError(f"dialogue must have shape [frames, dims], got {dialogue_values.shape}")
    frames = min(dialogue_values.shape[0], motion_values.shape[0])
    if frames <= 0:
        raise ValueError("dialogue and motion must contain at least one frame")
    return dialogue_values[:frames], motion_values[:frames]


def _motion_energy(motion: np.ndarray) -> np.ndarray:
    if motion.shape[0] <= 1:
        return np.linalg.norm(motion, axis=-1)
    velocity = np.diff(motion, axis=0, prepend=motion[:1])
    return np.linalg.norm(velocity, axis=-1)


def _as_speaker_tensor(array: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"{name} must have shape [frames, speakers, dims], got {values.shape}")
    if values.shape[1] < 2:
        raise ValueError(f"{name} must include at least two speakers")
    return values


def _speaker_activity(dialogue: np.ndarray, speaker_count: int) -> np.ndarray:
    if dialogue.shape[1] < speaker_count:
        raise ValueError("dialogue feature dimension is smaller than speaker count")
    return dialogue[:, :speaker_count] > 0.5


def _turn_event_mask(dialogue: np.ndarray, speaker_count: int) -> np.ndarray:
    if dialogue.shape[1] >= speaker_count + 3:
        return (dialogue[:, speaker_count + 1] > 0.5) | (dialogue[:, speaker_count + 2] > 0.5)
    activity = _speaker_activity(dialogue, speaker_count)
    starts = activity & ~np.concatenate([np.zeros((1, speaker_count), dtype=bool), activity[:-1]], axis=0)
    ends = activity & ~np.concatenate([activity[1:], np.zeros((1, speaker_count), dtype=bool)], axis=0)
    return starts.any(axis=1) | ends.any(axis=1)


def listener_responsiveness(dialogue: np.ndarray, motion: np.ndarray) -> float:
    dialogue_values, motion_values = _align(dialogue, motion)
    speaker_count = motion_values.shape[1]
    activity = _speaker_activity(dialogue_values, speaker_count)
    active_count = activity.sum(axis=1)
    single_speaker = active_count == 1
    if not np.any(single_speaker):
        return 0.0
    energy = _motion_energy(motion_values)
    listener_mask = single_speaker[:, None] & ~activity
    speaker_mask = single_speaker[:, None] & activity
    listener_energy = float(energy[listener_mask].mean()) if np.any(listener_mask) else 0.0
    speaker_energy = float(energy[speaker_mask].mean()) if np.any(speaker_mask) else 0.0
    if speaker_energy <= 1e-8:
        return 0.0
    return listener_energy / speaker_energy


def turn_timing_error_ms(dialogue: np.ndarray, motion: np.ndarray, *, fps: float, threshold_quantile: float = 0.75) -> float:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be in [0, 1]")
    dialogue_values, motion_values = _align(dialogue, motion)
    turn_events = np.flatnonzero(_turn_event_mask(dialogue_values, motion_values.shape[1]))
    if turn_events.size == 0:
        return 0.0
    energy = _motion_energy(motion_values).mean(axis=1)
    threshold = float(np.quantile(energy, threshold_quantile))
    motion_events = np.flatnonzero(energy >= threshold)
    if motion_events.size == 0:
        return float("inf")
    errors = [float(np.min(np.abs(motion_events - event))) * 1000.0 / fps for event in turn_events]
    return float(np.mean(errors))


def mutual_gaze_consistency(gaze: np.ndarray) -> float:
    gaze_values = _as_speaker_tensor(gaze, name="gaze")
    flattened = gaze_values.reshape(gaze_values.shape[0], gaze_values.shape[1], -1)
    norms = np.linalg.norm(flattened, axis=-1, keepdims=True)
    valid = norms.squeeze(-1) > 1e-8
    if not np.any(valid):
        return 0.0
    unit = np.divide(flattened, np.maximum(norms, 1e-8))
    scores: list[float] = []
    for left in range(unit.shape[1]):
        for right in range(left + 1, unit.shape[1]):
            pair_valid = valid[:, left] & valid[:, right]
            if np.any(pair_valid):
                cosine = np.sum(unit[pair_valid, left] * unit[pair_valid, right], axis=-1)
                scores.append(float(np.mean((1.0 - cosine) / 2.0)))
    return float(np.mean(scores)) if scores else 0.0


def relative_pose_consistency(relative_pose: np.ndarray) -> float:
    pose = _as_speaker_tensor(relative_pose, name="relative_pose")
    flattened = pose.reshape(pose.shape[0], pose.shape[1], -1)
    distances: list[np.ndarray] = []
    for left in range(flattened.shape[1]):
        for right in range(left + 1, flattened.shape[1]):
            distances.append(np.linalg.norm(flattened[:, left] - flattened[:, right], axis=-1))
    if not distances:
        return 0.0
    stacked = np.stack(distances, axis=1)
    drift = float(np.mean(np.std(stacked, axis=0)))
    return 1.0 / (1.0 + drift)


def interaction_summary(
    dialogue: np.ndarray,
    motion: np.ndarray,
    *,
    fps: float,
    gaze: np.ndarray | None = None,
    relative_pose: np.ndarray | None = None,
) -> dict[str, float]:
    summary = {
        "listener_responsiveness": listener_responsiveness(dialogue, motion),
        "turn_timing_error_ms": turn_timing_error_ms(dialogue, motion, fps=fps),
    }
    if gaze is not None:
        summary["mutual_gaze"] = mutual_gaze_consistency(gaze)
    if relative_pose is not None:
        summary["relative_pose_consistency"] = relative_pose_consistency(relative_pose)
    return summary


def _load_npz_key(path: Path, key: str) -> np.ndarray:
    payload = np.load(path)
    if key not in payload:
        raise ValueError(f"{path} does not contain key {key!r}")
    return np.asarray(payload[key], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dialogue-aware CIViC interaction metrics.")
    parser.add_argument("--dialogue-npz", type=Path, required=True)
    parser.add_argument("--motion-npz", type=Path, required=True)
    parser.add_argument("--gaze-npz", type=Path)
    parser.add_argument("--relative-pose-npz", type=Path)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--dialogue-key", default="values")
    parser.add_argument("--motion-key", default="motion")
    parser.add_argument("--gaze-key", default="gaze")
    parser.add_argument("--relative-pose-key", default="relative_pose")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = interaction_summary(
        _load_npz_key(args.dialogue_npz, args.dialogue_key),
        _load_npz_key(args.motion_npz, args.motion_key),
        fps=args.fps,
        gaze=_load_npz_key(args.gaze_npz, args.gaze_key) if args.gaze_npz else None,
        relative_pose=_load_npz_key(args.relative_pose_npz, args.relative_pose_key) if args.relative_pose_npz else None,
    )
    text = json.dumps(summary, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
