"""Motion-delta diagnostics for counterfactual audio interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from a2v.metrics.interaction import interaction_summary


def _as_speaker_motion(array: np.ndarray, *, flatten_window_axis: bool = False) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if flatten_window_axis:
        if values.ndim != 4:
            raise ValueError(f"flatten_window_axis expects shape [windows, frames, speakers, dims], got {values.shape}")
        return values.reshape(values.shape[0] * values.shape[1], values.shape[2], values.shape[3])
    if values.ndim == 2:
        return values[:, None, :]
    if values.ndim == 3:
        return values
    raise ValueError(f"motion arrays must have shape [frames, dims] or [frames, speakers, dims], got {values.shape}")


def aligned_motion_pair(
    reference: np.ndarray,
    counterfactual: np.ndarray,
    *,
    flatten_window_axis: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    reference_motion = _as_speaker_motion(reference, flatten_window_axis=flatten_window_axis)
    counterfactual_motion = _as_speaker_motion(counterfactual, flatten_window_axis=flatten_window_axis)
    if reference_motion.shape[1:] != counterfactual_motion.shape[1:]:
        raise ValueError(
            "motion arrays must share speaker and feature dimensions: "
            f"{reference_motion.shape} vs {counterfactual_motion.shape}"
        )
    frame_count = min(reference_motion.shape[0], counterfactual_motion.shape[0])
    if frame_count <= 0:
        raise ValueError("motion arrays must contain at least one frame")
    return reference_motion[:frame_count], counterfactual_motion[:frame_count]


def per_speaker_motion_delta(reference: np.ndarray, counterfactual: np.ndarray, *, flatten_window_axis: bool = False) -> np.ndarray:
    reference_motion, counterfactual_motion = aligned_motion_pair(
        reference,
        counterfactual,
        flatten_window_axis=flatten_window_axis,
    )
    return np.linalg.norm(counterfactual_motion - reference_motion, axis=-1).mean(axis=0)


def leakage_summary(
    reference: np.ndarray,
    counterfactual: np.ndarray,
    *,
    speaker_ids: Iterable[str],
    target_speaker_id: str,
    flatten_window_axis: bool = False,
) -> dict[str, float | dict[str, float]]:
    speaker_list = [str(speaker) for speaker in speaker_ids]
    if target_speaker_id not in speaker_list:
        raise ValueError(f"target_speaker_id {target_speaker_id!r} not found in speaker_ids")
    deltas = per_speaker_motion_delta(reference, counterfactual, flatten_window_axis=flatten_window_axis)
    if len(speaker_list) != int(deltas.shape[0]):
        raise ValueError(f"speaker_ids length {len(speaker_list)} does not match motion speakers {deltas.shape[0]}")
    target_index = speaker_list.index(target_speaker_id)
    non_target_indices = [index for index in range(len(speaker_list)) if index != target_index]
    target_delta = float(deltas[target_index])
    non_target_delta = float(np.mean(deltas[non_target_indices])) if non_target_indices else 0.0
    return {
        "target_motion_delta": target_delta,
        "non_target_motion_delta": non_target_delta,
        "causal_selectivity": target_delta - non_target_delta,
        "non_target_leakage": non_target_delta,
        "per_speaker_delta": {speaker: float(deltas[index]) for index, speaker in enumerate(speaker_list)},
    }


def civic_row_from_motion_pair(
    *,
    clip_id: str,
    intervention_kind: str,
    target_speaker_id: str,
    speaker_ids: Iterable[str],
    reference: np.ndarray,
    counterfactual: np.ndarray,
    identity_stability: float = 1.0,
    listener_responsiveness: float = 0.0,
    turn_timing_error_ms: float = 0.0,
    long_form_drift: float = 0.0,
    flatten_window_axis: bool = False,
) -> dict[str, float | str]:
    summary = leakage_summary(
        reference,
        counterfactual,
        speaker_ids=speaker_ids,
        target_speaker_id=target_speaker_id,
        flatten_window_axis=flatten_window_axis,
    )
    return {
        "clip_id": clip_id,
        "intervention_kind": intervention_kind,
        "causal_selectivity": float(summary["causal_selectivity"]),
        "non_target_leakage": float(summary["non_target_leakage"]),
        "identity_stability": identity_stability,
        "listener_responsiveness": listener_responsiveness,
        "turn_timing_error_ms": turn_timing_error_ms,
        "long_form_drift": long_form_drift,
    }


def _load_npz_key(path: Path, key: str) -> np.ndarray:
    with np.load(path) as payload:
        if key not in payload:
            raise ValueError(f"{path} does not contain key {key!r}; available={sorted(payload.files)}")
        return np.asarray(payload[key])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CIViC leakage metrics from paired motion arrays.")
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--counterfactual-npz", type=Path, required=True)
    parser.add_argument("--key", default="motion")
    parser.add_argument("--speaker-ids", required=True, help="Comma-separated speaker IDs in array speaker-axis order.")
    parser.add_argument("--target-speaker-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--intervention-kind", required=True)
    parser.add_argument("--dialogue-npz", type=Path, help="Optional dialogue feature .npz for listener/turn metrics.")
    parser.add_argument("--dialogue-key", default="values")
    parser.add_argument("--fps", type=float, help="Required with --dialogue-npz for turn-timing metrics.")
    parser.add_argument("--flatten-window-axis", action="store_true", help="Treat [windows, frames, speakers, dims] as one time axis.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference = _load_npz_key(args.reference_npz, args.key)
    counterfactual = _load_npz_key(args.counterfactual_npz, args.key)
    interaction_metrics = {}
    if args.dialogue_npz is not None:
        if args.fps is None:
            raise SystemExit("--fps is required when --dialogue-npz is provided")
        interaction_metrics = interaction_summary(
            _load_npz_key(args.dialogue_npz, args.dialogue_key),
            counterfactual,
            fps=args.fps,
        )
    row = civic_row_from_motion_pair(
        clip_id=args.clip_id,
        intervention_kind=args.intervention_kind,
        target_speaker_id=args.target_speaker_id,
        speaker_ids=[item.strip() for item in args.speaker_ids.split(",") if item.strip()],
        reference=reference,
        counterfactual=counterfactual,
        flatten_window_axis=args.flatten_window_axis,
        listener_responsiveness=float(interaction_metrics.get("listener_responsiveness", 0.0)),
        turn_timing_error_ms=float(interaction_metrics.get("turn_timing_error_ms", 0.0)),
    )
    line = json.dumps(row, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(line + "\n", encoding="utf-8")
    print(line)


if __name__ == "__main__":
    main()
