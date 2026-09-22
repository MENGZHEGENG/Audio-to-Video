"""Instruction-following metrics for speaker-conditioned A2V controls."""

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
            return np.asarray(payload[selected_key])
        finally:
            payload.close()
    return np.asarray(payload)


def _as_class_indices(target: np.ndarray) -> np.ndarray:
    values = np.asarray(target)
    if values.ndim == 1:
        return values.astype(np.int64)
    if values.ndim == 2:
        return np.argmax(values, axis=1).astype(np.int64)
    raise ValueError(f"target must have shape [examples] or [examples, speakers], got {values.shape}")


def instruction_following_report(predicted_scores: np.ndarray, target_speaker: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(predicted_scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"predicted_scores must have shape [examples, speakers], got {scores.shape}")
    targets = _as_class_indices(target_speaker)
    examples = min(scores.shape[0], targets.shape[0])
    if examples <= 0:
        raise ValueError("predicted_scores and target_speaker must contain at least one example")
    scores = scores[:examples]
    targets = targets[:examples]
    if np.any(targets < 0) or np.any(targets >= scores.shape[1]):
        raise ValueError("target_speaker contains class indices outside predicted_scores")
    predictions = np.argmax(scores, axis=1)
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2] if scores.shape[1] > 1 else sorted_scores[:, -1]
    return {
        "instruction_following": float(np.mean(predictions == targets)),
        "mean_instruction_margin": float(np.mean(margins)),
        "examples": int(examples),
        "speakers": int(scores.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute speaker-instruction following metrics.")
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--predicted-key", default="speaker_scores")
    parser.add_argument("--target-key", default="target_speaker")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = instruction_following_report(
        load_array(args.predicted, key=args.predicted_key),
        load_array(args.target, key=args.target_key),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
