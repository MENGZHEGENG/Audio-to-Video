"""Speaker-order sensitivity diagnostics for multi-person A2V predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_KEYS = ("motion", "articulation", "expression", "social_reaction")


def parse_permutation(value: str) -> tuple[int, ...]:
    permutation = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not permutation:
        raise ValueError("permutation must contain at least one index")
    if sorted(permutation) != list(range(len(permutation))):
        raise ValueError(f"permutation must contain each index once from 0 to {len(permutation) - 1}: {permutation}")
    return permutation


def inverse_permutation(permutation: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(item) for item in permutation)
    inverse = [0] * len(values)
    for source_index, permuted_index in enumerate(values):
        inverse[permuted_index] = source_index
    return tuple(inverse)


def speaker_order_delta(
    reference: np.ndarray,
    permuted: np.ndarray,
    *,
    permutation: tuple[int, ...],
    speaker_axis: int = -2,
) -> float:
    if reference.shape != permuted.shape:
        raise ValueError(f"prediction shapes must match: {reference.shape} vs {permuted.shape}")
    axis = speaker_axis if speaker_axis >= 0 else reference.ndim + speaker_axis
    if axis < 0 or axis >= reference.ndim:
        raise ValueError(f"speaker_axis {speaker_axis} out of bounds for shape {reference.shape}")
    if reference.shape[axis] != len(permutation):
        raise ValueError(
            f"permutation length {len(permutation)} does not match speaker dimension {reference.shape[axis]}"
        )
    aligned = np.take(permuted, inverse_permutation(permutation), axis=axis)
    return float(np.mean(np.square(reference.astype("float32") - aligned.astype("float32"))))


def _load_arrays(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    if path.suffix == ".npy":
        return {"motion": np.load(path)}
    with np.load(path) as payload:
        arrays = {key: payload[key] for key in keys if key in payload}
    if not arrays:
        raise ValueError(f"no requested prediction keys found in {path}: {keys}")
    return arrays


def speaker_order_report(
    *,
    reference_path: Path,
    permuted_path: Path,
    permutation: tuple[int, ...],
    keys: tuple[str, ...] = DEFAULT_KEYS,
    speaker_axis: int = -2,
) -> dict[str, object]:
    reference_arrays = _load_arrays(reference_path, keys)
    permuted_arrays = _load_arrays(permuted_path, keys)
    common_keys = tuple(key for key in keys if key in reference_arrays and key in permuted_arrays)
    if not common_keys:
        raise ValueError("reference and permuted predictions share no requested keys")
    per_key = {
        key: speaker_order_delta(
            reference_arrays[key],
            permuted_arrays[key],
            permutation=permutation,
            speaker_axis=speaker_axis,
        )
        for key in common_keys
    }
    return {
        "reference": str(reference_path),
        "permuted": str(permuted_path),
        "permutation": permutation,
        "speaker_axis": speaker_axis,
        "keys": common_keys,
        "per_key": per_key,
        "speaker_order_delta": float(sum(per_key.values()) / len(per_key)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute speaker-order sensitivity between paired predictions.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--permuted", type=Path, required=True)
    parser.add_argument("--permutation", required=True, help="Comma-separated speaker permutation used for the permuted prediction.")
    parser.add_argument("--keys", default=",".join(DEFAULT_KEYS), help="Comma-separated .npz keys to compare.")
    parser.add_argument("--speaker-axis", type=int, default=-2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = speaker_order_report(
        reference_path=args.reference,
        permuted_path=args.permuted,
        permutation=parse_permutation(args.permutation),
        keys=tuple(item.strip() for item in args.keys.split(",") if item.strip()),
        speaker_axis=args.speaker_axis,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
