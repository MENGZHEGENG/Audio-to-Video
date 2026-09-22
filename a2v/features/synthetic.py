"""Generate tiny deterministic feature packs for smoke tests.

This module is only for validating I/O contracts. It does not create paper
results and should not be used as a substitute for real extracted features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from a2v.features.schema import FeatureRecord, FeatureStream, load_feature_index


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def synthetic_values(record: FeatureRecord, stream: FeatureStream) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(record.clip_id, stream.stream_type))
    values = rng.normal(loc=0.0, scale=0.05, size=(stream.frames, stream.dim)).astype("float32")
    if stream.stream_type == "audio":
        values[:, 0] += np.linspace(0.0, 1.0, stream.frames, dtype="float32")
    if stream.stream_type == "face":
        values[:, 0] += np.sin(np.linspace(0.0, np.pi, stream.frames, dtype="float32"))
    return values


def write_synthetic_feature_pack(
    index_path: Path,
    output_root: Path,
    updated_index_path: Path,
) -> int:
    records = load_feature_index(index_path)
    updated_records: list[FeatureRecord] = []
    count = 0
    for record in records:
        updated_streams: list[FeatureStream] = []
        for stream in record.streams:
            stream_path = output_root / record.clip_id / f"{stream.stream_type}.npz"
            stream_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                stream_path,
                values=synthetic_values(record, stream),
                stream_type=np.array(stream.stream_type),
                clip_id=np.array(record.clip_id),
                rate_hz=np.array(stream.rate_hz, dtype="float32"),
            )
            updated_streams.append(replace(stream, path=str(stream_path)))
            count += 1
        updated_records.append(replace(record, streams=tuple(updated_streams)))

    updated_index_path.parent.mkdir(parents=True, exist_ok=True)
    with updated_index_path.open("w", encoding="utf-8") as handle:
        for record in updated_records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tiny synthetic feature packs.")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--updated-index", type=Path, required=True)
    args = parser.parse_args()
    count = write_synthetic_feature_pack(args.index, args.output_root, args.updated_index)
    print(f"wrote_feature_streams={count}")
    print(f"updated_index={args.updated_index}")


if __name__ == "__main__":
    main()
