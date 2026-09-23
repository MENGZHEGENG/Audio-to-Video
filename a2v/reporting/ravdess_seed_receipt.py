"""Verify the released aggregate RAVDESS seed-sensitivity receipt."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def validate_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("schema") != "civic-ravdess-seed-receipt/v1":
        raise ValueError("unexpected receipt schema")
    if receipt.get("target_timestamp_seconds") != 2.0 or receipt.get("targets_per_seed") != 240:
        raise ValueError("target contract differs")
    rows = receipt.get("seed_results")
    if not isinstance(rows, list) or [row.get("seed") for row in rows] != [1701, 1702, 1703, 1704]:
        raise ValueError("four ordered seeds required")
    gaps = []
    for row in rows:
        audio, constant, delta = (row[key] for key in ("aligned_audio_mae", "constant_audio_mae", "constant_minus_audio_mae"))
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (audio, constant, delta)):
            raise ValueError("finite numeric MAEs required")
        if not math.isclose(constant - audio, delta, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("seed delta does not match its operands")
        gaps.append(delta)
    return {"seeds": len(gaps), "negative_seed_count": sum(value < 0 for value in gaps), "mean_seed_delta": sum(gaps) / len(gaps), "min_seed_delta": min(gaps), "max_seed_delta": max(gaps)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_receipt(json.loads(args.receipt.read_text())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
