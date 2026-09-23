"""Validate the released aggregate same-trajectory RAVDESS receipt."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def validate_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    if receipt.get("schema") != "civic-ravdess-trajectory-receipt/v1":
        raise ValueError("unexpected receipt schema")
    if receipt.get("targets_per_seed") != 240 or receipt.get("actor_groups") != 4:
        raise ValueError("target contract differs")
    if receipt.get("milestone_epoch") != 40 or receipt.get("maximum_epochs") != 80:
        raise ValueError("checkpoint schedule differs")
    rows = receipt.get("seed_results")
    if not isinstance(rows, list) or [row.get("seed") for row in rows] != [1701, 1702, 1703, 1704]:
        raise ValueError("four ordered seed results required")
    changed = []
    negative_at_cap = 0
    negative_final = 0
    for row in rows:
        selected = row["selected_epochs"]
        scores = row["mean_mae"]
        for mode in ("aligned_audio_fit", "constant_zero_audio_fit"):
            epochs = selected[mode]
            if not (1 <= epochs["at_cap"] <= 40 and epochs["at_cap"] <= epochs["final"] <= 80):
                raise ValueError("selected epoch is outside the checkpoint schedule")
            for stage in ("at_cap", "final"):
                score = scores[mode][stage]
                if not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
                    raise ValueError("MAE must be finite and nonnegative")
        for stage in ("at_cap", "final"):
            delta = scores["constant_zero_audio_fit"][stage] - scores["aligned_audio_fit"][stage]
            if not math.isclose(delta, row["constant_minus_audio_delta"][stage], rel_tol=0, abs_tol=1e-9):
                raise ValueError("constant-minus-aligned delta differs from scores")
            if stage == "at_cap":
                negative_at_cap += delta < 0
            else:
                negative_final += delta < 0
        for mode in ("aligned_audio_fit", "constant_zero_audio_fit"):
            observed = scores[mode]["final"] - scores[mode]["at_cap"]
            if not math.isclose(observed, row["after_minus_before_mae"][mode], rel_tol=0, abs_tol=1e-9):
                raise ValueError("within-trajectory difference differs from scores")
            if selected[mode]["final"] > selected[mode]["at_cap"]:
                changed.append({"seed": row["seed"], "condition": mode, "mae_change": observed})
            elif not math.isclose(observed, 0.0, rel_tol=0, abs_tol=1e-8):
                raise ValueError("unchanged selected checkpoint has different score")
    return {
        "negative_seed_count_at_cap": negative_at_cap,
        "negative_seed_count_final": negative_final,
        "changed_selected_checkpoints": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_receipt(json.loads(args.receipt.read_text())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
