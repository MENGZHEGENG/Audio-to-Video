"""Score before/after-cap checkpoints from each continued RAVDESS fit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from a2v.data.ravdess_raster_feature_materialization import (
    TARGET_TIMESTAMP_SECONDS,
    _decode_audio,
    _decode_frame,
    _digest,
    _log_mel,
)
from a2v.evaluation.ravdess_budget_sensitivity import check_plan as check_budget_plan
from a2v.evaluation.ravdess_raster_confirmation import _sha256, _tensor


def check_plan(plan: dict[str, Any], budget_plan: dict[str, Any]) -> None:
    if plan.get("status") != "exploratory_same_trajectory_plan_frozen_before_execution":
        raise ValueError("frozen same-trajectory plan required")
    if plan.get("budget_plan_sha256") != _digest(budget_plan):
        raise ValueError("budget plan digest mismatch")
    for field in ("seeds", "conditions", "inputs_sha256", "expected_development_records", "expected_validation_records", "expected_confirmation_pairs", "target_timestamp_seconds", "maximum_epochs", "patience"):
        if plan.get(field) != budget_plan.get(field):
            raise ValueError(f"budget setting changed: {field}")
    if plan.get("milestone_epoch") != 40:
        raise ValueError("the registered maximum epoch must be the milestone")


def score_checkpoints(
    spec: dict[str, Any], confirmation: dict[str, Any], budget_plan: dict[str, Any],
    trajectory_plan: dict[str, Any], *, seed: int, reports: dict[str, dict[str, Any]],
    output_root: Path, source_root: Path, ffmpeg: str, device: Any,
) -> dict[str, Any]:
    import torch
    from a2v.training.crema_raster_renderer import _renderer_from_spec

    models = {}
    checkpoint_hashes = {}
    for mode in trajectory_plan["conditions"]:
        report = reports[mode]
        if report["seed"] != seed or report["mode"] != mode or report["spec_sha256"] != _digest(spec) or report["plan_sha256"] != _digest(budget_plan):
            raise ValueError("fit report and frozen plan disagree")
        for stage, name, expected_hash, expected_epoch in (
            ("at_cap", "checkpoint_at_epoch_40.pt", report["milestone"]["checkpoint_sha256"], report["milestone"]["selected_epoch"]),
            ("final", "checkpoint.pt", report["checkpoint_sha256"], report["selected_epoch"]),
        ):
            checkpoint = output_root / mode / name
            actual_hash = _sha256(checkpoint)
            if actual_hash != expected_hash:
                raise ValueError("checkpoint hash differs from fit report")
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
            if (payload["seed"], payload["mode"], payload["plan_sha256"], payload["spec_sha256"], payload["selected_epoch"]) != (seed, mode, _digest(budget_plan), _digest(spec), expected_epoch):
                raise ValueError("checkpoint metadata differs from frozen fit")
            model = _renderer_from_spec(spec).to(device)
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            models[(mode, stage)] = model
            checkpoint_hashes[f"{mode}/{stage}"] = actual_hash

    rows = []
    with torch.inference_mode():
        for pair in confirmation["pairs"]:
            target_id = pair["target_clip_id"]
            identity = _decode_frame(source_root, ffmpeg, target_id, 0.25)
            target = _decode_frame(source_root, ffmpeg, target_id, TARGET_TIMESTAMP_SECONDS)
            audio = _log_mel(_decode_audio(source_root, ffmpeg, target_id))
            identity_tensor, audio_tensor = _tensor(identity, audio, torch, device)
            target_chw = target.transpose(2, 0, 1).astype(np.float32) / 255.0
            row: dict[str, Any] = {
                "pair_sha256": hashlib.sha256(target_id.encode()).hexdigest(),
                "actor_id": pair["target_actor_id"],
            }
            for mode in trajectory_plan["conditions"]:
                model_audio = torch.zeros_like(audio_tensor) if mode == "constant_zero_audio_fit" else audio_tensor
                for stage in ("at_cap", "final"):
                    prediction = models[(mode, stage)](identity_tensor[None], model_audio[None])[0].cpu().numpy()
                    row[f"{mode}_{stage}_mae"] = float(np.abs(prediction - target_chw).mean())
            rows.append(row)
    if len(rows) != 240 or len({row["pair_sha256"] for row in rows}) != 240:
        raise ValueError("240 unique confirmation targets required")
    actors = sorted({row["actor_id"] for row in rows})
    if len(actors) != 4 or any(sum(row["actor_id"] == actor for row in rows) != 60 for actor in actors):
        raise ValueError("four actors with 60 targets each required")
    summary: dict[str, Any] = {"pairs": len(rows), "actors": actors, "selected_epochs": {}, "mean_mae": {}, "after_minus_before_mae": {}, "by_actor_after_minus_before_mae": {}}
    for mode in trajectory_plan["conditions"]:
        summary["selected_epochs"][mode] = {"at_cap": reports[mode]["milestone"]["selected_epoch"], "final": reports[mode]["selected_epoch"], "stopped_before_cap": reports[mode]["milestone"]["stopped_before_milestone"]}
        before = np.asarray([row[f"{mode}_at_cap_mae"] for row in rows])
        after = np.asarray([row[f"{mode}_final_mae"] for row in rows])
        summary["mean_mae"][mode] = {"at_cap": float(before.mean()), "final": float(after.mean())}
        summary["after_minus_before_mae"][mode] = float((after - before).mean())
        summary["by_actor_after_minus_before_mae"][mode] = {actor: float((after - before)[[row["actor_id"] == actor for row in rows]].mean()) for actor in actors}
    summary["constant_minus_audio_delta"] = {
        stage: float(np.mean([row[f"constant_zero_audio_fit_{stage}_mae"] - row[f"aligned_audio_fit_{stage}_mae"] for row in rows]))
        for stage in ("at_cap", "final")
    }
    return {
        "status": "exploratory_same_trajectory_scored",
        "scope": "retrospective within-trajectory selection check on previously opened actors",
        "seed": seed,
        "registered_spec_sha256": trajectory_plan["inputs_sha256"]["spec_json"],
        "extended_spec_sha256": _digest(spec),
        "confirmation_protocol_sha256": _digest(confirmation),
        "budget_plan_sha256": _digest(budget_plan),
        "trajectory_plan_sha256": _digest(trajectory_plan),
        "target_timestamp_seconds": TARGET_TIMESTAMP_SECONDS,
        "train_reports_sha256": {mode: _sha256(output_root / mode / "train_report.json") for mode in trajectory_plan["conditions"]},
        "checkpoint_sha256": checkpoint_hashes,
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    from a2v.evaluation.ravdess_seed_sensitivity import fit_one, validate

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "features", "feature-gate", "confirmation", "base-plan", "budget-plan", "trajectory-plan", "feature-root", "source-root", "ffmpeg", "output-root", "seed"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    input_names = ("spec", "features", "feature-gate", "confirmation", "base-plan", "budget-plan", "trajectory-plan")
    paths = {name: Path(getattr(args, name.replace("-", "_"))) for name in input_names}
    spec, features, gate, confirmation, base_plan, budget_plan, trajectory_plan = [json.loads(paths[name].read_text()) for name in input_names]
    check_budget_plan(budget_plan, base_plan, spec)
    check_plan(trajectory_plan, budget_plan)
    for name in ("spec", "features", "feature-gate", "confirmation"):
        key = name.replace("-", "_") + "_json"
        if _sha256(paths[name]) != trajectory_plan["inputs_sha256"][key]:
            raise ValueError("raw input hash mismatch: " + name)
    dev, val = validate(spec, features, gate, confirmation, base_plan)
    seed = int(args.seed)
    if seed not in trajectory_plan["seeds"]:
        raise ValueError("unplanned seed")
    import torch

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    device = torch.device(args.device)
    extended_spec = copy.deepcopy(spec)
    extended_spec["fitting"]["maximum_epochs"] = trajectory_plan["maximum_epochs"]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    reports = {
        mode: fit_one(extended_spec, budget_plan, seed=seed, mode=mode, dev=dev, val=val,
                      feature_root=Path(args.feature_root), output_root=output_root / mode,
                      device=device, milestone_epoch=trajectory_plan["milestone_epoch"])
        for mode in trajectory_plan["conditions"]
    }
    scored = score_checkpoints(extended_spec, confirmation, budget_plan, trajectory_plan,
                               seed=seed, reports=reports, output_root=output_root,
                               source_root=Path(args.source_root), ffmpeg=args.ffmpeg,
                               device=device)
    (output_root / "score_report.json").write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": scored["status"], "seed": seed, "summary": scored["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
