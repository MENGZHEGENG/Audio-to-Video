"""Matched-seed, post-confirmation RAVDESS audio-versus-constant fit diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from a2v.data.ravdess_raster_feature_materialization import (
    TARGET_TIMESTAMP_SECONDS,
    _decode_audio,
    _decode_frame,
    _digest,
    _log_mel,
    require_registered_target_timestamp,
)
from a2v.evaluation.ravdess_raster_confirmation import _sha256, _tensor
from a2v.training.crema_raster_renderer import _RasterDataset, _metrics, _renderer_from_spec


def validate(spec: dict[str, Any], features: dict[str, Any], gate: dict[str, Any], confirmation: dict[str, Any], plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if plan.get("status") != "exploratory_seed_sensitivity_frozen_before_execution":
        raise ValueError("frozen exploratory plan required")
    if spec.get("status") != "ravdess_raster_renderer_spec_registered" or spec.get("source") != "RAVDESS":
        raise ValueError("registered RAVDESS specification required")
    require_registered_target_timestamp(spec)
    if features.get("status") != "ravdess_raster_features_materialized" or features.get("confirmation_actor_accessed") is not False:
        raise ValueError("unopened development and validation features required")
    if gate.get("status") != "ravdess_raster_features_provenance_gate_passed" or gate.get("feature_report_sha256") != _digest(features) or gate.get("spec_sha256") != _digest(spec):
        raise ValueError("feature provenance gate does not bind inputs")
    if confirmation.get("status") != "ravdess_raster_confirmation_protocol_registered" or confirmation.get("spec_sha256") != _digest(spec):
        raise ValueError("registered confirmation protocol required")
    if len(confirmation.get("pairs", [])) != plan["expected_confirmation_pairs"] or TARGET_TIMESTAMP_SECONDS != plan["target_timestamp_seconds"]:
        raise ValueError("confirmation target differs from plan")
    fit = spec["fitting"]
    if (fit["seed"], fit["batch_size"], fit["maximum_epochs"], fit["learning_rate"]) != (1701, 64, 40, 0.0003):
        raise ValueError("registered fitting contract differs")
    if sorted(plan["seeds"]) != [1702, 1703, 1704] or plan["conditions"] != ["aligned_audio_fit", "constant_zero_audio_fit"]:
        raise ValueError("seed and condition contract differs")
    dev = [row for row in features["records"] if row["role"] == "development"]
    val = [row for row in features["records"] if row["role"] == "validation"]
    if len(dev) != plan["expected_development_records"] or len(val) != plan["expected_validation_records"] or ({r["actor_id"] for r in dev} & {r["actor_id"] for r in val}):
        raise ValueError("development and validation population differs")
    return dev, val


def _metrics_for_mode(model: Any, loader: Any, device: Any, *, zero_audio: bool) -> dict[str, float]:
    if not zero_audio:
        return _metrics(model, loader, device)
    import torch

    model.eval()
    absolute = squared = count = 0.0
    with torch.inference_mode():
        for identity, audio, target in loader:
            identity, target = identity.to(device), target.to(device)
            prediction = model(identity, torch.zeros_like(audio, device=device))
            diff = prediction - target
            absolute += float(diff.abs().sum())
            squared += float(diff.square().sum())
            count += float(target.numel())
    return {"mae": absolute / count, "mse": squared / count}


def fit_one(spec: dict[str, Any], plan: dict[str, Any], *, seed: int, mode: str, dev: list[dict[str, Any]], val: list[dict[str, Any]], feature_root: Path, output_root: Path, device: Any, milestone_epoch: int | None = None) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    if seed not in plan["seeds"] or mode not in plan["conditions"]:
        raise ValueError("unplanned seed or mode")
    if milestone_epoch is not None and not (1 <= milestone_epoch <= int(spec["fitting"]["maximum_epochs"])):
        raise ValueError("milestone must fall within the training budget")
    zero = mode == "constant_zero_audio_fit"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = _renderer_from_spec(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["fitting"]["learning_rate"]), weight_decay=1e-4)
    batch_size = int(spec["fitting"]["batch_size"])
    dev_loader = DataLoader(_RasterDataset(dev, feature_root), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(_RasterDataset(val, feature_root), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    output_root.mkdir(parents=True, exist_ok=False)
    best, selected, stale, history = float("inf"), 0, 0, []
    milestone = None
    for epoch in range(1, int(spec["fitting"]["maximum_epochs"]) + 1):
        model.train()
        for identity, audio, target in dev_loader:
            identity, audio, target = identity.to(device), audio.to(device), target.to(device)
            optimizer.zero_grad()
            prediction = model(identity, torch.zeros_like(audio) if zero else audio)
            loss = (prediction - target).abs().mean()
            loss.backward()
            optimizer.step()
        metrics = _metrics_for_mode(model, val_loader, device, zero_audio=zero)
        history.append({"epoch": epoch, **metrics})
        if metrics["mae"] < best:
            best, selected, stale = metrics["mae"], epoch, 0
            temporary = output_root / f".checkpoint.{os.getpid()}.tmp"
            torch.save({"model_state_dict": model.state_dict(), "spec_sha256": _digest(spec), "plan_sha256": _digest(plan), "seed": seed, "mode": mode, "selected_epoch": epoch}, temporary)
            os.replace(temporary, output_root / "checkpoint.pt")
        else:
            stale += 1
        if epoch == milestone_epoch:
            milestone_path = output_root / f"checkpoint_at_epoch_{epoch}.pt"
            shutil.copyfile(output_root / "checkpoint.pt", milestone_path)
            milestone = {"epoch": epoch, "selected_epoch": selected, "checkpoint_sha256": _sha256(milestone_path), "validation_metrics": min(history, key=lambda row: row["mae"])}
            os.chmod(milestone_path, 0o444)
        if stale >= 6:
            break
    report = {"status": "exploratory_seed_condition_fitted", "seed": seed, "mode": mode, "selected_epoch": selected, "validation_metrics": min(history, key=lambda row: row["mae"]), "history": history, "spec_sha256": _digest(spec), "plan_sha256": _digest(plan), "checkpoint_sha256": _sha256(output_root / "checkpoint.pt"), "development_records": len(dev), "validation_records": len(val)}
    if milestone_epoch is not None:
        if milestone is None:
            milestone_path = output_root / f"checkpoint_at_epoch_{milestone_epoch}.pt"
            shutil.copyfile(output_root / "checkpoint.pt", milestone_path)
            milestone = {"epoch": len(history), "selected_epoch": selected, "checkpoint_sha256": _sha256(milestone_path), "validation_metrics": min(history, key=lambda row: row["mae"]), "stopped_before_milestone": True}
            os.chmod(milestone_path, 0o444)
        else:
            milestone["stopped_before_milestone"] = False
        report["milestone"] = milestone
    (output_root / "train_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.chmod(output_root / "checkpoint.pt", 0o444)
    return report


def score_pair(spec: dict[str, Any], confirmation: dict[str, Any], plan: dict[str, Any], *, seed: int, reports: dict[str, dict[str, Any]], output_root: Path, source_root: Path, ffmpeg: str, device: Any) -> dict[str, Any]:
    import torch

    from a2v.models.crema_audiofilm_raster import AudioFiLMRasterRenderer

    models = {}
    for mode in plan["conditions"]:
        checkpoint = output_root / mode / "checkpoint.pt"
        report = reports[mode]
        if report["seed"] != seed or report["mode"] != mode or report["checkpoint_sha256"] != _sha256(checkpoint):
            raise ValueError("checkpoint and report disagree")
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if payload["seed"] != seed or payload["mode"] != mode or payload["plan_sha256"] != _digest(plan) or payload["spec_sha256"] != _digest(spec):
            raise ValueError("checkpoint not bound to frozen plan")
        model = AudioFiLMRasterRenderer().to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        models[mode] = model
    rows = []
    with torch.inference_mode():
        for pair in confirmation["pairs"]:
            target_id = pair["target_clip_id"]
            identity = _decode_frame(source_root, ffmpeg, target_id, 0.25)
            target = _decode_frame(source_root, ffmpeg, target_id, TARGET_TIMESTAMP_SECONDS)
            audio = _log_mel(_decode_audio(source_root, ffmpeg, target_id))
            identity_tensor, audio_tensor = _tensor(identity, audio, torch, device)
            target_chw = target.transpose(2, 0, 1).astype(np.float32) / 255.0
            errors = {}
            for mode in plan["conditions"]:
                actual_audio = torch.zeros_like(audio_tensor) if mode == "constant_zero_audio_fit" else audio_tensor
                pred = models[mode](identity_tensor[None], actual_audio[None])[0].cpu().numpy()
                errors[mode] = float(np.abs(pred - target_chw).mean())
            rows.append({"pair_sha256": hashlib.sha256(target_id.encode()).hexdigest(), "actor_id": pair["target_actor_id"], "aligned_audio_fit_mae": errors["aligned_audio_fit"], "constant_zero_audio_fit_mae": errors["constant_zero_audio_fit"]})
    if len(rows) != 240 or len({row["pair_sha256"] for row in rows}) != 240:
        raise ValueError("240 unique pairs required")
    audio = np.asarray([row["aligned_audio_fit_mae"] for row in rows]);zero = np.asarray([row["constant_zero_audio_fit_mae"] for row in rows]);delta = zero - audio
    actors = sorted({row["actor_id"] for row in rows})
    if len(actors) != 4:
        raise ValueError("four confirmation actors required")
    summary = {"pairs": 240, "actors": actors, "aligned_audio_fit_mae": float(audio.mean()), "constant_zero_audio_fit_mae": float(zero.mean()), "constant_minus_audio_delta": float(delta.mean()), "by_actor_mean_delta": {actor: float(delta[[row["actor_id"] == actor for row in rows]].mean()) for actor in actors}}
    return {"status": "exploratory_seed_sensitivity_scored", "scope": "post-confirmation descriptive seed repeat; not a registered gate", "seed": seed, "target_timestamp_seconds": TARGET_TIMESTAMP_SECONDS, "spec_sha256": _digest(spec), "confirmation_protocol_sha256": _digest(confirmation), "plan_sha256": _digest(plan), "train_reports_sha256": {mode: _sha256(output_root / mode / "train_report.json") for mode in plan["conditions"]}, "summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "features", "feature-gate", "confirmation", "plan", "feature-root", "source-root", "ffmpeg", "output-root", "seed"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name.replace("-", "_"))) for name in ("spec", "features", "feature-gate", "confirmation", "plan")}
    spec, features, gate, confirmation, plan = [json.loads(paths[name].read_text()) for name in ("spec", "features", "feature-gate", "confirmation", "plan")]
    for name, path in paths.items():
        if name != "plan" and _sha256(path) != plan["inputs_sha256"][name.replace("-", "_") + "_json"]:
            raise ValueError("raw input hash mismatch: " + name)
    dev, val = validate(spec, features, gate, confirmation, plan)
    import torch

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    device = torch.device(args.device)
    seed = int(args.seed)
    if seed not in plan["seeds"]:
        raise ValueError("seed not in frozen plan")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    reports = {mode: fit_one(spec, plan, seed=seed, mode=mode, dev=dev, val=val, feature_root=Path(args.feature_root), output_root=output_root / mode, device=device) for mode in plan["conditions"]}
    scored = score_pair(spec, confirmation, plan, seed=seed, reports=reports, output_root=output_root, source_root=Path(args.source_root), ffmpeg=args.ffmpeg, device=device)
    (output_root / "score_report.json").write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": scored["status"], "seed": seed, "summary": scored["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
