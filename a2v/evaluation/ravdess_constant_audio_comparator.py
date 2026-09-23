"""Exploratory same-architecture constant-audio comparator for RAVDESS rasters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from a2v.data.ravdess_raster_feature_materialization import (
    TARGET_TIMESTAMP_SECONDS,
    _decode_frame,
    _digest,
    require_registered_target_timestamp,
)
from a2v.evaluation.ravdess_raster_confirmation import _sha256, _tensor


def validate_inputs(spec: dict[str, Any], features: dict[str, Any], gate: dict[str, Any], plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the original split and selection settings; change only the audio values."""

    if plan.get("status") != "exploratory_constant_audio_comparator_frozen_before_scoring":
        raise ValueError("frozen exploratory plan required")
    if spec.get("status") != "ravdess_raster_renderer_spec_registered" or spec.get("source") != "RAVDESS":
        raise ValueError("registered RAVDESS renderer specification required")
    require_registered_target_timestamp(spec)
    if features.get("status") != "ravdess_raster_features_materialized" or features.get("confirmation_actor_accessed") is not False:
        raise ValueError("development and validation features without confirmation access required")
    if gate.get("status") != "ravdess_raster_features_provenance_gate_passed" or gate.get("feature_report_sha256") != _digest(features):
        raise ValueError("feature provenance gate does not bind the features")
    if spec.get("protocol_sha256") != features.get("protocol_sha256") or gate.get("spec_sha256") != _digest(spec):
        raise ValueError("specification and features disagree")
    if spec.get("model", {}).get("name") != "AudioFiLMRasterRenderer":
        raise ValueError("the same registered architecture is required")
    fit = spec["fitting"]
    if (fit.get("seed"), fit.get("batch_size"), fit.get("maximum_epochs"), fit.get("learning_rate")) != (1701, 64, 40, 0.0003):
        raise ValueError("registered fitting settings changed")
    development = [row for row in features["records"] if row["role"] == "development"]
    validation = [row for row in features["records"] if row["role"] == "validation"]
    if len(development) != plan["expected_development_records"] or len(validation) != plan["expected_validation_records"]:
        raise ValueError("feature population differs from the frozen plan")
    if {row["actor_id"] for row in development} & {row["actor_id"] for row in validation}:
        raise ValueError("development and validation actors overlap")
    return development, validation


def _zero_audio_metrics(model: Any, loader: Any, device: Any) -> dict[str, float]:
    import torch

    model.eval()
    absolute = squared = count = 0.0
    with torch.inference_mode():
        for identity, audio, target in loader:
            identity, target = identity.to(device), target.to(device)
            prediction = model(identity, torch.zeros_like(audio, device=device))
            difference = prediction - target
            absolute += float(difference.abs().sum())
            squared += float(difference.square().sum())
            count += float(target.numel())
    return {"mae": absolute / count, "mse": squared / count}


def train(spec: dict[str, Any], features: dict[str, Any], gate: dict[str, Any], plan: dict[str, Any], *, feature_root: Path, output_root: Path, device_name: str) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from a2v.training.crema_raster_renderer import _RasterDataset, _renderer_from_spec

    development, validation = validate_inputs(spec, features, gate, plan)
    if device_name != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("requested GPU unavailable")
    device = torch.device(device_name)
    seed = int(spec["fitting"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = _renderer_from_spec(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["fitting"]["learning_rate"]), weight_decay=1e-4)
    batch_size = int(spec["fitting"]["batch_size"])
    train_loader = DataLoader(_RasterDataset(development, feature_root), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(_RasterDataset(validation, feature_root), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    output_root.mkdir(parents=True, exist_ok=False)
    best, selected_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, int(spec["fitting"]["maximum_epochs"]) + 1):
        model.train()
        for identity, audio, target in train_loader:
            identity, target = identity.to(device), target.to(device)
            optimizer.zero_grad()
            loss = (model(identity, torch.zeros_like(audio, device=device)) - target).abs().mean()
            loss.backward()
            optimizer.step()
        metrics = _zero_audio_metrics(model, validation_loader, device)
        history.append({"epoch": epoch, **metrics})
        if metrics["mae"] < best:
            best, selected_epoch, stale = metrics["mae"], epoch, 0
            temporary = output_root / f".checkpoint.{os.getpid()}.tmp"
            torch.save({"model_state_dict": model.state_dict(), "spec_sha256": _digest(spec), "plan_sha256": _digest(plan), "selected_epoch": epoch, "validation_metrics": metrics}, temporary)
            os.replace(temporary, output_root / "checkpoint.pt")
        else:
            stale += 1
            if stale >= 6:
                break
    checkpoint = output_root / "checkpoint.pt"
    report = {
        "status": "exploratory_constant_audio_renderer_trained",
        "source": "RAVDESS",
        "scope": "same-architecture retrospective comparator fitted on development actors and selected on validation actors",
        "audio_input_during_fitting": "constant_zero_for_every_example",
        "spec_sha256": _digest(spec),
        "features_sha256": _digest(features),
        "feature_gate_sha256": _digest(gate),
        "plan_sha256": _digest(plan),
        "checkpoint_sha256": _sha256(checkpoint),
        "selected_epoch": selected_epoch,
        "validation_metrics": min(history, key=lambda item: item["mae"]),
        "history": history,
        "development_records": len(development),
        "validation_records": len(validation),
        "development_actors": sorted({row["actor_id"] for row in development}),
        "validation_actors": sorted({row["actor_id"] for row in validation}),
    }
    (output_root / "train_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.chmod(checkpoint, 0o444)
    return report


def summarize(rows: list[dict[str, Any]], *, seed: int, replicates: int) -> dict[str, Any]:
    if len(rows) != 240 or replicates < 1000:
        raise ValueError("240 complete pairs and at least 1,000 bootstrap replicates required")
    actors = sorted({row["actor_id"] for row in rows})
    if len(actors) != 4 or len({row["pair_sha256"] for row in rows}) != 240:
        raise ValueError("four actors and unique confirmation pairs required")
    comparator = np.asarray([row["constant_audio_mae"] for row in rows], dtype=np.float64)
    original = np.asarray([row["original_aligned_audio_mae"] for row in rows], dtype=np.float64)
    if not np.isfinite(comparator).all() or not np.isfinite(original).all():
        raise ValueError("nonfinite error")
    delta = comparator - original
    group = np.asarray([row["actor_id"] for row in rows])
    rng = np.random.default_rng(seed)
    pair_means = delta[rng.integers(0, len(rows), size=(replicates, len(rows)))].mean(axis=1)
    actor_means = np.asarray([delta[group == actor].mean() for actor in actors])
    actor_samples = actor_means[rng.integers(0, len(actors), size=(replicates, len(actors)))].mean(axis=1)
    return {
        "pairs": len(rows), "actors": actors,
        "original_aligned_audio_mae": float(original.mean()),
        "constant_audio_trained_mae": float(comparator.mean()),
        "constant_minus_original_mean_delta": float(delta.mean()),
        "pair_bootstrap_95": [float(x) for x in np.quantile(pair_means, [0.025, 0.975])],
        "actor_bootstrap_95": [float(x) for x in np.quantile(actor_samples, [0.025, 0.975])],
        "by_actor_mean_delta": {actor: float(value) for actor, value in zip(actors, actor_means)},
    }


def score(confirmation: dict[str, Any], spec: dict[str, Any], plan: dict[str, Any], train_report: dict[str, Any], original_report: dict[str, Any], *, checkpoint: Path, source_root: Path, ffmpeg: str, device_name: str) -> dict[str, Any]:
    import torch

    from a2v.models.crema_audiofilm_raster import AudioFiLMRasterRenderer

    if confirmation.get("status") != "ravdess_raster_confirmation_protocol_registered" or _digest(spec) != confirmation.get("spec_sha256"):
        raise ValueError("registered confirmation protocol and specification required")
    require_registered_target_timestamp(spec)
    if plan.get("expected_confirmation_pairs") != len(confirmation["pairs"]) or len(confirmation["pairs"]) != 240:
        raise ValueError("confirmation set differs from frozen plan")
    if train_report.get("status") != "exploratory_constant_audio_renderer_trained" or train_report.get("plan_sha256") != _digest(plan) or train_report.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("constant-audio checkpoint and training report disagree")
    if train_report.get("spec_sha256") != _digest(spec) or original_report.get("status") != "exploratory_ravdess_factorial_scored":
        raise ValueError("comparator and original report are incompatible")
    if original_report.get("checkpoint_sha256") != plan["inputs_sha256"]["original_checkpoint_pt"] or len(original_report["rows"]) != 240:
        raise ValueError("original aligned report differs from frozen plan")
    original = {row["pair_sha256"]: row for row in original_report["rows"]}
    if len(original) != 240:
        raise ValueError("original paired rows are duplicated")
    if device_name != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("requested GPU unavailable")
    device = torch.device(device_name)
    model = AudioFiLMRasterRenderer().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("plan_sha256") != _digest(plan) or payload.get("spec_sha256") != _digest(spec):
        raise ValueError("checkpoint payload does not bind the frozen plan")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for pair in confirmation["pairs"]:
            target_id = pair["target_clip_id"]
            target_path = source_root / target_id
            if not target_path.is_file():
                raise FileNotFoundError(target_path)
            identity = _decode_frame(source_root, ffmpeg, target_id, 0.25)
            target = _decode_frame(source_root, ffmpeg, target_id, TARGET_TIMESTAMP_SECONDS)
            identity_tensor, zero_audio = _tensor(identity, np.zeros((40, 48), dtype=np.float32), torch, device)
            prediction = model(identity_tensor[None], zero_audio[None])[0].cpu().numpy()
            target_chw = target.transpose(2, 0, 1).astype(np.float32) / 255.0
            key = hashlib.sha256(target_id.encode()).hexdigest()
            reference = original.get(key)
            if reference is None or reference["actor_id"] != pair["target_actor_id"]:
                raise ValueError("pair-level original report does not match confirmation target")
            rows.append({"pair_sha256": key, "actor_id": pair["target_actor_id"], "constant_audio_mae": float(np.abs(prediction - target_chw).mean()), "original_aligned_audio_mae": float(reference["errors"]["aligned"])})
    return {
        "status": "exploratory_constant_audio_comparator_scored",
        "scope": "same-target retrospective diagnostic after confirmation actors were opened; not a registered benefit gate",
        "target_timestamp_seconds": TARGET_TIMESTAMP_SECONDS,
        "plan_sha256": _digest(plan),
        "spec_sha256": _digest(spec),
        "confirmation_protocol_sha256": _digest(confirmation),
        "comparator_checkpoint_sha256": _sha256(checkpoint),
        "original_report_sha256": plan["inputs_sha256"]["original_pair_report_json"],
        "metadata_used_for_pairing": True,
        "external_performance_labels_read": False,
        "stimulus_codes_passed_to_model": False,
        "summary": summarize(rows, seed=int(plan["bootstrap_seed"]), replicates=int(plan["bootstrap_replicates"])),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    for name in ("spec", "features", "feature-gate", "plan", "feature-root", "output-root"):
        training.add_argument("--" + name, type=Path, required=True)
    training.add_argument("--device", default="cuda")
    scoring = sub.add_parser("score")
    for name in ("confirmation", "spec", "plan", "train-report", "original-report", "checkpoint", "source-root", "output"):
        scoring.add_argument("--" + name, type=Path, required=True)
    scoring.add_argument("--ffmpeg", required=True)
    scoring.add_argument("--device", default="cuda")
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text())
    if args.command == "train":
        plan = load(args.plan)
        for field, path in (("spec_json", args.spec), ("features_json", args.features), ("feature_gate_json", args.feature_gate)):
            if _sha256(path) != plan["inputs_sha256"][field]:
                raise ValueError(f"{field} differs from the frozen plan")
        result = train(load(args.spec), load(args.features), load(args.feature_gate), plan, feature_root=args.feature_root, output_root=args.output_root, device_name=args.device)
    else:
        plan = load(args.plan)
        for field, path in (("confirmation_json", args.confirmation), ("spec_json", args.spec), ("original_pair_report_json", args.original_report)):
            if _sha256(path) != plan["inputs_sha256"][field]:
                raise ValueError(f"{field} differs from the frozen plan")
        result = score(load(args.confirmation), load(args.spec), plan, load(args.train_report), load(args.original_report), checkpoint=args.checkpoint, source_root=args.source_root, ffmpeg=args.ffmpeg, device_name=args.device)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
