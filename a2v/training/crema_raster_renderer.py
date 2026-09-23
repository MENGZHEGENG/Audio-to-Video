"""Development-only fitting for the frozen CREMA AudioFiLM raster renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from a2v.models.crema_audiofilm_raster import AudioConcatRasterRenderer, AudioFiLMRasterRenderer


def _renderer_from_spec(spec: dict[str, Any]) -> torch.nn.Module:
    name = str(spec.get("model", {}).get("name", "AudioFiLMRasterRenderer"))
    renderers = {"AudioFiLMRasterRenderer": AudioFiLMRasterRenderer, "AudioConcatRasterRenderer": AudioConcatRasterRenderer}
    if name not in renderers:
        raise ValueError(f"unsupported registered renderer: {name}")
    return renderers[name]()


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


class _RasterDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], root: Path) -> None:
        self.records, self.root = records, root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        data = np.load(self.root / self.records[index]["relative_path"])
        return tuple(torch.from_numpy(data[key].astype(np.float32)) / 255.0 if key != "audio_log_mel" else torch.from_numpy(data[key].astype(np.float32)) for key in ("identity_rgb", "audio_log_mel", "target_rgb"))


def _metrics(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval(); mae = mse = zero = swapped = total = 0.0
    with torch.no_grad():
        for identity, audio, target in loader:
            identity, audio, target = identity.to(device), audio.to(device), target.to(device)
            predicted = model(identity, audio)
            mae += float(torch.abs(predicted - target).sum()); mse += float(((predicted - target) ** 2).sum()); total += float(target.numel())
            zero += float((torch.abs(model(identity, torch.zeros_like(audio)) - target) - torch.abs(predicted - target)).sum())
            swapped_audio = audio.roll(1, dims=0) if audio.shape[0] > 1 else torch.zeros_like(audio)
            swapped += float((torch.abs(model(identity, swapped_audio) - target) - torch.abs(predicted - target)).sum())
    return {"mae": mae / total, "mse": mse / total, "zero_audio_mae_delta": zero / total, "actor_swapped_audio_mae_delta": swapped / total}


def train_renderer(spec: dict[str, Any], features: dict[str, Any], feature_gate: dict[str, Any], *, feature_root: Path, output_root: Path, device_name: str = "cuda") -> dict[str, Any]:
    if spec.get("status") != "crema_raster_renderer_spec_registered" or spec.get("training_allowed") is not True:
        raise ValueError("registered training-allowed renderer specification required")
    if features.get("status") != "crema_raster_features_materialized" or features.get("training_allowed") is not True:
        raise ValueError("gated materialized features required")
    if spec.get("protocol_sha256") != features.get("protocol_sha256"):
        raise ValueError("specification and features must bind the same protocol")
    if features.get("confirmation_actor_accessed") is not False:
        raise ValueError("confirmation data must not be accessed during fitting")
    if feature_gate.get("status") != "crema_raster_features_provenance_gate_passed" or feature_gate.get("training_allowed") is not True:
        raise ValueError("passed feature provenance gate required")
    if feature_gate.get("feature_report_sha256") != _digest(features) or feature_gate.get("protocol_sha256") != spec.get("protocol_sha256") or feature_gate.get("spec_sha256") != _digest(spec):
        raise ValueError("feature provenance gate does not bind this exact feature report and specification")
    development = [r for r in features["records"] if r["role"] == "development"]
    validation = [r for r in features["records"] if r["role"] == "validation"]
    if not development or not validation:
        raise ValueError("development and validation feature records are required")
    seed = int(spec["fitting"]["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = _renderer_from_spec(spec).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["fitting"]["learning_rate"]), weight_decay=1e-4)
    train_loader = DataLoader(_RasterDataset(development, feature_root), batch_size=int(spec["fitting"]["batch_size"]), shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(_RasterDataset(validation, feature_root), batch_size=int(spec["fitting"]["batch_size"]), shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    output_root.mkdir(parents=True, exist_ok=True)
    best, selected_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, int(spec["fitting"]["maximum_epochs"]) + 1):
        model.train()
        for identity, audio, target in train_loader:
            optimizer.zero_grad(); loss = torch.abs(model(identity.to(device), audio.to(device)) - target.to(device)).mean(); loss.backward(); optimizer.step()
        metrics = _metrics(model, validation_loader, device); history.append({"epoch": epoch, **metrics})
        if metrics["mae"] < best:
            best, selected_epoch, stale = metrics["mae"], epoch, 0
            temporary, checkpoint = output_root / f".checkpoint.{os.getpid()}.tmp", output_root / "checkpoint.pt"
            torch.save({"model_state_dict": model.state_dict(), "spec_sha256": _digest(spec), "selected_epoch": epoch, "validation_metrics": metrics}, temporary); os.replace(temporary, checkpoint)
        else:
            stale += 1
            if stale >= 6: break
    checkpoint = output_root / "checkpoint.pt"; checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {"status": "crema_raster_renderer_trained", "source": "CREMA-D", "model_name": spec.get("model", {}).get("name", "AudioFiLMRasterRenderer"), "scope": "development fit and actor-disjoint validation only", "device": str(device), "selected_epoch": selected_epoch, "validation_metrics": min(history, key=lambda item: item["mae"]), "history": history, "checkpoint_sha256": checkpoint_hash, "spec_sha256": _digest(spec), "training_allowed": False, "confirmation_allowed": False, "claim_allowed": False}
    (output_root / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = {"checkpoint_sha256": checkpoint_hash, "spec_sha256": _digest(spec), "development_actor_ids": sorted({r["actor_id"] for r in development}), "validation_actor_ids": sorted({r["actor_id"] for r in validation}), "seed": seed, "selected_epoch": selected_epoch, "validation_metrics": report["validation_metrics"]}
    manifest_path = output_root / "checkpoint_manifest.json"; manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n"); os.chmod(checkpoint, 0o444); os.chmod(manifest_path, 0o444)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True); parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True); parser.add_argument("--feature-gate", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); report = train_renderer(json.loads(args.spec.read_text()), json.loads(args.features.read_text()), json.loads(args.feature_gate.read_text()), feature_root=args.feature_root, output_root=args.output_root, device_name=args.device)
    print(json.dumps({"status": report["status"], "selected_epoch": report["selected_epoch"]}, sort_keys=True))


if __name__ == "__main__": main()
