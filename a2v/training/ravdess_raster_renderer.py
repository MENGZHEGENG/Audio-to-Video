"""Development-only fitting for the frozen RAVDESS AudioFiLM raster renderer."""

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
from torch.utils.data import DataLoader

from a2v.training.crema_raster_renderer import _RasterDataset, _metrics, _renderer_from_spec


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def train_renderer(spec: dict[str, Any], features: dict[str, Any], feature_gate: dict[str, Any], *, feature_root: Path, output_root: Path, device_name: str = "cuda") -> dict[str, Any]:
    """Fit only development actors and select only on validation actors."""

    if spec.get("status") != "ravdess_raster_renderer_spec_registered" or spec.get("source") != "RAVDESS" or spec.get("training_allowed") is not True:
        raise ValueError("registered training-allowed RAVDESS renderer specification required")
    if features.get("status") != "ravdess_raster_features_materialized" or features.get("source") != "RAVDESS" or features.get("training_allowed") is not True:
        raise ValueError("gated RAVDESS materialized features required")
    if spec.get("protocol_sha256") != features.get("protocol_sha256") or features.get("confirmation_actor_accessed") is not False:
        raise ValueError("specification and non-confirmation features must bind the same protocol")
    if feature_gate.get("status") != "ravdess_raster_features_provenance_gate_passed" or feature_gate.get("training_allowed") is not True:
        raise ValueError("passed RAVDESS feature provenance gate required")
    if feature_gate.get("feature_report_sha256") != _digest(features) or feature_gate.get("protocol_sha256") != spec.get("protocol_sha256") or feature_gate.get("spec_sha256") != _digest(spec):
        raise ValueError("feature provenance gate does not bind this exact feature report and specification")
    development = [record for record in features.get("records", []) if record.get("role") == "development"]
    validation = [record for record in features.get("records", []) if record.get("role") == "validation"]
    if not development or not validation:
        raise ValueError("development and validation feature records are required")

    seed = int(spec["fitting"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = _renderer_from_spec(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["fitting"]["learning_rate"]), weight_decay=1e-4)
    batch_size = int(spec["fitting"]["batch_size"])
    train_loader = DataLoader(_RasterDataset(development, feature_root), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(_RasterDataset(validation, feature_root), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    output_root.mkdir(parents=True, exist_ok=True)
    best, selected_epoch, stale, history = float("inf"), 0, 0, []
    for epoch in range(1, int(spec["fitting"]["maximum_epochs"]) + 1):
        model.train()
        for identity, audio, target in train_loader:
            optimizer.zero_grad()
            loss = torch.abs(model(identity.to(device), audio.to(device)) - target.to(device)).mean()
            loss.backward(); optimizer.step()
        metrics = _metrics(model, validation_loader, device)
        history.append({"epoch": epoch, **metrics})
        if metrics["mae"] < best:
            best, selected_epoch, stale = metrics["mae"], epoch, 0
            temporary, checkpoint = output_root / f".checkpoint.{os.getpid()}.tmp", output_root / "checkpoint.pt"
            torch.save({"model_state_dict": model.state_dict(), "spec_sha256": _digest(spec), "selected_epoch": epoch, "validation_metrics": metrics}, temporary)
            os.replace(temporary, checkpoint)
        else:
            stale += 1
            if stale >= 6:
                break
    checkpoint = output_root / "checkpoint.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "status": "ravdess_raster_renderer_trained", "source": "RAVDESS", "model_name": spec.get("model", {}).get("name", "AudioFiLMRasterRenderer"),
        "scope": "development fit and actor-disjoint validation only", "device": str(device), "selected_epoch": selected_epoch,
        "validation_metrics": min(history, key=lambda item: item["mae"]), "history": history, "checkpoint_sha256": checkpoint_hash,
        "spec_sha256": _digest(spec), "training_allowed": False, "confirmation_allowed": False, "claim_allowed": False,
    }
    (output_root / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = {"checkpoint_sha256": checkpoint_hash, "spec_sha256": _digest(spec), "development_actor_ids": sorted({record["actor_id"] for record in development}), "validation_actor_ids": sorted({record["actor_id"] for record in validation}), "seed": seed, "selected_epoch": selected_epoch, "validation_metrics": report["validation_metrics"]}
    manifest_path = output_root / "checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(checkpoint, 0o444); os.chmod(manifest_path, 0o444)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--feature-gate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = train_renderer(json.loads(args.spec.read_text()), json.loads(args.features.read_text()), json.loads(args.feature_gate.read_text()), feature_root=args.feature_root, output_root=args.output_root, device_name=args.device)
    print(json.dumps({"status": report["status"], "selected_epoch": report["selected_epoch"]}, sort_keys=True))


if __name__ == "__main__":
    main()
