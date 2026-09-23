"""Score a sealed CREMA renderer on the pre-registered held-out composition controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from a2v.data.crema_raster_feature_materialization import _decode_audio, _decode_frame, _digest, _log_mel


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_checkpoint_gate(checkpoint: Path, manifest: dict[str, Any], report: dict[str, Any], confirmation_protocol: dict[str, Any]) -> None:
    required = {"checkpoint_sha256", "spec_sha256", "development_actor_ids", "validation_actor_ids", "selected_epoch", "validation_metrics"}
    if not checkpoint.is_file() or not required.issubset(manifest):
        raise ValueError("sealed checkpoint manifest is incomplete")
    checkpoint_sha256 = _sha256(checkpoint)
    if manifest["checkpoint_sha256"] != checkpoint_sha256 or report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("checkpoint hash does not match sealed artifacts")
    if report.get("status") != "crema_raster_renderer_trained" or report.get("training_allowed") is not False:
        raise ValueError("completed closed training report required")
    if manifest["spec_sha256"] != confirmation_protocol.get("spec_sha256") or report.get("spec_sha256") != confirmation_protocol.get("spec_sha256"):
        raise ValueError("checkpoint does not bind the frozen confirmation specification")


def summarize_controls(rows: list[dict[str, float]], confirmation_protocol: dict[str, Any]) -> dict[str, Any]:
    conditions = confirmation_protocol.get("required_conditions")
    rule = confirmation_protocol.get("decision_rule", {})
    controls = rule.get("controls", [])
    if conditions != ["aligned", "zero_audio", "actor_swapped_audio", "identity_reference_swap"] or not rows:
        raise ValueError("complete registered condition rows required")
    if any(set(row) != set(conditions) or not all(np.isfinite(value) for value in row.values()) for row in rows):
        raise ValueError("finite metric for every registered condition required")
    generator = np.random.default_rng(int(rule["seed"]))
    count, replicates = len(rows), int(rule["bootstrap_replicates"])
    if replicates <= 0:
        raise ValueError("positive bootstrap replicate count required")
    aligned = np.asarray([row["aligned"] for row in rows], dtype=np.float64)
    details: dict[str, dict[str, float]] = {}
    for control in controls:
        values = np.asarray([row[control] for row in rows], dtype=np.float64) - aligned
        samples = values[generator.integers(0, count, size=(replicates, count))].mean(axis=1)
        details[control] = {"mean_delta": float(values.mean()), "lower_95": float(np.quantile(samples, 0.025)), "upper_95": float(np.quantile(samples, 0.975))}
    return {"rows": count, "aligned_mae": float(aligned.mean()), "control_deltas": details, "claim_allowed": all(detail["lower_95"] > 0.0 for detail in details.values())}


def _tensor(frame: np.ndarray, audio: np.ndarray, torch, device):
    return torch.from_numpy(frame.transpose(2, 0, 1).astype(np.float32) / 255.0).to(device), torch.from_numpy(audio.astype(np.float32)).to(device)


def run_confirmation(
    confirmation_protocol: dict[str, Any], protocol: dict[str, Any], spec: dict[str, Any], *, checkpoint: Path, manifest: dict[str, Any], train_report: dict[str, Any], source_root: Path, ffmpeg: str, device_name: str
) -> dict[str, Any]:
    if confirmation_protocol.get("status") != "crema_raster_confirmation_protocol_registered" or confirmation_protocol.get("confirmation_allowed") is not False:
        raise ValueError("closed registered confirmation protocol required")
    if _digest(protocol) != confirmation_protocol.get("protocol_sha256") or _digest(spec) != confirmation_protocol.get("spec_sha256"):
        raise ValueError("confirmation protocol must bind the exact protocol and specification")
    validate_checkpoint_gate(checkpoint, manifest, train_report, confirmation_protocol)
    import torch
    from a2v.models.crema_audiofilm_raster import AudioFiLMRasterRenderer

    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = AudioFiLMRasterRenderer().to(device)
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    if payload.get("spec_sha256") != confirmation_protocol["spec_sha256"]:
        raise ValueError("checkpoint payload has an unexpected specification hash")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    rows, composition_hashes = [], []
    with torch.no_grad():
        for pair in confirmation_protocol["pairs"]:
            target_id, partner_id = pair["target_clip_id"], pair["partner_clip_id"]
            target_identity = _decode_frame(source_root, ffmpeg, target_id, 0.25)
            target_raster = _decode_frame(source_root, ffmpeg, target_id, 1.0)
            partner_identity = _decode_frame(source_root, ffmpeg, partner_id, 0.25)
            target_audio = _log_mel(_decode_audio(source_root, ffmpeg, target_id))
            partner_audio = _log_mel(_decode_audio(source_root, ffmpeg, partner_id))
            target_identity_tensor, target_audio_tensor = _tensor(target_identity, target_audio, torch, device)
            partner_identity_tensor, partner_audio_tensor = _tensor(partner_identity, partner_audio, torch, device)
            identities = torch.stack([partner_identity_tensor, target_identity_tensor, target_identity_tensor, target_identity_tensor, partner_identity_tensor])
            audios = torch.stack([partner_audio_tensor, target_audio_tensor, torch.zeros_like(target_audio_tensor), partner_audio_tensor, target_audio_tensor])
            predictions = model(identities, audios).detach().cpu().numpy()
            target = target_raster.transpose(2, 0, 1).astype(np.float32) / 255.0
            conditions = ["aligned", "zero_audio", "actor_swapped_audio", "identity_reference_swap"]
            metrics = {condition: float(np.abs(predictions[index + 1] - target).mean()) for index, condition in enumerate(conditions)}
            rows.append(metrics)
            composite = np.concatenate([predictions[0], predictions[1]], axis=2)
            composition_hashes.append(hashlib.sha256(composite.tobytes()).hexdigest())
    summary = summarize_controls(rows, confirmation_protocol)
    return {
        "status": "crema_raster_confirmation_scored",
        "source": "CREMA-D",
        "scope": confirmation_protocol["scope"],
        "checkpoint_sha256": _sha256(checkpoint),
        "protocol_sha256": confirmation_protocol["protocol_sha256"],
        "spec_sha256": confirmation_protocol["spec_sha256"],
        "device": str(device),
        "confirmation_actor_accessed": True,
        "labels_read": False,
        "composition_hashes": composition_hashes,
        "summary": summary,
        "claim_allowed": summary["claim_allowed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-protocol", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_confirmation(json.loads(args.confirmation_protocol.read_text()), json.loads(args.protocol.read_text()), json.loads(args.spec.read_text()), checkpoint=args.checkpoint, manifest=json.loads(args.manifest.read_text()), train_report=json.loads(args.train_report.read_text()), source_root=args.source_root, ffmpeg=args.ffmpeg, device_name=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "claim_allowed": result["claim_allowed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
