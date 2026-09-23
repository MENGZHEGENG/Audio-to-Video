"""Freeze a development-only RAVDESS raster-renderer specification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def build_renderer_spec(protocol: dict[str, Any], feature_audit: dict[str, Any]) -> dict[str, Any]:
    """Lock fitting choices from a passing RAVDESS development-only audit."""

    if protocol.get("status") != "ravdess_raster_actor_disjoint_protocol_registered" or protocol.get("source") != "RAVDESS":
        raise ValueError("registered RAVDESS actor-disjoint protocol required")
    if feature_audit.get("status") != "ravdess_raster_feature_audit_complete" or feature_audit.get("source") != "RAVDESS" or feature_audit.get("failure_count") != 0:
        raise ValueError("passing RAVDESS development/validation feature audit required")
    if feature_audit.get("protocol_sha256") != _digest(protocol) or feature_audit.get("confirmation_actor_accessed") is not False:
        raise ValueError("feature audit must bind the exact protocol without confirmation access")
    counts = feature_audit.get("role_counts", {})
    if any(int(counts.get(role, {}).get("clips", 0)) <= 0 or int(counts.get(role, {}).get("failures", -1)) != 0 for role in ("development", "validation")):
        raise ValueError("all development and validation records must pass")
    return {
        "status": "ravdess_raster_renderer_spec_registered",
        "source": "RAVDESS",
        "protocol_sha256": _digest(protocol),
        "feature_audit_sha256": _digest(feature_audit),
        "model": {
            "name": "AudioFiLMRasterRenderer",
            "visual_input": "64x64 RGB identity reference",
            "audio_input": "40-bin log-mel from the fixed 1.0--1.5 second source window",
            "target": "64x64 RGB centre-square raster at 2.0 seconds, after the audio window",
            "identity_encoder": "four stride-2 convolution blocks with channels 32, 64, 128, 256",
            "audio_encoder": "two-layer MLP over flattened log-mel features to 512 FiLM parameters",
            "fusion": "audio-conditioned scale and bias after the 256-channel identity bottleneck",
            "decoder": "four transpose-convolution blocks ending in sigmoid RGB output",
            "excluded_inputs": ["emotion labels", "intensity labels", "statement labels", "actor identifiers"],
        },
        "fitting": {
            "seed": 1701,
            "fit_role": "development",
            "selection_role": "validation",
            "optimizer": "AdamW(beta1=0.9,beta2=0.999,weight_decay=1e-4)",
            "learning_rate": 0.0003,
            "batch_size": 64,
            "maximum_epochs": 40,
            "early_stopping": "patience 6 on validation mean absolute error only; retain the minimum-validation-error checkpoint",
            "loss": "mean absolute RGB reconstruction error",
            "report_metrics": ["validation mean absolute error", "validation mean squared error", "zero-audio delta", "actor-swapped-audio delta"],
        },
        "checkpoint_gate": {
            "seal_before_confirmation": True,
            "required_manifest_fields": ["checkpoint_sha256", "spec_sha256", "development_actor_ids", "validation_actor_ids", "seed", "selected_epoch", "validation_metrics"],
            "confirmation_prohibition": "Do not read, extract, train on, tune against, or threshold on confirmation actors before the sealed manifest exists.",
        },
        "training_allowed": True,
        "confirmation_allowed": False,
        "claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--feature-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = build_renderer_spec(json.loads(args.protocol.read_text()), json.loads(args.feature_audit.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": spec["status"], "training_allowed": spec["training_allowed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
