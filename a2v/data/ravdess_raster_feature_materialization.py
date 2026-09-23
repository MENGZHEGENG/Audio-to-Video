"""Materialize audited RAVDESS raster/audio tensors without opening confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

from a2v.data.crema_raster_feature_materialization import _log_mel


FrameReader = Callable[[str, float], np.ndarray]
TARGET_TIMESTAMP_SECONDS = 2.0


def require_registered_target_timestamp(spec: dict[str, Any]) -> None:
    """Fail if the frozen target description disagrees with decoded features."""

    target = spec.get("model", {}).get("target", "")
    match = re.search(r"\bat ([0-9]+(?:\.[0-9]+)?) seconds\b", target)
    if match is None or float(match.group(1)) != TARGET_TIMESTAMP_SECONDS:
        raise ValueError("renderer specification target timestamp differs from feature decoding")


AudioReader = Callable[[str], np.ndarray]


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _feature_name(clip_id: str) -> str:
    return hashlib.sha256(clip_id.encode("utf-8")).hexdigest() + ".npz"


def materialize_features(
    protocol: dict[str, Any], media_audit: dict[str, Any], audit: dict[str, Any], spec: dict[str, Any], *, output_root: Path, frame_reader: FrameReader, audio_reader: AudioReader
) -> dict[str, Any]:
    """Create feature tensors exactly for the approved audit records."""

    if protocol.get("status") != "ravdess_raster_actor_disjoint_protocol_registered" or protocol.get("source") != "RAVDESS":
        raise ValueError("registered RAVDESS protocol required")
    protocol_hash = _digest(protocol)
    if media_audit.get("status") != "ravdess_development_validation_media_audited" or media_audit.get("source") != "RAVDESS" or media_audit.get("protocol_sha256") != protocol_hash:
        raise ValueError("bound RAVDESS development/validation media audit required")
    if audit.get("status") != "ravdess_raster_feature_audit_complete" or audit.get("source") != "RAVDESS" or audit.get("failure_count") != 0 or audit.get("feature_extraction_allowed") is not True:
        raise ValueError("passing RAVDESS feature audit required")
    if audit.get("protocol_sha256") != protocol_hash or audit.get("media_audit_sha256") != _digest(media_audit) or audit.get("confirmation_actor_accessed") is not False:
        raise ValueError("feature audit must bind the exact non-confirmation inputs")
    if spec.get("status") != "ravdess_raster_renderer_spec_registered" or spec.get("source") != "RAVDESS" or spec.get("training_allowed") is not True:
        raise ValueError("registered RAVDESS renderer specification required")
    if spec.get("protocol_sha256") != protocol_hash or spec.get("feature_audit_sha256") != _digest(audit):
        raise ValueError("renderer specification does not bind this exact audit")
    require_registered_target_timestamp(spec)

    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for clip in audit.get("records", []):
        if clip.get("role") not in {"development", "validation"} or clip.get("passed") is not True:
            continue
        clip_id = str(clip["clip_id"])
        identity = np.asarray(frame_reader(clip_id, 0.25), dtype=np.uint8)
        target = np.asarray(frame_reader(clip_id, TARGET_TIMESTAMP_SECONDS), dtype=np.uint8)
        if identity.shape != (64, 64, 3) or target.shape != (64, 64, 3):
            raise ValueError(f"{clip_id}: expected 64x64 RGB identity and target rasters")
        payload = {
            "identity_rgb": identity.transpose(2, 0, 1),
            "target_rgb": target.transpose(2, 0, 1),
            "audio_log_mel": _log_mel(np.asarray(audio_reader(clip_id), dtype=np.float32)),
        }
        relative_path = _feature_name(clip_id)
        destination = output_root / relative_path
        temporary = output_root / f".{relative_path}.{os.getpid()}.npz"
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, destination)
        records.append({"clip_id": clip_id, "actor_id": str(clip["actor_id"]), "role": str(clip["role"]), "relative_path": relative_path, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    if not records or len(records) != len(audit.get("records", [])):
        raise ValueError("every audited development/validation record must materialize exactly once")
    return {
        "status": "ravdess_raster_features_materialized",
        "source": "RAVDESS",
        "protocol_sha256": protocol_hash,
        "media_audit_sha256": _digest(media_audit),
        "feature_audit_sha256": _digest(audit),
        "source_scope": "development_and_validation_actors_only",
        "confirmation_actor_accessed": False,
        "records": records,
        "feature_count": len(records),
        "training_allowed": True,
        "confirmation_allowed": False,
        "claim_allowed": False,
    }


def _decode_frame(source_root: Path, ffmpeg: str, clip_id: str, timestamp: float) -> np.ndarray:
    command = [ffmpeg, "-v", "error", "-ss", str(timestamp), "-i", str(source_root / clip_id), "-frames:v", "1", "-vf", "crop=min(iw\\,ih):min(iw\\,ih):(iw-ow)/2:(ih-oh)/2,scale=64:64", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode or len(result.stdout) != 64 * 64 * 3:
        raise ValueError(f"{clip_id}: raster decode failed")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(64, 64, 3)


def _decode_audio(source_root: Path, ffmpeg: str, clip_id: str) -> np.ndarray:
    command = [ffmpeg, "-v", "error", "-ss", "1.0", "-t", "0.5", "-i", str(source_root / clip_id), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode or len(result.stdout) != 8000 * 4:
        raise ValueError(f"{clip_id}: audio decode failed")
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--media-audit", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = materialize_features(json.loads(args.protocol.read_text()), json.loads(args.media_audit.read_text()), json.loads(args.audit.read_text()), json.loads(args.spec.read_text()), output_root=args.feature_root, frame_reader=lambda clip, timestamp: _decode_frame(args.source_root, args.ffmpeg, clip, timestamp), audio_reader=lambda clip: _decode_audio(args.source_root, args.ffmpeg, clip))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "feature_count": report["feature_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
