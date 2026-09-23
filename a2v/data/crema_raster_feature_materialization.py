"""Materialize source-only raster/audio tensors after all input gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np


FrameReader = Callable[[str, float], np.ndarray]
AudioReader = Callable[[str], np.ndarray]


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _log_mel(samples: np.ndarray) -> np.ndarray:
    if samples.shape != (8000,):
        raise ValueError("registered audio window must contain exactly 8,000 samples")
    window = np.hanning(400).astype(np.float32)
    frames = np.stack([samples[start : start + 400] * window for start in range(0, 8000 - 400 + 1, 160)])
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    frequencies = np.linspace(0.0, 8000.0, power.shape[1])
    mel = 2595.0 * np.log10(1.0 + frequencies / 700.0)
    points = np.linspace(mel[0], mel[-1], 42)
    filters = np.zeros((40, power.shape[1]), dtype=np.float32)
    for index in range(1, 41):
        left, center, right = points[index - 1], points[index], points[index + 1]
        filters[index - 1] = np.maximum(0.0, np.minimum((mel - left) / (center - left), (right - mel) / (right - center)))
    return np.log(np.maximum(filters @ power.T, 1e-6)).astype(np.float32)


def materialize_features(
    protocol: dict[str, Any], audit: dict[str, Any], spec: dict[str, Any], *, output_root: Path, frame_reader: FrameReader, audio_reader: AudioReader
) -> dict[str, Any]:
    if protocol.get("status") != "crema_raster_renderer_protocol_amended" or protocol.get("source") != "CREMA-D":
        raise ValueError("amended CREMA raster protocol required")
    protocol_hash = _digest(protocol)
    if audit.get("status") != "crema_raster_feature_audit_complete" or audit.get("failure_count") != 0 or audit.get("feature_extraction_allowed") is not True:
        raise ValueError("passing feature audit required")
    if audit.get("protocol_sha256") != protocol_hash or spec.get("protocol_sha256") != protocol_hash:
        raise ValueError("audit and specification must bind the exact amended protocol")
    if spec.get("status") != "crema_raster_renderer_spec_registered" or spec.get("training_allowed") is not True:
        raise ValueError("registered renderer specification required")
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for clip in protocol.get("clips", []):
        if clip.get("role") not in {"development", "validation"}:
            continue
        clip_id = str(clip["clip_id"])
        identity = np.asarray(frame_reader(clip_id, 0.25), dtype=np.uint8)
        target = np.asarray(frame_reader(clip_id, 1.0), dtype=np.uint8)
        if identity.shape != (64, 64, 3) or target.shape != (64, 64, 3):
            raise ValueError(f"{clip_id}: expected 64x64 RGB identity and target rasters")
        payload = {"identity_rgb": identity.transpose(2, 0, 1), "target_rgb": target.transpose(2, 0, 1), "audio_log_mel": _log_mel(np.asarray(audio_reader(clip_id), dtype=np.float32))}
        destination = output_root / f"{clip_id}.npz"
        temporary = output_root / f".{clip_id}.{os.getpid()}.npz"
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, destination)
        records.append({"clip_id": clip_id, "actor_id": clip["actor_id"], "role": clip["role"], "relative_path": destination.name, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    return {"status": "crema_raster_features_materialized", "source": "CREMA-D", "protocol_sha256": protocol_hash, "source_scope": "development_and_validation_actors_only", "confirmation_actor_accessed": False, "records": records, "feature_count": len(records), "training_allowed": True, "confirmation_allowed": False, "claim_allowed": False}


def _decode_frame(source_root: Path, ffmpeg: str, clip_id: str, timestamp: float) -> np.ndarray:
    command = [ffmpeg, "-v", "error", "-ss", str(timestamp), "-i", str(source_root / "VideoFlash" / f"{clip_id}.flv"), "-frames:v", "1", "-vf", "crop=ih:ih:(iw-ih)/2:0,scale=64:64", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode or len(result.stdout) != 64 * 64 * 3:
        raise ValueError(f"{clip_id}: raster decode failed")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(64, 64, 3)


def _decode_audio(source_root: Path, ffmpeg: str, clip_id: str) -> np.ndarray:
    command = [ffmpeg, "-v", "error", "-ss", "0.75", "-t", "0.5", "-i", str(source_root / "AudioWAV" / f"{clip_id}.wav"), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode or len(result.stdout) != 8000 * 4:
        raise ValueError(f"{clip_id}: audio decode failed")
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = materialize_features(json.loads(args.protocol.read_text()), json.loads(args.audit.read_text()), json.loads(args.spec.read_text()), output_root=args.feature_root, frame_reader=lambda clip, t: _decode_frame(args.source_root, args.ffmpeg, clip, t), audio_reader=lambda clip: _decode_audio(args.source_root, args.ffmpeg, clip))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "feature_count": report["feature_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
