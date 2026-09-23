"""Exploratory, metadata-matched controls for the sealed RAVDESS raster renderer."""

from __future__ import annotations

import argparse
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
    require_registered_target_timestamp,
)
from a2v.evaluation.ravdess_raster_confirmation import _sha256, _tensor, validate_checkpoint_gate


CONDITIONS = (
    "aligned",
    "zero_audio",
    "registered_actor_swap",
    "matched_actor_swap",
    "same_actor_statement_swap",
    "identity_reference_swap",
)


def control_clip_ids(pair: dict[str, str]) -> tuple[str, str]:
    """Use the registered partner actor, holding the six non-actor filename fields fixed."""

    target = Path(pair["target_clip_id"])
    fields = target.stem.split("-")
    if len(fields) != 7 or fields[-1] != pair["target_actor_id"]:
        raise ValueError("target does not follow the registered RAVDESS video-speech filename")
    partner_actor = pair["partner_actor_id"]
    if partner_actor == fields[-1] or not partner_actor.isdigit():
        raise ValueError("the matched replacement needs a distinct registered actor")
    matched_fields = [*fields[:6], partner_actor]
    matched = f"Actor_{partner_actor}/{'-'.join(matched_fields)}.mp4"
    if fields[4] not in {"01", "02"}:
        raise ValueError("the statement field must be 01 or 02")
    statement_fields = fields.copy()
    statement_fields[4] = "02" if fields[4] == "01" else "01"
    other_statement = f"{target.parent.as_posix()}/{'-'.join(statement_fields)}.mp4"
    return matched, other_statement


def summarize(rows: list[dict[str, Any]], *, seed: int, replicates: int) -> dict[str, Any]:
    if not rows or replicates < 1000:
        raise ValueError("nonempty rows and at least 1,000 bootstrap replicates required")
    actors = sorted({row["actor_id"] for row in rows})
    if len(actors) != 4 or any(set(row["errors"]) != set(CONDITIONS) for row in rows):
        raise ValueError("four confirmation actors and complete finite condition rows required")
    errors = np.asarray([[row["errors"][c] for c in CONDITIONS] for row in rows], dtype=np.float64)
    if not np.isfinite(errors).all():
        raise ValueError("nonfinite raster error")
    actor_ids = np.asarray([row["actor_id"] for row in rows])
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {"pairs": len(rows), "actors": actors, "aligned_mae": float(errors[:, 0].mean()), "controls": {}}
    for index, condition in enumerate(CONDITIONS[1:], start=1):
        delta = errors[:, index] - errors[:, 0]
        pair_means = delta[rng.integers(0, len(rows), size=(replicates, len(rows)))].mean(axis=1)
        actor_means = np.asarray([delta[actor_ids == actor].mean() for actor in actors])
        sampled_actors = actor_means[rng.integers(0, len(actors), size=(replicates, len(actors)))].mean(axis=1)
        result["controls"][condition] = {
            "mean_delta": float(delta.mean()),
            "pair_bootstrap_95": [float(x) for x in np.quantile(pair_means, [0.025, 0.975])],
            "actor_bootstrap_95": [float(x) for x in np.quantile(sampled_actors, [0.025, 0.975])],
            "by_actor_mean_delta": {actor: float(value) for actor, value in zip(actors, actor_means)},
        }
    return result


def run(
    confirmation: dict[str, Any],
    protocol: dict[str, Any],
    spec: dict[str, Any],
    plan: dict[str, Any],
    *,
    checkpoint: Path,
    manifest: dict[str, Any],
    train_report: dict[str, Any],
    source_root: Path,
    ffmpeg: str,
    device_name: str,
) -> dict[str, Any]:
    if confirmation.get("status") != "ravdess_raster_confirmation_protocol_registered":
        raise ValueError("registered confirmation protocol required")
    if _digest(protocol) != confirmation.get("protocol_sha256") or _digest(spec) != confirmation.get("spec_sha256"):
        raise ValueError("frozen protocol/specification mismatch")
    require_registered_target_timestamp(spec)
    validate_checkpoint_gate(checkpoint, manifest, train_report, confirmation)
    if (
        plan.get("status") != "exploratory_controls_frozen_before_scoring"
        or plan.get("confirmation_protocol_sha256") != _digest(confirmation)
        or plan.get("conditions") != list(CONDITIONS)
        or plan.get("target_timestamp_seconds") != TARGET_TIMESTAMP_SECONDS
        or plan.get("expected_pairs") != 240
    ):
        raise ValueError("exploratory control plan is incomplete or mismatched")

    import torch

    from a2v.models.crema_audiofilm_raster import AudioFiLMRasterRenderer

    if device_name != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("requested GPU unavailable")
    device = torch.device(device_name)
    model = AudioFiLMRasterRenderer().to(device)
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    if payload.get("spec_sha256") != confirmation["spec_sha256"]:
        raise ValueError("checkpoint payload specification mismatch")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for pair in confirmation["pairs"]:
            target_id = pair["target_clip_id"]
            registered_id = pair["partner_clip_id"]
            matched_id, statement_id = control_clip_ids(pair)
            for clip_id in (target_id, registered_id, matched_id, statement_id):
                if not (source_root / clip_id).is_file():
                    raise FileNotFoundError(source_root / clip_id)
            identity = _decode_frame(source_root, ffmpeg, target_id, 0.25)
            target = _decode_frame(source_root, ffmpeg, target_id, TARGET_TIMESTAMP_SECONDS)
            partner_identity = _decode_frame(source_root, ffmpeg, registered_id, 0.25)
            target_audio = _log_mel(_decode_audio(source_root, ffmpeg, target_id))
            registered_audio = _log_mel(_decode_audio(source_root, ffmpeg, registered_id))
            matched_audio = _log_mel(_decode_audio(source_root, ffmpeg, matched_id))
            statement_audio = _log_mel(_decode_audio(source_root, ffmpeg, statement_id))
            identity_tensor, target_audio_tensor = _tensor(identity, target_audio, torch, device)
            partner_tensor, registered_tensor = _tensor(partner_identity, registered_audio, torch, device)
            _, matched_tensor = _tensor(identity, matched_audio, torch, device)
            _, statement_tensor = _tensor(identity, statement_audio, torch, device)
            identities = torch.stack([identity_tensor] * 5 + [partner_tensor])
            audios = torch.stack([
                target_audio_tensor,
                torch.zeros_like(target_audio_tensor),
                registered_tensor,
                matched_tensor,
                statement_tensor,
                target_audio_tensor,
            ])
            predictions = model(identities, audios).detach().cpu().numpy()
            target_chw = target.transpose(2, 0, 1).astype(np.float32) / 255.0
            errors = {condition: float(np.abs(predictions[index] - target_chw).mean()) for index, condition in enumerate(CONDITIONS)}
            rows.append({
                "pair_sha256": hashlib.sha256(target_id.encode()).hexdigest(),
                "actor_id": pair["target_actor_id"],
                "errors": errors,
            })
    if len(rows) != plan["expected_pairs"] or len({row["pair_sha256"] for row in rows}) != len(rows):
        raise ValueError("incomplete or duplicated target pairs")
    return {
        "status": "exploratory_ravdess_factorial_scored",
        "source": "RAVDESS",
        "scope": "fixed_240_pair_confirmation_actors_exploratory_after_registered_replay",
        "target_timestamp_seconds": TARGET_TIMESTAMP_SECONDS,
        "checkpoint_sha256": _sha256(checkpoint),
        "confirmation_protocol_sha256": _digest(confirmation),
        "plan_sha256": _digest(plan),
        "conditions": list(CONDITIONS),
        "source_labels_read": False,
        "summary": summarize(rows, seed=int(plan["bootstrap_seed"]), replicates=int(plan["bootstrap_replicates"])),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("confirmation-protocol", "protocol", "spec", "plan", "checkpoint", "manifest", "train-report", "source-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(
        json.loads(args.confirmation_protocol.read_text()),
        json.loads(args.protocol.read_text()),
        json.loads(args.spec.read_text()),
        json.loads(args.plan.read_text()),
        checkpoint=args.checkpoint,
        manifest=json.loads(args.manifest.read_text()),
        train_report=json.loads(args.train_report.read_text()),
        source_root=args.source_root,
        ffmpeg=args.ffmpeg,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "pairs": result["summary"]["pairs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
