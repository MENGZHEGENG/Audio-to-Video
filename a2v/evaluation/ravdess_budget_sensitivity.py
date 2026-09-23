"""Post-confirmation matched budget check for the RAVDESS raster renderer."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from a2v.data.ravdess_raster_feature_materialization import _digest
from a2v.evaluation.ravdess_raster_confirmation import _sha256


def check_plan(plan: dict, base_plan: dict, registered_spec: dict) -> None:
    if plan.get("status") != "exploratory_budget_sensitivity_frozen_before_execution":
        raise ValueError("frozen exploratory budget plan required")
    if plan.get("base_plan_sha256") != _digest(base_plan):
        raise ValueError("base seed plan digest mismatch")
    if plan.get("seeds") != [1701, 1702, 1703, 1704]:
        raise ValueError("four predeclared seeds required")
    if plan.get("conditions") != ["aligned_audio_fit", "constant_zero_audio_fit"]:
        raise ValueError("paired conditions required")
    if plan.get("maximum_epochs") != 80 or registered_spec["fitting"]["maximum_epochs"] != 40:
        raise ValueError("only the maximum epoch budget may change from 40 to 80")
    if plan.get("patience") != 6:
        raise ValueError("validation-selection patience must remain six epochs")
    if plan.get("inputs_sha256") != base_plan.get("inputs_sha256"):
        raise ValueError("registered input digests changed")


def main() -> None:
    from a2v.evaluation.ravdess_seed_sensitivity import fit_one, score_pair, validate

    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "spec", "features", "feature-gate", "confirmation", "base-plan", "plan",
        "feature-root", "source-root", "ffmpeg", "output-root", "seed",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name.replace("-", "_"))) for name in (
        "spec", "features", "feature-gate", "confirmation", "base-plan", "plan"
    )}
    spec, features, gate, confirmation, base_plan, plan = [
        json.loads(paths[name].read_text()) for name in paths
    ]
    check_plan(plan, base_plan, spec)
    for name in ("spec", "features", "feature-gate", "confirmation"):
        key = name.replace("-", "_") + "_json"
        if _sha256(paths[name]) != plan["inputs_sha256"][key]:
            raise ValueError("raw input hash mismatch: " + name)
    dev, val = validate(spec, features, gate, confirmation, base_plan)
    seed = int(args.seed)
    if seed not in plan["seeds"]:
        raise ValueError("seed is outside the frozen plan")
    import torch

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    device = torch.device(args.device)
    extended_spec = copy.deepcopy(spec)
    extended_spec["fitting"]["maximum_epochs"] = plan["maximum_epochs"]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    reports = {
        mode: fit_one(
            extended_spec, plan, seed=seed, mode=mode, dev=dev, val=val,
            feature_root=Path(args.feature_root), output_root=output_root / mode,
            device=device,
        )
        for mode in plan["conditions"]
    }
    scored = score_pair(
        extended_spec, confirmation, plan, seed=seed, reports=reports,
        output_root=output_root, source_root=Path(args.source_root),
        ffmpeg=args.ffmpeg, device=device,
    )
    scored["status"] = "exploratory_budget_sensitivity_scored"
    scored["registered_spec_sha256"] = _digest(spec)
    scored["extended_spec_sha256"] = _digest(extended_spec)
    scored["maximum_epochs"] = plan["maximum_epochs"]
    scored["scope"] = "post-confirmation matched 80-epoch budget check; not a registered gate"
    (output_root / "score_report.json").write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": scored["status"], "seed": seed, "summary": scored["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
