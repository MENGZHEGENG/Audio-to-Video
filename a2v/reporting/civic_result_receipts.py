"""Create and validate privacy-safe aggregate receipts for CIViC results.

The public receipt intentionally contains no source paths, raw examples,
identifiers, features, checkpoints, or execution records. It retains aggregate
per-run values needed to recompute displayed means and population standard
deviations, plus a commitment to the withheld member-to-value mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SUMMARY_KEYS = {"n", "mean", "std", "values"}
PRIVATE_KEYS = {"by_seed", "devices", "generator", "run_ids", "seeds", "source_reports"}


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_summary(value: Any) -> bool:
    return isinstance(value, Mapping) and SUMMARY_KEYS.issubset(value)


def _safe_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    values = value["values"]
    if not isinstance(values, list) or not values or not all(isinstance(item, (int, float)) for item in values):
        raise ValueError("summary values must be a non-empty numeric list")
    if value["n"] != len(values):
        raise ValueError("summary n does not match values")
    if not all(isinstance(value[key], (int, float)) for key in ("mean", "std")):
        raise ValueError("summary mean and std must be numeric")
    result: dict[str, Any] = {
        "n": value["n"],
        "mean": value["mean"],
        "std_population": value["std"],
        "values": values,
    }
    private_members = value.get("by_seed")
    if isinstance(private_members, Mapping):
        result["membership_commitment_sha256"] = _canonical_digest(private_members)
    return result


def _safe_tree(value: Any) -> Any:
    if _is_summary(value):
        return _safe_summary(value)
    if isinstance(value, Mapping):
        return {
            str(key): _safe_tree(item)
            for key, item in sorted(value.items())
            if key not in PRIVATE_KEYS
        }
    if isinstance(value, list):
        return [_safe_tree(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"unsupported value type: {type(value).__name__}")


def build_receipt(input_paths: Mapping[str, Path]) -> dict[str, Any]:
    """Build a public-safe receipt from approved aggregate JSON inputs."""
    source_records: dict[str, Any] = {}
    for name, path in sorted(input_paths.items()):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"{name}: expected a JSON object")
        source_records[name] = {
            "source_record_sha256": _file_digest(path),
            "paired": _safe_tree(payload.get("paired", {})),
            "splits": _safe_tree(payload.get("splits", {})),
        }
    return {
        "schema": "civic-safe-result-receipts/v1",
        "scope": "aggregate mixed-condition calculations only",
        "source_records": source_records,
        "excluded": [
            "raw media",
            "identifiers",
            "raw features",
            "checkpoints",
            "execution records",
            "pair-level bootstrap inputs",
        ],
        "recomputation": "Means and population standard deviations are recomputable from values. Membership commitments require authorized access to the withheld member mapping.",
    }


def _validate_tree(value: Any, errors: list[str], label: str) -> None:
    if isinstance(value, Mapping):
        if {"n", "mean", "std_population", "values"}.issubset(value):
            values = value["values"]
            if not isinstance(values, list) or not values or not all(isinstance(item, (int, float)) for item in values):
                errors.append(f"{label}: invalid values")
                return
            if value["n"] != len(values):
                errors.append(f"{label}: n mismatch")
            observed_mean = sum(values) / len(values)
            observed_std = math.sqrt(sum((item - observed_mean) ** 2 for item in values) / len(values))
            if not math.isclose(observed_mean, value["mean"], rel_tol=0.0, abs_tol=1e-12):
                errors.append(f"{label}: mean mismatch")
            if not math.isclose(observed_std, value["std_population"], rel_tol=0.0, abs_tol=1e-12):
                errors.append(f"{label}: population std mismatch")
            return
        for key, item in value.items():
            _validate_tree(item, errors, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_tree(item, errors, f"{label}[{index}]")


def validate_receipt(receipt: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if receipt.get("schema") != "civic-safe-result-receipts/v1":
        errors.append("invalid schema")
    records = receipt.get("source_records")
    if not isinstance(records, Mapping) or not records:
        return errors + ["missing source records"]
    _validate_tree(records, errors, "source_records")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CIViC privacy-safe aggregate result receipts.")
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt({
        "mixed_primary": args.primary,
        "support_policies": args.policy,
        "fixed_interface_controls": args.controls,
    })
    errors = validate_receipt(receipt)
    if errors:
        raise SystemExit("ERROR: " + "; ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: wrote {args.output}")


if __name__ == "__main__":
    main()
