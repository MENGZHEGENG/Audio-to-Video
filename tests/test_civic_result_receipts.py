import json
from pathlib import Path

from a2v.reporting.civic_result_receipts import build_receipt, validate_receipt


def _record(path: Path, values: list[float]) -> None:
    mean = sum(values) / len(values)
    std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
    path.write_text(json.dumps({
        "devices": ["private-device"],
        "source_reports": [{"path": "/private/path"}],
        "paired": {
            "result": {
                "by_seed": {"123": values[0], "456": values[1]},
                "n": len(values), "mean": mean, "std": std, "values": values,
            }
        },
        "splits": {},
    }), encoding="utf-8")


def test_receipt_keeps_recomputable_values_and_removes_private_fields(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    policy = tmp_path / "policy.json"
    controls = tmp_path / "controls.json"
    for path in (primary, policy, controls):
        _record(path, [1.0, 3.0])

    receipt = build_receipt({"primary": primary, "policy": policy, "controls": controls})

    result = receipt["source_records"]["primary"]["paired"]["result"]
    assert result["values"] == [1.0, 3.0]
    assert result["mean"] == 2.0
    assert result["std_population"] == 1.0
    assert "membership_commitment_sha256" in result
    rendered = json.dumps(receipt)
    assert "private-device" not in rendered
    assert "/private/path" not in rendered
    assert "by_seed" not in rendered
    assert validate_receipt(receipt) == []


def test_receipt_validator_rejects_modified_summary_value(tmp_path: Path) -> None:
    paths = {name: tmp_path / f"{name}.json" for name in ("primary", "policy", "controls")}
    for path in paths.values():
        _record(path, [1.0, 3.0])
    receipt = build_receipt(paths)
    receipt["source_records"]["primary"]["paired"]["result"]["mean"] = 9.0

    assert "source_records.primary.paired.result: mean mismatch" in validate_receipt(receipt)
