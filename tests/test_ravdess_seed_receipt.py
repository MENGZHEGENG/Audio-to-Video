import json
from pathlib import Path

import pytest

from a2v.reporting.ravdess_seed_receipt import validate_receipt


def test_released_seed_summary_recomputes() -> None:
    path = Path(__file__).resolve().parents[1] / "docs" / "ravdess_seed_receipt.json"
    result = validate_receipt(json.loads(path.read_text()))
    assert result["seeds"] == 4
    assert result["negative_seed_count"] == 4
    assert -0.0034 < result["mean_seed_delta"] < -0.0021


def test_modified_seed_delta_is_rejected() -> None:
    path = Path(__file__).resolve().parents[1] / "docs" / "ravdess_seed_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["seed_results"][0]["constant_minus_audio_mae"] = 0.0
    with pytest.raises(ValueError, match="operands"):
        validate_receipt(receipt)
