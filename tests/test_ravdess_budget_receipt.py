import json
from pathlib import Path

import pytest

from a2v.reporting.ravdess_budget_receipt import validate_receipt


def test_released_budget_summary_recomputes() -> None:
    path = Path(__file__).resolve().parents[1] / "docs" / "ravdess_budget_receipt.json"
    result = validate_receipt(json.loads(path.read_text()))["budget_summaries"]
    assert result["40"]["negative_seed_count"] == 4
    assert result["80"]["negative_seed_count"] == 4
    assert -0.0019 < result["80"]["mean_seed_delta"] < -0.0016


def test_modified_budget_delta_is_rejected() -> None:
    path = Path(__file__).resolve().parents[1] / "docs" / "ravdess_budget_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["seed_results"][1]["constant_minus_audio_mae"] = 0.0
    with pytest.raises(ValueError, match="operands"):
        validate_receipt(receipt)
