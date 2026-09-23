import copy
import json
from pathlib import Path

import pytest

from a2v.reporting.ravdess_trajectory_receipt import validate_receipt


RECEIPT = json.loads((Path(__file__).parents[1] / "docs/ravdess_trajectory_receipt.json").read_text())


def test_released_same_trajectory_receipt():
    result = validate_receipt(RECEIPT)
    assert result["negative_seed_count_at_cap"] == 4
    assert result["negative_seed_count_final"] == 4
    assert len(result["changed_selected_checkpoints"]) == 1
    assert result["changed_selected_checkpoints"][0]["seed"] == 1704
    assert result["changed_selected_checkpoints"][0]["condition"] == "constant_zero_audio_fit"


def test_tampered_difference_fails():
    receipt = copy.deepcopy(RECEIPT)
    receipt["seed_results"][0]["constant_minus_audio_delta"]["final"] = 0.0
    with pytest.raises(ValueError, match="delta differs"):
        validate_receipt(receipt)


def test_missing_seed_fails():
    receipt = copy.deepcopy(RECEIPT)
    receipt["seed_results"].pop()
    with pytest.raises(ValueError, match="four ordered"):
        validate_receipt(receipt)
