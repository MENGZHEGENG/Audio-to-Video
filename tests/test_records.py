from pathlib import Path

from a2v.data.dataset_metrics import build_dataset_integrity_report
from a2v.data.interventions import build_interventions
from a2v.data.quality import build_quality_report
from a2v.data.records import load_records
from a2v.data.split import assign_splits
from a2v.data.split_audit import audit_split_leakage


ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "tiny_records.jsonl"


def test_records_quality_and_group_split_are_deterministic() -> None:
    records = load_records(EXAMPLE)
    assert len(records) == 6
    quality = build_quality_report(records)
    assert quality.warnings == ()
    assert quality.split_counts == {"test": 2, "train": 2, "val": 2}

    reassigned = assign_splits(records, seed=17)
    assert reassigned == assign_splits(records, seed=17)
    assert audit_split_leakage(reassigned).passed

    report = build_dataset_integrity_report(records, long_form_seconds=5)
    assert report["ready"] is True
    assert report["metrics"]["identity_disjoint_splits"] == 1.0


def test_intervention_plan_validates_every_visible_speaker() -> None:
    record = load_records(EXAMPLE)[0]
    interventions = build_interventions(record, delay_ms=250)
    assert len(interventions) == 8
    assert {item.kind for item in interventions} == {"mute", "delay", "swap_content", "swap_prosody"}
