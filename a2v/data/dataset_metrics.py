"""Dataset integrity metrics for CIViC-A2V evidence reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from a2v.data.records import ClipRecord, load_records
from a2v.data.quality import build_quality_report
from a2v.data.split_audit import audit_split_leakage


def _metadata_value(record: ClipRecord, key: str, *, fallbacks: tuple[str, ...] = ()) -> str:
    value = record.metadata.get(key)
    if value in (None, ""):
        for fallback in fallbacks:
            value = record.metadata.get(fallback)
            if value not in (None, ""):
                break
    return str(value) if value is not None else "unknown"


def _bounded_ratio(value: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, value / denominator))


def build_dataset_integrity_report(
    records: list[ClipRecord],
    *,
    long_form_seconds: float = 30.0,
    target_speakers: int = 20,
    target_sources: int = 3,
    target_scenes: int = 5,
    group_key: str = "metadata.conversation_id",
    fallback_key: str = "clip_id",
) -> dict[str, Any]:
    if long_form_seconds <= 0:
        raise ValueError("long_form_seconds must be positive")
    quality = build_quality_report(records)
    split_audit = audit_split_leakage(records, group_key=group_key, fallback_key=fallback_key)
    sources = {record.source or "unknown" for record in records}
    scenes = {_metadata_value(record, "scene", fallbacks=("interaction_type",)) for record in records}
    splits = set(quality.split_counts)
    long_form_records = sum(1 for record in records if record.duration >= long_form_seconds)
    long_form_coverage = _bounded_ratio(long_form_records, len(records)) if records else 0.0
    diversity_components = {
        "speaker_diversity": _bounded_ratio(float(quality.unique_speakers), float(target_speakers)),
        "source_diversity": _bounded_ratio(float(len(sources)), float(target_sources)),
        "scene_diversity": _bounded_ratio(float(len(scenes)), float(target_scenes)),
        "split_coverage": _bounded_ratio(float(len(splits.intersection({"train", "val", "test"}))), 3.0),
        "multi_speaker_coverage": _bounded_ratio(float(quality.multi_speaker_records), float(len(records))) if records else 0.0,
    }
    diversity = sum(diversity_components.values()) / len(diversity_components)
    warnings = list(quality.warnings)
    if not split_audit.passed:
        warnings.append(f"{len(split_audit.leaking_groups)} split-leaking groups")
    if long_form_records == 0:
        warnings.append(f"no records are at least {long_form_seconds:g} seconds")
    return {
        "records": len(records),
        "total_hours": quality.total_hours,
        "unique_speakers": quality.unique_speakers,
        "sources": sorted(sources),
        "scenes": sorted(scenes),
        "split_counts": quality.split_counts,
        "multi_speaker_records": quality.multi_speaker_records,
        "segment_coverage": quality.segment_coverage,
        "long_form_seconds": float(long_form_seconds),
        "long_form_records": long_form_records,
        "metrics": {
            "diversity": diversity,
            "long_form_coverage": long_form_coverage,
            "identity_disjoint_splits": 1.0 if split_audit.passed else 0.0,
        },
        "diversity_components": diversity_components,
        "split_leakage": split_audit.to_dict(),
        "warnings": warnings,
        "ready": not warnings,
    }


def render_dataset_integrity_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# CIViC-A2V Dataset Integrity Metrics",
        "",
        f"Status: **{'PASS' if report['ready'] else 'WARN'}**",
        "",
        "## Dataset Summary",
        f"- Records: {report['records']}",
        f"- Total hours: {float(report['total_hours']):.4f}",
        f"- Unique speakers: {report['unique_speakers']}",
        f"- Multi-speaker records: {report['multi_speaker_records']}",
        f"- Splits: {report['split_counts']}",
        f"- Sources: {', '.join(report['sources'])}",
        f"- Scenes: {', '.join(report['scenes'])}",
        "",
        "## Paper-Facing Metrics",
        "| Metric | Value |",
        "|---|---:|",
        f"| diversity | {metrics['diversity']:.4f} |",
        f"| long_form_coverage | {metrics['long_form_coverage']:.4f} |",
        f"| identity_disjoint_splits | {metrics['identity_disjoint_splits']:.4f} |",
        "",
        "## Diversity Components",
        "| Component | Value |",
        "|---|---:|",
    ]
    for key, value in report["diversity_components"].items():
        lines.append(f"| {key} | {float(value):.4f} |")
    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dataset integrity metrics for CIViC-A2V.")
    parser.add_argument("--records", type=Path, default=Path("examples/tiny_records.jsonl"))
    parser.add_argument("--long-form-seconds", type=float, default=30.0)
    parser.add_argument("--json-output", type=Path, default=Path("results/dataset_integrity_metrics.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("results/dataset_integrity_metrics.md"))
    parser.add_argument("--group-key", default="metadata.conversation_id")
    parser.add_argument("--fallback-key", default="clip_id")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_dataset_integrity_report(
        load_records(args.records),
        long_form_seconds=args.long_form_seconds,
        group_key=args.group_key,
        fallback_key=args.fallback_key,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_dataset_integrity_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ready"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
