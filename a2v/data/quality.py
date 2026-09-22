"""Records-level quality checks for conversational audio-to-video data."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from a2v.data.records import ClipRecord, iter_records


@dataclass(frozen=True)
class RecordsQualityReport:
    records: int
    total_hours: float
    split_counts: dict[str, int]
    unique_speakers: int
    multi_speaker_records: int
    segment_coverage: float
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def segment_seconds(record: ClipRecord) -> float:
    return sum(segment.end - segment.start for segment in record.segments)


def has_overlapping_speech(record: ClipRecord) -> bool:
    segments = sorted(record.segments, key=lambda segment: (segment.start, segment.end))
    for previous, current in zip(segments, segments[1:]):
        if current.start < previous.end and current.speaker_id != previous.speaker_id:
            return True
    return False


def build_quality_report(records: Iterable[ClipRecord], *, max_warnings: int = 100) -> RecordsQualityReport:
    warnings: list[str] = []

    split_counts: dict[str, int] = {}
    speakers: set[str] = set()
    total_duration = 0.0
    total_segment_seconds = 0.0
    multi_speaker_records = 0
    records_with_overlap = 0
    total_records = 0

    def add_warning(message: str) -> None:
        if len(warnings) < max_warnings:
            warnings.append(message)

    for record in records:
        total_records += 1
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
        speakers.update(record.visible_speaker_ids)
        total_duration += record.duration
        total_segment_seconds += min(segment_seconds(record), record.duration)
        if len(record.visible_speaker_ids) >= 2:
            multi_speaker_records += 1
        if has_overlapping_speech(record):
            records_with_overlap += 1
        if not record.segments:
            add_warning(f"{record.clip_id}: no diarized segments")
        if record.duration < 2.0:
            add_warning(f"{record.clip_id}: duration below 2 seconds")

    if total_records == 0:
        return RecordsQualityReport(0, 0.0, {}, 0, 0, 0.0, ("records is empty",))

    if "val" not in split_counts:
        add_warning("missing validation split")
    if "test" not in split_counts:
        add_warning("missing test split")
    if multi_speaker_records == 0:
        add_warning("no multi-speaker records")
    if records_with_overlap:
        add_warning(f"{records_with_overlap} records contain overlapping speech")

    coverage = total_segment_seconds / total_duration if total_duration else 0.0
    if coverage < 0.25:
        add_warning("low diarized speech coverage")

    return RecordsQualityReport(
        records=total_records,
        total_hours=total_duration / 3600.0,
        split_counts=dict(sorted(split_counts.items())),
        unique_speakers=len(speakers),
        multi_speaker_records=multi_speaker_records,
        segment_coverage=coverage,
        warnings=tuple(warnings),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and QA audio-to-video records.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if warnings are present.")
    args = parser.parse_args()
    report = build_quality_report(iter_records(args.records))
    text = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.strict and report.warnings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
