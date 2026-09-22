"""Validated clip records for conversational audio-to-video experiments.

The records format is JSONL. Each row represents one synchronized clip and
keeps enough metadata to test causal audio/video interventions without committing
large media files to Git.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class SpeakerSegment:
    """A diarized speech segment for one visible participant."""

    speaker_id: str
    start: float
    end: float
    text: str | None = None
    emotion: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpeakerSegment":
        required = {"speaker_id", "start", "end"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"speaker segment missing fields: {sorted(missing)}")
        segment = cls(
            speaker_id=str(payload["speaker_id"]),
            start=float(payload["start"]),
            end=float(payload["end"]),
            text=payload.get("text"),
            emotion=payload.get("emotion"),
        )
        if segment.start < 0 or segment.end <= segment.start:
            raise ValueError(f"invalid segment time span: {payload}")
        return segment


@dataclass(frozen=True)
class ClipRecord:
    """One audio/video training or evaluation clip."""

    clip_id: str
    video_path: str
    audio_path: str
    duration: float
    fps: float
    split: str
    visible_speaker_ids: tuple[str, ...]
    segments: tuple[SpeakerSegment, ...] = field(default_factory=tuple)
    source: str | None = None
    license: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClipRecord":
        required = {
            "clip_id",
            "video_path",
            "audio_path",
            "duration",
            "fps",
            "split",
            "visible_speaker_ids",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"clip missing fields: {sorted(missing)}")
        record = cls(
            clip_id=str(payload["clip_id"]),
            video_path=str(payload["video_path"]),
            audio_path=str(payload["audio_path"]),
            duration=float(payload["duration"]),
            fps=float(payload["fps"]),
            split=str(payload["split"]),
            visible_speaker_ids=tuple(str(x) for x in payload["visible_speaker_ids"]),
            segments=tuple(SpeakerSegment.from_dict(x) for x in payload.get("segments", [])),
            source=payload.get("source"),
            license=payload.get("license"),
            metadata=dict(payload.get("metadata", {})),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not self.clip_id:
            raise ValueError("clip_id must be non-empty")
        if self.duration <= 0:
            raise ValueError(f"duration must be positive for {self.clip_id}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive for {self.clip_id}")
        if self.split not in {"train", "val", "test", "ood", "counterfactual"}:
            raise ValueError(f"unknown split {self.split!r} for {self.clip_id}")
        if not self.visible_speaker_ids:
            raise ValueError(f"visible_speaker_ids must be non-empty for {self.clip_id}")
        visible = set(self.visible_speaker_ids)
        for segment in self.segments:
            if segment.speaker_id not in visible:
                raise ValueError(
                    f"segment speaker {segment.speaker_id!r} is not visible in {self.clip_id}"
                )
            if segment.end > self.duration + 1e-6:
                raise ValueError(f"segment exceeds duration in {self.clip_id}: {segment}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "video_path": self.video_path,
            "audio_path": self.audio_path,
            "duration": self.duration,
            "fps": self.fps,
            "split": self.split,
            "visible_speaker_ids": list(self.visible_speaker_ids),
            "segments": [segment.__dict__ for segment in self.segments],
            "source": self.source,
            "license": self.license,
            "metadata": self.metadata,
        }


def iter_records(path: Path) -> Iterator[ClipRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
                yield ClipRecord.from_dict(payload)
            except Exception as exc:  # noqa: BLE001 - preserve row context for CLI users.
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def load_records(path: Path) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    records.extend(iter_records(path))
    return records


def split_counts(records: Iterable[ClipRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.split] = counts.get(record.split, 0) + 1
    return dict(sorted(counts.items()))


def validate_records_file(path: Path) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    total = 0
    for record in iter_records(path):
        total += 1
        counts[record.split] = counts.get(record.split, 0) + 1
    return total, dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate audio-to-video JSONL records.")
    parser.add_argument("--records", type=Path, required=True)
    args = parser.parse_args()
    total, counts = validate_records_file(args.records)
    print(f"valid_records={total}")
    print(f"split_counts={counts}")


if __name__ == "__main__":
    main()
