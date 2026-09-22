"""Validated feature index records for audio-to-video training.

Feature indexes point to derived feature files on external storage. They do not
store raw media, generated videos, or checkpoints in Git.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


STREAM_TYPES = {"audio", "phoneme", "face", "pose", "identity", "gaze", "emotion", "dialogue"}


@dataclass(frozen=True)
class FeatureStream:
    stream_type: str
    path: str
    frames: int
    dim: int
    rate_hz: float
    format: str = "npz"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureStream":
        required = {"stream_type", "path", "frames", "dim", "rate_hz"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"feature stream missing fields: {sorted(missing)}")
        stream = cls(
            stream_type=str(payload["stream_type"]),
            path=str(payload["path"]),
            frames=int(payload["frames"]),
            dim=int(payload["dim"]),
            rate_hz=float(payload["rate_hz"]),
            format=str(payload.get("format", "npz")),
        )
        stream.validate()
        return stream

    def validate(self) -> None:
        if self.stream_type not in STREAM_TYPES:
            raise ValueError(f"unknown stream_type: {self.stream_type}")
        if self.frames <= 0:
            raise ValueError(f"frames must be positive for {self.path}")
        if self.dim <= 0:
            raise ValueError(f"dim must be positive for {self.path}")
        if self.rate_hz <= 0:
            raise ValueError(f"rate_hz must be positive for {self.path}")
        if self.path.startswith("/"):
            raise ValueError("feature paths must be relative or use an approved external root")


@dataclass(frozen=True)
class FeatureRecord:
    clip_id: str
    split: str
    duration: float
    visible_speaker_ids: tuple[str, ...]
    streams: tuple[FeatureStream, ...]
    source_clip_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureRecord":
        required = {"clip_id", "split", "duration", "visible_speaker_ids", "streams"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"feature record missing fields: {sorted(missing)}")
        record = cls(
            clip_id=str(payload["clip_id"]),
            split=str(payload["split"]),
            duration=float(payload["duration"]),
            visible_speaker_ids=tuple(str(item) for item in payload["visible_speaker_ids"]),
            streams=tuple(FeatureStream.from_dict(item) for item in payload["streams"]),
            source_clip_id=payload.get("source_clip_id"),
            metadata=dict(payload.get("metadata", {})),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if not self.clip_id:
            raise ValueError("clip_id must be non-empty")
        if self.duration <= 0:
            raise ValueError(f"duration must be positive for {self.clip_id}")
        if not self.visible_speaker_ids:
            raise ValueError(f"visible_speaker_ids must be non-empty for {self.clip_id}")
        stream_types = [stream.stream_type for stream in self.streams]
        if not stream_types:
            raise ValueError(f"at least one feature stream is required for {self.clip_id}")
        if len(set(stream_types)) != len(stream_types):
            raise ValueError(f"duplicate stream types for {self.clip_id}: {stream_types}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "split": self.split,
            "duration": self.duration,
            "visible_speaker_ids": list(self.visible_speaker_ids),
            "streams": [stream.__dict__ for stream in self.streams],
            "source_clip_id": self.source_clip_id,
            "metadata": self.metadata,
        }


def iter_feature_index(path: Path) -> Iterator[FeatureRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                yield FeatureRecord.from_dict(json.loads(stripped))
            except Exception as exc:  # noqa: BLE001 - preserve row context.
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def load_feature_index(path: Path) -> list[FeatureRecord]:
    return list(iter_feature_index(path))
    return records


def stream_counts(records: Iterable[FeatureRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for stream in record.streams:
            counts[stream.stream_type] = counts.get(stream.stream_type, 0) + 1
    return dict(sorted(counts.items()))


def resolve_feature_path(path: str | Path, *, root: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or root is None:
        return candidate
    return root / candidate


def feature_file_report(path: Path, *, root: Path | None = None, max_examples: int = 20) -> dict[str, object]:
    records = 0
    streams = 0
    missing = 0
    examples: list[str] = []
    root = root or Path.cwd()
    for record in iter_feature_index(path):
        records += 1
        for stream in record.streams:
            streams += 1
            resolved = resolve_feature_path(stream.path, root=root)
            if resolved.is_file():
                continue
            missing += 1
            if len(examples) < max_examples:
                try:
                    examples.append(str(resolved.relative_to(root)))
                except ValueError:
                    examples.append(str(resolved))
    return {
        "records": records,
        "streams": streams,
        "missing_stream_files": missing,
        "missing_stream_file_examples": examples,
        "missing_stream_file_examples_truncated": missing > len(examples),
    }


def validate_feature_index_file(path: Path) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    total = 0
    for record in iter_feature_index(path):
        total += 1
        for stream in record.streams:
            counts[stream.stream_type] = counts.get(stream.stream_type, 0) + 1
    return total, dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a CIViC-A2V feature index.")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--require-files", action="store_true")
    args = parser.parse_args()
    total, counts = validate_feature_index_file(args.index)
    print(f"valid_feature_records={total}")
    print(f"stream_counts={counts}")
    if args.require_files:
        report = feature_file_report(args.index, root=args.root)
        print(f"feature_streams={report['streams']}")
        print(f"missing_stream_files={report['missing_stream_files']}")
        if report["missing_stream_files"]:
            print(f"missing_stream_file_examples={report['missing_stream_file_examples']}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
