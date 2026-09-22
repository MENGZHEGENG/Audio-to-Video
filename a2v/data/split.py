"""Deterministic group-aware records splitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from a2v.data.records import ClipRecord, load_records, split_counts

SPLIT_NAMES = ("train", "val", "test")


def normalize_ratios(train: float, val: float, test: float) -> dict[str, float]:
    ratios = {"train": float(train), "val": float(val), "test": float(test)}
    if any(value < 0 for value in ratios.values()):
        raise ValueError(f"split ratios must be non-negative: {ratios}")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("at least one split ratio must be positive")
    return {key: value / total for key, value in ratios.items()}


def stable_group_key(record: ClipRecord, group_key: str, fallback_key: str = "clip_id") -> str:
    def lookup(key: str) -> Any:
        if key.startswith("metadata."):
            return record.metadata.get(key.split(".", 1)[1])
        if hasattr(record, key):
            return getattr(record, key)
        raise ValueError(f"unknown records group key: {key}")

    value = lookup(group_key)
    if value in (None, ""):
        value = lookup(fallback_key)
    return str(value)


def _stable_shuffle(items: list[tuple[str, list[ClipRecord]]], seed: int) -> list[tuple[str, list[ClipRecord]]]:
    keyed = []
    for key, records in items:
        digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
        keyed.append((digest, key, records))
    return [(key, records) for _, key, records in sorted(keyed)]


def assign_splits(
    records: list[ClipRecord],
    *,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 1234,
    group_key: str = "metadata.conversation_id",
    fallback_key: str = "clip_id",
) -> list[ClipRecord]:
    if not records:
        raise ValueError("records must be non-empty")
    ratios = normalize_ratios(train, val, test)
    grouped: dict[str, list[ClipRecord]] = {}
    for record in records:
        grouped.setdefault(stable_group_key(record, group_key, fallback_key=fallback_key), []).append(record)

    shuffled_groups = _stable_shuffle(list(grouped.items()), seed)
    targets = {split: ratios[split] * len(records) for split in SPLIT_NAMES}
    assigned_counts = {split: 0 for split in SPLIT_NAMES}
    assignments: dict[str, str] = {}
    for group, group_records in shuffled_groups:
        split = max(
            SPLIT_NAMES,
            key=lambda name: (targets[name] - assigned_counts[name], -SPLIT_NAMES.index(name)),
        )
        assignments[group] = split
        assigned_counts[split] += len(group_records)

    output: list[ClipRecord] = []
    for record in records:
        group = stable_group_key(record, group_key, fallback_key=fallback_key)
        output.append(replace(record, split=assignments[group]))
    return output


def write_records(records: list[ClipRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic group-aware train/val/test records splits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--group-key", default="metadata.conversation_id")
    parser.add_argument("--fallback-key", default="clip_id")
    args = parser.parse_args()
    records = assign_splits(
        load_records(args.input),
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed,
        group_key=args.group_key,
        fallback_key=args.fallback_key,
    )
    write_records(records, args.output)
    print(f"wrote_records={len(records)}")
    print(f"split_counts={split_counts(records)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
