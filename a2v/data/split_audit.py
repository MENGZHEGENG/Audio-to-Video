"""Audit train/validation/test records for group leakage."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from collections.abc import Iterable

from a2v.data.records import ClipRecord, iter_records
from a2v.data.split import stable_group_key


@dataclass(frozen=True)
class SplitLeakageReport:
    records: int
    group_key: str
    fallback_key: str
    groups: int
    leaking_groups: dict[str, list[str]]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_split_leakage(
    records: Iterable[ClipRecord],
    *,
    group_key: str = "metadata.conversation_id",
    fallback_key: str = "clip_id",
) -> SplitLeakageReport:
    group_to_splits: dict[str, set[str]] = {}
    total = 0
    for record in records:
        total += 1
        group = stable_group_key(record, group_key, fallback_key=fallback_key)
        group_to_splits.setdefault(group, set()).add(record.split)
    leaking_groups = {
        group: sorted(splits)
        for group, splits in sorted(group_to_splits.items())
        if len(splits) > 1
    }
    return SplitLeakageReport(
        records=total,
        group_key=group_key,
        fallback_key=fallback_key,
        groups=len(group_to_splits),
        leaking_groups=leaking_groups,
        passed=not leaking_groups,
    )


def render_split_leakage_markdown(report: SplitLeakageReport) -> str:
    lines = [
        "# Records Split Leakage Audit",
        "",
        f"Status: **{'PASS' if report.passed else 'FAIL'}**",
        f"Records: {report.records}",
        f"Groups: {report.groups}",
        f"Group key: `{report.group_key}`",
        f"Fallback key: `{report.fallback_key}`",
    ]
    if report.leaking_groups:
        lines.extend(
            [
                "",
                "## Leaking Groups",
                "| Group | Splits |",
                "|---|---|",
            ]
        )
        for group, splits in report.leaking_groups.items():
            lines.append(f"| `{group}` | `{splits}` |")
    else:
        lines.extend(["", "No group appears in more than one split."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit A2V records for identity/conversation split leakage.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--group-key", default="metadata.conversation_id")
    parser.add_argument("--fallback-key", default="clip_id")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = audit_split_leakage(
        iter_records(args.records),
        group_key=args.group_key,
        fallback_key=args.fallback_key,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_split_leakage_markdown(report), encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not report.passed and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
