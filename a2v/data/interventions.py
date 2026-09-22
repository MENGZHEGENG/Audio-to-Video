"""Counterfactual records generation for causal audio-to-video evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from a2v.data.records import ClipRecord, iter_records


@dataclass(frozen=True)
class AudioIntervention:
    """A planned audio intervention without modifying raw media files."""

    intervention_id: str
    clip_id: str
    kind: str
    target_speaker_id: str
    partner_speaker_id: str | None = None
    delay_ms: int | None = None

    def validate(self, record: ClipRecord) -> None:
        visible = set(record.visible_speaker_ids)
        if self.target_speaker_id not in visible:
            raise ValueError(f"target speaker {self.target_speaker_id!r} is not visible")
        if self.partner_speaker_id is not None and self.partner_speaker_id not in visible:
            raise ValueError(f"partner speaker {self.partner_speaker_id!r} is not visible")
        if self.kind not in {"mute", "delay", "swap_content", "swap_prosody"}:
            raise ValueError(f"unknown intervention kind: {self.kind}")
        if self.kind == "delay" and (self.delay_ms is None or self.delay_ms <= 0):
            raise ValueError("delay interventions require positive delay_ms")
        if self.kind in {"swap_content", "swap_prosody"} and not self.partner_speaker_id:
            raise ValueError(f"{self.kind} interventions require partner_speaker_id")


def build_interventions(record: ClipRecord, delay_ms: int = 400) -> list[AudioIntervention]:
    """Create intervention metadata for one clip."""

    interventions: list[AudioIntervention] = []
    speakers = tuple(record.visible_speaker_ids)
    for speaker_id in speakers:
        interventions.append(
            AudioIntervention(
                intervention_id=f"{record.clip_id}__mute__{speaker_id}",
                clip_id=record.clip_id,
                kind="mute",
                target_speaker_id=speaker_id,
            )
        )
        interventions.append(
            AudioIntervention(
                intervention_id=f"{record.clip_id}__delay{delay_ms}__{speaker_id}",
                clip_id=record.clip_id,
                kind="delay",
                target_speaker_id=speaker_id,
                delay_ms=delay_ms,
            )
        )
    if len(speakers) >= 2:
        for target in speakers:
            for partner in speakers:
                if target == partner:
                    continue
                interventions.append(
                    AudioIntervention(
                        intervention_id=f"{record.clip_id}__swap_content__{target}__{partner}",
                        clip_id=record.clip_id,
                        kind="swap_content",
                        target_speaker_id=target,
                        partner_speaker_id=partner,
                    )
                )
                interventions.append(
                    AudioIntervention(
                        intervention_id=f"{record.clip_id}__swap_prosody__{target}__{partner}",
                        clip_id=record.clip_id,
                        kind="swap_prosody",
                        target_speaker_id=target,
                        partner_speaker_id=partner,
                    )
                )
    for intervention in interventions:
        intervention.validate(record)
    return interventions


def intervention_record(record: ClipRecord, intervention: AudioIntervention) -> dict[str, Any]:
    payload = record.to_dict()
    payload["split"] = "counterfactual"
    payload["clip_id"] = intervention.intervention_id
    payload.setdefault("metadata", {})["source_clip_id"] = record.clip_id
    payload["metadata"]["intervention"] = asdict(intervention)
    return payload


def write_counterfactual_records(
    records: Iterable[ClipRecord],
    output: Path,
    delay_ms: int,
    *,
    max_records: int | None = None,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    source_records = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            if max_records is not None and source_records >= max_records:
                break
            source_records += 1
            for intervention in build_interventions(record, delay_ms=delay_ms):
                handle.write(json.dumps(intervention_record(record, intervention), sort_keys=True) + "\n")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata-only counterfactual records.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay-ms", type=int, default=400)
    parser.add_argument("--max-records", type=int, help="Limit source clips for a small pilot.")
    args = parser.parse_args()
    count = write_counterfactual_records(
        iter_records(args.records),
        args.output,
        args.delay_ms,
        max_records=args.max_records,
    )
    print(f"wrote_counterfactual_records={count}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
