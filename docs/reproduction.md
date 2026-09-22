# Reproduction Workflow

This document describes a small local workflow for the reusable contracts in
the toolkit. It does not require media for the record and split checks.

## 1. Install

Use Python 3.10 or newer in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The numerical metrics require NumPy. The optional YAML reader is not needed by
the portable modules.

## 2. Validate Records

Each JSONL row describes one synchronized clip. Required fields are
`clip_id`, `video_path`, `audio_path`, `duration`, `fps`, `split`, and
`visible_speaker_ids`. Optional diarized segments must use the same speaker IDs
and remain within the clip duration.

```bash
python -m a2v.data.records --records examples/tiny_records.jsonl
python -m a2v.data.quality --records examples/tiny_records.jsonl
```

The paths are metadata only. The validators do not open the media files.

## 3. Make A Group-Aware Split

Assign all rows from the same conversation or identity group to one split:

```bash
python -m a2v.data.split \
  --input examples/tiny_records.jsonl \
  --output outputs/split_records.jsonl \
  --group-key metadata.conversation_id \
  --seed 1234
python -m a2v.data.split_audit \
  --records outputs/split_records.jsonl
```

The audit fails if a group occurs in more than one split. Keep the seed and
group key with the run metadata so the assignment can be regenerated.

## 4. Plan Interventions

Intervention rows carry metadata only. They do not alter source media:

```bash
python -m a2v.data.interventions \
  --records examples/tiny_records.jsonl \
  --output outputs/counterfactual_records.jsonl \
  --max-records 2
```

## 5. Run Numerical Checks

The modules in `a2v/metrics/` accept NumPy arrays or JSONL metric rows. The
tests show small in-memory examples for alignment, interaction, causal
selectivity, and bootstrap calculations. For a new evaluation, record array
shapes, axis conventions, units, seeds, and the exact metric options.

## Reproducibility Boundary

This branch supplies reusable code and a toy input. It does not reproduce a
model training run by itself. A complete study also needs permissioned source
data, a fixed preprocessing recipe, model configuration, checkpoints held in
appropriate storage, and evaluation outputs with their provenance records.
