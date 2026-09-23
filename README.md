# Audio-to-Video Evaluation Toolkit

Small, dependency-light utilities for checking synchronized audio/video record
lists, making deterministic group-aware splits, describing feature streams,
planning audio interventions, and computing numerical evaluation metrics.

The portable branch is code-focused. It does not contain media, trained model
weights, generated videos, private study material, or site-specific launch
files. The tiny example uses relative placeholder paths and is suitable for
local contract checks only.

## Quick Start

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m a2v.data.records --records examples/tiny_records.jsonl
python -m a2v.data.quality --records examples/tiny_records.jsonl
python -m a2v.data.split --input examples/tiny_records.jsonl --output outputs/split_records.jsonl
python -m a2v.data.split_audit --records outputs/split_records.jsonl
```

For the exact lightweight test environment used for this release, use Python
3.12 or newer, create a fresh virtual environment, and install the pinned
requirements before the editable package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install -e .
python -m pytest
```

To compute the compact dataset checks used by the example:

```bash
python -m a2v.data.dataset_metrics \
  --records examples/tiny_records.jsonl \
  --long-form-seconds 5 \
  --json-output outputs/dataset_metrics.json \
  --markdown-output outputs/dataset_metrics.md
```

## Layout

- `a2v/data/` validates records, splits groups, checks leakage, and plans metadata-only interventions.
- `a2v/features/` defines feature-stream records and deterministic synthetic values for I/O tests.
- `a2v/metrics/` provides alignment, interaction, causal, drift, bootstrap, and runtime metrics.
- `examples/` contains a small JSONL record list with no media.
- `tests/` exercises the public Python contracts.
- `a2v/reporting/civic_result_receipts.py` verifies privacy-safe aggregate result receipts.
- `docs/civic_result_receipts.json` records the releasable mixed-condition summaries.
- `docs/reproduction.md` describes the data contract and a repeatable local workflow.
- `docs/ravdess_raster.md` explains the portable raster evaluator, frozen analysis settings, matched-seed and longer-budget receipts, and the exact-input boundary.
- `docs/ravdess_seed_receipt.json` supports a local aggregate check of four fitted-seed comparisons.

## Scope

The utilities operate on user-supplied records and feature files. They do not
claim that the placeholder example represents a benchmark or a trained system.
For real studies, preserve source permissions, file hashes, split decisions,
evaluation settings, and run metadata alongside the data held outside this
repository.
