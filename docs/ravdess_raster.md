# RAVDESS raster diagnostics

The `a2v/data`, `a2v/models`, `a2v/training`, and `a2v/evaluation` modules in
this release contain the 64 × 64 raster preprocessing, AudioFiLM renderer,
registered-control scoring, matched-stimulus controls, constant-audio fit, and
matched-seed and training-budget sensitivity code. The four JSON files in `experiments/` retain
the executed exploratory analysis settings and input digests.

## Lightweight numeric check

With the base environment from the README, run:

```bash
python -m a2v.reporting.ravdess_seed_receipt docs/ravdess_seed_receipt.json
python -m a2v.reporting.ravdess_budget_receipt docs/ravdess_budget_receipt.json
python -m pytest tests/test_ravdess_seed_receipt.py
python -m pytest tests/test_ravdess_budget_receipt.py
```

The receipt commands recompute the four seed-level constant-minus-audio MAE
differences for each maximum-epoch setting, their means and ranges, and the
number with a negative difference.
The receipt contains aggregate scores only. It does not contain the 240 scored
pairs, model weights, video, audio, or feature arrays, so it cannot recompute
the pair or actor bootstrap intervals.

## Full fit and score inputs

Use a separate environment with Python 3.10.20, the packages pinned in
`requirements-raster.txt`, and FFmpeg 9.0.1. On other operating systems,
install a compatible FFmpeg binary and verify that frame and audio decoding
matches the expected 64 × 64 RGB and 16 kHz mono arrays.

Obtain RAVDESS from its data provider under its terms. The scripts accept a
registered renderer specification, development and validation feature report,
feature-provenance gate, confirmation protocol, media root, and feature root
as explicit command-line inputs. The feature report lists 960 development and
240 validation records. Confirmation contains 240 targets from four separate
actors. Audio uses 1.0–1.5 seconds, identity uses the frame at 0.25 seconds,
and the target is the frame at 2.0 seconds. The executable entry points are:

- `python -m a2v.training.ravdess_raster_renderer --help`
- `python -m a2v.evaluation.ravdess_raster_confirmation --help`
- `python -m a2v.evaluation.ravdess_raster_factorial --help`
- `python -m a2v.evaluation.ravdess_constant_audio_comparator --help`
- `python -m a2v.evaluation.ravdess_seed_sensitivity --help`
- `python -m a2v.evaluation.ravdess_budget_sensitivity --help`

The full scored run needs the hash-matching input JSON and feature arrays,
which are not redistributed here. The original feature gate contains a
site-specific storage path. The released analysis plans preserve its digest,
so a different gate file fails their exact-input check. This is intentional:
replacing that path without a new receipt would falsely present a new run as
the executed one. The source and frozen settings support an authorized rerun
once the corresponding inputs are supplied. The aggregate verification above
is runnable without them.

The constant-audio, seed, and longer-budget analyses were designed after the
confirmation actors had been opened. They are descriptive same-target checks;
they do not change the registered audio-replacement decision or establish
performance across a population of actors or real video quality.
The longer-budget fits start afresh. Their validation histories differ before
epoch 40 from the earlier fits, so differences between budget settings are not
an isolated measure of extra training epochs.
