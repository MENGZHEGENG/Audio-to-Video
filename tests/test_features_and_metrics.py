import numpy as np
import pytest

from a2v.features.schema import FeatureRecord, FeatureStream
from a2v.features.synthetic import synthetic_values
from a2v.metrics.av_alignment import av_alignment_report
from a2v.metrics.counterfactual import CounterfactualScore, aggregate_counterfactual_scores
from a2v.metrics.civic import CivicMetricRow, aggregate_civic_rows


def test_feature_schema_rejects_absolute_paths_and_synthetic_values_are_stable() -> None:
    stream = FeatureStream("audio", "features/audio.npz", 8, 3, 25.0)
    record = FeatureRecord("clip", "train", 2.0, ("spk-1",), (stream,))
    first = synthetic_values(record, stream)
    second = synthetic_values(record, stream)
    assert first.shape == (8, 3)
    np.testing.assert_array_equal(first, second)

    bad = FeatureStream("audio", "/outside/audio.npz", 8, 3, 25.0)
    try:
        bad.validate()
    except ValueError as exc:
        assert "relative" in str(exc)
    else:  # pragma: no cover - defensive assertion.
        raise AssertionError("absolute feature path was accepted")


def test_alignment_and_causal_metric_contracts() -> None:
    signal = np.sin(np.linspace(0.0, 4.0 * np.pi, 32, dtype=np.float32))[:, None]
    report = av_alignment_report(audio=signal, video=signal, fps=25.0)
    assert report["frames"] == 32
    assert report["audio_video_sync"] > 0.99

    score = CounterfactualScore("clip", "spk-1", 0.8, 0.1, 0.9, 0.4, 0.7)
    aggregate = aggregate_counterfactual_scores([score])
    assert aggregate["causal_selectivity"] == pytest.approx(0.7)
    assert aggregate["identity_stability"] == pytest.approx(1.0 / 1.9)

    row = CivicMetricRow("clip", "mute", 0.8, 0.1, 0.9, 0.7, 10.0, 0.2)
    summary = aggregate_civic_rows([row])
    assert summary["causal_selectivity"] == 0.8
    assert summary["turn_timing_error_ms"] == 10.0
