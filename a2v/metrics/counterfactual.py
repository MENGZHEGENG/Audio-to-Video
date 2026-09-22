"""Counterfactual evaluation helpers.

These helpers define metric bookkeeping for testing whether audio controls the
right visual factors. They intentionally avoid heavyweight dependencies so the
contract can be unit-tested locally before resource-intensive runs are queued.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class CounterfactualScore:
    """Scores for one counterfactual audio/video intervention."""

    clip_id: str
    target_speaker_id: str
    lip_delta_target: float
    lip_delta_non_target: float
    identity_delta: float
    listener_motion_delta: float
    semantic_consistency: float

    @property
    def causal_selectivity(self) -> float:
        return self.lip_delta_target - self.lip_delta_non_target

    @property
    def stable_identity_score(self) -> float:
        return 1.0 / (1.0 + max(self.identity_delta, 0.0))


def aggregate_counterfactual_scores(scores: list[CounterfactualScore]) -> dict[str, float]:
    if not scores:
        raise ValueError("scores must be non-empty")
    return {
        "causal_selectivity": mean(score.causal_selectivity for score in scores),
        "target_lip_response": mean(score.lip_delta_target for score in scores),
        "non_target_leakage": mean(score.lip_delta_non_target for score in scores),
        "identity_stability": mean(score.stable_identity_score for score in scores),
        "listener_responsiveness": mean(score.listener_motion_delta for score in scores),
        "semantic_consistency": mean(score.semantic_consistency for score in scores),
    }
