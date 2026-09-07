from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingDecision:
    model: str
    predicted_strong_win_rate: float
    threshold: float
    reason: str


class CostQualityRouter:
    def __init__(self, *, weak_model: str, strong_model: str, threshold: float):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        self.weak_model = weak_model
        self.strong_model = strong_model
        self.threshold = threshold

    def route(self, predicted_strong_win_rate: float, *, budget_pressure: float = 0.0) -> RoutingDecision:
        probability = min(max(predicted_strong_win_rate, 0.0), 1.0)
        pressure = min(max(budget_pressure, 0.0), 1.0)
        effective_threshold = min(1.0, self.threshold + 0.25 * pressure)
        use_strong = probability >= effective_threshold
        return RoutingDecision(
            model=self.strong_model if use_strong else self.weak_model,
            predicted_strong_win_rate=probability,
            threshold=effective_threshold,
            reason="quality_gain_exceeds_threshold" if use_strong else "weak_model_satisfies_cost_quality_gate",
        )
