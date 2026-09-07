from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    name: str
    functional_pass_rate: float
    mean_cost: float
    mean_latency: float
    payload: dict[str, Any] = field(default_factory=dict)

    def dominates(self, other: "Candidate") -> bool:
        no_worse = (
            self.functional_pass_rate >= other.functional_pass_rate
            and self.mean_cost <= other.mean_cost
            and self.mean_latency <= other.mean_latency
        )
        strictly_better = (
            self.functional_pass_rate > other.functional_pass_rate
            or self.mean_cost < other.mean_cost
            or self.mean_latency < other.mean_latency
        )
        return no_worse and strictly_better


class ParetoArchive:
    def __init__(self, max_size: int = 8):
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._candidates: list[Candidate] = []

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        return tuple(self._candidates)

    def add(self, candidate: Candidate) -> bool:
        if any(existing.dominates(candidate) for existing in self._candidates):
            return False
        self._candidates = [existing for existing in self._candidates if not candidate.dominates(existing)]
        self._candidates.append(candidate)
        self._candidates.sort(key=lambda item: (-item.functional_pass_rate, item.mean_cost, item.mean_latency, item.name))
        self._candidates = self._candidates[: self.max_size]
        return candidate in self._candidates
