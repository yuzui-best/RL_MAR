from __future__ import annotations

from dataclasses import dataclass, field
import time


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetEnvelope:
    max_agents: int
    max_tokens: int
    max_wall_seconds: float
    max_repairs: int
    max_validations: int
    started_at: float = field(default_factory=time.monotonic)
    agents: int = 0
    tokens: int = 0
    repairs: int = 0
    validations: int = 0
    stop_reason: str | None = None

    def consume(self, *, agents: int = 0, tokens: int = 0, repairs: int = 0, validations: int = 0) -> None:
        proposed = {
            "agents": self.agents + agents,
            "tokens": self.tokens + tokens,
            "repairs": self.repairs + repairs,
            "validations": self.validations + validations,
        }
        limits = {
            "agents": self.max_agents,
            "tokens": self.max_tokens,
            "repairs": self.max_repairs,
            "validations": self.max_validations,
        }
        for name, value in proposed.items():
            if value > limits[name]:
                self.stop_reason = f"{name}_budget_exhausted"
                raise BudgetExceeded(self.stop_reason)
        if time.monotonic() - self.started_at > self.max_wall_seconds:
            self.stop_reason = "wall_time_budget_exhausted"
            raise BudgetExceeded(self.stop_reason)
        self.agents, self.tokens = proposed["agents"], proposed["tokens"]
        self.repairs, self.validations = proposed["repairs"], proposed["validations"]

    def snapshot(self) -> dict[str, int | float | str | None]:
        return {
            "agents": self.agents,
            "tokens": self.tokens,
            "repairs": self.repairs,
            "validations": self.validations,
            "wall_seconds": time.monotonic() - self.started_at,
            "stop_reason": self.stop_reason,
        }
