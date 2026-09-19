from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
import time

from .budget import BudgetExceeded


@dataclass
class RequestBudget:
    max_calls: int
    max_tokens: int
    max_wall_seconds: float
    started_at: float = field(default_factory=time.monotonic)
    calls: int = 0
    tokens: int = 0
    reserved_tokens: int = 0
    stop_reason: str | None = None
    _lock: Lock = field(default_factory=Lock, repr=False)

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_wall_seconds - (time.monotonic() - self.started_at))

    def before_call(self, requested_tokens: int = 0) -> float:
        with self._lock:
            if self.remaining_seconds() <= 0:
                self.stop_reason = "wall_time_budget_exhausted"
                raise BudgetExceeded(self.stop_reason)
            if self.calls >= self.max_calls:
                self.stop_reason = "model_call_budget_exhausted"
                raise BudgetExceeded(self.stop_reason)
            if self.tokens + self.reserved_tokens + max(requested_tokens, 0) > self.max_tokens:
                self.stop_reason = "tokens_budget_exhausted"
                raise BudgetExceeded(self.stop_reason)
            self.calls += 1
            self.reserved_tokens += max(requested_tokens, 0)
            return self.remaining_seconds()

    def after_call(self, prompt_tokens: int, completion_tokens: int, reserved_tokens: int = 0) -> None:
        with self._lock:
            self.reserved_tokens = max(0, self.reserved_tokens - max(reserved_tokens, 0))
            self.tokens += max(prompt_tokens, 0) + max(completion_tokens, 0)
            if self.tokens > self.max_tokens:
                self.stop_reason = "tokens_budget_exhausted"
                raise BudgetExceeded(self.stop_reason)

    def cancel_call(self, reserved_tokens: int) -> None:
        with self._lock:
            self.reserved_tokens = max(0, self.reserved_tokens - max(reserved_tokens, 0))

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "tokens": self.tokens,
            "reserved_tokens": self.reserved_tokens,
            "wall_seconds": time.monotonic() - self.started_at,
            "stop_reason": self.stop_reason,
        }


_ACTIVE_BUDGET: RequestBudget | None = None
_ACTIVE_LOCK = Lock()


def active_request_budget() -> RequestBudget | None:
    with _ACTIVE_LOCK:
        return _ACTIVE_BUDGET


@contextmanager
def request_budget(budget: RequestBudget):
    global _ACTIVE_BUDGET
    with _ACTIVE_LOCK:
        if _ACTIVE_BUDGET is not None:
            raise RuntimeError("nested request budgets are not supported")
        _ACTIVE_BUDGET = budget
    try:
        yield budget
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_BUDGET = None
