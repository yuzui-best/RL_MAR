from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .outcomes import Outcome, classify_outcome
from .reward import RewardConfig, RewardResult, compute_reward


@dataclass(frozen=True)
class BehaviorEvaluation:
    outcome: Outcome
    reward: RewardResult
    selected_attempt: Mapping[str, Any] | None


def selected_attempt(records: Sequence[Mapping[str, Any]] | None) -> Mapping[str, Any] | None:
    records = records or []
    selected = [record for record in records if record.get("selected")]
    if selected:
        return selected[-1]
    return records[-1] if records else None


def evaluate_attempts(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    total_tokens: int,
    latency_seconds: float,
    retries: int,
    agent_calls: int,
    reward_config: RewardConfig,
) -> BehaviorEvaluation:
    attempt = selected_attempt(records)
    result = {"status": attempt.get("status", "UNAVAILABLE")} if attempt else None
    outcome = classify_outcome(result)
    reward = compute_reward(
        outcome,
        total_tokens=total_tokens,
        latency_seconds=latency_seconds,
        retries=retries,
        agent_calls=agent_calls,
        config=reward_config,
    )
    return BehaviorEvaluation(outcome, reward, attempt)
