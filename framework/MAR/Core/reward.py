from __future__ import annotations

from dataclasses import dataclass

from .outcomes import Outcome
from .verifier import summarize_quality


@dataclass(frozen=True)
class RewardConfig:
    token_penalty: float = 0.0
    latency_penalty: float = 0.0
    retry_penalty: float = 0.0
    agent_call_penalty: float = 0.0
    penalize_failed_efficiency: bool = False


@dataclass(frozen=True)
class RewardResult:
    raw_reward: float | None
    shaped_reward: float | None
    policy_eligible: bool
    penalty: float
    components: dict[str, float] | None = None


@dataclass(frozen=True)
class RewardNormalizers:
    tokens_p95: float
    latency_p95: float
    retries_p95: float
    agent_calls_p95: float

    def __post_init__(self) -> None:
        if min(self.tokens_p95, self.latency_p95, self.retries_p95, self.agent_calls_p95) <= 0:
            raise ValueError("all P95 normalizers must be positive")


def _normalized(value: float, p95: float) -> float:
    return min(max(value, 0.0) / p95, 1.0)


def compute_behavior_reward(
    outcome: Outcome,
    *,
    verifier_result: dict,
    normalizers: RewardNormalizers,
    total_tokens: int = 0,
    latency_seconds: float = 0.0,
    retries: int = 0,
    agent_calls: int = 0,
) -> RewardResult:
    if not outcome.policy_eligible or outcome.raw_reward is None:
        return RewardResult(outcome.raw_reward, None, False, 0.0, {})
    if outcome.raw_reward == 0.0:
        quality = summarize_quality(verifier_result)
        components = {
            "stage_progress": 0.10 * quality["stage_fraction"],
            "check_progress": 0.15 * quality["check_fraction"],
            "activation_progress": 0.10 * quality["activation_fraction"],
            "violation_progress": 0.05 * quality["violation_reduction"],
        }
        shaped = sum(components.values())
        return RewardResult(0.0, min(shaped, 0.40), True, 0.0, components)
    components = {
        "token_penalty": 0.15 * _normalized(total_tokens, normalizers.tokens_p95),
        "latency_penalty": 0.08 * _normalized(latency_seconds, normalizers.latency_p95),
        "retry_penalty": 0.04 * _normalized(retries, normalizers.retries_p95),
        "agent_call_penalty": 0.03 * _normalized(agent_calls, normalizers.agent_calls_p95),
    }
    penalty = min(sum(components.values()), 0.30)
    return RewardResult(1.0, 1.0 - penalty, True, penalty, components)


def compute_reward(
    outcome: Outcome,
    *,
    total_tokens: int = 0,
    latency_seconds: float = 0.0,
    retries: int = 0,
    agent_calls: int = 0,
    config: RewardConfig | None = None,
) -> RewardResult:
    config = config or RewardConfig()
    if not outcome.policy_eligible or outcome.raw_reward is None:
        return RewardResult(outcome.raw_reward, None, False, 0.0)

    apply_efficiency = outcome.raw_reward > 0.0 or config.penalize_failed_efficiency
    penalty = 0.0
    if apply_efficiency:
        penalty = (
            max(total_tokens, 0) * config.token_penalty
            + max(latency_seconds, 0.0) * config.latency_penalty
            + max(retries, 0) * config.retry_penalty
            + max(agent_calls, 0) * config.agent_call_penalty
        )
    return RewardResult(outcome.raw_reward, outcome.raw_reward - penalty, True, penalty)
