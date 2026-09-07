from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrajectoryStep:
    stage: str
    status: str
    code_sha256: str | None = None
    raw_reward: float | None = None
    shaped_reward: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    event_type: str = "stage"
    request_sha256: str | None = None
    response_sha256: str | None = None
    retries: int = 0
    agent_calls: int = 0
    monetary_cost: float = 0.0
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryLedger:
    task_id: str
    model: str
    topology: str
    roles: list[str]
    source_sha256: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    stop_reason: str | None = None
    route_action: dict[str, Any] = field(default_factory=dict)
    action_log_probability: float | None = None
    raw_reward: float | None = None
    training_reward: float | None = None
    policy_eligible: bool = False

    @classmethod
    def create(cls, *, task_id: str, model: str, topology: str, roles: list[str], source: str) -> "TrajectoryLedger":
        return cls(task_id, model, topology, roles, sha256_text(source))

    def append(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
