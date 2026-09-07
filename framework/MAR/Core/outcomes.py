from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class OutcomeClass(str, Enum):
    PASS = "PASS"
    PROGRAM_FAILURE = "PROGRAM_FAILURE"
    CONTRACT_FAILURE = "CONTRACT_FAILURE"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    DATASET_FAILURE = "DATASET_FAILURE"
    VERIFIER_UNAVAILABLE = "VERIFIER_UNAVAILABLE"


@dataclass(frozen=True)
class Outcome:
    category: OutcomeClass
    status: str
    raw_reward: float | None
    policy_eligible: bool
    details: Mapping[str, Any]


_ENVIRONMENT_STATUSES = {
    "ENVIRONMENT_ERROR",
    "RUNTIME_ERROR",
    "DOCKER_ERROR",
    "SETUP_ERROR",
    "TIMEOUT_INFRASTRUCTURE",
}
_CONTRACT_STATUSES = {"AMBIGUOUS_SPEC", "MISSING_CONTRACT", "CLARIFICATION_REQUIRED"}
_PROGRAM_STATUSES = {
    "COMPILE_FAIL",
    "COMPILE_ERROR",
    "ASSERTION_FAIL",
    "ASSERTION_FAILURE",
    "INTERFACE_ERROR",
    "WRONG_OUTPUT",
    "FUNCTION_FAIL",
    "RUNTIME_FAIL",
    "FAIL",
    "BUDGET_EXHAUSTED",
}


def classify_outcome(result: Mapping[str, Any] | None) -> Outcome:
    if result is None:
        return Outcome(OutcomeClass.VERIFIER_UNAVAILABLE, "UNAVAILABLE", None, False, {})

    status = str(result.get("status", "UNKNOWN")).upper()
    details = dict(result)
    if status == "PASS":
        return Outcome(OutcomeClass.PASS, status, 1.0, True, details)
    if status == "DATASET_ERROR":
        return Outcome(OutcomeClass.DATASET_FAILURE, status, None, False, details)
    if status in _ENVIRONMENT_STATUSES:
        return Outcome(OutcomeClass.ENVIRONMENT_FAILURE, status, None, False, details)
    if status in _CONTRACT_STATUSES:
        return Outcome(OutcomeClass.CONTRACT_FAILURE, status, None, False, details)
    if status in _PROGRAM_STATUSES:
        return Outcome(OutcomeClass.PROGRAM_FAILURE, status, 0.0, True, details)
    return Outcome(OutcomeClass.ENVIRONMENT_FAILURE, status, None, False, details)
