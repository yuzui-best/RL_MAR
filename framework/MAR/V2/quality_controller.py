from __future__ import annotations

from enum import Enum

from .outcomes import Outcome, OutcomeClass


class QualityAction(str, Enum):
    SUBMIT = "SUBMIT"
    REPAIR = "REPAIR"
    CLARIFY = "CLARIFY"
    REVERT = "REVERT"
    ABORT_ENVIRONMENT = "ABORT_ENVIRONMENT"


class QualityController:
    def decide(self, current: Outcome, previous_score: tuple[int, ...] | None = None, current_score: tuple[int, ...] | None = None) -> QualityAction:
        if current.category is OutcomeClass.PASS:
            return QualityAction.SUBMIT
        if current.category is OutcomeClass.CONTRACT_FAILURE:
            return QualityAction.CLARIFY
        if current.category in {OutcomeClass.ENVIRONMENT_FAILURE, OutcomeClass.DATASET_FAILURE, OutcomeClass.VERIFIER_UNAVAILABLE}:
            return QualityAction.ABORT_ENVIRONMENT
        if previous_score is not None and current_score is not None and current_score < previous_score:
            return QualityAction.REVERT
        return QualityAction.REPAIR
