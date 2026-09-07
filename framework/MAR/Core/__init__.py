"""RL-MAR routing, verification, and optimisation primitives."""

from .budget import BudgetEnvelope, BudgetExceeded
from .evolution import Candidate, ParetoArchive
from .integration import BehaviorEvaluation, evaluate_attempts, selected_attempt
from .manifest import FrozenManifest, file_sha256
from .outcomes import Outcome, OutcomeClass, classify_outcome
from .quality_controller import QualityAction, QualityController
from .request_budget import RequestBudget, active_request_budget, request_budget
from .policy import ContextualRoutePolicy, PolicySample, RouteAction, RouteSpace, clipped_policy_loss, group_relative_advantages
from .reward import RewardConfig, RewardNormalizers, RewardResult, compute_behavior_reward, compute_reward
from .routing import CostQualityRouter, RoutingDecision
from .trajectory import TrajectoryLedger, TrajectoryStep
from .verifier import VerifierConfig, summarize_quality, verify_rewardplc

__all__ = [
    "BudgetEnvelope",
    "BudgetExceeded",
    "BehaviorEvaluation",
    "Candidate",
    "CostQualityRouter",
    "Outcome",
    "OutcomeClass",
    "QualityAction",
    "QualityController",
    "RequestBudget",
    "ContextualRoutePolicy",
    "FrozenManifest",
    "PolicySample",
    "RouteAction",
    "RouteSpace",
    "ParetoArchive",
    "RewardConfig",
    "RewardNormalizers",
    "RewardResult",
    "RoutingDecision",
    "TrajectoryLedger",
    "TrajectoryStep",
    "classify_outcome",
    "active_request_budget",
    "clipped_policy_loss",
    "compute_reward",
    "compute_behavior_reward",
    "evaluate_attempts",
    "file_sha256",
    "group_relative_advantages",
    "request_budget",
    "selected_attempt",
    "VerifierConfig",
    "summarize_quality",
    "verify_rewardplc",
]
