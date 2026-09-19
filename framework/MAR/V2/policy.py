from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RouteSpace:
    topologies: tuple[str, ...] = ("IO", "CoT", "Chain", "FullConnected", "Debate", "Reflection")
    max_roles: int = 6
    max_agents: int = 6
    rounds: tuple[int, ...] = (1, 2, 3)
    repair_budgets: tuple[int, ...] = (0, 1, 2, 4)
    token_budgets: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536)


@dataclass(frozen=True)
class RouteAction:
    topology_index: int
    role_indices: tuple[int, ...]
    agent_count: int
    rounds: int
    repair_budget: int
    token_budget: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PolicySample:
    action: RouteAction
    log_probability: torch.Tensor
    entropy: torch.Tensor


class ContextualRoutePolicy(nn.Module):
    def __init__(self, *, input_dim: int = 384, domains: int = 5, models: int = 5, hidden_dim: int = 128, space: RouteSpace | None = None):
        super().__init__()
        self.space = space or RouteSpace()
        self.model_embedding = nn.Embedding(models, 16)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim + domains + 16, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.topology_head = nn.Linear(hidden_dim, len(self.space.topologies))
        self.agent_head = nn.Linear(hidden_dim, self.space.max_agents)
        self.round_head = nn.Linear(hidden_dim, len(self.space.rounds))
        self.repair_head = nn.Linear(hidden_dim, len(self.space.repair_budgets))
        self.token_head = nn.Linear(hidden_dim, len(self.space.token_budgets))
        self.role_head = nn.Linear(hidden_dim, self.space.max_roles)

    def _features(self, embeddings: torch.Tensor, domain_probs: torch.Tensor, model_ids: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or domain_probs.ndim != 2:
            raise ValueError("embeddings and domain_probs must be rank-2")
        return self.trunk(torch.cat([embeddings, domain_probs, self.model_embedding(model_ids)], dim=-1))

    @staticmethod
    def _categorical(logits: torch.Tensor, deterministic: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = torch.distributions.Categorical(logits=logits)
        selected = logits.argmax(-1) if deterministic else distribution.sample()
        return selected, distribution.log_prob(selected), distribution.entropy()

    @staticmethod
    def _roles(logits: torch.Tensor, count: int, deterministic: bool) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor]:
        available = torch.ones_like(logits, dtype=torch.bool)
        chosen: list[int] = []
        log_probability = logits.new_zeros(())
        entropy = logits.new_zeros(())
        for _ in range(min(count, logits.numel())):
            masked = logits.masked_fill(~available, float("-inf"))
            distribution = torch.distributions.Categorical(logits=masked)
            selected = masked.argmax() if deterministic else distribution.sample()
            log_probability = log_probability + distribution.log_prob(selected)
            entropy = entropy + distribution.entropy()
            index = int(selected.item())
            chosen.append(index)
            available[index] = False
        return tuple(chosen), log_probability, entropy

    def sample(self, embeddings: torch.Tensor, domain_probs: torch.Tensor, model_ids: torch.Tensor, *, deterministic: bool = False, repair_budget: int | None = None) -> list[PolicySample]:
        if repair_budget is not None and repair_budget not in self.space.repair_budgets:
            raise ValueError(f"unsupported repair budget: {repair_budget}")
        hidden = self._features(embeddings, domain_probs, model_ids)
        outputs: list[PolicySample] = []
        for index in range(hidden.shape[0]):
            topo, topo_lp, topo_h = self._categorical(self.topology_head(hidden[index]), deterministic)
            agents, agents_lp, agents_h = self._categorical(self.agent_head(hidden[index]), deterministic)
            rounds, rounds_lp, rounds_h = self._categorical(self.round_head(hidden[index]), deterministic)
            repair_logits = self.repair_head(hidden[index])
            if repair_budget is not None:
                repair_mask = torch.zeros_like(repair_logits, dtype=torch.bool)
                repair_mask[self.space.repair_budgets.index(repair_budget)] = True
                repair_logits = repair_logits.masked_fill(~repair_mask, float("-inf"))
            repairs, repairs_lp, repairs_h = self._categorical(repair_logits, deterministic)
            tokens, tokens_lp, tokens_h = self._categorical(self.token_head(hidden[index]), deterministic)
            agent_count = int(agents.item()) + 1
            roles, roles_lp, roles_h = self._roles(self.role_head(hidden[index]), agent_count, deterministic)
            action = RouteAction(
                int(topo.item()), roles, agent_count, self.space.rounds[int(rounds.item())],
                self.space.repair_budgets[int(repairs.item())], self.space.token_budgets[int(tokens.item())],
            )
            outputs.append(PolicySample(action, topo_lp + agents_lp + rounds_lp + repairs_lp + tokens_lp + roles_lp, topo_h + agents_h + rounds_h + repairs_h + tokens_h + roles_h))
        return outputs

    def log_probability(self, embeddings: torch.Tensor, domain_probs: torch.Tensor, model_ids: torch.Tensor, actions: Sequence[RouteAction], *, repair_budgets: Sequence[int | None] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if repair_budgets is None:
            repair_budgets = [None] * len(actions)
        if len(repair_budgets) != len(actions):
            raise ValueError("repair constraints and actions must have the same length")
        hidden = self._features(embeddings, domain_probs, model_ids)
        log_probabilities, entropies = [], []
        for index, action in enumerate(actions):
            repair_logits = self.repair_head(hidden[index])
            constraint = repair_budgets[index]
            if constraint is not None:
                if constraint not in self.space.repair_budgets:
                    raise ValueError(f"unsupported repair budget: {constraint}")
                if action.repair_budget != constraint:
                    raise ValueError("action violates repair budget constraint")
                repair_mask = torch.zeros_like(repair_logits, dtype=torch.bool)
                repair_mask[self.space.repair_budgets.index(constraint)] = True
                repair_logits = repair_logits.masked_fill(~repair_mask, float("-inf"))
            specifications = (
                (self.topology_head(hidden[index]), action.topology_index),
                (self.agent_head(hidden[index]), action.agent_count - 1),
                (self.round_head(hidden[index]), self.space.rounds.index(action.rounds)),
                (repair_logits, self.space.repair_budgets.index(action.repair_budget)),
                (self.token_head(hidden[index]), self.space.token_budgets.index(action.token_budget)),
            )
            log_probability = hidden.new_zeros(())
            entropy = hidden.new_zeros(())
            for logits, selected in specifications:
                distribution = torch.distributions.Categorical(logits=logits)
                value = torch.tensor(selected, device=hidden.device)
                log_probability = log_probability + distribution.log_prob(value)
                entropy = entropy + distribution.entropy()
            available = torch.ones(self.space.max_roles, dtype=torch.bool, device=hidden.device)
            for selected in action.role_indices:
                distribution = torch.distributions.Categorical(logits=self.role_head(hidden[index]).masked_fill(~available, float("-inf")))
                value = torch.tensor(selected, device=hidden.device)
                log_probability = log_probability + distribution.log_prob(value)
                entropy = entropy + distribution.entropy()
                available[selected] = False
            log_probabilities.append(log_probability)
            entropies.append(entropy)
        return torch.stack(log_probabilities), torch.stack(entropies)


def group_relative_advantages(rewards: torch.Tensor, group_ids: Sequence[str], eligible: torch.Tensor | None = None, epsilon: float = 1e-6) -> torch.Tensor:
    if rewards.ndim != 1 or len(group_ids) != rewards.numel():
        raise ValueError("rewards and group_ids must describe the same flat batch")
    mask = torch.ones_like(rewards, dtype=torch.bool) if eligible is None else eligible.bool()
    advantages = torch.zeros_like(rewards)
    for group in dict.fromkeys(group_ids):
        indices = torch.tensor([i for i, value in enumerate(group_ids) if value == group], device=rewards.device)
        group_mask = mask[indices]
        valid = indices[group_mask]
        if valid.numel() < 2:
            continue
        values = rewards[valid]
        std = values.std(unbiased=False)
        advantages[valid] = (values - values.mean()) / (std + epsilon)
    return advantages


def clipped_policy_loss(new_log_probabilities: torch.Tensor, old_log_probabilities: torch.Tensor, advantages: torch.Tensor, *, eligible: torch.Tensor | None = None, entropy: torch.Tensor | None = None, clip: float = 0.2, entropy_weight: float = 0.01, reference_log_probabilities: torch.Tensor | None = None, kl_weight: float = 0.02) -> torch.Tensor:
    mask = torch.ones_like(advantages, dtype=torch.bool) if eligible is None else eligible.bool()
    if not mask.any():
        return new_log_probabilities.sum() * 0.0
    ratio = torch.exp(new_log_probabilities[mask] - old_log_probabilities[mask])
    objective = torch.minimum(ratio * advantages[mask], ratio.clamp(1.0 - clip, 1.0 + clip) * advantages[mask])
    loss = -objective.mean()
    if entropy is not None:
        loss = loss - entropy_weight * entropy[mask].mean()
    if reference_log_probabilities is not None:
        approximate_kl = (new_log_probabilities[mask] - reference_log_probabilities[mask]).pow(2).mean() * 0.5
        loss = loss + kl_weight * approximate_kl
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite route policy loss")
    return loss
