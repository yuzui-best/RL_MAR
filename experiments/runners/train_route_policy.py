from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
sys.path[:0] = [str(ROOT), str(FRAMEWORK)]

from MAR.LLM.llm_embedding import SentenceEncoder
from MAR.Core.dataset import DOMAINS, stable_sha256
from MAR.Core.outcomes import classify_outcome
from MAR.Core.policy import ContextualRoutePolicy, RouteAction, clipped_policy_loss, group_relative_advantages
from MAR.Core.reward import RewardNormalizers, compute_behavior_reward


def percentile95(values: list[float]) -> float:
    if not values:
        return 1.0
    ordered = sorted(max(float(value), 0.0) for value in values)
    return max(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1.0)


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalizers_from(rows: list[dict]) -> RewardNormalizers:
    eligible = [row for row in rows if row.get("policy_eligible")]
    return RewardNormalizers(
        percentile95([sum(row.get("costs", {}).get(key, 0) for key in ("prompt_tokens", "completion_tokens")) for row in eligible]),
        percentile95([row.get("elapsed_seconds", 0) for row in eligible]),
        percentile95([max(len(row.get("attempts", [])) - 1, 0) for row in eligible]),
        percentile95([row.get("runtime_budget", {}).get("calls", 1) for row in eligible]),
    )


def action_from(row: dict) -> RouteAction:
    action = row["route_action"]
    return RouteAction(
        action["topology_index"], tuple(action["role_indices"]), action["agent_count"],
        action["rounds"], action["repair_budget"], action["token_budget"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the frozen RL-MAR v2 contextual route policy")
    parser.add_argument("--rollouts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--behavior-policy", type=Path, required=True)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_rows(args.rollouts)
    expected_policy_hash = stable_sha256(args.behavior_policy.read_bytes())
    if any(row.get("schema_version") != "rlmar-trajectory-2.0" for row in rows):
        raise ValueError("unsupported rollout schema")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("route training accepts train split only")
    dataset_hashes = {row.get("dataset_sha256") for row in rows}
    if len(dataset_hashes) != 1 or None in dataset_hashes:
        raise ValueError("rollouts must share one recorded dataset hash")
    if any(row.get("behavior_policy_sha256") != expected_policy_hash for row in rows):
        raise ValueError("rollout behavior policy hash mismatch")
    deduplicated = {}
    for row in rows:
        key = (row["group_id"], row["candidate_index"])
        if key in deduplicated:
            raise ValueError(f"duplicate rollout candidate: {key}")
        deduplicated[key] = row
    rows = list(deduplicated.values())
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    valid_groups = []
    for group in groups.values():
        eligible_actions = {json.dumps(row["route_action"], sort_keys=True) for row in group if row.get("policy_eligible")}
        if len(eligible_actions) >= 2:
            valid_groups.extend(group)
    complete = valid_groups
    if not complete:
        raise ValueError("no route group has at least two candidates")
    normalizers = normalizers_from(complete)
    for row in complete:
        outcome = classify_outcome(row.get("validation"))
        reward = compute_behavior_reward(
            outcome, verifier_result=row.get("validation", {}), normalizers=normalizers,
            total_tokens=sum(row.get("costs", {}).get(key, 0) for key in ("prompt_tokens", "completion_tokens")),
            latency_seconds=row.get("elapsed_seconds", 0), retries=max(len(row.get("attempts", [])) - 1, 0),
            agent_calls=row.get("runtime_budget", {}).get("calls", 1),
        )
        row["training_reward"] = reward.shaped_reward
        row["policy_eligible"] = reward.policy_eligible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SentenceEncoder(device=device)
    embeddings = encoder([row["specification"] for row in complete]).detach()
    domains = torch.zeros(len(complete), len(DOMAINS), device=device)
    for index, row in enumerate(complete):
        domains[index, DOMAINS.index(row["domain"])] = 1.0
    model_ids = torch.tensor([row["model_id"] for row in complete], device=device)
    rewards = torch.tensor([row["training_reward"] or 0.0 for row in complete], device=device)
    eligible = torch.tensor([row["policy_eligible"] for row in complete], device=device)
    group_ids = [row["group_id"] for row in complete]
    advantages = group_relative_advantages(rewards, group_ids, eligible)
    actions = [action_from(row) for row in complete]
    repair_constraints = [
        (row.get("route_constraints", {}).get("repair_budget") or [None])[0]
        for row in complete
    ]
    old = torch.tensor([row["old_log_probability"] for row in complete], device=device)
    reference = torch.tensor([row["reference_log_probability"] for row in complete], device=device)
    policy = ContextualRoutePolicy().to(device)
    behavior_checkpoint = torch.load(args.behavior_policy, map_location=device, weights_only=True)
    policy.load_state_dict(behavior_checkpoint["state_dict"] if "state_dict" in behavior_checkpoint else behavior_checkpoint)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    losses = []
    for _ in range(args.epochs):
        optimizer.zero_grad()
        log_probabilities, entropy = policy.log_probability(
            embeddings, domains, model_ids, actions, repair_budgets=repair_constraints
        )
        loss = clipped_policy_loss(log_probabilities, old, advantages, eligible=eligible, entropy=entropy, reference_log_probabilities=reference)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dict": policy.state_dict(), "normalizers": normalizers.__dict__, "seed": args.seed, "behavior_policy_sha256": expected_policy_hash}
    torch.save(checkpoint, args.output)
    report = {
        "schema_version": "rlmar-route-training-2.0", "rows": len(complete), "groups": len({row["group_id"] for row in complete}),
        "eligible_rows": int(eligible.sum()), "epochs": args.epochs, "losses": losses,
        "normalizers": normalizers.__dict__, "rollout_sha256": [stable_sha256(path.read_text(encoding="utf-8")) for path in args.rollouts],
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
