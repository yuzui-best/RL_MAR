"""GPU REINFORCE over newly measured development action rewards, without authored targets."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "framework"))
from MAR.Routing.core import ACTIONS, MODELS, domain, features, reward


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def empirical_tables(out, ids, stage, coefficient):
    """All feasible actions must be observed; unobserved actions are never fabricated failures."""
    import numpy as np
    feasible = read(out / "feasible_actions.json")
    jobs = [j for j in read(out / "development_rewards.json") if j["stage"] == stage and j["task_id"] in ids]
    cells = {}
    for job in jobs:
        row = job
        if row["status"] != "COMPLETE" or row["unknown_upper"]:
            raise ValueError("training requires complete, measured development executions")
        if row["known_tokens"] > 16384:
            raise ValueError("over-budget development execution")
        cells.setdefault((job["task_id"], job["action_index"]), []).append(row)
    values = np.zeros((len(ids), len(ACTIONS)), dtype="float32")
    allowed = np.zeros_like(values, dtype=bool)
    pass_table = np.zeros_like(values)
    tokens = np.zeros_like(values)
    for i, tid in enumerate(ids):
        observed = [a for (task_id, a) in cells if task_id == tid]
        expected = feasible[tid] if stage == "action_matrix" else observed
        if not expected or set(observed) != set(expected):
            raise ValueError(f"incomplete action reward matrix: {tid}")
        for a in expected:
            rows = cells[tid, a]
            allowed[i, a] = True
            values[i, a] = sum(reward(r["functional_pass"], r["known_tokens"], r["calls"], coefficient) for r in rows) / len(rows)
            pass_table[i, a] = sum(r["functional_pass"] for r in rows) / len(rows)
            tokens[i, a] = sum(r["known_tokens"] for r in rows) / len(rows)
    return values, allowed, pass_table, tokens


def fit(out, seed, tuning_round, initialize_only=False, shuffle_rewards=False):
    if (out / "DIAGNOSTIC_ONLY.json").exists():
        raise RuntimeError("Mechanism corpus is diagnostic only; do not use it for policy training or selection")
    import numpy as np
    import torch
    from torch import nn

    config = read(out / "FROZEN_CONFIG.json")
    hp = config["algorithm"]
    if seed not in hp["seeds"] or tuning_round not in (1, 2):
        raise ValueError("seed or tuning round is outside frozen configurations")
    coefficient = config["tuning_rounds"][tuning_round-1]["cost_lambda"]
    split = read(out / "data/split.json")
    tasks = {t["task_id"]: t for t in read(out / "data/private_tasks.json")}
    train_ids, val_ids = split["train"], split["validation"]
    init_ids = split["initialization"]
    init, init_mask, _, _ = empirical_tables(out, init_ids, "initialization", coefficient)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requested; CUDA unavailable")
    vectors = lambda ids: np.array([features(tasks[tid], "deepseek-v4-flash") for tid in ids], dtype="float32")
    raw = vectors(train_ids)
    mean, scale = raw.mean(0), raw.std(0)
    inactive=scale<1e-6
    scale[scale < 1e-6] = 1
    tensor = lambda a, dtype=None: torch.as_tensor(a, device=device, dtype=dtype)
    x, vx, ix = [tensor((vectors(ids)-mean)/scale) for ids in (train_ids, val_ids, init_ids)]
    policy = nn.Sequential(nn.Linear(raw.shape[1], hp["hidden_dim"]), nn.Tanh(),
                           nn.Linear(hp["hidden_dim"], len(ACTIONS))).to(device)
    # Features constant throughout training have no learned effect on unseen public inputs.
    with torch.no_grad():
        policy[0].weight[:,tensor(inactive)] = 0
    optimizer = torch.optim.Adam(policy.parameters(), lr=hp["learning_rate"])
    moving = deque(maxlen=hp["baseline_window"])
    init_allowed = tensor(init_mask)
    logs = []

    def update(inputs, rewards, permitted):
        dist = torch.distributions.Categorical(logits=policy(inputs).masked_fill(~permitted, -torch.inf))
        action = dist.sample((4,)).T
        observed = rewards.gather(1, action)
        baseline = sum(moving)/len(moving) if moving else 0.0
        advantage = observed - baseline
        loss = -(dist.log_prob(action.T).T * advantage.detach()).mean() - hp["entropy_weight"]*dist.entropy().mean()
        optimizer.zero_grad()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite policy gradient; preserve checkpoint and diagnose")
        optimizer.step()
        moving.append(float(observed.mean()))
        return dict(loss=float(loss.detach()), reward=float(observed.mean()), baseline=baseline,
                    entropy=float(dist.entropy().mean().detach()))

    init_directory = out / "train" / "initialization" / f"round{tuning_round}_seed{seed}"
    init_checkpoint = init_directory / "policy.pt"
    if init_checkpoint.exists():
        saved = torch.load(init_checkpoint, map_location=device, weights_only=True)
        if saved["normalization_mean"] != mean.tolist() or saved["normalization_scale"] != scale.tolist():
            raise ValueError("public feature normalization differs from saved initialization")
        policy.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        torch.set_rng_state(saved["cpu_rng"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device=device)
        logs = read(init_directory / "training_log.json")
    else:
        init_directory.mkdir(parents=True, exist_ok=False)
        for epoch in range(hp["initialization_epochs"]):
            row = update(ix, tensor(init), init_allowed)
            logs.append(dict(stage="initialization", epoch=epoch, **row))
        torch.save(dict(state_dict=policy.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            normalization_mean=mean.tolist(), normalization_scale=scale.tolist(),
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device), seed=seed,
            reward_lambda=coefficient, initialization_epochs=hp["initialization_epochs"]), init_checkpoint)
        put(init_directory / "training_log.json", logs)
        put(init_directory / "REPORT.json", dict(device=str(device), seed=seed, tuning_round=tuning_round,
            epochs=hp["initialization_epochs"], source_tasks=init_ids,
            measured_executions=4, new_inference_requests=0, authored_targets_used=False,
            objective="REINFORCE on measured execution rewards; moving baseline", final_update=logs[-1]))
    if initialize_only:
        print(json.dumps(dict(status="REWARD_INITIALIZATION_COMPLETE", seed=seed, epochs=hp["initialization_epochs"],
                              checkpoint=str(init_checkpoint), new_inference_requests=0)))
        return
    train, train_mask, _, _ = empirical_tables(out, train_ids, "action_matrix", coefficient)
    real_train=train.copy()
    val, val_mask, val_pass, val_tokens = empirical_tables(out, val_ids, "action_matrix", coefficient)
    training_rewards, validation_rewards = tensor(train), tensor(val)
    train_allowed, val_allowed = tensor(train_mask), tensor(val_mask)
    directory = out / "train" / (f"round{tuning_round}_seed{seed}"+("_shuffled" if shuffle_rewards else ""))
    directory.mkdir(parents=True, exist_ok=False)
    permutations=[]
    if shuffle_rewards:
        randomizer=np.random.default_rng(91818)
        for i,tid in enumerate(train_ids):
            available=np.flatnonzero(train_mask[i])
            permuted=randomizer.permutation(available)
            train[i,available]=real_train[i,permuted]
            permutations.append(dict(task_id=tid,action_indices=available.tolist(),permuted_source_actions=permuted.tolist()))
        put(directory/"reward_permutation.json",dict(seed=91818,scope="Training rewards only; within-task feasible actions; validation untouched",rows=permutations))
        training_rewards=tensor(train)
    real_rewards=tensor(real_train)
    def training_metrics():
        with torch.no_grad():
            distribution=torch.distributions.Categorical(logits=policy(x).masked_fill(~train_allowed,-torch.inf))
            choices=distribution.logits.argmax(1)
            return dict(expected_optimizer_reward=float((distribution.probs*training_rewards).sum(1).mean()),
                expected_real_reward=float((distribution.probs*real_rewards).sum(1).mean()),
                greedy_real_reward=float(real_rewards.gather(1,choices[:,None]).mean()),
                greedy_actions=choices.cpu().tolist())
    initial_metrics=training_metrics()
    put(directory/"initial_development_metrics.json",initial_metrics)
    moving.clear()
    best, last_improvement = -float("inf"), 0
    for epoch in range(hp["epochs"]):
        policy.train()
        row = update(x, training_rewards, train_allowed)
        row.update(stage="development", epoch=epoch)
        if epoch % hp["evaluation_interval"] == 0 or epoch+1 == hp["epochs"]:
            policy.eval()
            with torch.no_grad():
                distribution = torch.distributions.Categorical(logits=policy(vx).masked_fill(~val_allowed, -torch.inf))
                choices = distribution.logits.argmax(1)
                selected_rewards = validation_rewards.gather(1, choices[:, None]).squeeze(1)
                objective = float(selected_rewards.mean())
            selected = choices.cpu().tolist()
            rows = [dict(task_id=tid, family_id=tasks[tid]["family_id"], domain=domain(tasks[tid]),
                action_index=a, action=ACTIONS[a], functional_pass=float(val_pass[i, a]), tokens=float(val_tokens[i, a]))
                for i, (tid, a) in enumerate(zip(val_ids, selected))]
            row.update(validation_reward=objective, validation_passes=sum(r["functional_pass"] for r in rows),
                validation_mean_tokens=sum(r["tokens"] for r in rows)/len(rows))
            current_metrics=training_metrics()
            row.update(training_expected_optimizer_reward=current_metrics["expected_optimizer_reward"],
                training_expected_real_reward=current_metrics["expected_real_reward"],
                training_greedy_real_reward=current_metrics["greedy_real_reward"],
                training_mapping_changes_from_initial=sum(a!=b for a,b in zip(current_metrics["greedy_actions"],initial_metrics["greedy_actions"])))
            if objective > best+1e-8:
                best, last_improvement = objective, epoch
                checkpoint = dict(state_dict=policy.state_dict(), normalization_mean=mean.tolist(),
                    normalization_scale=scale.tolist(), features=len(mean), hidden_dim=hp["hidden_dim"],
                    actions=list(ACTIONS), seed=seed, round=tuning_round, epoch=epoch,
                    reward_lambda=coefficient, constant_training_features="zero weights, including unseen backbone bits; no fabricated learned effect",
                    training_task_ids=train_ids, validation_task_ids=val_ids)
                torch.save(checkpoint, directory / "policy.pt")
                put(directory / "best_validation.json", dict(reward=objective, epoch=epoch, predictions=rows,
                    entropy_mean=float(distribution.entropy().mean()), seed=seed, round=tuning_round,
                    shuffled_training_rewards=shuffle_rewards,training_metrics=current_metrics))
        logs.append(row)
        put(directory / "training_log.json", logs)
        if epoch-last_improvement >= hp["early_stopping_patience"]:
            break
    best_row = read(directory / "best_validation.json")
    put(directory / "REPORT.json", dict(algorithm="REINFORCE with moving baseline", device=str(device),
        measured_training_tasks=len(train_ids), measured_validation_tasks=len(val_ids),
        empirical_environment=True, authored_targets_used=False, benchmark_labels_used=False,
        epochs=len([r for r in logs if r["stage"] == "development"]),
        best_validation_reward=best, best_validation_epoch=best_row["epoch"],
        shuffled_training_rewards=shuffle_rewards,initial_development_metrics=initial_metrics,
        best_training_metrics=best_row["training_metrics"],
        identity_counts=dict(Counter(r["action"]["identity"] for r in best_row["predictions"]))))
    print(json.dumps(dict(seed=seed, round=tuning_round, best_validation_reward=best,
        passes=sum(r["functional_pass"] for r in best_row["predictions"]), tasks=len(val_ids))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--shuffle-rewards",action="store_true")
    args = parser.parse_args()
    if (args.out/"CONTROLLER_RELOAD_REQUIRED.json").exists() and not args.initialize_only:
        raise RuntimeError("Data queue may finish normally; reload its supervisor before main training so it uses the amended screening rules")
    fit(args.out, args.seed, args.round, args.initialize_only,args.shuffle_rewards)
