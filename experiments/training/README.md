# Measured-reward policy training

The deployed seed1818, round1 checkpoint is at `../comparison/rlmar/route_policy.pt`.

The 20-dimensional state contains deterministic public features. The policy is
a 20–64–14 tanh MLP with native-action masking. Training uses REINFORCE, a moving
reward baseline, and entropy regularization. The reward is
`functional_pass - tokens/16384 - 0.02*calls` for the selected round.

`development_rewards.json` contains 454 measured executions: 180 training
(task, action) cells with one execution each, 90 validation cells with three
executions each, and four initialization executions. The split contains 50
training tasks, 25 validation tasks, and a separate initialization task.
`data/split.json` and `data/private_tasks.json` supply the split and development tasks for replay.

Replay requires CUDA and issues no model/API calls:

```bash
python experiments/training/prepare_replay.py --out runs/training_replay
python experiments/training/train.py --out runs/training_replay --seed 1818 --round 1
```

Use a fresh output directory. The published checkpoint is unchanged by replay.
For a shuffled-reward control, prepare another fresh directory and add
`--shuffle-rewards`. Seeds 2411 and 3203 use the same command with a different seed.
The empirical action table, feasible mask and public features are unchanged from
the frozen study; `train.py` replaces server record lookup with the published table.

The original execution logs and server-specific controllers are retained in the
separate research archive. This repository contains the portable training and
inference implementation, frozen configuration and measured reward table.
