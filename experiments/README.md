# Experiments

| Entry point | Purpose | Required resources |
| --- | --- | --- |
| `runners/build_rewardplc.py` | Construct RewardPLC | Explicit output path |
| `runners/validate_rewardplc.py` | Compile corpus references | RuSTy, Docker, mount root |
| `runners/train_domain_classifier.py` | Train the domain classifier | Training corpus and local encoder |
| `runners/collect_rewardplc_rollouts.py` | Collect trajectories | Models, policy and verifier |
| `runners/train_route_policy.py` | Train the route policy | Train-only rollouts, matching behavior policy |
| `runners/revalidate_rollouts.py` | Recheck saved candidates | Corpus, rollouts and RuSTy |
| `runners/evaluate_five_models.py` | Frozen Functional94 evaluation | Contract JSONL, exact policy, settings, providers, external evaluator |
| `runners/evaluate_agents4plc23.py` | Agents4PLC evaluation | External benchmark and toolchain |

`runners/route_execution_adapter.py` executes route actions for the five-model
runner. `profiles/` holds workflow configurations. `paper/` contains analysis
scripts for authorized saved-result archives. `baselines/` contains sanitized
snapshots of Direct Generation, AutoPLC and LLM4PLC, plus provenance notices.

The old `run_plc.py` command is absent in this snapshot. Review
`docs/REPRODUCIBILITY.md` before attempting training or evaluation.
`evaluate_five_models.py --preflight` performs a real provider request after
checking its inputs; it is not an offline validation command.
