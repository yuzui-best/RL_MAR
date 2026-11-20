# Published experiment results

| Experiment | Published material | Coverage |
|---|---|---|
| Main comparison | [Source data](main_comparison/source_data.csv) | 2,256 runs across six models and four methods |
| Matched budgets | [Source data](matched_budget/source_data.csv) | 6,768 runs across three budgets, including the main comparison at 16,384 tokens |
| Routing and identity | [Source data](ablation/six_arm/source_data.csv) | Six arms, each with 94 tasks and three repetitions |
| Verifier feedback | [Retention rules](ablation/D_feedback/RETENTION_RULES.json) | Candidate selection rules for comparisons within each budget |
| Formal verification | [Source data](formal_alignment/source_data.csv) | 92 checks covering 23 tasks, two models and two source conditions |

The main comparison CSV replaces the former experiments/comparison/task_results.csv. Call time sums model API durations and includes recorded service time for cached calls. Compilation and terminal evaluation are timed separately.

The [Holm results](main_comparison/HOLM_VERIFICATION.json) report six paired RL-MAR versus AutoPLC comparisons. The [independent replication](replication/REPLICATION_RESULT.json) records 32 passes on 94 tasks and the stopping threshold. The [formal evaluation guide](../../evaluation/agents4plc/README.md) explains the denominators for compilation, definite verdicts and validation.
