# LLM4PLC comparison baseline

Sanitized runnable snapshot of the three-stage planning, generation, and repair
baseline. It includes the runner, validator, cost calculator, and prompts.
Datasets, generated programs, result workspaces, logs, caches, credentials, and
Functional94 test/oracle material are excluded.

## Provenance and republication caveat

- Source: local comparison workspace; machine-specific path removed.
- Snapshot date: 2026-09-04
- Source Git remote/commit: unavailable; the directory is not a Git checkout.
- License: **unclear**. No license file was found. Inclusion in this public
  research snapshot does not establish an open-source license for these files.

This is comparison-baseline material, not evidence of copied implementation in
RL-MAR.

## Run

```bash
python experiments/baselines/llm4plc/run_llm4plc_codessy_reproduce.py --help
```

Supply an authorized dataset, provider settings, and the external CODESYS
validation service at runtime. Execution may incur provider charges.
