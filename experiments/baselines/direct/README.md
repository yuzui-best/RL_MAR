# Direct Generation baseline

Sanitized snapshot of the four single-pass runners used for the Direct
Generation comparison. The snapshot contains runner code only; datasets,
outputs, logs, credentials, and Functional94 test/oracle material are excluded.

## Provenance and license

- Snapshot date: 2026-09-04
- Upstream remote/commit: unavailable because the source tree is not a Git
  checkout.
- License: no license file was found at the source. Redistribution rights
  therefore remain unclear; inclusion in this public research snapshot does
  not establish an open-source license for these files.

No claim is made that this code originates from any design-reference project.

## Run

Choose the runner matching the provider, install the parent framework
dependencies, and pass an authorized public dataset:

```bash
export PYTHONPATH="$PWD:$PWD/framework${PYTHONPATH:+:$PYTHONPATH}"
python experiments/baselines/direct/run_gpt4o.py \
  --dataset_path /path/to/public/tasks.jsonl \
  --api_key "\$GPT4O_API_KEY"
```

Use `--help` for provider-specific flags. Running invokes paid APIs and is
never performed by repository materialization or tests.
