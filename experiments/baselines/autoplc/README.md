# AutoPLC comparison baseline

Sanitized runnable snapshot of the AutoPLC ST/SCL pipeline. It includes Python
sources, generated ANTLR parser sources, two generic example configs, upstream
README, requirements, and license. It excludes datasets, Functional94 configs,
experiment outputs, logs, archives, credentials, and the vendored ANTLR JAR.

## Provenance and license

- Source: local adaptation of the linked AutoPLC project; machine-specific path removed.
- Snapshot date: 2026-09-04
- Source Git remote/commit: unavailable in the local source directory.
- License: MIT; see `LICENSE.upstream`.

## Run

```bash
python -m pip install -r experiments/baselines/autoplc/requirements.txt
PYTHONPATH=experiments/baselines/autoplc/autoplc \
  python experiments/baselines/autoplc/autoplc/main_st.py --help
```

Copy and edit a generic config outside the repository. Supply an authorized
dataset and provider credentials at runtime. The snapshot's external CODESYS
service and ANTLR runtime remain environment prerequisites.
