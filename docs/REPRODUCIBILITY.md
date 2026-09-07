# Reproduction scope

## Available without model credentials or external evaluators

- Inspect the public 94-task JSON-array dataset under `framework/Datasets/`.
- Read routing, budgeting, outcome and reward implementations.
- Regenerate figures from `results/paper_plot_report.json` using numpy and
  matplotlib. This reuses saved results and does not establish a fresh pass rate.

## Full training and evaluation

Training requires a compatible corpus, local encoder, rollout records and
matching behavior-policy checkpoint. Frozen evaluation requires the expected
policy checkpoint, run settings, model bindings, contract-bearing inputs and
external verifier. Private saved-result archives, test oracles and credentials
are not distributed here.

`evaluate_five_models.py` reads JSONL records with fields including `cases`,
`oracles` and `stimulus`; the public JSON-array dataset is not that contract.
The runner enforces a checkpoint hash. Do not remove that check to make an
unrelated checkpoint appear reproducible. Its `--preflight` also performs a
provider request.

ABI alignment and replay tools under `experiments/paper/` use an explicitly
supplied external evaluator. They do not silently replace the evaluator or
regenerate missing archived candidates.

## Platform and startup limitations

- The five-model runner imports `resource`. The RewardPLC verifier uses Linux
  shared libraries, a C compiler, Docker and `os.getuid/getgid`. Use a
  compatible Linux environment for these paths.
- The CODESYS service requires Windows and CODESYS ScriptEngine. Check its
  configured bind address and executable path independently.
- Several training/data runners derive `ROOT` from `parents[1]`, pointing to
  `experiments/` in this layout. Add the repository root and `framework/` to
  `PYTHONPATH`, and supply explicit paths instead of relying on derived
  defaults. This workaround does not establish that every runner works.
- `psutil` is imported by resource monitoring but not explicitly listed in
  framework requirements. Install it separately when needed. ANTLR runtime
  and plot dependencies also need the installations in the root README.
- Sentence-encoder loading requires a local directory. The public selection
  omits weights; see `framework/models/PROVENANCE.md`.
- Dependencies are not locked for all workflows. A minimum Python version
  does not guarantee compatibility of current packages with every backend.

Report syntax checks, offline checks and plot generation separately from
model generation, CODESYS compilation and functional evaluation.
