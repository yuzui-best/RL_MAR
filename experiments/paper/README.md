# Manuscript Analysis and ABI Alignment

The aggregate files in `results/paper_*.json` match the current manuscript.
`paper_plot_report.json` contains only the current plotted results, excluding
historical comparison statistics. Original generation costs are unchanged.

Install `requirements-statistics.txt` for private paired-statistics reproduction.
The analysis scripts require externally supplied, authorized saved-result archives
and training corpora. They are not included in this package.

`aligned_abi.py` is an isolated adapter for the archived RuSTy evaluator.
`validate_aligned_abi.py --root AUTHORIZED_EVALUATOR_ROOT --out OUTPUT.json`
compiles four synthetic ABI fixtures using that evaluator's toolchain.
`replay_aligned_abi.py --saved SAVED_RUN --out NEW_RUN` replays compiled candidates
without changing ST; `--reference` processes the reference group separately.
`aligned_worker.py` supports newly supplied candidates under the same adapter.
The archived evaluator and compiler are external prerequisites, not vendored here.

Render the quantitative figures with Python and matplotlib:

```bash
python docs/figures/generate_experiment_figures.py --report results/paper_plot_report.json --outdir figure_output
```

The framework and execution adapter retain their existing paths. The new ABI
adapter is explicit and does not silently replace the archived evaluator.
Package validation checks distribution contents; full experiment reproduction
also requires the external benchmark, model credentials and checkpoint artifacts.
