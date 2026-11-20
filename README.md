# RL-MAR: Verifier-Guided Multi-Agent Routing for PLC Program Generation

[中文说明](README_zh.md)

This repository contains the public implementation and experiment artifacts for
**"Harnessing Large Language Models for PLC Program Generation via
Verifier-Guided Multi-Agent Routing."** RL-MAR generates IEC 61131-3 Structured
Text (ST) programs from natural-language specifications using domain-conditioned
agent routing, resource-aware execution, and bounded verifier-guided correction.

![RL-MAR framework](PLC_paper_picture/PLC%20Agent%20Framework.png)

The evaluation covers six LLM backbones on **Functional94**, formal-property
verification on **Agents4PLC-23**, routing and verifier-feedback ablations, and an
archived Siemens S7-1200 stereoscopic warehouse case study.

## Repository Structure

```text
RL_MAR/
|-- framework/
|   |-- MAR/                         # RL-MAR runtime, domain roles, and PLC tools
|   `-- CODESYSCompileService/       # Optional CODESYS compilation service
|-- evaluation/
|   |-- functional94/                # Frozen 94-task functional evaluator
|   `-- agents4plc/                  # 23-task formal-property evaluator
|-- experiments/
|   |-- comparison/                  # Baseline/RL-MAR runners and reported results
|   |-- training/                    # Measured rewards, GPU training replay
|   |-- results/                     # Main table, three budgets, ablations, terminal measurements
|   `-- case_study/                  # Generated ST, measurements, and TIA project
`-- PLC_paper_picture/               # Framework figure
```

## Installation

Python 3.10+ is recommended. Install the core dependencies and create a private
environment file:

```bash
pip install -r framework/requirements.txt
cp framework/template.env framework/.env
```

Fill `framework/.env` with the endpoints, model names, and API keys for the
providers you intend to use. Do not commit this file.

Functional94 execution additionally requires Docker, a C compiler, and RuSTy.
Agents4PLC-23 formal verification requires RuSTy, PLCverif, nuXmv, and CBMC. The
CODESYS service is optional and requires CODESYS V3.5 SP20 with ScriptEngine on
Windows:

```bash
pip install -r framework/CODESYSCompileService/requirements.txt
cd framework/CODESYSCompileService
python debug_server.py
```

## Running the Experiments

Validate the frozen Functional94 manifest:

```bash
python evaluation/functional94/scripts/validate_manifest.py \
  evaluation/functional94/data/plc94.functional.v1.3.jsonl
```

Inspect an RL-MAR comparison command before execution:

```bash
python experiments/comparison/run_comparison.py \
  --method rlmar \
  --model gpt-4o \
  --dataset evaluation/functional94/data/plc94.functional.v1.3.jsonl \
  --output-root runs \
  --env-file framework/.env \
  --dry-run
```

Set `RUSTY_BIN` to the RuSTy binary, then remove `--dry-run` to generate programs.
Use Linux for native ABI verification. A containerized compiler can be selected
with `RUSTY_DOCKER_IMAGE=ubuntu:24.04` and `RUSTY_MOUNT_ROOT` covering the compiler
and output directories. For DeepSeek/Qwen, set `<PREFIX>_TOKENIZER` to the matching
`tokenizer.json`; GPT models use `o200k_base`. The runner uses Chat Completions;
GPT-5.5 providers that require Responses need a compatible loopback endpoint.
The same launcher supports `direct_generation`, `autoplc`, `llm4plc`, and
`agents4plc23` (official rechecking of the retained programs; see the
[formal guide](evaluation/agents4plc/README.md)). Saved paper results and task-level records are provided under
`experiments/comparison/` and `experiments/results/`.

## Reported Results

The frozen policy is seed **1818**, round **1**: a 20–64–14 tanh MLP trained
with REINFORCE on measured development rewards. It selects a native identity
and chain depth from public task features. The six backbones produce identical
routes on Functional94 with this checkpoint.

At 16,384 tokens, RL-MAR solves **189/564** task-runs, compared with AutoPLC
**167/564**, Direct Generation **147/564**, and LLM4PLC **87/564**, using
**31.5–43.6%** fewer tokens than AutoPLC across the six backbones. The three-budget
comparison (4,096 / 8,192 / 16,384) contains all **72 conditions / 6,768 records**.

In the 94×3 identity/routing study, the learned policy achieves **98/282** and
the strongest fixed identity (analyzer) **96/282**; their paired interval crosses
zero, and the analyzer uses **23.2%** fewer tokens. Against the original domain
rule (**81/282**), the policy gains **6.03 pp**, with a task-cluster 95% interval
of **[2.13, 10.99]**.

The latest Agents4PLC recheck follows the official implementation: verified means
that the tool gives a definite verdict for the required fraction of properties;
validation satisfied requires **>80%** satisfied. With the separately reported
semicolon compatibility adapter, GPT-4o / mini compile **17/23 / 15/23**, verify
**16/17 / 15/15**, and satisfy validation for **13/17 / 10/15** compiled programs.
Original-source results and full 23-task denominators are also included.

See [the artifact index](experiments/results/README.md) for each table and its
per-task terminal measurements. Check recorded results without model calls:

```bash
python -m unittest discover -s experiments/tests -v
```

For GPU training replay, see [the training guide](experiments/training/README.md).

The archived warehouse case predates the frozen seed1818 policy.

## Acknowledgments

The comparison and verification workflows build on
[AutoPLC](https://github.com/cangkui/AutoPLC),
[LLM4PLC](https://github.com/AICPS/LLM_4_PLC),
[Agents4PLC](https://github.com/Luoji-zju/Agents4PLC_release), and
[CODESYSCompileService](https://github.com/cangkui/CODESYSCompileService).
See the repository and subdirectory license notices for applicable terms.
