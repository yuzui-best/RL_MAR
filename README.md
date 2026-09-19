# RL-MAR: Verifier-Guided Multi-Agent Routing for PLC Program Generation

[中文说明](README_zh.md)

This repository contains the public implementation and experiment artifacts for
**"Harnessing Large Language Models for PLC Program Generation via
Verifier-Guided Multi-Agent Routing."** RL-MAR generates IEC 61131-3 Structured
Text (ST) programs from natural-language specifications using domain-conditioned
agent routing, resource-aware execution, and bounded verifier-guided correction.

![RL-MAR framework](PLC_paper_picture/PLC%20Agent%20Framework.png)

The evaluation covers six LLM backbones on **Functional94**, formal-property
verification on **Agents4PLC-23**, routing and verifier-feedback ablations, and a
Siemens S7-1200 stereoscopic warehouse case study.

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
|   |-- results/                     # Ablations and failure analysis
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
Agents4PLC-23 formal verification requires RuSTy, PLCverif, and nuXmv. The
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

Remove `--dry-run` after the required compiler/evaluator tools are configured.
The same launcher supports `direct_generation`, `autoplc`, `llm4plc`, and
`agents4plc23`. Saved paper results and task-level records are provided under
`experiments/comparison/` and `experiments/results/`.

## Reported Results

On Functional94, RL-MAR matches or exceeds AutoPLC on most of the six evaluated
backbones while using 16.5-46.4% fewer recorded generation tokens under the
original workflow settings. The matched routing ablation reduces recorded
tokens by 29.7% and execution time by 22.5% with comparable functional results.
On Agents4PLC-23, RL-MAR verifies 69.6% of tasks with GPT-4o and 65.2% with
GPT-4o-mini.

## Acknowledgments

The comparison and verification workflows build on
[AutoPLC](https://github.com/cangkui/AutoPLC),
[LLM4PLC](https://github.com/AICPS/LLM_4_PLC),
[Agents4PLC](https://github.com/Luoji-zju/Agents4PLC_release), and
[CODESYSCompileService](https://github.com/cangkui/CODESYSCompileService).
See the repository and subdirectory license notices for applicable terms.
