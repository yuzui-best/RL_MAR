# Agents4PLC-23: official verification alignment

The release includes 46 retained generated programs (GPT-4o and GPT-4o-mini),
and the unchanged 23 tasks / 101 properties per backbone.
Verification runs both original source and the separately identified compound
statement semicolon adapter. No new model generation is needed for rechecking.

The official implementation in `official_source/` uses `Satisfied` and `Violated`
as definite verdicts. Verified follows its exact threshold arithmetic;
validation satisfied requires `satisfied > 0.80 * total`. The official summary
uses compiled programs as the denominator for those two rates. The full 23-task
denominator, unresolved classifications and all-satisfied counts remain in the
per-task data. Compilation is evaluated independently.

Install RuSTy, PLCverif, nuXmv and CBMC (the recorded run used CBMC 6.11.0), then:

```bash
python evaluation/agents4plc/scripts/run_alignment.py \
  --out runs/formal_recheck \
  --rusty-bin /path/to/plc \
  --plcverif-bin /path/to/plcverif-cli \
  --nuxmv-bin /path/to/nuXmv \
  --cbmc-bin /path/to/cbmc
python evaluation/agents4plc/scripts/summarize_agents4plc23.py \
  --results runs/formal_recheck/RESULTS.json --output-dir runs/formal_recheck/summary
```

The official NuSMV-to-CBMC fallback is retained. `lxml` is required to parse
counterexamples. Tool output is saved per property. The outer operational limit
is 1,200 seconds; each compiler check retains its 60-second limit.

Latest recorded counts (compiled / verified / validation satisfied):

| Source condition | GPT-4o | GPT-4o-mini |
|---|---|---|
| Original | 17 / 13 / 12 | 15 / 15 / 10 |
| Semicolon compatibility | 17 / 16 / 13 | 15 / 15 / 10 |

Each model has 23 tasks. These counts and both denominators are exported in
`experiments/results/formal_alignment/`. 
The public formal entry point rechecks the retained final programs. Generation
trajectories and tool logs remain in the separate research archive.
