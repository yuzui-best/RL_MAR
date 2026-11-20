"""Dispatch the public comparison methods without hiding method-specific knobs.

This is intentionally an orchestration layer, not a second implementation of
the baselines.  Use --dry-run first; commands are printed before execution.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", choices=["direct_generation", "autoplc", "llm4plc", "rlmar", "agents4plc23"], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--autoplc-root", type=Path)
    parser.add_argument("--llm4plc-script", type=Path)
    parser.add_argument("--rlmar-runner", type=Path)
    parser.add_argument("--agents4plc23-runner", type=Path)
    parser.add_argument("--agents-evaluator", type=Path)
    parser.add_argument("--rusty-bin", type=Path)
    parser.add_argument("--plcverif-bin", type=Path)
    parser.add_argument("--nuxmv-bin", type=Path)
    parser.add_argument("--tool-log-dir", type=Path)
    parser.add_argument("--cbmc-bin", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = args.model or ["gpt-5.5"]
    commands: list[list[str]] = []
    for method in args.method:
        if method == "direct_generation":
            for model in models:
                command = [sys.executable, str(HERE / method / "run.py"), "--dataset", str(args.dataset), "--model", model, "--output", str(args.output_root / method / model / "candidates.json")]
                if args.env_file:
                    command += ["--env-file", str(args.env_file)]
                commands.append(command)
        elif method == "autoplc":
            if not args.autoplc_root:
                raise SystemExit("--autoplc-root is required for AutoPLC")
            commands.append([sys.executable, str(HERE / method / "run.py"), "--upstream-root", str(args.autoplc_root), "--dataset", str(args.dataset), "--output-dir", str(args.output_root / method)])
        elif method == "llm4plc":
            if not args.llm4plc_script:
                raise SystemExit("--llm4plc-script is required for LLM4PLC")
            commands.append([sys.executable, str(HERE / method / "run.py"), "--upstream-script", str(args.llm4plc_script), "--dataset", str(args.dataset), "--output-dir", str(args.output_root / method)])
        elif method == "rlmar":
            if not args.env_file:
                raise SystemExit("--env-file is required for RL-MAR")
            for model in models:
                command = [sys.executable, str(HERE / method / "run.py")]
                if args.rlmar_runner:
                    command += ["--runner", str(args.rlmar_runner)]
                command += ["--", "--dataset", str(args.dataset), "--model", model, "--output", str(args.output_root / method), "--run-id", f"comparison_{model}", "--policy", str(HERE / method / "route_policy.pt"), "--env-file", str(args.env_file)]
                commands.append(command)
        else:
            required = {"--rusty-bin": args.rusty_bin, "--plcverif-bin": args.plcverif_bin,
                        "--nuxmv-bin": args.nuxmv_bin, "--cbmc-bin": args.cbmc_bin}
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise SystemExit("Agents4PLC rechecking requires " + ", ".join(missing))
            command = [sys.executable, str(HERE / method / "run.py")]
            if args.agents4plc23_runner:
                command += ["--runner", str(args.agents4plc23_runner)]
            command += ["--", "--out", str(args.output_root / method),
                "--rusty-bin", str(args.rusty_bin), "--plcverif-bin", str(args.plcverif_bin),
                "--nuxmv-bin", str(args.nuxmv_bin), "--cbmc-bin", str(args.cbmc_bin)]
            commands.append(command)

    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=HERE, check=True)


if __name__ == "__main__":
    main()
