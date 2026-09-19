"""Invoke the authors' LLM4PLC reproduction script on Functional94.

The upstream implementation is intentionally not duplicated in this release;
pass its reproduction entry point explicitly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-script", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-verification", type=int, default=3)
    parser.add_argument("--single-task")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.upstream_script.is_file():
        raise SystemExit(f"LLM4PLC script not found: {args.upstream_script}")
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        str(args.upstream_script.resolve()),
        "--data_path",
        str(args.dataset.resolve()),
        "--result_dir",
        str(args.output_dir.resolve()),
        "--max_attempts",
        str(args.max_attempts),
        "--max_verification",
        str(args.max_verification),
    ]
    if args.single_task:
        command += ["--single_task", args.single_task]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=args.upstream_script.resolve().parent, check=True)


if __name__ == "__main__":
    main()
