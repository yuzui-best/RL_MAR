"""Run the frozen Functional94 evaluator on generated ST packages.

``input-index`` follows the evaluator's published package-index schema. The
evaluation artifacts are written to ``EVALUATION_ROOT/runs/RUN_ID``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2] / "evaluation" / "functional94"
    parser.add_argument("--evaluation-root", type=Path, default=default_root)
    parser.add_argument("--manifest", default="data/plc94.functional.v1.3.jsonl")
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--compile-timeout", type=int, default=120)
    parser.add_argument("--task-timeout", type=int, default=180)
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    root = args.evaluation_root.resolve()
    evaluator = root / "scripts" / "run_functional94.py"
    if not evaluator.is_file():
        raise SystemExit(f"Frozen evaluator not found: {evaluator}")
    if not args.input_index.is_file():
        raise SystemExit(f"Input index not found: {args.input_index}")
    try:
        index_argument = str(args.input_index.resolve().relative_to(root))
    except ValueError as exc:
        raise SystemExit("--input-index must be inside --evaluation-root") from exc

    command = [
        args.python,
        str(evaluator),
        "--root", str(root),
        "--manifest", args.manifest,
        "--input-index", index_argument,
        "--run-id", args.run_id,
        "--compile-timeout", str(args.compile_timeout),
        "--task-timeout", str(args.task_timeout),
    ]
    if args.methods:
        command += ["--methods", *args.methods]
    if args.tasks:
        command += ["--tasks", *args.tasks]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=root, check=True)
    print(f"Results: {root / 'runs' / args.run_id}")


if __name__ == "__main__":
    main()
