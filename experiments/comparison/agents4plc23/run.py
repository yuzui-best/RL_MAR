"""Stable launcher for the frozen Agents4PLC'23 comparison runner.

Arguments following ``--`` are passed to the archived runner verbatim.  The
launcher exists to keep third-party/runtime-specific code out of the public
result directory while preserving an exact reproduction entry point.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("evaluate_agents4plc23.py"),
        help="Frozen evaluate_agents4plc23.py used for the paper",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.runner.is_file():
        raise SystemExit(f"Agents4PLC'23 runner not found: {args.runner}")
    forwarded = args.runner_args[1:] if args.runner_args[:1] == ["--"] else args.runner_args
    command = [args.python, str(args.runner.resolve()), *forwarded]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    environment = os.environ.copy()
    environment.setdefault(
        "RL_MAR_FIVE_MODEL_SETTINGS",
        str(Path(__file__).resolve().parents[1] / "rlmar" / "settings.json"),
    )
    subprocess.run(command, cwd=args.runner.resolve().parent, env=environment, check=True)


if __name__ == "__main__":
    main()
