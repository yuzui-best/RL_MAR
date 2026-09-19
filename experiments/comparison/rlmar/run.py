"""Stable public launcher for the RL-MAR comparison run.

The experiment engine is selected explicitly so that the frozen paper runner
can be archived independently of this stable command line.  Arguments after
``--`` are forwarded unchanged to that engine.
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
        default=Path(__file__).with_name("evaluate_five_models.py"),
        help="Frozen evaluate_five_models.py used for the paper",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.runner.is_file():
        raise SystemExit(f"RL-MAR paper runner not found: {args.runner}")
    forwarded = args.runner_args[1:] if args.runner_args[:1] == ["--"] else args.runner_args
    command = [args.python, str(args.runner.resolve()), *forwarded]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    environment = os.environ.copy()
    environment.setdefault("RL_MAR_FIVE_MODEL_SETTINGS", str(Path(__file__).with_name("settings.json")))
    environment.setdefault(
        "PLC_FUNCTIONAL_ROOT",
        str(Path(__file__).resolve().parents[3] / "evaluation" / "functional94"),
    )
    subprocess.run(command, cwd=args.runner.resolve().parent, env=environment, check=True)


if __name__ == "__main__":
    main()
