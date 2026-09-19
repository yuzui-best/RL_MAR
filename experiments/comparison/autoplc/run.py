"""Reproduce the AutoPLC baseline using a separately obtained upstream checkout.

AutoPLC is not vendored here.  This adapter copies the supplied checkout into a
temporary working directory, injects Functional94, and invokes its public
benchmark entry point.  This keeps third-party source and local caches out of
the RL-MAR release.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("functional94_gpt55_consistent.yaml"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--single-task")
    parser.add_argument("--task-range")
    parser.add_argument("--keep-workdir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entry = args.upstream_root.resolve() / "autoplc" / "main_st.py"
    if not entry.is_file():
        raise SystemExit(f"AutoPLC entry point not found: {entry}")
    for required in (args.dataset, args.config):
        if not required.is_file():
            raise SystemExit(f"Required file not found: {required}")

    owner = None
    if args.keep_workdir:
        work = args.keep_workdir.resolve()
        if work.exists():
            raise SystemExit(f"Refusing to overwrite work directory: {work}")
        shutil.copytree(args.upstream_root.resolve(), work)
    else:
        owner = tempfile.TemporaryDirectory(prefix="rlmar-autoplc-")
        work = Path(owner.name) / "source"
        shutil.copytree(args.upstream_root.resolve(), work)

    benchmark = work / "data" / "benchmarks" / "functional94.jsonl"
    benchmark.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.dataset.resolve(), benchmark)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python,
        str(work / "autoplc" / "main_st.py"),
        "--benchmark",
        "--config",
        str(args.config.resolve()),
        "--checkpoint_dir",
        str(args.output_dir.resolve()),
    ]
    if args.single_task:
        command += ["--single_task", args.single_task]
    if args.task_range:
        command += ["--task_range", args.task_range]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    try:
        subprocess.run(command, cwd=work, check=True)
    finally:
        if owner is not None:
            owner.cleanup()


if __name__ == "__main__":
    main()
