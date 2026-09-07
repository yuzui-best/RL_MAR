from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.MAR.Core.dataset import load_reward_records, stable_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile every RewardPLC reference with RuSTy")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "rewardplc" / "rewardplc_750.json")
    parser.add_argument("--rusty", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "datasets" / "rewardplc" / "compile_audit.json")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--docker-image", default="ubuntu:24.04")
    parser.add_argument("--mount-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    records = load_reward_records(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]
    failures: list[dict[str, str]] = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="rewardplc-rusty-", dir=args.dataset.parent) as directory:
        work = Path(directory)
        for record in records:
            source = work / f"{record.task_id}.st"
            target = work / f"{record.task_id}.o"
            source.write_text(record.reference_st, encoding="utf-8")
            try:
                compiler = [
                    "docker", "run", "--rm",
                    "-v", f"{args.mount_root}:{args.mount_root}",
                    args.docker_image,
                    str(args.rusty),
                ]
                result = subprocess.run(
                    compiler + [str(source), "-o", str(target), "-c", "--target", "x86_64-linux-gnu"],
                    text=True,
                    capture_output=True,
                    timeout=args.timeout,
                )
            except subprocess.TimeoutExpired:
                failures.append({"task_id": record.task_id, "diagnostic": "TIMEOUT"})
                continue
            if result.returncode:
                failures.append({"task_id": record.task_id, "diagnostic": (result.stderr or result.stdout)[-4000:]})

    report = {
        "dataset_sha256": stable_sha256(args.dataset.read_bytes()),
        "rusty_sha256": stable_sha256(args.rusty.read_bytes()),
        "tasks": len(records),
        "compiled": len(records) - len(failures),
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
        "status": "PASS" if not failures else "FAIL",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
