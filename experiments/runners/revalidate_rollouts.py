from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
sys.path[:0] = [str(ROOT), str(FRAMEWORK)]

from MAR.Core.dataset import load_reward_records, stable_sha256
from MAR.Core.outcomes import classify_outcome
from MAR.Core.verifier import VerifierConfig, verify_rewardplc


def selected_code(row: dict) -> str:
    selected = [attempt for attempt in row.get("attempts", []) if attempt.get("selected")]
    attempt = selected[-1] if selected else (row.get("attempts") or [{}])[-1]
    return attempt.get("code", "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministically revalidate saved RewardPLC rollout code")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rusty", type=Path, required=True)
    parser.add_argument("--mount-root", type=Path, required=True)
    args = parser.parse_args()
    records = {record.task_id: record for record in load_reward_records(args.dataset)}
    dataset_sha256 = stable_sha256(args.dataset.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with args.output.open("w", encoding="utf-8") as stream:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("dataset_sha256") != dataset_sha256:
                raise ValueError(f"dataset hash mismatch for {row.get('task_id')}")
            record = records[row["task_id"]]
            if row.get("family_id") != record.family_id or row.get("split") != record.split:
                raise ValueError(f"task lineage mismatch for {row['task_id']}")
            validation = verify_rewardplc(
                record, selected_code(row),
                VerifierConfig(rusty=args.rusty.resolve(), mount_root=args.mount_root.resolve()),
            )
            outcome = classify_outcome(validation)
            row["validation"] = validation
            row["raw_reward"] = outcome.raw_reward
            row["policy_eligible"] = outcome.policy_eligible
            row["deterministically_revalidated"] = True
            counts[validation["status"]] = counts.get(validation["status"], 0) + 1
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    print(json.dumps({"rows": sum(counts.values()), "statuses": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
