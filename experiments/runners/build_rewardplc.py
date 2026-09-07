from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.MAR.Core.dataset import write_dataset
from framework.MAR.Core.rewardplc_generator import generate_rewardplc, mutation_kill_rate


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate RewardPLC-750")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "rewardplc" / "rewardplc_750.json")
    parser.add_argument("--minimum-mutation-score", type=float, default=0.25)
    args = parser.parse_args()
    records = generate_rewardplc()
    weak = {record.task_id: mutation_kill_rate(record) for record in records if mutation_kill_rate(record) < args.minimum_mutation_score}
    if weak:
        print(json.dumps({"status": "FAIL", "weak_mutation_scores": weak}, indent=2))
        return 2
    manifest = write_dataset(records, args.output)
    manifest["minimum_mutation_score"] = min(mutation_kill_rate(record) for record in records)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
