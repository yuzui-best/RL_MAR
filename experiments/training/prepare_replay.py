"""Create a fresh GPU training workspace from the published measured rewards."""
import argparse
from pathlib import Path
import shutil


def prepare(output):
    here = Path(__file__).resolve().parent
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("FROZEN_CONFIG.json", "feasible_actions.json", "development_rewards.json"):
        shutil.copy2(here / name, output / name)
    shutil.copytree(here / "data", output / "data")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.out))
