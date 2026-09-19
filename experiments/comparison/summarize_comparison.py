"""Build paper-facing comparison tables from Functional94 evaluation JSON.

Each input is labelled ``METHOD:MODEL:PATH`` and must contain a task list
either at the top level or under ``tasks``/``results``.  The script writes a
compact summary, a task-level CSV, and paired exact McNemar statistics.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any


def rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("tasks", "results", "records"):
            if isinstance(value.get(key), list):
                return value[key]
    raise ValueError("Input does not contain a task list")


def pick(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def truth(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "pass", "passed", "success", "ok"}
    return bool(value)


def exact_mcnemar(a: dict[str, bool], b: dict[str, bool]) -> dict[str, Any]:
    common = sorted(set(a) & set(b))
    n10 = sum(a[key] and not b[key] for key in common)
    n01 = sum(not a[key] and b[key] for key in common)
    n = n10 + n01
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(n10, n01) + 1)) / (2**n)
        p = min(1.0, 2.0 * tail)
    return {"common_tasks": len(common), "a_only_pass": n10, "b_only_pass": n01, "p_exact_two_sided": p}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="METHOD:MODEL:PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    task_rows: list[dict[str, Any]] = []
    pass_maps: dict[str, dict[str, bool]] = {}
    for specification in args.input:
        try:
            method, model, raw_path = specification.split(":", 2)
        except ValueError as exc:
            raise SystemExit(f"Invalid --input value: {specification}") from exc
        path = Path(raw_path)
        value = json.loads(path.read_text(encoding="utf-8"))
        label = f"{method}/{model}"
        pass_maps[label] = {}
        for index, row in enumerate(rows(value), start=1):
            identifier = str(pick(row, "task_id", "id", "name", default=f"task_{index:03d}"))
            passed = truth(pick(row, "functional_pass", "passed", "success", "pass", "status", default=False))
            pass_maps[label][identifier] = passed
            task_rows.append(
                {
                    "method": method,
                    "model": model,
                    "task_id": identifier,
                    "functional_pass": int(passed),
                    "compile_pass": int(truth(pick(row, "compile_pass", "compiled", default=False))),
                    "tokens": pick(row, "tokens", "total_tokens", default=""),
                    "seconds": pick(row, "seconds", "latency_seconds", default=""),
                }
            )

    summary = []
    for (method, model), group in itertools.groupby(
        sorted(task_rows, key=lambda row: (row["method"], row["model"], row["task_id"])),
        key=lambda row: (row["method"], row["model"]),
    ):
        values = list(group)
        total = len(values)
        summary.append(
            {
                "method": method,
                "model": model,
                "tasks": total,
                "functional_passes": sum(row["functional_pass"] for row in values),
                "functional_pass_rate": (sum(row["functional_pass"] for row in values) / total if total else None),
                "compile_pass_rate": (sum(row["compile_pass"] for row in values) / total if total else None),
            }
        )

    paired = {}
    for left, right in itertools.combinations(sorted(pass_maps), 2):
        paired[f"{left} vs {right}"] = exact_mcnemar(pass_maps[left], pass_maps[right])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "paired_statistics.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "task_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]) if task_rows else ["method", "model", "task_id"])
        writer.writeheader()
        writer.writerows(task_rows)


if __name__ == "__main__":
    main()
