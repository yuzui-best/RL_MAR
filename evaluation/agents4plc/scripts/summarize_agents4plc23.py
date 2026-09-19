#!/usr/bin/env python3
"""
Summarize Agents4PLC 23-task evaluation results.
Outputs summary.json, summary.csv, and table_agents4plc.tex.

Usage:
  python3 scripts/summarize_agents4plc23.py \
    --eval results/evaluation.jsonl \
    --out_dir results \
    --method "RL-MAR" \
    --model "gpt-4o"
"""

import argparse
import csv
import json
from pathlib import Path


def load_eval(path):
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def make_summary(results, method, model):
    splits = {"Easy": [], "Medium": [], "Hard": []}
    for r in results:
        diff = r.get("difficulty", "Unknown")
        if diff not in splits:
            splits[diff] = []
        splits[diff].append(r)

    all_tasks = results
    rows = []

    for split_name in ["Easy", "Medium", "Hard", "Overall"]:
        if split_name == "Overall":
            subset = all_tasks
        else:
            subset = splits.get(split_name, [])

        if not subset:
            continue

        total = len(subset)
        comp_solved = sum(1 for r in subset if r.get("comp_pass"))
        verif_solved = sum(1 for r in subset if r.get("verif_pass"))
        comp_rate = 100 * comp_solved / total if total > 0 else 0
        verif_rate = 100 * verif_solved / total if total > 0 else 0

        rows.append(
            {
                "method": method,
                "model": model,
                "split": split_name,
                "total": total,
                "comp_solved": comp_solved,
                "verif_solved": verif_solved,
                "comp_rate": round(comp_rate, 1),
                "verif_rate": round(verif_rate, 1),
            }
        )

    return rows


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "model", "split", "total",
                "comp_solved", "verif_solved", "comp_rate", "verif_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows, path):
    """Write a simple LaTeX table."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[h]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{" + f"{rows[0]['method']} ({rows[0]['model']}) on Agents4PLC v1" + r"}" + "\n")
        f.write(r"\label{tab:agents4plc}" + "\n")
        f.write(r"\begin{tabular}{lrrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Split & Total & Comp. & Verif. & Comp.\% & Verif.\% \\" + "\n")
        f.write(r"\midrule" + "\n")
        for row in rows:
            f.write(
                f"{row['split']} & {row['total']} & {row['comp_solved']} & "
                f"{row['verif_solved']} & {row['comp_rate']} & {row['verif_rate']} \\\\\n"
            )
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize Agents4PLC evaluation")
    parser.add_argument("--eval", required=True, help="Path to evaluation.jsonl")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--method", required=True, help="Method name")
    parser.add_argument("--model", required=True, help="Model name")
    args = parser.parse_args()

    results = load_eval(args.eval)
    rows = make_summary(results, args.method, args.model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write summary.json
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # Write summary.csv
    csv_path = out_dir / "summary.csv"
    write_csv(rows, csv_path)

    # Write LaTeX table
    tex_path = out_dir / "table_agents4plc.tex"
    write_latex(rows, tex_path)

    print(f"Summary written to:")
    print(f"  {summary_path}")
    print(f"  {csv_path}")
    print(f"  {tex_path}")

    # Print to stdout
    print(f"\n{'='*60}")
    print(f"Method: {args.method} | Model: {args.model}")
    print(f"{'Split':<10} {'Total':>6} {'Comp':>6} {'Verif':>6} {'Comp%':>7} {'Verif%':>7}")
    print(f"{'-'*46}")
    for row in rows:
        print(
            f"{row['split']:<10} {row['total']:>6} {row['comp_solved']:>6} "
            f"{row['verif_solved']:>6} {row['comp_rate']:>6.1f} {row['verif_rate']:>6.1f}"
        )


if __name__ == "__main__":
    main()
