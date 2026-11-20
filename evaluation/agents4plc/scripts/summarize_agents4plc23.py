"""Summarize official Agents4PLC verdicts, preserving both denominators."""
import argparse
import csv
import json
from pathlib import Path

from run_alignment import official_counts


def summarize(results, samples=10000):
    import numpy as np
    rng = np.random.default_rng(1818)
    summaries = []
    for condition in sorted({r['condition'] for r in results}):
        for model in sorted({r['model'] for r in results}):
            for difficulty in ('Easy', 'Medium', 'All'):
                group = [r for r in results if r['condition'] == condition and r['model'] == model
                         and (difficulty == 'All' or r['difficulty'] == difficulty)]
                if not group:
                    continue
                metrics = []
                for record in group:
                    counts = official_counts([p['verdict'] for p in record['properties']])
                    compiled = bool(record['compile'])
                    metrics.append([compiled, compiled and counts['verified'],
                                    compiled and counts['satisfied80']])
                matrix = np.array(metrics, dtype=float)
                boot = matrix[rng.integers(len(group), size=(samples, len(group)))].sum(axis=1)
                totals = matrix.sum(axis=0).astype(int).tolist()
                compiled, verified, satisfied = totals
                row = dict(condition=condition, model=model, difficulty=difficulty, n=len(group),
                    compiled=compiled, verified=verified, satisfied80=satisfied,
                    compilation_percent=100 * compiled / len(group),
                    verified_percent_official=100 * verified / compiled if compiled else None,
                    satisfied80_percent_official=100 * satisfied / compiled if compiled else None,
                    verified_percent_full=100 * verified / len(group),
                    satisfied80_percent_full=100 * satisfied / len(group),
                    mean_generation_tokens=sum(r['generation_tokens'] for r in group) / len(group),
                    mean_generation_calls=sum(r['generation_calls'] for r in group) / len(group),
                    mean_reverification_seconds=sum(r['seconds'] for r in group) / len(group))
                for i, name in enumerate(('compile', 'verified_full', 'satisfied80_full')):
                    row[name + '_ci95_pp'] = np.percentile(100 * boot[:, i] / len(group), [2.5, 97.5]).tolist()
                nonzero = boot[:, 0] > 0
                for i, name in ((1, 'verified_official'), (2, 'satisfied80_official')):
                    row[name + '_ci95_pp'] = np.percentile(
                        100 * boot[nonzero, i] / boot[nonzero, 0], [2.5, 97.5]).tolist() if nonzero.any() else None
                summaries.append(row)
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True, help='run_alignment RESULTS.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = summarize(json.loads(args.results.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(data, indent=2) + '\n')
    with (args.output_dir / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
