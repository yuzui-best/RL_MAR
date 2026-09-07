#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

MODELS = ['GPT-4o', 'GPT-4o-mini', 'DeepSeek-V3.2', 'Qwen3.5-9B', 'DeepSeek-v4-flash']
BLUE, GREEN, RED = '#315794', '#40662B', '#A85B5B'


def save(fig, outdir, name):
    fig.savefig(outdir / (name + '.pdf'))
    fig.savefig(outdir / (name + '.svg'))
    preview_dir = outdir / 'previews'
    preview_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_dir / (name + '.png'), dpi=300)
    plt.close(fig)


def functional94(report, outdir):
    fig, (ax, stages) = plt.subplots(2, 1, figsize=(3.5, 4.5), layout='constrained')
    offsets = [(4, 8), (4, -13), (-40, -18), (4, -13), (-22, 12)]
    for model, offset in zip(MODELS, offsets):
        rows = [report['functional94'][model][run] for run in ['primary', 'confirmation']]
        x = [r['tokens'] / 1000 for r in rows]
        y = [100 * r['pass_count'] / 94 for r in rows]
        ax.plot(x, y, color='#BAC2CA', lw=0.9, zorder=1)
        ax.scatter(x[0], y[0], color=BLUE, s=23)
        ax.scatter(x[1], y[1], edgecolors=GREEN, facecolors='white', marker='s', s=23)
        ax.annotate(model, (x[0], y[0]), xytext=offset, textcoords='offset points', fontsize=6.5, arrowprops={'arrowstyle': '-', 'color': '#747C87', 'lw': .5})
    ax.scatter([], [], color=BLUE, s=23, label='Primary')
    ax.scatter([], [], edgecolors=GREEN, facecolors='white', marker='s', s=23, label='Confirmation')
    ax.legend(loc='upper right', fontsize=6.5)
    values = [report['functional94'][m][r] for m in MODELS for r in ['primary', 'confirmation']]
    xs = [v['tokens'] / 1000 for v in values]
    ys = [100 * v['pass_count'] / 94 for v in values]
    ax.set(xlabel='Total tokens (k)', ylabel='Func. Pass (%)', xlim=(min(xs)-15, max(xs)+20), ylim=(min(ys)-4, max(ys)+5), title='(a) Quality and token use')
    categories = [('PASS', 'Func. Pass', BLUE), ('FUNCTION_FAIL', 'Func. fail', RED), ('COMPILE_FAIL', 'Comp. fail', '#747C87'), ('RUNTIME_FAIL', 'Runtime fail', '#C99846'), ('TIMEOUT', 'Timeout', '#C8B4D1'), ('EVALUATION_ERROR', 'Evaluation error', '#DBDFE4'), ('DATASET_ERROR', 'Dataset error', '#B8C8C2'), ('MISSING_OUTPUT', 'Missing', '#E5E5E5')]
    left = np.zeros(5)
    for key, label, color in categories:
        values = np.array([report['functional94'][m]['primary']['status_counts'].get(key, 0) for m in MODELS])
        if not values.any():
            continue
        stages.barh(np.arange(5), values, left=left, color=color, label=label, height=.58)
        for i, value in enumerate(values):
            if value >= 7:
                stages.text(left[i] + value / 2, i, str(value), ha='center', va='center', color='white', fontsize=7)
        left += values
    assert np.all(left == 94)
    stages.set(yticks=np.arange(5), yticklabels=MODELS, xlim=(0, 94), xticks=[0, 23, 47, 70, 94], xlabel='Tasks (/94)', title='(b) Primary terminal outcomes')
    stages.invert_yaxis()
    fig.legend(*stages.get_legend_handles_labels(), ncol=3, loc='lower center', bbox_to_anchor=(.5, 0), fontsize=7, handlelength=1, columnspacing=1.0)
    fig.get_layout_engine().set(rect=(0, .10, 1, .90))
    save(fig, outdir, 'rq1_backbone_comparison')


def agents4plc23(report, outdir):
    fig, ax = plt.subplots(figsize=(3.5, 2.15), layout='constrained')
    rows = [report['agents4plc23'][m][method] for m in ['GPT-4o', 'GPT-4o-mini'] for method in ['rl_mar', 'agents4plc_reported']]
    verified = np.array([r['verification_passes'] for r in rows])
    compiled = np.array([r['compile_passes'] for r in rows])
    assert np.all((0 <= verified) & (verified <= compiled) & (compiled <= 23))
    left = np.zeros(4)
    for values, label, color in [(verified, 'Verif.', BLUE), (compiled - verified, 'Compiled, unverified', '#747C87'), (23 - compiled, 'Comp. fail', RED)]:
        ax.barh(np.arange(4), values, left=left, label=label, color=color, height=.6)
        for i, value in enumerate(values):
            if value:
                ax.text(left[i] + value / 2, i, str(value), color='white', ha='center', va='center')
        left += values
    ax.set(yticks=np.arange(4), yticklabels=['RL-MAR / 4o', 'Published / 4o', 'RL-MAR / 4o-mini', 'Published / 4o-mini'], xlim=(0, 23), xticks=[0, 5, 10, 15, 20, 23], xlabel='Tasks (/23)')
    ax.invert_yaxis()
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, 0), ncol=3, fontsize=6.5, handlelength=.9, columnspacing=.8, handletextpad=.4)
    fig.get_layout_engine().set(rect=(0, .12, 1, .88))
    save(fig, outdir, 'rq1_formal_pass_verif')


def route_ablation(report, outdir):
    ab = report['ablations']
    fig, (ax, repair) = plt.subplots(1, 2, figsize=(3.5, 2.05), layout='constrained')
    for run, balanced, quality, color in zip([1, 2, 3], ab['balanced_three_seed'], ab['quality_three_seed'], [BLUE, RED, GREEN]):
        values = [quality['pass_count'], balanced['pass_count']]
        ax.plot([0, 1], values, 'o-', color=color, ms=3, lw=1, label=f'Run {run}')
    ax.set(xticks=[0, 1], xticklabels=['Quality', 'Balanced'], xlim=(-.2, 1.2), ylim=(3.5, 7.5), yticks=[4, 5, 6, 7], ylabel='Func. Pass (/12)', title='(a) Individual runs')
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, 0), ncol=3, fontsize=6.5, handlelength=1, columnspacing=1)
    fig.get_layout_engine().set(rect=(0, .12, 1, .88))
    rows = [ab['no_repair'], ab['repair_budget_one_reference']]
    tokens = [(r['prompt_tokens'] + r['completion_tokens']) / 1000 for r in rows]
    passes = [r['pass_count'] for r in rows]
    assert passes == [5, 7]
    repair.plot(tokens, passes, 'o-', color=BLUE, ms=4)
    for x, y, label, offset in zip(tokens, passes, ['No correction', 'One correction'], [(0, -15), (-52, 8)]):
        repair.annotate(label, (x, y), xytext=offset, textcoords='offset points', fontsize=6.5)
    repair.set(xlim=(49, 64), ylim=(3.5, 8), yticks=[4, 5, 6, 7, 8], xticks=[50, 55, 60], xlabel='Total tokens (k)', title='(b) Feedback cost')
    save(fig, outdir, 'rq2_postproc_recovery')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6, 'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.bbox': 'tight', 'savefig.pad_inches': .04})
    report = json.loads(args.report.read_text())
    for model in MODELS:
        for run in ['primary', 'confirmation']:
            row = report['functional94'][model][run]
            assert sum(row['status_counts'].values()) == 94
            assert row['pass_count'] == row['status_counts'].get('PASS', 0)
    functional94(report, args.outdir)
    agents4plc23(report, args.outdir)
    route_ablation(report, args.outdir)


if __name__ == '__main__':
    main()
