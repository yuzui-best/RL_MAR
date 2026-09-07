"""Build traceable CSV, manuscript tables, and plot input from unified results."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

MODELS = ['gpt-4o', 'gpt-4o-mini', 'deepseek-v3.2', 'qwen3.5-9b', 'deepseek-v4-flash']
NAMES = ['GPT-4o', 'GPT-4o-mini', 'DeepSeek-V3.2', 'Qwen3.5-9B', 'DeepSeek-v4-flash']
METHODS = ['Direct Generation', 'AutoPLC', 'LLM4PLC', 'RL-MAR']
FAMILIES = [('Parsing / delimiters', {'E006', 'E007'}), ('Unresolved references', {'E048'}), ('Unknown types', {'E052'}), ('Type / argument mismatch', {'E032', 'E073', 'E094'}), ('Duplicate / nonconstant symbols', {'E004', 'E080'})]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    groups = {(g['method'], g['model'], g['run']): g for g in data['groups']}
    assert len(groups) == 25
    flat = []
    summaries = []
    known = set().union(*(codes for _, codes in FAMILIES))
    members = {}
    report = json.loads(args.report.read_text())
    for method, model, run in groups:
        group = groups[method, model, run]
        rows = group['tasks']
        assert len(rows) == len({r['task_id'] for r in rows}) == 94
        assert all(r.get('aggregation_complete') for r in rows)
        assert all(r['compiled'] for r in rows if r['status'] == 'PASS')
        assert not any(r['status'] in {'EVALUATION_ERROR', 'DATASET_ERROR'} for r in rows), (method, model, run)
        assert all(r['compiled'] for r in rows if r['status'] in {'RUNTIME_FAIL', 'TIMEOUT'}), (method, model, run)
        resources = {r['task_id']: r for r in group['resources']}
        assert len(resources) == 94
        total_tokens = sum(r['tokens'] for r in resources.values())
        total_seconds = sum(r['seconds'] for r in resources.values())
        status = Counter(r['status'] for r in rows)
        summary = {'method': method, 'model': model, 'run': run, 'compiled': sum(r['compiled'] for r in rows), 'passed': status['PASS'], 'tokens': total_tokens, 'seconds': total_seconds, 'statuses': dict(status)}
        assert summary['compiled'] == group['summary']['compiled']
        assert summary['passed'] == group['summary']['passed']
        summaries.append(summary)
        for row in rows:
            flat.append({'method': method, 'model': model, 'run': run, 'task_id': row['task_id'], 'compiled': row['compiled'], 'status': row['status'], 'payload_sha256': row['payload_sha256'], 'extracted_sha256': row['extracted_sha256'], 'tokens': resources[row['task_id']]['tokens'], 'generation_seconds': resources[row['task_id']]['seconds']})
        if method == 'RL-MAR':
            report['functional94'][NAMES[MODELS.index(model)]][run].update(pass_count=status['PASS'], compile_pass_count=summary['compiled'], tokens=total_tokens, wall_seconds_sum=total_seconds, status_counts=dict(status))
            if run == 'primary':
                failed = [r for r in rows if r['status'] == 'COMPILE_FAIL']
                members[model] = {label: [] for label, _ in FAMILIES}
                members[model]['Other diagnostics'] = []
                for row in failed:
                    diagnostic = re.sub(r'\x1b\[[0-9;]*m', '', row.get('diagnostic', ''))
                    codes = set(re.findall(r'error\[(E\d+)\]', diagnostic))
                    for label, family in FAMILIES:
                        if codes & family:
                            members[model][label].append(row['task_id'])
                    if not codes or codes - known:
                        members[model]['Other diagnostics'].append(row['task_id'])
    out = args.results.parent
    with (out / 'task_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    for name, value in [('summary.json', summaries), ('diagnostic_membership.json', members), ('plot_report.json', report)]:
        (out / name).write_text(json.dumps(value, indent=2) + '\n')
    selected = {(s['method'], s['model']): s for s in summaries if s['run'] == 'primary'}
    table = [r'\begin{table*}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.18}',
             r'\caption{Functional94 correctness and original generation costs. Rates use 94 tasks; bold marks the best value per backbone.}',
             r'\label{tab:rq1}', r'\begin{tabularx}{\textwidth}{@{}lp{3pt}l*{4}{>{\raggedleft\arraybackslash}X}p{3pt}l*{4}{>{\raggedleft\arraybackslash}X}@{}}', r'\toprule[0.9pt]',
             r'\textbf{Model} & & \textbf{Method} & \textbf{Comp.} & \textbf{Func.\ Pass} & \textbf{Avg.\ $t$} & \textbf{Tokens} & & \textbf{Method} & \textbf{Comp.} & \textbf{Func.\ Pass} & \textbf{Avg.\ $t$} & \textbf{Tokens} \\',
             r' & & & (\%) & (\%) & (s) & (k) & & & (\%) & (\%) & (s) & (k) \\', r'\midrule[0.5pt]']
    for index, (model, name) in enumerate(zip(MODELS, NAMES)):
        if index:
            table.append(r'\addlinespace[2pt]')
        rows = [selected[method, model] for method in METHODS]
        best = [max(r['compiled'] for r in rows), max(r['passed'] for r in rows),
                min(r['seconds'] for r in rows), min(r['tokens'] for r in rows)]
        for j, pair in enumerate((('Direct Generation', 'LLM4PLC'), ('AutoPLC', 'RL-MAR'))):
            if index % 2 == 0:
                table.append(r'\rowcolor[gray]{0.95}')
            cells = [rf'\multirow{{-2}}{{*}}{{\textbf{{{name}}}}}' if j == 1 else '']
            for method in pair:
                cells.append('')
                row = selected[method, model]
                cells.append('Direct Gen.' if method == 'Direct Generation' else (r'\textbf{RL-MAR}' if method == 'RL-MAR' else method))
                values = [row['compiled'], row['passed'], row['seconds'], row['tokens']]
                for value, optimum, factor in zip(values, best, (100 / 94, 100 / 94, 1 / 94, 1 / 1000)):
                    formatted = f'{value * factor:.1f}'
                    cells.append(r'\textbf{' + formatted + '}' if value == optimum else formatted)
            table.append(' & '.join(cells) + r' \\')
    table += [r'\bottomrule[0.9pt]', r'\end{tabularx}', r'\end{table*}']
    diagnostics = [r'\begin{table}[htbp]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.12}',
                   r'\caption{Compiler diagnostics for RL-MAR on Functional94.}',
                   r'\label{tab:rq3_errors}', r'\begin{tabularx}{\columnwidth}{@{}>{\raggedright\arraybackslash}Xrrrrr@{}}', r'\toprule',
                   r'\textbf{Diagnostic family} & \textbf{4o} & \textbf{mini} & \textbf{V3.2} & \textbf{Qwen} & \textbf{v4} \\', r'\midrule']
    for index, label in enumerate([label for label, _ in FAMILIES] + ['Other diagnostics']):
        if index % 2 == 0:
            diagnostics.append(r'\rowcolor[gray]{0.95}')
        diagnostics.append(label + ' & ' + ' & '.join(str(len(members[model][label])) for model in MODELS) + r' \\')
    diagnostics += [r'\midrule', r'\textbf{Compile-failed tasks} & ' + ' & '.join(r'\textbf{' + str(selected['RL-MAR', model]['statuses'].get('COMPILE_FAIL', 0)) + '}' for model in MODELS) + r' \\', r'\bottomrule', r'\end{tabularx}', r'\end{table}']
    paper = (args.paper / 'paper.tex').read_text()
    for name, label, lines, environment in [('table_iii_audited.tex', 'tab:rq1', table, 'table*'), ('table_vi_audited.tex', 'tab:rq3_errors', diagnostics, 'table')]:
        text = '\n'.join(lines) + '\n'
        (args.paper / 'resubmission' / name).write_text(text)
        marker = paper.index('\\label{' + label + '}')
        start = paper.rfind('\\begin{' + environment + '}', 0, marker)
        end = paper.index('\\end{' + environment + '}', marker) + len('\\end{' + environment + '}')
        assert start >= 0
        paper = paper[:start] + text.strip() + paper[end:]
    (args.paper / 'paper.tex').write_text(paper)
    print(f'Validated {len(flat)} task rows across {len(summaries)} groups; tables and plot input updated.')


if __name__ == '__main__':
    main()
