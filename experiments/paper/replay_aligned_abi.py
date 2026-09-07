"""Replay saved compiled candidates; no ST changes, compilation, or API calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from aligned_abi import install


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def worker(args):
    request = json.loads(args.worker.read_text())
    root = Path(request['root'])
    sys.path.insert(0, str(root / 'scripts'))
    spec = importlib.util.spec_from_file_location('aligned_engine', root / 'scripts/run_functional94.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    install(engine)
    work = root / 'runs' / request['run_id'] / 'build' / request['method_slug'] / request['task']['task_id']

    def cached_compile(root, run_id, method_slug, task, source, timeout):
        assert (work / 'program.st').read_bytes() == source.encode('utf-8'), 'cached source mismatch'
        assert (work / 'program.so').is_file() and (work / 'program.h').is_file()
        return work / 'program.so', work / 'program.h', 0.0

    engine.compile_source = cached_compile
    result = engine.worker(request)
    result['cached_artifacts'] = {n: sha(work / n) for n in ['program.st', 'program.h', 'program.so']}
    dump(args.output, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--reference', action='store_true')
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    filename = 'reference_results.json' if args.reference else 'unified_results.json'
    original = json.loads((args.saved / filename).read_text())
    jobs = [(g, r) for g in original['groups'] for r in g['tasks']]

    def run(job):
        group, old = job
        if not old['compiled']:
            return dict(old, aligned_replay='unchanged_compile_failure')
        request = args.saved / 'responses' / group['slug'] / (old['task_id'] + '.request.json')
        output = args.out / 'responses' / group['slug'] / (old['task_id'] + '.json')
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            process = subprocess.run([sys.executable, __file__, '--worker', str(request), '--output', str(output)], capture_output=True, timeout=180)
        except subprocess.TimeoutExpired:
            process = None
        if process is None:
            new = dict(status='TIMEOUT', diagnostic='Worker exceeded 180 seconds', checks=[])
        elif process.returncode:
            if process.returncode >= 0:
                raise RuntimeError(f'worker failed: {group["slug"]}/{old["task_id"]}: {process.stderr.decode()[-1000:]}')
            new = dict(status='RUNTIME_FAIL', diagnostic=f'worker signal {-process.returncode}', checks=[], returncode=process.returncode)
        else:
            new = json.loads(output.read_text())
        result = dict(old)
        for key in ['diagnostic', 'returncode', 'failure_category']:
            result.pop(key, None)
        result.update(new)
        result.update(compiled=True, aligned_replay='cached_binary', original_status=old['status'], adapter_sha256=sha(Path(__file__).with_name('aligned_abi.py')))
        dump(output, result)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run, jobs))
    transitions = []
    index = 0
    for group in original['groups']:
        count = len(group['tasks'])
        old_rows = group['tasks']
        group['tasks'] = rows[index:index + count]
        index += count
        for old, new in zip(old_rows, group['tasks']):
            if old['status'] != new['status']:
                transitions.append(dict(method=group['method'], model=group['model'], run=group['run'], task_id=old['task_id'], before=old['status'], after=new['status']))
        group['summary'] = dict(compiled=sum(r['compiled'] for r in group['tasks']), passed=sum(r['status'] == 'PASS' for r in group['tasks']), statuses=dict(Counter(r['status'] for r in group['tasks'])))
    original['alignment'] = dict(original_results_sha256=sha(args.saved / filename), adapter_sha256=sha(Path(__file__).with_name('aligned_abi.py')), replay_sha256=sha(Path(__file__)), source_and_vectors='unchanged', mode='cached compiled binaries; compile failures retained')
    dump(args.out / filename, original)
    dump(args.out / ('reference_transitions.json' if args.reference else 'transitions.json'), transitions)
    print(json.dumps(dict(records=len(rows), transitions=transitions, groups=[dict(method=g['method'], model=g['model'], run=g['run'], **g['summary']) for g in original['groups']]), indent=2))


if __name__ == '__main__':
    main()
