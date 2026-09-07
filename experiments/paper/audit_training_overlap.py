"""Current-manifest overlap screens; outputs hashes/counts, never hidden traces."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import re
import tarfile

from audit_saved_evidence import sha


def tokens(text):
    return re.findall(r'[a-z0-9_]+', text.lower())


def signature(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def interface(record, reward=False):
    result = {k: {} for k in ['inputs', 'outputs', 'internal', 'inout']}
    if reward:
        pairs = [(section, name, kind) for section, fields in record['pou_interface'].items() for name, kind in fields.items()]
    else:
        sections = {'input': 'inputs', 'output': 'outputs', 'internal': 'internal', 'var': 'internal', 'inout': 'inout'}
        pairs = [(sections[d['section']], d['name'], d['type']) for d in record['pou_declarations']]
    for section, name, kind in pairs:
        result[section][name.lower()] = re.sub(r'\s+', '', kind.upper())
    return result


def trace(record, reward=False):
    cases = []
    for case in record['cases']:
        if reward:
            initial, scans, expected = case['initial_state'], case['scans'], case['expected']
        else:
            oracle = case['oracle']
            initial = {**case.get('initial_outputs', {}), **case.get('initial_state', {})}
            scans = [{k: v[i] for k, v in case['input_trace'].items()} for i in range(case['simulation_rounds'])]
            outputs = {**oracle.get('expected_output_trace', {}), **oracle.get('expected_internal_state', {})}
            expected = [{k: v[i] for k, v in outputs.items()} for i in range(case['simulation_rounds'])]
        lower = lambda row: {k.lower(): v for k, v in row.items()}
        cases.append(dict(initial=lower(initial), inputs=[lower(s) for s in scans], expected=[lower(s) for s in expected]))
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    args = p.parse_args()
    private = args.experiments/audit/'private'
    corpus_bytes = (private/'rewardplc_750.json').read_bytes()
    reward = json.loads(corpus_bytes)['tasks']
    with tarfile.open(args.results/'evaluation_archive.tar.gz') as outer:
        snapshot = outer.extractfile('source_snapshot.tar.gz').read()
        refs = {t['task_id']: json.load(outer.extractfile('responses/reference_reference_pilot/'+t['task_id']+'.request.json'))['source']
                for t in json.loads((args.results/'reference_results.json').read_text())['groups'][0]['tasks']}
    with tarfile.open(fileobj=io.BytesIO(snapshot)) as archive:
        path = next(n for n in archive.getnames() if n.endswith('/data/plc94.functional.v1.3.jsonl'))
        manifest_bytes = archive.extractfile(path).read()
    functional = [json.loads(line) for line in manifest_bytes.splitlines()]
    features = []
    for task in functional:
        features.append(dict(task_id=task['task_id'], words=set(tokens(task['instruction'])),
            interface=signature(interface(task)),
            trace=signature(trace(task)) if task['validation_mode']=='exact_trace' else None,
            reference=signature(tokens(refs[task['task_id']])) if task['task_id'] in refs else None))
    findings, closest = [], []
    for task in reward:
        words = set(tokens(task['specification']))
        inter, tr, ref = signature(interface(task, True)), signature(trace(task, True)), signature(tokens(task['reference_st']))
        similarities = []
        for other in features:
            similarity = len(words & other['words'])/len(words | other['words'])
            similarities.append((similarity, other['task_id']))
            matches = []
            if similarity >= .85: matches.append('specification_jaccard')
            if inter == other['interface']: matches.append('interface')
            if other['trace'] and tr == other['trace']: matches.append('trace_projection')
            if other['reference'] and ref == other['reference']: matches.append('reference_tokens')
            if matches: findings.append(dict(reward_task=task['task_id'], functional_task=other['task_id'], matches=matches))
        best, other_id = max(similarities)
        closest.append(dict(reward_task=task['task_id'], functional_task=other_id, jaccard=best))
    roll_bytes = (private/'deepseek_valid_train.jsonl').read_bytes()
    rollouts = [json.loads(line) for line in roll_bytes.splitlines()]
    build = json.loads((args.audit.parents[1]/'experiments/audit/stage4_policy_build_018.json').read_text())
    report = json.loads((args.audit.parents[1]/'experiments/runners/checkpoints/deepseek_route_valid_v2_report.json').read_text())
    behavior_hash = sha((private/'deepseek_valid_train.behavior_policy.pt').read_bytes())
    base_hash = sha((private/'deepseek_route_valid_v2.pt').read_bytes())
    assert base_hash == build['base_policy_sha256']
    assert sha(roll_bytes) in report['rollout_sha256']
    assert all(r['dataset_sha256']==sha(corpus_bytes) and r['behavior_policy_sha256']==behavior_hash for r in rollouts)
    corpus = {t['task_id']: t for t in reward}
    assert all(r['specification']==corpus[r['task_id']]['specification'] and r['split']=='train' for r in rollouts)
    final_ids = {r['task_id'] for r in build['routes']}
    assert all(corpus[i]['split']=='evolution_dev' for i in final_ids)
    output = dict(hashes=dict(rewardplc=sha(corpus_bytes), functional94=sha(manifest_bytes),
                  script=sha(Path(__file__).read_bytes()), generator=sha((private/'rewardplc_generator.py').read_bytes())),
        coverage=dict(reward_tasks=len(reward), functional_tasks=len(functional), specification_pairs=len(reward)*len(functional),
                      interface_pairs=len(reward)*len(functional), reference_pairs=len(reward)*len(refs),
                      exact_trace_tasks=sum(t['validation_mode']=='exact_trace' for t in functional)),
        criteria=dict(jaccard_threshold=.85, specification_tokenization='lowercase ASCII words, numbers, underscores; token sets',
            interface='case-insensitive field names, whitespace-normalized types, four distinct sections including IN_OUT',
            traces='scan-major input/output/internal-state projection with explicit initial state; drops IDs and tolerances; exact equality only',
            reference='token sequence equality of 18 archived reference candidates; other 76 references unavailable in this screen'),
        findings=findings, closest_specs=sorted(closest,key=lambda r:r['jaccard'],reverse=True)[:10],
        corpus=dict(families=len({r['family_id'] for r in reward}), domains=dict(Counter(r['domain'] for r in reward)),
                    splits=dict(Counter(r['split'] for r in reward)), independently_authored_templates=5),
        lineage=dict(base_checkpoint_sha256=base_hash, rollout_sha256=sha(roll_bytes), behavior_sha256=behavior_hash,
            rollout_rows=len(rollouts), unique_tasks=sorted({r['task_id'] for r in rollouts}),
            base_objective='clipped policy loss per train_route_policy.py; report has 20 epochs and one group',
            final_objective='negative target-route log likelihood; 600 epochs, 12 development tasks',
            final_model_conditioning='build_stage4_policy.py uses model ID 4 (DeepSeek-v4-flash) for every fitted pair',
            fitting_script_sha256=sha((args.audit.parents[1]/'experiments/build_stage4_policy.py').read_bytes()),
            base_final_task_overlap=sorted(final_ids & {r['task_id'] for r in rollouts}),
            initialization_boundary='behavior checkpoint stores seed 10234; collector supports fresh initialization; exact initialization command and encoder provenance not recovered'),
        limitations=['No semantic equivalence or paraphrase detection.', 'Reference candidates are not a certified oracle corpus.',
                     'Exact trace projection is a narrow equality screen; property-mode tasks are excluded from trace matching.',
                     'Family split does not hold out template semantics.', 'No claim about pretraining contamination of the sentence encoder or generation backbones.'])
    (args.experiments/audit/'training_overlap.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['coverage','findings','lineage']},indent=2))
    print('Maximum specification Jaccard:', output['closest_specs'][0]['jaccard'])


if __name__=='__main__':
    main()
