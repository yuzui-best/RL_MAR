"""Frozen seed1818 identity/depth routing and Functional94 execution.

Public-only routing is decoded before verification. Each run writes its own
routes and candidates; recorded paper results are never overwritten.
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "framework"))
from MAR.V2.execution import Execution
from MAR.V2.budget import BudgetExceeded
from MAR.V2.request_budget import active_request_budget
PUBLIC_PROMPT_FIELDS = ("instruction", "instruction_sha256", "entry_point", "pou_interface")
PFX = {"gpt-4o":"GPT4O", "gpt-4o-mini":"GPT4O_MINI", "gpt-5.5":"GPT55", "deepseek-v3.2":"DEEPSEEK_V32", "deepseek-v4-flash":"DEEPSEEK_V4_FLASH", "qwen3.5-9b":"QWEN35"}


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_env(path):
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def public_prompt(task):
    """Only released name/category/interface are model-visible; verifier fields remain private."""
    instruction = str(task.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("missing public instruction")
    if hashlib.sha256(instruction.encode()).hexdigest() != task.get("instruction_sha256"):
        raise ValueError("instruction hash mismatch")
    interface = task.get("pou_interface") or []
    lines = [f"- {x['section'].upper()} {x['name']}: {x['type']}" for x in interface]
    return ("Implement this public PLC requirement in IEC 61131-3 Structured Text.\n"
            f"Public requirement:\n{instruction}\n\n"
            f"The entry point and interface are immutable: `{task['entry_point']}`. Do not rename, remove, or add interface fields.\n"
            "Public interface (including declared IEC types and array bounds):\n" + "\n".join(lines) +
            "\nUse compiler-portable IEC ST: close every declaration and POU, terminate statements with semicolons, "
            "avoid methods, unsupported standard-library blocks, and C-style casts unless the public requirement explicitly needs them.\n"
            "Return only complete IEC Structured Text. Hidden tests, stimuli, expected outputs, and oracles are unavailable.")

def extract_candidate(text):
    stripped = text.strip()
    fenced = re.search(r"```[^\n]*\n(.*?)```", stripped, re.DOTALL)
    if fenced: return fenced.group(1).strip()
    if stripped.startswith("```"): return "\n".join(stripped.splitlines()[1:]).strip()
    return stripped

def select_tasks(tasks, requested_ids):
    if not requested_ids: return tasks
    requested = set(requested_ids); available = {task["task_id"] for task in tasks}
    missing = sorted(requested - available)
    if missing: raise ValueError(f"unknown task ids: {missing}")
    return [task for task in tasks if task["task_id"] in requested]

def budgeted_action(action, settings):
    """Preserve the sampled route; RequestBudget enforces limits at execution time."""
    required = {"topology_index", "agent_count", "role_indices", "rounds", "repair_budget", "token_budget"}
    missing = sorted(required - action.keys())
    if missing:
        raise ValueError(f"route action missing fields: {missing}")
    return dict(action)

def assert_prompt_isolated(task, prompt):
    public_task = {field: task[field] for field in PUBLIC_PROMPT_FIELDS}
    if prompt != public_prompt(public_task):
        raise AssertionError("prompt depends on fields outside the public allowlist")

def public_validation(result):
    status = result.get("status", "ENVIRONMENT_ERROR")
    checks = result.get("checks") or []
    checks_passed = sum(check.get("status") == "PASS" for check in checks)
    stages_completed = 0 if status in {"COMPILE_FAIL", "ENVIRONMENT_ERROR", "API_ERROR"} else 1
    if status == "PASS": stages_completed = 2
    public = {"status": status, "stages_completed": stages_completed, "checks_passed": checks_passed}
    if status == "COMPILE_FAIL": public["diagnostic"] = str(result.get("diagnostic", ""))[-3000:]
    elif status in {"ENVIRONMENT_ERROR", "API_ERROR"}: public["diagnostic"] = str(result.get("diagnostic", status))[-1000:]
    else: public["diagnostic"] = "Compiled candidate failed hidden functional checks." if status != "PASS" else "PASS"
    return public


class Backend:
    """OpenAI-compatible transport; provider usage is the accounting source.

    GPT-5.5 may use a user-managed Responses-to-Chat loopback. The endpoint and
    model are always taken from the user's environment, with SDK retries off.
    """
    def __init__(self, model):
        self.model, self.usage, self.calls, self.spent = model, [], [], 0

    def prompt_tokens(self, messages):
        prefix = PFX[self.model]
        tokenizer_file = os.getenv(prefix + '_TOKENIZER')
        if tokenizer_file:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(tokenizer_file)
            count = lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids)
        elif self.model.startswith('gpt-'):
            import tiktoken
            encoding = tiktoken.get_encoding('o200k_base')
            count = lambda s: len(encoding.encode(s, disallowed_special=()))
        else:
            raise ValueError(prefix + '_TOKENIZER must point to the model tokenizer.json')
        content_tokens = sum(count(m['content']) for m in messages)
        if self.model == 'deepseek-v4-flash':
            return content_tokens + 128
        extra = int(os.getenv('GPT55_PROMPT_EXTRA_RESERVE', '512')) if self.model == 'gpt-5.5' else 0
        return content_tokens + 16 + 128 * len(messages) + extra

    async def agen(self, messages, max_tokens=2048, **_):
        from openai import AsyncOpenAI
        prefix = PFX[self.model]
        cfg = {name: os.environ[prefix + '_' + name] for name in ('URL', 'KEY', 'MODEL')}
        budget = active_request_budget()
        prompt = self.prompt_tokens(messages)
        # RequestBudget reserves completion first; the provider ledger must also
        # account for prompt tokens, as in the frozen experimental client.
        remaining = budget.max_tokens - self.spent if budget else 16384 - self.spent
        limit = min(max_tokens, remaining - prompt)
        if limit < 1:
            if budget:
                budget.stop_reason = 'prompt_budget_exhausted'
            raise BudgetExceeded('prompt_budget_exhausted')
        options = dict(model=cfg['MODEL'], messages=messages, temperature=0, max_tokens=limit)
        if self.model == 'qwen3.5-9b':
            options['extra_body'] = {'enable_thinking': False}
        elif self.model == 'deepseek-v4-flash':
            options['extra_body'] = {'thinking': {'type': 'disabled'}}
        started = time.monotonic()
        record = dict(status='RUNNING', requested_max_tokens=limit)
        self.calls.append(record)
        try:
            async with AsyncOpenAI(base_url=cfg['URL'], api_key=cfg['KEY'], max_retries=0,
                                   timeout=float(os.getenv('RL_MAR_API_TIMEOUT', '180'))) as client:
                response = await client.chat.completions.create(**options)
            if response.usage is None:
                record.update(status='UNKNOWN_USAGE', unknown_upper=prompt + limit)
                raise RuntimeError('provider did not return token usage')
            usage = dict(prompt_tokens=response.usage.prompt_tokens,
                         completion_tokens=response.usage.completion_tokens, cost=None)
            self.spent += usage['prompt_tokens'] + usage['completion_tokens']
            self.usage.append(usage)
            record.update(status='COMPLETE', **usage, response_model=response.model,
                          finish_reason=response.choices[0].finish_reason)
            if usage['prompt_tokens'] > prompt:
                record['prompt_under_reserved'] = True
                raise BudgetExceeded('measured_prompt_reservation_exceeded')
            if usage['completion_tokens'] > limit:
                record['over_call_budget'] = True
                raise BudgetExceeded('measured_completion_cap_exceeded')
            if usage['prompt_tokens'] + usage['completion_tokens'] > remaining:
                # Total measured usage remains in the ledger on a provider overrun.
                record['over_budget'] = True
                raise BudgetExceeded('measured_total_budget_exhausted')
            return response.choices[0].message.content or ''
        except Exception as exc:
            if record['status'] == 'RUNNING':
                record.update(status='FAILED', error_type=type(exc).__name__, unknown_upper=prompt + limit)
            raise
        finally:
            record['seconds'] = time.monotonic() - started

    def pop_usage(self):
        return self.usage.pop(0)


def verify(output, run_id, model, task, source, index):
    root = Path(os.environ.get('PLC_FUNCTIONAL_ROOT') or ROOT / 'evaluation/functional94')
    work = output / run_id / 'verification' / model / task['task_id'] / str(index)
    work.mkdir(parents=True, exist_ok=False)
    request, response = work / 'request.json', work / 'response.json'
    put(request, dict(root=str(work.resolve()), run_id='evaluation', method='RL-MAR',
        method_slug=model, task=task, source=extract_candidate(source),
        test_vector_sha256='released-exact-trace', compile_timeout=60))
    try:
        process = subprocess.run([sys.executable, str(root / 'scripts/portable_worker.py'),
            '--worker', str(request), '--response', str(response)], capture_output=True,
            text=True, timeout=float(os.getenv('RL_MAR_VERIFIER_TIMEOUT', '120')))
    except subprocess.TimeoutExpired:
        return {'status': 'TIMEOUT', 'checks': [], 'diagnostic': 'verifier outer timeout'}
    if response.is_file():
        return json.loads(response.read_text())
    return {'status': 'ENVIRONMENT_ERROR', 'checks': [],
            'diagnostic': (process.stderr or process.stdout)[-2000:]}


def main():
    from MAR.Routing.policy import FrozenRouter
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=Path(__file__).with_name('route_policy.pt'))
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--model', choices=PFX, action='append')
    parser.add_argument('--budget', type=int, choices=(4096, 8192, 16384), default=16384)
    parser.add_argument('--task-id', action='append')
    parser.add_argument('--routes-only', action='store_true')
    args = parser.parse_args()
    if args.env_file:
        load_env(args.env_file)
    value = args.dataset.read_text()
    tasks = json.loads(value) if value.lstrip().startswith('[') else [json.loads(s) for s in value.splitlines() if s.strip()]
    tasks = select_tasks(tasks, args.task_id)
    models = args.model or list(PFX)
    router = FrozenRouter(args.policy)
    routes = {model: [router.predict(task, model, args.budget) for task in tasks] for model in models}
    # The entire route plan is fixed before any model call or hidden verification.
    run = args.output.resolve() / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    for model in models:
        put(run / f'routes_{model}.json', routes[model])
    put(run / 'settings.json', dict(seed=1818, round=1, budget=args.budget,
        max_call_tokens=2048, corrections=1, temperature=0, models=models,
        task_ids=[t['task_id'] for t in tasks]))
    if args.routes_only:
        print(f'Frozen {len(tasks) * len(models)} routes; no model calls')
        return
    if not os.getenv('RUSTY_BIN'):
        raise ValueError('Set RUSTY_BIN to your RuSTy compiler binary before generation')
    for model in models:
        for task, route in zip(tasks, routes[model]):
            prompt = public_prompt(task)
            assert_prompt_isolated(task, prompt)
            backend = Backend(model)
            checks = []
            def verifier(source):
                result = verify(args.output.resolve(), args.run_id, model, task, source, len(checks))
                checks.append(dict(source=extract_candidate(source), result=result))
                return public_validation(result)
            record = dict(task_id=task['task_id'], model=model, route=route)
            started = time.monotonic()
            try:
                execution = asyncio.run(Execution(route['execution_action'], backend, verifier,
                    max_call_tokens=2048, role_names=tuple(route['roles']),
                    role_instructions=tuple(route['role_instructions'])).run({'query': prompt}))
                record.update(execution=execution, selected_source=extract_candidate(execution['candidate']))
                matching = [v['result'] for v in checks if v['source'] == record['selected_source']]
                record['terminal'] = matching[-1] if matching else {'status': 'MISSING_CANDIDATE'}
            except Exception as exc:
                record.update(error_type=type(exc).__name__, status='FAILED')
            record.update(calls=backend.calls, verification=checks, known_tokens=backend.spent,
                seconds=time.monotonic() - started,
                functional_pass=record.get('terminal', {}).get('status') == 'PASS')
            put(run / 'results' / model / (task['task_id'] + '.json'), record)
    print(f'Completed {len(tasks) * len(models)} workflows in {run}')


if __name__ == '__main__':
    main()
