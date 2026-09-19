#!/usr/bin/env python3
"""Evaluation-only, resumable Functional94 runner.  Never train or mutate policy."""
from __future__ import annotations
import argparse, asyncio, hashlib, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path
import torch
from openai import AsyncOpenAI

try:
    import resource
except ImportError:  # Windows does not provide the Unix resource module.
    resource = None

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "framework"), str(ROOT / "experiments" / "runners")]
from MAR.V2.execution import Execution, TOPOLOGIES
from MAR.V2.policy import ContextualRoutePolicy

IDS = {"gpt-4o": 0, "gpt-4o-mini": 1, "deepseek-v3.2": 2, "qwen3.5-9b": 3, "deepseek-v4-flash": 4, "gpt-5.5": 0}
PFX = {"gpt-4o":"GPT4O", "gpt-4o-mini":"GPT4O_MINI", "deepseek-v3.2":"DEEPSEEK_V32", "qwen3.5-9b":"QWEN35", "deepseek-v4-flash":"DEEPSEEK_V4_FLASH", "gpt-5.5":"GPT55"}
ROUTE_INDEX_SOURCE = {"gpt-5.5": "gpt-4o"}
CK = "e4006a0245bd5947d8a45fc7404db41969cfd368bd57ca8bf208b88323cb546d"
HIDDEN = ("cases", "oracles", "stimulus", "oracle", "expected_output", "input_trace", "expected_internal")
PUBLIC_PROMPT_FIELDS = ("instruction", "instruction_sha256", "entry_point", "pou_interface")
DOMAINS = ("SequentialLogicControl", "ProcessControl", "MotionAndSortingControl", "DataHandlingAndCommunication", "SafetyAndMonitoring")
GENERATOR_ROLE = {"SequentialLogicControl":3, "ProcessControl":5, "MotionAndSortingControl":5, "DataHandlingAndCommunication":5, "SafetyAndMonitoring":3}
BALANCED_GENERATOR_DOMAINS = {"SequentialLogicControl", "ProcessControl", "DataHandlingAndCommunication"}
DOMAIN_TERMS = {
    "SafetyAndMonitoring": ("emergency", "fault", "guard", "hazard", "interlock", "safe", "safety", "trip", "watchdog"),
    "MotionAndSortingControl": ("axis", "conveyor", "drive", "encoder", "jam", "motion", "position", "robot", "sort", "speed", "stepper", "target"),
    "DataHandlingAndCommunication": ("array", "buffer", "communication", "count", "crc", "data", "decode", "encode", "fifo", "message", "modbus", "protocol", "sample"),
    "ProcessControl": ("analog", "flow", "hysteresis", "level", "pid", "pressure", "pump", "tank", "temperature", "valve"),
    "SequentialLogicControl": ("latch", "permit", "sequence", "start", "state machine", "step", "stop"),
}

def peak_rss_bytes():
    return None if resource is None else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def stable_test_hash(task):
    payload = {"task_id":task["task_id"], "cases":task["cases"], "oracles":task["oracles"], "stimulus":task["stimulus"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"); tmp.replace(path)

def load_env(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1); os.environ[key.strip()] = value.strip()
    if not os.getenv("GPT55_MODEL"):
        blocks, block = [], {}
        for line in lines + [""]:
            if not line.strip():
                if block: blocks.append(block); block = {}
            elif not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1); block[key.strip()] = value.strip()
        match = next((item for item in blocks if item.get("MODEL") == "gpt-5.5"), None)
        if match:
            os.environ.update(GPT55_URL=match["URL"], GPT55_KEY=match["KEY"], GPT55_MODEL=match["MODEL"], GPT55_PROVIDER="openai-compatible")

def provider_bindings(models, env_file):
    bindings = {"env_file_sha256": sha(env_file), "models": {}}
    for model in models:
        prefix = PFX[model]
        values = {name: os.getenv(f"{prefix}_{name}", "") for name in ("URL", "KEY", "MODEL", "PROVIDER")}
        if not all(values.values()):
            raise ValueError(f"incomplete provider binding for {prefix}")
        bindings["models"][model] = {
            "provider_prefix": prefix,
            "provider": values["PROVIDER"],
            "base_url": values["URL"],
            "model_id": values["MODEL"],
            "key_configured": bool(values["KEY"]),
            "route_index": IDS[model],
            "route_index_source": ROUTE_INDEX_SOURCE.get(model, model),
            "metadata_sha256": {
                name.lower(): hashlib.sha256(value.encode()).hexdigest()
                for name, value in values.items() if name != "KEY"
            },
        }
    return bindings

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

class Backend:
    def __init__(self, model): self.model = model; self.usage = []
    async def agen(self, messages, max_tokens=None, **_):
        prefix = PFX[self.model]; cfg = {k: os.getenv(f"{prefix}_{k}", "") for k in ("URL", "KEY", "MODEL")}
        if not all(cfg.values()): raise RuntimeError("ENVIRONMENT_ERROR: incomplete provider")
        kwargs = {"model":cfg["MODEL"],"messages":messages,"temperature":0,"max_tokens":max_tokens or 2048}
        if self.model in {"deepseek-v4-flash", "gpt-5.5"}: kwargs["extra_body"] = {"thinking":{"type":"disabled"}}
        try:
            async with AsyncOpenAI(base_url=cfg["URL"], api_key=cfg["KEY"], timeout=float(os.getenv("RL_MAR_API_TIMEOUT", "180"))) as client:
                response = await client.chat.completions.create(**kwargs)
            usage = response.usage
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            self.usage.append({"prompt_tokens":prompt_tokens, "completion_tokens":completion_tokens, "cost":None, "pricing_status":"unavailable", "response_model":str(response.model), "reasoning_present":bool(getattr(response.choices[0].message, "reasoning_content", None))})
            return str(response.choices[0].message.content)
        except Exception as exc:
            raise RuntimeError("API_ERROR:" + type(exc).__name__) from exc
    def pop_usage(self): return self.usage.pop(0) if self.usage else {"prompt_tokens":0,"completion_tokens":0,"cost":None,"pricing_status":"unavailable"}

def infer_domain(task):
    text = " ".join(str(task.get(field, "")) for field in ("category", "title", "instruction")).lower()
    scores = {domain: sum(text.count(term) for term in terms) for domain, terms in DOMAIN_TERMS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "SequentialLogicControl"

def load_role_catalog(domain):
    directory = ROOT / "framework" / "MAR" / "Roles" / domain
    roles = tuple(json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json")))
    names = tuple(role["Name"] for role in roles)
    instructions = tuple(role["Description"] for role in roles)
    if len(names) != 6:
        raise ValueError(f"invalid role catalog for {domain}")
    return names, instructions

def route(policy, encoder, task, model, device, override, candidate_profile):
    text = public_prompt(task); inferred_domain = infer_domain(task)
    domain = torch.zeros(1, len(DOMAINS), device=device); domain[0, DOMAINS.index(inferred_domain)] = 1
    with torch.no_grad(): action = policy.sample(encoder([text]).to(device), domain, torch.tensor([IDS[model]], device=device), deterministic=True)[0].action.to_dict()
    if override == "random": action["topology_index"] = int(hashlib.sha256(task["task_id"].encode()).hexdigest(), 16) % len(TOPOLOGIES)
    elif override and override != "policy": action["topology_index"] = TOPOLOGIES.index(override)
    if candidate_profile == "balanced_domain" and inferred_domain in BALANCED_GENERATOR_DOMAINS:
        action["agent_count"] = 1
        action["role_indices"] = [GENERATOR_ROLE[inferred_domain]]
    return action, inferred_domain

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

def verify(root, run_id, model, task, source):
    source = extract_candidate(source)
    verifier_root = Path(os.environ["PLC_FUNCTIONAL_ROOT"])
    private_task = dict(task)  # verifier alone receives hidden cases/oracles/stimulus.
    with tempfile.TemporaryDirectory(prefix="rlmar-v2-verifier-") as directory:
        req = Path(directory) / "request.json"; response = Path(directory) / "response.json"
        put(req, {"root":str(verifier_root),"run_id":"rlmar_v2_"+run_id,"method":"RL-MAR-v2","method_slug":model,"task":private_task,"source":source,"test_vector_sha256":stable_test_hash(task),"compile_timeout":int(os.getenv("PLC_FUNCTIONAL_COMPILE_TIMEOUT", "120"))})
        try:
            process = subprocess.run([sys.executable, str(verifier_root / "scripts/run_functional94.py"), "--worker", str(req), "--response", str(response)], text=True, capture_output=True, timeout=float(os.getenv("RL_MAR_VERIFIER_TIMEOUT", "180")))
        except (OSError, subprocess.TimeoutExpired) as exc:
            return public_validation({"status":"ENVIRONMENT_ERROR", "diagnostic":type(exc).__name__})
        if response.is_file(): return public_validation(json.loads(response.read_text()))
        diagnostic = (process.stderr or process.stdout or f"worker exited without response ({process.returncode})")[-2000:]
        return public_validation({"status":"ENVIRONMENT_ERROR", "diagnostic":diagnostic})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True); ap.add_argument("--policy", type=Path, required=True); ap.add_argument("--env-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--run-id", required=True); ap.add_argument("--model", action="append", choices=IDS)
    ap.add_argument("--preflight", action="store_true"); ap.add_argument("--limit", type=int); ap.add_argument("--task-id", action="append"); ap.add_argument("--disable-quality-controller", action="store_true")
    ap.add_argument("--route", choices=("policy", "random") + TOPOLOGIES, default="policy"); ap.add_argument("--disable-hard-budget", action="store_true")
    ap.add_argument("--candidate-profile", choices=("quality", "balanced_domain"), required=True)
    ap.add_argument("--evolution-archive", type=Path, help="read-only archive materialized in preflight metadata only")
    args = ap.parse_args(); load_env(args.env_file)
    settings_path = Path(os.environ["RL_MAR_FIVE_MODEL_SETTINGS"])
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    from MAR.LLM.llm_embedding import SentenceEncoder
    if sha(args.policy) != CK: raise ValueError("checkpoint hash mismatch")
    verifier_root = Path(os.environ["PLC_FUNCTIONAL_ROOT"])
    official_dir = verifier_root / "vendor" / "Agents4PLC_release" / "benchmark_v2"
    validation_command = [sys.executable, str(verifier_root / "scripts/validate_manifest.py"), str(args.dataset)]
    if official_dir.is_dir():
        validation_command += ["--official-dir", str(official_dir)]
    check = subprocess.run(validation_command, capture_output=True)
    tasks = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if check.returncode or len(tasks) != 94 or len({t["task_id"] for t in tasks}) != 94 or any(t.get("schema_version") != "1.3" for t in tasks): raise ValueError("Functional94 v1.3 freeze validation failed")
    tasks = select_tasks(tasks, args.task_id)
    if args.limit: tasks = tasks[:args.limit]
    models = args.model or list(IDS); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ContextualRoutePolicy().to(device); policy.load_state_dict(torch.load(args.policy, map_location=device, weights_only=True).get("state_dict", torch.load(args.policy, map_location=device, weights_only=True))); policy.eval()
    init_wall = time.monotonic(); print("stage=encoder_load status=start", flush=True)
    encoder = SentenceEncoder(device=device); before = sha(args.policy)
    print(f"stage=encoder_load status=done wall_seconds={time.monotonic()-init_wall:.3f}", flush=True)
    routes = {m: route(policy, encoder, tasks[0], m, device, args.route, args.candidate_profile)[0] for m in models}
    proof = {"status":"PASS","dataset_sha256":sha(args.dataset),"checkpoint_sha256":before,"candidate_profile":args.candidate_profile,"provider_bindings":provider_bindings(models, args.env_file),"routes":routes,"overrides":{"quality":args.disable_quality_controller,"hard_budget":args.disable_hard_budget,"route":args.route},"evolution_archive":sha(args.evolution_archive) if args.evolution_archive else None}
    if args.preflight:
        # Explicit smoke only; ordinary/resume runs never issue a non-task API request.
        asyncio.run(Backend(models[0]).agen([{"role":"user","content":"Reply exactly OK"}], max_tokens=8)); print(json.dumps(proof, sort_keys=True)); return
    root = args.output / args.run_id
    contract_path = root / "preflight.json"
    if contract_path.is_file():
        prior = json.loads(contract_path.read_text())
        if prior != proof: raise ValueError("resume contract mismatch")
    else: put(contract_path, proof)
    for model in models:
        for task in tasks:
            out = root / "results" / model / f"{task['task_id']}.json"
            if out.is_file(): continue
            prompt = public_prompt(task); assert_prompt_isolated(task, prompt); wall, cpu = time.monotonic(), time.process_time()
            print(f"stage=task status=start model={model} task_id={task['task_id']}", flush=True)
            action, inferred_domain = route(policy, encoder, task, model, device, args.route, args.candidate_profile); execution_action = budgeted_action(action, settings); record = {"run_id":args.run_id,"task_id":task["task_id"],"model":model,"candidate_profile":args.candidate_profile,"inferred_domain":inferred_domain,"route_action":action,"execution_action":execution_action,"public_prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest()}
            try:
                role_names, role_instructions = load_role_catalog(inferred_domain)
                run = asyncio.run(Execution(execution_action, Backend(model), lambda code: verify(root,args.run_id,model,task,code), hard_budget=not args.disable_hard_budget, quality=not args.disable_quality_controller, max_call_tokens=int(settings["max_call_tokens"]), role_names=role_names, role_instructions=role_instructions).run({"query":prompt}))
                record.update({"execution":run,"generated_code":run.get("candidate", ""),"validation":run["validation"],"raw_reward":run.get("raw_reward")})
            except Exception as exc: record.update({"validation":{"status":"API_ERROR" if "API_ERROR" in str(exc) else "ENVIRONMENT_ERROR","diagnostic":str(exc)[:300]},"raw_reward":None})
            record["resources"] = {"wall_seconds":time.monotonic()-wall,"cpu_seconds":time.process_time()-cpu,"peak_rss_bytes":peak_rss_bytes()}
            put(out, record)
            print(f"stage=task status=done model={model} task_id={task['task_id']} validation={record['validation']['status']} wall_seconds={record['resources']['wall_seconds']:.3f}", flush=True)
    after = sha(args.policy)
    if after != before: raise RuntimeError("checkpoint changed during evaluation")
    put(root / "status.json", {"run_id":args.run_id,"completed":len(list((root/"results").glob("*/*.json"))),"expected":len(tasks)*len(models),"checkpoint_before":before,"checkpoint_after":after})
if __name__ == "__main__": main()
