from __future__ import annotations

import argparse
from dataclasses import replace
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import torch

def discover_root(source: Path) -> Path:
    """Find the nearest import root containing framework/MAR."""
    for candidate in (source.resolve().parent, *source.resolve().parents):
        if (candidate / "framework" / "MAR").is_dir():
            return candidate
    raise RuntimeError(f"could not locate framework/MAR from {source}")


ROOT = discover_root(Path(__file__))
FRAMEWORK = ROOT / "framework"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FRAMEWORK))
sys.path.insert(0, str(ROOT / "experiments"))

from MAR.Graph.graph import DynamicEngineeringTopology as Graph
from MAR.LLM.llm_embedding import SentenceEncoder
from MAR.LLM import gpt_chat as _llm_registrations
from MAR.LLM.price import MODEL_PRICE, get_currency, pricing_model_name
from MAR.Utils.globals import CompletionTokens, Cost, PromptTokens
from MAR.Utils.utils import get_kwargs
from MAR.Core.dataset import DOMAINS, load_reward_records, stable_sha256
from MAR.Core.outcomes import classify_outcome
from MAR.Core.policy import ContextualRoutePolicy, RouteAction
from MAR.Core.resources import ResourceMonitor
from MAR.Core.request_budget import BudgetExceeded, RequestBudget, request_budget
from MAR.Prompts.post_process_plc import extract_st_code, normalize_st

try:
    from stage2_selection import resolve_selection as resolve_stage2_selection
except ImportError:
    resolve_stage2_selection = None

try:
    from stage3_selection import resolve_selection as resolve_stage3_selection
except ImportError:
    resolve_stage3_selection = None


MODEL_IDS = {
    "gpt-4o": 0,
    "gpt-4o-mini": 1,
    "deepseek-ai/DeepSeek-V3.2": 2,
    "qwen3.5-9b": 3,
    "deepseek-v4-flash": 4,
}

PROVIDER_PREFIXES = {
    "gpt-4o": "GPT4O",
    "gpt-4o-mini": "GPT4O_MINI",
    "deepseek-ai/DeepSeek-V3.2": "DEEPSEEK_V32",
    "qwen3.5-9b": "QWEN35",
    "deepseek-v4-flash": "DEEPSEEK_V4_FLASH",
}

DOMAIN_SPECIALIST_ROLE = {
    "SequentialLogicControl": 4,
    "ProcessControl": 3,
    "MotionAndSortingControl": 3,
    "DataHandlingAndCommunication": 2,
    "SafetyAndMonitoring": 5,
}

GENERATOR_ROLE = {
    "SequentialLogicControl": 3,
    "ProcessControl": 5,
    "MotionAndSortingControl": 5,
    "DataHandlingAndCommunication": 5,
    "SafetyAndMonitoring": 3,
}

BALANCED_GENERATOR_DOMAINS = {
    "SequentialLogicControl",
    "ProcessControl",
    "DataHandlingAndCommunication",
}

THINKING_CONTROL_PREFIXES = {"DEEPSEEK_V32", "QWEN35", "DEEPSEEK_V4_FLASH"}


def activate_provider(model: str, prefix: str) -> dict:
    expected = PROVIDER_PREFIXES.get(model)
    if prefix != expected:
        raise ValueError(f"provider prefix {prefix!r} does not match logical model {model!r}")
    values = {name: os.environ.get(f"{prefix}_{name}", "").strip() for name in ("URL", "KEY", "MODEL")}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"provider prefix {prefix} is missing variables: {missing}")
    os.environ["URL"] = values["URL"]
    os.environ["KEY"] = values["KEY"]
    os.environ["RL_MAR_API_MODEL"] = values["MODEL"]
    os.environ["RL_MAR_USE_AUDITED_PROVIDER_TRANSPORT"] = "1"
    os.environ["RL_MAR_SUPPORTS_THINKING_CONTROL"] = "1" if prefix in THINKING_CONTROL_PREFIXES else "0"
    return {
        "prefix": prefix,
        "provider": os.environ.get(f"{prefix}_PROVIDER", "unspecified"),
        "api_model": values["MODEL"],
        "url_sha256": hashlib.sha256(values["URL"].encode()).hexdigest(),
    }


def resolve_provider(model: str, prefix: str | None, allow_legacy: bool) -> dict:
    if prefix:
        return activate_provider(model, prefix)
    if not allow_legacy:
        raise ValueError("--provider-prefix is required for auditable model comparisons")
    return {
        "prefix": None,
        "provider": "legacy_generic_environment",
        "api_model": model,
        "url_sha256": hashlib.sha256(os.environ.get("URL", "").encode()).hexdigest(),
        "eligible_for_formal_comparison": False,
    }


def load_roles(domain: str) -> list[dict]:
    directory = FRAMEWORK / "MAR" / "Roles" / domain
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]


def root_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def framework_path(path: Path) -> Path:
    return path if path.is_absolute() else FRAMEWORK / path


def selected_validation(attempts: list[dict]) -> dict:
    selected = next((attempt for attempt in reversed(attempts) if attempt.get("selected")), None)
    if selected and selected.get("validation_summary"):
        return selected["validation_summary"]
    if selected and selected.get("validation_result"):
        return public_validation_summary(selected["validation_result"])
    return {"status": "ENVIRONMENT_ERROR", "diagnostic": "no selected executable validation", "checks": []}


def unique_samples(policy, embedding, domain_probs, model_id, count, *, max_agents=None, max_rounds=None, min_token_budget=None, repair_budget=None, deterministic=False):
    samples, actions = [], set()
    attempts = 0
    while len(samples) < count and attempts < count * 200:
        sample = policy.sample(
            embedding, domain_probs, model_id,
            deterministic=deterministic, repair_budget=repair_budget,
        )[0]
        if max_agents is not None and sample.action.agent_count > max_agents:
            attempts += 1
            continue
        if max_rounds is not None and sample.action.rounds > max_rounds:
            attempts += 1
            continue
        if min_token_budget is not None and sample.action.token_budget < min_token_budget:
            attempts += 1
            continue
        key = json.dumps(sample.action.to_dict(), sort_keys=True)
        if key not in actions:
            actions.add(key)
            samples.append(sample)
        attempts += 1
    if len(samples) != count:
        raise RuntimeError(f"could not sample {count} unique routes")
    return samples


def select_records(dataset: Path, split: str, count: int, task_manifest: Path | None = None, task_index: int | None = None):
    if task_manifest is None:
        if task_index is not None: raise ValueError("--task-index requires --task-manifest")
        return load_reward_records(dataset, allowed_splits={split})[:count]
    manifest = json.loads(task_manifest.read_text(encoding="utf-8"))
    resolvers = {
        "rlmar-stage2-screen-1.0": resolve_stage2_selection,
        "rlmar-stage3-screen-1.0": resolve_stage3_selection,
    }
    schema = manifest.get("schema_version") if isinstance(manifest, dict) else None
    if schema not in resolvers:
        raise ValueError(f"unsupported task manifest schema: {schema}")
    resolver = resolvers[schema]
    if resolver is None:
        raise RuntimeError(f"selection resolver is unavailable for schema: {schema}")
    records = resolver(dataset, task_manifest)
    if split != "evolution_dev":
        raise ValueError("frozen task manifests require split evolution_dev")
    if task_index is not None:
        if count != 1: raise ValueError("--task-index requires --tasks 1")
        if not 0 <= task_index < len(records): raise ValueError("--task-index is outside frozen manifest")
        return [records[task_index]]
    if count != len(records):
        raise ValueError(f"--tasks must equal frozen manifest size ({len(records)})")
    return records


def candidate_provenance(response: str, extraction_profile: str, normalization_profile: str) -> dict:
    extracted = extract_st_code(response, extraction_profile)
    normalized, transformation_ids = normalize_st(extracted, normalization_profile)
    return {
        "raw_response": response,
        "response_sha256": stable_sha256(response),
        "extracted_candidate_sha256": stable_sha256(extracted),
        "normalized_candidate_sha256": stable_sha256(normalized),
        "transformation_ids": transformation_ids,
    }


def public_validation_summary(validation: dict) -> dict:
    checks = validation.get("checks") or []
    status = validation.get("status", "ENVIRONMENT_ERROR")
    summary = {
        "status": status,
        "stages_completed": validation.get("stages_completed", 0),
        "total_stages": validation.get("total_stages", 5),
        "checks_total": validation.get("checks_total", len(checks)),
        "checks_passed": validation.get(
            "checks_passed",
            sum(check.get("status") == "PASS" for check in checks),
        ),
    }
    if status in {"COMPILE_FAIL", "COMPILE_ERROR"}:
        diagnostic = str(validation.get("diagnostic") or "")
        summary["diagnostic"] = re.sub(
            r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", diagnostic
        )[-4000:]
    return summary


def public_attempt_summaries(attempts: list[dict]) -> list[dict]:
    summaries = []
    for attempt in attempts:
        validation = attempt.get("validation_summary")
        if validation is None:
            validation = public_validation_summary(attempt.get("validation_result") or {})
        summaries.append({
            "attempt": attempt.get("attempt"),
            "code_sha256": attempt.get("code_sha256"),
            "status": attempt.get("status"),
            "selected": bool(attempt.get("selected")),
            "validation_seconds": attempt.get("validation_seconds"),
            "stages_completed": validation.get("stages_completed", 0),
            "total_stages": validation.get("total_stages", 5),
            "checks_total": validation.get("checks_total", 0),
            "checks_passed": validation.get("checks_passed", 0),
        })
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect verifier-grounded RewardPLC route trajectories")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "evolution_dev", "selection"], default="train")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider-prefix", choices=tuple(PROVIDER_PREFIXES.values()))
    parser.add_argument("--allow-legacy-generic-provider", action="store_true")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, default=ROOT / "framework/models/all-MiniLM-L6-v2")
    parser.add_argument("--prompt-file", default=str(FRAMEWORK / "MAR/Roles/FinalNode/plc.json"))
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--role-profile", choices=("policy", "domain_specialist", "generator_only", "balanced_domain"), default="policy")
    parser.add_argument("--repair-budget", type=int)
    parser.add_argument("--repair-mode", choices=("compile_only", "all_stages"), default="all_stages")
    parser.add_argument("--max-stagnant-repairs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--compiler-diagnostic-limit", type=int, choices=(1000, 2000, 4000), default=4000)
    parser.add_argument("--repair-prompt-profile", choices=("baseline", "iec"), default="iec")
    parser.add_argument("--extraction-profile", default="baseline")
    parser.add_argument("--normalization-profile", default="baseline")
    parser.add_argument("--max-route-seconds", type=float, default=180.0)
    parser.add_argument("--max-model-calls", type=int, default=12)
    parser.add_argument("--max-call-tokens", type=int, default=2048)
    parser.add_argument("--max-agents", type=int)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--min-token-budget", type=int)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--rusty", type=Path)
    parser.add_argument("--behavior-policy-output", type=Path)
    args = parser.parse_args()
    if args.model not in MODEL_IDS:
        raise ValueError(f"model must be one of {sorted(MODEL_IDS)}")
    provider = resolve_provider(
        args.model, args.provider_prefix, args.allow_legacy_generic_provider
    )
    if args.deterministic and args.k != 1:
        parser.error("--deterministic requires --k 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.dataset = root_path(args.dataset)
    args.output = root_path(args.output)
    if args.task_manifest:
        args.task_manifest = root_path(args.task_manifest)
    args.encoder = root_path(args.encoder).resolve()
    os.environ["PLC_REWARDPLC_PATH"] = str(args.dataset)
    if args.rusty:
        os.environ["PLC_RUSTY_PATH"] = str(args.rusty.resolve())
    if args.policy:
        args.policy = root_path(args.policy)
    args.prompt_file = str(root_path(Path(args.prompt_file)))
    records = select_records(args.dataset, args.split, args.tasks, args.task_manifest, args.task_index)
    os.environ["RL_MAR_EXTRACTION_PROFILE"] = args.extraction_profile
    os.environ["RL_MAR_NORMALIZATION_PROFILE"] = args.normalization_profile
    extract_st_code("", args.extraction_profile)
    normalize_st("", args.normalization_profile)
    if args.disable_thinking:
        os.environ["RL_MAR_DISABLE_THINKING"] = "1"
    os.chdir(FRAMEWORK)
    encoder = SentenceEncoder(model_path=args.encoder, device=device)
    policy = ContextualRoutePolicy().to(device)
    if args.policy:
        checkpoint = torch.load(args.policy, map_location=device, weights_only=True)
        policy.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
        behavior_policy_path = args.policy
    else:
        behavior_policy_path = root_path(args.behavior_policy_output or args.output.with_suffix(".behavior_policy.pt"))
        behavior_policy_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": policy.state_dict(), "seed": args.seed}, behavior_policy_path)
    behavior_policy_sha256 = stable_sha256(behavior_policy_path.read_bytes())
    policy.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            existing.add((row["task_id"], row["model"], row["candidate_index"]))
    with args.output.open("a", encoding="utf-8") as stream:
        for task_index, record in enumerate(records):
            embedding = encoder([record.specification]).detach()
            domain_probs = torch.zeros(1, len(DOMAINS), device=device)
            domain_probs[0, DOMAINS.index(record.domain)] = 1.0
            model_id = torch.tensor([MODEL_IDS[args.model]], device=device)
            with torch.no_grad():
                samples = unique_samples(
                    policy, embedding, domain_probs, model_id, args.k,
                    max_agents=args.max_agents, max_rounds=args.max_rounds,
                    min_token_budget=args.min_token_budget,
                    repair_budget=args.repair_budget,
                    deterministic=args.deterministic,
                )
            for candidate_index, sample in enumerate(samples):
                key = (record.task_id, args.model, candidate_index)
                if key in existing:
                    continue
                action = sample.action
                policy_action = action
                if args.role_profile == "domain_specialist":
                    specialist = DOMAIN_SPECIALIST_ROLE[record.domain]
                    role_indices = tuple(dict.fromkeys((*action.role_indices, specialist)))
                    action = replace(action, role_indices=role_indices, agent_count=len(role_indices))
                elif args.role_profile == "generator_only" or (
                    args.role_profile == "balanced_domain" and record.domain in BALANCED_GENERATOR_DOMAINS
                ):
                    role_indices = (GENERATOR_ROLE[record.domain],)
                    action = replace(action, role_indices=role_indices, agent_count=1)
                roles = load_roles(record.domain)
                selected_roles = [roles[index]["Name"] for index in action.role_indices if index < len(roles)]
                topology = policy.space.topologies[action.topology_index]
                kwargs = get_kwargs(topology, len(selected_roles))
                kwargs["num_rounds"] = action.rounds
                previous = (Cost.instance().value, PromptTokens.instance().value, CompletionTokens.instance().value)
                os.environ["RL_MAR_MAX_REPAIRS"] = str(action.repair_budget)
                os.environ["RL_MAR_REPAIR_MODE"] = args.repair_mode
                os.environ["RL_MAR_MAX_STAGNANT_REPAIRS"] = str(args.max_stagnant_repairs)
                os.environ["RL_MAR_COMPILER_DIAGNOSTIC_LIMIT"] = str(args.compiler_diagnostic_limit)
                os.environ["RL_MAR_REPAIR_PROMPT_PROFILE"] = args.repair_prompt_profile
                os.environ["LLM_MAX_TOKENS"] = str(min(action.token_budget, args.max_call_tokens))
                started = time.time()
                attempts = []
                response = ""
                error = None
                runtime_budget = RequestBudget(args.max_model_calls, action.token_budget, args.max_route_seconds)
                with ResourceMonitor() as monitor, request_budget(runtime_budget):
                    try:
                        graph = Graph(
                            domain=record.domain,
                            llm_names=[args.model] * len(selected_roles),
                            agent_names=selected_roles,
                            decision_method="FinalRefer",
                            prompt_file=args.prompt_file,
                            reasoning_name=topology,
                            **kwargs,
                        )
                        response = graph.run(
                            inputs={"query": record.specification, "task_id": record.task_id},
                            num_rounds=action.rounds,
                            use_async=True,
                            max_concurrent=min(len(selected_roles), 3),
                        )[0][0]
                        attempts = getattr(graph.decision_node, "attempt_records", [])
                    except BudgetExceeded as exception:
                        error = f"BudgetExceeded: {exception}"
                    except Exception as exception:
                        error = type(exception).__name__
                resources = monitor.result()
                if attempts:
                    validation = selected_validation(attempts)
                elif error and error.startswith("BudgetExceeded:"):
                    validation = {
                        "status": "BUDGET_EXHAUSTED",
                        "diagnostic": runtime_budget.stop_reason or "request budget exhausted",
                        "checks": [],
                    }
                else:
                    validation = {"status": "ENVIRONMENT_ERROR", "diagnostic": error or "no attempts", "checks": []}
                outcome = classify_outcome(validation)
                public_validation = public_validation_summary(validation)
                public_attempts = public_attempt_summaries(attempts)
                candidate_audit = candidate_provenance(
                    response, args.extraction_profile, args.normalization_profile
                )
                costs = {
                    "monetary_cost": Cost.instance().value - previous[0],
                    "prompt_tokens": PromptTokens.instance().value - previous[1],
                    "completion_tokens": CompletionTokens.instance().value - previous[2],
                    "pricing_status": "available" if pricing_model_name(args.model) in MODEL_PRICE else "unavailable",
                    "currency": get_currency(args.model) if pricing_model_name(args.model) in MODEL_PRICE else None,
                }
                if costs["pricing_status"] == "unavailable":
                    costs["monetary_cost"] = None
                row = {
                    "schema_version": "rlmar-trajectory-2.0",
                    "group_id": f"{record.task_id}:{args.model}",
                    "task_id": record.task_id,
                    "family_id": record.family_id,
                    "domain": record.domain,
                    "split": record.split,
                    "specification": record.specification,
                    "model": args.model,
                    "model_id": MODEL_IDS[args.model],
                    "provider": provider,
                    "candidate_index": candidate_index,
                    "route_action": action.to_dict(),
                    "policy_route_action": policy_action.to_dict(),
                    "route_override": None if args.role_profile == "policy" else {"role_profile": args.role_profile},
                    "old_log_probability": float(sample.log_probability.item()),
                    "reference_log_probability": float(sample.log_probability.item()),
                    "policy_eligible": outcome.policy_eligible,
                    "raw_reward": outcome.raw_reward,
                    "validation": public_validation,
                    "attempts": public_attempts,
                    "costs": costs,
                    "resources": resources.to_dict(),
                    "runtime_budget": runtime_budget.snapshot(),
                    **candidate_audit,
                    "error": error,
                    "started_at": started,
                    "elapsed_seconds": time.time() - started,
                    "policy_seed": args.seed,
                    "behavior_policy_sha256": behavior_policy_sha256,
                    "selection_manifest_sha256": stable_sha256(args.task_manifest.read_bytes()) if args.task_manifest else None,
                    "prompt_sha256": stable_sha256(framework_path(Path(args.prompt_file)).read_bytes()),
                    "collector_sha256": stable_sha256(Path(__file__).read_bytes()),
                    "post_process_sha256": stable_sha256((FRAMEWORK / "MAR" / "Prompts" / "post_process_plc.py").read_bytes()),
                    "compiler_sha256": stable_sha256(args.rusty.resolve().read_bytes()) if args.rusty else None,
                    "extraction_profile": args.extraction_profile,
                    "normalization_profile": args.normalization_profile,
                    "route_constraints": {"repair_budget": [args.repair_budget]} if args.repair_budget is not None else {},
                    "repair_control": {
                        "mode": args.repair_mode,
                        "max_stagnant_retries": args.max_stagnant_repairs,
                        "compiler_diagnostic_limit": args.compiler_diagnostic_limit,
                        "prompt_profile": args.repair_prompt_profile,
                    },
                    "thinking_disabled": args.disable_thinking,
                    "deterministic_route": args.deterministic,
                    "dataset_sha256": stable_sha256(args.dataset.read_text(encoding="utf-8")),
                }
                stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                stream.flush()
                print(json.dumps({"task": record.task_id, "candidate": candidate_index, "status": validation["status"], "tokens": costs["prompt_tokens"] + costs["completion_tokens"], "seconds": row["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
