#!/usr/bin/env python3
"""Run the frozen balanced candidate on Agents4PLC-23 without verifier leakage."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "framework"), str(ROOT / "experiments" / "runners")]

from MAR.Core.policy import ContextualRoutePolicy
from evaluate_five_models import (
    BALANCED_GENERATOR_DOMAINS,
    CK,
    DOMAINS,
    GENERATOR_ROLE,
    IDS,
    PFX,
    Backend,
    budgeted_action,
    extract_candidate,
    infer_domain,
    load_env,
    load_role_catalog,
    put,
    sha,
)
from route_execution_adapter import Execution


def public_prompt(task: dict) -> str:
    instruction = str(task.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("missing instruction")
    return (
        "Implement the following PLC requirement in IEC 61131-3 Structured Text.\n\n"
        f"{instruction}\n\n"
        "Return only complete IEC Structured Text. Formal properties and verifier results are unavailable."
    )


def assert_prompt_isolated(task: dict, prompt: str) -> None:
    if prompt != public_prompt({"instruction": task.get("instruction")}):
        raise AssertionError("prompt depends on fields outside instruction")


def route(policy, encoder, prompt: str, task: dict, model: str, device) -> tuple[dict, str]:
    domain_name = infer_domain({"instruction": task["instruction"]})
    domain = torch.zeros(1, len(DOMAINS), device=device)
    domain[0, DOMAINS.index(domain_name)] = 1
    with torch.no_grad():
        action = policy.sample(
            encoder([prompt]).to(device),
            domain,
            torch.tensor([IDS[model]], device=device),
            deterministic=True,
        )[0].action.to_dict()
    if domain_name in BALANCED_GENERATOR_DOMAINS:
        action["agent_count"] = 1
        action["role_indices"] = [GENERATOR_ROLE[domain_name]]
    return action, domain_name


def public_validation(result: dict) -> dict:
    if result.get("comp_pass") and result.get("verif_pass"):
        return {"status": "PASS", "compile_pass": True, "verification_pass": True, "stages_completed": 2, "checks_passed": int(result.get("num_properties_passed", 0)), "diagnostic": "PASS"}
    if not result.get("comp_pass"):
        diagnostic = "Compilation failed."
        stderr_path = result.get("rusty_stderr_path")
        if stderr_path and Path(stderr_path).is_file():
            diagnostic = Path(stderr_path).read_text(encoding="utf-8", errors="replace")[-3000:]
        return {"status": "COMPILE_FAIL", "compile_pass": False, "verification_pass": False, "stages_completed": 0, "checks_passed": 0, "diagnostic": diagnostic}
    return {
        "status": "FUNCTION_FAIL",
        "compile_pass": True,
        "verification_pass": False,
        "stages_completed": 1,
        "checks_passed": int(result.get("num_properties_passed", 0)),
        "diagnostic": "Compiled candidate failed formal verification.",
    }


def verify(task: dict, source: str, args, validation_index: int) -> dict:
    source = extract_candidate(source)
    with tempfile.TemporaryDirectory(prefix="rlmar-agents4plc-") as directory:
        work = Path(directory)
        outputs = work / "outputs"
        outputs.mkdir()
        (outputs / f"{task['task_id']}.st").write_text(source + "\n", encoding="utf-8")
        manifest = work / "manifest.jsonl"
        manifest.write_text(json.dumps(task, ensure_ascii=False) + "\n", encoding="utf-8")
        evaluation = work / "evaluation.jsonl"
        command = [
            sys.executable,
            str(args.evaluator),
            "--manifest", str(manifest),
            "--outputs", str(outputs),
            "--out", str(evaluation),
            "--tool_log_dir", str(args.tool_log_dir / task["task_id"] / f"validation_{validation_index}"),
            "--rusty_bin", str(args.rusty_bin),
            "--plcverif_bin", str(args.plcverif_bin),
            "--nuxmv_bin", str(args.nuxmv_bin),
        ]
        try:
            process = subprocess.run(command, text=True, capture_output=True, timeout=args.verifier_timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"status": "ENVIRONMENT_ERROR", "compile_pass": False, "verification_pass": False, "stages_completed": 0, "checks_passed": 0, "diagnostic": type(exc).__name__}
        if process.returncode or not evaluation.is_file():
            return {"status": "ENVIRONMENT_ERROR", "compile_pass": False, "verification_pass": False, "stages_completed": 0, "checks_passed": 0, "diagnostic": "Agents4PLC evaluator failed"}
        rows = [json.loads(line) for line in evaluation.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 1 or rows[0].get("task_id") != task["task_id"]:
            return {"status": "ENVIRONMENT_ERROR", "compile_pass": False, "verification_pass": False, "stages_completed": 0, "checks_passed": 0, "diagnostic": "Agents4PLC evaluator contract mismatch"}
        return public_validation(rows[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", choices=IDS, required=True)
    parser.add_argument("--provider-prefix", choices=sorted(set(PFX.values())), required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--rusty-bin", type=Path, required=True)
    parser.add_argument("--plcverif-bin", type=Path, required=True)
    parser.add_argument("--nuxmv-bin", type=Path, required=True)
    parser.add_argument("--tool-log-dir", type=Path, required=True)
    parser.add_argument("--verifier-timeout", type=float, default=1200)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    load_env(args.env_file)
    expected_prefix = PFX[args.model]
    if args.provider_prefix != expected_prefix:
        raise ValueError(f"provider prefix mismatch for {args.model}")
    provider = {
        name: os.getenv(f"{expected_prefix}_{name}", "")
        for name in ("URL", "KEY", "MODEL", "PROVIDER")
    }
    if not all(provider.values()):
        raise ValueError(f"incomplete provider binding for {expected_prefix}")
    if sha(args.policy) != CK:
        raise ValueError("checkpoint hash mismatch")
    tasks = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tasks) != 23 or len({task["task_id"] for task in tasks}) != 23:
        raise ValueError("Agents4PLC manifest must contain 23 unique tasks")
    if args.limit:
        tasks = tasks[: args.limit]
    settings = json.loads(Path(os.environ["RL_MAR_FIVE_MODEL_SETTINGS"]).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ContextualRoutePolicy().to(device)
    checkpoint = torch.load(args.policy, map_location=device, weights_only=True)
    policy.load_state_dict(checkpoint.get("state_dict", checkpoint))
    policy.eval()
    from MAR.LLM.llm_embedding import SentenceEncoder
    encoder = SentenceEncoder(device=device)

    root = args.output / args.run_id
    contract = {
        "status": "PASS",
        "manifest_sha256": sha(args.manifest),
        "checkpoint_sha256": sha(args.policy),
        "model": args.model,
        "provider_prefix": expected_prefix,
        "provider_metadata_sha256": {
            name.lower(): hashlib.sha256(value.encode()).hexdigest()
            for name, value in provider.items()
        },
        "candidate_profile": "balanced_domain",
        "prompt_fields": ["instruction"],
    }
    contract_path = root / "preflight.json"
    if contract_path.is_file() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("resume contract mismatch")
    if not contract_path.is_file():
        put(contract_path, contract)

    for task in tasks:
        out = root / "results" / args.model / f"{task['task_id']}.json"
        if out.is_file():
            continue
        prompt = public_prompt(task)
        assert_prompt_isolated(task, prompt)
        action, domain = route(policy, encoder, prompt, task, args.model, device)
        execution_action = budgeted_action(action, settings)
        counter = {"value": 0}

        def validate(source: str) -> dict:
            counter["value"] += 1
            return verify(task, source, args, counter["value"])

        started_wall, started_cpu = time.monotonic(), time.process_time()
        record = {
            "run_id": args.run_id,
            "task_id": task["task_id"],
            "difficulty": task.get("difficulty"),
            "model": args.model,
            "candidate_profile": "balanced_domain",
            "inferred_domain": domain,
            "route_action": action,
            "execution_action": execution_action,
            "public_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        try:
            role_names, role_instructions = load_role_catalog(domain)
            run = asyncio.run(
                Execution(
                    execution_action,
                    Backend(args.model),
                    validate,
                    hard_budget=True,
                    quality=True,
                    max_call_tokens=int(settings["max_call_tokens"]),
                    role_names=role_names,
                    role_instructions=role_instructions,
                ).run({"query": prompt})
            )
            record.update({"execution": run, "generated_code": run.get("candidate", ""), "validation": run["validation"], "raw_reward": run.get("raw_reward")})
        except Exception as exc:
            status = "API_ERROR" if "API_ERROR" in str(exc) else "ENVIRONMENT_ERROR"
            record.update({"validation": {"status": status, "compile_pass": False, "verification_pass": False, "diagnostic": type(exc).__name__}, "raw_reward": None})
        record["resources"] = {
            "wall_seconds": time.monotonic() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
        put(out, record)
        generated = root / "generated_st" / f"{task['task_id']}.st"
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(extract_candidate(record.get("generated_code", "")) + "\n", encoding="utf-8")
        print(f"task_id={task['task_id']} status={record['validation']['status']}", flush=True)

    result_files = list((root / "results" / args.model).glob("*.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_files]
    completed = len(results)
    compile_passes = sum(bool(row.get("validation", {}).get("compile_pass")) for row in results)
    verification_passes = sum(bool(row.get("validation", {}).get("verification_pass")) for row in results)
    put(root / "status.json", {
        "run_id": args.run_id,
        "model": args.model,
        "completed": completed,
        "expected": len(tasks),
        "compile_passes": compile_passes,
        "verification_passes": verification_passes,
        "compile_rate": compile_passes / len(tasks) if tasks else 0.0,
        "end_to_end_verification_rate": verification_passes / len(tasks) if tasks else 0.0,
        "conditional_verification_rate": verification_passes / compile_passes if compile_passes else 0.0,
        "checkpoint_sha256": sha(args.policy),
    })


if __name__ == "__main__":
    main()
