#!/usr/bin/env python3
"""
Agents4PLC v1 23-task evaluation script for generated ST programs.
Runs rusty (compilation check) + PLCverif/nuXmv (formal verification) on generated ST files.

Usage:
  python3 scripts/eval_agents4plc23_rdmar.py \
    --manifest data/agents4plc_23_manifest.jsonl \
    --outputs generated_st \
    --out results/evaluation.jsonl \
    --tool_log_dir results/tool_logs \
    --rusty_bin tools/rusty/target/debug/plc \
    --plcverif_bin tools/plcverif/plcverif-cli \
    --nuxmv_bin tools/nuxmv/bin/nuXmv
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time
from pathlib import Path


def load_manifest(path):
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    return tasks


def find_st_file(task_id, outputs_dir):
    """Find .st file for a given task_id. Tries exact match first, then fuzzy."""
    st_dir = Path(outputs_dir)
    # Exact match
    exact = st_dir / f"{task_id}.st"
    if exact.exists():
        return exact
    # Case-insensitive
    for f in st_dir.glob("*.st"):
        if f.stem.lower() == task_id.lower():
            return f
    # Contains match
    for f in st_dir.glob("*.st"):
        if task_id.lower() in f.stem.lower():
            return f
    return None


def run_rusty(rusty_bin, st_path, log_dir):
    """Run rusty on an ST file. Returns (returncode, stdout, stderr)."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = log_dir / "rusty.stdout.txt"
    stderr_path = log_dir / "rusty.stderr.txt"

    try:
        result = subprocess.run(
            [rusty_bin, "--check", str(st_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            text=True,
        )
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        return result.returncode, str(stdout_path), str(stderr_path)
    except subprocess.TimeoutExpired:
        msg = "rusty timed out after 60s"
        stdout_path.write_text(msg, encoding="utf-8")
        stderr_path.write_text(msg, encoding="utf-8")
        return -1, str(stdout_path), str(stderr_path)
    except FileNotFoundError:
        msg = f"rusty binary not found: {rusty_bin}"
        stdout_path.write_text(msg, encoding="utf-8")
        stderr_path.write_text(msg, encoding="utf-8")
        return -1, str(stdout_path), str(stderr_path)


def build_plcverif_args(plcverif_bin, nuxmv_bin, task_id, prop_idx, property_def, st_path, log_dir):
    """Build PLCverif CLI arguments for a single property.

    Handles the nested property format from Agents4PLC benchmark:
    {
      "property_description": "...",
      "property": {
        "job_req": "assertion" | "pattern",
        "pattern_id": "pattern-implication" | "pattern-invariant",
        "pattern_params": {"1": "expr1", "2": "expr2"}
      }
    }
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    prop_log_dir = log_dir / f"property_{prop_idx}"
    prop_log_dir.mkdir(parents=True, exist_ok=True)

    # Resolve relative paths to absolute (PLCverif resolves relative paths from its own dir)
    _nuxmv_abs = str(Path(nuxmv_bin).resolve())
    _st_abs = str(Path(st_path).resolve())
    args = [
        plcverif_bin,
        f"-id={task_id}_{prop_idx}",
        "-job=verif",
        "-lf=step7",
        f"-sourcefiles={_st_abs}",
        "-job.backend=nusmv",
        f"-job.backend.binary_path={_nuxmv_abs}",
        "-job.reporters={plaintext,summary}",
        f"-output={prop_log_dir}",
    ]

    # Unwrap nested property format: {"property_description":..., "property": {...}}
    if "property" in property_def:
        prop_inner = property_def["property"]
    else:
        # Flattened format (already unwrapped)
        prop_inner = property_def

    job_req = prop_inner.get("job_req", "").lower()

    if job_req == "pattern":
        args.append("-job.req=pattern")
        pattern_id = prop_inner.get("pattern_id", "")
        if pattern_id:
            args.append(f"-job.req.pattern_id={pattern_id}")
        params = prop_inner.get("pattern_params", {})
        for k, v in params.items():
            args.append(f'-job.req.pattern_params.{k}="{v}"')

    elif job_req == "assertion":
        args.append("-job.req=assertion")

    elif job_req == "ctl":
        args.append("-job.req=ctl")
        formula = prop_inner.get("formula", "")
        if formula:
            args.append(f'-job.req.ctl.formula="{formula}"')

    elif job_req == "ltl":
        args.append("-job.req=ltl")
        formula = prop_inner.get("formula", "")
        if formula:
            args.append(f'-job.req.ltl.formula="{formula}"')

    else:
        # Unknown property type — try as assertion
        args.append("-job.req=assertion")

    return args, prop_log_dir


def run_plcverif(plcverif_bin, nuxmv_bin, task_id, properties, st_path, log_dir):
    """Run PLCverif for all properties of a task. Returns list of per-property results."""
    results = []

    for prop_idx, prop in enumerate(properties):
        args, prop_log_dir = build_plcverif_args(
            plcverif_bin, nuxmv_bin, task_id, prop_idx, prop, st_path, log_dir
        )

        stdout_path = prop_log_dir / "stdout.txt"
        stderr_path = prop_log_dir / "stderr.txt"

        try:
            result = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
                text=True,
            )
            stdout_path.write_text(result.stdout, encoding="utf-8")
            stderr_path.write_text(result.stderr, encoding="utf-8")
            passed = result.returncode == 0
            results.append(
                {
                    "property_index": prop_idx,
                    "passed": passed,
                    "returncode": result.returncode,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                }
            )
        except subprocess.TimeoutExpired:
            msg = "PLCverif timed out after 300s"
            stdout_path.write_text(msg, encoding="utf-8")
            stderr_path.write_text(msg, encoding="utf-8")
            results.append(
                {
                    "property_index": prop_idx,
                    "passed": False,
                    "returncode": -1,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                }
            )
        except FileNotFoundError:
            msg = f"PLCverif binary not found: {plcverif_bin}"
            stdout_path.write_text(msg, encoding="utf-8")
            stderr_path.write_text(msg, encoding="utf-8")
            results.append(
                {
                    "property_index": prop_idx,
                    "passed": False,
                    "returncode": -1,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Agents4PLC 23-task evaluation")
    parser.add_argument("--manifest", required=True, help="Path to agents4plc_23_manifest.jsonl")
    parser.add_argument("--outputs", required=True, help="Directory with generated .st files")
    parser.add_argument("--out", required=True, help="Output evaluation.jsonl path")
    parser.add_argument("--tool_log_dir", required=True, help="Directory for tool logs")
    parser.add_argument("--rusty_bin", required=True, help="Path to rusty 'plc' binary")
    parser.add_argument("--plcverif_bin", required=True, help="Path to PLCverif CLI binary")
    parser.add_argument("--nuxmv_bin", required=True, help="Path to nuXmv binary")
    args = parser.parse_args()

    tasks = load_manifest(args.manifest)
    print(f"Loaded {len(tasks)} tasks from manifest")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tool_log_dir = Path(args.tool_log_dir)
    outputs_dir = Path(args.outputs)

    results = []
    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        difficulty = task.get("difficulty", "Unknown")
        properties = task.get("properties_to_be_validated", [])

        print(f"[{i+1}/{len(tasks)}] {task_id} ({difficulty})")

        st_path = find_st_file(task_id, outputs_dir)
        if st_path is None:
            print(f"  WARNING: No .st file found for {task_id}, skipping")
            results.append(
                {
                    "task_id": task_id,
                    "difficulty": difficulty,
                    "code_path": None,
                    "comp_pass": False,
                    "verif_pass": False,
                    "num_properties": len(properties),
                    "num_properties_passed": 0,
                    "error": "ST file not found",
                }
            )
            continue

        task_log_dir = tool_log_dir / task_id

        # Step 1: rusty compilation check
        rc, stdout_p, stderr_p = run_rusty(args.rusty_bin, st_path, task_log_dir)
        comp_pass = rc == 0
        print(f"  rusty: {'PASS' if comp_pass else 'FAIL'} (rc={rc})")

        # Step 2: PLCverif verification (only if compilation passed)
        if comp_pass and properties:
            plcverif_results = run_plcverif(
                args.plcverif_bin,
                args.nuxmv_bin,
                task_id,
                properties,
                st_path,
                task_log_dir,
            )
            num_passed = sum(1 for r in plcverif_results if r["passed"])
            verif_pass = num_passed == len(properties)
            print(f"  PLCverif: {num_passed}/{len(properties)} properties passed")
        elif comp_pass and not properties:
            plcverif_results = []
            verif_pass = True  # No properties to verify
        else:
            plcverif_results = []
            verif_pass = False

        results.append(
            {
                "task_id": task_id,
                "difficulty": difficulty,
                "code_path": str(st_path),
                "comp_pass": comp_pass,
                "verif_pass": verif_pass,
                "num_properties": len(properties),
                "num_properties_passed": sum(1 for r in plcverif_results if r["passed"]),
                "rusty_returncode": rc,
                "rusty_stdout_path": stdout_p,
                "rusty_stderr_path": stderr_p,
                "plcverif_results": plcverif_results,
            }
        )

    # Write results
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Quick summary
    total = len(results)
    comp_ok = sum(1 for r in results if r["comp_pass"])
    verif_ok = sum(1 for r in results if r["verif_pass"])
    print(f"\n{'='*50}")
    print(f"Evaluation complete: {total} tasks")
    print(f"  Compilation passed: {comp_ok}/{total} ({100*comp_ok/total:.1f}%)")
    print(f"  Verification passed: {verif_ok}/{total} ({100*verif_ok/total:.1f}%)")
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
