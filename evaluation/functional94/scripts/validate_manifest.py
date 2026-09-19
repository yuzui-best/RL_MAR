#!/usr/bin/env python3
"""Fail-closed structural validation for the PLC94 functional manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXPECTED_IDS = (
    [f"hf_{index}" for index in range(1, 22)]
    + [f"med_{index}" for index in range(22, 92)]
    + [f"hard_{index}" for index in range(92, 95)]
)
SUPPORTED_SEMANTICS = {
    "same_cycle_implication",
    "post_cycle_invariant",
    "post_cycle_forbidden",
    "begin_to_end_transition",
    "between_cycle_transition",
    "bounded_reachability",
}


def array_length(type_name: str) -> int | None:
    match = re.fullmatch(r"ARRAY\[\s*(-?\d+)\s*\.\.\s*(-?\d+)\s*\]\s+OF\s+.+", type_name, re.IGNORECASE)
    return None if not match else int(match.group(2)) - int(match.group(1)) + 1


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def official_provenance(path: Path) -> dict[str, tuple[str, str, str]]:
    records = {}
    decoder = json.JSONDecoder()
    for source in ("high-fidelity.jsonl", "medium.jsonl", "hard.jsonl"):
        source_path = path / source
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        text, offset = source_path.read_text(encoding="utf-8"), 0
        while offset < len(text):
            while offset < len(text) and text[offset].isspace(): offset += 1
            if offset >= len(text): break
            record, offset = decoder.raw_decode(text, offset)
            instruction = str(record.get("instruction", "")).strip()
            records[record["id"]] = (instruction, canonical_hash(record), source_sha)
    return records


def validate(path: Path, official_dir: Path | None = None) -> list[str]:
    tasks = load(path)
    errors: list[str] = []
    task_ids = [task.get("task_id") for task in tasks]
    provenance = official_provenance(official_dir) if official_dir else {}
    if task_ids != EXPECTED_IDS:
        errors.append("task ids/order do not match hf_1..21, med_22..91, hard_92..94")

    exact_cases = 0
    for task in tasks:
        task_id = task.get("task_id", "<missing>")
        if task.get("schema_version") != "1.3":
            errors.append(f"{task_id}: wrong schema_version")
        if task.get("schema_version") == "1.3":
            if task.get("suite_version") != "plc94.functional.v1.3":
                errors.append(f"{task_id}: wrong or missing suite_version")
            instruction = task.get("instruction", "")
            expected_hash = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
            if not instruction.strip():
                errors.append(f"{task_id}: missing public instruction")
            if task.get("instruction_sha256") != expected_hash:
                errors.append(f"{task_id}: instruction hash mismatch")
            if official_dir is not None:
                source = provenance.get(task.get("source_task_id"))
                if source is None:
                    errors.append(f"{task_id}: official instruction provenance missing")
                elif (instruction, task.get("instruction_record_sha256"), task.get("instruction_source_sha256")) != source:
                    errors.append(f"{task_id}: official instruction provenance mismatch")
        if task.get("quality", {}).get("status") != "ready":
            errors.append(f"{task_id}: quality status is not ready")
        if task.get("quality", {}).get("blockers"):
            errors.append(f"{task_id}: quality blockers present")
        if not task.get("entry_point") or not task.get("pou_interface"):
            errors.append(f"{task_id}: missing entry point or interface")
        if str(task.get("title", "")).lower() in {"in", "out"}:
            errors.append(f"{task_id}: malformed title")

        if task.get("validation_mode") == "exact_trace":
            if not task.get("cases"):
                errors.append(f"{task_id}: no exact cases")
            exact_cases += len(task.get("cases", []))
            for case in task.get("cases", []):
                rounds = case.get("simulation_rounds")
                interface_types = {item["name"]: item["type"] for item in task.get("pou_interface", [])}
                for trace_name in ("input_trace",):
                    for symbol, values in case.get(trace_name, {}).items():
                        if len(values) != rounds:
                            errors.append(f"{task_id}/{case.get('case_id')}: {symbol} input length mismatch")
                oracle = case.get("oracle", {})
                for trace_name in ("expected_output_trace", "expected_internal_state"):
                    for symbol, values in oracle.get(trace_name, {}).items():
                        if not isinstance(values, list) or len(values) != rounds:
                            errors.append(f"{task_id}/{case.get('case_id')}: {symbol} oracle length mismatch")
        elif task.get("validation_mode") == "property_directed":
            if not task.get("oracles"):
                errors.append(f"{task_id}: no property oracles")
            for oracle in task.get("oracles", []):
                if oracle.get("semantics", {}).get("kind") not in SUPPORTED_SEMANTICS:
                    errors.append(f"{task_id}/{oracle.get('oracle_id')}: unsupported semantics")
                if oracle.get("unresolved_symbols"):
                    errors.append(f"{task_id}/{oracle.get('oracle_id')}: unresolved symbols")
                if oracle.get("observability") not in {"black_box", "white_box"}:
                    errors.append(f"{task_id}/{oracle.get('oracle_id')}: invalid observability")
        else:
            errors.append(f"{task_id}: invalid validation mode")

    expected_exact_cases = 121
    if exact_cases != expected_exact_cases:
        errors.append(f"expected {expected_exact_cases} exact cases, found {exact_cases}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--official-dir", type=Path)
    args = parser.parse_args()
    path = Path(args.manifest)
    errors = validate(path, args.official_dir)
    report = {"manifest": str(path), "status": "PASS" if not errors else "FAIL", "errors": errors}
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
