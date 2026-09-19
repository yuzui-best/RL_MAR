#!/usr/bin/env python3
"""Run the frozen PLC94 functional suite through the RuSTy C ABI."""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from iec_normalize import PREPROCESSING_VERSION, normalize_iec_st


C_TYPES: dict[str, type] = {
    "bool": ctypes.c_bool,
    "int8_t": ctypes.c_int8, "uint8_t": ctypes.c_uint8,
    "int16_t": ctypes.c_int16, "uint16_t": ctypes.c_uint16,
    "int32_t": ctypes.c_int32, "uint32_t": ctypes.c_uint32,
    "int64_t": ctypes.c_int64, "uint64_t": ctypes.c_uint64,
    "float": ctypes.c_float, "double": ctypes.c_double,
    "float_t": ctypes.c_float, "double_t": ctypes.c_double,
    "time_t": ctypes.c_int64,
    "void*": ctypes.c_void_p, "char": ctypes.c_char,
}
INTEGER_LIMITS = {
    "SINT": (-128, 127), "USINT": (0, 255), "BYTE": (0, 255),
    "INT": (-32768, 32767), "UINT": (0, 65535), "WORD": (0, 65535),
    "DINT": (-2147483648, 2147483647), "UDINT": (0, 4294967295), "DWORD": (0, 4294967295),
    "LINT": (-(2**63), 2**63 - 1), "ULINT": (0, 2**64 - 1), "LWORD": (0, 2**64 - 1),
    "TIME": (0, 24 * 60 * 60 * 1_000_000_000),
}
IEC_CTYPES: dict[str, type] = {
    "BOOL": ctypes.c_bool, "BYTE": ctypes.c_uint8, "SINT": ctypes.c_int8, "USINT": ctypes.c_uint8,
    "INT": ctypes.c_int16, "UINT": ctypes.c_uint16, "WORD": ctypes.c_uint16,
    "DINT": ctypes.c_int32, "UDINT": ctypes.c_uint32, "DWORD": ctypes.c_uint32,
    "LINT": ctypes.c_int64, "ULINT": ctypes.c_uint64, "LWORD": ctypes.c_uint64,
    "REAL": ctypes.c_float, "LREAL": ctypes.c_double, "TIME": ctypes.c_int64,
}
FIELD_RE = re.compile(r"^\s*([A-Za-z_][\w]*\s*\*?|void\s*\*)\s+([A-Za-z_][\w]*)(\s*(?:\[\d+\])+)?\s*;\s*$")
STRUCT_RE = re.compile(r"typedef\s+struct\s*\{(.*?)\}\s*([A-Za-z_]\w*)\s*;", re.DOTALL)
FUNCTION_RE_TEMPLATE = r"\bvoid\s+{entry}\s*\(\s*([A-Za-z_]\w*)\s*\*\s*self\s*\)"
ARRAY_TYPE_RE = re.compile(r"ARRAY\[\s*(-?\d+)\s*\.\.\s*(-?\d+)(?:\s*,\s*(-?\d+)\s*\.\.\s*(-?\d+))?\s*\]\s+OF\s+(.+)", re.IGNORECASE)
TIME_RE = re.compile(r"T#(\d+(?:\.\d+)?)(MS|S|M|H)", re.IGNORECASE)
INSTANCE_ARRAY_RE = re.compile(r"instance\.([A-Za-z_]\w*)\[(\d+)\]")
INSTANCE_RE = re.compile(r"instance\.([A-Za-z_]\w*)")


class FunctionalError(RuntimeError):
    pass


class CompileError(FunctionalError):
    pass


class DatasetError(FunctionalError):
    pass


class InterfaceError(FunctionalError):
    pass


class EntryPointError(FunctionalError):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_checked(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("command timeout") from error
    if result.returncode:
        diagnostic = (result.stderr or result.stdout)[-8000:]
        raise CompileError(diagnostic)
    return result


def compile_source(root: Path, run_id: str, method_slug: str, task: dict[str, Any], source: str, timeout: int) -> tuple[Path, Path, float]:
    work = root / "runs" / run_id / "build" / method_slug / task["task_id"]
    work.mkdir(parents=True, exist_ok=True)
    source_path = work / "program.st"
    source_path.write_bytes(source.encode("utf-8"))
    object_path = work / "program.o"
    library_path = work / "program.so"
    plc_root = root.parent
    rusty = plc_root / "tools/rusty/target/release/plc.real"
    docker = ["docker", "run", "--rm", "-v", f"{plc_root}:{plc_root}", "-w", str(root), "ubuntu:24.04", str(rusty)]
    started = time.monotonic()
    run_checked(docker + ["--header-output", str(work.relative_to(root)), "-o", "program", "--generate-headers", str(source_path.relative_to(root))], timeout)
    run_checked(docker + [str(source_path.relative_to(root)), "-o", str(object_path.relative_to(root)), "-c", "--fpic", "--target", "x86_64-linux-gnu"], timeout)
    run_checked([shutil.which("cc") or "cc", "-shared", "-o", str(library_path), str(object_path)], timeout)
    header = work / "program.h"
    if not header.is_file():
        raise FunctionalError("RuSTy did not generate program.h")
    return library_path, header, time.monotonic() - started


def parse_structs(header: Path, entry: str, task: dict[str, Any]) -> tuple[type[ctypes.Structure], dict[str, type], dict[str, str]]:
    text = header.read_text(encoding="utf-8")
    bodies = {name: body for body, name in STRUCT_RE.findall(text)}
    built: dict[str, type] = {}
    building: set[str] = set()
    inout_types = {item["name"]: item["type"] for item in task["pou_declarations"] if item["section"] == "inout"}

    def build(name: str) -> type:
        if name in built:
            return built[name]
        if name not in bodies or name in building:
            raise FunctionalError(f"unsupported or recursive C ABI type: {name}")
        building.add(name)
        fields = []
        for raw in bodies[name].splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("//"):
                continue
            match = FIELD_RE.match(raw)
            if not match:
                raise FunctionalError(f"unsupported C ABI field: {raw}")
            c_name, field_name, dimensions = match.groups()
            c_name = re.sub(r"\s+", "", c_name)
            manifest_inout = name == entry_name and field_name in inout_types
            pointer_field = c_name.endswith("*") or manifest_inout
            field_type = ctypes.c_void_p if pointer_field else C_TYPES.get(c_name) or IEC_CTYPES.get(c_name.upper()) or (build(c_name) if c_name in bodies else ctypes.c_int32 if c_name.startswith("__") else None)
            if field_type is None:
                raise FunctionalError(f"unsupported C ABI field type: {c_name}")
            if not pointer_field:
                for dimension in reversed([int(value) for value in re.findall(r"\[(\d+)\]", dimensions or "")]):
                    field_type = field_type * dimension
            fields.append((field_name, field_type))
        struct = type(name, (ctypes.Structure,), {"_fields_": fields})
        built[name] = struct
        building.remove(name)
        return struct

    function_match = re.search(FUNCTION_RE_TEMPLATE.format(entry=re.escape(entry)), text)
    if not function_match:
        raise EntryPointError(f"entry point not exported: {entry}")
    entry_name = function_match.group(1)
    entry_type = build(entry_name)
    for type_name in bodies:
        build(type_name)
    pointer_targets = dict(inout_types)
    for raw in bodies[entry_name].splitlines():
        match = FIELD_RE.match(raw.strip())
        if match and "*" in match.group(1) and match.group(2) not in pointer_targets:
            pointer_targets[match.group(2)] = match.group(1).replace("*", "").strip()
    return entry_type, built, pointer_targets


def iec_ctype(type_name: str, custom_types: dict[str, type] | None = None) -> type:
    normalized = type_name.strip().upper()
    if normalized in IEC_CTYPES:
        return IEC_CTYPES[normalized]
    if custom_types:
        for name, custom_type in custom_types.items():
            if name.upper() == normalized:
                return custom_type
    match = ARRAY_TYPE_RE.fullmatch(normalized)
    if not match:
        raise FunctionalError(f"unsupported IN_OUT type: {type_name}")
    first = int(match.group(2)) - int(match.group(1)) + 1
    second = int(match.group(4)) - int(match.group(3)) + 1 if match.group(3) else None
    result = iec_ctype(match.group(5), custom_types)
    if second is not None:
        result = result * second
    return result * first


def assign(current: Any, value: Any, path: str) -> None:
    if isinstance(current, ctypes.Array):
        if not isinstance(value, list) or len(value) != len(current):
            raise InterfaceError(f"{path}: expected array length {len(current)}, got {len(value) if isinstance(value, list) else 'scalar'}")
        for index, item in enumerate(value):
            assign(current[index], item, f"{path}[{index}]") if isinstance(current[index], (ctypes.Array, ctypes.Structure)) else current.__setitem__(index, item)
        return
    if isinstance(current, ctypes.Structure):
        if not isinstance(value, dict):
            raise InterfaceError(f"{path}: expected object")
        for name, item in value.items():
            if not hasattr(current, name):
                raise InterfaceError(f"{path}.{name}: missing field")
            child = getattr(current, name)
            assign(child, item, f"{path}.{name}") if isinstance(child, (ctypes.Array, ctypes.Structure)) else setattr(current, name, item)
        return
    raise FunctionalError(f"cannot assign compound value at {path}")


def set_field(instance: ctypes.Structure, name: str, value: Any, inout_buffers: dict[str, Any] | None = None) -> None:
    if inout_buffers and name in inout_buffers:
        target = inout_buffers[name]
        if isinstance(target, (ctypes.Array, ctypes.Structure)):
            assign(target, value, name)
        else:
            target.value = value
        return
    if not hasattr(instance, name):
        raise InterfaceError(f"generated ABI missing field: {name}")
    current = getattr(instance, name)
    if isinstance(current, (ctypes.Array, ctypes.Structure)):
        assign(current, value, name)
    else:
        setattr(instance, name, value)


def read_value(value: Any) -> Any:
    if isinstance(value, ctypes.Array):
        return [read_value(item) for item in value]
    if isinstance(value, ctypes.Structure):
        return {name: read_value(getattr(value, name)) for name, _ in value._fields_ if name != "__vtable"}
    if hasattr(value, "value"):
        return value.value
    return value


def get_field(instance: ctypes.Structure, name: str, index: int | None = None, inout_buffers: dict[str, Any] | None = None) -> Any:
    if inout_buffers and name in inout_buffers:
        value = inout_buffers[name]
        if index is not None:
            value = value[index - 1]
        return read_value(value)
    if not hasattr(instance, name):
        raise InterfaceError(f"generated ABI missing field: {name}")
    value = getattr(instance, name)
    if index is not None:
        if not isinstance(value, ctypes.Array) or not 1 <= index <= len(value):
            raise FunctionalError(f"invalid IEC array access: {name}[{index}]")
        value = value[index - 1]
    return read_value(value)


def equal(actual: Any, expected: Any, tolerance: dict[str, float]) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and equal(actual[key], value, tolerance) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(equal(a, e, tolerance) for a, e in zip(actual, expected))
    if isinstance(expected, float):
        return math.isclose(float(actual), expected, rel_tol=tolerance["relative"], abs_tol=tolerance["absolute"])
    return actual == expected


def execute_exact(task: dict[str, Any], library: ctypes.CDLL, struct_type: type[ctypes.Structure], custom_types: dict[str, type], pointer_targets: dict[str, str]) -> dict[str, Any]:
    function = getattr(library, task["entry_point"])
    function.argtypes = [ctypes.POINTER(struct_type)]
    function.restype = None
    constructor = getattr(library, f"{task['entry_point']}__ctor", None)
    if constructor:
        constructor.argtypes = [ctypes.POINTER(struct_type)]
        constructor.restype = None
    cases = []
    for case in task["cases"]:
        instance = struct_type()
        if constructor:
            constructor(ctypes.byref(instance))
        inout_buffers = {}
        for declaration in task["pou_declarations"]:
            if declaration["section"] != "inout":
                continue
            if not hasattr(instance, declaration["name"]):
                raise InterfaceError(f"generated ABI missing IN_OUT field: {declaration['name']}")
            if declaration["name"] not in pointer_targets:
                inline_value = getattr(instance, declaration["name"])
                if isinstance(inline_value, (ctypes.Array, ctypes.Structure)):
                    continue
                raise InterfaceError(f"generated ABI scalar IN_OUT is not a pointer: {declaration['name']}")
            buffer = iec_ctype(declaration["type"], custom_types)()
            inout_buffers[declaration["name"]] = buffer
            setattr(instance, declaration["name"], ctypes.addressof(buffer))
        for name, value in {**case.get("initial_outputs", {}), **case.get("initial_state", {})}.items():
            set_field(instance, name, value, inout_buffers)
        observations, mismatches = [], []
        tolerance = case["oracle"]["real_tolerance"]
        for cycle in range(case["simulation_rounds"]):
            for name, trace in case["input_trace"].items():
                set_field(instance, name, trace[cycle], inout_buffers)
            function(ctypes.byref(instance))
            observed = {}
            for group in ("expected_output_trace", "expected_internal_state"):
                for name, trace in case["oracle"].get(group, {}).items():
                    actual, expected = get_field(instance, name, inout_buffers=inout_buffers), trace[cycle]
                    observed[name] = actual
                    if not equal(actual, expected, tolerance):
                        mismatches.append({"cycle": cycle + 1, "field": name, "expected": expected, "actual": actual})
            observations.append(observed)
        cases.append({"case_id": case["case_id"], "status": "PASS" if not mismatches else "FUNCTION_FAIL", "observations": observations, "first_mismatch": mismatches[0] if mismatches else None})
    return {"status": "PASS" if all(case["status"] == "PASS" for case in cases) else "FUNCTION_FAIL", "checks": cases}


def time_literal(match: re.Match[str]) -> str:
    value, unit = float(match.group(1)), match.group(2).upper()
    scale = {"MS": 1_000_000, "S": 1_000_000_000, "M": 60_000_000_000, "H": 3_600_000_000_000}[unit]
    return str(int(value * scale))


def compile_expression(expression: str) -> Any:
    translated = TIME_RE.sub(time_literal, expression)
    translated = INSTANCE_ARRAY_RE.sub(r'v("\1", \2)', translated)
    translated = INSTANCE_RE.sub(r'v("\1")', translated)
    translated = re.sub(r"\bTRUE\b", "True", translated, flags=re.IGNORECASE)
    translated = re.sub(r"\bFALSE\b", "False", translated, flags=re.IGNORECASE)
    translated = translated.replace("<>", "!=")
    translated = re.sub(r"(?<![<>=!])=(?!=)", "==", translated)
    translated = re.sub(r"\bAND\b", "and", translated, flags=re.IGNORECASE)
    translated = re.sub(r"\bOR\b", "or", translated, flags=re.IGNORECASE)
    translated = re.sub(r"\bNOT\b", "not", translated, flags=re.IGNORECASE)
    tree = ast.parse(translated, mode="eval")
    allowed = (ast.Expression, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare, ast.Call, ast.Name, ast.Load, ast.Constant, ast.And, ast.Or, ast.Not, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.USub, ast.UAdd)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise DatasetError(f"unsupported oracle expression node: {type(node).__name__}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id == "v"):
            raise DatasetError("unsupported oracle call")
        if isinstance(node, ast.Name) and node.id not in {"v", "True", "False"}:
            raise DatasetError(f"unsupported oracle name: {node.id}")
    return compile(tree, "<oracle>", "eval")


def evaluate(code: Any, instance: ctypes.Structure) -> bool:
    return bool(eval(code, {"__builtins__": {}}, {"v": lambda name, index=None: get_field(instance, name, index)}))


def expression_constants(expressions: list[str]) -> list[float]:
    text = " ".join(TIME_RE.sub("", expression) for expression in expressions)
    return [float(value) for value in re.findall(r"(?<![\w#])-?\d+(?:\.\d+)?", text)]


def scalar_domain(type_name: str, constants: list[float]) -> list[Any]:
    normalized = type_name.upper()
    if normalized == "BOOL":
        return [False, True]
    values: list[Any] = [0, 1, -1]
    for constant in constants:
        values.extend([constant - 1, constant, constant + 1])
    if normalized in INTEGER_LIMITS:
        lower, upper = INTEGER_LIMITS[normalized]
        values.extend([lower, upper])
        return list(dict.fromkeys(max(lower, min(upper, int(value))) for value in values))
    if normalized in {"REAL", "LREAL"}:
        values.extend([-100.0, 100.0])
        return list(dict.fromkeys(float(value) for value in values))
    return [0]


def candidate_value(type_name: str, row: int, field_index: int, constants: list[float]) -> Any:
    match = ARRAY_TYPE_RE.fullmatch(type_name.strip())
    if match:
        first = int(match.group(2)) - int(match.group(1)) + 1
        second = int(match.group(4)) - int(match.group(3)) + 1 if match.group(3) else None
        element = match.group(5)
        if second is not None:
            return [[candidate_value(element, row + i * second + j, field_index, constants) for j in range(second)] for i in range(first)]
        domain = scalar_domain(element, constants)
        return [domain[(row + field_index + i) % len(domain)] if i == row % first else domain[0] for i in range(first)]
    domain = scalar_domain(type_name, constants)
    if type_name.strip().upper() == "BOOL":
        column_pattern = field_index % 63 + 1
        return bool(bin(row & column_pattern).count("1") % 2)
    return domain[(row * (field_index * 2 + 1) + field_index) % len(domain)]


def expression_fields(expressions: list[str]) -> set[str]:
    return set(INSTANCE_RE.findall(" ".join(expressions))) | {name for name, _ in INSTANCE_ARRAY_RE.findall(" ".join(expressions))}


def execute_properties(task: dict[str, Any], library: ctypes.CDLL, struct_type: type[ctypes.Structure]) -> dict[str, Any]:
    function = getattr(library, task["entry_point"])
    function.argtypes = [ctypes.POINTER(struct_type)]
    function.restype = None
    constructor = getattr(library, f"{task['entry_point']}__ctor", None)
    if constructor:
        constructor.argtypes = [ctypes.POINTER(struct_type)]
        constructor.restype = None
    declarations = {item["name"]: item for item in task["pou_declarations"]}
    inputs = [item for item in task["pou_declarations"] if item["section"] in {"input", "inout"}]
    checks = []
    for oracle in task["oracles"]:
        semantics = oracle["semantics"]
        kind = semantics["kind"]
        expressions = [value for key, value in semantics.items() if key != "kind" and isinstance(value, str)]
        compiled = {key: compile_expression(value) for key, value in semantics.items() if key != "kind" and isinstance(value, str)}
        constants = expression_constants(expressions)
        controlled = list(inputs)
        if kind == "begin_to_end_transition":
            for name in sorted(expression_fields([semantics["pre"]])):
                if name in declarations and declarations[name] not in controlled:
                    controlled.append(declarations[name])
        activations = violations = 0
        first_failure = None
        reached = False
        for row in range(task["stimulus"].get("max_cycles_per_oracle", 64)):
            instance = struct_type()
            if constructor:
                constructor(ctypes.byref(instance))
            vector = {}
            for field_index, item in enumerate(controlled):
                value = candidate_value(item["type"], row, field_index, constants)
                set_field(instance, item["name"], value)
                vector[item["name"]] = value
            if kind == "begin_to_end_transition":
                active = evaluate(compiled["pre"], instance)
                function(ctypes.byref(instance))
                if active:
                    activations += 1
                    if not evaluate(compiled["post"], instance):
                        violations += 1
            elif kind == "between_cycle_transition":
                function(ctypes.byref(instance))
                active = evaluate(compiled["cycle_n"], instance)
                function(ctypes.byref(instance))
                if active and evaluate(compiled["cycle_n_plus_1"], instance):
                    activations += 1
                    if not evaluate(compiled["post"], instance):
                        violations += 1
            elif kind == "same_cycle_implication":
                function(ctypes.byref(instance))
                if evaluate(compiled["pre"], instance):
                    activations += 1
                    if not evaluate(compiled["post"], instance):
                        violations += 1
            elif kind == "post_cycle_invariant":
                function(ctypes.byref(instance))
                activations += 1
                if not evaluate(compiled["post"], instance):
                    violations += 1
            elif kind == "post_cycle_forbidden":
                function(ctypes.byref(instance))
                activations += 1
                if evaluate(compiled["forbidden"], instance):
                    violations += 1
            elif kind == "bounded_reachability":
                function(ctypes.byref(instance))
                if evaluate(compiled["target"], instance):
                    reached = True
                    activations += 1
                    break
            else:
                raise DatasetError(f"unsupported semantics: {kind}")
            if violations and first_failure is None:
                first_failure = {"row": row, "input": vector}
        passed = (reached if kind == "bounded_reachability" else violations == 0 and (activations > 0 or kind in {"post_cycle_invariant", "post_cycle_forbidden"}))
        checks.append({"oracle_id": oracle["oracle_id"], "status": "PASS" if passed else "FUNCTION_FAIL", "activations": activations, "violations": violations, "first_failure": first_failure})
    return {"status": "PASS" if checks and all(check["status"] == "PASS" for check in checks) else "FUNCTION_FAIL", "checks": checks}


def worker(request: dict[str, Any]) -> dict[str, Any]:
    root = Path(request["root"])
    task = request["task"]
    started = time.monotonic()
    original_source = request["source"]
    normalized_source = normalize_iec_st(original_source)
    result = {
        "method": request["method"],
        "task_id": task["task_id"],
        "suite_version": task.get("suite_version", "plc94.functional.v1.3"),
        "preprocessing_version": PREPROCESSING_VERSION,
        "original_code_sha256": sha256_text(original_source),
        "normalized_code_sha256": sha256_text(normalized_source),
        "source_sha256": sha256_text(original_source),
        "test_vector_sha256": request["test_vector_sha256"],
    }
    try:
        library_path, header, compile_seconds = compile_source(root, request["run_id"], request["method_slug"], task, normalized_source, request["compile_timeout"])
        struct_type, custom_types, pointer_targets = parse_structs(header, task["entry_point"], task)
        library = ctypes.CDLL(str(library_path))
        execution = execute_exact(task, library, struct_type, custom_types, pointer_targets) if task["validation_mode"] == "exact_trace" else execute_properties(task, library, struct_type)
        result.update(execution)
        result["compile_seconds"] = compile_seconds
    except DatasetError as error:
        result.update({"status": "DATASET_ERROR", "diagnostic": str(error), "checks": []})
    except TimeoutError as error:
        result.update({"status": "TIMEOUT", "diagnostic": str(error), "checks": []})
    except CompileError as error:
        result.update({"status": "COMPILE_FAIL", "diagnostic": str(error), "checks": []})
    except SyntaxError as error:
        result.update({"status": "DATASET_ERROR", "diagnostic": str(error), "checks": []})
    except EntryPointError as error:
        result.update({"status": "FUNCTION_FAIL", "diagnostic": str(error), "checks": []})
    except (InterfaceError, TypeError) as error:
        result.update({"status": "FUNCTION_FAIL", "diagnostic": str(error), "checks": []})
    except FunctionalError as error:
        result.update({"status": "RUNTIME_FAIL", "diagnostic": str(error), "checks": []})
    except Exception as error:
        result.update({"status": "RUNTIME_FAIL", "diagnostic": f"{type(error).__name__}: {error}", "checks": []})
    result["failure_category"] = None if result["status"] == "PASS" else result["status"]
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def load_sources(package: dict[str, Any]) -> dict[str, str]:
    if package["source_kind"] == "st_directory":
        return {item["task_id"]: Path(item["source_path"]).read_text(encoding="utf-8") for item in package["tasks"] if item["status"] == "READY"}
    records = json.loads(Path(package["source_path"]).read_text(encoding="utf-8"))
    sources = {}
    for record in records:
        code = record.get("generated_code")
        if not isinstance(code, str) or not code.strip():
            continue
        stripped = code.strip()
        if stripped.startswith("START_ST") and stripped.endswith("END_ST"):
            stripped = stripped[len("START_ST"):-len("END_ST")].strip()
        sources[record["task_id"]] = stripped
    return sources


def stable_test_hash(task: dict[str, Any]) -> str:
    payload = {"task_id": task["task_id"], "cases": task["cases"], "oracles": task["oracles"], "stimulus": task["stimulus"]}
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def result_provenance(task: dict[str, Any], method: str, source: str | None) -> dict[str, Any]:
    normalized = normalize_iec_st(source) if source is not None else None
    return {
        "method": method,
        "task_id": task["task_id"],
        "suite_version": task.get("suite_version", "plc94.functional.v1.3"),
        "preprocessing_version": PREPROCESSING_VERSION,
        "original_code_sha256": sha256_text(source) if source is not None else None,
        "normalized_code_sha256": sha256_text(normalized) if normalized is not None else None,
        "source_sha256": sha256_text(source) if source is not None else None,
        "test_vector_sha256": stable_test_hash(task),
    }


def task_definition_error(task: dict[str, Any]) -> str | None:
    if task["validation_mode"] == "property_directed":
        try:
            for oracle in task["oracles"]:
                for key, expression in oracle["semantics"].items():
                    if key != "kind" and isinstance(expression, str):
                        compile_expression(expression)
        except (DatasetError, SyntaxError) as error:
            return f"invalid functional oracle: {error}"
        return None
    declared = {item["name"]: item["type"] for item in task["pou_declarations"]}
    for case in task["cases"]:
        for container_name in ("input_trace", "initial_state", "initial_outputs"):
            if not isinstance(case.get(container_name, {}), dict):
                return f"{case['case_id']}: {container_name} must be an object"
        oracle = case.get("oracle", {})
        for container_name in ("expected_output_trace", "expected_internal_state"):
            if not isinstance(oracle.get(container_name, {}), dict):
                return f"{case['case_id']}: oracle.{container_name} must be an object"
        referenced = set(case.get("input_trace", {})) | set(case.get("initial_state", {})) | set(case.get("initial_outputs", {}))
        referenced |= set(case.get("oracle", {}).get("expected_output_trace", {}))
        referenced |= set(case.get("oracle", {}).get("expected_internal_state", {}))
        unknown = sorted(referenced - set(declared))
        if unknown:
            return f"{case['case_id']}: fields absent from task declaration: {unknown}"
        for name, cycles in case.get("input_trace", {}).items():
            match = ARRAY_TYPE_RE.fullmatch(declared.get(name, "").strip())
            if not match:
                continue
            shape = [int(match.group(2)) - int(match.group(1)) + 1]
            if match.group(3):
                shape.append(int(match.group(4)) - int(match.group(3)) + 1)
            for cycle, value in enumerate(cycles, 1):
                observed = []
                cursor = value
                while isinstance(cursor, list):
                    observed.append(len(cursor))
                    cursor = cursor[0] if cursor else None
                if observed != shape:
                    return f"{case['case_id']} {name} cycle {cycle}: dataset shape {observed}, declared shape {shape}"
    return None


def master(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    tasks = load_jsonl(root / args.manifest)
    index = json.loads((root / args.input_index).read_text(encoding="utf-8"))
    output_dir = root / "runs" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_tasks = [task for task in tasks if not args.tasks or task["task_id"] in args.tasks]
    selected_packages = [package for package in index["packages"] if not args.methods or package["method"] in args.methods]
    all_results = []
    for package in selected_packages:
        sources = load_sources(package)
        method = package["method"]
        method_slug = re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")
        method_results = []
        for task in selected_tasks:
            source = sources.get(task["task_id"])
            definition_error = task_definition_error(task)
            if definition_error:
                result = result_provenance(task, method, source)
                result.update({"status": "DATASET_ERROR", "failure_category": "DATASET_ERROR", "diagnostic": definition_error, "checks": []})
                method_results.append(result)
                continue
            if source is None:
                result = result_provenance(task, method, None)
                result.update({"status": "MISSING_OUTPUT", "failure_category": "MISSING_OUTPUT", "checks": []})
                method_results.append(result)
                continue
            request = {"root": str(root), "run_id": args.run_id, "method": method, "method_slug": method_slug, "task": task, "source": source, "test_vector_sha256": stable_test_hash(task), "compile_timeout": args.compile_timeout}
            request_path = output_dir / "requests" / method_slug / f"{task['task_id']}.json"
            response_path = output_dir / "responses" / method_slug / f"{task['task_id']}.json"
            request_path.parent.mkdir(parents=True, exist_ok=True)
            response_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(json.dumps(request, sort_keys=True) + "\n", encoding="utf-8")
            try:
                process = subprocess.run([sys.executable, __file__, "--worker", str(request_path), "--response", str(response_path)], text=True, capture_output=True, timeout=args.task_timeout)
                if response_path.is_file():
                    result = json.loads(response_path.read_text(encoding="utf-8"))
                else:
                    result = result_provenance(task, method, source)
                    diagnostic = (process.stderr or process.stdout)[-4000:]
                    if not diagnostic:
                        diagnostic = f"worker exited without response (returncode={process.returncode})"
                    result.update({"status": "RUNTIME_FAIL", "failure_category": "RUNTIME_FAIL", "diagnostic": diagnostic, "worker_returncode": process.returncode, "checks": []})
            except subprocess.TimeoutExpired:
                result = result_provenance(task, method, source)
                result.update({"status": "TIMEOUT", "failure_category": "TIMEOUT", "diagnostic": "task timeout", "checks": []})
            method_results.append(result)
            print(f"{method} {task['task_id']}: {result['status']}", flush=True)
        counts = Counter(result["status"] for result in method_results)
        functional_pass = counts.get("PASS", 0)
        denominator = 94 if len(selected_tasks) == 94 else len(selected_tasks)
        summary = {"method": method, "model": index["expected_model_family"], "suite_version": selected_tasks[0].get("suite_version") if selected_tasks else None, "preprocessing_version": PREPROCESSING_VERSION, "functional_pass_tasks": functional_pass, "task_count": denominator, "pass_percent": round(functional_pass / denominator * 100, 1), "verified_tasks": package["verified_tasks"], "verif_percent": round(package["verified_tasks"] / 94 * 100, 1), "status_counts": dict(counts), "provenance_status": package["provenance_status"]}
        (output_dir / f"{method_slug}.json").write_text(json.dumps({"summary": summary, "tasks": method_results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        all_results.append({"summary": summary, "tasks": method_results})
    hashes_by_task: dict[str, set[str]] = {}
    for method in all_results:
        for result in method["tasks"]:
            hashes_by_task.setdefault(result["task_id"], set()).add(result["test_vector_sha256"])
    inconsistent = [task_id for task_id, hashes in hashes_by_task.items() if len(hashes) != 1]
    report = {"schema_version": "plc94-functional-run-1.2", "suite_version": selected_tasks[0].get("suite_version") if selected_tasks else None, "preprocessing_version": PREPROCESSING_VERSION, "run_id": args.run_id, "manifest": args.manifest, "task_count_per_method": len(selected_tasks), "formal_run": len(selected_tasks) == 94 and len(selected_packages) == 4, "test_vector_hash_consistent": not inconsistent, "inconsistent_tasks": inconsistent, "methods": [item["summary"] for item in all_results]}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if inconsistent:
        raise SystemExit(1)


def main() -> None:
    if "--worker" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker", required=True)
        parser.add_argument("--response", required=True)
        args = parser.parse_args()
        result = worker(json.loads(Path(args.worker).read_text(encoding="utf-8")))
        Path(args.response).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", default="data/plc94.functional.v1.3.jsonl")
    parser.add_argument("--input-index", required=True, help="JSON index describing generated ST packages")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--compile-timeout", type=int, default=120)
    parser.add_argument("--task-timeout", type=int, default=180)
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--tasks", nargs="*")
    master(parser.parse_args())


if __name__ == "__main__":
    main()
