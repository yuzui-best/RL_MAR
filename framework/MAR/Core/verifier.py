from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping

from .dataset import ScanCase, TaskRecord, stable_sha256


_C_TYPES: dict[str, type] = {
    "bool": ctypes.c_bool,
    "int8_t": ctypes.c_int8,
    "uint8_t": ctypes.c_uint8,
    "int16_t": ctypes.c_int16,
    "uint16_t": ctypes.c_uint16,
    "int32_t": ctypes.c_int32,
    "uint32_t": ctypes.c_uint32,
    "int64_t": ctypes.c_int64,
    "uint64_t": ctypes.c_uint64,
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "void*": ctypes.c_void_p,
}
_FIELD = re.compile(r"^\s*([A-Za-z_]\w*\s*\*?|void\s*\*)\s+([A-Za-z_]\w*)\s*;\s*$")
_STRUCT = re.compile(r"typedef\s+struct\s*\{(.*?)\}\s*([A-Za-z_]\w*)\s*;", re.DOTALL)


class VerificationInfrastructureError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifierConfig:
    rusty: Path
    mount_root: Path
    docker_image: str = "ubuntu:24.04"
    timeout_seconds: int = 120


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise VerificationInfrastructureError(str(error)) from error
    return result


def _compile(code: str, entry_point: str, config: VerifierConfig, work: Path) -> tuple[Path, Path, float]:
    source = work / "program.st"
    source.write_text(code, encoding="utf-8")
    header_dir = work / "headers"
    object_path = work / "program.o"
    library_path = work / "program.so"
    prefix = [
        "docker", "run", "--rm", "-v", f"{config.mount_root}:{config.mount_root}",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-w", str(work), config.docker_image, str(config.rusty),
    ]
    started = time.monotonic()
    header = _run(prefix + ["--header-output", str(header_dir), "-o", "program", "--generate-headers", str(source)], config.timeout_seconds)
    if header.returncode:
        raise ValueError((header.stderr or header.stdout)[-4000:])
    compiled = _run(prefix + [str(source), "-o", str(object_path), "-c", "--fpic", "--target", "x86_64-linux-gnu"], config.timeout_seconds)
    if compiled.returncode:
        raise ValueError((compiled.stderr or compiled.stdout)[-4000:])
    linked = _run([shutil.which("cc") or "cc", "-shared", "-o", str(library_path), str(object_path)], config.timeout_seconds)
    if linked.returncode:
        raise VerificationInfrastructureError((linked.stderr or linked.stdout)[-4000:])
    header_path = header_dir / "program.h"
    if not header_path.is_file():
        raise ValueError("RuSTy did not generate a header for the submitted POU")
    return library_path, header_path, time.monotonic() - started


def _load_abi(library_path: Path, header_path: Path, entry_point: str) -> tuple[Any, ctypes.Structure]:
    text = header_path.read_text(encoding="utf-8")
    structures = {name: body for body, name in _STRUCT.findall(text)}
    type_name = f"{entry_point}_type"
    if type_name not in structures:
        raise ValueError(f"entry point ABI not found: {entry_point}")
    fields = []
    for raw in structures[type_name].splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("//"):
            continue
        match = _FIELD.match(raw)
        if not match:
            raise ValueError(f"unsupported ABI field: {raw}")
        c_name, name = match.groups()
        c_name = re.sub(r"\s+", "", c_name)
        field_type = ctypes.c_void_p if c_name.endswith("*") else _C_TYPES.get(c_name)
        if field_type is None:
            raise ValueError(f"unsupported ABI type: {c_name}")
        fields.append((name, field_type))
    instance_type = type(type_name, (ctypes.Structure,), {"_fields_": fields})
    library = ctypes.CDLL(str(library_path))
    function = getattr(library, entry_point)
    function.argtypes = [ctypes.POINTER(instance_type)]
    function.restype = None
    return function, instance_type()


def _set_fields(instance: ctypes.Structure, values: Mapping[str, Any]) -> None:
    for name, value in values.items():
        if not hasattr(instance, name):
            raise ValueError(f"field absent from generated ABI: {name}")
        setattr(instance, name, value)


def _run_case(function: Any, instance_type: type, case: ScanCase, observed_names: tuple[str, ...]) -> dict[str, Any]:
    instance = instance_type()
    _set_fields(instance, case.initial_state)
    violations = []
    activations = 0
    for scan_index, (inputs, expected) in enumerate(zip(case.scans, case.expected)):
        _set_fields(instance, inputs)
        function(ctypes.byref(instance))
        actual = {name: getattr(instance, name) for name in observed_names}
        mismatch = {name: {"expected": value, "actual": actual.get(name)} for name, value in expected.items() if actual.get(name) != value}
        if mismatch:
            violations.append({"scan": scan_index, "mismatch": mismatch})
        else:
            activations += 1
    return {
        "case_id": case.case_id,
        "status": "PASS" if not violations else "FAIL",
        "activations": activations,
        "violations": len(violations),
        "details": violations[:4],
    }


def verify_rewardplc(record: TaskRecord, code: str, config: VerifierConfig) -> dict[str, Any]:
    entry_point = f"FB_{record.task_id.replace('-', '_')}"
    work_parent = Path(config.mount_root) / ".rlmar_rewardplc_runs"
    work_parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=f"{record.task_id}-", dir=work_parent) as directory:
            work = Path(directory)
            try:
                library, header, compile_seconds = _compile(code, entry_point, config, work)
            except ValueError as error:
                return {"status": "COMPILE_FAIL", "diagnostic": str(error), "checks": [], "stages_completed": 1, "total_stages": 5}
            try:
                function, prototype = _load_abi(library, header, entry_point)
            except (AttributeError, OSError, ValueError) as error:
                return {"status": "LOAD_FAIL", "diagnostic": str(error), "checks": [], "stages_completed": 2, "total_stages": 5}
            observed = tuple(record.pou_interface["outputs"]) + tuple(record.pou_interface["internal"])
            checks = [_run_case(function, type(prototype), case, observed) for case in record.cases]
            passed = all(check["status"] == "PASS" for check in checks)
            return {
                "status": "PASS" if passed else "ASSERTION_FAIL",
                "diagnostic": None if passed else "behavioral oracle mismatch",
                "checks": checks,
                "stages_completed": 5 if passed else 4,
                "total_stages": 5,
                "compile_seconds": compile_seconds,
                "code_sha256": stable_sha256(code),
            }
    except VerificationInfrastructureError as error:
        return {"status": "ENVIRONMENT_ERROR", "diagnostic": str(error), "checks": [], "stages_completed": 0, "total_stages": 5}


def summarize_quality(result: Mapping[str, Any]) -> dict[str, float]:
    checks = result.get("checks") or []
    passed = sum(check.get("status") == "PASS" for check in checks)
    activations = sum(int(check.get("activations", 0)) for check in checks)
    opportunities = sum(max(int(check.get("activations", 0)) + int(check.get("violations", 0)), 1) for check in checks)
    violations = sum(int(check.get("violations", 0)) for check in checks)
    return {
        "stage_fraction": min(max(float(result.get("stages_completed", 0)) / max(float(result.get("total_stages", 5)), 1.0), 0.0), 1.0),
        "check_fraction": passed / len(checks) if checks else 0.0,
        "activation_fraction": activations / opportunities if opportunities else 0.0,
        "violation_reduction": 1.0 / (1.0 + violations) if checks else 0.0,
    }
