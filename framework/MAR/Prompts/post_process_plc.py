
import importlib.util
import itertools
import json
import multiprocessing
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from loguru import logger
from MAR.Tools.PLC.validators import st_syntax_checker_tool
from MAR.Core.dataset import load_reward_records
from MAR.Core.verifier import VerifierConfig, verify_rewardplc


_FUNCTIONAL_ATTEMPTS = itertools.count(1)
_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_COMPILER_CONSTRAINTS = [
    "Generate IEC 61131-3 Structured Text accepted by the configured compiler.",
    "Preserve the exact entry point, interface names, types, and array bounds.",
    "Return one complete POU and do not call unsupported vendor libraries.",
]

# Pre-compiled ST code block extraction regex
_ST_CODE_PATTERN = re.compile(r"```st(.*?)```", re.DOTALL | re.MULTILINE)
EXTRACTION_PROFILES = ("baseline", "fence_labels", "bare_pou", "duplicate_blocks")
NORMALIZATION_PROFILES = ("baseline", "line_endings", "encoding_whitespace")
_FENCE_LABELS = {
    "baseline": ("st",),
    "fence_labels": ("st", "ST", "iecst"),
    "duplicate_blocks": ("st",),
}
_POU_END = {
    "FUNCTION_BLOCK": "END_FUNCTION_BLOCK",
    "FUNCTION": "END_FUNCTION",
    "PROGRAM": "END_PROGRAM",
    "TYPE": "END_TYPE",
}


def _load_functional_runner(runner_path: Path | str, module_name: str):
    path = Path(runner_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None
    runner = importlib.util.module_from_spec(spec)
    script_dir = str(path.parent)
    added = script_dir not in sys.path
    if added:
        sys.path.insert(0, script_dir)
    try:
        spec.loader.exec_module(runner)
    finally:
        if added:
            sys.path.remove(script_dir)
    return runner


def _functional_worker_process(runner_path: str, request: dict, queue) -> None:
    runner = _load_functional_runner(runner_path, "plc94_inline_worker")
    if runner is None:
        queue.put(None)
        return
    queue.put(runner.worker(request))

# CODESYS error formatting template
_CODESYS_ERROR_HEADER = "\n\n[CODESYS Compilation Report]: The ST code above failed CODESYS compilation.\n"
_CODESYS_ERROR_FOOTER = "\nPlease fix ALL errors based on this report and regenerate the correct code."


def _fenced_matches(response: str, labels: tuple[str, ...]) -> list[str]:
    alternatives = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(
        rf"```(?:{alternatives})[ \t]*\r?\n(.*?)```",
        re.DOTALL | re.MULTILINE,
    )
    return [match.group(1).strip() for match in pattern.finditer(response)]


def extract_st_code(response: str, profile: str = "baseline") -> str:
    """Extract one unambiguous ST candidate under an explicit profile."""
    if profile not in EXTRACTION_PROFILES:
        raise ValueError(f"unknown extraction profile: {profile}")
    if profile == "baseline":
        match = _ST_CODE_PATTERN.search(response)
        return match.group(1).strip() if match else ""
    if profile == "bare_pou":
        text = response.strip()
        if "```" in text:
            return ""
        if not re.match(r"(?i)^\s*(FUNCTION_BLOCK|FUNCTION|PROGRAM|TYPE)\b", text):
            return ""
        starts = re.findall(
            r"(?im)^\s*(FUNCTION_BLOCK|FUNCTION|PROGRAM|TYPE)\b",
            text,
        )
        if len(starts) != 1:
            return ""
        terminator = _POU_END[starts[0].upper()]
        ends = re.findall(rf"(?im)^\s*{terminator}\s*;?\s*$", text)
        if len(ends) != 1 or not re.search(rf"(?i){terminator}\s*;?\s*$", text):
            return ""
        return text

    matches = _fenced_matches(response, _FENCE_LABELS[profile])
    if profile == "fence_labels":
        return matches[0] if len(matches) == 1 else ""
    if not matches:
        return ""
    normalized = [match.replace("\r\n", "\n").replace("\r", "\n").strip() for match in matches]
    return normalized[0] if len(set(normalized)) == 1 else ""


def _extract_st_code(response: str) -> str:
    """Compatibility entry point configured once per experiment process."""
    return extract_st_code(response, os.getenv("RL_MAR_EXTRACTION_PROFILE", "baseline"))


def _canonicalize_terminators(code: str) -> str:
    return re.sub(
        r"(?im)^(\s*END_(?:FUNCTION_BLOCK|FUNCTION|PROGRAM|TYPE))\s*;\s*$",
        r"\1",
        code,
    )


def normalize_st(code: str, profile: str = "baseline") -> tuple[str, list[str]]:
    """Apply one audited, semantics-preserving normalization profile."""
    if profile not in NORMALIZATION_PROFILES:
        raise ValueError(f"unknown normalization profile: {profile}")
    baseline = _canonicalize_terminators(code)
    normalized = code
    if profile == "line_endings":
        lines = normalized.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        normalized = "\n".join(line.rstrip() for line in lines).rstrip()
    elif profile == "encoding_whitespace":
        normalized = normalized.removeprefix("\ufeff")
        normalized = re.sub(
            r"(?m)^[ \t\u00a0]+",
            lambda match: match.group(0).replace("\u00a0", " "),
            normalized,
        )
    before_terminators = normalized
    normalized = _canonicalize_terminators(normalized)
    if profile == "line_endings":
        normalized = normalized.rstrip() + "\n"
    transforms: list[str] = []
    if profile != "baseline" and normalized != baseline:
        transforms.append(profile)
    if before_terminators != _canonicalize_terminators(before_terminators):
        transforms.append("terminator_semicolon")
    return normalized, transforms


def canonicalize_st(code: str) -> str:
    """Compatibility entry point configured once per experiment process."""
    profile = os.getenv("RL_MAR_NORMALIZATION_PROFILE", "baseline")
    return normalize_st(code, profile)[0]


def replace_st_code(response: str, code: str) -> str:
    replacement = f"```st\n{code.strip()}\n```"
    if _ST_CODE_PATTERN.search(response):
        return _ST_CODE_PATTERN.sub(replacement, response, count=1)
    return replacement


def _functional_root() -> Path | None:
    configured = os.getenv("PLC_FUNCTIONAL94_ROOT")
    if not configured:
        return None
    root = Path(configured).resolve()
    return root if root.is_dir() else None


@lru_cache(maxsize=2)
def _rewardplc_index(configured_path: str) -> dict[str, object]:
    return {record.task_id: record for record in load_reward_records(configured_path)}


def _rewardplc_task(task_id: str):
    configured = os.getenv("PLC_REWARDPLC_PATH")
    if not configured or not task_id:
        return None
    return _rewardplc_index(str(Path(configured).resolve())).get(task_id)


def _functional_task(task_id: str) -> dict | None:
    root = _functional_root()
    if root is None or not task_id:
        return None
    manifest_name = os.getenv("PLC_FUNCTIONAL94_MANIFEST", "plc94.functional.v1.1.jsonl")
    manifest = root / "data" / manifest_name
    if not manifest.is_file():
        return None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        task = json.loads(line)
        if task.get("task_id") == task_id:
            return task
    return None


def functional_context(task_id: str) -> str:
    reward_task = _rewardplc_task(task_id)
    if reward_task is not None:
        payload = {
            "entry_point": f"FB_{reward_task.task_id.replace('-', '_')}",
            "interface": reward_task.pou_interface,
            "compiler_constraints": _COMPILER_CONSTRAINTS,
        }
        return "\n\nExecutable acceptance contract:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    task = _functional_task(task_id)
    if task is None:
        return ""
    payload = {
        "entry_point": task.get("entry_point"),
        "interface": task.get("pou_interface", []),
        "compiler_constraints": _COMPILER_CONSTRAINTS,
    }
    return "\n\nExecutable acceptance contract:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def functional_available(task_id: str) -> bool:
    return _rewardplc_task(task_id) is not None or _functional_task(task_id) is not None


def functional_repair_instructions() -> str:
    profile = os.getenv("RL_MAR_REPAIR_PROMPT_PROFILE", "iec")
    if profile == "baseline":
        return "Preserve the public contract and correct only the reported validation stage."
    if profile != "iec":
        raise ValueError(f"unknown repair prompt profile: {profile}")
    return (
        "The executable contract is authoritative if older task prose or the previous code conflicts "
        "with it. Preserve the contract's exact entry point, variable sections, types, and array bounds. "
        "IEC 61131-3 logic is evaluated in scan order: changing a state does not execute the new state's "
        "branch in the same scan. Satisfy same-cycle postconditions with explicit assignments in the "
        "currently executing branch. Address every reported mismatch with a behavioral code change; "
        "do not merely add comments or rename state labels."
    )


def run_functional_check(task_id: str, code: str) -> dict | None:
    reward_task = _rewardplc_task(task_id)
    if reward_task is not None:
        rusty = os.getenv("PLC_RUSTY_PATH")
        mount_root = os.getenv("PLC_VERIFIER_MOUNT_ROOT")
        if not rusty or not mount_root:
            return {"status": "ENVIRONMENT_ERROR", "diagnostic": "PLC_RUSTY_PATH and PLC_VERIFIER_MOUNT_ROOT must be configured", "checks": []}
        return verify_rewardplc(
            reward_task,
            canonicalize_st(code),
            VerifierConfig(
                rusty=Path(rusty).resolve(),
                mount_root=Path(mount_root).resolve(),
                docker_image=os.getenv("PLC_VERIFIER_DOCKER_IMAGE", "ubuntu:24.04"),
                timeout_seconds=int(os.getenv("PLC_FUNCTIONAL_COMPILE_TIMEOUT", "120")),
            ),
        )
    root = _functional_root()
    task = _functional_task(task_id)
    if root is None or task is None:
        return None
    runner_path = root / "scripts/run_functional94.py"
    runner = _load_functional_runner(runner_path, "plc94_inline_runner")
    if runner is None:
        return None
    definition_error = runner.task_definition_error(task)
    if definition_error:
        return {
            "method": "RL-MAR",
            "task_id": task_id,
            "status": "DATASET_ERROR",
            "diagnostic": definition_error,
            "checks": [],
        }
    request = {
        "root": str(root),
        "run_id": os.getenv("PLC_FUNCTIONAL_RUN_ID", "rlmar_inline"),
        "method": "RL-MAR",
        "method_slug": f"rl_mar_{next(_FUNCTIONAL_ATTEMPTS)}",
        "task": task,
        "source": canonicalize_st(code),
        "test_vector_sha256": runner.stable_test_hash(task),
        "compile_timeout": int(os.getenv("PLC_FUNCTIONAL_COMPILE_TIMEOUT", "120")),
    }
    context = multiprocessing.get_context("fork")
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_functional_worker_process,
        args=(str(runner_path), request, queue),
    )
    process.start()
    process.join(int(os.getenv("PLC_FUNCTIONAL_TASK_TIMEOUT", "180")))
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "method": "RL-MAR",
            "task_id": task_id,
            "status": "TIMEOUT",
            "diagnostic": "task timeout",
            "checks": [],
        }
    return queue.get() if not queue.empty() else {
        "method": "RL-MAR",
        "task_id": task_id,
        "status": "RUNTIME_FAIL",
        "diagnostic": f"functional worker exited with code {process.exitcode}",
        "checks": [],
    }


def sanitize_compiler_diagnostic(value: object, limit: int | None = None) -> str:
    if limit is None:
        limit = int(os.getenv("RL_MAR_COMPILER_DIAGNOSTIC_LIMIT", "4000"))
    if limit not in {1000, 2000, 4000}:
        raise ValueError(f"unsupported compiler diagnostic limit: {limit}")
    return _ANSI_ESCAPE.sub("", str(value or ""))[-limit:]


def public_functional_summary(result: dict) -> dict:
    """Project verifier output onto the only fields safe outside the hidden evaluator."""
    status = result.get("status", "RUNTIME_FAIL")
    checks = result.get("checks") or []
    summary = {
        "status": status,
        "stages_completed": int(result.get("stages_completed", 0) or 0),
        "checks_passed": sum(check.get("status") == "PASS" for check in checks),
        "checks_total": len(checks),
    }
    if status in {"COMPILE_FAIL", "COMPILE_ERROR"}:
        summary["diagnostic"] = sanitize_compiler_diagnostic(result.get("diagnostic"))
    return summary


def functional_feedback(result: dict, task_id: str = "") -> str:
    summary = public_functional_summary(result)
    return "Executable validation summary:\n" + json.dumps(
        summary, ensure_ascii=False, separators=(",", ":")
    )


def functional_score(result: dict | None) -> tuple[int, int, int, int, int]:
    """Rank candidates without using model-specific or task-specific rules."""
    if result is None:
        return (0, 0, 0, 0, 0)
    status = result.get("status", "RUNTIME_FAIL")
    checks = result.get("checks") or []
    passed = sum(check.get("status") == "PASS" for check in checks)
    activated = sum(int(check.get("activations", 0) or 0) > 0 for check in checks)
    violations = sum(int(check.get("violations", 0) or 0) for check in checks)
    stages_completed = int(result.get("stages_completed", 0) or 0)
    return (int(status == "PASS"), stages_completed, passed, activated, -violations)


# IEC 61131-3 reserved words that conflict with built-in functions
_RESERVED_NAMES = {
    'min': 'MIN()', 'max': 'MAX()', 'len': 'LEN()', 'limit': 'LIMIT()',
    'sel': 'SEL()', 'mux': 'MUX()', 'sin': 'SIN()', 'cos': 'COS()',
    'tan': 'TAN()', 'exp': 'EXP()', 'expt': 'EXPT()', 'abs': 'ABS()',
    'sqrt': 'SQRT()', 'log': 'LOG()', 'ln': 'LN()', 'trunc': 'TRUNC()',
}


def _format_codesys_errors(errors: list) -> str:
    """Compress CODESYS errors by Error ID, filter cascade noise, add root-cause hints."""
    if not errors:
        return "Unknown compilation error."

    from collections import OrderedDict

    # Group by Error ID, preserving order of first occurrence
    groups = OrderedDict()
    for e in errors:
        eid = e.get("ID", "Unknown")
        if eid not in groups:
            groups[eid] = []
        groups[eid].append(e)

    PURE_CASCADE = {'C0189'}       # "expected X instead of Y" — pure parser noise
    SECONDARY = {'C0032', 'C0035'}  # type mismatch / need function name — cascade but informative

    root_lines = []
    secondary_lines = []
    cascade_lines = []

    for eid, group in groups.items():
        first = group[0]
        desc = first.get("ErrorDesc", str(first))
        count = len(group)
        count_str = f" x{count}" if count > 1 else ""
        hint = None

        if eid == 'C0046':  # Undefined identifier
            m = re.search(r"标识符'(\w+)'", desc)
            if m:
                name = m.group(1)
                lower = name.lower()
                if lower in _RESERVED_NAMES:
                    hint = f"HINT: '{name}' conflicts with built-in {_RESERVED_NAMES[lower]}. Rename this variable (e.g., {name}Value)."
                elif len(name) == 1 and name.isalpha():
                    hint = (f"HINT: '{name}' appears to be an undeclared FOR loop counter. "
                            "In IEC 61131-3, ALL loop counters must be explicitly declared "
                            f"in the VAR section (e.g., \"i : INT;\"). Add the declaration and retry.")
                else:
                    hint = f"HINT: '{name}' is not a standard IEC 61131-3 function. Use built-in equivalent (e.g., MIN) or define it yourself."

        elif eid == 'C0009':  # Unexpected token
            m = re.search(r"记号'(\w+)'", desc)
            if m:
                token = m.group(1)
                lower = token.lower()
                if lower in _RESERVED_NAMES:
                    hint = f"HINT: '{token}' is a reserved word (built-in {_RESERVED_NAMES[lower]}). Rename this variable."
                elif lower in ('function', 'end_function'):
                    hint = ("HINT: Nested POU detected. You cannot declare a FUNCTION inside a "
                            "FUNCTION_BLOCK. In IEC 61131-3, each POU must be standalone. "
                            "Move the nested POU outside, after END_FUNCTION_BLOCK of the main block.")

        elif eid == 'C0212':
            hint = ("HINT: Structural error — you cannot nest a FUNCTION or FUNCTION_BLOCK "
                    "inside another POU. In IEC 61131-3, each POU must be a standalone "
                    "top-level unit. Move the nested POU outside, placing it after the "
                    "END_FUNCTION_BLOCK of the main block.")

        elif eid == 'C0032':
            hint = "(cascade: type mismatch caused by an undefined identifier — fix C0046 errors first)"
        elif eid == 'C0035':
            hint = "(cascade: caused by an undefined identifier — fix C0046 errors first)"
        elif eid == 'C0189':
            hint = "(cascade: parser confused by an earlier unexpected token — fix C0009 errors first)"

        line = f"[{eid}{count_str}] {desc}"
        if hint:
            line += f"  ← {hint}"

        if eid in PURE_CASCADE:
            cascade_lines.append(line)
        elif eid in SECONDARY:
            secondary_lines.append(line)
        else:
            root_lines.append(line)

    all_lines = root_lines + secondary_lines + cascade_lines
    total = len(errors)
    unique = len(groups)
    header = f"{total} errors compressed to {unique} unique types:\n"

    return header + "\n".join(all_lines)


def _build_rename_hint(errors: list) -> str:
    """Scan CODESYS errors for naming conflicts with IEC 61131-3 built-ins.

    Returns a prominent rename instruction block, or empty string if none found.
    """
    conflicts = {}
    for e in errors:
        if e.get("ID") == "C0009":
            m = re.search(r"记号'(\w+)'", e.get("ErrorDesc", ""))
            if m:
                token = m.group(1)
                lower = token.lower()
                if lower in _RESERVED_NAMES:
                    conflicts[lower] = token

    if not conflicts:
        return ""

    items = []
    for low, orig in conflicts.items():
        suggestion = f"{orig}Value"
        items.append(f"  - '{orig}' conflicts with built-in {_RESERVED_NAMES[low]}. Rename to '{suggestion}'.")

    return (
        "\n\n"
        "========================================\n"
        "CRITICAL: Variable name conflicts with IEC 61131-3 built-in functions.\n"
        "Fix these FIRST — they cause cascading parser errors that hide other issues:\n"
        + "\n".join(items) +
        "\n========================================"
    )


def post_process(raw_inputs, response, method="None"):
    """
    Post-process agent output.

    Args:
        raw_inputs: Original input data
        response: Raw response from LLM
        method: Post-processing method name (e.g. "STSyntaxCheck", "CodesysSemanticCheck")

    Returns:
        tuple: (Post-processed response text, validation_info dict)
        validation_info = {"passed": bool, "errors": [...]}
    """

    # 1. No post-processing: return original response
    if method == "None" or not method:
        return response, {"passed": True, "errors": []}

    # 2. ANTLR syntax check
    elif method == "STSyntaxCheck":
        code_snippet = _extract_st_code(response)
        if code_snippet:
            check_result = st_syntax_checker_tool(code_snippet)
            if check_result['passed']:
                return response, {"passed": True, "errors": []}
            else:
                error_msg = check_result.get('error', 'Unknown syntax error')
                error_feedback = (
                    f"\n\n[System Feedback]: The ST code above contains syntax errors:\n"
                    f"{error_msg}\n"
                    f"Please fix these errors."
                )
                return response + error_feedback, {"passed": False, "errors": [{"ErrorDesc": error_msg}]}
        else:
            return response, {"passed": True, "errors": []}

    # 3. CODESYS semantic validation
    elif method == "CodesysSemanticCheck":
        try:
            from MAR.Tools.PLC.codesys_client import get_codesys_client
            client = get_codesys_client()
        except ImportError:
            logger.warning("CODESYS client not available (import failed), falling back to ANTLR syntax check.")
            resp, info = post_process(raw_inputs, response, "STSyntaxCheck")
            info["codesys_used"] = False
            return resp, info

        code_snippet = _extract_st_code(response)
        if not code_snippet:
            return response, {"passed": True, "errors": [], "codesys_used": False}

        try:
            result = client.compile_code(code_snippet)
        except Exception as e:
            logger.warning(f"CODESYS server unreachable ({client.host}:{client.port}): {e}, falling back to ANTLR syntax check.")
            resp, info = post_process(raw_inputs, response, "STSyntaxCheck")
            info["codesys_used"] = False
            return resp, info

        if result.get("success", False):
            return response, {"passed": True, "errors": [], "codesys_used": True}
        else:
            errors = result.get("errors", [])
            error_text = _format_codesys_errors(errors)
            error_feedback = f"{_CODESYS_ERROR_HEADER}{error_text}{_CODESYS_ERROR_FOOTER}"
            return response + error_feedback, {"passed": False, "errors": errors, "codesys_used": True}

    # 4. Unknown method: fallback to original
    else:
        return response, {"passed": True, "errors": []}
