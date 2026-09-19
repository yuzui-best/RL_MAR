"""Small shared helpers for the public comparison runners."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load a JSON or JSONL benchmark without assuming one exact envelope."""
    if path.suffix.lower() == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    value = load_json(path)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("tasks", "data", "records", "examples"):
            if isinstance(value.get(key), list):
                return value[key]
    raise ValueError(f"Unsupported benchmark envelope: {path}")


def task_id(record: dict[str, Any], index: int) -> str:
    for key in ("task_id", "id", "name", "problem_id"):
        if record.get(key) not in (None, ""):
            return str(record[key])
    return f"task_{index:03d}"


def task_prompt(record: dict[str, Any]) -> str:
    for key in ("prompt", "description", "problem", "instruction", "query"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("Task has no prompt/description/problem/instruction/query field")


def load_env_file(path: Path | None) -> None:
    if path is None:
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)
