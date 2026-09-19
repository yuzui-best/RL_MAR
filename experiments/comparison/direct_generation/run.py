"""Run the direct-generation baseline through an OpenAI-compatible endpoint.

The script writes a canonical candidate package.  It never overwrites the
published ``results.json`` unless that path is explicitly passed to --output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

COMPARISON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPARISON))
from _common import dump_json, load_env_file, load_records, task_id, task_prompt  # noqa: E402


SYSTEM_PROMPT = (
    "Generate IEC 61131-3 Structured Text for the requested PLC task. "
    "Return only the program text, without Markdown fences or explanation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def endpoint(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value if value.endswith("/chat/completions") else value + "/chat/completions"


def request_completion(args: argparse.Namespace, prompt: str) -> tuple[str, dict]:
    key = os.environ.get(args.api_key_env)
    if not key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        endpoint(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    content = body["choices"][0]["message"]["content"]
    return content.strip(), body.get("usage", {})


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    records = load_records(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    package = {"schema_version": 1, "method": "direct_generation", "model": args.model, "tasks": []}
    if args.resume and args.output.exists():
        package = json.loads(args.output.read_text(encoding="utf-8"))
    completed = {str(row["task_id"]) for row in package.get("tasks", [])}

    for index, record in enumerate(records, start=1):
        identifier = task_id(record, index)
        if identifier in completed:
            continue
        started = time.perf_counter()
        try:
            code, usage = request_completion(args, task_prompt(record))
            row = {
                "task_id": identifier,
                "generated_code": code,
                "seconds": round(time.perf_counter() - started, 6),
                "usage": usage,
                "error": None,
            }
        except Exception as exc:  # preserve partial runs for expensive experiments
            row = {
                "task_id": identifier,
                "generated_code": None,
                "seconds": round(time.perf_counter() - started, 6),
                "usage": {},
                "error": str(exc),
            }
        package.setdefault("tasks", []).append(row)
        dump_json(args.output, package)
        print(f"[{len(package['tasks'])}/{len(records)}] {identifier}", flush=True)


if __name__ == "__main__":
    main()
