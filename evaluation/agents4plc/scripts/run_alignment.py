"""Re-evaluate fixed retained programs using the published Agents4PLC verifier."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import contextlib
import copy
import csv
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import types

from normalize_semicolons import normalize, verify_insertions


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def official_counts(verdicts):
    counts = Counter(verdicts)
    total = len(verdicts)
    success, failure = counts["Satisfied"], counts["Violated"]
    unresolved = total - success - failure
    # Match the published if/elif arithmetic, including strict > for satisfaction.
    satisfied80 = success > 0.80 * total
    verified = satisfied80 or unresolved <= (1 - 0.80) * total
    return dict(satisfied=success, violated=failure,
                unknown=counts["Unknown"], error=counts["Error"],
                missing=counts["Missing"], total=total, definite=success + failure,
                verified=bool(total and verified),
                satisfied80=bool(total and satisfied80),
                all_satisfied=bool(total and success == total))


class ToolProcess:
    """Run the unchanged official call with an outer operational time limit."""

    CalledProcessError = subprocess.CalledProcessError
    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, events, timeout):
        self.events, self.timeout = events, timeout
        self.last_folder = None

    def getoutput(self, command):
        return subprocess.getoutput(command)

    def run(self, command, check=False, capture_output=False, text=False, **kwargs):
        shell = kwargs.pop("shell", False)
        requested_timeout = kwargs.pop("timeout", None)
        if not capture_output or not text or kwargs:
            raise ValueError("Unexpected official subprocess contract")
        if isinstance(command, list):
            folder = Path(command[command.index("-output") + 1])
            self.last_folder = folder
        else:
            if not shell or self.last_folder is None:
                raise ValueError("Unexpected official recovery subprocess")
            folder = self.last_folder
        timeout = min(self.timeout, requested_timeout or self.timeout)
        started = time.monotonic()
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True,
                                   start_new_session=True, shell=shell)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            stdout += "\nTimeout: alignment operational outer limit reached.\n"
        event = dict(command=command, returncode=process.returncode,
                     timed_out=timed_out, seconds=time.monotonic() - started,
                     outer_timeout_seconds=timeout, output=str(folder))
        prefix = f"ALIGN_{len(self.events):03d}"
        (folder / (prefix + "_STDOUT.txt")).write_text(stdout)
        (folder / (prefix + "_STDERR.txt")).write_text(stderr)
        put(folder / (prefix + "_EXECUTION.json"), event)
        self.events.append(event)
        if timed_out and requested_timeout is not None:
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
        if check and process.returncode and not timed_out:
            raise subprocess.CalledProcessError(process.returncode, command,
                                                output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


class PreserveFallback:
    def __init__(self, events):
        self.events = events

    def rmtree(self, path):
        """The official fallback overwrites NuSMV output; retain its first output."""
        path = Path(path)
        destination = path.with_name(path.name + "_prior_" + str(time.time_ns()))
        path.rename(destination)
        for event in self.events:
            if event["output"] == str(path):
                event["output"] = str(destination)


def load_official(path, events, timeout):
    tree = ast.parse(path.read_text())
    # ChatOpenAI is an unused import in this verification-only module.
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom)
                         and node.module == "langchain_openai")]
    module = types.ModuleType("published_agents4plc_verifier")
    exec(compile(tree, str(path), "exec"), module.__dict__)
    module.subprocess = ToolProcess(events, timeout)
    module.shutil = PreserveFallback(events)
    return module


def evaluate(job):
    out = Path(job["out"])
    result_file = out / "RESULT.json"
    if result_file.exists():
        return json.loads(result_file.read_text())
    out.mkdir(parents=True, exist_ok=True)
    record = json.loads(Path(job["record"]).read_text())
    source = record["selected_source"]
    edits = []
    if job["condition"] == "semicolon_compatibility":
        normalized, edits = normalize(source)
        verify_insertions(source, normalized, edits)
        source = normalized
    candidate = out / "candidate.st"
    candidate.write_text(source + "\n")
    events = []
    module = load_official(Path(job["official"]), events, job["outer_timeout"])
    started = time.monotonic()
    compiled = subprocess.run([job["rusty"], "--check", str(candidate)],
                              capture_output=True, text=True, timeout=60)
    (out / "compile.stdout.txt").write_text(compiled.stdout)
    (out / "compile.stderr.txt").write_text(compiled.stderr)
    compile_pass = compiled.returncode == 0
    properties = []
    with (out / "official_console.txt").open("w") as console:
        with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
            for index, definition in enumerate(job["task"]["properties_to_be_validated"]):
                entry = dict(index=index, definition=copy.deepcopy(definition))
                if not compile_pass:
                    entry.update(verdict="Missing", invoked=False,
                                 reason="COMPILE_BLOCKED")
                else:
                    folder = out / "properties" / str(index)
                    try:
                        first_event = len(events)
                        summaries = module.plcverif_validation(
                            str(candidate), [copy.deepcopy(definition)], base_dir=str(folder))
                        assert len(summaries) == 1
                        text = summaries[0]
                        # Match official evaluation: violated checked before satisfied.
                        verdict = ("Violated" if "violated" in text else
                                   "Satisfied" if "satisfied" in text else "Unknown")
                        if verdict == "Unknown":
                            logs = []
                            for event in events[first_event:]:
                                logs.extend(Path(event["output"]).glob("*.report.txt"))
                                logs.extend(Path(event["output"]).glob("ALIGN_*_STDOUT.txt"))
                                logs.extend(Path(event["output"]).glob("ALIGN_*_STDERR.txt"))
                            report_text = "\n".join(p.read_text(errors="replace") for p in logs)
                            if ("requirement is ERROR" in report_text
                                    or "Unable to generate the CFA" in report_text):
                                verdict = "Error"
                            elif "failed smv file generation" in text:
                                verdict = "Missing"
                        entry.update(verdict=verdict, invoked=True, summary=text)
                    except Exception as exc:
                        entry.update(verdict="Error", invoked=True,
                                     error=f"{type(exc).__name__}: {exc}")
                properties.append(entry)
                put(out / "PROGRESS.json", dict(completed=len(properties),
                    total=len(job["task"]["properties_to_be_validated"]),
                    task_id=job["task"]["task_id"]))
    counts = official_counts([p["verdict"] for p in properties])
    if not compile_pass:
        counts.update(verified=False, satisfied80=False, all_satisfied=False)
    result = dict(condition=job["condition"], model=job["model"],
                  task_id=job["task"]["task_id"],
                  difficulty="Easy" if "-E-" in job["task"]["task_id"] else "Medium",
                  compile=compile_pass, **counts, properties=properties, events=events,
                  source_record=job["record"], candidate=str(candidate),
                  normalization_edits=edits, new_model_calls=0,
                  generation_tokens=record["known_tokens"],
                  generation_calls=len(record["calls"]),
                  seconds=time.monotonic() - started)
    put(result_file, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "retained_programs")
    parser.add_argument("--rusty-bin", type=Path, required=True)
    parser.add_argument("--plcverif-bin", type=Path, required=True)
    parser.add_argument("--nuxmv-bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cbmc-bin", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--outer-timeout", type=int, default=1200)
    args = parser.parse_args()
    args.root, args.out = args.root.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    shim = args.out / "bin"
    shim.mkdir(exist_ok=True)
    for name, target in (("plcverif-cli", args.plcverif_bin),
                         ("nuXmv", args.nuxmv_bin),
                         ("cbmc", args.cbmc_bin)):
        if not target.is_file():
            raise FileNotFoundError(target)
        link = shim / name
        if not link.exists():
            link.symlink_to(target.resolve())
    os.environ["PATH"] = str(shim) + os.pathsep + os.environ["PATH"]
    os.environ["nuXmv_PATH"] = str(shim)
    os.environ["CBMC_PATH"] = str(shim)
    jobs = []
    for model in ("gpt-4o", "gpt-4o-mini"):
        parent = args.root / model
        properties = json.loads((parent / "data/private_properties.json").read_text())
        tasks = {task["task_id"]: dict(task, properties_to_be_validated=properties[task["task_id"]])
                 for task in json.loads((parent / "data/public_tasks.json").read_text())}
        files = sorted((parent / "generation").glob("*.json"))
        assert len(files) == 23 and len(tasks) == 23
        for condition in ("original_source", "semicolon_compatibility"):
            for file in files:
                record = json.loads(file.read_text())
                jobs.append(dict(condition=condition, model=model, record=str(file),
                    task=tasks[record["task_id"]],
                    out=str(args.out / condition / model / record["task_id"]),
                    official=str(Path(__file__).resolve().parents[1] / "official_source/plcverif.py"),
                    rusty=str(args.rusty_bin.resolve()),
                    outer_timeout=args.outer_timeout))
    put(args.out / "RUN_CONFIG.json", dict(
        conditions=["original_source", "semicolon_compatibility"], workflows=92,
        source_batch="formal_feedbackfix3_20261113", new_model_calls=0,
        verifier="published src/plcverif.py (master)", backend_timeout="published tool default",
        fallback="official NuSMV-to-CBMC branch", cbmc=str(args.cbmc_bin),
        outer_timeout=args.outer_timeout, workers=args.workers,
        thresholds="official >0.8 satisfied; <=(1-0.8) not_verified",
        source_normalization="separate condition: missing compound-statement semicolons only"))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # map preserves the predeclared order, with no score-dependent selection.
        for result in pool.map(evaluate, jobs):
            results.append(result)
            put(args.out / "PROGRESS.json", dict(completed=len(results), total=len(jobs),
                latest=result["task_id"], latest_model=result["model"]))
    put(args.out / "RESULTS.json", results)
    fields = ["condition", "model", "task_id", "difficulty", "compile", "satisfied",
              "violated", "unknown", "error", "missing", "total", "definite",
              "verified", "satisfied80", "all_satisfied", "generation_tokens",
              "generation_calls", "seconds", "source_record", "candidate"]
    with (args.out / "source_data.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    summaries = []
    for condition in ("original_source", "semicolon_compatibility"):
        for model in ("gpt-4o", "gpt-4o-mini"):
            for difficulty in ("Easy", "Medium", "All"):
                group = [r for r in results if r["condition"] == condition and r["model"] == model
                         and (difficulty == "All" or r["difficulty"] == difficulty)]
                summaries.append(dict(condition=condition, model=model,
                    difficulty=difficulty, n=len(group),
                    mean_generation_tokens=sum(r["generation_tokens"] for r in group)/len(group),
                    mean_generation_calls=sum(r["generation_calls"] for r in group)/len(group),
                    mean_reverification_seconds=sum(r["seconds"] for r in group)/len(group),
                    **{key: sum(r[key] for r in group) for key in
                       ("compile", "verified", "satisfied80", "all_satisfied",
                        "satisfied", "violated", "unknown", "error", "missing", "total")}))
    put(args.out / "SUMMARY.json", dict(status="COMPLETE", new_model_calls=0, groups=summaries))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
