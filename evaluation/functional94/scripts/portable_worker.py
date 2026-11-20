"""Frozen exact-trace evaluator with configurable RuSTy build paths.

Run this in a subprocess: native candidate execution must not share the runner's
process. ABI installation and the scan evaluator match the experiment snapshot.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import run_functional94 as engine
import aligned_abi


def install():
    aligned_abi.install(engine)

    def compile_source(root, run_id, method_slug, task, source, timeout):
        work = root / "runs" / run_id / "build" / method_slug / task["task_id"]
        work.mkdir(parents=True, exist_ok=False)
        (work / "program.st").write_text(source)
        rusty = Path(os.environ["RUSTY_BIN"]).resolve()
        image = os.environ.get("RUSTY_DOCKER_IMAGE")
        if image:
            mount = Path(os.environ.get("RUSTY_MOUNT_ROOT") or "/").resolve()
            command = ["docker", "run", "--rm", "-v", f"{mount}:{mount}",
                       "-w", str(work), image, str(rusty)]
        else:
            command = [str(rusty)]
        started = time.monotonic()
        # The compiler's cwd must be work, including in native mode.
        engine.run_checked(command + ["--header-output", str(work), "-o",
            str(work / "program"), "--generate-headers", str(work / "program.st")], timeout)
        engine.run_checked(command + [str(work / "program.st"), "-o",
            str(work / "program.o"), "-c", "--fpic", "--target",
            "x86_64-linux-gnu"], timeout)
        library, header = work / "program.so", work / "program.h"
        engine.run_checked([shutil.which("cc") or "cc", "-shared", "-o",
                            str(library), str(work / "program.o")], timeout)
        if not header.is_file():
            raise RuntimeError("RuSTy did not generate program.h")
        return library, header, time.monotonic() - started

    engine.compile_source = compile_source


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    install()
    result = engine.worker(json.loads(args.worker.read_text()))
    args.response.write_text(json.dumps(result, default=str) + "\n")
