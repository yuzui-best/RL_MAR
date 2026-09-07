"""Compile and evaluate a new candidate using the isolated ABI adapter."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from aligned_abi import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--response', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    scripts = Path(request['root']) / 'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('aligned_engine', scripts / 'run_functional94.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    install(engine)
    args.response.write_text(json.dumps(engine.worker(request), indent=2) + '\n')


if __name__ == '__main__':
    main()
