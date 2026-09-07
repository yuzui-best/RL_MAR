"""Compile synthetic ABI fixtures with the existing RuSTy toolchain."""
import argparse
import ctypes
import importlib.util
import json
from pathlib import Path
import sys

from aligned_abi import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / 'scripts'))
    spec = importlib.util.spec_from_file_location('fixture_engine', args.root / 'scripts/run_functional94.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    original = engine.parse_structs
    install(engine)
    reports = []
    fixtures = [
        ('scalar', 'VAR_IN_OUT value : INT; END_VAR\nVAR_OUTPUT result : INT; END_VAR', 'value := value + 1; result := value;', [{'name': 'value', 'type': 'INT', 'section': 'inout'}], '', 8),
        ('array', 'VAR_IN_OUT value : ARRAY[1..2] OF INT; END_VAR\nVAR_OUTPUT result : INT; END_VAR', 'value[1] := value[1] + 1; result := value[1];', [{'name': 'value', 'type': 'ARRAY[1..2] OF INT', 'section': 'inout'}], '', 8),
        ('nested', 'VAR child : TON; END_VAR\nVAR_OUTPUT result : INT; END_VAR', 'child(); result := child.ticks;', [], 'FUNCTION_BLOCK TON\nVAR_OUTPUT ticks : INT := 3; END_VAR\nticks := ticks + 1;\nEND_FUNCTION_BLOCK\n', 4),
        ('mismatch', 'VAR_INPUT value : INT; END_VAR\nVAR_OUTPUT result : INT; END_VAR', 'result := value;', [{'name': 'value', 'type': 'INT', 'section': 'inout'}], '', None),
    ]
    for name, declarations, body, inouts, prefix, expected in fixtures:
        source = prefix + 'FUNCTION_BLOCK Fixture\n' + declarations + '\n' + body + '\nEND_FUNCTION_BLOCK\n'
        task = dict(task_id=name, entry_point='Fixture', pou_declarations=inouts)
        lib, header, _ = engine.compile_source(args.root, 'paper_alignment_fixtures_20260907', 'synthetic', task, source, 120)
        baseline = 'parsed'
        try:
            original(header, 'Fixture', task)
        except engine.FunctionalError as error:
            baseline = type(error).__name__
        try:
            struct, _, targets = engine.parse_structs(header, 'Fixture', task)
        except engine.InterfaceError:
            assert name == 'mismatch'
            reports.append(dict(fixture=name, original=baseline, aligned='contract_rejected_before_execution', passed=True))
            continue
        library = ctypes.CDLL(str(lib))
        instance = struct()
        constructor = getattr(library, 'Fixture__ctor', None)
        if constructor:
            constructor.argtypes = [ctypes.POINTER(struct)]
            constructor(ctypes.byref(instance))
        if inouts:
            buffer = engine.iec_ctype(inouts[0]['type'])()
            if isinstance(buffer, ctypes.Array):
                buffer[0] = 7
            else:
                buffer.value = 7
            assert 'value' in targets
            instance.value = ctypes.addressof(buffer)
        function = library.Fixture
        function.argtypes = [ctypes.POINTER(struct)]
        function(ctypes.byref(instance))
        assert instance.result == expected, (name, instance.result, expected)
        reports.append(dict(fixture=name, original=baseline, aligned='executed', expected=expected, actual=instance.result, passed=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
