import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "framework")]

from experiments.comparison.rlmar.evaluate_five_models import (
    assert_prompt_isolated,
    public_prompt,
    public_validation,
)


def task():
    instruction = "Implement a bounded minimum stack."
    return {
        "task_id": "hf_1",
        "title": "StackMin",
        "category": "DataHandlingAndCommunication",
        "entry_point": "StackMin",
        "instruction": instruction,
        "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        "pou_interface": [{"section": "input", "name": "push", "type": "BOOL"}],
        "cases": [{"expected_output_trace": {"secret": [42]}}],
        "oracles": [{"expected": "secret"}],
        "stimulus": {"secret": [1]},
    }


def test_public_prompt_contains_frozen_instruction_and_interface():
    prompt = public_prompt(task())
    assert "Implement a bounded minimum stack." in prompt
    assert "INPUT push: BOOL" in prompt
    assert "secret" not in prompt and "42" not in prompt


def test_prompt_isolation_rejects_hidden_payload():
    value = task()
    prompt = public_prompt(value)
    assert_prompt_isolated(value, prompt)


def test_public_prompt_rejects_instruction_hash_mismatch():
    value = task()
    value["instruction_sha256"] = "0" * 64
    try:
        public_prompt(value)
    except ValueError as exc:
        assert "instruction hash mismatch" in str(exc)
    else:
        raise AssertionError("hash mismatch was accepted")


def test_public_validation_redacts_hidden_function_details():
    result = public_validation({
        "status": "FUNCTION_FAIL",
        "diagnostic": "expected SECRET=42 actual=7",
        "checks": [{"status": "PASS", "observations": ["SECRET"]}, {"status": "FUNCTION_FAIL"}],
    })
    assert result == {
        "status": "FUNCTION_FAIL",
        "stages_completed": 1,
        "checks_passed": 1,
        "diagnostic": "Compiled candidate failed hidden functional checks.",
    }


def load_tests(loader, tests, pattern):
    """Run the same contract functions under stdlib unittest and pytest."""
    import unittest
    return unittest.TestSuite(unittest.FunctionTestCase(value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value))
