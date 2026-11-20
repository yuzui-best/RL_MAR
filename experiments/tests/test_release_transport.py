"""Provider-free checks of prompt reservation and measured token accounting."""
import asyncio
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "framework")]
from experiments.comparison.rlmar.evaluate_five_models import Backend
from MAR.V2.budget import BudgetExceeded
from MAR.V2.request_budget import RequestBudget, request_budget


class TransportTests(unittest.TestCase):
    def test_model_specific_envelopes_match_frozen_clients(self):
        encoding = SimpleNamespace(encode=lambda text, **kwargs: list(range(len(text))))
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs:
                                    SimpleNamespace(ids=list(range(len(text)))))
        modules = {
            "tiktoken": SimpleNamespace(get_encoding=lambda _: encoding),
            "tokenizers": SimpleNamespace(Tokenizer=SimpleNamespace(
                from_file=lambda _: tokenizer))}
        messages = [{"role": "user", "content": "abcd"}]
        with patch.dict(sys.modules, modules), patch.dict(os.environ,
                {"DEEPSEEK_V4_FLASH_TOKENIZER": "not-read.json",
                 "GPT55_PROMPT_EXTRA_RESERVE": "512"}):
            self.assertEqual(132, Backend("deepseek-v4-flash").prompt_tokens(messages))
            self.assertEqual(148, Backend("gpt-4o").prompt_tokens(messages))
            self.assertEqual(660, Backend("gpt-5.5").prompt_tokens(messages))

    def run_call(self, usage, prompt=3000, cap=2048, total=4096):
        options = []

        class Client:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def create(self, **kwargs):
                options.append(kwargs)
                return SimpleNamespace(usage=usage, model="gpt-4o",
                    choices=[SimpleNamespace(finish_reason="stop",
                                             message=SimpleNamespace(content="program"))])

        backend = Backend("gpt-4o")
        backend.prompt_tokens = lambda messages: prompt
        self.backend, self.options = backend, options
        environment = dict(GPT4O_URL="http://invalid.local/v1", GPT4O_KEY="test",
                           GPT4O_MODEL="gpt-4o")
        budget = RequestBudget(2, total, 180)
        with patch.dict(os.environ, environment), patch.dict(sys.modules,
                {"openai": SimpleNamespace(AsyncOpenAI=Client)}), request_budget(budget):
            return asyncio.run(backend.agen([{"role": "user", "content": "public"}],
                                            max_tokens=cap))

    def test_prompt_is_reserved_before_completion(self):
        self.assertEqual("program", self.run_call(
            SimpleNamespace(prompt_tokens=3000, completion_tokens=1000)))
        self.assertEqual(1096, self.options[0]["max_tokens"])
        self.assertEqual(4000, self.backend.spent)

    def test_provider_completion_overrun_retains_cost(self):
        with self.assertRaisesRegex(BudgetExceeded, "measured_completion_cap"):
            self.run_call(SimpleNamespace(prompt_tokens=3000, completion_tokens=1200))
        self.assertEqual(4200, self.backend.spent)
        self.assertTrue(self.backend.calls[0]["over_call_budget"])

    def test_no_request_when_prompt_exhausts_budget(self):
        with self.assertRaises(BudgetExceeded):
            self.run_call(None, prompt=4096)
        self.assertEqual([], self.options)

    def test_missing_usage_is_not_zero_cost(self):
        with self.assertRaisesRegex(RuntimeError, "token usage"):
            self.run_call(None)
        self.assertEqual("UNKNOWN_USAGE", self.backend.calls[0]["status"])
        self.assertEqual(4096, self.backend.calls[0]["unknown_upper"])

    def test_prompt_reservation_underestimate_retains_cost(self):
        with self.assertRaisesRegex(BudgetExceeded, "prompt_reservation"):
            self.run_call(SimpleNamespace(prompt_tokens=3050, completion_tokens=1000))
        self.assertEqual(4050, self.backend.spent)
        self.assertTrue(self.backend.calls[0]["prompt_under_reserved"])


if __name__ == "__main__":
    unittest.main()
