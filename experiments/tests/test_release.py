"""Offline release checks: recorded results, measured rewards and frozen routes."""
import collections
import csv
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "framework")]


def read(path):
    return json.loads((ROOT / path).read_text())


def rows(path):
    with (ROOT / path).open() as handle:
        return list(csv.DictReader(handle))


def truth(value):
    return str(value).lower() in ("1", "true")


class PublishedDataTests(unittest.TestCase):
    def test_main_comparison_matches_current_totals(self):
        data = rows("experiments/results/main_comparison/source_data.csv")
        self.assertEqual(2256, len(data))
        expected = {"rlmar": (189, 483), "autoplc": (167, 491),
                    "direct": (147, 422), "llm4plc": (87, 489)}
        for method, (passes, compiled) in expected.items():
            group = [r for r in data if r["method"] == method]
            self.assertEqual(564, len(group))
            self.assertEqual(passes, sum(truth(r["functional_pass"]) for r in group))
            self.assertEqual(compiled, sum(truth(r["compile_pass"]) for r in group))
            self.assertTrue(all(r["call_seconds"] and not truth(r["timing_missing"]) for r in group))
            self.assertEqual(564, len({(r["model"], r["task_id"]) for r in group}))

    def test_matched_budget_has_all_conditions(self):
        data = rows("experiments/results/matched_budget/source_data.csv")
        self.assertEqual(6768, len(data))
        conditions = collections.Counter((r["model"], r["method"], r["budget"]) for r in data)
        self.assertEqual(72, len(conditions))
        self.assertEqual({94}, set(conditions.values()))
        expected = {"4096": 164, "8192": 174, "16384": 189}
        for budget, passes in expected.items():
            self.assertEqual(passes, sum(truth(r["passed"]) for r in data
                if r["method"] == "rlmar" and r["budget"] == budget))

    def test_six_arm_rows_and_repeated_task_denominators(self):
        data = rows("experiments/results/ablation/six_arm/source_data.csv")
        expected = {"learned": 98, "analyzer": 96, "fixed": 90,
                    "single": 81, "rule": 81, "costrule450": 90}
        for arm, passes in expected.items():
            group = [r for r in data if r["arm"] == arm]
            self.assertEqual(282, len(group))
            self.assertEqual(passes, sum(truth(r["functional_pass"]) for r in group))
            self.assertEqual({3}, set(collections.Counter(r["task_id"] for r in group).values()))
            self.assertTrue(all(r["source_record"] for r in group))
        self.assertEqual(set(expected), {r["arm"] for r in data})

    def test_development_has_full_measured_feasible_matrix(self):
        rewards = read("experiments/training/development_rewards.json")
        self.assertEqual(454, len(rewards))
        self.assertEqual(450, sum(r["stage"] == "action_matrix" for r in rewards))
        self.assertTrue(all(r["status"] == "COMPLETE" and r["unknown_upper"] == 0 for r in rewards))
        feasible = read("experiments/training/feasible_actions.json")
        split = read("experiments/training/data/split.json")
        self.assertEqual((50, 25), (len(split["train"]), len(split["validation"])))
        for tid in split["train"] + split["validation"]:
            group = [r for r in rewards if r["task_id"] == tid and r["stage"] == "action_matrix"]
            self.assertEqual(set(feasible[tid]), {r["action_index"] for r in group})
            self.assertEqual({1 if tid in split["train"] else 3},
                set(collections.Counter(r["action_index"] for r in group).values()))

    def test_formal_official_counts_are_preserved(self):
        data = rows("experiments/results/formal_alignment/source_data.csv")
        self.assertEqual(92, len(data))
        expected = {
            ("original_source", "gpt-4o"): (17, 13, 12),
            ("original_source", "gpt-4o-mini"): (15, 15, 10),
            ("semicolon_compatibility", "gpt-4o"): (17, 16, 13),
            ("semicolon_compatibility", "gpt-4o-mini"): (15, 15, 10)}
        for (condition, model), totals in expected.items():
            group = [r for r in data if r["condition"] == condition and r["model"] == model]
            self.assertEqual(23, len(group))
            self.assertEqual(totals, tuple(sum(truth(r[key]) for r in group)
                for key in ("compile", "verified", "satisfied80")))

    def test_public_prompt_contract(self):
        from experiments.comparison.rlmar.evaluate_five_models import public_prompt
        task = read("evaluation/functional94/data/private_tasks.json")[0]
        public = {k: task[k] for k in ("instruction", "instruction_sha256", "entry_point", "pou_interface")}
        self.assertEqual(public_prompt(task), public_prompt(public))


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch required for checkpoint inference")
class FrozenPolicyTests(unittest.TestCase):
    def test_all_564_routes_match_registered_routes(self):
        from MAR.Routing.policy import FrozenRouter
        router = FrozenRouter(ROOT / "experiments/comparison/rlmar/route_policy.pt")
        tasks = read("evaluation/functional94/data/public_tasks.json")
        files = sorted((ROOT / "experiments/comparison/rlmar/routes").glob("routes_*.json"))
        self.assertEqual(6, len(files))
        for file in files:
            model = file.stem.removeprefix("routes_")
            frozen = json.loads(file.read_text())
            if isinstance(frozen, dict):
                frozen = frozen["routes"]
            expected = {r["task_id"]: r["action_index"] for r in frozen}
            actual = {t["task_id"]: router.predict(t, model)["action_index"] for t in tasks}
            self.assertEqual(expected, actual, model)


if __name__ == "__main__":
    unittest.main()
