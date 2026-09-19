import asyncio
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "framework")]
from experiments.comparison.rlmar.evaluate_five_models import assert_prompt_isolated, budgeted_action, extract_candidate, public_prompt, select_tasks
from MAR.V2.execution import Execution

class FakeBackend:
    def __init__(self): self.messages=[]; self.max_tokens=[]; self.number=0
    async def agen(self, messages, max_tokens=None, **_):
        self.messages.append(messages[-1]["content"]); self.max_tokens.append(max_tokens); self.number += 1
        return f"FUNCTION_BLOCK X_{self.number}"
    def pop_usage(self): return {"prompt_tokens": 2, "completion_tokens": 3, "cost": 0.01}

def action(topology, **changes):
    return {"topology_index": topology, "role_indices": [0,1,2], "agent_count": 3, "rounds": 2, "repair_budget": 2, "token_budget": 1024, **changes}

class RunnerTests(unittest.TestCase):
    def test_public_prompt_excludes_hidden_vectors(self):
        instruction="Control a pump from input x."
        task={"title":"Pump control","category":"Control","entry_point":"FB_Pump","instruction":instruction,"instruction_sha256":hashlib.sha256(instruction.encode()).hexdigest(),"pou_interface":[{"section":"input","name":"x","type":"ARRAY[0..3] OF INT"}],"cases":[{"expected_output":"SECRET"}],"oracles":["SECRET"],"stimulus":{"expected":"SECRET"}}
        prompt=public_prompt(task); assert_prompt_isolated(task,prompt)
        self.assertIn("ARRAY[0..3] OF INT",prompt); self.assertNotIn("SECRET",prompt)
    def test_candidate_extraction_and_formal_budget_projection(self):
        self.assertEqual("FUNCTION_BLOCK X\nEND_FUNCTION_BLOCK", extract_candidate("```iecst\nFUNCTION_BLOCK X\nEND_FUNCTION_BLOCK\n```\nExplanation"))
        projected=budgeted_action(action(4, token_budget=16384), {"max_call_tokens":2048,"max_model_calls":1})
        self.assertEqual(action(4, token_budget=16384), projected)

    def test_task_selection_rejects_unknown_ids(self):
        tasks=[{"task_id":"hf_1"},{"task_id":"hf_2"}]
        self.assertEqual([tasks[1]], select_tasks(tasks, ["hf_2"]))
        with self.assertRaisesRegex(ValueError, "unknown task ids"):
            select_tasks(tasks, ["typo"])

    def test_topology_event_graphs_differ(self):
        graphs=[]
        for top in (2,3,4,5):
            result=asyncio.run(Execution(action(top),FakeBackend(),lambda _: {"status":"PASS","stages_completed":5,"checks_passed":5}).run({"query":"public"}))
            graphs.append([(e["purpose"],tuple(e["parents"])) for e in result["events"] if e["kind"]=="role"])
        self.assertEqual(4,len({str(graph) for graph in graphs}))
    def test_repair_budget_environment_and_best_revert(self):
        backend=FakeBackend()
        def verifier(code):
            return {"status":"COMPILE_FAIL","stages_completed":2 if code.endswith("_3") else 1,"checks_passed":0,"diagnostic":"bad semicolon at line 7"}
        result=asyncio.run(Execution(action(0,rounds=1),backend,verifier).run({"query":"public"}))
        self.assertIn("Previous candidate:",backend.messages[-1]); self.assertIn("bad semicolon at line 7",backend.messages[-1]); self.assertTrue(result["candidate"].endswith("_3"))
        self.assertEqual(result["budget"]["tokens"], result["budget"]["calls"]*5)
        self.assertEqual(1024, backend.max_tokens[0])
        env=asyncio.run(Execution(action(0),FakeBackend(),lambda _: {"status":"ENVIRONMENT_ERROR"}).run({"query":"public"}))
        self.assertIsNone(env["raw_reward"]); self.assertEqual(env["quality_action"],"ABORT_ENVIRONMENT")
    def test_concurrent_debate_accounts_all_usage(self):
        result=asyncio.run(Execution(action(4,rounds=1),FakeBackend(),lambda _: {"status":"PASS","stages_completed":5,"checks_passed":5}).run({"query":"public"}))
        calls=len([event for event in result["events"] if event["kind"]=="role"])
        self.assertEqual(result["budget"]["calls"],calls); self.assertEqual(result["budget"]["tokens"],calls*5)
    def test_small_reflection_and_full_connected_receive_real_budget(self):
        for topology, agents in ((3,2),(5,1)):
            a=action(topology, agent_count=agents, role_indices=list(range(agents)), rounds=1, repair_budget=0)
            r=asyncio.run(Execution(a,FakeBackend(),lambda _: {"status":"PASS","stages_completed":5,"checks_passed":5}).run({"query":"public"}))
            self.assertIsNone(r["budget"]["stop_reason"])
if __name__ == "__main__": unittest.main()
