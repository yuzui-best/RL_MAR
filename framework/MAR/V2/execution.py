"""Frozen RouteAction executor used only by the Functional94 evaluation runner."""
from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass

from MAR.V2.request_budget import RequestBudget, request_budget
from MAR.V2.budget import BudgetExceeded
from MAR.V2.quality_controller import QualityAction, QualityController
from MAR.V2.outcomes import classify_outcome

ROLE_NAMES = ("Architect", "PLCImplementer", "Verifier", "SafetyReviewer", "Integrator", "Reviewer")
TOPOLOGIES = ("IO", "CoT", "Chain", "FullConnected", "Debate", "Reflection")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def score(result: dict) -> tuple[int, int]:
    return (int(result.get("stages_completed", 0)), int(result.get("checks_passed", 0)))


def compact_diagnostic(result: dict) -> str:
    if result.get("status") == "COMPILE_FAIL":
        diagnostic = re.sub(r"\x1b\[[0-9;]*m", "", str(result.get("diagnostic", "")))
        return diagnostic[-3000:] or "The candidate did not compile."
    return "The candidate compiled but failed hidden functional checks; re-check only the public requirement."


@dataclass
class Execution:
    action: dict
    backend: object
    verifier: object
    hard_budget: bool = True
    quality: bool = True
    max_call_tokens: int = 2048
    role_names: tuple[str, ...] = ROLE_NAMES
    role_instructions: tuple[str, ...] | None = None

    def __post_init__(self):
        self._validate_action()

    def _validate_action(self):
        a = self.action
        if not 0 <= int(a["topology_index"]) < len(TOPOLOGIES):
            raise ValueError("invalid topology")
        if not 1 <= int(a["agent_count"]) <= len(self.role_names):
            raise ValueError("invalid agent_count")
        if len(a["role_indices"]) != int(a["agent_count"]) or len(set(a["role_indices"])) != len(a["role_indices"]):
            raise ValueError("role_indices must select each agent exactly once")
        if any(not 0 <= int(i) < len(self.role_names) for i in a["role_indices"]):
            raise ValueError("invalid role")
        if self.role_instructions is not None and len(self.role_instructions) != len(self.role_names):
            raise ValueError("role instructions must align with role names")
        if int(a["rounds"]) < 1 or int(a["repair_budget"]) < 0 or int(a["token_budget"]) < 1:
            raise ValueError("invalid frozen budget")

    async def run(self, task: dict) -> dict:
        a = self.action
        topology = a["topology_index"]
        calls_per_round = (a["agent_count"] if topology == 4 else 3) if topology in (4, 5) else a["agent_count"] + (1 if topology == 3 and a["agent_count"] > 1 else 0)
        max_calls = calls_per_round * a["rounds"] + a["repair_budget"]
        budget = RequestBudget(max_calls=max_calls, max_tokens=a["token_budget"], max_wall_seconds=180)
        events, outputs = [], {}
        best_candidate, best_result, best_score = "", None, None
        final_result, repairs, stagnant = {"status": "COMPILE_FAIL"}, 0, 0

        async def call(role_index: int, round_index: int, parents: list[str], purpose: str, extra: str = "", token_limit: int | None = None, node: str | None = None) -> str:
            role_index_value = a["role_indices"][role_index]
            role = self.role_names[role_index_value]
            instruction = self.role_instructions[role_index_value] if self.role_instructions is not None else f"You are {role}."
            parent_text = "\n\n".join(outputs[p] for p in parents if p in outputs)
            prompt = f"{instruction}\n\n{purpose}\n\n{task['query']}"
            if parent_text:
                prompt += "\n\nPrior permitted work products:\n" + parent_text
            if extra:
                prompt += "\n\n" + extra
            limit = token_limit if token_limit is not None else max(1, min(self.max_call_tokens, a["token_budget"] - budget.tokens - budget.reserved_tokens))
            try:
                text = await self.backend.agen([{"role": "user", "content": prompt}], max_tokens=limit)
            except BudgetExceeded:
                raise
            usage = getattr(self.backend, "pop_usage", lambda: {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0})()
            events.append({"kind": "role", "round": round_index, "node": node or f"r{round_index}:a{role_index}", "role": role,
                           "purpose": purpose, "parents": parents, "response_sha256": digest(text), "usage": usage})
            return text

        async def call_budgeted(*args, **kwargs):
            limit = max(1, min(self.max_call_tokens, a["token_budget"] - budget.tokens - budget.reserved_tokens))
            if self.hard_budget:
                budget.before_call(limit)
            try:
                value = await call(*args, **kwargs, token_limit=limit)
                usage = events[-1]["usage"]
                if self.hard_budget:
                    budget.after_call(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), limit)
                return value
            except Exception:
                if self.hard_budget:
                    budget.cancel_call(limit)
                raise

        async def execute_round(rnd: int):
            topology = a["topology_index"]
            ids = list(range(a["agent_count"]))
            if topology == 2:  # Chain
                previous = []
                for i in ids:
                    key = f"r{rnd}:a{i}"
                    outputs[key] = await call_budgeted(i, rnd, previous, "Produce an IEC ST candidate or improve the predecessor.")
                    previous = [key]
                return outputs[previous[0]]
            if topology == 5:  # Reflection: generate, review, repair
                generator, reviewer, repairer = ids[0], ids[min(1, len(ids)-1)], ids[-1]
                g = f"r{rnd}:a{generator}"; outputs[g] = await call_budgeted(generator, rnd, [], "Draft the IEC ST implementation.")
                v = f"r{rnd}:a{reviewer}"; outputs[v] = await call_budgeted(reviewer, rnd, [g], "Review the draft for contract and IEC ST errors; give a concise repair plan.")
                z = f"r{rnd}:a{repairer}"; outputs[z] = await call_budgeted(repairer, rnd, [g, v], "Return only the repaired IEC ST candidate.")
                return outputs[z]
            if topology == 1:  # CoT: planner trace, then parallel implementations consume it.
                plan = f"r{rnd}:a0"; outputs[plan] = await call_budgeted(0, rnd, [], "Create a concise IEC ST implementation plan.")
                workers = ids[1:]
                if not workers:
                    return outputs[plan]
                keys = [f"r{rnd}:a{i}" for i in workers]
                values = await asyncio.gather(*(call_budgeted(i, rnd, [plan], "Use the plan to produce only IEC ST.") for i in workers))
                outputs.update(dict(zip(keys, values)))
                return outputs[keys[-1]]
            if topology == 4:  # Debate: concurrent proposals, critics, synthesis
                if len(ids) == 1:
                    k = f"r{rnd}:a0"; outputs[k] = await call_budgeted(0, rnd, [], "Produce and self-check an IEC ST solution.", node=f"r{rnd}:proposal:0")
                    return outputs[k]
                if len(ids) == 2:
                    g = f"r{rnd}:a0"; outputs[g] = await call_budgeted(0, rnd, [], "Independently propose an IEC ST solution.", node=f"r{rnd}:proposal:0")
                    s = f"r{rnd}:a1"; outputs[s] = await call_budgeted(1, rnd, [g], "Critique the proposal and return only the corrected IEC ST.", node=f"r{rnd}:synthesis:1")
                    return outputs[s]
                proposal_ids = ids[:max(1, len(ids)//2)]
                proposal_keys = [f"r{rnd}:a{i}" for i in proposal_ids]
                values = await asyncio.gather(*(call_budgeted(i, rnd, [], "Independently propose an IEC ST solution.", node=f"r{rnd}:proposal:{i}") for i in proposal_ids))
                outputs.update(dict(zip(proposal_keys, values)))
                critic_ids = ids[len(proposal_ids):-1] or [proposal_ids[-1]]
                critic_keys = [f"r{rnd}:a{i}" for i in critic_ids]
                values = await asyncio.gather(*(call_budgeted(i, rnd, proposal_keys, "Critique the proposals and identify the best repair.", node=f"r{rnd}:critic:{i}") for i in critic_ids))
                outputs.update(dict(zip(critic_keys, values)))
                s = ids[-1]; k = f"r{rnd}:a{s}"; outputs[k] = await call_budgeted(s, rnd, proposal_keys + critic_keys, "Synthesize and return only IEC ST.", node=f"r{rnd}:synthesis:{s}")
                return outputs[k]
            # IO / CoT are one-layer; FullyConnected aggregates all prior-round work.
            parents = [k for k in outputs if k.startswith(f"r{rnd-1}:")] if topology == 3 and rnd else []
            keys = [f"r{rnd}:a{i}" for i in ids]
            values = await asyncio.gather(*(call_budgeted(i, rnd, parents, "Produce an IEC ST candidate." if topology == 0 else "Reason then produce an IEC ST candidate.") for i in ids))
            outputs.update(dict(zip(keys, values)))
            if topology == 3 and len(ids) > 1:
                # A distinct aggregator creates dependency edges from every previous-round artifact.
                k = f"r{rnd}:aggregate"; outputs[k] = await call_budgeted(ids[-1], rnd, keys + parents, "Integrate all supplied artifacts; return only IEC ST.")
                return outputs[k]
            return outputs[keys[-1]]

        with request_budget(budget) if self.hard_budget else _NullContext():
            for rnd in range(a["rounds"]):
                try:
                    candidate = await execute_round(rnd)
                except BudgetExceeded:
                    events.append({"kind": "budget_stop", "round": rnd, "reason": budget.stop_reason})
                    break
                final_result = self.verifier(candidate)
                current_score = score(final_result)
                outcome = classify_outcome(final_result)
                events.append({"kind": "validation", "round": rnd, "status": outcome.status, "score": current_score})
                if outcome.raw_reward is None:
                    return {"route_executed": True, "events": events, "candidate": candidate, "validation": final_result, "raw_reward": None,
                            "quality_action": QualityAction.ABORT_ENVIRONMENT.value, "budget": budget.snapshot()}
                if best_score is None or current_score > best_score:
                    best_candidate, best_result, best_score, stagnant = candidate, final_result, current_score, 0
                else:
                    stagnant += 1
                decision = QualityController().decide(outcome, best_score if candidate != best_candidate else None, current_score) if self.quality else QualityAction.SUBMIT
                events.append({"kind": "quality", "round": rnd, "action": decision.value, "best_score": best_score, "repairs": repairs})
                if decision == QualityAction.SUBMIT:
                    return {"route_executed": True, "events": events, "candidate": candidate, "validation": final_result, "raw_reward": outcome.raw_reward, "budget": budget.snapshot()}
                if decision == QualityAction.REVERT:
                    final_result = best_result
                if self.quality and decision in (QualityAction.REPAIR, QualityAction.CLARIFY) and repairs < a["repair_budget"]:
                    repairs += 1
                    feedback = "Verifier diagnostic: " + compact_diagnostic(final_result) + "\nPrevious candidate:\n" + candidate
                    try:
                        repaired = await call_budgeted(a["agent_count"]-1, rnd, [], "Repair the previous IEC ST. Return only IEC ST.", feedback)
                    except BudgetExceeded:
                        break
                    final_result = self.verifier(repaired); repaired_score = score(final_result); repaired_outcome = classify_outcome(final_result)
                    events.append({"kind": "repair", "round": rnd, "status": repaired_outcome.status, "score": repaired_score, "feedback": compact_diagnostic(final_result)})
                    if repaired_outcome.raw_reward is None:
                        return {"route_executed": True, "events": events, "validation": final_result, "raw_reward": None, "quality_action": "ABORT_ENVIRONMENT", "budget": budget.snapshot()}
                    if repaired_score > best_score:
                        best_candidate, best_result, best_score, stagnant = repaired, final_result, repaired_score, 0
                    else:
                        stagnant += 1
                    if repaired_outcome.status == "PASS":
                        return {"route_executed": True, "events": events, "candidate": repaired, "validation": final_result, "raw_reward": 1.0, "budget": budget.snapshot()}
                if stagnant >= 2:
                    events.append({"kind": "quality", "round": rnd, "action": "STOP_NO_IMPROVEMENT"})
                    break
        return {"route_executed": True, "events": events, "candidate": best_candidate, "validation": best_result or final_result,
                "raw_reward": classify_outcome(best_result or final_result).raw_reward, "budget": budget.snapshot()}


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *args): return False
