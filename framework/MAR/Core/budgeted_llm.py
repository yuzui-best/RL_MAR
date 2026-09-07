from __future__ import annotations

import asyncio
import os

from MAR.Utils.globals import CompletionTokens, PromptTokens

from .request_budget import active_request_budget


class BudgetedLLM:
    """Apply the active route budget to any registered LLM backend."""

    def __init__(self, backend):
        self._backend = backend

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def _before(self, max_tokens):
        budget = active_request_budget()
        limit = int(max_tokens or os.getenv("LLM_MAX_TOKENS", getattr(self._backend, "DEFAULT_MAX_TOKENS", 4096)))
        previous = (PromptTokens.instance().value, CompletionTokens.instance().value)
        if budget:
            budget.before_call(limit)
        return budget, limit, previous

    @staticmethod
    def _after(budget, limit, previous):
        if budget:
            budget.after_call(
                PromptTokens.instance().value - previous[0],
                CompletionTokens.instance().value - previous[1],
                limit,
            )

    def gen(self, messages, max_tokens=None, temperature=None, num_comps=None):
        budget, limit, previous = self._before(max_tokens)
        try:
            response = self._backend.gen(messages, limit, temperature, num_comps)
        except Exception:
            if budget:
                budget.cancel_call(limit)
            raise
        self._after(budget, limit, previous)
        return response

    async def agen(self, messages, max_tokens=None, temperature=None, num_comps=None):
        if active_request_budget() is None:
            return await self._backend.agen(messages, max_tokens, temperature, num_comps)
        # Backends without per-request usage expose only process-global counters.
        # Serialize their budgeted async calls so counter deltas remain exact.
        lock = _async_budget_lock()
        async with lock:
            return await self._budgeted_agen(messages, max_tokens, temperature, num_comps)

    async def _budgeted_agen(self, messages, max_tokens, temperature, num_comps):
        budget, limit, previous = self._before(max_tokens)
        try:
            response = await self._backend.agen(messages, limit, temperature, num_comps)
        except Exception:
            if budget:
                budget.cancel_call(limit)
            raise
        self._after(budget, limit, previous)
        return response


_ASYNC_LOCKS = {}


def _async_budget_lock():
    loop = asyncio.get_running_loop()
    return _ASYNC_LOCKS.setdefault(id(loop), asyncio.Lock())
