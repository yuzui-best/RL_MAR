import os
from typing import Optional
from class_registry import ClassRegistry

from MAR.LLM.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)

    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None or model_name == "":
            model_name = "gpt-4o-mini"
        audited_transport = os.environ.get(
            "RL_MAR_USE_AUDITED_PROVIDER_TRANSPORT", ""
        ).lower() in {"1", "true", "yes"}
        if audited_transport:
            model = cls.registry.get("ALLChat", model_name)
        elif "DeepSeek-V3" in model_name:
            model = cls.registry.get("Deepseek", model_name)
        elif model_name.lower().startswith("gpt"):
            model = cls.registry.get("AIGBest", model_name)
        elif model_name.lower().startswith("qwen"):
            model = cls.registry.get("Qwen", model_name)
        else:
            model = cls.registry.get("ALLChat", model_name)
        if type(model).__name__ == "ALLChat":
            return model
        from MAR.Core.budgeted_llm import BudgetedLLM
        return BudgetedLLM(model)
