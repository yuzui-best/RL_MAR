"""Load the paper's public-feature policy and decode native role chains."""
from pathlib import Path
import json

from .core import ACTIONS, chain, domain, features, mask


class FrozenRouter:
    def __init__(self, checkpoint, catalogs=None):
        import torch
        from torch import nn

        self.torch = torch
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["actions"] != list(ACTIONS):
            raise ValueError("checkpoint action library differs from the released library")
        self.network = nn.Sequential(
            nn.Linear(saved["features"], saved["hidden_dim"]),
            nn.Tanh(), nn.Linear(saved["hidden_dim"], len(ACTIONS)))
        self.network.load_state_dict(saved["state_dict"])
        self.network.eval()
        self.mean = torch.tensor(saved["normalization_mean"], dtype=torch.float32)
        self.scale = torch.tensor(saved["normalization_scale"], dtype=torch.float32)
        if not (torch.isfinite(self.mean).all() and torch.isfinite(self.scale).all()
                and (self.scale > 0).all()):
            raise ValueError("invalid feature normalization")
        root = Path(__file__).resolve().parents[3]
        catalog_path = (Path(catalogs) if catalogs else
                        root / "experiments/comparison/rlmar/role_catalogs.json")
        self.catalogs = json.loads(catalog_path.read_text())

    def predict(self, task, model="deepseek-v4-flash", budget=16384):
        torch = self.torch
        name = domain(task)
        catalog = self.catalogs[name]
        available = [r["Name"] for r in catalog]
        permitted = torch.tensor(mask(task, available), dtype=torch.bool)
        if not permitted.any():
            raise ValueError(f"no native action for {task['task_id']}")
        with torch.no_grad():
            x = torch.tensor(features(task, model), dtype=torch.float32)
            logits = self.network((x - self.mean) / self.scale)
            index = int(logits.masked_fill(~permitted, -torch.inf).argmax())
        roles = chain(ACTIONS[index])
        selected = [catalog[available.index(role)] for role in roles]
        return dict(task_id=task["task_id"], domain=name, action_index=index,
                    action=ACTIONS[index], roles=roles,
                    role_instructions=[r["Description"] for r in selected],
                    execution_action=dict(topology_index=2, agent_count=len(roles),
                        role_indices=list(range(len(roles))), rounds=1,
                        repair_budget=1, token_budget=budget))
