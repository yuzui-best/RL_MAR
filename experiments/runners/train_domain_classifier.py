from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.MAR.Core.dataset import DOMAINS, load_reward_records, stable_sha256


def build_classifier(input_dim: int = 384) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5))


def accuracy(model: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float((model(features).argmax(-1) == labels).float().mean().item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the RL-MAR v2 domain classifier on RewardPLC only")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-path", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = load_reward_records(args.dataset)
    train = [row for row in records if row.split == "train"]
    validation = [row for row in records if row.split == "evolution_dev"]
    encoder = SentenceTransformer(args.model_path, device=str(device))
    train_x = encoder.encode([row.specification for row in train], convert_to_tensor=True, device=str(device))
    validation_x = encoder.encode([row.specification for row in validation], convert_to_tensor=True, device=str(device))
    label_index = {name: index for index, name in enumerate(DOMAINS)}
    train_y = torch.tensor([label_index[row.domain] for row in train], device=device)
    validation_y = torch.tensor([label_index[row.domain] for row in validation], device=device)
    counts = torch.bincount(train_y, minlength=len(DOMAINS)).float()
    weights = counts.sum() / (len(DOMAINS) * counts.clamp_min(1))

    model = build_classifier(train_x.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=weights)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(train_x.cpu(), train_y.cpu()), batch_size=args.batch_size, shuffle=True, generator=generator)
    best = None
    best_accuracy = -1.0
    stale = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        score = accuracy(model, validation_x, validation_y)
        history.append({"epoch": epoch + 1, "loss": sum(losses) / len(losses), "validation_accuracy": score})
        if score > best_accuracy:
            best_accuracy = score
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best is None:
        raise RuntimeError("training produced no checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, args.output)
    report = {
        "dataset_sha256": stable_sha256(args.dataset.read_bytes()),
        "checkpoint_sha256": stable_sha256(args.output.read_bytes()),
        "encoder": args.model_path,
        "seed": args.seed,
        "train_tasks": len(train),
        "validation_tasks": len(validation),
        "best_validation_accuracy": best_accuracy,
        "epochs_completed": len(history),
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
