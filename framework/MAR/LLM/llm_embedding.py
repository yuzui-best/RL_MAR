import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
import torch

def _resolve_model_path(requested=None):
    value = requested or os.environ.get("LOCAL_MODEL_PATH", "")
    if not value:
        raise FileNotFoundError("a local sentence encoder path is required")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"local sentence encoder is unavailable: {path}")
    return path


def get_sentence_embedding(sentence, model_path=None):
    path = _resolve_model_path(model_path)
    model = SentenceTransformer(str(path), local_files_only=True)
    embeddings = model.encode(sentence)
    return torch.tensor(embeddings)


class SentenceEncoder(torch.nn.Module):
    def __init__(self, model_path=None, device=None):
        super().__init__()
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'

        path = _resolve_model_path(model_path)
        self.model = SentenceTransformer(str(path), device=self.device, local_files_only=True)
        
    def forward(self, sentence):
        if len(sentence) == 0:
            return torch.tensor([]).to(self.device)
        embeddings = self.model.encode(sentence,convert_to_tensor=True,device=self.device)
        return embeddings
