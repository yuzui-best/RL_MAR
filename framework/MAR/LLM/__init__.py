"""Embedding support used by the public RL-MAR route policy."""

from .llm_embedding import SentenceEncoder, get_sentence_embedding

__all__ = ["SentenceEncoder", "get_sentence_embedding"]
