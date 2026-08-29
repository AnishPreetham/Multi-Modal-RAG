"""
embeddings.py -- local sentence embeddings via all-MiniLM-L6-v2.

The model is loaded once per process (module-level singleton, thread-safe)
and reused for every call. Vectors are L2-normalized so that FAISS inner
product is exactly cosine similarity.

No cloud embedding API is used anywhere.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from config import settings
from schemas import ContentItem

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()

# Cache of text -> vector, so re-embedding identical content (common during
# evaluation reruns and repeated queries) costs nothing.
_cache: dict[str, np.ndarray] = {}
_CACHE_LIMIT = 4096


def get_model():
    """Load the embedding model once. ~26s on first call including download;
    subsequent calls return the resident model."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s", settings.embedding_model)
                _model = SentenceTransformer(
                    settings.embedding_model,
                    device=settings.torch_device,
                )
    return _model


def embedding_dimension() -> int:
    """Actual dimension reported by the loaded model."""
    model = get_model()
    getter = getattr(model, "get_embedding_dimension", None)
    if getter is None:
        getter = model.get_sentence_embedding_dimension
    return int(getter())


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of strings -> float32 array of shape (n, dim), normalized."""
    if not texts:
        return np.zeros((0, settings.embedding_dim), dtype=np.float32)

    todo = [t for t in texts if t not in _cache]
    if todo:
        unique = list(dict.fromkeys(todo))
        vectors = get_model().encode(
            unique,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        if len(_cache) > _CACHE_LIMIT:
            _cache.clear()
        for text, vector in zip(unique, vectors):
            _cache[text] = vector

    return np.vstack([_cache[t] for t in texts]).astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query -> shape (dim,)."""
    return embed_texts([query])[0]


def embed_items(items: list[ContentItem], batch_size: int = 32) -> np.ndarray:
    """Embed the `content` of each ContentItem, in order."""
    return embed_texts([item.content for item in items], batch_size=batch_size)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Inputs are already normalized
    in this system, but the denominator is kept for correctness."""
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)
