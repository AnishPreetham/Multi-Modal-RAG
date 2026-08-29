"""
reranker.py -- RetrievalResult[] -> RerankedResult[].

A local cross-encoder scores the query against each candidate jointly,
which is far more accurate than comparing two independent embeddings.

The original retrieval_score is carried through untouched alongside the new
rerank_score, so the UI can demonstrate that reranking changed the ordering.

The model is loaded once per process and reused.
"""
from __future__ import annotations

import logging
import math
import threading

from config import settings
from schemas import RerankedResult, RetrievalResult

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import CrossEncoder

                logger.info("Loading reranker %s", settings.reranker_model)
                _model = CrossEncoder(
                    settings.reranker_model,
                    device=settings.torch_device,
                )
    return _model


def score_to_confidence(rerank_score: float) -> float:
    """Squash an unbounded cross-encoder logit into [0,1].

    This is a monotonic transform for display and for the confidence field.
    It is NOT a calibrated probability of correctness -- see README.
    """
    return 1.0 / (1.0 + math.exp(-rerank_score))


class Reranker:
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        if not results:
            return []
        k = top_k or settings.rerank_top_k

        pairs = [(query, r.item.content) for r in results]
        scores = get_model().predict(pairs, show_progress_bar=False)

        ordered = sorted(
            zip(results, scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        return [
            RerankedResult(
                item=result.item,
                retrieval_score=result.score,
                rerank_score=float(score),
                rank=position,
            )
            for position, (result, score) in enumerate(ordered[:k], start=1)
        ]
