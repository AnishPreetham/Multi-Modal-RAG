"""
retriever.py -- query -> RetrievalResult[].

    query text -> MiniLM embedding -> FAISS cosine search -> ranked results

All modalities compete in one ranked list. That is what makes a cross-modal
query genuinely cross-modal rather than a stitched-together per-modality
search.
"""
from __future__ import annotations

import logging

from config import settings
from embeddings import embed_query
from schemas import Modality, RetrievalResult
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, store: VectorStore):
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        modality_filter: Modality | None = None,
        exclude_source_ids: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """Semantic search over the whole index.

        `modality_filter` and `exclude_source_ids` over-fetch then filter, so
        a constrained search still returns up to top_k results rather than a
        truncated handful.

        `exclude_source_ids` is what makes "find sources related to this
        file" meaningful: without it a document query retrieves mostly its
        own chunks, which are trivially its own nearest neighbours.
        """
        k = top_k or settings.top_k
        if not query.strip():
            return []

        excluded = exclude_source_ids or set()
        fetch_k = k * 4 if (modality_filter or excluded) else k
        query_vector = embed_query(query)
        hits = self.store.search(query_vector, fetch_k)

        results: list[RetrievalResult] = []
        for item, score in hits:
            if modality_filter and item.modality is not modality_filter:
                continue
            if item.source_id in excluded:
                continue
            results.append(RetrievalResult(item=item, score=score, rank=len(results) + 1))
            if len(results) >= k:
                break
        return results
