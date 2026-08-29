"""
relationships.py -- two-stage relationship computation.

INGESTION TIME (deterministic, persisted in SQLite):
    SAME_SOURCE     items from the same file
    SAME_PAGE       items from the same PDF page      (documents.py)
    TRANSCRIPT_OF   adjacent audio segments           (audio.py)
    OCR_OF          text derived from an image

QUERY TIME (semantic, transient, never persisted):
    RELATED_TO      cosine similarity above threshold, computed ONLY among
                    the handful of already-retrieved items.

The query-time stage compares at most a few dozen pairs. It never scans the
database, which is what keeps this O(k^2) on k=5 rather than O(n^2) on the
whole corpus.
"""
from __future__ import annotations

import itertools
import logging

import numpy as np

from config import settings
from embeddings import embed_texts
from schemas import (
    ContentItem,
    Relationship,
    RelationshipType,
    RerankedResult,
    derive_id,
)
from vector_store import VectorStore

logger = logging.getLogger(__name__)


def build_same_source_relationships(items: list[ContentItem]) -> list[Relationship]:
    """Chain items of one source rather than fully connecting them: a full
    mesh of n items is O(n^2) rows for no retrieval benefit."""
    by_source: dict[str, list[ContentItem]] = {}
    for item in items:
        by_source.setdefault(item.source_id, []).append(item)

    out: list[Relationship] = []
    for source_id, group in by_source.items():
        for a, b in zip(group, group[1:]):
            out.append(Relationship(
                relationship_id=derive_id(a.item_id, b.item_id, "same_source"),
                source_item_id=a.item_id,
                target_item_id=b.item_id,
                relationship_type=RelationshipType.SAME_SOURCE,
                confidence=1.0,
                metadata={"source_id": source_id},
            ))
    return out


def compute_related_to(
    evidence: list[RerankedResult],
    threshold: float | None = None,
) -> list[Relationship]:
    """Semantic RELATED_TO among the reranked evidence only.

    Cosine similarity is clamped into [0,1] before it becomes `confidence`,
    because the schema bounds that field and a negative similarity is never
    a relationship worth keeping anyway.
    """
    limit = threshold if threshold is not None else settings.related_to_threshold
    if len(evidence) < 2:
        return []

    items = [result.item for result in evidence]
    vectors = embed_texts([item.content for item in items])

    out: list[Relationship] = []
    for (i, a), (j, b) in itertools.combinations(enumerate(items), 2):
        similarity = float(np.dot(vectors[i], vectors[j]))
        if similarity < limit:
            continue
        # Cross-modal links are the interesting ones; keep same-source pairs
        # out because SAME_SOURCE already covers them deterministically.
        if a.source_id == b.source_id:
            continue
        out.append(Relationship(
            relationship_id=derive_id(a.item_id, b.item_id, "related_to"),
            source_item_id=a.item_id,
            target_item_id=b.item_id,
            relationship_type=RelationshipType.RELATED_TO,
            confidence=max(0.0, min(1.0, similarity)),
            metadata={
                "similarity": round(similarity, 4),
                "cross_modal": a.modality is not b.modality,
                "transient": True,
            },
        ))
    return sorted(out, key=lambda r: r.confidence or 0.0, reverse=True)


def expand_evidence(
    store: VectorStore,
    evidence: list[RerankedResult],
    max_expansion: int | None = None,
) -> tuple[list[RerankedResult], list[Relationship]]:
    """Pull in deterministically-linked neighbours of the reranked evidence.

    Expanded items are appended after the direct evidence and marked in
    metadata, so the prompt keeps direct evidence at higher priority. The
    number added is capped so relationship expansion cannot flood the
    context window.
    """
    cap = max_expansion if max_expansion is not None else settings.max_relationship_expansion
    if not evidence or cap <= 0:
        return evidence, []

    seen = {result.item.item_id for result in evidence}
    stored = store.get_relationships_for_items(list(seen))

    added: list[RerankedResult] = []
    used: list[Relationship] = []
    weakest = min(result.rerank_score for result in evidence)

    for relationship in stored:
        if len(added) >= cap:
            break
        partner_id = (
            relationship.target_item_id
            if relationship.source_item_id in seen
            else relationship.source_item_id
        )
        if partner_id in seen:
            continue
        item = store.get_item(partner_id)
        if item is None:
            continue

        seen.add(partner_id)
        item.metadata["expanded_via"] = relationship.relationship_type.value
        added.append(RerankedResult(
            item=item,
            retrieval_score=0.0,
            # Ranked strictly below the weakest direct hit so that expanded
            # context can never outrank real evidence.
            rerank_score=weakest - 1.0,
            rank=len(evidence) + len(added) + 1,
        ))
        used.append(relationship)

    return evidence + added, used
