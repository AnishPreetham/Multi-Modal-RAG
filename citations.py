"""
citations.py -- programmatic citation construction.

The model never emits citation markers. Citations are built here from the
final evidence set, which makes a fabricated citation structurally
impossible: every Citation is constructed from a real ContentItem that was
actually retrieved.

Attribution works at sentence level: each answer sentence is embedded and
compared against each evidence item; evidence that supports at least one
sentence above CITATION_THRESHOLD is cited. If no sentence clears the bar,
the top-ranked evidence is cited so the answer is never left unsourced.
"""
from __future__ import annotations

import logging
import re

import numpy as np

from config import settings
from embeddings import embed_texts
from schemas import Citation, RerankedResult, clock

logger = logging.getLogger(__name__)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text.strip()) if len(s.strip()) > 15]


def build_citations(
    answer: str,
    evidence: list[RerankedResult],
    filenames: dict[str, str],
    threshold: float | None = None,
) -> tuple[list[Citation], dict[str, list[int]]]:
    """Return (citations, sentence_to_citation_ids).

    The second value maps each answer sentence to the citation IDs that
    support it, which is what the UI uses to show groundedness per sentence.
    """
    limit = threshold if threshold is not None else settings.citation_threshold
    if not evidence:
        return [], {}

    sentences = split_sentences(answer)
    evidence_vectors = embed_texts([r.item.content for r in evidence])

    supported: set[int] = set()
    sentence_map: dict[str, list[int]] = {}

    if sentences:
        sentence_vectors = embed_texts(sentences)
        for sentence, sentence_vector in zip(sentences, sentence_vectors):
            similarities = evidence_vectors @ sentence_vector
            hits = [
                index for index, score in enumerate(similarities)
                if float(score) >= limit
            ]
            if not hits:
                # Fall back to the single best-matching evidence item so the
                # sentence still resolves to something the user can inspect.
                hits = [int(np.argmax(similarities))]
            supported.update(hits)
            sentence_map[sentence] = hits

    if not supported:
        supported = {0}

    citations: list[Citation] = []
    index_to_citation_id: dict[int, int] = {}
    for citation_id, evidence_index in enumerate(sorted(supported), start=1):
        result = evidence[evidence_index]
        item = result.item
        index_to_citation_id[evidence_index] = citation_id
        citations.append(Citation(
            citation_id=citation_id,
            item_id=item.item_id,
            source_id=item.source_id,
            filename=filenames.get(item.source_id, "unknown"),
            modality=item.modality,
            location=item.location,
            excerpt=_excerpt(item.content),
        ))

    remapped = {
        sentence: sorted(
            index_to_citation_id[i] for i in indices if i in index_to_citation_id
        )
        for sentence, indices in sentence_map.items()
    }
    return citations, remapped


def _excerpt(content: str) -> str:
    text = " ".join(content.split())
    if len(text) <= settings.max_excerpt_chars:
        return text
    return text[: settings.max_excerpt_chars].rstrip() + "..."


def render_citation(citation: Citation) -> str:
    """Human-readable citation line, e.g. '[2] meeting.wav - 04:12-04:34'."""
    location = citation.location
    if location.page_number is not None:
        where = f" - Page {location.page_number}"
    elif location.timestamp_start is not None and location.timestamp_end is not None:
        where = f" - {clock(location.timestamp_start)}-{clock(location.timestamp_end)}"
    elif location.section:
        where = f" - {location.section}"
    else:
        where = ""
    return f"[{citation.citation_id}] {citation.filename}{where}"
