"""
vector_store.py -- persistent local storage.

    FAISS   IndexFlatIP over L2-normalized 384-d vectors (inner product ==
            cosine). Exact search: no training step, no recall loss, nothing
            to tune, and faster than an approximate index at this scale.
    SQLite  sources, content items, and deterministic relationships. It is
            the authority mapping a FAISS row ordinal back to a ContentItem.

Both are written on every ingest, so the index survives a restart.
Everything is local; no remote vector database is involved.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path

import numpy as np

from config import settings
from schemas import (
    ContentItem,
    Modality,
    Relationship,
    RelationshipType,
    Source,
    SourceLocation,
    SourceType,
)

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id   TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    source_type TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    mime_type   TEXT,
    file_size   INTEGER
);

CREATE TABLE IF NOT EXISTS items (
    item_id   TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    modality  TEXT NOT NULL,
    content   TEXT NOT NULL,
    location  TEXT NOT NULL,
    parent_id TEXT,
    vector_id INTEGER UNIQUE,
    metadata  TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS relationships (
    relationship_id   TEXT PRIMARY KEY,
    source_item_id    TEXT NOT NULL,
    target_item_id    TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    confidence        REAL,
    metadata          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_source    ON items(source_id);
CREATE INDEX IF NOT EXISTS idx_items_vector    ON items(vector_id);
CREATE INDEX IF NOT EXISTS idx_rel_source_item ON relationships(source_item_id);
CREATE INDEX IF NOT EXISTS idx_rel_target_item ON relationships(target_item_id);
"""


class VectorStore:
    """FAISS + SQLite behind one interface."""

    def __init__(self, index_path: Path | None = None, db_path: Path | None = None):
        self.index_path = Path(index_path or settings.faiss_path)
        self.db_path = Path(db_path or settings.db_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._index = self._load_or_create_index()

    # ---- index lifecycle -------------------------------------------------

    def _load_or_create_index(self):
        import faiss

        dim = settings.embedding_dim
        if self.index_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                if index.d != dim:
                    logger.warning(
                        "Index dim %s != configured %s; rebuilding empty index.",
                        index.d, dim,
                    )
                    return faiss.IndexFlatIP(dim)
                return index
            except Exception as exc:
                logger.error("Failed to read FAISS index, starting fresh: %s", exc)
        return faiss.IndexFlatIP(dim)

    def save(self) -> None:
        import faiss

        with self._lock:
            faiss.write_index(self._index, str(self.index_path))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()

    # ---- writes -----------------------------------------------------------

    def add_source(self, source: Source) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?)",
                (
                    source.source_id, source.filename, source.source_type.value,
                    source.file_path, source.mime_type, source.file_size,
                ),
            )
            self._conn.commit()

    def source_exists(self, source_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        return row is not None

    def add_items(self, items: list[ContentItem], vectors: np.ndarray) -> list[ContentItem]:
        """Append vectors to FAISS and persist items with their row ordinals.
        Items whose item_id already exists are skipped, making re-ingestion of
        an identical file idempotent."""
        if not items:
            return []
        if len(items) != len(vectors):
            raise ValueError(
                f"items ({len(items)}) and vectors ({len(vectors)}) length mismatch"
            )

        with self._lock:
            existing = {
                row["item_id"]
                for row in self._conn.execute(
                    "SELECT item_id FROM items WHERE item_id IN "
                    f"({','.join('?' * len(items))})",
                    [i.item_id for i in items],
                )
            }
            fresh_items, fresh_vectors = [], []
            for item, vector in zip(items, vectors):
                if item.item_id not in existing:
                    fresh_items.append(item)
                    fresh_vectors.append(vector)

            if not fresh_items:
                return []

            start = self._index.ntotal
            self._index.add(np.ascontiguousarray(
                np.vstack(fresh_vectors), dtype=np.float32
            ))

            stored: list[ContentItem] = []
            for offset, item in enumerate(fresh_items):
                item.vector_id = start + offset
                self._conn.execute(
                    "INSERT OR REPLACE INTO items VALUES (?,?,?,?,?,?,?,?)",
                    (
                        item.item_id, item.source_id, item.modality.value,
                        item.content, item.location.model_dump_json(),
                        item.parent_id, item.vector_id, json.dumps(item.metadata),
                    ),
                )
                stored.append(item)
            self._conn.commit()
            self.save()
            return stored

    def add_relationships(self, relationships: list[Relationship]) -> None:
        if not relationships:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO relationships VALUES (?,?,?,?,?,?)",
                [
                    (
                        r.relationship_id, r.source_item_id, r.target_item_id,
                        r.relationship_type.value, r.confidence,
                        json.dumps(r.metadata),
                    )
                    for r in relationships
                ],
            )
            self._conn.commit()

    # ---- reads -------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[ContentItem, float]]:
        """Cosine search. Returns (item, score) ordered best-first."""
        if self._index.ntotal == 0:
            return []
        vector = np.ascontiguousarray(
            query_vector.reshape(1, -1), dtype=np.float32
        )
        scores, indices = self._index.search(vector, min(top_k, self._index.ntotal))

        out: list[tuple[ContentItem, float]] = []
        for score, vector_id in zip(scores[0], indices[0]):
            if vector_id < 0:
                continue
            item = self.get_item_by_vector_id(int(vector_id))
            if item is not None:
                out.append((item, float(score)))
        return out

    def _row_to_item(self, row: sqlite3.Row) -> ContentItem:
        return ContentItem(
            item_id=row["item_id"],
            source_id=row["source_id"],
            modality=Modality(row["modality"]),
            content=row["content"],
            location=SourceLocation.model_validate_json(row["location"]),
            parent_id=row["parent_id"],
            vector_id=row["vector_id"],
            metadata=json.loads(row["metadata"]),
        )

    def get_item_by_vector_id(self, vector_id: int) -> ContentItem | None:
        row = self._conn.execute(
            "SELECT * FROM items WHERE vector_id = ?", (vector_id,)
        ).fetchone()
        return self._row_to_item(row) if row else None

    def get_item(self, item_id: str) -> ContentItem | None:
        row = self._conn.execute(
            "SELECT * FROM items WHERE item_id = ?", (item_id,)
        ).fetchone()
        return self._row_to_item(row) if row else None

    def get_items_by_source(self, source_id: str) -> list[ContentItem]:
        rows = self._conn.execute(
            "SELECT * FROM items WHERE source_id = ? ORDER BY vector_id", (source_id,)
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def get_source(self, source_id: str) -> Source | None:
        row = self._conn.execute(
            "SELECT * FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        if not row:
            return None
        return Source(
            source_id=row["source_id"],
            filename=row["filename"],
            source_type=SourceType(row["source_type"]),
            file_path=row["file_path"],
            mime_type=row["mime_type"],
            file_size=row["file_size"],
        )

    def list_sources(self) -> list[Source]:
        rows = self._conn.execute("SELECT * FROM sources ORDER BY filename").fetchall()
        return [
            Source(
                source_id=r["source_id"], filename=r["filename"],
                source_type=SourceType(r["source_type"]), file_path=r["file_path"],
                mime_type=r["mime_type"], file_size=r["file_size"],
            )
            for r in rows
        ]

    def get_relationships_for_items(self, item_ids: list[str]) -> list[Relationship]:
        """Deterministic relationships touching any of the given items."""
        if not item_ids:
            return []
        placeholders = ",".join("?" * len(item_ids))
        rows = self._conn.execute(
            f"SELECT * FROM relationships WHERE source_item_id IN ({placeholders}) "
            f"OR target_item_id IN ({placeholders})",
            item_ids + item_ids,
        ).fetchall()
        return [
            Relationship(
                relationship_id=r["relationship_id"],
                source_item_id=r["source_item_id"],
                target_item_id=r["target_item_id"],
                relationship_type=RelationshipType(r["relationship_type"]),
                confidence=r["confidence"],
                metadata=json.loads(r["metadata"]),
            )
            for r in rows
        ]

    # ---- stats --------------------------------------------------------------

    def stats(self) -> dict:
        item_count = self._conn.execute("SELECT COUNT(*) c FROM items").fetchone()["c"]
        source_count = self._conn.execute("SELECT COUNT(*) c FROM sources").fetchone()["c"]
        rel_count = self._conn.execute(
            "SELECT COUNT(*) c FROM relationships"
        ).fetchone()["c"]
        by_modality = {
            row["modality"]: row["c"]
            for row in self._conn.execute(
                "SELECT modality, COUNT(*) c FROM items GROUP BY modality"
            )
        }
        return {
            "sources": source_count,
            "items": item_count,
            "relationships": rel_count,
            "vectors": int(self._index.ntotal),
            "by_modality": by_modality,
            "index_path": str(self.index_path),
            "db_path": str(self.db_path),
        }
