"""
schemas.py -- the shared data contract for every module in Evidence AI.

This file defines formats only; it processes nothing. Every ingestion
module (documents / images / audio) returns an IngestionResult carrying
ContentItem objects, and every downstream stage consumes that same shape:

    file -> IngestionResult -> ContentItem[] -> embedding -> FAISS
         -> RetrievalResult[] -> RerankedResult[] -> Citation[] -> QueryResponse

Modality-specific meaning of ContentItem.content:
    TEXT   extracted and chunked document text
    IMAGE  Qwen visual description + Tesseract OCR text  (never raw pixels)
    AUDIO  one timestamped transcript segment
"""
from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# Enumerations
# ============================================================

class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


class SourceType(str, Enum):
    PDF = "pdf"
    DOC = "doc"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"
    IMAGE = "image"
    AUDIO = "audio"
    UNKNOWN = "unknown"


class RelationshipType(str, Enum):
    SAME_SOURCE = "same_source"
    SAME_PAGE = "same_page"
    OCR_OF = "ocr_of"
    TRANSCRIPT_OF = "transcript_of"
    REFERENCES = "references"
    RELATED_TO = "related_to"


class InferenceMode(str, Enum):
    """Both modes run entirely locally; they differ in whether the original
    image is handed to the vision model at generation time."""
    LOCAL = "local"
    LOCAL_VISION = "local_vision"


class QueryModality(str, Enum):
    """What the user supplied as the query itself."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"


class ErrorCode(str, Enum):
    """Explicit, actionable failure codes surfaced to API and UI callers."""
    OLLAMA_NOT_AVAILABLE = "OLLAMA_NOT_AVAILABLE"
    QWEN_MODEL_NOT_AVAILABLE = "QWEN_MODEL_NOT_AVAILABLE"
    OCR_NOT_AVAILABLE = "OCR_NOT_AVAILABLE"
    VISION_FAILED = "VISION_FAILED"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    CORRUPT_FILE = "CORRUPT_FILE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INVALID_QUERY = "INVALID_QUERY"
    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    CONVERSION_UNAVAILABLE = "CONVERSION_UNAVAILABLE"
    INDEX_ERROR = "INDEX_ERROR"


# ============================================================
# Deterministic identity
# ============================================================

def content_hash(data: bytes) -> str:
    """Stable 16-hex-char digest. Re-ingesting the same file yields the same
    source_id, which is what makes evaluation reproducible and re-ingestion
    idempotent."""
    return hashlib.sha256(data).hexdigest()[:16]


def derive_id(*parts: Any) -> str:
    """Deterministic ID from any set of components."""
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def clock(seconds: float) -> str:
    """Render seconds as MM:SS for audio citations."""
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


# ============================================================
# Core models
# ============================================================

class Source(BaseModel):
    """One original uploaded file."""
    model_config = ConfigDict(extra="forbid")

    source_id: str
    filename: str
    source_type: SourceType
    file_path: str
    mime_type: str | None = None
    file_size: int | None = None


class SourceLocation(BaseModel):
    """Where an item sits inside its source. Only fields meaningful for the
    modality are populated; the rest stay None rather than being invented."""
    model_config = ConfigDict(extra="forbid")

    page_number: int | None = None
    section: str | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    image_path: str | None = None

    def human(self) -> str:
        """Render the location the way a citation should show it."""
        if self.page_number is not None:
            return f"Page {self.page_number}"
        if self.timestamp_start is not None and self.timestamp_end is not None:
            return f"{clock(self.timestamp_start)}-{clock(self.timestamp_end)}"
        if self.section:
            return self.section
        return ""


class ContentItem(BaseModel):
    """The atomic unit of evidence. Everything indexed is one of these."""
    model_config = ConfigDict(extra="forbid")

    item_id: str
    source_id: str
    modality: Modality
    content: str
    location: SourceLocation = Field(default_factory=SourceLocation)
    parent_id: str | None = None
    vector_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    """A connection between two ContentItems."""
    model_config = ConfigDict(extra="forbid")

    relationship_id: str
    source_item_id: str
    target_item_id: str
    relationship_type: RelationshipType
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: ContentItem
    score: float
    rank: int


class RerankedResult(BaseModel):
    """Carries the retrieval score through untouched alongside the new
    cross-encoder score, so the UI can show that reranking changed order."""
    model_config = ConfigDict(extra="forbid")

    item: ContentItem
    retrieval_score: float
    rerank_score: float
    rank: int


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: int
    item_id: str
    source_id: str
    filename: str
    modality: Modality
    location: SourceLocation
    excerpt: str | None = None


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=50)
    inference_mode: InferenceMode = InferenceMode.LOCAL
    query_modality: QueryModality = QueryModality.TEXT
    query_file_path: str | None = None


class QueryResponse(BaseModel):
    """
    `confidence` is a normalised [0,1] transform of the best cross-encoder
    rerank score. It is a relative ordering signal, NOT a calibrated
    probability that the answer is correct. See README > Limitations.
    """
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_items: list[RerankedResult] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    abstained: bool = False
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    error_code: ErrorCode | None = None
    query_representation: str | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Source
    items: list[ContentItem] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    success: bool = True
    error: str | None = None
    error_code: ErrorCode | None = None
    warnings: list[str] = Field(default_factory=list)
