"""
rag_pipeline.py -- the single orchestrator.

Both the Streamlit UI and the FastAPI service call this module and nothing
else, so there is exactly one implementation of retrieval, reranking,
relationship expansion, abstention, generation, and citation logic.

Canonical query pipeline:

    query (text / image / audio / document)
      -> query representation
      -> embedding
      -> FAISS retrieval          top_k
      -> cross-encoder rerank     rerank_top_k
      -> ABSTENTION GATE          (before the LLM, so it cannot improvise)
      -> deterministic relationship expansion
      -> query-time RELATED_TO
      -> final evidence set       capped
      -> Qwen2.5-VL generation
      -> programmatic citations
      -> QueryResponse
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path

from config import settings
from schemas import (
    Citation,
    ContentItem,
    ErrorCode,
    IngestionResult,
    InferenceMode,
    Modality,
    QueryModality,
    QueryRequest,
    QueryResponse,
    Relationship,
    RerankedResult,
    Source,
    SourceType,
)

logger = logging.getLogger(__name__)

ABSTENTION_MESSAGE = (
    "I could not find sufficient evidence in the available sources to "
    "answer this question."
)

DOCUMENT_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".md", ".csv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def _safe_name(filename: str) -> str:
    """Strip any directory component so an upload cannot traverse paths."""
    return Path(filename).name.replace("\\", "_").replace("/", "_")


# Technical error detail for the most recent query on this thread. Kept
# outside QueryResponse so that schemas.py stays unchanged, and thread-local
# so concurrent API requests cannot read each other's detail. The UI renders
# it behind a "Technical details" expander; it is never the headline.
_local = threading.local()


def _set_technical_detail(detail: str) -> None:
    _local.technical_detail = detail


def last_technical_detail() -> str:
    """Technical detail from the last query on this thread, or ''."""
    return getattr(_local, "technical_detail", "")


class RAGPipeline:
    """Owns the store and the stateless stages. Models load lazily on first
    use and are then reused for the life of the process."""

    def __init__(self, store=None):
        from retriever import Retriever
        from reranker import Reranker
        from vector_store import VectorStore

        self.store = store or VectorStore()
        self.retriever = Retriever(self.store)
        self.reranker = Reranker()

    # ---------------------------------------------------------------- health

    def health(self, backend=None) -> dict:
        """Readiness for the selected backend.

        Only the selected backend is probed. In online mode this never
        contacts Ollama, so the application is fully usable on a machine
        where Ollama is not installed.
        """
        from providers import Backend, get_provider, resolve_backend

        resolved = resolve_backend(backend)
        provider = get_provider(resolved)
        readiness = provider.readiness()

        tesseract_ok = Path(settings.tesseract_cmd).exists() or bool(
            shutil.which("tesseract")
        )
        health = {
            "status": "ok" if readiness.ready else "degraded",
            "backend": resolved.value,
            "inference": {
                "backend": resolved.value,
                "label": provider.label,
                "model": provider.model_name,
                "ready": readiness.ready,
                "message": readiness.message,
                "technical": readiness.technical,
                "can_autostart": readiness.can_autostart,
            },
            "tesseract": {
                "available": tesseract_ok,
                "detail": settings.tesseract_cmd if tesseract_ok
                else ErrorCode.OCR_NOT_AVAILABLE.value,
            },
            "libreoffice": {"available": Path(settings.soffice_cmd).exists()},
            "models": {
                "llm": provider.model_name,
                "embedding": settings.embedding_model,
                "reranker": settings.reranker_model,
                "whisper": settings.whisper_model_size,
            },
            "index": self.store.stats(),
        }
        # Preserved for existing callers and tests that read health["ollama"].
        # Populated without a probe when online, so online stays Ollama-free.
        if resolved is Backend.OFFLINE:
            health["ollama"] = {
                "available": readiness.ready,
                "detail": readiness.technical or readiness.message,
            }
        else:
            health["ollama"] = {
                "available": False,
                "detail": "Not required in online mode; not checked.",
            }
        return health

    # ------------------------------------------------------------- ingestion

    @staticmethod
    def route(path: Path):
        """Pick the ingestion module for a file extension."""
        suffix = path.suffix.lower()
        if suffix in DOCUMENT_EXTENSIONS:
            from documents import process_document
            return process_document
        if suffix in IMAGE_EXTENSIONS:
            from images import process_image
            return process_image
        if suffix in AUDIO_EXTENSIONS:
            from audio import process_audio
            return process_audio
        return None

    def store_upload(self, filename: str, data: bytes) -> Path:
        """Persist an uploaded file into data/ under a sanitised name."""
        safe = _safe_name(filename)
        suffix = Path(safe).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            target_dir = settings.images_dir
        elif suffix in AUDIO_EXTENSIONS:
            target_dir = settings.audio_dir
        else:
            target_dir = settings.documents_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = target_dir / safe
        destination.write_bytes(data)
        return destination

    def ingest_file(self, path_str: str) -> IngestionResult:
        """Ingest one file end to end: extract, embed, persist, relate."""
        path = Path(path_str).resolve()
        processor = self.route(path)
        if processor is None:
            return IngestionResult(
                source=Source(
                    source_id="unknown",
                    filename=path.name,
                    source_type=SourceType.UNKNOWN,
                    file_path=str(path),
                ),
                success=False,
                error=f"Unsupported file type '{path.suffix}'.",
                error_code=ErrorCode.UNSUPPORTED_FILE_TYPE,
            )

        result = processor(str(path))
        if not result.success or not result.items:
            return result

        return self._index_result(result)

    def _index_result(self, result: IngestionResult) -> IngestionResult:
        from embeddings import embed_items
        from relationships import build_same_source_relationships

        try:
            self.store.add_source(result.source)
            vectors = embed_items(result.items)
            stored = self.store.add_items(result.items, vectors)

            if stored:
                relationships = list(result.relationships)
                relationships.extend(build_same_source_relationships(stored))
                self.store.add_relationships(relationships)
                result.relationships = relationships
            else:
                result.warnings.append(
                    "Already indexed (identical content); no new vectors added."
                )
            result.items = stored or result.items
            return result
        except Exception as exc:
            logger.exception("Indexing failed")
            result.success = False
            result.error = f"{ErrorCode.INDEX_ERROR.value}: {exc}"
            result.error_code = ErrorCode.INDEX_ERROR
            return result

    def ingest_directory(self, directory: str) -> list[IngestionResult]:
        results: list[IngestionResult] = []
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file() and self.route(path) is not None:
                results.append(self.ingest_file(str(path)))
        return results

    # ------------------------------------------------- query representation

    def build_query_representation(self, request: QueryRequest) -> str:
        """Turn whatever the user supplied into the text that gets embedded.

        For image and audio queries the uploaded file is converted using the
        same local pipeline used at ingestion, so an image query and an
        indexed image live in the same representation space.
        """
        instruction = request.query.strip()

        if request.query_modality is QueryModality.TEXT or not request.query_file_path:
            return instruction

        path = Path(request.query_file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{ErrorCode.FILE_NOT_FOUND.value}: {path}")

        if request.query_modality is QueryModality.IMAGE:
            from images import build_representation, describe_image, run_ocr
            from PIL import Image

            with Image.open(path) as handle:
                ocr_text = run_ocr(handle.convert("RGB"))
            try:
                description = describe_image(path)
            except Exception as exc:
                logger.warning("Vision failed for image query: %s", exc)
                description = ""
            representation = build_representation(description, ocr_text)

        elif request.query_modality is QueryModality.AUDIO:
            from audio import transcribe_to_text
            representation = transcribe_to_text(str(path))

        else:  # DOCUMENT
            from documents import process_document
            result = process_document(str(path))
            if not result.success:
                raise ValueError(result.error or "Could not read document query.")
            # Use the leading chunks rather than the whole file: sending an
            # entire document as a query embeds to noise and blows context.
            representation = " ".join(i.content for i in result.items[:5])

        return f"{instruction}\n\n{representation}".strip() if instruction else representation

    # ------------------------------------------------------------------ query

    def query(self, request: QueryRequest, backend=None) -> QueryResponse:
        """Run the shared RAG pipeline and generate with the selected backend.

        `backend` selects offline (local Ollama) or online (cloud API). It is
        a keyword argument rather than a QueryRequest field so that the
        locked schemas.py contract is unchanged; it defaults to
        settings.inference_backend.

        Everything before generation -- retrieval, reranking, relationship
        expansion, the abstention gate -- is identical on both backends.
        """
        from citations import build_citations
        from llm import pick_inspectable_image
        from providers import ProviderError, get_provider, supports_vision
        from relationships import compute_related_to, expand_evidence
        from reranker import score_to_confidence

        provider = get_provider(backend)
        _set_technical_detail("")  # never leak detail from a previous query

        timings: dict[str, float] = {}
        started = time.perf_counter()

        text = request.query.strip()
        if not text or len(text) > settings.max_query_chars:
            return QueryResponse(
                answer="Query must be between 1 and "
                       f"{settings.max_query_chars} characters.",
                abstained=True,
                error_code=ErrorCode.INVALID_QUERY,
            )

        # 1. Query representation
        mark = time.perf_counter()
        try:
            representation = self.build_query_representation(request)
        except Exception as exc:
            return QueryResponse(
                answer=f"Could not read the supplied query file: {exc}",
                abstained=True,
                error_code=ErrorCode.INVALID_QUERY,
            )
        if not representation:
            return QueryResponse(
                answer="Empty query.", abstained=True,
                error_code=ErrorCode.INVALID_QUERY,
            )
        timings["representation_ms"] = (time.perf_counter() - mark) * 1000

        # 2. Retrieval. A file-based query excludes its own indexed source so
        # that "find sources related to this" returns other material rather
        # than the query file's own chunks.
        mark = time.perf_counter()
        excluded = self._self_source_ids(request)
        retrieved = self.retriever.retrieve(
            representation, top_k=request.top_k, exclude_source_ids=excluded
        )
        timings["retrieval_ms"] = (time.perf_counter() - mark) * 1000

        if not retrieved:
            return QueryResponse(
                answer=ABSTENTION_MESSAGE,
                abstained=True,
                error_code=ErrorCode.INSUFFICIENT_EVIDENCE,
                query_representation=representation,
                latency_ms=timings,
            )

        # 3. Reranking
        mark = time.perf_counter()
        reranked = self.reranker.rerank(
            representation, retrieved, top_k=request.rerank_top_k
        )
        timings["rerank_ms"] = (time.perf_counter() - mark) * 1000

        best_score = reranked[0].rerank_score if reranked else -99.0
        confidence = score_to_confidence(best_score)

        # 4. Abstention gate -- before any LLM call
        if best_score < settings.abstention_threshold:
            return QueryResponse(
                answer=ABSTENTION_MESSAGE,
                retrieved_items=reranked,
                abstained=True,
                confidence=confidence,
                error_code=ErrorCode.INSUFFICIENT_EVIDENCE,
                query_representation=representation,
                latency_ms=timings,
            )

        # 5. Relationship expansion
        mark = time.perf_counter()
        expanded, deterministic = expand_evidence(self.store, reranked)
        semantic = compute_related_to(reranked)
        relationships = deterministic + semantic
        final_evidence = expanded[: settings.max_context_items
                                  + settings.max_relationship_expansion]
        timings["relationships_ms"] = (time.perf_counter() - mark) * 1000

        filenames = self._filenames_for(final_evidence)

        # 6. Generation
        mark = time.perf_counter()
        inspect_image = (
            pick_inspectable_image(final_evidence)
            if (request.inference_mode is InferenceMode.LOCAL_VISION
                and supports_vision(backend))
            else None
        )
        try:
            answer = provider.generate(
                representation, final_evidence, filenames, inspect_image=inspect_image
            )
        except ProviderError as exc:
            # The headline stays user-facing. The raw detail travels through
            # last_technical_detail() rather than QueryResponse, so the
            # locked schemas.py contract needs no new field.
            _set_technical_detail(exc.technical)
            return QueryResponse(
                answer=exc.user_message,
                retrieved_items=reranked,
                relationships=relationships,
                abstained=True,
                confidence=confidence,
                error_code=exc.code,
                query_representation=representation,
                latency_ms=timings,
            )
        timings["generation_ms"] = (time.perf_counter() - mark) * 1000

        # 7. Model-signalled abstention
        if "INSUFFICIENT_EVIDENCE" in answer.upper():
            return QueryResponse(
                answer=ABSTENTION_MESSAGE,
                retrieved_items=reranked,
                relationships=relationships,
                abstained=True,
                confidence=confidence,
                error_code=ErrorCode.INSUFFICIENT_EVIDENCE,
                query_representation=representation,
                latency_ms=timings,
            )

        # 8. Programmatic citations
        mark = time.perf_counter()
        citations, _sentence_map = build_citations(answer, final_evidence, filenames)
        timings["citations_ms"] = (time.perf_counter() - mark) * 1000
        timings["total_ms"] = (time.perf_counter() - started) * 1000

        return QueryResponse(
            answer=answer,
            citations=citations,
            retrieved_items=reranked,
            relationships=relationships,
            abstained=False,
            confidence=confidence,
            query_representation=representation,
            latency_ms=timings,
        )

    # ----------------------------------------------------------------- lookup

    def _self_source_ids(self, request: QueryRequest) -> set[str]:
        """Source IDs to exclude from a file-based query's own results.

        Because source_id is a content hash, an uploaded copy of an already
        indexed file resolves to the same ID and is correctly excluded.
        """
        if not request.query_file_path or request.query_modality is QueryModality.TEXT:
            return set()
        try:
            from schemas import content_hash

            path = Path(request.query_file_path)
            if not path.exists():
                return set()
            return {content_hash(path.read_bytes())}
        except Exception as exc:
            logger.warning("Could not hash query file: %s", exc)
            return set()

    def _filenames_for(self, evidence: list[RerankedResult]) -> dict[str, str]:
        names: dict[str, str] = {}
        for result in evidence:
            source_id = result.item.source_id
            if source_id not in names:
                source = self.store.get_source(source_id)
                names[source_id] = source.filename if source else "unknown"
        return names

    def get_source_detail(self, source_id: str) -> dict | None:
        source = self.store.get_source(source_id)
        if source is None:
            return None
        items = self.store.get_items_by_source(source_id)
        return {
            "source": source.model_dump(),
            "item_count": len(items),
            "items": [item.model_dump() for item in items],
        }


_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Process-wide singleton so models and the index load only once."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
