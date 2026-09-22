"""
api.py -- FastAPI surface over rag_pipeline.

    GET  /health              runtime and index status
    POST /ingest              upload and index one or more files
    POST /query               ask a question (text / image / audio / document)
    GET  /source/{source_id}  inspect an indexed source and its items
    GET  /sources             list indexed sources

Every endpoint delegates to the same RAGPipeline the Streamlit UI uses, so
there is one implementation of the RAG logic, not two.

Run:  .venv\\Scripts\\uvicorn api:app --port 8000
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from rag_pipeline import get_pipeline
from schemas import (
    ErrorCode,
    InferenceMode,
    QueryModality,
    QueryRequest,
    QueryResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Evidence AI",
    description="Offline-runtime multimodal RAG over documents, images and audio.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    # Local dev frontend, plus your deployed Vercel URL once you have one.
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health(backend: str | None = None) -> dict:
    """Runtime status for the selected inference backend.

    `backend` is "offline" (local Ollama) or "online" (cloud API); it
    defaults to settings.inference_backend. Online mode is not probed
    against Ollama, so this endpoint works on a host with no Ollama.
    """
    try:
        return get_pipeline().health(backend)
    except Exception as exc:
        logger.exception("Health check failed")
        return {"status": "error", "detail": str(exc)}


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)) -> dict:
    """Upload and index files. Returns a per-file result rather than failing
    the whole batch when one file is bad."""
    pipeline = get_pipeline()
    results = []

    for upload in files:
        try:
            data = await upload.read()
            if len(data) > settings.max_file_bytes:
                results.append({
                    "filename": upload.filename,
                    "success": False,
                    "error_code": ErrorCode.FILE_TOO_LARGE.value,
                    "error": f"Exceeds {settings.max_file_bytes} bytes.",
                })
                continue

            stored_path = pipeline.store_upload(upload.filename or "upload", data)
            result = pipeline.ingest_file(str(stored_path))
            results.append({
                "filename": result.source.filename,
                "source_id": result.source.source_id,
                "source_type": result.source.source_type.value,
                "items_indexed": len(result.items),
                "relationships": len(result.relationships),
                "success": result.success,
                "error": result.error,
                "error_code": result.error_code.value if result.error_code else None,
                "warnings": result.warnings,
            })
        except Exception as exc:
            logger.exception("Ingest failed for %s", upload.filename)
            results.append({
                "filename": upload.filename,
                "success": False,
                "error": str(exc),
            })

    return {
        "results": results,
        "indexed": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "index": pipeline.store.stats(),
    }


@app.post("/query")
def query(request: QueryRequest, backend: str | None = None) -> dict:
    """Text query against the index.

    `backend` ("offline" | "online") selects the inference provider and
    defaults to settings.inference_backend. It is a query parameter rather
    than a QueryRequest field so the locked schemas.py contract is unchanged.

    The response is the QueryResponse contract plus a sibling
    `technical_detail` string, which carries provider error detail for a
    "Technical details" view without altering QueryResponse itself.
    """
    from rag_pipeline import last_technical_detail

    response = get_pipeline().query(request, backend=backend)
    return {
        **response.model_dump(mode="json"),
        "technical_detail": last_technical_detail(),
    }


@app.post("/query_file", response_model=QueryResponse)
async def query_file(
    file: UploadFile = File(...),
    instruction: str = Form(""),
    query_modality: QueryModality = Form(QueryModality.IMAGE),
    inference_mode: InferenceMode = Form(InferenceMode.LOCAL),
    top_k: int = Form(settings.top_k),
    rerank_top_k: int = Form(settings.rerank_top_k),
    backend: str | None = Form(None),
) -> QueryResponse:
    """Query using an uploaded image, audio clip, or document as the query.

    `backend` ("offline" | "online") selects the inference provider.
    """
    suffix = Path(file.filename or "query").suffix or ".bin"
    data = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        temp_path = handle.name

    try:
        return get_pipeline().query(QueryRequest(
            query=instruction or f"Find evidence related to this {query_modality.value}.",
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            inference_mode=inference_mode,
            query_modality=query_modality,
            query_file_path=temp_path,
        ), backend=backend)
    finally:
        Path(temp_path).unlink(missing_ok=True)


@app.get("/sources")
def list_sources() -> dict:
    sources = get_pipeline().store.list_sources()
    return {"count": len(sources), "sources": [s.model_dump() for s in sources]}


@app.get("/source/{source_id}")
def get_source(source_id: str) -> dict:
    detail = get_pipeline().get_source_detail(source_id)
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": ErrorCode.SOURCE_NOT_FOUND.value,
                    "source_id": source_id},
        )
    return detail


@app.get("/source/{source_id}/file")
def get_source_file(source_id: str) -> FileResponse:
    """Serve the original file. Constrained to data/ so a crafted source_id
    cannot read arbitrary paths off the filesystem."""
    pipeline = get_pipeline()
    source = pipeline.store.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404,
                            detail={"error_code": ErrorCode.SOURCE_NOT_FOUND.value})

    path = Path(source.file_path).resolve()
    data_root = settings.data_dir.resolve()
    if not path.is_file() or data_root not in path.parents:
        raise HTTPException(
            status_code=403,
            detail={"error": "Source file is outside the managed data directory."},
        )
    return FileResponse(path, filename=source.filename)
