"""
llm.py -- local grounded generation with Ollama + qwen2.5vl:3b.

Verified against the installed runtime (Ollama 0.33.1): POST /api/generate
with stream=false and base64 images in the `images` array performs real
local image understanding.

Two measured constraints shape the calls made here:
  * ~1030 prompt tokens per image, so num_ctx is raised to 8192 and at most
    one image is ever attached to a query.
  * Ollama unloads after 5 minutes idle, costing a ~14s reload, so every
    call passes an explicit keep_alive.

The model is a writer, not a source of truth. It never emits citation
markers; citations are built programmatically in citations.py.
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

from config import settings
from schemas import ErrorCode, Modality, RerankedResult

logger = logging.getLogger(__name__)

SYSTEM_RULES = """You are Evidence AI, a grounded question-answering system.

Rules you must follow without exception:
1. Use ONLY the evidence provided below. You have no other knowledge.
2. Never invent facts, figures, dates, or names that are not in the evidence.
3. Never write citation markers such as [1] or (Source 2). Citations are
   attached automatically after you answer.
4. Never invent page numbers, timestamps, or filenames.
5. Copy every number, percentage, date, and proper noun EXACTLY as written in
   the evidence. Do not round, reformat, or add digits. If the evidence says
   "99.4 percent", write "99.4 percent" and never "99.44".
6. If the evidence does not answer the question, say exactly:
   INSUFFICIENT_EVIDENCE
7. If two pieces of evidence conflict, say so explicitly and describe both
   rather than silently picking one.
8. Prefer direct evidence over loosely related evidence.
9. Be concise: at most one short paragraph unless the question needs more.
"""

DISCOVERY_RULES = """You are Evidence AI, a grounded evidence-discovery system.

The user is not asking a factual question. They are asking WHICH SOURCES in
the collection relate to their material. The evidence below is the answer.

Rules you must follow without exception:
1. Summarise what each piece of evidence below contains and why it relates
   to the user's material.
2. Name the source files and say what kind of material each one is (report
   page, spreadsheet row, dashboard image, audio segment).
3. Use ONLY the evidence below. Never invent sources or content.
4. Copy numbers, percentages and dates EXACTLY as written. Never round or
   add digits.
5. Never write citation markers such as [1]. Citations are attached
   automatically.
6. Do NOT answer INSUFFICIENT_EVIDENCE when evidence is present -- listing
   and describing the related evidence IS the answer.
7. Be concise: one short sentence per related source.
"""

# Phrasings that ask "what else is connected to this?" rather than a factual
# question. These need the discovery prompt: the retrieved set is the answer,
# so the factual-QA rule about insufficient evidence must not fire.
DISCOVERY_MARKERS = (
    "find document", "find screenshot", "find recording", "find image",
    "find audio", "find evidence", "find related", "find sources",
    "related to this", "corresponds to this", "connected to this",
    "associated with this", "which sources", "what sources",
    "show me related", "anything related",
)


def is_discovery_query(query: str) -> bool:
    """True when the user wants an inventory of related evidence."""
    lowered = query.lower()
    return any(marker in lowered for marker in DISCOVERY_MARKERS)


class OllamaError(RuntimeError):
    """Raised when the local Ollama runtime cannot serve a request."""

    def __init__(self, message: str, code: ErrorCode):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _post(path: str, payload: dict, timeout: int | None = None) -> dict:
    url = f"{settings.ollama_host.rstrip('/')}{path}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout or settings.ollama_timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        if "not found" in body.lower():
            raise OllamaError(
                f"Model '{settings.ollama_model}' is not installed. "
                f"Run: ollama pull {settings.ollama_model}",
                ErrorCode.QWEN_MODEL_NOT_AVAILABLE,
            ) from exc
        raise OllamaError(f"Ollama HTTP {exc.code}: {body}",
                          ErrorCode.OLLAMA_NOT_AVAILABLE) from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {settings.ollama_host}. Is it running? ({exc.reason})",
            ErrorCode.OLLAMA_NOT_AVAILABLE,
        ) from exc


def ollama_available() -> tuple[bool, str]:
    """Check the daemon is up and the configured model is present."""
    try:
        url = f"{settings.ollama_host.rstrip('/')}/api/tags"
        with urllib.request.urlopen(url, timeout=10) as resp:
            tags = json.loads(resp.read())
    except Exception as exc:
        return False, f"{ErrorCode.OLLAMA_NOT_AVAILABLE.value}: {exc}"

    names = {m.get("name", "") for m in tags.get("models", [])}
    if settings.ollama_model not in names:
        return False, (
            f"{ErrorCode.QWEN_MODEL_NOT_AVAILABLE.value}: "
            f"{settings.ollama_model} not in {sorted(names)}"
        )
    return True, f"Ollama ready with {settings.ollama_model}"


def ollama_generate(prompt: str, image_paths: list[Path] | None = None) -> str:
    """One non-streaming local generation. Optionally attaches local images."""
    payload: dict = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": settings.ollama_keep_alive,
        "options": {
            "num_ctx": settings.ollama_num_ctx,
            "temperature": settings.ollama_temperature,
        },
    }

    if image_paths:
        encoded: list[str] = []
        for path in image_paths[: settings.max_inspect_images]:
            resolved = Path(path).resolve()
            if not resolved.exists():
                raise OllamaError(
                    f"{ErrorCode.FILE_NOT_FOUND.value}: {resolved}",
                    ErrorCode.FILE_NOT_FOUND,
                )
            encoded.append(base64.b64encode(resolved.read_bytes()).decode("utf-8"))
        payload["images"] = encoded

    data = _post("/api/generate", payload)
    text = (data.get("response") or "").strip()
    if not text:
        raise OllamaError("Ollama returned an empty response.", ErrorCode.VISION_FAILED)
    return text


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def format_evidence(evidence: list[RerankedResult], filenames: dict[str, str]) -> str:
    """Render evidence with provenance. Order encodes priority: the caller
    passes direct evidence first, relationship-expanded evidence last."""
    blocks: list[str] = []
    budget = settings.max_context_chars

    for position, result in enumerate(evidence, start=1):
        item = result.item
        filename = filenames.get(item.source_id, "unknown")
        where = item.location.human()
        header = f"EVIDENCE {position} | {filename}"
        if where:
            header += f" | {where}"
        header += f" | modality={item.modality.value}"
        if item.metadata.get("expanded_via"):
            header += f" | linked via {item.metadata['expanded_via']}"

        body = item.content
        if len(body) > budget:
            body = body[:budget] + "..."
        budget -= len(body)
        blocks.append(f"{header}\n{body}")
        if budget <= 0:
            break
    return "\n\n".join(blocks)


def build_prompt(
    query: str,
    evidence: list[RerankedResult],
    filenames: dict[str, str],
) -> str:
    discovery = is_discovery_query(query)
    rules = DISCOVERY_RULES if discovery else SYSTEM_RULES
    closing = (
        "List and describe the related evidence above."
        if discovery
        else "Answer using only the evidence above."
    )
    return (
        f"{rules}\n"
        f"=================== EVIDENCE ===================\n"
        f"{format_evidence(evidence, filenames)}\n"
        f"================================================\n\n"
        f"USER REQUEST: {query}\n\n"
        f"{closing}"
    )


def generate_answer(
    query: str,
    evidence: list[RerankedResult],
    filenames: dict[str, str],
    inspect_image: Path | None = None,
) -> str:
    """Produce a grounded answer. `inspect_image` optionally hands the
    original image to the vision model for direct inspection."""
    prompt = build_prompt(query, evidence, filenames)
    images = [inspect_image] if inspect_image else None
    return ollama_generate(prompt, image_paths=images)


def pick_inspectable_image(evidence: list[RerankedResult]) -> Path | None:
    """Return the single highest-ranked image worth inspecting directly.
    Capped at one because each image costs ~1030 of the 8192 context tokens."""
    for result in evidence:
        if result.item.modality is Modality.IMAGE:
            path_str = result.item.location.image_path
            if path_str and Path(path_str).exists():
                return Path(path_str)
    return None
