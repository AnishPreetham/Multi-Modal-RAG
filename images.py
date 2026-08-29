"""
images.py -- local image ingestion.

    image -> Pillow load -> OpenCV grayscale -> Tesseract OCR
                         -> Ollama qwen2.5vl:3b concise visual description
          -> content = "visual description + OCR text"
          -> ContentItem (original image path preserved on location)

ContentItem.content for an image is TEXT, never pixels: it is the Qwen
description concatenated with the OCR text. Both halves are also kept
separately in metadata so the UI can show them apart. This matters because
the combined string is what gets embedded -- putting OCR in metadata only
would make text inside screenshots unsearchable.

If Qwen fails we degrade to an OCR-only representation and flag it in
metadata. We never fabricate a description.
"""
from __future__ import annotations

import logging
from pathlib import Path

from config import settings
from schemas import (
    ContentItem,
    ErrorCode,
    IngestionResult,
    Modality,
    Source,
    SourceLocation,
    SourceType,
    content_hash,
    derive_id,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

MIME_BY_EXTENSION = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}

VISION_PROMPT = (
    "Describe this image in at most two concise sentences for search and "
    "retrieval. State what the image shows: its subject, layout, and any "
    "charts, diagrams, screenshots, tables or notable objects. Report only "
    "what is actually visible. Do not speculate and do not invent details."
)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _validate(path_str: str) -> Path:
    path = Path(path_str).resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{ErrorCode.FILE_NOT_FOUND.value}: {path_str}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"{ErrorCode.UNSUPPORTED_FILE_TYPE.value}: {path.suffix}")
    if path.stat().st_size > settings.max_file_bytes:
        raise ValueError(f"{ErrorCode.FILE_TOO_LARGE.value}: {path.name}")
    return path


def _load_image(path: Path):
    """Verify then load. verify() invalidates the handle, so the file is
    reopened; both opens are closed properly to avoid leaking handles."""
    from PIL import Image

    try:
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as handle:
            return handle.convert("RGB")
    except Exception as exc:
        raise ValueError(f"{ErrorCode.CORRUPT_FILE.value}: {exc}") from exc


def run_ocr(pil_image) -> str:
    """Plain grayscale first (best for clean screenshots and rendered text),
    Otsu threshold as a fallback for low-contrast scans. Returns '' on
    failure -- OCR is a supporting signal, not a hard requirement."""
    import cv2
    import numpy as np
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    try:
        bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()
        if text:
            return text
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return pytesseract.image_to_string(otsu).strip()
    except Exception as exc:
        logger.warning("Tesseract OCR failed: %s", exc)
        return ""


def describe_image(path: Path) -> str:
    """Concise visual description from the local vision model."""
    from llm import ollama_generate

    return ollama_generate(VISION_PROMPT, image_paths=[path]).strip()


def build_representation(description: str, ocr_text: str) -> str:
    """The string that actually gets embedded."""
    parts: list[str] = []
    if description:
        parts.append(f"Visual description: {description}")
    if ocr_text:
        parts.append(f"Text visible in image: {ocr_text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_image(path_str: str) -> IngestionResult:
    """Turn one image into an IngestionResult. Never raises."""
    source: Source | None = None
    warnings: list[str] = []
    try:
        path = _validate(path_str)
        data = path.read_bytes()
        source = Source(
            source_id=content_hash(data),
            filename=path.name,
            source_type=SourceType.IMAGE,
            file_path=str(path),
            mime_type=MIME_BY_EXTENSION.get(path.suffix.lower(), "image/png"),
            file_size=len(data),
        )

        pil_image = _load_image(path)
        width, height = pil_image.size
        ocr_text = run_ocr(pil_image)

        description = ""
        vision_ok = True
        try:
            description = describe_image(path)
        except Exception as exc:
            vision_ok = False
            warnings.append(f"Vision model unavailable, OCR-only item: {exc}")
            logger.warning("Qwen vision failed for %s: %s", path.name, exc)

        content = build_representation(description, ocr_text)
        if not content:
            return IngestionResult(
                source=source,
                items=[],
                success=False,
                error="Image produced neither OCR text nor a visual description.",
                error_code=ErrorCode.VISION_FAILED,
                warnings=warnings,
            )

        item = ContentItem(
            item_id=derive_id(source.source_id, "image", 0),
            source_id=source.source_id,
            modality=Modality.IMAGE,
            content=content,
            location=SourceLocation(image_path=str(path)),
            metadata={
                "ocr_text": ocr_text,
                "visual_description": description,
                "vision_ok": vision_ok,
                "width": width,
                "height": height,
            },
        )
        return IngestionResult(
            source=source,
            items=[item],
            success=True,
            warnings=warnings,
        )

    except Exception as exc:
        logger.error("process_image failed for %s: %s", path_str, exc)
        return _failure(path_str, source, exc, warnings)


def _failure(
    path_str: str,
    source: Source | None,
    exc: Exception,
    warnings: list[str],
) -> IngestionResult:
    message = str(exc)
    code = ErrorCode.CORRUPT_FILE
    for candidate in ErrorCode:
        if message.startswith(candidate.value):
            code = candidate
            break
    if source is None:
        path = Path(path_str)
        source = Source(
            source_id=derive_id(path_str),
            filename=path.name or "unknown",
            source_type=SourceType.IMAGE,
            file_path=str(path),
        )
    return IngestionResult(
        source=source,
        items=[],
        success=False,
        error=message,
        error_code=code,
        warnings=warnings,
    )
