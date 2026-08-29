"""
audio.py -- local speech-to-text with faster-whisper.

    audio -> WhisperModel.transcribe -> timestamped segments
          -> one ContentItem per meaningful segment
          -> TRANSCRIPT_OF relationships between consecutive segments

The Whisper model is loaded once per process and reused (see _get_model).
Speaker identities are never invented -- Whisper does not perform
diarization and this module does not pretend otherwise.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from config import settings
from schemas import (
    ContentItem,
    ErrorCode,
    IngestionResult,
    Modality,
    Relationship,
    RelationshipType,
    Source,
    SourceLocation,
    SourceType,
    content_hash,
    derive_id,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

MIME_BY_EXTENSION = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}

# Segments shorter than this carry no retrievable meaning on their own.
MIN_SEGMENT_CHARS = 15

# Whisper will happily emit text for non-speech audio (music, noise, silence)
# when its VAD lets a chunk through. Those segments come back with a high
# no_speech_prob and/or a very poor avg_logprob. Dropping them is what stops
# a song from being indexed as if it were a fabricated transcript.
MAX_NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.0

# Whisper emits one segment per breath group -- often 4-8 seconds and a
# single clause. Indexed alone, those lose to longer document chunks in both
# the bi-encoder and the cross-encoder purely because they carry less text.
# Adjacent segments are therefore merged into windows before indexing, which
# keeps full timestamp traceability (start of the first, end of the last)
# while giving retrieval a unit with comparable information density.
AUDIO_WINDOW_CHARS = 320
# A pause longer than this ends a window: it usually marks a topic change.
AUDIO_WINDOW_MAX_GAP_S = 2.0

_model = None
_model_lock = threading.Lock()


def _get_model():
    """Load faster-whisper once per process. Loading takes ~20s on this CPU,
    so doing it per file (let alone per segment) would be unusable."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from faster_whisper import WhisperModel

                logger.info(
                    "Loading faster-whisper %s (%s, %s)",
                    settings.whisper_model_size,
                    settings.whisper_device,
                    settings.whisper_compute_type,
                )
                _model = WhisperModel(
                    settings.whisper_model_size,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type,
                )
    return _model


def _validate(path_str: str) -> Path:
    path = Path(path_str).resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{ErrorCode.FILE_NOT_FOUND.value}: {path_str}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"{ErrorCode.UNSUPPORTED_FILE_TYPE.value}: {path.suffix}")
    if path.stat().st_size > settings.max_file_bytes:
        raise ValueError(f"{ErrorCode.FILE_TOO_LARGE.value}: {path.name}")
    return path


def transcribe(path: Path) -> list[dict]:
    """Return [{start, end, text}] for one audio file.

    Decoding is delegated to faster-whisper, which reads WAV/MP3/M4A/FLAC/OGG
    through PyAV and resamples to the 16 kHz mono the model expects. Sample
    rate and channel count therefore need no handling here.

    Segments Whisper is not confident are speech are discarded rather than
    indexed: see MAX_NO_SPEECH_PROB.
    """
    model = _get_model()
    try:
        segments, _info = model.transcribe(
            str(path),
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            # Without this, a non-speech stretch makes Whisper loop on its own
            # previous output ("thank you. thank you. thank you.").
            condition_on_previous_text=False,
        )
        kept: list[dict] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            no_speech = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
            avg_logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
            if no_speech > MAX_NO_SPEECH_PROB or avg_logprob < MIN_AVG_LOGPROB:
                logger.info(
                    "Dropping low-confidence segment [%.1f-%.1f] "
                    "no_speech_prob=%.2f avg_logprob=%.2f",
                    segment.start, segment.end, no_speech, avg_logprob,
                )
                continue
            kept.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
                "no_speech_prob": no_speech,
                "avg_logprob": avg_logprob,
            })
        return kept
    except Exception as exc:
        raise RuntimeError(f"{ErrorCode.TRANSCRIPTION_FAILED.value}: {exc}") from exc


def merge_segments(
    segments: list[dict],
    window_chars: int = AUDIO_WINDOW_CHARS,
    max_gap_s: float = AUDIO_WINDOW_MAX_GAP_S,
) -> list[dict]:
    """Group consecutive Whisper segments into retrieval-sized windows.

    A window closes when adding the next segment would exceed `window_chars`,
    or when the silence before it exceeds `max_gap_s`. Each window keeps the
    start time of its first segment and the end time of its last, so audio
    citations still point at a real, playable span of the recording.
    """
    if not segments:
        return []

    windows: list[dict] = []
    current = dict(segments[0])
    current["segment_count"] = 1

    for segment in segments[1:]:
        gap = segment["start"] - current["end"]
        too_long = len(current["text"]) + len(segment["text"]) + 1 > window_chars
        if too_long or gap > max_gap_s:
            windows.append(current)
            current = dict(segment)
            current["segment_count"] = 1
            continue
        current["text"] = f"{current['text']} {segment['text']}".strip()
        current["end"] = segment["end"]
        current["segment_count"] += 1

    windows.append(current)
    return windows


def transcribe_to_text(path_str: str) -> str:
    """Flat transcript, used when audio is the query rather than a source."""
    return " ".join(s["text"] for s in transcribe(Path(path_str).resolve())).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_audio(path_str: str) -> IngestionResult:
    """Turn one audio file into per-segment ContentItems. Never raises."""
    source: Source | None = None
    warnings: list[str] = []
    try:
        path = _validate(path_str)
        data = path.read_bytes()
        source = Source(
            source_id=content_hash(data),
            filename=path.name,
            source_type=SourceType.AUDIO,
            file_path=str(path),
            mime_type=MIME_BY_EXTENSION.get(path.suffix.lower(), "audio/wav"),
            file_size=len(data),
        )

        raw_segments = transcribe(path)
        segments = merge_segments(raw_segments)
        kept = [s for s in segments if len(s["text"]) >= MIN_SEGMENT_CHARS]
        if not kept:
            # These are different problems and lead to different user actions,
            # so they get different messages rather than one generic failure.
            if not raw_segments:
                detail = (
                    "No speech was detected. The file appears to contain music, "
                    "silence or noise rather than spoken words. Ingest a "
                    "recording of speech instead."
                )
            else:
                detail = (
                    f"Speech was detected but every passage was shorter than "
                    f"{MIN_SEGMENT_CHARS} characters, which is too little to "
                    f"retrieve on. Use a longer recording."
                )
            return IngestionResult(
                source=source,
                items=[],
                success=False,
                error=detail,
                error_code=ErrorCode.TRANSCRIPTION_FAILED,
                warnings=warnings,
            )
        if len(kept) < len(segments):
            warnings.append(
                f"Dropped {len(segments) - len(kept)} segment(s) below "
                f"{MIN_SEGMENT_CHARS} characters."
            )

        items = [
            ContentItem(
                item_id=derive_id(source.source_id, "audio", index),
                source_id=source.source_id,
                modality=Modality.AUDIO,
                content=segment["text"],
                location=SourceLocation(
                    timestamp_start=segment["start"],
                    timestamp_end=segment["end"],
                ),
                metadata={
                    "segment_index": index,
                    "duration_s": round(segment["end"] - segment["start"], 2),
                    "merged_segments": segment.get("segment_count", 1),
                    "whisper_model": settings.whisper_model_size,
                },
            )
            for index, segment in enumerate(kept)
        ]
        return IngestionResult(
            source=source,
            items=items,
            relationships=build_audio_relationships(items),
            success=True,
            warnings=warnings,
        )

    except Exception as exc:
        logger.error("process_audio failed for %s: %s", path_str, exc)
        return _failure(path_str, source, exc, warnings)


def build_audio_relationships(items: list[ContentItem]) -> list[Relationship]:
    """TRANSCRIPT_OF links consecutive segments of the same recording, which
    is what lets the pipeline pull in the sentence before or after a hit."""
    relationships: list[Relationship] = []
    for i in range(len(items) - 1):
        a, b = items[i], items[i + 1]
        relationships.append(Relationship(
            relationship_id=derive_id(a.item_id, b.item_id, "transcript_of"),
            source_item_id=a.item_id,
            target_item_id=b.item_id,
            relationship_type=RelationshipType.TRANSCRIPT_OF,
            confidence=1.0,
            metadata={"adjacent_segments": True},
        ))
    return relationships


def _failure(
    path_str: str,
    source: Source | None,
    exc: Exception,
    warnings: list[str],
) -> IngestionResult:
    message = str(exc)
    code = ErrorCode.TRANSCRIPTION_FAILED
    for candidate in ErrorCode:
        if message.startswith(candidate.value):
            code = candidate
            break
    if source is None:
        path = Path(path_str)
        source = Source(
            source_id=derive_id(path_str),
            filename=path.name or "unknown",
            source_type=SourceType.AUDIO,
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
