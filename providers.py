"""
providers.py -- the inference provider layer.

Evidence AI supports two mandatory inference backends. Everything upstream
of generation (ingestion, embeddings, FAISS, retrieval, reranking,
relationships, abstention, citations) is shared and provider-independent;
only the final "evidence -> prose" step differs.

    RAG pipeline
         |
    InferenceProvider
       /            \
    OllamaProvider   OpenAIProvider
    (offline, local) (online, cloud)

Design notes:

* The `Backend` axis lives here, NOT in schemas.py. `InferenceMode` in the
  locked contract already means something else (whether the original image
  is handed to the vision model), so overloading it would conflate two
  independent choices. schemas.py is unchanged.

* Both providers reuse `llm.build_prompt`, so the grounding rules,
  discovery-vs-factual prompt selection, evidence ordering and context
  budget are identical online and offline. There is exactly one set of
  prompt rules in this codebase.

* Online mode never touches Ollama. `readiness()` for ONLINE checks
  credentials only, so the application is fully usable on a machine where
  Ollama is not installed.

* No API key is ever logged, echoed into an exception message, or returned
  in a readiness detail. See `_redact`.
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from config import settings
from schemas import ErrorCode, Modality, RerankedResult

logger = logging.getLogger(__name__)


class Backend(str, Enum):
    """Which inference provider serves generation."""
    OFFLINE = "offline"
    ONLINE = "online"


@dataclass(frozen=True)
class Readiness:
    """Whether a backend can serve a request right now.

    `message` is written for a non-technical user. `technical` carries the
    detail that belongs behind a "Technical details" expander -- never in
    the headline.
    """
    ready: bool
    message: str
    technical: str = ""
    code: ErrorCode | None = None
    # UI hint: offline setup problems are actionable in different ways.
    can_autostart: bool = False


class ProviderError(RuntimeError):
    """Generation failed. Carries a user-safe message plus technical detail."""

    def __init__(self, user_message: str, technical: str, code: ErrorCode):
        super().__init__(user_message)
        self.user_message = user_message
        self.technical = technical
        self.code = code


def _redact(text: str) -> str:
    """Strip anything resembling a credential out of text bound for a user,
    a log, or an exception. Defence in depth: the key should never reach
    these paths in the first place."""
    key = settings.openai_api_key
    if key and len(key) > 8:
        text = text.replace(key, "***REDACTED***")
    return text


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class InferenceProvider(ABC):
    """Turns retrieved evidence into a grounded answer."""

    backend: Backend
    label: str

    @abstractmethod
    def readiness(self) -> Readiness:
        """Can this provider serve a request? Must not raise."""

    @abstractmethod
    def generate(
        self,
        query: str,
        evidence: list[RerankedResult],
        filenames: dict[str, str],
        inspect_image: Path | None = None,
    ) -> str:
        """Produce the answer text. Raises ProviderError on failure."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier shown in the UI."""


# ---------------------------------------------------------------------------
# Offline -- local Ollama
# ---------------------------------------------------------------------------

class OllamaProvider(InferenceProvider):
    """Wraps the existing local Ollama integration in llm.py.

    This adds no new inference logic: it delegates to the functions that
    already power offline mode, so offline behaviour is unchanged.
    """

    backend = Backend.OFFLINE
    label = "Offline AI"

    @property
    def model_name(self) -> str:
        return settings.ollama_model

    def readiness(self) -> Readiness:
        from llm import ollama_available

        ok, detail = ollama_available()
        if ok:
            return Readiness(True, f"Offline AI ready — {settings.ollama_model}", detail)

        if detail.startswith(ErrorCode.QWEN_MODEL_NOT_AVAILABLE.value):
            return Readiness(
                False,
                f"The local model '{settings.ollama_model}' isn't installed yet.",
                detail,
                ErrorCode.QWEN_MODEL_NOT_AVAILABLE,
            )
        return Readiness(
            False,
            "Offline AI isn't ready yet.",
            detail,
            ErrorCode.OLLAMA_NOT_AVAILABLE,
            can_autostart=True,
        )

    def generate(
        self,
        query: str,
        evidence: list[RerankedResult],
        filenames: dict[str, str],
        inspect_image: Path | None = None,
    ) -> str:
        from llm import OllamaError, generate_answer

        try:
            return generate_answer(query, evidence, filenames,
                                   inspect_image=inspect_image)
        except OllamaError as exc:
            raise ProviderError(
                "Offline AI could not generate an answer.",
                _redact(str(exc)),
                exc.code,
            ) from exc


def start_ollama() -> tuple[bool, str]:
    """Best-effort attempt to start a locally installed Ollama service.

    Returns (started, detail). Never raises -- a failure here is a UI state,
    not a crash.
    """
    import shutil
    import subprocess
    import time

    from llm import ollama_available

    if ollama_available()[0]:
        return True, "Ollama was already running."

    binary = shutil.which("ollama")
    if not binary:
        return False, "Ollama is not installed (no 'ollama' on PATH)."

    try:
        subprocess.Popen(
            [binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except Exception as exc:
        return False, f"Could not launch Ollama: {exc}"

    for _ in range(10):
        time.sleep(1.5)
        if ollama_available()[0]:
            return True, "Ollama started."
    return False, "Ollama was launched but did not become ready in time."


def ollama_installed() -> bool:
    """Whether the Ollama binary exists, independent of whether it runs."""
    import shutil
    return bool(shutil.which("ollama"))


# ---------------------------------------------------------------------------
# Online -- OpenAI-compatible cloud API
# ---------------------------------------------------------------------------

class OpenAIProvider(InferenceProvider):
    """Cloud generation over the OpenAI-compatible chat-completions API.

    Uses the standard library rather than the `openai` SDK: the payload is a
    single JSON POST, and avoiding the SDK keeps the offline dependency set
    untouched and adds no new package to install.

    `online_base_url` makes this usable with any OpenAI-compatible endpoint
    (Azure OpenAI, OpenRouter, a self-hosted gateway) without code changes.
    """

    backend = Backend.ONLINE
    label = "Online AI"

    @property
    def model_name(self) -> str:
        return settings.online_model

    def readiness(self) -> Readiness:
        """Credentials only. Deliberately does NOT probe Ollama, and does not
        spend an API call to validate the key -- authentication problems are
        reported when a request is actually made."""
        if not settings.openai_api_key.strip():
            return Readiness(
                False,
                "Online AI isn't configured yet.",
                f"No API key found. Set {settings.online_api_key_env} in your "
                f".env file (or your host's secret settings) and reload.",
                ErrorCode.OLLAMA_NOT_AVAILABLE,
            )
        return Readiness(
            True,
            f"Online AI ready — {settings.online_model}",
            f"Endpoint {settings.online_base_url}; key loaded from "
            f"{settings.online_api_key_env}.",
        )

    def _build_messages(
        self,
        prompt: str,
        inspect_image: Path | None,
    ) -> list[dict]:
        """Chat payload. When an image is supplied it is inlined as a base64
        data URI, which is how the chat-completions API accepts local images
        without uploading them anywhere first."""
        if inspect_image is None:
            return [{"role": "user", "content": prompt}]

        path = Path(inspect_image)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                },
            ],
        }]

    def generate(
        self,
        query: str,
        evidence: list[RerankedResult],
        filenames: dict[str, str],
        inspect_image: Path | None = None,
    ) -> str:
        # Same prompt builder as offline: identical grounding rules,
        # identical evidence formatting, identical abstention instruction.
        from llm import build_prompt

        if not settings.openai_api_key.strip():
            raise ProviderError(
                "Online AI isn't configured yet.",
                f"Set {settings.online_api_key_env} in your .env file and reload.",
                ErrorCode.OLLAMA_NOT_AVAILABLE,
            )

        prompt = build_prompt(query, evidence, filenames)
        payload = {
            "model": settings.online_model,
            "messages": self._build_messages(prompt, inspect_image),
            "temperature": settings.ollama_temperature,
        }

        request = urllib.request.Request(
            f"{settings.online_base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.openai_api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=settings.online_timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = _redact(exc.read().decode("utf-8", errors="replace")[:400])
            if exc.code in (401, 403):
                raise ProviderError(
                    "Online AI could not authenticate.",
                    f"HTTP {exc.code}. The API key was rejected. Check "
                    f"{settings.online_api_key_env} in your .env file. {body}",
                    ErrorCode.OLLAMA_NOT_AVAILABLE,
                ) from exc
            if exc.code == 429:
                raise ProviderError(
                    "Online AI is rate-limited or out of quota.",
                    f"HTTP 429. {body}",
                    ErrorCode.OLLAMA_NOT_AVAILABLE,
                ) from exc
            if exc.code == 404:
                raise ProviderError(
                    f"The online model '{settings.online_model}' is not available.",
                    f"HTTP 404. {body}",
                    ErrorCode.QWEN_MODEL_NOT_AVAILABLE,
                ) from exc
            raise ProviderError(
                "Online AI returned an error.",
                f"HTTP {exc.code}. {body}",
                ErrorCode.OLLAMA_NOT_AVAILABLE,
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(
                "Online AI is currently unavailable. Please check your "
                "internet connection or the API service.",
                _redact(f"{type(exc).__name__}: {exc.reason}"),
                ErrorCode.OLLAMA_NOT_AVAILABLE,
            ) from exc
        except Exception as exc:
            raise ProviderError(
                "Online AI is currently unavailable.",
                _redact(f"{type(exc).__name__}: {exc}"),
                ErrorCode.OLLAMA_NOT_AVAILABLE,
            ) from exc

        try:
            answer = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise ProviderError(
                "Online AI returned an unexpected response.",
                _redact(f"Malformed payload: {json.dumps(data)[:300]}"),
                ErrorCode.OLLAMA_NOT_AVAILABLE,
            ) from exc

        if not answer:
            raise ProviderError(
                "Online AI returned an empty answer.",
                "The API responded successfully but the message was empty.",
                ErrorCode.OLLAMA_NOT_AVAILABLE,
            )
        return answer


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

_PROVIDERS: dict[Backend, InferenceProvider] = {}


def resolve_backend(backend: Backend | str | None) -> Backend:
    """Normalise a backend selection, falling back to configuration."""
    if backend is None:
        backend = settings.inference_backend
    if isinstance(backend, Backend):
        return backend
    try:
        return Backend(str(backend).strip().lower())
    except ValueError:
        logger.warning("Unknown backend %r; using offline.", backend)
        return Backend.OFFLINE


def get_provider(backend: Backend | str | None = None) -> InferenceProvider:
    """Return the provider for a backend. Instances are cached and stateless."""
    resolved = resolve_backend(backend)
    if resolved not in _PROVIDERS:
        _PROVIDERS[resolved] = (
            OpenAIProvider() if resolved is Backend.ONLINE else OllamaProvider()
        )
    return _PROVIDERS[resolved]


def supports_vision(backend: Backend | str | None = None) -> bool:
    """Whether direct image inspection is available on this backend.

    Offline uses Qwen2.5-VL, which is a vision model. Online depends on the
    configured model, so this reflects the configured capability flag rather
    than guessing from the model name.
    """
    if resolve_backend(backend) is Backend.OFFLINE:
        return True
    return settings.online_model_supports_vision


def image_evidence_present(evidence: list[RerankedResult]) -> bool:
    """Whether any retrieved evidence is an image."""
    return any(r.item.modality is Modality.IMAGE for r in evidence)
