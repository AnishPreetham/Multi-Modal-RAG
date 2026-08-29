"""
config.py — single source of truth for every tunable in Evidence AI.

Values resolve in this order: environment variable -> .env file -> default
below. Import `settings` anywhere; it is constructed once at import time.

Nothing here contacts the network. The two HF_* environment variables are
set before any HuggingFace import so that model loading stays local and,
on Windows without Developer Mode, does not attempt unprivileged symlinks.
"""
from __future__ import annotations

import os
from pathlib import Path

# Must be set before huggingface_hub / sentence_transformers are imported
# anywhere in the process. Windows without Developer Mode cannot create
# symlinks (WinError 1314), so the hub is told to copy blobs instead.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


def _bridge_streamlit_secrets() -> None:
    """Copy Streamlit Cloud secrets into the environment.

    Hosted Streamlit supplies secrets through `st.secrets`, which
    pydantic-settings does not read. Bridging them here means the same
    Settings class works locally (.env) and hosted (platform secrets) with
    no code change. Existing environment variables always win, and nothing
    is logged -- only key names are ever touched.
    """
    try:
        import streamlit as st

        for name, value in st.secrets.items():
            if isinstance(value, str) and name not in os.environ:
                os.environ[name] = value
    except Exception:
        # Not running under Streamlit, or no secrets configured. Both fine.
        pass


_bridge_streamlit_secrets()


class Settings(BaseSettings):
    """Runtime configuration. Every field is overridable from .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Inference backend ---------------------------------------------
    # "offline" -> local Ollama;  "online" -> cloud API.
    # This is the backend the UI starts on; the user can switch at runtime.
    inference_backend: str = "offline"

    # ---- Online (cloud) inference ---------------------------------------
    # Read from the environment / .env only. Never hard-code a key here.
    openai_api_key: str = ""
    online_model: str = "gpt-4o-mini"
    # Any OpenAI-compatible endpoint works (Azure OpenAI, OpenRouter, a
    # self-hosted gateway) without touching provider code.
    online_base_url: str = "https://api.openai.com/v1"
    online_timeout: int = 120
    online_model_supports_vision: bool = True
    # Named in user-facing setup messages so they know exactly what to set.
    online_api_key_env: str = "OPENAI_API_KEY"

    # ---- Local LLM (Ollama) ------------------------------------------
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5vl:3b"
    ollama_timeout: int = 300
    # Keeps the model resident between queries. Ollama's 5-minute default
    # causes a measured ~14s reload penalty mid-demo.
    ollama_keep_alive: str = "30m"
    # One image costs ~1030 prompt tokens; the 4096 default leaves too
    # little room for text evidence alongside it.
    ollama_num_ctx: int = 8192
    ollama_temperature: float = 0.0

    # ---- Local models -------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # 4GB VRAM is fully committed to Ollama; PyTorch stays on CPU.
    torch_device: str = "cpu"

    # ---- Chunking -----------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_chunk_chars: int = 40

    # ---- Retrieval ----------------------------------------------------
    top_k: int = 10
    rerank_top_k: int = 5

    # ---- Relationships ------------------------------------------------
    related_to_threshold: float = 0.70
    max_relationship_expansion: int = 3

    # ---- Generation context -------------------------------------------
    max_context_items: int = 5
    max_context_chars: int = 6000
    max_excerpt_chars: int = 400
    # At most one image may be passed to the vision model per query.
    max_inspect_images: int = 1

    # ---- Grounding / abstention ----------------------------------------
    # Cross-encoder logit below which evidence is considered unsupportive.
    abstention_threshold: float = -6.0
    # Cosine similarity required to attribute an answer sentence to evidence.
    citation_threshold: float = 0.35

    # ---- Ingestion limits (security) -----------------------------------
    max_file_bytes: int = 200 * 1024 * 1024
    max_query_chars: int = 2000

    # ---- Paths ---------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    index_dir: Path = PROJECT_ROOT / "index"
    evaluation_dir: Path = PROJECT_ROOT / "evaluation"

    # ---- External binaries ---------------------------------------------
    tesseract_cmd: str = Field(default=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    soffice_cmd: str = Field(default=r"C:\Program Files\LibreOffice\program\soffice.exe")

    # ---- Derived paths --------------------------------------------------
    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def db_path(self) -> Path:
        return self.index_dir / "store.db"

    def ensure_dirs(self) -> None:
        for path in (
            self.documents_dir, self.images_dir, self.audio_dir,
            self.index_dir, self.evaluation_dir / "reports",
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

# Point pytesseract at the installed binary if the default path exists.
if Path(settings.tesseract_cmd).exists():
    os.environ.setdefault("TESSERACT_CMD", settings.tesseract_cmd)
