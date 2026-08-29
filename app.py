"""
app.py -- Streamlit interface for Evidence AI.

Calls rag_pipeline directly (the same object the API uses), so the UI and
the API cannot drift apart.

Run:  .venv\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from config import settings
from schemas import InferenceMode, Modality, QueryModality, QueryRequest, clock

st.set_page_config(page_title="Evidence AI", page_icon="🔍", layout="wide")

MODALITY_ICON = {
    Modality.TEXT: "📄",
    Modality.IMAGE: "🖼️",
    Modality.AUDIO: "🎧",
}


@st.cache_resource(show_spinner="Loading local models and index...")
def load_pipeline():
    from rag_pipeline import get_pipeline
    return get_pipeline()


def select_backend() -> "object":
    """Inference-mode selector. Returns the chosen Backend."""
    from providers import Backend

    with st.sidebar:
        st.subheader("Inference mode")
        choice = st.radio(
            "Where should the AI run?",
            ["Offline", "Online"],
            index=0 if settings.inference_backend == "offline" else 1,
            captions=[
                "Local AI on this machine",
                "Cloud AI over the internet",
            ],
            key="backend_choice",
            label_visibility="collapsed",
        )
    return Backend.OFFLINE if choice == "Offline" else Backend.ONLINE


def render_backend_status(pipeline, backend) -> bool:
    """Show readiness for the selected backend and offer recovery actions.

    Returns True when the backend can serve a query. Never raises, and never
    silently switches backend -- the user's choice is always respected.
    """
    from providers import Backend, ollama_installed, start_ollama

    info = pipeline.health(backend)["inference"]

    with st.sidebar:
        if info["ready"]:
            st.success(f"✓ {info['message']}")
        else:
            st.warning(f"⚠ {info['message']}")

        if not info["ready"]:
            if backend is Backend.ONLINE:
                st.caption(
                    f"Add your cloud API key as `{settings.online_api_key_env}` "
                    "to the local `.env` file, then reload the page."
                )
                st.code(f"{settings.online_api_key_env}=your_api_key_here",
                        language="bash")
                if st.button("Check again", key="recheck_online"):
                    st.cache_resource.clear()
                    st.rerun()
            else:
                st.caption(
                    "Evidence AI uses local AI for Offline mode. "
                    "It runs on this machine, so nothing leaves your computer."
                )
                installed = ollama_installed()
                if not installed:
                    st.caption("Offline AI requires Ollama to run AI locally.")
                    st.link_button("Install Ollama", "https://ollama.com/download")
                else:
                    if st.button("Start Offline AI", key="start_ollama"):
                        with st.spinner("Starting local AI service..."):
                            started, detail = start_ollama()
                        if started:
                            st.cache_resource.clear()
                            st.rerun()
                        else:
                            st.error("Could not start Offline AI automatically.")
                            with st.expander("Technical details"):
                                st.code(detail)
                    if info.get("code") != "OLLAMA_NOT_AVAILABLE":
                        st.caption(
                            f"Required local model: `{settings.ollama_model}`"
                        )
                        st.code(f"ollama pull {settings.ollama_model}",
                                language="bash")

                cols = st.columns(2)
                if cols[0].button("Check again", key="recheck_offline"):
                    st.cache_resource.clear()
                    st.rerun()
                if cols[1].button("Use Online", key="switch_online"):
                    st.session_state["backend_choice"] = "Online"
                    st.rerun()

            if info["technical"]:
                with st.expander("Technical details"):
                    st.code(info["technical"])

    return bool(info["ready"])


def render_health(pipeline, backend) -> None:
    health = pipeline.health(backend)
    index = health["index"]

    with st.sidebar:
        st.subheader("Runtime")
        st.write(("🟢" if health["tesseract"]["available"] else "🔴") + " Tesseract OCR")
        st.write(("🟢" if health["libreoffice"]["available"] else "⚪")
                 + " LibreOffice (.doc)")

        st.subheader("Index")
        col_a, col_b = st.columns(2)
        col_a.metric("Sources", index["sources"])
        col_b.metric("Items", index["items"])
        col_a.metric("Vectors", index["vectors"])
        col_b.metric("Links", index["relationships"])
        if index["by_modality"]:
            st.caption(" · ".join(f"{k}: {v}" for k, v in index["by_modality"].items()))

        st.subheader("Models")
        for label, value in health["models"].items():
            st.caption(f"{label}: `{value}`")
        st.caption(
            "Ingestion (OCR, image understanding, transcription, embeddings, "
            "retrieval) always runs locally. Only answer generation uses the "
            "cloud in Online mode."
        )


def render_evidence(pipeline, response) -> None:
    st.subheader("Evidence")
    for result in response.retrieved_items:
        item = result.item
        source = pipeline.store.get_source(item.source_id)
        filename = source.filename if source else "unknown"
        icon = MODALITY_ICON.get(item.modality, "📄")
        where = item.location.human()
        label = f"{icon} #{result.rank} · {filename}" + (f" · {where}" if where else "")

        with st.expander(label):
            cols = st.columns(3)
            cols[0].metric("Retrieval", f"{result.retrieval_score:.3f}")
            cols[1].metric("Rerank", f"{result.rerank_score:.3f}")
            cols[2].metric("Modality", item.modality.value)

            if item.modality is Modality.IMAGE and item.location.image_path:
                image_path = Path(item.location.image_path)
                if image_path.exists():
                    st.image(str(image_path), width=420)
                description = item.metadata.get("visual_description")
                ocr_text = item.metadata.get("ocr_text")
                if description:
                    st.markdown("**Visual description (Qwen2.5-VL)**")
                    st.write(description)
                if ocr_text:
                    st.markdown("**OCR text (Tesseract)**")
                    st.code(ocr_text[:1200])
                if item.metadata.get("vision_ok") is False:
                    st.warning("Vision model unavailable — OCR-only representation.")
            elif item.modality is Modality.AUDIO:
                start = item.location.timestamp_start
                end = item.location.timestamp_end
                if start is not None and end is not None:
                    st.caption(f"Transcript segment {clock(start)}–{clock(end)}")
                if source and Path(source.file_path).exists():
                    st.audio(source.file_path)
                st.write(item.content)
            else:
                st.write(item.content)

            st.caption(f"item_id `{item.item_id}` · source_id `{item.source_id}`")


def render_relationships(response) -> None:
    if not response.relationships:
        return
    st.subheader("Relationships")
    for rel in response.relationships[:12]:
        confidence = f" · {rel.confidence:.2f}" if rel.confidence is not None else ""
        cross = " · cross-modal" if rel.metadata.get("cross_modal") else ""
        st.caption(
            f"`{rel.relationship_type.value}`{confidence}{cross} — "
            f"{rel.source_item_id[:8]} ↔ {rel.target_item_id[:8]}"
        )


def render_response(pipeline, response) -> None:
    if response.abstained:
        st.warning(f"**Abstained.** {response.answer}")
        if response.error_code:
            st.caption(f"Reason: `{response.error_code.value}`")
    else:
        st.success("Grounded answer")
        st.markdown(response.answer)

    if response.confidence is not None:
        st.caption(
            f"Confidence {response.confidence:.2f} — a normalised rerank score, "
            "not a calibrated probability."
        )

    if response.citations:
        from citations import render_citation

        st.subheader("Citations")
        for citation in response.citations:
            with st.container(border=True):
                st.markdown(f"**{render_citation(citation)}**")
                st.caption(f"{citation.modality.value} · source `{citation.source_id}`")
                if citation.excerpt:
                    st.caption(citation.excerpt)
                if citation.modality is Modality.IMAGE and citation.location.image_path:
                    path = Path(citation.location.image_path)
                    if path.exists():
                        st.image(str(path), width=320)

    render_relationships(response)
    render_evidence(pipeline, response)

    if response.latency_ms:
        with st.expander("Timing"):
            for stage, value in response.latency_ms.items():
                st.caption(f"{stage}: {value:.0f} ms")


def main() -> None:
    from providers import Backend

    st.title("🔍 Evidence AI")
    st.caption(
        "Multimodal RAG over documents, images and audio — with grounded "
        "answers, citations and abstention."
    )

    backend = select_backend()
    pipeline = load_pipeline()
    backend_ready = render_backend_status(pipeline, backend)
    render_health(pipeline, backend)

    if backend is Backend.OFFLINE:
        st.caption("🔒 Offline mode — every stage runs on this machine.")
    else:
        st.caption(
            "☁️ Online mode — retrieval stays local; retrieved evidence is "
            "sent to the cloud model to write the answer."
        )

    tab_query, tab_ingest, tab_sources = st.tabs(
        ["Ask", "Ingest", "Sources"]
    )

    # ---------------------------------------------------------------- ingest
    with tab_ingest:
        st.subheader("Add sources to the index")
        st.caption(
            "Upload each source **once**. It is processed into indexed items "
            "that persist across restarts, so you can then ask any number of "
            "questions in the **Ask** tab without uploading it again."
        )
        uploads = st.file_uploader(
            "PDF, DOC, DOCX, TXT, CSV, PNG, JPG, WEBP, WAV, MP3, M4A",
            type=["pdf", "doc", "docx", "txt", "csv", "png", "jpg", "jpeg",
                  "webp", "wav", "mp3", "m4a"],
            accept_multiple_files=True,
        )
        if uploads and st.button("Ingest files", type="primary"):
            for upload in uploads:
                with st.spinner(f"Processing {upload.name}..."):
                    path = pipeline.store_upload(upload.name, upload.getvalue())
                    result = pipeline.ingest_file(str(path))
                if result.success:
                    st.success(
                        f"{result.source.filename} — {len(result.items)} items, "
                        f"{len(result.relationships)} relationships"
                    )
                else:
                    st.error(
                        f"{result.source.filename} — "
                        f"{result.error_code.value if result.error_code else 'FAILED'}: "
                        f"{result.error}"
                    )
                for warning in result.warnings:
                    st.warning(warning)
            st.cache_resource.clear()

    # ----------------------------------------------------------------- query
    with tab_query:
        stats = pipeline.store.stats()
        if stats["items"] == 0:
            st.warning(
                "Nothing is indexed yet. Add sources in the **Ingest** tab first — "
                "you only need to do that once, then you can ask as many "
                "questions as you like."
            )

        st.caption(
            f"Searching {stats['items']} indexed items from "
            f"{stats['sources']} source(s). Ingested sources stay indexed — "
            "you never need to upload them again to ask about them."
        )

        mode = st.radio(
            "How do you want to ask?",
            ["Text", "Image", "Audio", "Document"],
            horizontal=True,
            captions=[
                "Type a question",
                "Query by example image",
                "Query by example audio",
                "Query by example document",
            ],
            help="Text searches everything already indexed. The other three let "
                 "you use a NEW file as the query itself — for finding indexed "
                 "material related to it. They do not re-ingest anything.",
        )
        modality = QueryModality[mode.upper()]

        query_file = None
        if modality is QueryModality.TEXT:
            st.caption(
                "Asks your question against the whole indexed corpus — "
                "documents, images and audio together. No upload needed."
            )
        else:
            extensions = {
                QueryModality.IMAGE: ["png", "jpg", "jpeg", "webp"],
                QueryModality.AUDIO: ["wav", "mp3", "m4a"],
                QueryModality.DOCUMENT: ["pdf", "docx", "txt", "csv"],
            }[modality]
            st.caption(
                f"Optional. Upload a **new** {mode.lower()} to use as the query "
                f"itself and find indexed material related to it. This file is "
                f"not added to the index. To add sources, use the Ingest tab."
            )
            query_file = st.file_uploader(
                f"{mode} to query with", type=extensions, key=f"q_{mode}"
            )

        placeholder = (
            "e.g. What were the main findings across the reports?"
            if modality is QueryModality.TEXT
            else f"Optional — leave blank to just find related material."
        )
        default_question = (
            "" if modality is QueryModality.TEXT
            else f"Find evidence related to this {mode.lower()}."
        )
        question = st.text_area(
            "Question", value=default_question, height=80, placeholder=placeholder,
        )

        col_a, col_b, col_c = st.columns(3)
        top_k = col_a.slider("Retrieve (top_k)", 1, 50, settings.top_k)
        rerank_top_k = col_b.slider("Rerank to", 1, 20, settings.rerank_top_k)
        use_vision = col_c.checkbox(
            "Let Qwen inspect the top image",
            value=False,
            help="Passes one original image to the vision model. Slower "
                 "(~1030 extra context tokens).",
        )

        if st.button("Search", type="primary", disabled=not backend_ready):
            temp_path = None
            if modality is QueryModality.TEXT:
                if not question.strip():
                    st.error("Type a question to search the indexed corpus.")
                    st.stop()
            else:
                if query_file is None:
                    st.error(
                        f"Select a {mode.lower()} file to query with, or switch "
                        f"to **Text** to ask a question about what is already "
                        f"indexed."
                    )
                    st.stop()
                suffix = Path(query_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    handle.write(query_file.getvalue())
                    temp_path = handle.name

            request = QueryRequest(
                query=question.strip() or f"Find evidence related to this {mode.lower()}.",
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                inference_mode=(InferenceMode.LOCAL_VISION if use_vision
                                else InferenceMode.LOCAL),
                query_modality=modality,
                query_file_path=temp_path,
            )
            from rag_pipeline import last_technical_detail

            with st.spinner("Retrieving, reranking, generating..."):
                response = pipeline.query(request, backend=backend)
            technical = last_technical_detail()
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

            if technical:
                with st.expander("Technical details"):
                    st.code(technical)

            if response.query_representation and modality is not QueryModality.TEXT:
                with st.expander("Query representation used for retrieval"):
                    st.code(response.query_representation[:2000])

            render_response(pipeline, response)

    # --------------------------------------------------------------- sources
    with tab_sources:
        sources = pipeline.store.list_sources()
        if not sources:
            st.info("Nothing indexed yet. Use the Ingest tab.")
        for source in sources:
            with st.expander(f"{source.filename}  ·  {source.source_type.value}"):
                st.caption(
                    f"source_id `{source.source_id}` · "
                    f"{(source.file_size or 0) / 1024:.0f} KB · {source.file_path}"
                )
                items = pipeline.store.get_items_by_source(source.source_id)
                st.write(f"{len(items)} indexed items")
                for item in items[:20]:
                    where = item.location.human()
                    prefix = f"{where} — " if where else ""
                    st.caption(f"{prefix}{item.content[:180]}")


if __name__ == "__main__":
    main()
