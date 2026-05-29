"""
app.py — PaperLens Streamlit interface.

Run:  streamlit run app.py
URL:  http://localhost:8501
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

import config
from config import validate_required_keys
from db.ingestion_pipeline import IngestionPipeline
from db.weaviate_client import delete_collections, get_client
from generation.generator import get_generator
from memory.chat_history import ChatHistory
from retrieval.searcher import HybridSearcher

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PaperLens",
    page_icon="🔬",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation (runs once per browser session)
# ─────────────────────────────────────────────────────────────────────────────

def init_session() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatHistory()
    if "ingested_pdfs" not in st.session_state:
        st.session_state.ingested_pdfs: list[str] = []
    if "active_pdf" not in st.session_state:
        st.session_state.active_pdf: str | None = None
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = IngestionPipeline()
    if "searcher" not in st.session_state:
        st.session_state.searcher = HybridSearcher()


# ─────────────────────────────────────────────────────────────────────────────
# Startup validation
# ─────────────────────────────────────────────────────────────────────────────

def check_env() -> bool:
    missing = validate_required_keys()
    if missing:
        st.error(
            f"⚠️ Missing environment variables: **{', '.join(missing)}**\n\n"
            "Copy `.env.example` to `.env` and fill in your API keys."
        )
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — PDF management
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.title("🔬 PaperLens")
        st.caption("Multimodal RAG for scientific papers")

        st.divider()

        # ── Upload PDF ────────────────────────────────────────────────────
        st.subheader("📄 Upload PDF")
        uploaded = st.file_uploader(
            "Choose a PDF", type=["pdf"], label_visibility="collapsed"
        )

        title_input = st.text_input(
            "Document title (optional)",
            placeholder="e.g. Stanford AI Index 2025",
        )

        if uploaded and st.button("⚡ Ingest PDF", use_container_width=True):
            _handle_ingestion(uploaded, title_input)

        # ── Active document selector ──────────────────────────────────────
        if st.session_state.ingested_pdfs:
            st.divider()
            st.subheader("📚 Loaded Documents")
            options = ["All documents"] + st.session_state.ingested_pdfs
            selected = st.selectbox("Filter by:", options, label_visibility="collapsed")
            st.session_state.active_pdf = None if selected == "All documents" else selected

        # ── Danger zone ───────────────────────────────────────────────────
        st.divider()
        with st.expander("⚠️ Danger Zone"):
            if st.button("🗑️ Clear all data", type="secondary", use_container_width=True):
                _handle_clear_all()

        # ── Config info ───────────────────────────────────────────────────
        st.divider()
        st.caption(f"**Image model:** `{config.COLSMOL_MODEL}`")
        st.caption(f"**Text model:** `{config.MINILM_MODEL}`")
        st.caption(f"**Generator:** Gemini → Groq fallback")


def _handle_ingestion(uploaded_file, title: str) -> None:
    # Save to uploads dir
    save_path = config.UPLOADS_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner(f"Ingesting **{uploaded_file.name}**… (this may take a few minutes)"):
        result = st.session_state.pipeline.ingest(
            pdf_path=save_path,
            document_title=title or uploaded_file.name,
        )

    if result.success:
        st.success(
            f"✅ **{result.source_pdf}** ingested!\n\n"
            f"{result.pages_inserted} pages · {result.chunks_inserted} chunks"
        )
        if uploaded_file.name not in st.session_state.ingested_pdfs:
            st.session_state.ingested_pdfs.append(uploaded_file.name)
    else:
        st.error(f"❌ Ingestion failed:\n" + "\n".join(result.errors))


def _handle_clear_all() -> None:
    with st.spinner("Clearing Weaviate collections…"):
        try:
            delete_collections(get_client())
            st.session_state.ingested_pdfs = []
            st.session_state.active_pdf = None
            st.session_state.chat_history.clear()
            st.success("All data cleared.")
            st.rerun()
        except Exception as exc:
            st.error(f"Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────────────────────────────────────

def render_chat() -> None:
    st.title("Ask Your Research Papers")

    if not st.session_state.ingested_pdfs:
        st.info("👈 Upload a PDF in the sidebar to get started.")
        return

    # ── Display chat history ──────────────────────────────────────────────
    history = st.session_state.chat_history
    for msg in history.as_list():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────
    if query := st.chat_input("Ask a question about the paper…"):
        with st.chat_message("user"):
            st.markdown(query)
        history.add_user(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating…"):
                answer, pages, model_used = _answer_query(query)

            st.markdown(answer)

            # Show retrieved pages as thumbnails
            if pages:
                with st.expander(f"📎 {len(pages)} relevant page(s) retrieved"):
                    cols = st.columns(min(len(pages), 3))
                    for i, page in enumerate(pages):
                        with cols[i % 3]:
                            st.image(
                                page.image,
                                caption=f"Page {page.page_number} — {page.source_pdf}",
                                use_column_width=True,
                            )

            # Show which model was used
            st.caption(f"_Answered by **{model_used}**_")

        history.add_assistant(answer)

    # ── Clear chat button ─────────────────────────────────────────────────
    if not history.is_empty():
        if st.button("🔄 Clear chat", key="clear_chat"):
            history.clear()
            st.rerun()


def _answer_query(query: str):
    """Run search + generation; return (answer_text, pages, model_name)."""
    searcher: HybridSearcher = st.session_state.searcher
    results = searcher.search(
        query=query,
        source_pdf=st.session_state.active_pdf,
    )

    generator = get_generator()
    response = generator.generate(
        query=query,
        results=results,
        chat_history=st.session_state.chat_history.as_list(),
    )

    return response.text, results.pages, response.model_used


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    init_session()
    if not check_env():
        return
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
else:
    # Streamlit imports the file directly — main() must be called at module level
    main()
