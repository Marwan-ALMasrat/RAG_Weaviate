"""
ingestion/chunker.py — Semantic chunking of extracted page text.

Strategy:
  1. Split text into sentences.
  2. Embed each sentence with all-MiniLM-L6-v2 (same model used for chunk embeddings).
  3. Compute cosine similarity between adjacent sentences.
  4. When similarity drops below threshold → start a new chunk.

Design note:
  The SentenceTransformer model is injected via `get_text_embedder()` from
  embedder.py so we never load it twice. But this module can also stand alone
  (it will load the model internally if no embedder is provided).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

import config
from ingestion.pdf_processor import PageData

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A semantically coherent chunk of text from one page."""

    text: str
    page_number: int
    chunk_index: int          # 0-indexed within the whole document
    source_pdf: str
    document_title: str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Naive but fast sentence splitter for scientific text."""
    # Split on '. ', '? ', '! ' but keep abbreviations like 'Fig. 1' intact-ish
    parts = re.split(r"(?<=[.?!])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Chunker
# ─────────────────────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Split pages into semantically coherent text chunks.

    Parameters
    ----------
    model_name : str
        SentenceTransformer model name. Defaults to config.MINILM_MODEL.
    threshold : float
        Cosine similarity below this value → new chunk boundary.
    min_sentences : int
        Don't break before accumulating at least this many sentences.
    max_sentences : int
        Force a break after this many sentences regardless of similarity.
    embedder : optional pre-loaded SentenceTransformer
        Pass the already-loaded model to avoid loading it twice.
    """

    def __init__(
        self,
        model_name: str = config.MINILM_MODEL,
        threshold: float = config.CHUNK_SIMILARITY_THRESHOLD,
        min_sentences: int = config.CHUNK_MIN_SENTENCES,
        max_sentences: int = config.CHUNK_MAX_SENTENCES,
        embedder=None,
    ) -> None:
        self.threshold = threshold
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

        if embedder is not None:
            self._model = embedder
        else:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence transformer: %s", model_name)
            self._model = SentenceTransformer(
                model_name,
                cache_folder=config.HF_CACHE_DIR,
            )

    def chunk_pages(self, pages: list[PageData]) -> list[TextChunk]:
        """
        Convert a list of PageData into semantically coherent TextChunks.

        Chunks do NOT cross page boundaries — each page is chunked independently.
        chunk_index is global across the whole document.
        """
        all_chunks: list[TextChunk] = []
        global_index = 0

        for page in pages:
            if not page.text:
                continue
            page_chunks = self._chunk_text(
                text=page.text,
                page_number=page.page_number,
                source_pdf=page.source_pdf,
                document_title=page.document_title,
                start_index=global_index,
            )
            all_chunks.extend(page_chunks)
            global_index += len(page_chunks)

        logger.info("Chunking complete: %d chunks from %d pages", len(all_chunks), len(pages))
        return all_chunks

    # ── Internal ──────────────────────────────────────────────────────────────

    def _chunk_text(
        self,
        text: str,
        page_number: int,
        source_pdf: str,
        document_title: str,
        start_index: int,
    ) -> list[TextChunk]:

        sentences = _split_sentences(text)
        if not sentences:
            return []

        # Short pages → single chunk
        if len(sentences) <= self.min_sentences:
            return [
                TextChunk(
                    text=text,
                    page_number=page_number,
                    chunk_index=start_index,
                    source_pdf=source_pdf,
                    document_title=document_title,
                )
            ]

        # Embed all sentences at once (batch = fast)
        embeddings: np.ndarray = self._model.encode(
            sentences, convert_to_numpy=True, show_progress_bar=False
        )

        # Group sentences into chunks based on similarity drops
        chunks: list[TextChunk] = []
        current_sentences: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = _cosine(embeddings[i - 1], embeddings[i])
            force_break = len(current_sentences) >= self.max_sentences
            natural_break = (
                sim < self.threshold and len(current_sentences) >= self.min_sentences
            )

            if force_break or natural_break:
                chunks.append(
                    TextChunk(
                        text=" ".join(current_sentences),
                        page_number=page_number,
                        chunk_index=start_index + len(chunks),
                        source_pdf=source_pdf,
                        document_title=document_title,
                    )
                )
                current_sentences = [sentences[i]]
            else:
                current_sentences.append(sentences[i])

        # Flush remaining sentences
        if current_sentences:
            chunks.append(
                TextChunk(
                    text=" ".join(current_sentences),
                    page_number=page_number,
                    chunk_index=start_index + len(chunks),
                    source_pdf=source_pdf,
                    document_title=document_title,
                )
            )

        return chunks
