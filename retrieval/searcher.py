"""
retrieval/searcher.py — Hybrid search over Pages and Chunks collections.

For each query:
  - Embed query with colSmol  → search Pages  (image-space)
  - Embed query with MiniLM   → search Chunks (text-space)
  - Both use Hybrid (BM25 + vector) with alpha from config

Design note:
  HybridSearcher is stateless and thread-safe.
  Returns typed result objects — no raw Weaviate dicts leak out.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image

import config
from db.weaviate_client import get_client
from ingestion.embedder import get_image_embedder, get_text_embedder

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PageResult:
    page_number: int
    source_pdf: str
    document_title: str
    image_b64: str
    filename: str

    @property
    def image(self) -> Image.Image:
        """Decode base64 to PIL Image on demand."""
        return Image.open(BytesIO(base64.b64decode(self.image_b64)))


@dataclass
class ChunkResult:
    text: str
    page_number: int
    source_pdf: str
    document_title: str
    chunk_index: int


@dataclass
class SearchResults:
    query: str
    pages: list[PageResult] = field(default_factory=list)
    chunks: list[ChunkResult] = field(default_factory=list)

    @property
    def context_text(self) -> str:
        """Concatenated chunk texts — ready to paste into the LLM prompt."""
        parts = [
            f"[Page {c.page_number}, {c.source_pdf}]\n{c.text}"
            for c in self.chunks
        ]
        return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Searcher
# ─────────────────────────────────────────────────────────────────────────────

class HybridSearcher:
    """
    Run hybrid search (BM25 + vector) over Pages and Chunks.

    Parameters
    ----------
    top_k_pages  : number of page images to retrieve
    top_k_chunks : number of text chunks to retrieve
    alpha        : hybrid weight — 0 = pure BM25, 1 = pure vector
    """

    def __init__(
        self,
        top_k_pages: int = config.RETRIEVAL_TOP_K_PAGES,
        top_k_chunks: int = config.RETRIEVAL_TOP_K_CHUNKS,
        alpha: float = config.HYBRID_ALPHA,
    ) -> None:
        self.top_k_pages = top_k_pages
        self.top_k_chunks = top_k_chunks
        self.alpha = alpha

    def search(self, query: str, source_pdf: str | None = None) -> SearchResults:
        """
        Run hybrid search for a user query.

        Parameters
        ----------
        query      : natural language question
        source_pdf : if set, restrict results to this PDF filename

        Returns
        -------
        SearchResults with page images and text chunks
        """
        logger.info("Searching for: %r", query)
        client = get_client()

        pages = self._search_pages(client, query, source_pdf)
        chunks = self._search_chunks(client, query, source_pdf)

        logger.info("Retrieved %d pages, %d chunks", len(pages), len(chunks))
        return SearchResults(query=query, pages=pages, chunks=chunks)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _search_pages(self, client, query: str, source_pdf: str | None) -> list[PageResult]:
        image_embedder = get_image_embedder()
        query_vec = image_embedder.embed_query(query)

        pages_col = client.collections.use(config.WEAVIATE_PAGES_COLLECTION)

        filters = None
        if source_pdf:
            from weaviate.classes.query import Filter
            filters = Filter.by_property("source_pdf").equal(source_pdf)

        try:
            response = pages_col.query.hybrid(
                query=query,
                vector=query_vec,
                target_vector="default",
                alpha=self.alpha,
                limit=self.top_k_pages,
                filters=filters,
                return_properties=[
                    "page_number", "source_pdf", "document_title",
                    "page_image", "filename",
                ],
            )
        except Exception as exc:
            logger.error("Pages search failed: %s", exc)
            return []

        results = []
        for obj in response.objects:
            p = obj.properties
            results.append(
                PageResult(
                    page_number=p.get("page_number", 0),
                    source_pdf=p.get("source_pdf", ""),
                    document_title=p.get("document_title", ""),
                    image_b64=p.get("page_image", ""),
                    filename=p.get("filename", ""),
                )
            )
        return results

    def _search_chunks(self, client, query: str, source_pdf: str | None) -> list[ChunkResult]:
        text_embedder = get_text_embedder()
        query_vec = text_embedder.embed_text(query)

        chunks_col = client.collections.use(config.WEAVIATE_CHUNKS_COLLECTION)

        filters = None
        if source_pdf:
            from weaviate.classes.query import Filter
            filters = Filter.by_property("source_pdf").equal(source_pdf)

        try:
            response = chunks_col.query.hybrid(
                query=query,
                vector=query_vec,
                target_vector="default",
                alpha=self.alpha,
                limit=self.top_k_chunks,
                filters=filters,
                return_properties=[
                    "text", "page_number", "source_pdf",
                    "document_title", "chunk_index",
                ],
            )
        except Exception as exc:
            logger.error("Chunks search failed: %s", exc)
            return []

        results = []
        for obj in response.objects:
            p = obj.properties
            results.append(
                ChunkResult(
                    text=p.get("text", ""),
                    page_number=p.get("page_number", 0),
                    source_pdf=p.get("source_pdf", ""),
                    document_title=p.get("document_title", ""),
                    chunk_index=p.get("chunk_index", 0),
                )
            )
        return results