"""
db/ingestion_pipeline.py — Full ingestion orchestration.

Flow for a single PDF:
  1. PDFProcessor     → list[PageData]
  2. SemanticChunker  → list[TextChunk]
  3. ImageEmbedder    → multi-vector per page
  4. TextEmbedder     → single-vector per chunk
  5. Weaviate batch   → insert Pages + Chunks

Design note:
  IngestionPipeline is stateless — safe to call multiple times with
  different PDFs. Progress is reported via tqdm and logging.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
from weaviate.util import generate_uuid5

import config
from db.weaviate_client import ensure_collections, get_client
from ingestion.chunker import SemanticChunker, TextChunk
from ingestion.embedder import get_image_embedder, get_text_embedder
from ingestion.pdf_processor import PDFProcessor, PageData

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result summary
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IngestionResult:
    source_pdf: str
    pages_inserted: int
    chunks_inserted: int
    errors: list[str]

    @property
    def success(self) -> bool:
        return not self.errors


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class IngestionPipeline:
    """
    End-to-end PDF ingestion pipeline.

    Parameters
    ----------
    batch_size : int
        Weaviate batch size. 50 is safe for Weaviate Cloud free tier.
    dpi : int
        PDF rendering resolution. 150 balances quality vs speed on CPU.
    """

    def __init__(self, batch_size: int = 50, dpi: int = 150) -> None:
        self.batch_size = batch_size

        # Lazy — models load on first call
        self._pdf_processor = PDFProcessor(dpi=dpi, save_images=True)
        self._chunker: SemanticChunker | None = None   # init after text embedder loaded

    def ingest(self, pdf_path: str | Path, document_title: str = "") -> IngestionResult:
        """
        Ingest one PDF into Weaviate.

        Parameters
        ----------
        pdf_path : path to PDF
        document_title : optional human-readable title

        Returns
        -------
        IngestionResult with counts and any error messages
        """
        pdf_path = Path(pdf_path)
        errors: list[str] = []

        # ── Step 1: Extract pages ─────────────────────────────────────────
        logger.info("=== Ingestion started: %s ===", pdf_path.name)
        try:
            pages: list[PageData] = self._pdf_processor.process(pdf_path, document_title)
        except Exception as exc:
            logger.error("PDF extraction failed: %s", exc)
            return IngestionResult(pdf_path.name, 0, 0, [str(exc)])

        # ── Step 2: Semantic chunking ─────────────────────────────────────
        text_embedder = get_text_embedder()
        if self._chunker is None:
            # Reuse the already-loaded SentenceTransformer model
            self._chunker = SemanticChunker(embedder=text_embedder.get_model())

        chunks: list[TextChunk] = self._chunker.chunk_pages(pages)

        # ── Step 3: Connect + ensure collections ─────────────────────────
        client = get_client()
        ensure_collections(client)

        pages_inserted = 0
        chunks_inserted = 0

        # ── Step 4: Insert page images ────────────────────────────────────
        logger.info("Embedding and inserting %d pages…", len(pages))
        image_embedder = get_image_embedder()
        pages_collection = client.collections.use(config.WEAVIATE_PAGES_COLLECTION)

        with pages_collection.batch.fixed_size(batch_size=self.batch_size) as batch:
            for page in tqdm(pages, desc="Pages", unit="pg"):
                try:
                    embedding = image_embedder.embed_image(page.image)

                    # Encode image to base64 for BLOB storage
                    import io
                    buf = io.BytesIO()
                    page.image.save(buf, format="PNG")
                    b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

                    uid = generate_uuid5(f"{page.source_pdf}::page::{page.page_number}")
                    batch.add_object(
                        properties={
                            "document_title": page.document_title,
                            "page_image":     b64_image,
                            "filename":       f"{Path(page.source_pdf).stem}_page_{page.page_number:04d}.png",
                            "page_number":    page.page_number,
                            "source_pdf":     page.source_pdf,
                        },
                        uuid=uid,
                        vector={"default": embedding},
                    )
                    pages_inserted += 1

                except Exception as exc:
                    msg = f"Page {page.page_number} embedding failed: {exc}"
                    logger.error(msg)
                    errors.append(msg)

        # ── Step 5: Insert text chunks ────────────────────────────────────
        logger.info("Embedding and inserting %d chunks…", len(chunks))
        chunks_collection = client.collections.use(config.WEAVIATE_CHUNKS_COLLECTION)

        # Batch-embed all chunk texts at once (much faster than one-by-one)
        chunk_texts = [c.text for c in chunks]
        try:
            all_embeddings = text_embedder.embed_batch(chunk_texts)
        except Exception as exc:
            logger.error("Batch text embedding failed: %s", exc)
            errors.append(str(exc))
            all_embeddings = []

        if all_embeddings:
            with chunks_collection.batch.fixed_size(batch_size=self.batch_size) as batch:
                for chunk, embedding in tqdm(
                    zip(chunks, all_embeddings), total=len(chunks), desc="Chunks", unit="ck"
                ):
                    try:
                        uid = generate_uuid5(
                            f"{chunk.source_pdf}::chunk::{chunk.chunk_index}"
                        )
                        batch.add_object(
                            properties={
                                "text":           chunk.text,
                                "page_number":    chunk.page_number,
                                "source_pdf":     chunk.source_pdf,
                                "chunk_index":    chunk.chunk_index,
                                "document_title": chunk.document_title,
                            },
                            uuid=uid,
                            vector={"default": embedding},
                        )
                        chunks_inserted += 1
                    except Exception as exc:
                        msg = f"Chunk {chunk.chunk_index} insert failed: {exc}"
                        logger.error(msg)
                        errors.append(msg)

        result = IngestionResult(
            source_pdf=pdf_path.name,
            pages_inserted=pages_inserted,
            chunks_inserted=chunks_inserted,
            errors=errors,
        )
        logger.info(
            "=== Ingestion complete: %d pages, %d chunks, %d errors ===",
            pages_inserted,
            chunks_inserted,
            len(errors),
        )
        return result
