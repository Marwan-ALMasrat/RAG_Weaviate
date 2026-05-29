"""
ingestion/pdf_processor.py — Extract page images and raw text from a PDF.

Responsibilities:
  - Render every page as a PIL Image (for colSmol embedding)
  - Extract raw text per page (for semantic chunking)
  - Return structured PageData objects

Design note:
  Nothing here knows about embeddings or Weaviate — pure I/O layer.
  Swap PyMuPDF for another library by only touching this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PageData:
    """All data extracted from a single PDF page."""

    page_number: int          # 1-indexed
    image: Image.Image        # PIL image of the rendered page
    text: str                 # raw text extracted by PyMuPDF
    source_pdf: str           # original PDF filename (not full path)
    document_title: str = ""  # filled in by caller if known


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

class PDFProcessor:
    """
    Extract page images and text from a PDF file.

    Parameters
    ----------
    dpi : int
        Rendering DPI for page images. 150 is good for colSmol; higher → slower.
    save_images : bool
        If True, saves rendered images to config.IMAGES_DIR/<pdf_stem>/.
    """

    def __init__(self, dpi: int = 150, save_images: bool = True) -> None:
        self.dpi = dpi
        self.save_images = save_images

    def process(self, pdf_path: str | Path, document_title: str = "") -> list[PageData]:
        """
        Process a PDF and return one PageData per page.

        Parameters
        ----------
        pdf_path : path to the PDF file
        document_title : human-readable title (defaults to filename stem)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_title = document_title or pdf_path.stem
        source_pdf = pdf_path.name

        # Create per-PDF image subfolder
        images_dir = config.IMAGES_DIR / pdf_path.stem
        if self.save_images:
            images_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Processing PDF: %s (%d pages)", source_pdf, self._page_count(pdf_path))

        pages: list[PageData] = []
        doc = fitz.open(str(pdf_path))

        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                page_number = page_index + 1  # 1-indexed

                # ── Render image ──────────────────────────────────────────
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                if self.save_images:
                    img_path = images_dir / f"page_{page_number:04d}.png"
                    img.save(img_path, "PNG")

                # ── Extract text ──────────────────────────────────────────
                text = page.get_text("text").strip()

                pages.append(
                    PageData(
                        page_number=page_number,
                        image=img,
                        text=text,
                        source_pdf=source_pdf,
                        document_title=doc_title,
                    )
                )

                logger.debug("  Page %d: %d chars extracted", page_number, len(text))
        finally:
            doc.close()

        logger.info("PDF processing complete: %d pages extracted", len(pages))
        return pages

    @staticmethod
    def _page_count(pdf_path: Path) -> int:
        doc = fitz.open(str(pdf_path))
        n = len(doc)
        doc.close()
        return n
