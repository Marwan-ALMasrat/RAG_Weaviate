"""
ingestion/embedder.py — Produce embeddings for page images and text chunks.

Two independent embedders:
  - ImageEmbedder  → colSmol-500M → multi-vector (late interaction)
  - TextEmbedder   → all-MiniLM-L6-v2 → single dense vector

Design note:
  Both embedders are lazy-loaded (first use triggers model load).
  Call `get_image_embedder()` / `get_text_embedder()` to get singletons —
  the models are loaded once and reused across the whole session.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Image Embedder  (colSmol-500M, multi-vector)
# ─────────────────────────────────────────────────────────────────────────────

class ImageEmbedder:
    """
    Embed page images using colSmol-500M.

    Returns a list-of-lists (multi-vector / late-interaction format)
    compatible with Weaviate's Configure.MultiVectors.
    """

    def __init__(self, model_name: str = config.COLSMOL_MODEL) -> None:
        from colpali_engine.models import ColIdefics3, ColIdefics3Processor

        logger.info("Loading colSmol model: %s  (CPU, float32)", model_name)
        logger.info("This takes ~45 seconds on first load…")

        self._model = ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=torch.float32,   # bfloat16 is slow on CPU
            device_map="cpu",
            cache_dir=config.HF_CACHE_DIR,
        ).eval()

        self._processor = ColIdefics3Processor.from_pretrained(
            model_name,
            cache_dir=config.HF_CACHE_DIR,
        )
        # CPU speed-up settings from the technical notes
        self._processor.image_processor.token_pooling = True
        self._processor.image_processor.size = {
            "longest_edge": config.COLSMOL_IMAGE_LONGEST_EDGE
        }

        logger.info("colSmol loaded successfully")

    def embed_image(self, image: Image.Image) -> list[list[float]]:
        """
        Embed a single PIL image.

        Returns
        -------
        list[list[float]]
            Multi-vector representation (shape: [n_tokens, dim]).
        """
        batch = self._processor.process_images([image]).to("cpu")
        with torch.no_grad():
            output = self._model(**batch)
        # output shape: [1, n_tokens, dim] → take first item → list-of-lists
        return output[0].to("cpu").tolist()

    def embed_query(self, query: str) -> list[list[float]]:
        """
        Embed a text query for image-space retrieval.

        Uses process_queries (not process_images) — same vector space.
        """
        batch = self._processor.process_queries([query]).to("cpu")
        with torch.no_grad():
            output = self._model(**batch)
        return output[0].to("cpu").tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Text Embedder  (all-MiniLM-L6-v2, single vector)
# ─────────────────────────────────────────────────────────────────────────────

class TextEmbedder:
    """
    Embed text chunks using all-MiniLM-L6-v2.

    Returns a flat list[float] (single dense vector, 384 dims).
    """

    def __init__(self, model_name: str = config.MINILM_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading text embedder: %s", model_name)
        self._model = SentenceTransformer(
            model_name,
            cache_folder=config.HF_CACHE_DIR,
        )
        logger.info("Text embedder loaded")

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        vec: np.ndarray = self._model.encode(
            text, convert_to_numpy=True, show_progress_bar=False
        )
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts — batched for speed."""
        vecs: np.ndarray = self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64
        )
        return vecs.tolist()

    def get_model(self):
        """Expose underlying SentenceTransformer (for SemanticChunker reuse)."""
        return self._model


# ─────────────────────────────────────────────────────────────────────────────
# Singletons — load once, reuse everywhere
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_image_embedder() -> ImageEmbedder:
    """Return the shared ImageEmbedder instance (loaded on first call)."""
    return ImageEmbedder()


@lru_cache(maxsize=1)
def get_text_embedder() -> TextEmbedder:
    """Return the shared TextEmbedder instance (loaded on first call)."""
    return TextEmbedder()
