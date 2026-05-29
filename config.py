"""
config.py — Central settings for VisionRAG.

All tuneable knobs live here. Change a value once → takes effect everywhere.
Loaded from environment variables (via .env) with safe defaults.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env before anything reads os.environ ────────────────────────────────
load_dotenv(override=False)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY: str = os.getenv("WEAVIATE_API_KEY", "")

# ─────────────────────────────────────────────────────────────────────────────
# Model Names
# ─────────────────────────────────────────────────────────────────────────────

COLSMOL_MODEL: str = os.getenv("COLSMOL_MODEL", "vidore/colSmol-500M")
MINILM_MODEL: str = os.getenv("MINILM_MODEL", "all-MiniLM-L6-v2")
HF_CACHE_DIR: str = os.getenv("HF_CACHE_DIR", "./hf_model_cache")

# ─────────────────────────────────────────────────────────────────────────────
# CPU Optimisation Settings for colSmol
# ─────────────────────────────────────────────────────────────────────────────

COLSMOL_TOKEN_POOLING: bool = True
COLSMOL_IMAGE_LONGEST_EDGE: int = 224

# ─────────────────────────────────────────────────────────────────────────────
# Weaviate Collection Names
# ─────────────────────────────────────────────────────────────────────────────

WEAVIATE_PAGES_COLLECTION: str = "Pages"
WEAVIATE_CHUNKS_COLLECTION: str = "Chunks"

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

HYBRID_ALPHA: float = 0.75
RETRIEVAL_TOP_K_PAGES: int = 3
RETRIEVAL_TOP_K_CHUNKS: int = 5

# ─────────────────────────────────────────────────────────────────────────────
# Semantic Chunking
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SIMILARITY_THRESHOLD: float = 0.4
CHUNK_MIN_SENTENCES: int = 2
CHUNK_MAX_SENTENCES: int = 20

# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GENERATION_MAX_TOKENS: int = 1024
GENERATION_TEMPERATURE: float = 0.2
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR: Path = Path(__file__).parent
DATA_DIR: Path = BASE_DIR / "data"
UPLOADS_DIR: Path = DATA_DIR / "uploads"
IMAGES_DIR: Path = DATA_DIR / "images"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Validation helper
# ─────────────────────────────────────────────────────────────────────────────

def validate_required_keys() -> list[str]:
    """Return a list of missing required environment variables."""
    required = {
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "GROQ_API_KEY": GROQ_API_KEY,
        "WEAVIATE_URL": WEAVIATE_URL,
        "WEAVIATE_API_KEY": WEAVIATE_API_KEY,
    }    
    return [k for k, v in required.items() if not v]