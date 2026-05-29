"""
generation/gemini_client.py — Google Gemini 2.0 Flash multimodal generation.

Sends: query + retrieved text chunks + retrieved page images → answer.
Images are passed as base64 inline (no upload step required).

Raises
------
RateLimitError  : when the free tier limit (15 req/min or 1500/day) is hit
GenerationError : for all other API errors
"""

from __future__ import annotations

import base64
import logging

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, TooManyRequests

import config
from retrieval.searcher import SearchResults

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────

class RateLimitError(Exception):
    """Raised when Gemini free-tier rate limit is exceeded."""


class GenerationError(Exception):
    """Raised for all other Gemini API errors."""


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────

class GeminiClient:
    """
    Generate answers using Gemini 2.0 Flash.

    Parameters
    ----------
    model_name : str  (default: config.GEMINI_MODEL)
    max_tokens : int
    temperature : float
    """

    def __init__(
        self,
        model_name: str = config.GEMINI_MODEL,
        max_tokens: int = config.GENERATION_MAX_TOKENS,
        temperature: float = config.GENERATION_TEMPERATURE,
    ) -> None:
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env")

        genai.configure(api_key=config.GEMINI_API_KEY)
        self._model = genai.GenerativeModel(model_name)
        self._gen_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        logger.info("GeminiClient ready: %s", model_name)

    def generate(
        self,
        query: str,
        results: SearchResults,
        chat_history: list[dict] | None = None,
    ) -> str:
        """
        Generate an answer from retrieved context.

        Parameters
        ----------
        query        : user's question
        results      : SearchResults from HybridSearcher
        chat_history : list of {"role": "user"|"assistant", "content": str}

        Returns
        -------
        str  : model's answer text

        Raises
        ------
        RateLimitError   : rate limit hit → caller should switch to Groq
        GenerationError  : other API error
        """
        prompt_parts = self._build_prompt(query, results, chat_history)

        try:
            response = self._model.generate_content(
                prompt_parts,
                generation_config=self._gen_config,
            )
            return response.text

        except (ResourceExhausted, TooManyRequests) as exc:
            logger.warning("Gemini rate limit hit: %s", exc)
            raise RateLimitError(str(exc)) from exc

        except Exception as exc:
            logger.error("Gemini API error: %s", exc)
            raise GenerationError(str(exc)) from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        query: str,
        results: SearchResults,
        chat_history: list[dict] | None,
    ) -> list:
        """
        Build the multimodal prompt list for Gemini.

        Structure:
          [system instruction] + [history] + [text context] + [images] + [question]
        """
        parts: list = []

        # System instruction
        parts.append(
            "You are a scientific research assistant. "
            "Answer questions accurately based on the provided document pages and text. "
            "Always cite page numbers when referencing specific content. "
            "If the answer is not in the provided context, say so clearly."
        )

        # Chat history (for multi-turn)
        if chat_history:
            history_text = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in chat_history[-6:]  # last 3 turns
            )
            parts.append(f"\n--- Conversation history ---\n{history_text}\n")

        # Retrieved text chunks
        if results.chunks:
            parts.append(f"\n--- Retrieved text context ---\n{results.context_text}\n")

        # Retrieved page images (inline base64)
        if results.pages:
            parts.append("\n--- Relevant document pages ---\n")
            for page in results.pages:
                if page.image_b64:
                    parts.append(
                        {
                            "mime_type": "image/png",
                            "data": base64.b64decode(page.image_b64),
                        }
                    )
                    parts.append(
                        f"[Page {page.page_number} from {page.source_pdf}]\n"
                    )

        # The actual question
        parts.append(f"\nQuestion: {query}\n\nAnswer:")

        return parts
