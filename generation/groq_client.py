"""
generation/groq_client.py — Groq / Llama 3.2 11B Vision fallback.

Activated automatically when Gemini hits its rate limit.
Supports vision via base64 image URLs in the messages array.

Raises
------
GenerationError : for all Groq API errors
"""

from __future__ import annotations

import base64
import logging

from groq import Groq, RateLimitError as GroqRateLimitError

import config
from retrieval.searcher import SearchResults

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Raised for Groq API errors."""


class GroqClient:
    """
    Generate answers using Groq — Llama 3.2 11B Vision.

    Parameters
    ----------
    model_name  : str
    max_tokens  : int
    temperature : float
    """

    def __init__(
        self,
        model_name: str = config.GROQ_MODEL,
        max_tokens: int = config.GENERATION_MAX_TOKENS,
        temperature: float = config.GENERATION_TEMPERATURE,
    ) -> None:
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")

        self._client = Groq(api_key=config.GROQ_API_KEY)
        self._model = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        logger.info("GroqClient ready: %s", model_name)

    def generate(
        self,
        query: str,
        results: SearchResults,
        chat_history: list[dict] | None = None,
    ) -> str:
        """
        Generate an answer from retrieved context using Groq.

        Parameters
        ----------
        query        : user's question
        results      : SearchResults from HybridSearcher
        chat_history : list of {"role": "user"|"assistant", "content": str}

        Returns
        -------
        str : model's answer text
        """
        messages = self._build_messages(query, results, chat_history)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            return response.choices[0].message.content

        except GroqRateLimitError as exc:
            logger.error("Groq rate limit hit: %s", exc)
            raise GenerationError(f"Both Gemini and Groq rate limits hit: {exc}") from exc

        except Exception as exc:
            logger.error("Groq API error: %s", exc)
            raise GenerationError(str(exc)) from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_messages(
        self,
        query: str,
        results: SearchResults,
        chat_history: list[dict] | None,
    ) -> list[dict]:
        """
        Build the messages array for the Groq chat completion API.
        Images are embedded as base64 data URIs in the content array.
        """
        system_msg = {
            "role": "system",
            "content": (
                "You are a scientific research assistant. "
                "Answer questions accurately based on the provided document pages and text. "
                "Always cite page numbers when referencing specific content. "
                "If the answer is not in the provided context, say so clearly."
            ),
        }

        # Build user content — mix of text and image parts
        content: list[dict] = []

        # Text context
        if results.chunks:
            content.append({
                "type": "text",
                "text": f"--- Retrieved text context ---\n{results.context_text}\n",
            })

        # Page images (base64 data URIs)
        for page in results.pages:
            if page.image_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{page.image_b64}"
                    },
                })
                content.append({
                    "type": "text",
                    "text": f"[Page {page.page_number} from {page.source_pdf}]",
                })

        # History summary (Groq has smaller context window)
        history_text = ""
        if chat_history:
            history_text = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in chat_history[-4:]  # last 2 turns
            )
            content.append({
                "type": "text",
                "text": f"\n--- Previous conversation ---\n{history_text}\n",
            })

        # Final question
        content.append({"type": "text", "text": f"Question: {query}\n\nAnswer:"})

        messages = [system_msg]
        if content:
            messages.append({"role": "user", "content": content})

        return messages
