"""
generation/openrouter_client.py — OpenRouter Gemma 4 31B fallback.

Activated automatically when Gemini hits its rate limit.
Supports vision via base64 image URLs.

Raises
------
RateLimitError  : when OpenRouter rate limit is hit
GenerationError : for all other API errors
"""

from __future__ import annotations

import logging
from openai import OpenAI, RateLimitError as OpenAIRateLimit

import config
from retrieval.searcher import SearchResults

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when OpenRouter rate limit is hit."""


class GenerationError(Exception):
    """Raised for all other OpenRouter API errors."""


class OpenRouterClient:
    """
    Generate answers using OpenRouter — Gemma 4 31B.

    Parameters
    ----------
    model_name  : str
    max_tokens  : int
    temperature : float
    """

    def __init__(
        self,
        model_name: str = config.OPENROUTER_MODEL,
        max_tokens: int = config.GENERATION_MAX_TOKENS,
        temperature: float = config.GENERATION_TEMPERATURE,
    ) -> None:
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set in .env")

        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY,
        )
        self._model = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        logger.info("OpenRouterClient ready: %s", model_name)

    def generate(
        self,
        query: str,
        results: SearchResults,
        chat_history: list[dict] | None = None,
    ) -> str:
        messages = self._build_messages(query, results, chat_history)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            return response.choices[0].message.content

        except OpenAIRateLimit as exc:
            logger.error("OpenRouter rate limit hit: %s", exc)
            raise RateLimitError(str(exc)) from exc

        except Exception as exc:
            logger.error("OpenRouter API error: %s", exc)
            raise GenerationError(str(exc)) from exc

    def _build_messages(
        self,
        query: str,
        results: SearchResults,
        chat_history: list[dict] | None,
    ) -> list[dict]:
        system_msg = {
            "role": "system",
            "content": (
                "You are a scientific research assistant. "
                "Answer questions accurately based on the provided document pages and text. "
                "Always cite page numbers when referencing specific content. "
                "If the answer is not in the provided context, say so clearly."
            ),
        }

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

        # Chat history
        if chat_history:
            history_text = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in chat_history[-6:]
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