"""
generation/generator.py — Unified generator with automatic fallback chain.

Fallback order:
  1. Gemini 2.0 Flash
  2. OpenRouter — Gemma 4 31B (free)
  3. Groq — Llama 4 Scout
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import config
from generation.gemini_client import GeminiClient, RateLimitError as GeminiRateLimit
from generation.openrouter_client import OpenRouterClient, RateLimitError as OpenRouterRateLimit
from generation.groq_client import GroqClient, GenerationError
from retrieval.searcher import SearchResults

logger = logging.getLogger(__name__)


@dataclass
class GenerationResponse:
    text: str
    model_used: str          # "gemini", "openrouter", or "groq"
    fallback_triggered: bool


class Generator:
    """
    Multimodal answer generator with automatic fallback chain.

    Primary    : Gemini 2.0 Flash
    Fallback 1 : OpenRouter — Gemma 4 31B (free)
    Fallback 2 : Groq — Llama 4 Scout
    """

    def __init__(self, force_groq: bool = False) -> None:
        self._force_groq = force_groq
        self._gemini: GeminiClient | None = None
        self._openrouter: OpenRouterClient | None = None
        self._groq: GroqClient | None = None

    def generate(
        self,
        query: str,
        results: SearchResults,
        chat_history: list[dict] | None = None,
    ) -> GenerationResponse:

        if not self._force_groq:
            # 1. Try Gemini
            try:
                gemini = self._get_gemini()
                text = gemini.generate(query, results, chat_history)
                return GenerationResponse(
                    text=text, model_used="gemini", fallback_triggered=False
                )
            except GeminiRateLimit:
                logger.warning("Gemini rate limit — trying OpenRouter")
            except Exception as exc:
                logger.error("Gemini error: %s — trying OpenRouter", exc)

            # 2. Try OpenRouter
            try:
                openrouter = self._get_openrouter()
                text = openrouter.generate(query, results, chat_history)
                return GenerationResponse(
                    text=text, model_used="openrouter", fallback_triggered=True
                )
            except OpenRouterRateLimit:
                logger.warning("OpenRouter rate limit — trying Groq")
            except Exception as exc:
                logger.error("OpenRouter error: %s — trying Groq", exc)

        # 3. Fallback: Groq
        groq = self._get_groq()
        text = groq.generate(query, results, chat_history)
        return GenerationResponse(
            text=text,
            model_used="groq",
            fallback_triggered=True,
        )

    def _get_gemini(self) -> GeminiClient:
        if self._gemini is None:
            self._gemini = GeminiClient()
        return self._gemini

    def _get_openrouter(self) -> OpenRouterClient:
        if self._openrouter is None:
            self._openrouter = OpenRouterClient()
        return self._openrouter

    def _get_groq(self) -> GroqClient:
        if self._groq is None:
            self._groq = GroqClient()
        return self._groq


@lru_cache(maxsize=1)
def get_generator() -> Generator:
    """Return the shared Generator instance."""
    return Generator()