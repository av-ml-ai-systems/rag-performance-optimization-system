# Module: answer_generator
# Path: src/rag_performance/answer_generator.py
# Purpose: Generate natural-language answers using a local Ollama model.
# Description: Encapsulates LangChain's Ollama integration with model
# configuration capture, execution timing, TTFT measurement, and logging.

from __future__ import annotations

import time
from typing import Any

from langchain_ollama import ChatOllama

from rag_performance.logging import get_logger
from rag_performance.telemetry import ExecutionTimer, TokenUsage

logger = get_logger(__name__)


class AnswerGenerator:
    """Generate answers using a locally hosted language model."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        temperature: float = 0.0,
        stream: bool = False,
    ) -> None:
        """Initialize generator with Ollama model and configuration."""
        if not model.strip():
            logger.warning("AnswerGenerator initialization rejected: model is empty")
            raise ValueError("model must not be empty.")

        if temperature < 0.0:
            logger.warning(
                "AnswerGenerator initialization rejected: negative temperature"
            )
            raise ValueError("temperature must be non-negative.")

        self._model_name = model.strip()
        self._temperature = temperature
        self._stream = stream
        self._model = ChatOllama(
            model=self._model_name,
            temperature=self._temperature,
        )
        self._last_latency_ms: float = 0.0
        self._last_ttft_ms: float | None = None
        self._last_token_usage: TokenUsage = TokenUsage()

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model_name

    @property
    def temperature(self) -> float:
        """Return the configured generation temperature."""
        return self._temperature

    @property
    def stream(self) -> bool:
        """Return whether streaming generation is enabled."""
        return self._stream

    @property
    def last_latency_ms(self) -> float:
        """Return execution latency of the most recent generation in ms."""
        return self._last_latency_ms

    @property
    def last_ttft_ms(self) -> float | None:
        """Return Time to First Token in ms, or None if non-streaming."""
        return self._last_ttft_ms

    @property
    def last_token_usage(self) -> TokenUsage:
        """Return token usage extracted from the most recent generation."""
        return self._last_token_usage

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        """Extract token counts defensively from message metadata."""
        usage_meta = getattr(response, "usage_metadata", None)
        if isinstance(usage_meta, dict):
            inp = usage_meta.get("input_tokens")
            out = usage_meta.get("output_tokens")
            tot = usage_meta.get("total_tokens")
            if inp is not None or out is not None:
                return TokenUsage(
                    prompt_tokens=inp,
                    generation_tokens=out,
                    total_tokens=tot,
                )

        resp_meta = getattr(response, "response_metadata", None)
        if isinstance(resp_meta, dict):
            prompt_eval = resp_meta.get("prompt_eval_count")
            eval_count = resp_meta.get("eval_count")
            if prompt_eval is not None or eval_count is not None:
                total = None
                if prompt_eval is not None and eval_count is not None:
                    total = prompt_eval + eval_count
                return TokenUsage(
                    prompt_tokens=prompt_eval,
                    generation_tokens=eval_count,
                    total_tokens=total,
                )

        return TokenUsage()

    def generate(self, prompt: str) -> str:
        """Generate an answer from a non-empty prompt."""
        if not prompt.strip():
            logger.warning("Generation rejected: prompt must not be empty")
            raise ValueError("prompt must not be empty.")

        timer = ExecutionTimer()
        logger.info(
            "LLM generation started "
            "(model=%s, temp=%.2f, stream=%s, prompt_size_chars=%d)",
            self._model_name,
            self._temperature,
            self._stream,
            len(prompt),
        )

        if not self._stream:
            return self._generate_blocking(prompt, timer)
        return self._generate_streaming(prompt, timer)

    def _generate_blocking(self, prompt: str, timer: ExecutionTimer) -> str:
        """Generate response in standard blocking mode."""
        self._last_ttft_ms = None
        try:
            with timer:
                response = self._model.invoke(prompt)
        except Exception as exc:
            logger.error(
                "LLM generation failed: %s",
                type(exc).__name__,
                exc_info=True,
            )
            raise
        finally:
            self._last_latency_ms = timer.elapsed_ms

        self._last_token_usage = self._extract_token_usage(response)

        answer = response.content
        if not isinstance(answer, str):
            logger.error(
                "LLM returned unexpected content type: %s",
                type(answer).__name__,
            )
            raise TypeError("The language model returned non-text content.")

        resolved_answer = answer.strip()
        logger.info(
            "LLM generation completed successfully "
            "(latency_ms=%.2f, answer_size_chars=%d, "
            "prompt_tokens=%s, gen_tokens=%s)",
            self._last_latency_ms,
            len(resolved_answer),
            self._last_token_usage.prompt_tokens,
            self._last_token_usage.generation_tokens,
        )
        return resolved_answer

    def _generate_streaming(self, prompt: str, timer: ExecutionTimer) -> str:
        """Generate response via streaming and measure TTFT."""
        self._last_ttft_ms = None
        accumulated: list[str] = []
        final_chunk: Any = None

        try:
            with timer:
                stream_start = time.perf_counter()
                for chunk in self._model.stream(prompt):
                    final_chunk = chunk
                    content = chunk.content
                    if isinstance(content, str) and content:
                        if self._last_ttft_ms is None:
                            self._last_ttft_ms = (
                                time.perf_counter() - stream_start
                            ) * 1000.0
                        accumulated.append(content)
        except Exception as exc:
            logger.error(
                "LLM streaming generation failed: %s",
                type(exc).__name__,
                exc_info=True,
            )
            raise
        finally:
            self._last_latency_ms = timer.elapsed_ms

        self._last_token_usage = self._extract_token_usage(final_chunk)
        resolved_answer = "".join(accumulated).strip()

        logger.info(
            "LLM streaming completed successfully "
            "(latency_ms=%.2f, ttft_ms=%s, answer_size_chars=%d, "
            "prompt_tokens=%s, gen_tokens=%s)",
            self._last_latency_ms,
            f"{self._last_ttft_ms:.2f}" if self._last_ttft_ms is not None else "None",
            len(resolved_answer),
            self._last_token_usage.prompt_tokens,
            self._last_token_usage.generation_tokens,
        )
        return resolved_answer
