"""Anthropic Claude provider implementation."""

import time
from typing import List, Optional

from anthropic import Anthropic, APIError, RateLimitError, APITimeoutError

from .base import LLMProvider, Response, ProviderStatus
from ..utils.logging import get_logger
from ..utils.retry import retry_with_backoff, RetryConfig

logger = get_logger(__name__)


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider."""

    MODEL_PREFERENCES = [
        "claude-opus-4", "claude-sonnet-4", "claude-3-5-sonnet",
        "claude-3-5-haiku", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
    ]
    MODEL_FALLBACK_PATTERN = "claude-"

    @property
    def name(self) -> str:
        return "claude"

    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-20250514"

    def _create_client(self) -> None:
        try:
            self._client = Anthropic(api_key=self.api_key, timeout=self.timeout)
            self.status = ProviderStatus.HEALTHY
        except Exception:
            self.status = ProviderStatus.UNHEALTHY
            logger.error("Failed to create Anthropic client")
            raise

    def discover_models(self) -> List[str]:
        try:
            models = self._client.models.list()
            return [m.id for m in models.data]
        except Exception:
            logger.warning("Claude model discovery failed")
            return []

    @retry_with_backoff(RetryConfig(max_retries=3, retryable_exceptions=(RateLimitError, APITimeoutError, APIError)))
    def generate(self, prompt: str, **kwargs) -> Response:
        start_time = time.time()
        system_message = kwargs.get("system_message", "You are a helpful AI assistant.")

        try:
            response = self._client.messages.create(
                model=self.model_id,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                system=system_message,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
            )

            text = "".join(block.text for block in response.content if hasattr(block, "text"))
            usage = response.usage
            self.status = ProviderStatus.HEALTHY

            return Response(
                provider=self.name,
                model_id=response.model,
                text=text,
                latency_ms=int((time.time() - start_time) * 1000),
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
                finish_reason=response.stop_reason or "stop",
                raw_response=response,
            )

        except RateLimitError:
            self.status = ProviderStatus.DEGRADED
            raise
        except APITimeoutError:
            self.status = ProviderStatus.DEGRADED
            raise
        except APIError:
            self.status = ProviderStatus.UNHEALTHY
            raise
        except Exception:
            self.status = ProviderStatus.UNHEALTHY
            raise
