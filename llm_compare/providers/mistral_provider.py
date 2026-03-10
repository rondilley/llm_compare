"""Mistral AI provider implementation."""

import time
from typing import List, Optional

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from .base import LLMProvider, Response, ProviderStatus
from ..utils.logging import get_logger
from ..utils.retry import retry_with_backoff, RetryConfig

logger = get_logger(__name__)


class MistralProvider(LLMProvider):
    """Mistral AI provider (OpenAI-compatible API)."""

    MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
    MODEL_PREFERENCES = [
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
        "open-mistral-nemo",
        "open-mixtral-8x22b",
        "open-mixtral-8x7b",
    ]
    MODEL_FALLBACK_PATTERN = "mistral"

    @property
    def name(self) -> str:
        return "mistral"

    @property
    def default_model(self) -> str:
        return "mistral-large-latest"

    def _create_client(self) -> None:
        try:
            self._client = OpenAI(api_key=self.api_key, base_url=self.MISTRAL_BASE_URL, timeout=self.timeout)
            self.status = ProviderStatus.HEALTHY
        except Exception:
            self.status = ProviderStatus.UNHEALTHY
            logger.error("Failed to create Mistral client")
            raise

    def discover_models(self) -> List[str]:
        try:
            models = self._client.models.list()
            return [m.id for m in models.data if "mistral" in m.id.lower() or "mixtral" in m.id.lower()]
        except Exception:
            logger.warning("Mistral model discovery failed")
            return []

    @retry_with_backoff(RetryConfig(max_retries=3, retryable_exceptions=(RateLimitError, APITimeoutError, APIError)))
    def _generate_impl(self, prompt: str, **kwargs) -> Response:
        start_time = time.time()
        system_message = kwargs.get("system_message", "You are a helpful AI assistant.")

        try:
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )

            choice = response.choices[0]
            usage = response.usage
            self.status = ProviderStatus.HEALTHY

            return Response(
                provider=self.name,
                model_id=response.model,
                text=choice.message.content or "",
                latency_ms=int((time.time() - start_time) * 1000),
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                finish_reason=choice.finish_reason or "stop",
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
