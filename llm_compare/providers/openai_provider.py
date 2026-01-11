"""OpenAI provider implementation."""

import time
from typing import List, Optional

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from .base import LLMProvider, Response, ProviderStatus
from ..utils.logging import get_logger
from ..utils.retry import retry_with_backoff, RetryConfig

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    MODEL_PREFERENCES = [
        "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini",
        "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    ]
    MODEL_FALLBACK_PATTERN = "gpt-4"

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return "gpt-4o"

    def _create_client(self) -> None:
        try:
            self._client = OpenAI(api_key=self.api_key, timeout=self.timeout)
            self.status = ProviderStatus.HEALTHY
        except Exception:
            self.status = ProviderStatus.UNHEALTHY
            logger.error("Failed to create OpenAI client")
            raise

    def discover_models(self) -> List[str]:
        try:
            models = self._client.models.list()
            return [m.id for m in models.data if m.id.startswith("gpt-") and "instruct" not in m.id]
        except Exception:
            logger.warning("OpenAI model discovery failed")
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
