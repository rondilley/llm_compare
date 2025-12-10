"""OpenAI provider implementation."""

import time
from typing import List, Optional

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from .base import LLMProvider, Response, ProviderStatus
from ..utils.logging import get_logger
from ..utils.retry import retry_with_backoff, RetryConfig

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider implementation."""

    # Model preferences in order of priority (best first)
    MODEL_PREFERENCES = [
        "gpt-4.1",          # Latest GPT-4.1
        "gpt-4.1-mini",     # GPT-4.1 mini
        "gpt-4o",           # GPT-4o
        "gpt-4o-mini",      # GPT-4o mini
        "gpt-4-turbo",      # GPT-4 Turbo
        "gpt-4",            # Base GPT-4
        "gpt-3.5-turbo",    # Fallback
    ]

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return "gpt-4o"

    def _create_client(self) -> None:
        """Create the OpenAI client."""
        try:
            self._client = OpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
            )
            self.status = ProviderStatus.HEALTHY
            logger.debug(f"OpenAI client created with model {self.model_id}")
        except Exception as e:
            self.status = ProviderStatus.UNHEALTHY
            logger.error(f"Failed to create OpenAI client: {e}")
            raise

    def discover_models(self) -> List[str]:
        """Discover available models from OpenAI API."""
        try:
            models = self._client.models.list()
            # Filter for chat models (gpt-*)
            chat_models = [
                m.id for m in models.data
                if m.id.startswith("gpt-") and "instruct" not in m.id
            ]
            logger.debug(f"OpenAI discovered {len(chat_models)} chat models")
            return chat_models
        except Exception as e:
            logger.warning(f"OpenAI model discovery failed: {e}")
            return []

    def _rank_models(self, models: List[str]) -> Optional[str]:
        """Select the best OpenAI model based on preferences."""
        models_set = set(models)

        # Try each preferred model in order
        for preferred in self.MODEL_PREFERENCES:
            # Check for exact match
            if preferred in models_set:
                return preferred
            # Check for prefix match (e.g., "gpt-4o" matches "gpt-4o-2024-...")
            for model in models:
                if model.startswith(preferred):
                    return model

        # If no preferred model found, return the first gpt-4 variant
        for model in sorted(models, reverse=True):
            if "gpt-4" in model:
                return model

        return None

    @retry_with_backoff(RetryConfig(
        max_retries=3,
        retryable_exceptions=(RateLimitError, APITimeoutError, APIError),
    ))
    def generate(self, prompt: str, **kwargs) -> Response:
        """
        Generate a response using OpenAI's API.

        Args:
            prompt: The prompt to send
            **kwargs: Additional arguments (system_message, temperature, etc.)

        Returns:
            Response object
        """
        start_time = time.time()

        system_message = kwargs.get("system_message", "You are a helpful AI assistant.")
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        try:
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract response data
            choice = response.choices[0]
            text = choice.message.content or ""
            finish_reason = choice.finish_reason or "stop"

            # Token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            self.status = ProviderStatus.HEALTHY

            return Response(
                provider=self.name,
                model_id=response.model,
                text=text,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                raw_response=response,
            )

        except RateLimitError as e:
            self.status = ProviderStatus.DEGRADED
            logger.warning(f"OpenAI rate limit: {e}")
            raise
        except APITimeoutError as e:
            self.status = ProviderStatus.DEGRADED
            logger.warning(f"OpenAI timeout: {e}")
            raise
        except APIError as e:
            self.status = ProviderStatus.UNHEALTHY
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            self.status = ProviderStatus.UNHEALTHY
            logger.error(f"OpenAI unexpected error: {e}")
            raise
