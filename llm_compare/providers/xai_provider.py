"""xAI Grok provider implementation."""

import time
from typing import List, Optional

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from .base import LLMProvider, Response, ProviderStatus
from ..utils.logging import get_logger
from ..utils.retry import retry_with_backoff, RetryConfig

logger = get_logger(__name__)


class XAIProvider(LLMProvider):
    """xAI Grok provider implementation using OpenAI-compatible API."""

    XAI_BASE_URL = "https://api.x.ai/v1"

    # Model preferences in order of priority (best first)
    MODEL_PREFERENCES = [
        "grok-3",           # Grok 3 (latest/best)
        "grok-3-fast",      # Grok 3 Fast
        "grok-2",           # Grok 2
        "grok-2-mini",      # Grok 2 Mini
    ]

    @property
    def name(self) -> str:
        return "xai"

    @property
    def default_model(self) -> str:
        return "grok-3"

    def _create_client(self) -> None:
        """Create the xAI client using OpenAI SDK with custom base URL."""
        try:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.XAI_BASE_URL,
                timeout=self.timeout,
            )
            self.status = ProviderStatus.HEALTHY
            logger.debug(f"xAI client created with model {self.model_id}")
        except Exception as e:
            self.status = ProviderStatus.UNHEALTHY
            logger.error(f"Failed to create xAI client: {e}")
            raise

    def discover_models(self) -> List[str]:
        """Discover available models from xAI API."""
        try:
            models = self._client.models.list()
            # Filter for grok models
            grok_models = [
                m.id for m in models.data
                if "grok" in m.id.lower()
            ]
            logger.debug(f"xAI discovered {len(grok_models)} Grok models")
            return grok_models
        except Exception as e:
            logger.warning(f"xAI model discovery failed: {e}")
            return []

    def _rank_models(self, models: List[str]) -> Optional[str]:
        """Select the best xAI model based on preferences."""
        models_set = set(models)

        # Try each preferred model in order
        for preferred in self.MODEL_PREFERENCES:
            # Check for exact match
            if preferred in models_set:
                return preferred
            # Check for prefix match
            for model in models:
                if model.startswith(preferred):
                    return model

        # If no preferred model found, return the first grok variant
        for model in sorted(models, reverse=True):
            if "grok" in model:
                return model

        return None

    @retry_with_backoff(RetryConfig(
        max_retries=3,
        retryable_exceptions=(RateLimitError, APITimeoutError, APIError),
    ))
    def generate(self, prompt: str, **kwargs) -> Response:
        """
        Generate a response using xAI's API.

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
            logger.warning(f"xAI rate limit: {e}")
            raise
        except APITimeoutError as e:
            self.status = ProviderStatus.DEGRADED
            logger.warning(f"xAI timeout: {e}")
            raise
        except APIError as e:
            self.status = ProviderStatus.UNHEALTHY
            logger.error(f"xAI API error: {e}")
            raise
        except Exception as e:
            self.status = ProviderStatus.UNHEALTHY
            logger.error(f"xAI unexpected error: {e}")
            raise
