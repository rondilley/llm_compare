"""Anthropic Claude provider implementation."""

import time
from typing import List, Optional

from anthropic import Anthropic, APIError, RateLimitError, APITimeoutError

from .base import LLMProvider, Response, ProviderStatus
from ..utils.logging import get_logger
from ..utils.retry import retry_with_backoff, RetryConfig

logger = get_logger(__name__)


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider implementation."""

    # Model preferences in order of priority (best first)
    MODEL_PREFERENCES = [
        "claude-opus-4",        # Claude Opus 4 (best)
        "claude-sonnet-4",      # Claude Sonnet 4
        "claude-3-5-sonnet",    # Claude 3.5 Sonnet
        "claude-3-5-haiku",     # Claude 3.5 Haiku
        "claude-3-opus",        # Claude 3 Opus
        "claude-3-sonnet",      # Claude 3 Sonnet
        "claude-3-haiku",       # Claude 3 Haiku
    ]

    @property
    def name(self) -> str:
        return "claude"

    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-20250514"

    def _create_client(self) -> None:
        """Create the Anthropic client."""
        try:
            self._client = Anthropic(
                api_key=self.api_key,
                timeout=self.timeout,
            )
            self.status = ProviderStatus.HEALTHY
            logger.debug(f"Anthropic client created with model {self.model_id}")
        except Exception as e:
            self.status = ProviderStatus.UNHEALTHY
            logger.error(f"Failed to create Anthropic client: {e}")
            raise

    def discover_models(self) -> List[str]:
        """Discover available models from Anthropic API."""
        try:
            models = self._client.models.list()
            model_ids = [m.id for m in models.data]
            logger.debug(f"Claude discovered {len(model_ids)} models")
            return model_ids
        except Exception as e:
            logger.warning(f"Claude model discovery failed: {e}")
            return []

    def _rank_models(self, models: List[str]) -> Optional[str]:
        """Select the best Claude model based on preferences."""
        models_set = set(models)

        # Try each preferred model in order
        for preferred in self.MODEL_PREFERENCES:
            # Check for exact match
            if preferred in models_set:
                return preferred
            # Check for prefix match (e.g., "claude-sonnet-4" matches "claude-sonnet-4-20250514")
            for model in models:
                if model.startswith(preferred):
                    return model

        # If no preferred model found, return the first claude-3 or claude-4 variant
        for model in sorted(models, reverse=True):
            if "claude-" in model and ("opus" in model or "sonnet" in model):
                return model

        return None

    @retry_with_backoff(RetryConfig(
        max_retries=3,
        retryable_exceptions=(RateLimitError, APITimeoutError, APIError),
    ))
    def generate(self, prompt: str, **kwargs) -> Response:
        """
        Generate a response using Anthropic's API.

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
            response = self._client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                system=system_message,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract response data
            text = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        text += block.text

            finish_reason = response.stop_reason or "stop"

            # Token usage
            usage = response.usage
            input_tokens = usage.input_tokens if usage else 0
            output_tokens = usage.output_tokens if usage else 0

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
            logger.warning(f"Claude rate limit: {e}")
            raise
        except APITimeoutError as e:
            self.status = ProviderStatus.DEGRADED
            logger.warning(f"Claude timeout: {e}")
            raise
        except APIError as e:
            self.status = ProviderStatus.UNHEALTHY
            logger.error(f"Claude API error: {e}")
            raise
        except Exception as e:
            self.status = ProviderStatus.UNHEALTHY
            logger.error(f"Claude unexpected error: {e}")
            raise
