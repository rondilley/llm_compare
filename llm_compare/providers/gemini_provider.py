"""Google Gemini provider implementation."""

import time
from typing import List, Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from google.api_core import exceptions as google_exceptions

from .base import LLMProvider, Response, ProviderStatus
from ..utils.logging import get_logger
from ..utils.retry import retry_with_backoff, RetryConfig

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation."""

    # Model preferences in order of priority (best first)
    MODEL_PREFERENCES = [
        "gemini-2.5-pro",       # Gemini 2.5 Pro (best)
        "gemini-2.5-flash",     # Gemini 2.5 Flash
        "gemini-2.0-pro",       # Gemini 2.0 Pro
        "gemini-2.0-flash",     # Gemini 2.0 Flash
        "gemini-1.5-pro",       # Gemini 1.5 Pro
        "gemini-1.5-flash",     # Gemini 1.5 Flash
        "gemini-pro",           # Legacy Gemini Pro
    ]

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        return "gemini-2.0-flash"

    def _create_client(self) -> None:
        """Create the Gemini client."""
        try:
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model_id)
            self.status = ProviderStatus.HEALTHY
            logger.debug(f"Gemini client created with model {self.model_id}")
        except Exception as e:
            self.status = ProviderStatus.UNHEALTHY
            logger.error(f"Failed to create Gemini client: {e}")
            raise

    def discover_models(self) -> List[str]:
        """Discover available models from Gemini API."""
        try:
            models = genai.list_models()
            # Filter for models that support generateContent
            chat_models = []
            for m in models:
                # Check if model supports generateContent
                supported_methods = getattr(m, 'supported_generation_methods', [])
                if isinstance(supported_methods, list):
                    method_names = [
                        getattr(method, 'name', method) if hasattr(method, 'name') else str(method)
                        for method in supported_methods
                    ]
                    if "generateContent" in method_names and "gemini" in m.name.lower():
                        model_name = m.name.replace("models/", "")
                        chat_models.append(model_name)
            logger.debug(f"Gemini discovered {len(chat_models)} models")
            return chat_models
        except Exception as e:
            logger.warning(f"Gemini model discovery failed: {e}")
            return []

    def _rank_models(self, models: List[str]) -> Optional[str]:
        """Select the best Gemini model based on preferences."""
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

        # If no preferred model found, return the first gemini-2 or gemini-1.5 variant
        for model in sorted(models, reverse=True):
            if "gemini-2" in model or "gemini-1.5" in model:
                return model

        return None

    def _update_client_model(self, model_id: str) -> None:
        """Update the client to use a different model."""
        self._model_id = model_id
        self._client = genai.GenerativeModel(model_id)
        logger.debug(f"Gemini client updated to model {model_id}")

    @retry_with_backoff(RetryConfig(
        max_retries=3,
        retryable_exceptions=(
            google_exceptions.ResourceExhausted,
            google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded,
        ),
    ))
    def generate(self, prompt: str, **kwargs) -> Response:
        """
        Generate a response using Google's Gemini API.

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

        # Combine system message with prompt for Gemini
        full_prompt = f"{system_message}\n\n{prompt}"

        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        try:
            response = self._client.generate_content(
                full_prompt,
                generation_config=generation_config,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract response data
            text = ""
            if response.text:
                text = response.text
            elif response.parts:
                text = "".join(part.text for part in response.parts if hasattr(part, "text"))

            # Determine finish reason
            finish_reason = "stop"
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason"):
                    finish_reason = str(candidate.finish_reason).lower()

            # Token usage (Gemini provides this differently)
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                input_tokens = getattr(usage, "prompt_token_count", 0)
                output_tokens = getattr(usage, "candidates_token_count", 0)

            self.status = ProviderStatus.HEALTHY

            return Response(
                provider=self.name,
                model_id=self.model_id,
                text=text,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                raw_response=response,
            )

        except google_exceptions.ResourceExhausted as e:
            self.status = ProviderStatus.DEGRADED
            logger.warning(f"Gemini rate limit: {e}")
            raise
        except google_exceptions.DeadlineExceeded as e:
            self.status = ProviderStatus.DEGRADED
            logger.warning(f"Gemini timeout: {e}")
            raise
        except google_exceptions.ServiceUnavailable as e:
            self.status = ProviderStatus.UNHEALTHY
            logger.error(f"Gemini service unavailable: {e}")
            raise
        except Exception as e:
            self.status = ProviderStatus.UNHEALTHY
            logger.error(f"Gemini unexpected error: {e}")
            raise
