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
    """Google Gemini provider."""

    MODEL_PREFERENCES = [
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-pro",
        "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro",
    ]
    MODEL_FALLBACK_PATTERN = "gemini-"

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        return "gemini-2.0-flash"

    def _create_client(self) -> None:
        try:
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model_id)
            self.status = ProviderStatus.HEALTHY
        except Exception:
            self.status = ProviderStatus.UNHEALTHY
            logger.error("Failed to create Gemini client")
            raise

    def discover_models(self) -> List[str]:
        try:
            chat_models = []
            for m in genai.list_models():
                supported = getattr(m, 'supported_generation_methods', [])
                if isinstance(supported, list):
                    methods = [getattr(x, 'name', x) if hasattr(x, 'name') else str(x) for x in supported]
                    if "generateContent" in methods and "gemini" in m.name.lower():
                        chat_models.append(m.name.replace("models/", ""))
            return chat_models
        except Exception:
            logger.warning("Gemini model discovery failed")
            return []

    @retry_with_backoff(RetryConfig(
        max_retries=3,
        retryable_exceptions=(google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable, google_exceptions.DeadlineExceeded),
    ))
    def generate(self, prompt: str, **kwargs) -> Response:
        start_time = time.time()
        system_message = kwargs.get("system_message", "You are a helpful AI assistant.")
        full_prompt = f"{system_message}\n\n{prompt}"

        generation_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        try:
            response = self._client.generate_content(full_prompt, generation_config=generation_config)

            text = response.text if response.text else "".join(
                part.text for part in (response.parts or []) if hasattr(part, "text")
            )

            finish_reason = "stop"
            if hasattr(response, "candidates") and response.candidates:
                if hasattr(response.candidates[0], "finish_reason"):
                    finish_reason = str(response.candidates[0].finish_reason).lower()

            input_tokens = output_tokens = 0
            if hasattr(response, "usage_metadata"):
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

            self.status = ProviderStatus.HEALTHY
            return Response(
                provider=self.name,
                model_id=self.model_id,
                text=text,
                latency_ms=int((time.time() - start_time) * 1000),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                raw_response=response,
            )

        except google_exceptions.ResourceExhausted:
            self.status = ProviderStatus.DEGRADED
            raise
        except google_exceptions.DeadlineExceeded:
            self.status = ProviderStatus.DEGRADED
            raise
        except google_exceptions.ServiceUnavailable:
            self.status = ProviderStatus.UNHEALTHY
            raise
        except Exception:
            self.status = ProviderStatus.UNHEALTHY
            raise
