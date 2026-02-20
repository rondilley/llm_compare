"""Google Gemini provider implementation."""

import time
from typing import List, Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from google.api_core import exceptions as google_exceptions

from .base import LLMProvider, Response, ProviderStatus
from ..prompting.repetition import RepetitionMode
from ..utils.logging import get_logger
from ..utils.retry import retry_with_backoff, RetryConfig

logger = get_logger(__name__)


class PermanentQuotaError(Exception):
    """Model not available on the current billing plan (quota limit: 0)."""
    def __init__(self, model_id: str, original_error: Exception):
        self.model_id = model_id
        self.original_error = original_error
        super().__init__(f"Model {model_id} not available on current plan (quota limit: 0)")


class GeminiProvider(LLMProvider):
    """Google Gemini provider."""

    MODEL_PREFERENCES = [
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-pro",
        "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro",
    ]
    MODEL_FALLBACK_PATTERN = "gemini-"

    def __init__(self, *args, **kwargs):
        self._failed_models: set = set()
        super().__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        return "gemini-2.5-flash"

    def _create_client(self) -> None:
        try:
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model_id)
            self.status = ProviderStatus.HEALTHY
        except Exception:
            self.status = ProviderStatus.UNHEALTHY
            logger.error("Failed to create Gemini client")
            raise

    def _try_fallback_model(self) -> bool:
        """Fall back to the next model in preference order. Returns True if switched."""
        self._failed_models.add(self.model_id)
        for model in self.MODEL_PREFERENCES:
            if model not in self._failed_models:
                logger.warning(
                    f"gemini: {self.model_id} not available on current plan, "
                    f"falling back to {model}"
                )
                self._model_id = model
                self._client = genai.GenerativeModel(model)
                return True
        return False

    def generate(self, prompt, repetition_mode=RepetitionMode.NONE, **kwargs):
        while True:
            try:
                return super().generate(prompt, repetition_mode, **kwargs)
            except PermanentQuotaError:
                if not self._try_fallback_model():
                    raise

    def discover_models(self) -> List[str]:
        try:
            chat_models = []
            for m in genai.list_models():
                supported = getattr(m, 'supported_generation_methods', [])
                methods = [str(x) for x in supported] if supported else []
                if "generateContent" in methods and "gemini" in m.name.lower():
                    model_id = m.name.replace("models/", "")
                    chat_models.append(model_id)
            # Prefer canonical names: for each MODEL_PREFERENCES prefix, if both
            # an exact match and versioned variants exist, keep the exact match.
            # If only versioned variants exist, keep the shortest (most canonical).
            return self._deduplicate_models(chat_models)
        except Exception as e:
            logger.warning(f"Gemini model discovery failed: {e}")
            return []

    def _deduplicate_models(self, models: List[str]) -> List[str]:
        """Prefer canonical model names over versioned/preview variants."""
        models_set = set(models)
        result = set()
        matched = set()

        for preferred in self.MODEL_PREFERENCES:
            if preferred in models_set:
                # Exact match exists - use it, skip versioned variants
                result.add(preferred)
                matched.update(m for m in models if m.startswith(preferred + "-"))
                matched.add(preferred)
            else:
                # No exact match - pick the best variant (prefer "-latest", then shortest)
                variants = sorted(
                    [m for m in models if m.startswith(preferred) and m not in matched],
                    key=lambda m: (0 if m.endswith("-latest") else 1, len(m)),
                )
                if variants:
                    result.add(variants[0])
                    matched.update(variants)

        # Include any remaining models not matched by preferences
        for m in models:
            if m not in matched:
                result.add(m)

        return list(result)

    @retry_with_backoff(RetryConfig(
        max_retries=3,
        retryable_exceptions=(google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable, google_exceptions.DeadlineExceeded),
    ))
    def _generate_impl(self, prompt: str, **kwargs) -> Response:
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

        except google_exceptions.ResourceExhausted as e:
            if "limit: 0" in str(e):
                raise PermanentQuotaError(self.model_id, e) from e
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
