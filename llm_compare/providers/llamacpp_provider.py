"""llama.cpp provider implementation for local GGUF models."""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from .base import LLMProvider, Response, ProviderStatus
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Lazy import to avoid requiring llama-cpp-python when not used
Llama = None


def _ensure_llama_imported():
    """Lazily import llama-cpp-python."""
    global Llama
    if Llama is None:
        try:
            from llama_cpp import Llama as _Llama
            Llama = _Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for local model support. "
                "Install with: pip install llama-cpp-python"
            )


class LlamaCppProvider(LLMProvider):
    """llama.cpp provider for local GGUF model inference."""

    def __init__(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        chat_format: Optional[str] = None,
        timeout: int = 300,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize the llama.cpp provider.

        Args:
            model_path: Path to the GGUF model file
            model_name: Optional friendly name for the model
            n_ctx: Context window size (default 4096)
            n_gpu_layers: Number of layers to offload to GPU (0 for CPU only)
            chat_format: Chat format template (e.g., "llama-2", "chatml", "mistral-instruct")
            timeout: Generation timeout in seconds
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            **kwargs: Additional arguments passed to Llama constructor
        """
        self.model_path = Path(model_path)
        self._model_name = model_name or self.model_path.stem
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.chat_format = chat_format
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._extra_kwargs = kwargs
        self.status = ProviderStatus.HEALTHY
        self._client = None

        # Don't call parent __init__ since we handle initialization differently
        self._create_client()

    @property
    def name(self) -> str:
        return f"llamacpp_{self._model_name}"

    @property
    def default_model(self) -> str:
        return self._model_name

    @property
    def model_id(self) -> str:
        return self._model_name

    def _create_client(self) -> None:
        """Create the llama.cpp model instance."""
        _ensure_llama_imported()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            logger.info(f"Loading llama.cpp model: {self.model_path}")
            logger.info(f"Context size: {self.n_ctx}, GPU layers: {self.n_gpu_layers}")

            kwargs = {
                "model_path": str(self.model_path),
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "verbose": False,
            }

            if self.chat_format:
                kwargs["chat_format"] = self.chat_format

            kwargs.update(self._extra_kwargs)

            self._client = Llama(**kwargs)
            self.status = ProviderStatus.HEALTHY
            logger.info(f"llama.cpp model loaded: {self._model_name}")

        except Exception as e:
            self.status = ProviderStatus.UNHEALTHY
            logger.error(f"Failed to load llama.cpp model: {e}")
            raise

    def generate(self, prompt: str, **kwargs) -> Response:
        """
        Generate a response using the local llama.cpp model.

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
            # Use chat completion if chat_format is set, otherwise use raw completion
            if self.chat_format or self._supports_chat():
                response = self._client.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Extract response from chat format
                text = response["choices"][0]["message"]["content"]
                finish_reason = response["choices"][0].get("finish_reason", "stop")

                # Token usage
                usage = response.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
            else:
                # Raw completion for models without chat format
                full_prompt = f"{system_message}\n\nUser: {prompt}\n\nAssistant:"
                response = self._client(
                    full_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    echo=False,
                )

                text = response["choices"][0]["text"].strip()
                finish_reason = response["choices"][0].get("finish_reason", "stop")

                usage = response.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

            latency_ms = int((time.time() - start_time) * 1000)

            self.status = ProviderStatus.HEALTHY

            return Response(
                provider=self.name,
                model_id=self._model_name,
                text=text,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                raw_response=response,
            )

        except Exception as e:
            self.status = ProviderStatus.UNHEALTHY
            logger.error(f"llama.cpp generation error: {e}")
            raise

    def _supports_chat(self) -> bool:
        """Check if the model supports chat completion."""
        # Try to detect chat format from model metadata
        if hasattr(self._client, "metadata"):
            metadata = self._client.metadata
            # Check for common chat indicators
            if any(key in str(metadata).lower() for key in ["chat", "instruct", "conversation"]):
                return True
        return False

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._client is not None:
            del self._client
            self._client = None
            self.status = ProviderStatus.DISABLED
            logger.info(f"Unloaded model: {self._model_name}")


class LlamaCppModelConfig:
    """Configuration for a llama.cpp model."""

    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        chat_format: Optional[str] = None,
        **kwargs
    ):
        self.path = path
        self.name = name
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.chat_format = chat_format
        self.extra_kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "chat_format": self.chat_format,
            **self.extra_kwargs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LlamaCppModelConfig":
        return cls(
            path=data["path"],
            name=data.get("name"),
            n_ctx=data.get("n_ctx", 4096),
            n_gpu_layers=data.get("n_gpu_layers", 0),
            chat_format=data.get("chat_format"),
            **{k: v for k, v in data.items() if k not in [
                "path", "name", "n_ctx", "n_gpu_layers", "chat_format"
            ]},
        )


def discover_gguf_models(
    search_paths: List[Path],
    default_n_ctx: int = 4096,
    default_n_gpu_layers: int = 0,
) -> List[LlamaCppModelConfig]:
    """
    Discover GGUF model files in the given paths.

    Args:
        search_paths: List of directories to search
        default_n_ctx: Default context size
        default_n_gpu_layers: Default GPU layers

    Returns:
        List of model configurations
    """
    configs = []

    for search_path in search_paths:
        path = Path(search_path)
        if not path.exists():
            continue

        # Find all .gguf files
        for model_file in path.glob("**/*.gguf"):
            # Infer chat format from filename
            chat_format = _infer_chat_format(model_file.name)

            config = LlamaCppModelConfig(
                path=str(model_file),
                name=model_file.stem,
                n_ctx=default_n_ctx,
                n_gpu_layers=default_n_gpu_layers,
                chat_format=chat_format,
            )
            configs.append(config)
            logger.debug(f"Discovered GGUF model: {model_file}")

    return configs


def _infer_chat_format(filename: str) -> Optional[str]:
    """Infer chat format from model filename."""
    filename_lower = filename.lower()

    # Common chat format patterns
    if "llama-2" in filename_lower or "llama2" in filename_lower:
        return "llama-2"
    elif "llama-3" in filename_lower or "llama3" in filename_lower:
        return "llama-3"
    elif "mistral" in filename_lower:
        return "mistral-instruct"
    elif "chatml" in filename_lower or "qwen" in filename_lower:
        return "chatml"
    elif "gemma" in filename_lower:
        return "gemma"
    elif "phi" in filename_lower:
        return "chatml"
    elif "vicuna" in filename_lower:
        return "vicuna"
    elif "alpaca" in filename_lower:
        return "alpaca"

    # Default to chatml for instruct models
    if "instruct" in filename_lower or "chat" in filename_lower:
        return "chatml"

    return None
