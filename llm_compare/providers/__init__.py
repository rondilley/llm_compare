"""LLM Provider implementations."""

from .base import LLMProvider, Response, ProviderStatus
from .discovery import ProviderDiscovery
from .openai_provider import OpenAIProvider
from .claude_provider import ClaudeProvider
from .gemini_provider import GeminiProvider
from .xai_provider import XAIProvider
from .mistral_provider import MistralProvider

# Conditional import for llama.cpp (requires llama-cpp-python)
try:
    from .llamacpp_provider import LlamaCppProvider
    _LLAMACPP_AVAILABLE = True
except ImportError:
    LlamaCppProvider = None
    _LLAMACPP_AVAILABLE = False

__all__ = [
    "LLMProvider",
    "Response",
    "ProviderStatus",
    "ProviderDiscovery",
    "OpenAIProvider",
    "ClaudeProvider",
    "GeminiProvider",
    "XAIProvider",
    "MistralProvider",
    "LlamaCppProvider",
]
