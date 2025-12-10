"""Provider discovery and management."""

from pathlib import Path
from typing import Dict, List, Optional, Type

from ..config import Config, default_config
from ..utils.logging import get_logger
from .base import LLMProvider, ProviderStatus

logger = get_logger(__name__)


class ProviderDiscovery:
    """Discovers and manages available LLM providers."""

    # Registry of provider classes
    _provider_classes: Dict[str, Type[LLMProvider]] = {}

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize provider discovery.

        Args:
            config: Configuration object (uses default if not specified)
        """
        self.config = config or default_config
        self._providers: Dict[str, LLMProvider] = {}
        self._registration_done = False

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[LLMProvider]) -> None:
        """
        Register a provider class.

        Args:
            name: Provider name identifier
            provider_class: Provider class to register
        """
        cls._provider_classes[name] = provider_class
        logger.debug(f"Registered provider: {name}")

    def _ensure_providers_registered(self) -> None:
        """Ensure all provider classes are registered."""
        if self._registration_done:
            return

        # Import providers to trigger registration
        from .openai_provider import OpenAIProvider
        from .claude_provider import ClaudeProvider
        from .gemini_provider import GeminiProvider
        from .xai_provider import XAIProvider

        self.register_provider("openai", OpenAIProvider)
        self.register_provider("claude", ClaudeProvider)
        self.register_provider("gemini", GeminiProvider)
        self.register_provider("xai", XAIProvider)

        self._registration_done = True

    def discover(self) -> Dict[str, LLMProvider]:
        """
        Discover and initialize all available providers.

        Returns:
            Dictionary of provider name to provider instance
        """
        self._ensure_providers_registered()
        self._providers.clear()

        # Discover API-based providers
        self._discover_api_providers()

        # Discover local llama.cpp models
        self._discover_llamacpp_models()

        if not self._providers:
            logger.warning("No providers discovered. Check API key files or model directories.")

        return self._providers

    def _discover_api_providers(self) -> None:
        """Discover API-based providers from key files."""
        for provider_name, key_file in self.config.provider_key_files.items():
            key_path = self.config.project_root / key_file

            if not key_path.exists():
                logger.debug(f"Key file not found for {provider_name}: {key_path}")
                continue

            try:
                api_key = self._read_key_file(key_path)
                if not api_key:
                    logger.warning(f"Empty key file for {provider_name}: {key_path}")
                    continue

                provider = self._create_provider(provider_name, api_key)
                if provider:
                    self._providers[provider_name] = provider
                    logger.info(f"Discovered provider: {provider_name} ({provider.model_id})")

            except Exception as e:
                logger.error(f"Failed to initialize {provider_name}: {e}")

    def _discover_llamacpp_models(self) -> None:
        """Discover local llama.cpp GGUF models."""
        # Load any external configuration
        self.config.load_llamacpp_config()

        # First, try explicitly configured models
        for model_name, model_config in self.config.llamacpp.models.items():
            try:
                provider = self._create_llamacpp_provider(model_name, model_config)
                if provider:
                    self._providers[provider.name] = provider
                    logger.info(f"Discovered llama.cpp model: {provider.name}")
            except Exception as e:
                logger.error(f"Failed to load llama.cpp model {model_name}: {e}")

        # Then, auto-discover from model directories
        for model_dir in self.config.llamacpp.model_dirs:
            # Expand user home directory
            model_path = Path(model_dir).expanduser()
            if not model_path.exists():
                continue

            # Find all .gguf files
            for gguf_file in model_path.glob("**/*.gguf"):
                model_name = gguf_file.stem

                # Skip if already configured explicitly
                if model_name in self.config.llamacpp.models:
                    continue

                # Skip if a provider with this name already exists
                provider_name = f"llamacpp_{model_name}"
                if provider_name in self._providers:
                    continue

                try:
                    model_config = {
                        "path": str(gguf_file),
                        "n_ctx": self.config.llamacpp.default_n_ctx,
                        "n_gpu_layers": self.config.llamacpp.default_n_gpu_layers,
                    }
                    provider = self._create_llamacpp_provider(model_name, model_config)
                    if provider:
                        self._providers[provider.name] = provider
                        logger.info(f"Auto-discovered llama.cpp model: {provider.name}")
                except Exception as e:
                    logger.warning(f"Failed to auto-load llama.cpp model {gguf_file}: {e}")

    def _create_llamacpp_provider(
        self,
        model_name: str,
        model_config: Dict
    ) -> Optional[LLMProvider]:
        """
        Create a llama.cpp provider instance.

        Args:
            model_name: Name for the model
            model_config: Model configuration dict

        Returns:
            LlamaCppProvider instance or None
        """
        try:
            from .llamacpp_provider import LlamaCppProvider

            provider = LlamaCppProvider(
                model_path=model_config["path"],
                model_name=model_name,
                n_ctx=model_config.get("n_ctx", self.config.llamacpp.default_n_ctx),
                n_gpu_layers=model_config.get("n_gpu_layers", self.config.llamacpp.default_n_gpu_layers),
                chat_format=model_config.get("chat_format"),
                timeout=self.config.provider_config.timeout,
                max_tokens=self.config.provider_config.max_tokens,
                temperature=self.config.provider_config.temperature,
            )
            return provider
        except ImportError:
            logger.debug("llama-cpp-python not installed, skipping local model support")
            return None
        except Exception as e:
            logger.error(f"Failed to create llama.cpp provider: {e}")
            return None

    def _read_key_file(self, key_path: Path) -> Optional[str]:
        """
        Read API key from file.

        Args:
            key_path: Path to key file

        Returns:
            API key string or None
        """
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                # Read first line, strip whitespace
                key = f.readline().strip()
                return key if key else None
        except Exception as e:
            logger.error(f"Failed to read key file {key_path}: {e}")
            return None

    def _create_provider(
        self,
        provider_name: str,
        api_key: str
    ) -> Optional[LLMProvider]:
        """
        Create a provider instance with dynamic model selection.

        Args:
            provider_name: Name of the provider
            api_key: API key for authentication

        Returns:
            Provider instance or None if creation failed
        """
        if provider_name not in self._provider_classes:
            logger.error(f"Unknown provider: {provider_name}")
            return None

        provider_class = self._provider_classes[provider_name]
        configured_model = self.config.provider_models.get(provider_name)

        try:
            # First create provider with default/configured model
            provider = provider_class(
                api_key=api_key,
                model_id=configured_model,
                timeout=self.config.provider_config.timeout,
                max_tokens=self.config.provider_config.max_tokens,
                temperature=self.config.provider_config.temperature,
            )

            # If no model was explicitly configured, discover and select the best available
            if not configured_model:
                try:
                    best_model = provider.select_best_model()
                    if best_model and best_model != provider.model_id:
                        logger.info(f"{provider_name}: Auto-selected model: {best_model}")
                        # Recreate provider with the best model
                        provider = provider_class(
                            api_key=api_key,
                            model_id=best_model,
                            timeout=self.config.provider_config.timeout,
                            max_tokens=self.config.provider_config.max_tokens,
                            temperature=self.config.provider_config.temperature,
                        )
                except Exception as e:
                    logger.warning(f"{provider_name}: Model discovery failed, using default: {e}")

            return provider
        except Exception as e:
            logger.error(f"Failed to create provider {provider_name}: {e}")
            return None

    def get_provider(self, name: str) -> Optional[LLMProvider]:
        """
        Get a specific provider by name.

        Args:
            name: Provider name

        Returns:
            Provider instance or None if not available
        """
        return self._providers.get(name)

    def get_active_providers(self) -> List[LLMProvider]:
        """
        Get all active (healthy or degraded) providers.

        Returns:
            List of active provider instances
        """
        return [
            p for p in self._providers.values()
            if p.status in (ProviderStatus.HEALTHY, ProviderStatus.DEGRADED)
        ]

    def get_provider_names(self) -> List[str]:
        """
        Get names of all discovered providers.

        Returns:
            List of provider names
        """
        return list(self._providers.keys())

    @property
    def providers(self) -> Dict[str, LLMProvider]:
        """Get all discovered providers."""
        return self._providers
