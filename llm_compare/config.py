"""Configuration management for LLM Compare."""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    timeout: int = 120
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class LlamaCppConfig:
    """Configuration for llama.cpp local models."""
    # Directories to search for GGUF models
    model_dirs: List[str] = field(default_factory=lambda: ["./models", "~/models"])

    # Default context size
    default_n_ctx: int = 4096

    # Default GPU layers (0 = CPU only)
    default_n_gpu_layers: int = 0

    # Specific model configurations (override auto-discovery)
    # Format: {"model_name": {"path": "...", "n_ctx": ..., "chat_format": "..."}}
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Config file path for external configuration
    config_file: str = "llamacpp.config.json"


@dataclass
class EvaluationWeights:
    """Weights for each evaluation phase."""
    pointwise: float = 0.30
    pairwise: float = 0.30
    adversarial: float = 0.25
    collaborative: float = 0.15

    def validate(self) -> bool:
        """Validate that weights sum to 1.0."""
        total = self.pointwise + self.pairwise + self.adversarial + self.collaborative
        return abs(total - 1.0) < 0.001


@dataclass
class Config:
    """Main configuration for LLM Compare."""
    # Paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "evaluations")

    # Provider settings
    provider_config: ProviderConfig = field(default_factory=ProviderConfig)

    # Evaluation settings
    evaluation_weights: EvaluationWeights = field(default_factory=EvaluationWeights)
    debate_rounds: int = 2

    # Concurrency
    max_concurrent_requests: int = 4

    # Key file pattern
    key_file_pattern: str = "*.key.txt"

    # Provider to key file mapping
    provider_key_files: Dict[str, str] = field(default_factory=lambda: {
        "openai": "openai.key.txt",
        "claude": "claude.key.txt",
        "gemini": "gemini.key.txt",
        "xai": "xai.key.txt",
    })

    # Model IDs for each provider (empty = auto-select best available)
    provider_models: Dict[str, str] = field(default_factory=dict)

    # llama.cpp configuration
    llamacpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)

    def get_key_file_path(self, provider: str) -> Optional[Path]:
        """Get the full path to a provider's key file."""
        if provider in self.provider_key_files:
            return self.project_root / self.provider_key_files[provider]
        return None

    def get_available_providers(self) -> List[str]:
        """Get list of providers with available key files."""
        available = []
        for provider, key_file in self.provider_key_files.items():
            key_path = self.project_root / key_file
            if key_path.exists():
                available.append(provider)
        return available

    def load_llamacpp_config(self) -> None:
        """Load llama.cpp configuration from JSON file if it exists."""
        config_path = self.project_root / self.llamacpp.config_file
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                if "model_dirs" in data:
                    self.llamacpp.model_dirs = data["model_dirs"]
                if "default_n_ctx" in data:
                    self.llamacpp.default_n_ctx = data["default_n_ctx"]
                if "default_n_gpu_layers" in data:
                    self.llamacpp.default_n_gpu_layers = data["default_n_gpu_layers"]
                if "models" in data:
                    self.llamacpp.models = data["models"]
            except Exception as e:
                pass  # Silently ignore config errors


# Global default configuration
default_config = Config()
