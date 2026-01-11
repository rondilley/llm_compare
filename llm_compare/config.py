"""Configuration management for LLM Compare."""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class RepetitionMode(Enum):
    """Prompt repetition strategies (re-exported from prompting module)."""
    NONE = "none"
    SIMPLE = "simple"
    VERBOSE = "verbose"
    TRIPLE = "triple"


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


# JSON Schema for llama.cpp configuration
LLAMACPP_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "model_dirs": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 20,
        },
        "default_n_ctx": {
            "type": "integer",
            "minimum": 128,
            "maximum": 131072,
        },
        "default_n_gpu_layers": {
            "type": "integer",
            "minimum": 0,
            "maximum": 1000,
        },
        "models": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "n_ctx": {
                        "type": "integer",
                        "minimum": 128,
                        "maximum": 131072,
                    },
                    "n_gpu_layers": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1000,
                    },
                    "chat_format": {"type": "string", "maxLength": 100},
                },
                "required": ["path"],
            },
            "maxProperties": 50,
        },
    },
    "additionalProperties": False,
}

# Maximum prompt size (characters)
MAX_PROMPT_SIZE = 100000


def _validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any], path: str = "") -> List[str]:
    """
    Simple JSON schema validator (subset of JSON Schema spec).

    Args:
        data: Data to validate
        schema: Schema to validate against
        path: Current path for error messages

    Returns:
        List of validation error messages
    """
    errors = []

    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: Expected object, got {type(data).__name__}")
            return errors

        # Check additionalProperties
        if schema.get("additionalProperties") is False:
            allowed_keys = set(schema.get("properties", {}).keys())
            extra_keys = set(data.keys()) - allowed_keys
            if extra_keys:
                errors.append(f"{path}: Unexpected properties: {extra_keys}")

        # Check maxProperties
        max_props = schema.get("maxProperties")
        if max_props and len(data) > max_props:
            errors.append(f"{path}: Too many properties ({len(data)} > {max_props})")

        # Check required properties
        for required in schema.get("required", []):
            if required not in data:
                errors.append(f"{path}: Missing required property '{required}'")

        # Validate properties
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                sub_path = f"{path}.{key}" if path else key
                errors.extend(_validate_json_schema(value, properties[key], sub_path))
            elif "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
                sub_path = f"{path}.{key}" if path else key
                errors.extend(_validate_json_schema(value, schema["additionalProperties"], sub_path))

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: Expected array, got {type(data).__name__}")
            return errors

        # Check maxItems
        max_items = schema.get("maxItems")
        if max_items and len(data) > max_items:
            errors.append(f"{path}: Too many items ({len(data)} > {max_items})")

        # Validate items
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                sub_path = f"{path}[{i}]"
                errors.extend(_validate_json_schema(item, items_schema, sub_path))

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: Expected string, got {type(data).__name__}")
        else:
            max_length = schema.get("maxLength")
            if max_length and len(data) > max_length:
                errors.append(f"{path}: String too long ({len(data)} > {max_length})")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: Expected integer, got {type(data).__name__}")
        else:
            minimum = schema.get("minimum")
            maximum = schema.get("maximum")
            if minimum is not None and data < minimum:
                errors.append(f"{path}: Value {data} below minimum {minimum}")
            if maximum is not None and data > maximum:
                errors.append(f"{path}: Value {data} above maximum {maximum}")

    return errors


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
    """Weights for each evaluation phase (should sum to 1.0)."""
    pointwise: float = 0.30
    pairwise: float = 0.30
    adversarial: float = 0.25
    collaborative: float = 0.15


@dataclass
class RepetitionConfig:
    """
    Configuration for prompt repetition.

    Based on "Prompt Repetition Improves Non-Reasoning LLMs" (Leviathan et al., 2025).
    Repeating prompts allows each token to attend to all other tokens,
    improving performance without increasing output length or latency.
    """
    # Repetition mode to use
    mode: RepetitionMode = RepetitionMode.NONE

    # Compare repeated vs non-repeated responses
    compare_modes: bool = False

    # Auto-detect best mode based on prompt structure
    auto_detect: bool = False


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

    # Prompt repetition configuration
    repetition: RepetitionConfig = field(default_factory=RepetitionConfig)

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
        """
        Load llama.cpp configuration from JSON file if it exists.

        Validates the configuration against a schema and ensures
        model paths are within allowed directories.

        Raises:
            ConfigurationError: If configuration is invalid
        """
        config_path = self.project_root / self.llamacpp.config_file
        if not config_path.exists():
            return

        try:
            with open(config_path, "r") as f:
                data = json.load(f)

            # Validate against schema
            errors = _validate_json_schema(data, LLAMACPP_CONFIG_SCHEMA)
            if errors:
                logger.warning(f"llama.cpp config validation errors: {errors}")
                raise ConfigurationError(
                    f"Invalid llama.cpp configuration: {'; '.join(errors)}"
                )

            # Validate and apply model_dirs
            if "model_dirs" in data:
                validated_dirs = []
                for dir_path in data["model_dirs"]:
                    validated = self._validate_model_directory(dir_path)
                    if validated:
                        validated_dirs.append(validated)
                self.llamacpp.model_dirs = validated_dirs

            if "default_n_ctx" in data:
                self.llamacpp.default_n_ctx = data["default_n_ctx"]
            if "default_n_gpu_layers" in data:
                self.llamacpp.default_n_gpu_layers = data["default_n_gpu_layers"]

            # Validate model paths
            if "models" in data:
                validated_models = {}
                for model_name, model_config in data["models"].items():
                    if self._validate_model_path(model_config.get("path", "")):
                        validated_models[model_name] = model_config
                    else:
                        logger.warning(f"Skipping model {model_name}: invalid path")
                self.llamacpp.models = validated_models

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in llama.cpp config: parse error")
            raise ConfigurationError("Invalid JSON in llama.cpp configuration")
        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Failed to load llama.cpp config: unexpected error")
            raise ConfigurationError("Failed to load llama.cpp configuration")

    def _validate_model_directory(self, dir_path: str) -> Optional[str]:
        """
        Validate a model directory path.

        Args:
            dir_path: Directory path to validate

        Returns:
            Validated path string or None if invalid
        """
        try:
            path = Path(dir_path).expanduser().resolve()

            # Security: Check for path traversal attempts
            if ".." in str(dir_path):
                logger.warning(f"Rejecting model directory with path traversal: {dir_path}")
                return None

            # Must be a valid directory path (or creatable)
            # Note: Directory doesn't need to exist yet
            return str(path)
        except Exception:
            return None

    def _validate_model_path(self, model_path: str) -> bool:
        """
        Validate a model file path for security.

        Ensures the path:
        - Does not contain path traversal sequences
        - Has a valid .gguf extension
        - Is an absolute path or relative to allowed directories

        Args:
            model_path: Path to the model file

        Returns:
            True if path is valid, False otherwise
        """
        if not model_path:
            return False

        # Security: Reject path traversal attempts
        if ".." in model_path:
            logger.warning(f"Rejecting model path with traversal: path contains '..'")
            return False

        try:
            path = Path(model_path).expanduser()

            # Must have .gguf extension
            if path.suffix.lower() != ".gguf":
                logger.warning(f"Rejecting model path: must have .gguf extension")
                return False

            # If absolute, verify it's a reasonable path
            if path.is_absolute():
                resolved = path.resolve()
                # Ensure resolved path doesn't escape via symlinks
                if ".." in str(resolved):
                    return False
                return True

            # Relative paths are resolved against model_dirs during discovery
            return True

        except Exception:
            return False


# Global default configuration
default_config = Config()
