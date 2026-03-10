"""Generate llamacpp.config.json from model recommendation."""

import json
from pathlib import Path
from typing import Dict, Any

from .model_selector import ModelRecommendation
from ..config import LLAMACPP_CONFIG_SCHEMA, _validate_json_schema
from ..utils.logging import get_logger

logger = get_logger(__name__)


def generate_llamacpp_config(
    recommendation: ModelRecommendation,
    model_path: Path,
    models_dir: Path,
) -> Dict[str, Any]:
    """Generate a llamacpp.config.json dict from a recommendation.

    Args:
        recommendation: The model recommendation.
        model_path: Path to the downloaded model file.
        models_dir: Directory containing model files.

    Returns:
        Config dict matching LLAMACPP_CONFIG_SCHEMA.
    """
    model_key = model_path.stem

    config = {
        "model_dirs": [str(models_dir)],
        "default_n_ctx": recommendation.n_ctx,
        "default_n_gpu_layers": recommendation.n_gpu_layers,
        "models": {
            model_key: {
                "path": str(model_path),
                "n_ctx": recommendation.n_ctx,
                "n_gpu_layers": recommendation.n_gpu_layers,
                "chat_format": recommendation.chat_format,
            }
        },
    }

    # Validate before returning
    errors = _validate_json_schema(config, LLAMACPP_CONFIG_SCHEMA)
    if errors:
        logger.warning(f"Generated config has validation issues: {errors}")

    return config


def write_config(config_dict: Dict[str, Any], output_path: Path) -> None:
    """Write config dict to a JSON file.

    Args:
        config_dict: The config dictionary to write.
        output_path: Path for the output file.
    """
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Config written to {output_path}")
