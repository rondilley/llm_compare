"""Model downloader using Hugging Face Hub."""

import shutil
from pathlib import Path
from typing import Optional

from .model_selector import ModelRecommendation
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DownloadError(Exception):
    """Raised when model download fails."""
    pass


class ModelDownloader:
    """Downloads GGUF models from Hugging Face Hub."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)

    def check_disk_space(self, required_mb: int) -> bool:
        """Check if sufficient disk space is available.

        Args:
            required_mb: Required space in MB.

        Returns:
            True if enough space is available.
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(self.models_dir)
        free_mb = usage.free // (1024 * 1024)
        # Require 10% extra for safety
        needed = int(required_mb * 1.1)
        return free_mb >= needed

    def get_free_space_mb(self) -> int:
        """Get free disk space in MB."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(self.models_dir)
        return usage.free // (1024 * 1024)

    def check_existing(self, filename: str) -> Optional[Path]:
        """Check if a model file already exists.

        Args:
            filename: The GGUF filename to check.

        Returns:
            Path to existing file, or None.
        """
        path = self.models_dir / filename
        if path.exists() and path.stat().st_size > 0:
            return path
        return None

    def download(self, recommendation: ModelRecommendation) -> Path:
        """Download a model from Hugging Face Hub.

        Args:
            recommendation: The model recommendation to download.

        Returns:
            Path to downloaded model file.

        Raises:
            DownloadError: If download fails.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise DownloadError(
                "huggingface_hub is required for model download.\n"
                "Install with: pip install huggingface_hub>=0.20.0"
            )

        self.models_dir.mkdir(parents=True, exist_ok=True)

        if not self.check_disk_space(recommendation.estimated_size_mb):
            free = self.get_free_space_mb()
            raise DownloadError(
                f"Insufficient disk space. Need ~{recommendation.estimated_size_mb}MB, "
                f"have {free}MB free in {self.models_dir}"
            )

        try:
            logger.info(
                f"Downloading {recommendation.filename} from {recommendation.repo_id}"
            )
            path = hf_hub_download(
                repo_id=recommendation.repo_id,
                filename=recommendation.filename,
                local_dir=str(self.models_dir),
            )
            result_path = Path(path)
            logger.info(f"Download complete: {result_path}")
            return result_path
        except Exception as e:
            raise DownloadError(f"Download failed: {e}") from e
