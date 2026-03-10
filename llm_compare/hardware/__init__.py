"""Hardware detection and local model setup."""

from .detector import detect_hardware, HardwareProfile, GPUInfo
from .model_selector import select_model, ModelRecommendation, InsufficientResourcesError
from .downloader import ModelDownloader, DownloadError
from .config_generator import generate_llamacpp_config, write_config

__all__ = [
    "detect_hardware",
    "HardwareProfile",
    "GPUInfo",
    "select_model",
    "ModelRecommendation",
    "InsufficientResourcesError",
    "ModelDownloader",
    "DownloadError",
    "generate_llamacpp_config",
    "write_config",
]
