"""Model selection based on hardware capabilities."""

from dataclasses import dataclass
from typing import List, Optional

from .detector import HardwareProfile
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelSpec:
    """A GGUF model specification in the catalog."""
    repo_id: str
    family: str           # e.g. "llama-3.1-8b-instruct"
    total_layers: int
    chat_format: str
    quantizations: dict   # quant_name -> {"filename": str, "size_mb": int, "vram_mb": int}


# Curated catalog of Llama GGUF models from bartowski's HF repos
MODEL_CATALOG: List[ModelSpec] = [
    ModelSpec(
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        family="llama-3.1-8b-instruct",
        total_layers=33,
        chat_format="llama-3",
        quantizations={
            "Q8_0": {
                "filename": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
                "size_mb": 8540,
                "vram_mb": 9500,
            },
            "Q6_K": {
                "filename": "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
                "size_mb": 6600,
                "vram_mb": 7500,
            },
            "Q4_K_M": {
                "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                "size_mb": 4920,
                "vram_mb": 5800,
            },
        },
    ),
    ModelSpec(
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        family="llama-3.2-3b-instruct",
        total_layers=29,
        chat_format="llama-3",
        quantizations={
            "Q8_0": {
                "filename": "Llama-3.2-3B-Instruct-Q8_0.gguf",
                "size_mb": 3420,
                "vram_mb": 4200,
            },
            "Q4_K_M": {
                "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                "size_mb": 2020,
                "vram_mb": 2800,
            },
        },
    ),
    ModelSpec(
        repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
        family="llama-3.2-1b-instruct",
        total_layers=17,
        chat_format="llama-3",
        quantizations={
            "Q8_0": {
                "filename": "Llama-3.2-1B-Instruct-Q8_0.gguf",
                "size_mb": 1320,
                "vram_mb": 1800,
            },
            "Q4_K_M": {
                "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                "size_mb": 760,
                "vram_mb": 1200,
            },
        },
    ),
]


@dataclass
class ModelRecommendation:
    """A recommended model based on hardware detection."""
    repo_id: str
    filename: str
    model_family: str
    quantization: str
    estimated_size_mb: int
    estimated_vram_mb: int
    n_gpu_layers: int       # -1 = all layers on GPU
    n_ctx: int
    chat_format: str
    total_layers: int
    reason: str


# VRAM headroom reserved for OS/other apps
VRAM_HEADROOM_MB = 1500
# RAM headroom reserved for OS/other apps
RAM_HEADROOM_MB = 4000

# GPU selection table: (min_vram_mb, model_family, quant, n_ctx)
# Context sizes set high enough for adversarial debate prompts (~10K+ tokens)
_GPU_TIERS = [
    (12000, "llama-3.1-8b-instruct", "Q8_0", 32768),
    (8000,  "llama-3.1-8b-instruct", "Q6_K", 16384),
    (7000,  "llama-3.1-8b-instruct", "Q4_K_M", 16384),
    (5000,  "llama-3.2-3b-instruct", "Q8_0", 16384),
    (4000,  "llama-3.2-3b-instruct", "Q4_K_M", 16384),
    (3000,  "llama-3.2-1b-instruct", "Q8_0", 8192),
]

# CPU-only selection table: (min_ram_mb, model_family, quant, n_ctx)
_CPU_TIERS = [
    (16000, "llama-3.1-8b-instruct", "Q4_K_M", 4096),
    (8000,  "llama-3.2-3b-instruct", "Q4_K_M", 4096),
    (4000,  "llama-3.2-1b-instruct", "Q4_K_M", 2048),
]


class InsufficientResourcesError(Exception):
    """Raised when no model fits the detected hardware."""
    pass


def select_model(profile: HardwareProfile) -> ModelRecommendation:
    """Select the best model for the given hardware profile.

    Args:
        profile: Detected hardware capabilities.

    Returns:
        ModelRecommendation with download/config details.

    Raises:
        InsufficientResourcesError: If hardware is too limited for any model.
    """
    usable_vram = profile.total_vram_mb - VRAM_HEADROOM_MB if profile.has_gpu else 0
    usable_ram = profile.available_ram_mb - RAM_HEADROOM_MB

    # Try full GPU offload first
    if profile.has_gpu and usable_vram > 0:
        for min_vram, family, quant, n_ctx in _GPU_TIERS:
            if usable_vram >= min_vram:
                spec = _find_spec(family)
                q = spec.quantizations[quant]
                gpu_name = profile.best_gpu.name
                return ModelRecommendation(
                    repo_id=spec.repo_id,
                    filename=q["filename"],
                    model_family=family,
                    quantization=quant,
                    estimated_size_mb=q["size_mb"],
                    estimated_vram_mb=q["vram_mb"],
                    n_gpu_layers=-1,
                    n_ctx=n_ctx,
                    chat_format=spec.chat_format,
                    total_layers=spec.total_layers,
                    reason=f"Full GPU offload on {gpu_name} "
                           f"({usable_vram + VRAM_HEADROOM_MB}MB VRAM, "
                           f"{usable_vram}MB usable after headroom)",
                )

        # Partial GPU offload: GPU exists but not enough VRAM for smallest full-offload tier
        rec = _try_partial_offload(profile, usable_vram, usable_ram)
        if rec:
            return rec

    # CPU-only selection
    for min_ram, family, quant, n_ctx in _CPU_TIERS:
        if usable_ram >= min_ram:
            spec = _find_spec(family)
            q = spec.quantizations[quant]
            return ModelRecommendation(
                repo_id=spec.repo_id,
                filename=q["filename"],
                model_family=family,
                quantization=quant,
                estimated_size_mb=q["size_mb"],
                estimated_vram_mb=0,
                n_gpu_layers=0,
                n_ctx=n_ctx,
                chat_format=spec.chat_format,
                total_layers=spec.total_layers,
                reason=f"CPU-only mode ({profile.available_ram_mb}MB RAM available, "
                       f"{usable_ram}MB usable after headroom)",
            )

    raise InsufficientResourcesError(
        f"Insufficient resources for any supported model. "
        f"Available RAM: {profile.available_ram_mb}MB "
        f"(need at least {4000 + RAM_HEADROOM_MB}MB). "
        f"VRAM: {profile.total_vram_mb}MB."
    )


def _try_partial_offload(
    profile: HardwareProfile,
    usable_vram: int,
    usable_ram: int,
) -> Optional[ModelRecommendation]:
    """Try partial GPU offload when VRAM is too small for full offload."""
    # Pick the best model that fits in RAM + partial VRAM
    total_available = usable_vram + usable_ram

    for min_ram, family, quant, n_ctx in _CPU_TIERS:
        spec = _find_spec(family)
        q = spec.quantizations[quant]
        if total_available >= q["vram_mb"]:
            # Calculate how many layers fit in VRAM
            layers = int(spec.total_layers * usable_vram / q["vram_mb"])
            layers = max(0, min(layers, spec.total_layers))
            if layers == 0:
                continue
            gpu_name = profile.best_gpu.name
            return ModelRecommendation(
                repo_id=spec.repo_id,
                filename=q["filename"],
                model_family=family,
                quantization=quant,
                estimated_size_mb=q["size_mb"],
                estimated_vram_mb=q["vram_mb"],
                n_gpu_layers=layers,
                n_ctx=n_ctx,
                chat_format=spec.chat_format,
                total_layers=spec.total_layers,
                reason=f"Partial GPU offload: {layers}/{spec.total_layers} layers "
                       f"on {gpu_name} ({usable_vram}MB usable VRAM), "
                       f"rest on CPU ({usable_ram}MB usable RAM)",
            )
    return None


def _find_spec(family: str) -> ModelSpec:
    """Find a ModelSpec by family name."""
    for spec in MODEL_CATALOG:
        if spec.family == family:
            return spec
    raise ValueError(f"Unknown model family: {family}")
