"""Prompting strategies and transformations."""

from .repetition import (
    RepetitionMode,
    PromptRepetition,
    apply_repetition,
    detect_recommended_mode,
)

__all__ = [
    "RepetitionMode",
    "PromptRepetition",
    "apply_repetition",
    "detect_recommended_mode",
]
