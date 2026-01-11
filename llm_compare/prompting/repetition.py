"""
Prompt repetition strategies based on:
"Prompt Repetition Improves Non-Reasoning LLMs" (Leviathan et al., 2025)

Key insight: Repeating the prompt allows each token to attend to every other
prompt token, improving performance without increasing output length or latency.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class RepetitionMode(Enum):
    """Prompt repetition strategies."""
    NONE = "none"           # No repetition (baseline)
    SIMPLE = "simple"       # <QUERY><QUERY>
    VERBOSE = "verbose"     # <QUERY> Let me repeat that: <QUERY>
    TRIPLE = "triple"       # 3x with transitions


@dataclass
class PromptRepetition:
    """Configuration for prompt repetition."""
    mode: RepetitionMode = RepetitionMode.NONE
    compare_modes: bool = False  # Run both repeated and non-repeated

    def apply(self, prompt: str) -> str:
        """Apply repetition to a prompt."""
        return apply_repetition(prompt, self.mode)

    def should_compare(self) -> bool:
        """Whether to run comparison between modes."""
        return self.compare_modes


def apply_repetition(prompt: str, mode: RepetitionMode) -> str:
    """
    Apply prompt repetition according to the specified mode.

    Args:
        prompt: Original prompt text
        mode: Repetition strategy to apply

    Returns:
        Transformed prompt with repetition applied
    """
    if mode == RepetitionMode.NONE:
        return prompt

    elif mode == RepetitionMode.SIMPLE:
        # Simple repetition: <QUERY><QUERY>
        return f"{prompt}\n\n{prompt}"

    elif mode == RepetitionMode.VERBOSE:
        # Verbose: <QUERY> Let me repeat that: <QUERY>
        return f"{prompt}\n\nLet me repeat that:\n\n{prompt}"

    elif mode == RepetitionMode.TRIPLE:
        # Triple repetition with transitions
        return (
            f"{prompt}\n\n"
            f"Let me repeat that:\n\n"
            f"{prompt}\n\n"
            f"Let me repeat that one more time:\n\n"
            f"{prompt}"
        )

    return prompt


def detect_recommended_mode(prompt: str) -> Tuple[RepetitionMode, str]:
    """
    Analyze prompt structure and recommend a repetition mode.

    Based on the paper's findings:
    - Options-first prompts benefit most from repetition
    - Question-first prompts benefit less but still improve
    - Very long prompts may not benefit (context limit concerns)

    Args:
        prompt: Prompt to analyze

    Returns:
        Tuple of (recommended mode, reason string)
    """
    prompt_lower = prompt.lower()
    prompt_len = len(prompt)

    # Very long prompts - skip repetition (may hit context limits)
    if prompt_len > 50000:
        return (
            RepetitionMode.NONE,
            "Prompt too long for repetition (>50K chars)"
        )

    # Detect options-first pattern (multiple choice with options before question)
    options_pattern = r'^[A-D][.):]\s+.+\n'
    has_early_options = bool(re.search(options_pattern, prompt[:500], re.MULTILINE))

    # Detect question at end
    ends_with_question = prompt.rstrip().endswith('?')

    # Detect list/sequence patterns (NameIndex/MiddleMatch-like tasks)
    has_long_list = (
        prompt.count(',') > 10 or
        prompt.count('\n') > 20 or
        bool(re.search(r'\d+\.\s+\w+', prompt))  # Numbered list
    )

    # Check for explicit "think step by step" or reasoning instructions
    has_reasoning = any(phrase in prompt_lower for phrase in [
        'think step by step',
        'let\'s think',
        'reason through',
        'chain of thought',
        'show your work',
        'explain your reasoning'
    ])

    # Reasoning mode - repetition is neutral, skip it
    if has_reasoning:
        return (
            RepetitionMode.NONE,
            "Prompt requests reasoning; repetition neutral per paper"
        )

    # Options-first or list tasks benefit most
    if has_early_options and ends_with_question:
        return (
            RepetitionMode.SIMPLE,
            "Options-first pattern detected; high benefit from repetition"
        )

    # Long list/sequence tasks - triple repetition often best
    if has_long_list:
        return (
            RepetitionMode.TRIPLE,
            "List/sequence task detected; triple repetition recommended"
        )

    # Standard prompts - simple repetition
    if prompt_len < 10000:
        return (
            RepetitionMode.SIMPLE,
            "Standard prompt; simple repetition recommended"
        )

    # Medium-length prompts - verbose to be safe
    return (
        RepetitionMode.VERBOSE,
        "Medium-length prompt; verbose repetition recommended"
    )


def get_repetition_info(mode: RepetitionMode) -> dict:
    """Get descriptive information about a repetition mode."""
    info = {
        RepetitionMode.NONE: {
            "name": "None (Baseline)",
            "description": "No prompt modification",
            "token_multiplier": 1.0,
        },
        RepetitionMode.SIMPLE: {
            "name": "Simple Repetition",
            "description": "Prompt repeated twice: <QUERY><QUERY>",
            "token_multiplier": 2.0,
        },
        RepetitionMode.VERBOSE: {
            "name": "Verbose Repetition",
            "description": "Prompt with transition: <QUERY> Let me repeat: <QUERY>",
            "token_multiplier": 2.05,
        },
        RepetitionMode.TRIPLE: {
            "name": "Triple Repetition",
            "description": "Prompt repeated 3x with transitions",
            "token_multiplier": 3.1,
        },
    }
    return info.get(mode, info[RepetitionMode.NONE])
