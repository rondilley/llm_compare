"""Base provider interface for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger
from ..prompting.repetition import RepetitionMode, apply_repetition

logger = get_logger(__name__)


class ProviderStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"


@dataclass
class Response:
    """Response from an LLM provider."""
    provider: str
    model_id: str
    text: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_response: Optional[Any] = field(default=None, repr=False)
    repetition_mode: RepetitionMode = RepetitionMode.NONE

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        del d["raw_response"]  # Not serializable
        d["timestamp"] = self.timestamp.isoformat()
        d["repetition_mode"] = self.repetition_mode.value
        return d


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    # Subclasses should override with their preferred models (best first)
    MODEL_PREFERENCES: List[str] = []
    # Fallback pattern to match if no preference found (e.g., "gpt-4", "claude-")
    MODEL_FALLBACK_PATTERN: Optional[str] = None

    def __init__(
        self,
        api_key: str,
        model_id: Optional[str] = None,
        timeout: int = 120,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        self.api_key = api_key
        self._model_id = model_id
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.status = ProviderStatus.HEALTHY
        self._client = None
        self._create_client()

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        pass

    @property
    def model_id(self) -> str:
        return self._model_id or self.default_model

    def discover_models(self) -> List[str]:
        """Override in subclasses to query API for available models."""
        return []

    def select_best_model(self) -> str:
        """Select the best available model based on MODEL_PREFERENCES."""
        available = self.discover_models()
        if not available:
            return self.default_model

        best = self._rank_models(available)
        if best:
            logger.info(f"{self.name}: Selected model: {best}")
            return best
        return self.default_model

    def _rank_models(self, models: List[str]) -> Optional[str]:
        """Rank models by MODEL_PREFERENCES, with fallback pattern matching."""
        models_set = set(models)

        # Try each preferred model (exact match, then prefix match)
        for preferred in self.MODEL_PREFERENCES:
            if preferred in models_set:
                return preferred
            for model in models:
                if model.startswith(preferred):
                    return model

        # Fallback: find any model matching the fallback pattern
        if self.MODEL_FALLBACK_PATTERN:
            for model in sorted(models, reverse=True):
                if self.MODEL_FALLBACK_PATTERN in model:
                    return model

        return None

    @abstractmethod
    def _create_client(self) -> None:
        pass

    @abstractmethod
    def _generate_impl(self, prompt: str, **kwargs) -> Response:
        """Generate a response. Override in subclasses."""
        pass

    def generate(
        self,
        prompt: str,
        repetition_mode: RepetitionMode = RepetitionMode.NONE,
        **kwargs
    ) -> Response:
        """
        Generate a response with optional prompt repetition.

        Args:
            prompt: Original prompt text
            repetition_mode: Repetition strategy to apply
            **kwargs: Additional arguments passed to provider

        Returns:
            Response object with repetition_mode field set
        """
        # Apply repetition transformation
        effective_prompt = apply_repetition(prompt, repetition_mode)

        # Call the provider-specific implementation
        response = self._generate_impl(effective_prompt, **kwargs)

        # Record which repetition mode was used
        response.repetition_mode = repetition_mode

        return response

    def evaluate(self, prompt: str, response_text: str, rubric: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a response against a rubric using G-Eval methodology."""
        eval_prompt = self._build_evaluation_prompt(prompt, response_text, rubric)
        result = self.generate(eval_prompt)
        return self._parse_evaluation_response(result.text, rubric)

    def _build_evaluation_prompt(
        self,
        prompt: str,
        response_text: str,
        rubric: Dict[str, Any]
    ) -> str:
        """Build an evaluation prompt using G-Eval methodology."""
        return f"""You are evaluating an AI response to the following prompt:

ORIGINAL PROMPT:
{prompt}

RESPONSE TO EVALUATE:
{response_text}

EVALUATION RUBRIC - {rubric['name']}:
{rubric['description']}

Scale: {rubric.get('scale_min', 0)} to {rubric.get('scale_max', 10)}

{rubric.get('scoring_guidance', '')}

Please evaluate this response on a scale of {rubric.get('scale_min', 0)}-{rubric.get('scale_max', 10)} for {rubric['name']}.

First, think through your evaluation step by step:
1. What aspects of {rubric['name']} does this response demonstrate well?
2. What aspects are lacking or could be improved?
3. How does it compare to an ideal response?

Then provide your final score and a brief justification.

OUTPUT FORMAT (you must follow this exactly):
REASONING: [Your step-by-step analysis]
SCORE: [A single number from {rubric.get('scale_min', 0)} to {rubric.get('scale_max', 10)}]
JUSTIFICATION: [Brief explanation of score]"""

    def _parse_evaluation_response(
        self,
        response_text: str,
        rubric: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse an evaluation response to extract score and reasoning."""
        import re

        result = {
            "score": 5.0,  # Default middle score
            "reasoning": "",
            "justification": "",
            "raw_response": response_text,
        }

        lines = response_text.strip().split("\n")
        current_section = None
        section_content = []

        for line in lines:
            # Remove markdown formatting for matching
            line_clean = re.sub(r'\*+', '', line).strip()
            line_upper = line_clean.upper()

            if line_upper.startswith("REASONING:") or line_upper.startswith("REASONING"):
                if current_section:
                    result[current_section] = " ".join(section_content).strip()
                current_section = "reasoning"
                content = line_clean.split(":", 1)[1].strip() if ":" in line_clean else ""
                section_content = [content]
            elif line_upper.startswith("SCORE:") or line_upper.startswith("SCORE"):
                if current_section:
                    result[current_section] = " ".join(section_content).strip()
                current_section = None
                # Extract score - try multiple formats
                score_text = line_clean.split(":", 1)[1].strip() if ":" in line_clean else line_clean
                parsed_score = self._extract_score(score_text, rubric)
                if parsed_score is not None:
                    result["score"] = parsed_score
            elif line_upper.startswith("JUSTIFICATION:") or line_upper.startswith("JUSTIFICATION"):
                if current_section:
                    result[current_section] = " ".join(section_content).strip()
                current_section = "justification"
                content = line_clean.split(":", 1)[1].strip() if ":" in line_clean else ""
                section_content = [content]
            elif current_section:
                section_content.append(line)

        # Save last section
        if current_section:
            result[current_section] = " ".join(section_content).strip()

        # If we still have default score, try to find any number in the response
        if result["score"] == 5.0:
            fallback_score = self._extract_score_fallback(response_text, rubric)
            if fallback_score is not None:
                result["score"] = fallback_score

        return result

    def _extract_score(self, score_text: str, rubric: Dict[str, Any]) -> Optional[float]:
        """Extract a numeric score from text."""
        import re

        if not score_text:
            return None

        min_score = rubric.get("scale_min", 0)
        max_score = rubric.get("scale_max", 10)

        try:
            # Remove common formatting
            score_text = score_text.strip()

            # Handle "8/10" format
            if "/" in score_text:
                score_text = score_text.split("/")[0].strip()

            # Handle "8 out of 10" format
            if " out of " in score_text.lower():
                score_text = score_text.lower().split(" out of ")[0].strip()

            # Handle "(8)" or "[8]" format
            score_text = re.sub(r'[\(\)\[\]]', '', score_text)

            # Extract first number found
            match = re.search(r'(\d+\.?\d*)', score_text)
            if match:
                score = float(match.group(1))
                # Clamp to valid range
                return max(min_score, min(max_score, score))

        except (ValueError, IndexError, AttributeError):
            pass

        return None

    def _extract_score_fallback(self, response_text: str, rubric: Dict[str, Any]) -> Optional[float]:
        """Fallback: try to find a score pattern anywhere in the response."""
        import re

        min_score = rubric.get("scale_min", 0)
        max_score = rubric.get("scale_max", 10)

        # Look for common patterns like "Score: 8", "8/10", "rating: 8"
        patterns = [
            r'[Ss]core[:\s]+(\d+\.?\d*)',
            r'[Rr]ating[:\s]+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*/\s*10',
            r'(\d+\.?\d*)\s+out\s+of\s+10',
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text)
            if match:
                try:
                    score = float(match.group(1))
                    if min_score <= score <= max_score:
                        return score
                except (ValueError, IndexError):
                    continue

        return None

    def compare(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        provider_a: str,
        provider_b: str
    ) -> Dict[str, Any]:
        """
        Compare two responses head-to-head.

        Args:
            prompt: Original prompt
            response_a: First response text
            response_b: Second response text
            provider_a: Name of provider A
            provider_b: Name of provider B

        Returns:
            Dictionary with winner, confidence, and reasoning
        """
        compare_prompt = f"""You are comparing two AI responses to the same prompt.

ORIGINAL PROMPT:
{prompt}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Which response better answers the prompt? Consider:
- Accuracy and correctness
- Completeness and depth
- Clarity and coherence
- Relevance to the question

Provide your analysis and verdict.

OUTPUT FORMAT (you must follow this exactly):
ANALYSIS: [Your comparison analysis]
VERDICT: [A, B, or TIE]
CONFIDENCE: [A number from 0.0 to 1.0]
REASONING: [Brief explanation of verdict]"""

        result = self.generate(compare_prompt)
        return self._parse_comparison_response(result.text, provider_a, provider_b)

    def _parse_comparison_response(
        self,
        response_text: str,
        provider_a: str,
        provider_b: str
    ) -> Dict[str, Any]:
        """Parse a comparison response."""
        result = {
            "winner": "tie",
            "winner_provider": None,
            "confidence": 0.5,
            "analysis": "",
            "reasoning": "",
            "raw_response": response_text,
        }

        lines = response_text.strip().split("\n")
        current_section = None
        section_content = []

        for line in lines:
            line_upper = line.upper().strip()
            if line_upper.startswith("ANALYSIS:"):
                if current_section:
                    result[current_section] = " ".join(section_content).strip()
                current_section = "analysis"
                section_content = [line.split(":", 1)[1].strip() if ":" in line else ""]
            elif line_upper.startswith("VERDICT:"):
                if current_section:
                    result[current_section] = " ".join(section_content).strip()
                current_section = None
                verdict_text = line.split(":", 1)[1].strip().upper() if ":" in line else ""
                if "A" in verdict_text and "B" not in verdict_text:
                    result["winner"] = "a"
                    result["winner_provider"] = provider_a
                elif "B" in verdict_text and "A" not in verdict_text:
                    result["winner"] = "b"
                    result["winner_provider"] = provider_b
                else:
                    result["winner"] = "tie"
                    result["winner_provider"] = None
            elif line_upper.startswith("CONFIDENCE:"):
                if current_section:
                    result[current_section] = " ".join(section_content).strip()
                current_section = None
                conf_text = line.split(":", 1)[1].strip() if ":" in line else ""
                try:
                    result["confidence"] = float(conf_text)
                    result["confidence"] = max(0.0, min(1.0, result["confidence"]))
                except ValueError:
                    logger.warning(f"Could not parse confidence from: {conf_text}")
            elif line_upper.startswith("REASONING:"):
                if current_section:
                    result[current_section] = " ".join(section_content).strip()
                current_section = "reasoning"
                section_content = [line.split(":", 1)[1].strip() if ":" in line else ""]
            elif current_section:
                section_content.append(line)

        if current_section:
            result[current_section] = " ".join(section_content).strip()

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_id}, status={self.status.value})"
