"""Base provider interface for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ProviderStatus(Enum):
    """Status of an LLM provider."""
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
    raw_response: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "text": self.text,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "finish_reason": self.finish_reason,
            "timestamp": self.timestamp.isoformat(),
        }


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        api_key: str,
        model_id: Optional[str] = None,
        timeout: int = 120,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """
        Initialize the provider.

        Args:
            api_key: API key for authentication
            model_id: Model identifier (uses default if not specified)
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
        """
        self.api_key = api_key
        self._model_id = model_id
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.status = ProviderStatus.HEALTHY
        self._client = None

        # Initialize the client
        self._create_client()

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model ID for this provider."""
        pass

    @property
    def model_id(self) -> str:
        """Current model ID."""
        return self._model_id or self.default_model

    def discover_models(self) -> List[str]:
        """
        Discover available models from the provider's API.

        Returns:
            List of available model IDs
        """
        return []

    def select_best_model(self) -> str:
        """
        Select the best available model from the provider.

        This method queries the API for available models and selects
        the best one based on provider-specific preferences.

        Returns:
            The model ID of the best available model
        """
        available = self.discover_models()
        if not available:
            logger.warning(f"{self.name}: No models discovered, using default: {self.default_model}")
            return self.default_model

        best = self._rank_models(available)
        if best:
            logger.info(f"{self.name}: Selected best model: {best}")
            return best

        return self.default_model

    def _rank_models(self, models: List[str]) -> Optional[str]:
        """
        Rank models and return the best one.

        Override this in subclasses to implement provider-specific ranking.

        Args:
            models: List of available model IDs

        Returns:
            The best model ID, or None to use default
        """
        return None

    @abstractmethod
    def _create_client(self) -> None:
        """Create the API client. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Response:
        """
        Generate a response to a prompt.

        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional provider-specific arguments

        Returns:
            Response object with the generated text and metadata
        """
        pass

    def evaluate(self, prompt: str, response_text: str, rubric: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a response against a rubric.

        This is a convenience method that constructs an evaluation prompt
        and calls generate().

        Args:
            prompt: Original prompt that generated the response
            response_text: The response text to evaluate
            rubric: Evaluation rubric with name, description, and scale

        Returns:
            Dictionary with score and reasoning
        """
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
