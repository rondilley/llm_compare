"""Pointwise evaluation - each response scored independently against rubrics."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..providers.base import LLMProvider, Response
from ..utils.logging import get_logger
from .rubrics import Rubric, RubricSet, default_rubrics

logger = get_logger(__name__)


@dataclass
class PointwiseScore:
    """Score from a single evaluator for a single response."""
    evaluator: str
    evaluated: str
    rubric_name: str
    score: float
    reasoning: str
    justification: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "evaluator": self.evaluator,
            "evaluated": self.evaluated,
            "rubric_name": self.rubric_name,
            "score": self.score,
            "reasoning": self.reasoning,
            "justification": self.justification,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PointwiseEvaluation:
    """Complete pointwise evaluation for a single response."""
    evaluated: str
    scores: Dict[str, List[PointwiseScore]] = field(default_factory=dict)
    aggregated_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0

    def add_score(self, score: PointwiseScore) -> None:
        """Add a score from an evaluator."""
        if score.rubric_name not in self.scores:
            self.scores[score.rubric_name] = []
        self.scores[score.rubric_name].append(score)

    def aggregate(self, rubric_set: RubricSet) -> None:
        """Calculate aggregated scores across evaluators."""
        self.aggregated_scores = {}
        weighted_total = 0.0
        total_weight = 0.0

        for rubric in rubric_set:
            if rubric.name in self.scores and self.scores[rubric.name]:
                scores = [s.score for s in self.scores[rubric.name]]
                avg_score = sum(scores) / len(scores)
                self.aggregated_scores[rubric.name] = avg_score
                weighted_total += avg_score * rubric.weight
                total_weight += rubric.weight

        if total_weight > 0:
            self.overall_score = weighted_total / total_weight
        else:
            self.overall_score = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "evaluated": self.evaluated,
            "scores": {
                rubric: [s.to_dict() for s in scores]
                for rubric, scores in self.scores.items()
            },
            "aggregated_scores": self.aggregated_scores,
            "overall_score": self.overall_score,
        }


@dataclass
class PointwiseResults:
    """Results from pointwise evaluation phase."""
    prompt: str
    evaluations: Dict[str, PointwiseEvaluation] = field(default_factory=dict)

    def get_evaluation(self, provider: str) -> Optional[PointwiseEvaluation]:
        """Get evaluation for a specific provider."""
        return self.evaluations.get(provider)

    def get_rankings(self) -> List[tuple]:
        """Get providers ranked by overall score."""
        rankings = [
            (provider, eval.overall_score)
            for provider, eval in self.evaluations.items()
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "evaluations": {
                provider: eval.to_dict()
                for provider, eval in self.evaluations.items()
            },
        }


class PointwiseEvaluator:
    """Orchestrates pointwise evaluation of responses."""

    def __init__(
        self,
        providers: Dict[str, LLMProvider],
        rubric_set: Optional[RubricSet] = None,
    ):
        """
        Initialize the pointwise evaluator.

        Args:
            providers: Dictionary of provider name to provider instance
            rubric_set: Evaluation rubrics (uses defaults if not specified)
        """
        self.providers = providers
        self.rubric_set = rubric_set or default_rubrics

    def evaluate(
        self,
        prompt: str,
        responses: Dict[str, Response],
    ) -> PointwiseResults:
        """
        Evaluate all responses using pointwise scoring.

        Each provider evaluates all responses except its own.

        Args:
            prompt: Original prompt
            responses: Dictionary of provider name to response

        Returns:
            PointwiseResults with all evaluations
        """
        results = PointwiseResults(prompt=prompt)

        # Initialize evaluations for each response
        for provider_name in responses:
            results.evaluations[provider_name] = PointwiseEvaluation(
                evaluated=provider_name
            )

        # Each provider evaluates all other responses
        for evaluator_name, evaluator in self.providers.items():
            logger.info(f"Evaluator {evaluator_name} starting pointwise evaluation")

            for response_provider, response in responses.items():
                # Skip self-evaluation
                if evaluator_name == response_provider:
                    continue

                # Evaluate against each rubric
                for rubric in self.rubric_set:
                    try:
                        score = self._evaluate_single(
                            evaluator=evaluator,
                            evaluator_name=evaluator_name,
                            prompt=prompt,
                            response=response,
                            response_provider=response_provider,
                            rubric=rubric,
                        )
                        results.evaluations[response_provider].add_score(score)
                        logger.debug(
                            f"{evaluator_name} -> {response_provider} "
                            f"[{rubric.name}]: {score.score}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Evaluation failed: {evaluator_name} -> {response_provider} "
                            f"[{rubric.name}]: {e}"
                        )

        # Aggregate scores for each response
        for evaluation in results.evaluations.values():
            evaluation.aggregate(self.rubric_set)
            logger.info(
                f"Aggregated score for {evaluation.evaluated}: "
                f"{evaluation.overall_score:.2f}"
            )

        return results

    def _evaluate_single(
        self,
        evaluator: LLMProvider,
        evaluator_name: str,
        prompt: str,
        response: Response,
        response_provider: str,
        rubric: Rubric,
    ) -> PointwiseScore:
        """
        Perform a single evaluation.

        Args:
            evaluator: Provider to use as evaluator
            evaluator_name: Name of evaluator
            prompt: Original prompt
            response: Response to evaluate
            response_provider: Name of response provider
            rubric: Rubric to evaluate against

        Returns:
            PointwiseScore with evaluation results
        """
        result = evaluator.evaluate(
            prompt=prompt,
            response_text=response.text,
            rubric=rubric.to_dict(),
        )

        return PointwiseScore(
            evaluator=evaluator_name,
            evaluated=response_provider,
            rubric_name=rubric.name,
            score=result["score"],
            reasoning=result.get("reasoning", ""),
            justification=result.get("justification", ""),
        )
