"""Pointwise evaluation - each response scored independently against rubrics."""

from dataclasses import dataclass, field, asdict
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
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class PointwiseEvaluation:
    """Complete pointwise evaluation for a single response."""
    evaluated: str
    scores: Dict[str, List[PointwiseScore]] = field(default_factory=dict)
    aggregated_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0

    def add_score(self, score: PointwiseScore) -> None:
        if score.rubric_name not in self.scores:
            self.scores[score.rubric_name] = []
        self.scores[score.rubric_name].append(score)

    def aggregate(self, rubric_set: RubricSet) -> None:
        self.aggregated_scores = {}
        weighted_total = total_weight = 0.0

        for rubric in rubric_set:
            if rubric.name in self.scores and self.scores[rubric.name]:
                avg = sum(s.score for s in self.scores[rubric.name]) / len(self.scores[rubric.name])
                self.aggregated_scores[rubric.name] = avg
                weighted_total += avg * rubric.weight
                total_weight += rubric.weight

        self.overall_score = weighted_total / total_weight if total_weight > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluated": self.evaluated,
            "scores": {k: [s.to_dict() for s in v] for k, v in self.scores.items()},
            "aggregated_scores": self.aggregated_scores,
            "overall_score": self.overall_score,
        }


@dataclass
class PointwiseResults:
    """Results from pointwise evaluation phase."""
    prompt: str
    evaluations: Dict[str, PointwiseEvaluation] = field(default_factory=dict)

    def get_rankings(self) -> List[tuple]:
        return sorted(
            [(p, e.overall_score) for p, e in self.evaluations.items()],
            key=lambda x: x[1], reverse=True
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "evaluations": {k: v.to_dict() for k, v in self.evaluations.items()},
        }


class PointwiseEvaluator:
    """Orchestrates pointwise evaluation of responses."""

    def __init__(self, providers: Dict[str, LLMProvider], rubric_set: Optional[RubricSet] = None):
        self.providers = providers
        self.rubric_set = rubric_set or default_rubrics

    def evaluate(self, prompt: str, responses: Dict[str, Response]) -> PointwiseResults:
        """Evaluate all responses using pointwise scoring. Each provider evaluates others."""
        results = PointwiseResults(prompt=prompt)
        for name in responses:
            results.evaluations[name] = PointwiseEvaluation(evaluated=name)

        for evaluator_name, evaluator in self.providers.items():
            for response_provider, response in responses.items():
                if evaluator_name == response_provider:
                    continue
                for rubric in self.rubric_set:
                    try:
                        score = self._evaluate_single(evaluator, evaluator_name, prompt, response, response_provider, rubric)
                        results.evaluations[response_provider].add_score(score)
                    except Exception as e:
                        logger.error(f"Evaluation failed: {evaluator_name} -> {response_provider} [{rubric.name}]")

        for evaluation in results.evaluations.values():
            evaluation.aggregate(self.rubric_set)

        return results

    def _evaluate_single(self, evaluator: LLMProvider, evaluator_name: str, prompt: str,
                         response: Response, response_provider: str, rubric: Rubric) -> PointwiseScore:
        result = evaluator.evaluate(prompt=prompt, response_text=response.text, rubric=rubric.to_dict())
        return PointwiseScore(
            evaluator=evaluator_name,
            evaluated=response_provider,
            rubric_name=rubric.name,
            score=result["score"],
            reasoning=result.get("reasoning", ""),
            justification=result.get("justification", ""),
        )
