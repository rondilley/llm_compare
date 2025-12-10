"""Pairwise comparison - head-to-head evaluation of responses."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import random

from ..providers.base import LLMProvider, Response
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PairwiseComparison:
    """Result of a single pairwise comparison."""
    response_a: str
    response_b: str
    judge: str
    winner: str  # "a", "b", or "tie"
    winner_provider: Optional[str]
    confidence: float
    analysis: str
    reasoning: str
    order_shown: str  # "a_first" or "b_first"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class PairwiseResults:
    """Results from pairwise comparison phase."""
    prompt: str
    comparisons: List[PairwiseComparison] = field(default_factory=list)
    win_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    win_rates: Dict[str, float] = field(default_factory=dict)

    def add_comparison(self, comparison: PairwiseComparison) -> None:
        self.comparisons.append(comparison)

    def calculate_matrix(self, providers: List[str]) -> None:
        """Calculate win matrix from comparisons."""
        self.win_matrix = {p: {q: 0.0 for q in providers if q != p} for p in providers}
        pair_results: Dict[Tuple[str, str], List[str]] = {}

        for comp in self.comparisons:
            pair = tuple(sorted([comp.response_a, comp.response_b]))
            pair_results.setdefault(pair, []).append(comp.winner_provider)

        for (a, b), winners in pair_results.items():
            total = len(winners)
            if total > 0:
                a_wins = sum(1 for w in winners if w == a)
                b_wins = sum(1 for w in winners if w == b)
                ties = sum(1 for w in winners if w is None)
                self.win_matrix[a][b] = (a_wins + 0.5 * ties) / total
                self.win_matrix[b][a] = (b_wins + 0.5 * ties) / total

    def calculate_win_rates(self, providers: List[str]) -> None:
        self.win_rates = {}
        for provider in providers:
            scores = list(self.win_matrix.get(provider, {}).values())
            self.win_rates[provider] = sum(scores) / len(scores) if scores else 0.5

    def get_rankings(self) -> List[tuple]:
        return sorted(self.win_rates.items(), key=lambda x: x[1], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "win_matrix": self.win_matrix,
            "win_rates": self.win_rates,
        }


class PairwiseEvaluator:
    """Orchestrates pairwise comparison of responses."""

    def __init__(self, providers: Dict[str, LLMProvider]):
        self.providers = providers

    def evaluate(self, prompt: str, responses: Dict[str, Response]) -> PairwiseResults:
        """Perform pairwise comparisons. Each pair judged by non-participating providers."""
        results = PairwiseResults(prompt=prompt)
        provider_names = list(responses.keys())
        pairs = [(a, b) for i, a in enumerate(provider_names) for b in provider_names[i + 1:]]

        for response_a, response_b in pairs:
            judges = [n for n in provider_names if n not in (response_a, response_b)] or provider_names

            for judge_name in judges:
                try:
                    # Randomize order to mitigate position bias
                    if random.random() < 0.5:
                        first, second, order = response_a, response_b, "a_first"
                    else:
                        first, second, order = response_b, response_a, "b_first"

                    comparison = self._compare_single(
                        self.providers[judge_name], judge_name, prompt,
                        responses[first], responses[second],
                        first, second, response_a, response_b, order
                    )
                    results.add_comparison(comparison)
                except Exception:
                    logger.error(f"Comparison failed: {judge_name} judging {response_a} vs {response_b}")

        results.calculate_matrix(provider_names)
        results.calculate_win_rates(provider_names)
        return results

    def _compare_single(self, judge: LLMProvider, judge_name: str, prompt: str,
                        resp_first: Response, resp_second: Response,
                        prov_first: str, prov_second: str,
                        orig_a: str, orig_b: str, order: str) -> PairwiseComparison:
        result = judge.compare(prompt, resp_first.text, resp_second.text, prov_first, prov_second)

        winner = result["winner"]
        if winner == "a":
            winner = "a" if order == "a_first" else "b"
        elif winner == "b":
            winner = "b" if order == "a_first" else "a"
        else:
            winner = "tie"

        winner_provider = orig_a if winner == "a" else (orig_b if winner == "b" else None)

        return PairwiseComparison(
            response_a=orig_a, response_b=orig_b, judge=judge_name,
            winner=winner, winner_provider=winner_provider,
            confidence=result["confidence"],
            analysis=result.get("analysis", ""),
            reasoning=result.get("reasoning", ""),
            order_shown=order,
        )
