"""Pairwise comparison - head-to-head evaluation of responses."""

from dataclasses import dataclass, field
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
        """Convert to dictionary for serialization."""
        return {
            "response_a": self.response_a,
            "response_b": self.response_b,
            "judge": self.judge,
            "winner": self.winner,
            "winner_provider": self.winner_provider,
            "confidence": self.confidence,
            "analysis": self.analysis,
            "reasoning": self.reasoning,
            "order_shown": self.order_shown,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PairwiseResults:
    """Results from pairwise comparison phase."""
    prompt: str
    comparisons: List[PairwiseComparison] = field(default_factory=list)
    win_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    win_rates: Dict[str, float] = field(default_factory=dict)

    def add_comparison(self, comparison: PairwiseComparison) -> None:
        """Add a comparison result."""
        self.comparisons.append(comparison)

    def calculate_matrix(self, providers: List[str]) -> None:
        """
        Calculate win matrix from comparisons.

        For each pair (A, B), the matrix[A][B] contains:
        - 1.0 if A consistently beats B
        - 0.5 if tied
        - 0.0 if B consistently beats A
        """
        # Initialize matrix
        self.win_matrix = {p: {q: 0.0 for q in providers if q != p} for p in providers}

        # Count wins for each pair
        pair_results: Dict[Tuple[str, str], List[str]] = {}

        for comp in self.comparisons:
            pair = tuple(sorted([comp.response_a, comp.response_b]))
            if pair not in pair_results:
                pair_results[pair] = []
            pair_results[pair].append(comp.winner_provider)

        # Calculate win rates for each pair
        for (a, b), winners in pair_results.items():
            a_wins = sum(1 for w in winners if w == a)
            b_wins = sum(1 for w in winners if w == b)
            ties = sum(1 for w in winners if w is None)
            total = len(winners)

            if total > 0:
                # A's score against B: wins + 0.5*ties
                a_score = (a_wins + 0.5 * ties) / total
                b_score = (b_wins + 0.5 * ties) / total

                self.win_matrix[a][b] = a_score
                self.win_matrix[b][a] = b_score

    def calculate_win_rates(self, providers: List[str]) -> None:
        """Calculate overall win rate for each provider."""
        self.win_rates = {}

        for provider in providers:
            if provider in self.win_matrix:
                scores = list(self.win_matrix[provider].values())
                if scores:
                    self.win_rates[provider] = sum(scores) / len(scores)
                else:
                    self.win_rates[provider] = 0.5
            else:
                self.win_rates[provider] = 0.5

    def get_rankings(self) -> List[tuple]:
        """Get providers ranked by win rate."""
        return sorted(self.win_rates.items(), key=lambda x: x[1], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "win_matrix": self.win_matrix,
            "win_rates": self.win_rates,
        }


class PairwiseEvaluator:
    """Orchestrates pairwise comparison of responses."""

    def __init__(self, providers: Dict[str, LLMProvider]):
        """
        Initialize the pairwise evaluator.

        Args:
            providers: Dictionary of provider name to provider instance
        """
        self.providers = providers

    def evaluate(
        self,
        prompt: str,
        responses: Dict[str, Response],
    ) -> PairwiseResults:
        """
        Perform pairwise comparisons between all responses.

        Each pair is judged by providers not involved in the comparison.
        Position bias is mitigated by randomizing presentation order.

        Args:
            prompt: Original prompt
            responses: Dictionary of provider name to response

        Returns:
            PairwiseResults with all comparisons
        """
        results = PairwiseResults(prompt=prompt)
        provider_names = list(responses.keys())

        # Generate all pairs
        pairs = []
        for i, a in enumerate(provider_names):
            for b in provider_names[i + 1:]:
                pairs.append((a, b))

        logger.info(f"Performing {len(pairs)} pairwise comparisons")

        # For each pair, get judgments from non-participating providers
        for response_a, response_b in pairs:
            judges = [
                name for name in provider_names
                if name not in (response_a, response_b)
            ]

            if not judges:
                # If only 2 providers, they judge each other's comparison
                # (not ideal but necessary for 2-provider scenarios)
                judges = provider_names

            for judge_name in judges:
                judge = self.providers[judge_name]

                try:
                    # Randomize order to mitigate position bias
                    if random.random() < 0.5:
                        first, second = response_a, response_b
                        order = "a_first"
                    else:
                        first, second = response_b, response_a
                        order = "b_first"

                    comparison = self._compare_single(
                        judge=judge,
                        judge_name=judge_name,
                        prompt=prompt,
                        response_first=responses[first],
                        response_second=responses[second],
                        provider_first=first,
                        provider_second=second,
                        original_a=response_a,
                        original_b=response_b,
                        order=order,
                    )
                    results.add_comparison(comparison)

                    logger.debug(
                        f"{judge_name} judged {response_a} vs {response_b}: "
                        f"winner={comparison.winner_provider or 'tie'}"
                    )

                except Exception as e:
                    logger.error(
                        f"Comparison failed: {judge_name} judging "
                        f"{response_a} vs {response_b}: {e}"
                    )

        # Calculate matrices and rankings
        results.calculate_matrix(provider_names)
        results.calculate_win_rates(provider_names)

        logger.info(f"Pairwise win rates: {results.win_rates}")

        return results

    def _compare_single(
        self,
        judge: LLMProvider,
        judge_name: str,
        prompt: str,
        response_first: Response,
        response_second: Response,
        provider_first: str,
        provider_second: str,
        original_a: str,
        original_b: str,
        order: str,
    ) -> PairwiseComparison:
        """
        Perform a single pairwise comparison.

        Args:
            judge: Provider to use as judge
            judge_name: Name of judge
            prompt: Original prompt
            response_first: First response (as shown)
            response_second: Second response (as shown)
            provider_first: Provider of first response
            provider_second: Provider of second response
            original_a: Original "A" provider
            original_b: Original "B" provider
            order: "a_first" or "b_first"

        Returns:
            PairwiseComparison with results
        """
        result = judge.compare(
            prompt=prompt,
            response_a=response_first.text,
            response_b=response_second.text,
            provider_a=provider_first,
            provider_b=provider_second,
        )

        # Map winner back to original A/B designation
        winner = result["winner"]
        winner_provider = None

        if winner == "a":
            winner_provider = provider_first
            # Translate to original a/b
            if order == "a_first":
                winner = "a"
            else:
                winner = "b"
        elif winner == "b":
            winner_provider = provider_second
            if order == "a_first":
                winner = "b"
            else:
                winner = "a"
        else:
            winner = "tie"

        # Map winner_provider to actual provider
        if winner == "a":
            winner_provider = original_a
        elif winner == "b":
            winner_provider = original_b
        else:
            winner_provider = None

        return PairwiseComparison(
            response_a=original_a,
            response_b=original_b,
            judge=judge_name,
            winner=winner,
            winner_provider=winner_provider,
            confidence=result["confidence"],
            analysis=result.get("analysis", ""),
            reasoning=result.get("reasoning", ""),
            order_shown=order,
        )
