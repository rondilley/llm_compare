"""Ranking engine - Bradley-Terry model and score aggregation."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import math

from ..utils.logging import get_logger
from ..config import EvaluationWeights

logger = get_logger(__name__)


@dataclass
class RankedResponse:
    """A response with its final ranking."""
    rank: int
    provider: str
    score: float
    confidence_interval: Tuple[float, float]
    score_breakdown: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confidence_interval"] = list(self.confidence_interval)
        return d


@dataclass
class FinalRankings:
    """Final rankings with all metadata."""
    rankings: List[RankedResponse] = field(default_factory=list)
    methodology: str = "bradley_terry_weighted"
    confidence_level: float = 0.95

    def get_winner(self) -> Optional[str]:
        return self.rankings[0].provider if self.rankings else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rankings": [r.to_dict() for r in self.rankings],
            "methodology": self.methodology,
            "confidence_level": self.confidence_level,
        }


class RankingEngine:
    """Computes final rankings using Bradley-Terry model and weighted aggregation."""

    def __init__(self, weights: Optional[EvaluationWeights] = None):
        """
        Initialize the ranking engine.

        Args:
            weights: Evaluation phase weights (uses defaults if not specified)
        """
        self.weights = weights or EvaluationWeights()

    def compute_rankings(
        self,
        pointwise_scores: Dict[str, float],
        pairwise_win_rates: Dict[str, float],
        adversarial_scores: Dict[str, float],
        consensus_scores: Dict[str, float],
        pairwise_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> FinalRankings:
        """
        Compute final rankings from all evaluation phases.

        Args:
            pointwise_scores: Aggregated pointwise scores per provider
            pairwise_win_rates: Win rates from pairwise comparisons
            adversarial_scores: Scores from adversarial debates
            consensus_scores: Weighted scores from collaborative consensus
            pairwise_matrix: Optional win matrix for Bradley-Terry refinement

        Returns:
            FinalRankings with ordered results
        """
        providers = list(pointwise_scores.keys())

        # Normalize all scores to 0-1 range
        norm_pointwise = self._normalize_scores(pointwise_scores, scale=10)
        norm_pairwise = pairwise_win_rates  # Already 0-1
        norm_adversarial = self._normalize_scores(adversarial_scores, scale=10)
        norm_consensus = consensus_scores  # Already 0-1

        # Compute weighted aggregate scores
        aggregate_scores = {}
        for provider in providers:
            score = (
                self.weights.pointwise * norm_pointwise.get(provider, 0.5) +
                self.weights.pairwise * norm_pairwise.get(provider, 0.5) +
                self.weights.adversarial * norm_adversarial.get(provider, 0.5) +
                self.weights.collaborative * norm_consensus.get(provider, 0.5)
            )
            aggregate_scores[provider] = score

        # Apply Bradley-Terry refinement if pairwise matrix available
        if pairwise_matrix:
            bt_scores = self._bradley_terry(pairwise_matrix, providers)
            # Blend Bradley-Terry with aggregate (70/30)
            for provider in providers:
                aggregate_scores[provider] = (
                    0.7 * aggregate_scores[provider] +
                    0.3 * bt_scores.get(provider, 0.5)
                )

        # Sort by score
        sorted_providers = sorted(
            aggregate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Build rankings
        rankings = FinalRankings()
        for rank, (provider, score) in enumerate(sorted_providers, 1):
            # Calculate confidence interval (simplified)
            ci = self._calculate_confidence_interval(score, n_comparisons=10)

            # Score breakdown
            breakdown = {
                "pointwise": pointwise_scores.get(provider, 0) / 10,
                "pairwise_winrate": pairwise_win_rates.get(provider, 0.5),
                "debate": adversarial_scores.get(provider, 5) / 10,
                "consensus": consensus_scores.get(provider, 0.5),
                "weighted_total": score,
            }

            rankings.rankings.append(RankedResponse(
                rank=rank,
                provider=provider,
                score=score * 10,  # Scale back to 0-10
                confidence_interval=(ci[0] * 10, ci[1] * 10),
                score_breakdown=breakdown,
            ))

        logger.info(
            f"Final rankings: {[(r.provider, r.score) for r in rankings.rankings]}"
        )

        return rankings

    def _normalize_scores(
        self,
        scores: Dict[str, float],
        scale: float = 10.0,
    ) -> Dict[str, float]:
        """
        Normalize scores to 0-1 range.

        Args:
            scores: Raw scores
            scale: Maximum value of raw scores

        Returns:
            Normalized scores
        """
        if not scores:
            return {}

        return {
            provider: score / scale
            for provider, score in scores.items()
        }

    def _bradley_terry(
        self,
        win_matrix: Dict[str, Dict[str, float]],
        providers: List[str],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> Dict[str, float]:
        """
        Compute Bradley-Terry scores from pairwise comparison matrix.

        Uses iterative maximum likelihood estimation.

        Args:
            win_matrix: Matrix of win rates, win_matrix[a][b] = P(a beats b)
            providers: List of provider names
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance

        Returns:
            Bradley-Terry strength scores (normalized to 0-1)
        """
        n = len(providers)
        if n < 2:
            return {p: 0.5 for p in providers}

        # Initialize strengths
        strengths = {p: 1.0 for p in providers}

        for iteration in range(max_iterations):
            old_strengths = strengths.copy()

            for i, p_i in enumerate(providers):
                numerator = 0.0
                denominator = 0.0

                for j, p_j in enumerate(providers):
                    if i == j:
                        continue

                    # Get win count (treating win_matrix as win probability)
                    w_ij = win_matrix.get(p_i, {}).get(p_j, 0.5)
                    w_ji = win_matrix.get(p_j, {}).get(p_i, 0.5)

                    # Number of games (assume 1 for each comparison)
                    n_ij = 1

                    numerator += w_ij * n_ij
                    if old_strengths[p_i] + old_strengths[p_j] > 0:
                        denominator += n_ij / (old_strengths[p_i] + old_strengths[p_j])

                if denominator > 0:
                    strengths[p_i] = numerator / denominator
                else:
                    strengths[p_i] = 1.0

            # Normalize (sum to n for stability)
            total = sum(strengths.values())
            if total > 0:
                strengths = {p: s * n / total for p, s in strengths.items()}

            # Check convergence
            max_change = max(
                abs(strengths[p] - old_strengths[p])
                for p in providers
            )
            if max_change < tolerance:
                logger.debug(f"Bradley-Terry converged in {iteration + 1} iterations")
                break

        # Normalize to 0-1 range
        min_s = min(strengths.values())
        max_s = max(strengths.values())
        range_s = max_s - min_s

        if range_s > 0:
            return {
                p: (s - min_s) / range_s
                for p, s in strengths.items()
            }
        else:
            return {p: 0.5 for p in providers}

    def _calculate_confidence_interval(
        self,
        score: float,
        n_comparisons: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for a score.

        Uses simplified normal approximation.

        Args:
            score: Point estimate
            n_comparisons: Number of comparisons used
            confidence: Confidence level (default 95%)

        Returns:
            Tuple of (lower, upper) bounds
        """
        # Z-score for confidence level
        z = 1.96 if confidence == 0.95 else 1.645

        # Estimate standard error
        # Simplified: assume variance proportional to score*(1-score)
        variance = score * (1 - score) if 0 < score < 1 else 0.25
        se = math.sqrt(variance / max(n_comparisons, 1))

        margin = z * se
        lower = max(0, score - margin)
        upper = min(1, score + margin)

        return (lower, upper)
