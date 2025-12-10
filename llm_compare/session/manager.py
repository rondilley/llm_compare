"""Session manager - orchestrates the complete evaluation pipeline."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..config import Config, default_config
from ..providers.base import LLMProvider, Response
from ..providers.discovery import ProviderDiscovery
from ..evaluation.rubrics import RubricSet, default_rubrics
from ..evaluation.pointwise import PointwiseEvaluator, PointwiseResults
from ..evaluation.pairwise import PairwiseEvaluator, PairwiseResults
from ..evaluation.adversarial import AdversarialDebate, AdversarialResults
from ..evaluation.collaborative import CollaborativeConsensus, ConsensusResults
from ..evaluation.ranking import RankingEngine, FinalRankings
from ..utils.logging import get_logger, SessionLogger
from .storage import SessionStorage

logger = get_logger(__name__)


class SessionStatus(Enum):
    """Status of an evaluation session."""
    CREATED = "created"
    COLLECTING = "collecting"
    EVALUATING = "evaluating"
    RANKING = "ranking"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SessionMetadata:
    """Metadata about a session."""
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    provider_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "total_duration_ms": self.total_duration_ms,
            "provider_stats": self.provider_stats,
        }


@dataclass
class Session:
    """An evaluation session."""
    session_id: str
    prompt: str
    created_at: datetime
    status: SessionStatus = SessionStatus.CREATED
    config: Optional[Config] = None
    responses: Dict[str, Response] = field(default_factory=dict)
    pointwise_results: Optional[PointwiseResults] = None
    pairwise_results: Optional[PairwiseResults] = None
    adversarial_results: Optional[AdversarialResults] = None
    consensus_results: Optional[ConsensusResults] = None
    final_rankings: Optional[FinalRankings] = None
    metadata: SessionMetadata = field(default_factory=SessionMetadata)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "responses": {
                name: resp.to_dict() for name, resp in self.responses.items()
            },
            "evaluations": {
                "pointwise": self.pointwise_results.to_dict() if self.pointwise_results else None,
                "pairwise": self.pairwise_results.to_dict() if self.pairwise_results else None,
                "adversarial": self.adversarial_results.to_dict() if self.adversarial_results else None,
                "collaborative": self.consensus_results.to_dict() if self.consensus_results else None,
            },
            "rankings": self.final_rankings.to_dict() if self.final_rankings else None,
            "metadata": self.metadata.to_dict(),
            "error": self.error,
        }


class SessionManager:
    """Manages evaluation sessions and orchestrates the pipeline."""

    def __init__(
        self,
        config: Optional[Config] = None,
        rubric_set: Optional[RubricSet] = None,
    ):
        """
        Initialize the session manager.

        Args:
            config: Configuration (uses defaults if not specified)
            rubric_set: Evaluation rubrics (uses defaults if not specified)
        """
        self.config = config or default_config
        self.rubric_set = rubric_set or default_rubrics
        self.storage = SessionStorage(self.config.output_dir)
        self.providers: Dict[str, LLMProvider] = {}
        self._discovery = ProviderDiscovery(self.config)

    def discover_providers(self) -> List[str]:
        """
        Discover and initialize available providers.

        Returns:
            List of available provider names
        """
        self.providers = self._discovery.discover()
        return list(self.providers.keys())

    def create_session(self, prompt: str) -> Session:
        """
        Create a new evaluation session.

        Args:
            prompt: The prompt to evaluate

        Returns:
            New Session object
        """
        session_id = str(uuid.uuid4())[:8]
        session = Session(
            session_id=session_id,
            prompt=prompt,
            created_at=datetime.utcnow(),
            config=self.config,
        )
        logger.info(f"Created session: {session_id}")
        return session

    def run_session(
        self,
        session: Session,
        skip_phases: Optional[List[str]] = None,
    ) -> Session:
        """
        Run a complete evaluation session.

        Args:
            session: Session to run
            skip_phases: Optional list of phases to skip

        Returns:
            Completed session
        """
        skip_phases = skip_phases or []
        session_logger = SessionLogger(
            self.storage.get_session_dir(session.session_id),
            session.session_id
        )
        start_time = datetime.utcnow()

        try:
            # Ensure providers are discovered
            if not self.providers:
                self.discover_providers()

            if len(self.providers) < 2:
                raise ValueError(
                    f"Need at least 2 providers, found {len(self.providers)}: "
                    f"{list(self.providers.keys())}"
                )

            # Phase 0: Collect responses
            session_logger.phase_start("Response Collection")
            session.status = SessionStatus.COLLECTING
            session.responses = self._collect_responses(session.prompt, session_logger)
            self.storage.save_responses(
                session.session_id,
                {name: resp.to_dict() for name, resp in session.responses.items()}
            )
            session_logger.phase_end("Response Collection")

            # Phase 1: Pointwise evaluation
            session.status = SessionStatus.EVALUATING
            if "pointwise" not in skip_phases:
                session_logger.phase_start("Pointwise Evaluation")
                session.pointwise_results = self._run_pointwise(
                    session.prompt, session.responses, session_logger
                )
                self.storage.save_intermediate(
                    session.session_id, "pointwise",
                    session.pointwise_results.to_dict()
                )
                session_logger.phase_end("Pointwise Evaluation")

            # Phase 2: Pairwise comparison
            if "pairwise" not in skip_phases:
                session_logger.phase_start("Pairwise Comparison")
                session.pairwise_results = self._run_pairwise(
                    session.prompt, session.responses, session_logger
                )
                self.storage.save_intermediate(
                    session.session_id, "pairwise",
                    session.pairwise_results.to_dict()
                )
                session_logger.phase_end("Pairwise Comparison")

            # Phase 3: Adversarial debate
            if "adversarial" not in skip_phases:
                session_logger.phase_start("Adversarial Debate")
                session.adversarial_results = self._run_adversarial(
                    session.prompt, session.responses, session_logger
                )
                self.storage.save_intermediate(
                    session.session_id, "adversarial",
                    session.adversarial_results.to_dict()
                )
                session_logger.phase_end("Adversarial Debate")

            # Phase 4: Collaborative consensus
            if "collaborative" not in skip_phases:
                session_logger.phase_start("Collaborative Consensus")
                session.consensus_results = self._run_collaborative(
                    session.prompt, session.responses, session_logger
                )
                self.storage.save_intermediate(
                    session.session_id, "collaborative",
                    session.consensus_results.to_dict()
                )
                session_logger.phase_end("Collaborative Consensus")

            # Phase 5: Final ranking
            session_logger.phase_start("Final Ranking")
            session.status = SessionStatus.RANKING
            session.final_rankings = self._compute_rankings(session)
            session_logger.phase_end("Final Ranking")

            # Complete
            session.status = SessionStatus.COMPLETED
            session.completed_at = datetime.utcnow()
            session.metadata.total_duration_ms = int(
                (session.completed_at - start_time).total_seconds() * 1000
            )

            # Save final session data
            self.storage.save_session(session.session_id, session.to_dict())
            session_logger.info(
                f"Session completed in {session.metadata.total_duration_ms}ms"
            )

            return session

        except Exception as e:
            session.status = SessionStatus.FAILED
            session.error = str(e)
            session_logger.error(f"Session failed: {e}")
            logger.error(f"Session {session.session_id} failed: {e}")
            # Save partial results
            self.storage.save_session(session.session_id, session.to_dict())
            raise

    def _collect_responses(
        self,
        prompt: str,
        session_logger: SessionLogger
    ) -> Dict[str, Response]:
        """Collect responses from all providers."""
        responses = {}

        for name, provider in self.providers.items():
            try:
                session_logger.provider_call(name, f"Generating response")
                response = provider.generate(prompt)
                responses[name] = response
                session_logger.info(
                    f"Response from {name}: {len(response.text)} chars, "
                    f"{response.latency_ms}ms"
                )
            except Exception as e:
                session_logger.error(f"Failed to get response from {name}: {e}")
                logger.error(f"Provider {name} failed: {e}")

        return responses

    def _run_pointwise(
        self,
        prompt: str,
        responses: Dict[str, Response],
        session_logger: SessionLogger
    ) -> PointwiseResults:
        """Run pointwise evaluation phase."""
        evaluator = PointwiseEvaluator(
            providers=self.providers,
            rubric_set=self.rubric_set,
        )
        results = evaluator.evaluate(prompt, responses)

        for provider, eval_result in results.evaluations.items():
            session_logger.info(
                f"Pointwise score for {provider}: {eval_result.overall_score:.2f}"
            )

        return results

    def _run_pairwise(
        self,
        prompt: str,
        responses: Dict[str, Response],
        session_logger: SessionLogger
    ) -> PairwiseResults:
        """Run pairwise comparison phase."""
        evaluator = PairwiseEvaluator(providers=self.providers)
        results = evaluator.evaluate(prompt, responses)

        for provider, win_rate in results.win_rates.items():
            session_logger.info(f"Pairwise win rate for {provider}: {win_rate:.2f}")

        return results

    def _run_adversarial(
        self,
        prompt: str,
        responses: Dict[str, Response],
        session_logger: SessionLogger
    ) -> AdversarialResults:
        """Run adversarial debate phase."""
        debate = AdversarialDebate(
            providers=self.providers,
            num_rounds=self.config.debate_rounds,
        )
        results = debate.evaluate(prompt, responses)

        for provider, score in results.scores.items():
            session_logger.info(f"Debate score for {provider}: {score:.2f}")

        return results

    def _run_collaborative(
        self,
        prompt: str,
        responses: Dict[str, Response],
        session_logger: SessionLogger
    ) -> ConsensusResults:
        """Run collaborative consensus phase."""
        consensus = CollaborativeConsensus(providers=self.providers)
        results = consensus.evaluate(prompt, responses)

        for provider, score in results.weighted_scores.items():
            session_logger.info(f"Consensus score for {provider}: {score:.2f}")

        return results

    def _compute_rankings(self, session: Session) -> FinalRankings:
        """Compute final rankings from all evaluation phases."""
        engine = RankingEngine(weights=self.config.evaluation_weights)

        # Gather scores from each phase
        pointwise_scores = {}
        if session.pointwise_results:
            for provider, eval_result in session.pointwise_results.evaluations.items():
                pointwise_scores[provider] = eval_result.overall_score

        pairwise_win_rates = {}
        pairwise_matrix = None
        if session.pairwise_results:
            pairwise_win_rates = session.pairwise_results.win_rates
            pairwise_matrix = session.pairwise_results.win_matrix

        adversarial_scores = {}
        if session.adversarial_results:
            adversarial_scores = session.adversarial_results.scores

        consensus_scores = {}
        if session.consensus_results:
            consensus_scores = session.consensus_results.weighted_scores

        return engine.compute_rankings(
            pointwise_scores=pointwise_scores,
            pairwise_win_rates=pairwise_win_rates,
            adversarial_scores=adversarial_scores,
            consensus_scores=consensus_scores,
            pairwise_matrix=pairwise_matrix,
        )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a saved session."""
        return self.storage.load_session(session_id)

    def list_sessions(self) -> List[str]:
        """List all saved sessions."""
        return self.storage.list_sessions()
