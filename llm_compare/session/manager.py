"""Session manager - orchestrates the complete evaluation pipeline."""

import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List

from ..config import Config, default_config
from ..providers.base import LLMProvider, Response
from ..providers.discovery import ProviderDiscovery
from ..evaluation.rubrics import RubricSet, default_rubrics
from ..evaluation.pointwise import PointwiseEvaluator, PointwiseResults
from ..evaluation.pairwise import PairwiseEvaluator, PairwiseResults
from ..evaluation.adversarial import AdversarialDebate, AdversarialResults
from ..evaluation.collaborative import CollaborativeConsensus, ConsensusResults
from ..evaluation.ranking import RankingEngine, FinalRankings
from ..utils.logging import get_logger, create_session_logger
from .storage import SessionStorage

logger = get_logger(__name__)


class SessionStatus(Enum):
    CREATED = "created"
    COLLECTING = "collecting"
    EVALUATING = "evaluating"
    RANKING = "ranking"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SessionMetadata:
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    provider_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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

    def __init__(self, config: Optional[Config] = None, rubric_set: Optional[RubricSet] = None):
        self.config = config or default_config
        self.rubric_set = rubric_set or default_rubrics
        self.storage = SessionStorage(self.config.output_dir)
        self.providers: Dict[str, LLMProvider] = {}
        self._discovery = ProviderDiscovery(self.config)

    def discover_providers(self) -> List[str]:
        self.providers = self._discovery.discover()
        return list(self.providers.keys())

    def create_session(self, prompt: str) -> Session:
        session_id = str(uuid.uuid4())[:8]
        return Session(session_id=session_id, prompt=prompt, created_at=datetime.utcnow(), config=self.config)

    def run_session(self, session: Session, skip_phases: Optional[List[str]] = None) -> Session:
        """Run a complete evaluation session through all phases."""
        skip_phases = skip_phases or []
        slog = create_session_logger(self.storage.get_session_dir(session.session_id), session.session_id)
        start_time = datetime.utcnow()

        try:
            if not self.providers:
                self.discover_providers()
            if len(self.providers) < 2:
                raise ValueError(f"Need at least 2 providers, found {len(self.providers)}")

            # Phase 0: Collect responses
            slog.info("=== Starting Phase: Response Collection ===")
            session.status = SessionStatus.COLLECTING
            session.responses = {}
            for name, provider in self.providers.items():
                try:
                    response = provider.generate(session.prompt)
                    session.responses[name] = response
                except Exception:
                    logger.error(f"Provider {name} failed")
            self.storage.save_responses(session.session_id, {k: v.to_dict() for k, v in session.responses.items()})

            session.status = SessionStatus.EVALUATING

            # Phase 1: Pointwise
            if "pointwise" not in skip_phases:
                slog.info("=== Starting Phase: Pointwise Evaluation ===")
                session.pointwise_results = PointwiseEvaluator(self.providers, self.rubric_set).evaluate(
                    session.prompt, session.responses
                )
                self.storage.save_intermediate(session.session_id, "pointwise", session.pointwise_results.to_dict())

            # Phase 2: Pairwise
            if "pairwise" not in skip_phases:
                slog.info("=== Starting Phase: Pairwise Comparison ===")
                session.pairwise_results = PairwiseEvaluator(self.providers).evaluate(session.prompt, session.responses)
                self.storage.save_intermediate(session.session_id, "pairwise", session.pairwise_results.to_dict())

            # Phase 3: Adversarial
            if "adversarial" not in skip_phases:
                slog.info("=== Starting Phase: Adversarial Debate ===")
                session.adversarial_results = AdversarialDebate(self.providers, self.config.debate_rounds).evaluate(
                    session.prompt, session.responses
                )
                self.storage.save_intermediate(session.session_id, "adversarial", session.adversarial_results.to_dict())

            # Phase 4: Collaborative
            if "collaborative" not in skip_phases:
                slog.info("=== Starting Phase: Collaborative Consensus ===")
                session.consensus_results = CollaborativeConsensus(self.providers).evaluate(
                    session.prompt, session.responses
                )
                self.storage.save_intermediate(session.session_id, "collaborative", session.consensus_results.to_dict())

            # Phase 5: Ranking
            slog.info("=== Starting Phase: Final Ranking ===")
            session.status = SessionStatus.RANKING
            session.final_rankings = self._compute_rankings(session)

            session.status = SessionStatus.COMPLETED
            session.completed_at = datetime.utcnow()
            session.metadata.total_duration_ms = int((session.completed_at - start_time).total_seconds() * 1000)
            self.storage.save_session(session.session_id, session.to_dict())
            return session

        except Exception as e:
            session.status = SessionStatus.FAILED
            session.error = str(e)
            slog.error(f"Session failed: {e}")
            self.storage.save_session(session.session_id, session.to_dict())
            raise

    def _compute_rankings(self, session: Session) -> FinalRankings:
        engine = RankingEngine(weights=self.config.evaluation_weights)

        pointwise = {p: e.overall_score for p, e in session.pointwise_results.evaluations.items()} if session.pointwise_results else {}
        pairwise_rates = session.pairwise_results.win_rates if session.pairwise_results else {}
        pairwise_matrix = session.pairwise_results.win_matrix if session.pairwise_results else None
        adversarial = session.adversarial_results.scores if session.adversarial_results else {}
        consensus = session.consensus_results.weighted_scores if session.consensus_results else {}

        return engine.compute_rankings(pointwise, pairwise_rates, adversarial, consensus, pairwise_matrix)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.storage.load_session(session_id)

    def list_sessions(self) -> List[str]:
        return self.storage.list_sessions()
