"""Session manager - orchestrates the complete evaluation pipeline."""

import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List

from ..config import Config, default_config, RepetitionMode
from ..providers.base import LLMProvider, Response
from ..providers.discovery import ProviderDiscovery
from ..evaluation.rubrics import RubricSet, default_rubrics
from ..evaluation.pointwise import PointwiseEvaluator, PointwiseResults
from ..evaluation.pairwise import PairwiseEvaluator, PairwiseResults
from ..evaluation.adversarial import AdversarialDebate, AdversarialResults
from ..evaluation.collaborative import CollaborativeConsensus, ConsensusResults
from ..evaluation.ranking import RankingEngine, FinalRankings
from ..prompting.repetition import (
    apply_repetition,
    detect_recommended_mode,
    get_repetition_info,
)
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
class RepetitionAnalysis:
    """Analysis of prompt repetition effects on responses."""
    mode_used: RepetitionMode = RepetitionMode.NONE
    compare_enabled: bool = False
    recommended_mode: Optional[RepetitionMode] = None
    recommendation_reason: str = ""
    # Per-provider comparison (if compare_enabled)
    provider_comparisons: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mode_used"] = self.mode_used.value
        d["recommended_mode"] = self.recommended_mode.value if self.recommended_mode else None
        return d


@dataclass
class Session:
    """An evaluation session."""
    session_id: str
    prompt: str
    created_at: datetime
    status: SessionStatus = SessionStatus.CREATED
    config: Optional[Config] = None
    responses: Dict[str, Response] = field(default_factory=dict)
    # Responses without repetition (for comparison mode)
    baseline_responses: Dict[str, Response] = field(default_factory=dict)
    pointwise_results: Optional[PointwiseResults] = None
    pairwise_results: Optional[PairwiseResults] = None
    adversarial_results: Optional[AdversarialResults] = None
    consensus_results: Optional[ConsensusResults] = None
    final_rankings: Optional[FinalRankings] = None
    metadata: SessionMetadata = field(default_factory=SessionMetadata)
    repetition_analysis: Optional[RepetitionAnalysis] = None
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
            "baseline_responses": {
                name: resp.to_dict() for name, resp in self.baseline_responses.items()
            } if self.baseline_responses else {},
            "evaluations": {
                "pointwise": self.pointwise_results.to_dict() if self.pointwise_results else None,
                "pairwise": self.pairwise_results.to_dict() if self.pairwise_results else None,
                "adversarial": self.adversarial_results.to_dict() if self.adversarial_results else None,
                "collaborative": self.consensus_results.to_dict() if self.consensus_results else None,
            },
            "rankings": self.final_rankings.to_dict() if self.final_rankings else None,
            "metadata": self.metadata.to_dict(),
            "repetition_analysis": self.repetition_analysis.to_dict() if self.repetition_analysis else None,
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

    def run_session(
        self,
        session: Session,
        skip_phases: Optional[List[str]] = None,
        repetition_mode: Optional[RepetitionMode] = None,
        compare_repetition: bool = False,
    ) -> Session:
        """
        Run a complete evaluation session through all phases.

        Args:
            session: The session to run
            skip_phases: List of phase names to skip
            repetition_mode: Repetition mode to use (overrides config)
            compare_repetition: If True, run both baseline and repeated prompts
        """
        skip_phases = skip_phases or []
        slog = create_session_logger(self.storage.get_session_dir(session.session_id), session.session_id)
        start_time = datetime.utcnow()

        # Determine repetition settings
        rep_mode = repetition_mode or self.config.repetition.mode
        do_compare = compare_repetition or self.config.repetition.compare_modes

        # Auto-detect if enabled
        if self.config.repetition.auto_detect and rep_mode == RepetitionMode.NONE:
            detected_mode, reason = detect_recommended_mode(session.prompt)
            slog.info(f"Auto-detected repetition mode: {detected_mode.value} ({reason})")
            rep_mode = detected_mode

        try:
            if not self.providers:
                self.discover_providers()
            if len(self.providers) < 2:
                raise ValueError(f"Need at least 2 providers, found {len(self.providers)}")

            # Initialize repetition analysis
            recommended, rec_reason = detect_recommended_mode(session.prompt)
            session.repetition_analysis = RepetitionAnalysis(
                mode_used=rep_mode,
                compare_enabled=do_compare,
                recommended_mode=recommended,
                recommendation_reason=rec_reason,
            )

            # Phase 0: Collect responses
            slog.info("=== Starting Phase: Response Collection ===")
            if rep_mode != RepetitionMode.NONE:
                slog.info(f"Using repetition mode: {rep_mode.value}")
            session.status = SessionStatus.COLLECTING
            session.responses = {}

            # Collect baseline responses (no repetition) if comparing
            if do_compare:
                slog.info("Collecting baseline responses (no repetition)...")
                session.baseline_responses = {}
                for name, provider in self.providers.items():
                    try:
                        response = provider.generate(
                            session.prompt,
                            repetition_mode=RepetitionMode.NONE
                        )
                        session.baseline_responses[name] = response
                    except Exception:
                        logger.error(f"Provider {name} failed (baseline)")

            # Collect main responses (with repetition if enabled)
            slog.info(f"Collecting responses with repetition mode: {rep_mode.value}")
            for name, provider in self.providers.items():
                try:
                    response = provider.generate(session.prompt, repetition_mode=rep_mode)
                    session.responses[name] = response
                except Exception:
                    logger.error(f"Provider {name} failed")

            # Analyze repetition effects if comparing
            if do_compare and session.baseline_responses:
                session.repetition_analysis.provider_comparisons = (
                    self._analyze_repetition_effects(session)
                )

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

    def _analyze_repetition_effects(self, session: Session) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the effects of prompt repetition by comparing baseline vs repeated responses.

        Returns per-provider analysis including:
        - latency_delta_ms: Change in latency
        - output_length_delta: Change in output length
        - preferred_mode: Which mode produced a longer/richer response
        """
        comparisons = {}

        for provider_name in session.responses.keys():
            if provider_name not in session.baseline_responses:
                continue

            baseline = session.baseline_responses[provider_name]
            repeated = session.responses[provider_name]

            latency_delta = repeated.latency_ms - baseline.latency_ms
            baseline_len = len(baseline.text)
            repeated_len = len(repeated.text)
            length_delta = repeated_len - baseline_len

            # Determine preferred mode based on response richness
            # (longer responses often indicate more thorough answers)
            preferred = "repeated" if repeated_len >= baseline_len else "baseline"

            comparisons[provider_name] = {
                "baseline_latency_ms": baseline.latency_ms,
                "repeated_latency_ms": repeated.latency_ms,
                "latency_delta_ms": latency_delta,
                "latency_delta_pct": (latency_delta / baseline.latency_ms * 100) if baseline.latency_ms > 0 else 0,
                "baseline_output_tokens": baseline.output_tokens,
                "repeated_output_tokens": repeated.output_tokens,
                "baseline_length": baseline_len,
                "repeated_length": repeated_len,
                "length_delta": length_delta,
                "length_delta_pct": (length_delta / baseline_len * 100) if baseline_len > 0 else 0,
                "preferred_mode": preferred,
            }

        return comparisons

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.storage.load_session(session_id)

    def list_sessions(self) -> List[str]:
        return self.storage.list_sessions()
