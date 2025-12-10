"""Collaborative consensus - multi-model discussion and voting."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..providers.base import LLMProvider, Response
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Discussion:
    """A single discussion contribution."""
    round: int
    participant: str
    content: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "round": self.round,
            "participant": self.participant,
            "content": self.content,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Insight:
    """A shared insight (strength or weakness)."""
    description: str
    supporting_providers: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "description": self.description,
            "supporting_providers": self.supporting_providers,
            "confidence": self.confidence,
        }


@dataclass
class Disagreement:
    """A point of disagreement between evaluators."""
    topic: str
    positions: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "topic": self.topic,
            "positions": self.positions,
        }


@dataclass
class Vote:
    """A ranking vote from a single provider."""
    voter: str
    rankings: List[str]  # Ordered list of provider names, best first
    confidence: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "voter": self.voter,
            "rankings": self.rankings,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class ConsensusResults:
    """Results from collaborative consensus phase."""
    prompt: str
    discussions: List[Discussion] = field(default_factory=list)
    shared_strengths: List[Insight] = field(default_factory=list)
    shared_weaknesses: List[Insight] = field(default_factory=list)
    disagreements: List[Disagreement] = field(default_factory=list)
    votes: List[Vote] = field(default_factory=list)
    synthesis: str = ""
    weighted_scores: Dict[str, float] = field(default_factory=dict)

    def calculate_weighted_scores(self, providers: List[str]) -> None:
        """Calculate weighted scores from votes."""
        scores = {p: 0.0 for p in providers}
        total_weight = 0.0

        for vote in self.votes:
            weight = vote.confidence
            total_weight += weight

            # Assign points based on ranking (Borda count style)
            n = len(vote.rankings)
            for i, provider in enumerate(vote.rankings):
                if provider in scores:
                    points = (n - i) / n  # 1.0 for first, decreasing
                    scores[provider] += points * weight

        if total_weight > 0:
            for provider in scores:
                scores[provider] /= total_weight

        self.weighted_scores = scores

    def get_rankings(self) -> List[tuple]:
        """Get providers ranked by weighted score."""
        return sorted(self.weighted_scores.items(), key=lambda x: x[1], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "discussions": [d.to_dict() for d in self.discussions],
            "shared_strengths": [s.to_dict() for s in self.shared_strengths],
            "shared_weaknesses": [w.to_dict() for w in self.shared_weaknesses],
            "disagreements": [d.to_dict() for d in self.disagreements],
            "votes": [v.to_dict() for v in self.votes],
            "synthesis": self.synthesis,
            "weighted_scores": self.weighted_scores,
        }


class CollaborativeConsensus:
    """Orchestrates collaborative consensus building."""

    def __init__(
        self,
        providers: Dict[str, LLMProvider],
        num_rounds: int = 2,
    ):
        """
        Initialize the collaborative consensus evaluator.

        Args:
            providers: Dictionary of provider name to provider instance
            num_rounds: Number of discussion rounds (default 2)
        """
        self.providers = providers
        self.num_rounds = num_rounds

    def evaluate(
        self,
        prompt: str,
        responses: Dict[str, Response],
    ) -> ConsensusResults:
        """
        Build collaborative consensus on responses.

        Process:
        1. Each provider gives initial assessment of all responses
        2. Share assessments and discuss
        3. Identify shared insights and disagreements
        4. Vote on final rankings with confidence

        Args:
            prompt: Original prompt
            responses: Dictionary of provider name to response

        Returns:
            ConsensusResults with discussions, insights, and votes
        """
        results = ConsensusResults(prompt=prompt)
        provider_names = list(responses.keys())

        # Round 1: Initial assessments
        logger.info("Collaborative consensus: Round 1 - Initial assessments")
        assessments: Dict[str, str] = {}

        for evaluator_name, evaluator in self.providers.items():
            try:
                assessment = self._get_initial_assessment(
                    evaluator=evaluator,
                    evaluator_name=evaluator_name,
                    prompt=prompt,
                    responses=responses,
                )
                assessments[evaluator_name] = assessment
                results.discussions.append(Discussion(
                    round=1,
                    participant=evaluator_name,
                    content=assessment,
                    confidence=0.7,  # Initial confidence
                ))
            except Exception as e:
                logger.error(f"Initial assessment failed for {evaluator_name}: {e}")

        # Round 2: Discussion with awareness of others' views
        logger.info("Collaborative consensus: Round 2 - Discussion")
        for evaluator_name, evaluator in self.providers.items():
            try:
                # Share other assessments
                other_assessments = {
                    k: v for k, v in assessments.items()
                    if k != evaluator_name
                }

                updated = self._get_updated_assessment(
                    evaluator=evaluator,
                    evaluator_name=evaluator_name,
                    prompt=prompt,
                    responses=responses,
                    other_assessments=other_assessments,
                )
                results.discussions.append(Discussion(
                    round=2,
                    participant=evaluator_name,
                    content=updated,
                    confidence=0.8,  # Higher confidence after discussion
                ))
            except Exception as e:
                logger.error(f"Discussion round failed for {evaluator_name}: {e}")

        # Identify shared insights
        logger.info("Collaborative consensus: Identifying shared insights")
        insights = self._identify_insights(
            prompt=prompt,
            responses=responses,
            discussions=[d for d in results.discussions if d.round == 2],
        )
        results.shared_strengths = insights["strengths"]
        results.shared_weaknesses = insights["weaknesses"]
        results.disagreements = insights["disagreements"]

        # Collect votes
        logger.info("Collaborative consensus: Collecting votes")
        for evaluator_name, evaluator in self.providers.items():
            try:
                vote = self._get_vote(
                    evaluator=evaluator,
                    evaluator_name=evaluator_name,
                    prompt=prompt,
                    responses=responses,
                    provider_names=provider_names,
                )
                results.votes.append(vote)
                logger.debug(f"Vote from {evaluator_name}: {vote.rankings}")
            except Exception as e:
                logger.error(f"Voting failed for {evaluator_name}: {e}")

        # Calculate weighted scores
        results.calculate_weighted_scores(provider_names)

        # Generate synthesis
        results.synthesis = self._generate_synthesis(results)

        logger.info(f"Consensus scores: {results.weighted_scores}")

        return results

    def _get_initial_assessment(
        self,
        evaluator: LLMProvider,
        evaluator_name: str,
        prompt: str,
        responses: Dict[str, Response],
    ) -> str:
        """Get initial assessment from an evaluator."""
        responses_text = "\n\n".join([
            f"=== Response from {name} ===\n{resp.text}"
            for name, resp in responses.items()
            if name != evaluator_name  # Don't evaluate own response
        ])

        assessment_prompt = f"""You are participating in a collaborative evaluation of AI responses.

ORIGINAL PROMPT:
{prompt}

RESPONSES TO EVALUATE:
{responses_text}

Provide your initial assessment:
1. What are the key strengths you observe across the responses?
2. What are the common weaknesses or issues?
3. Which response do you think is best and why?
4. What aspects would you like other evaluators' opinions on?

Be specific and cite evidence from the responses.

Your assessment:"""

        result = evaluator.generate(assessment_prompt)
        return result.text

    def _get_updated_assessment(
        self,
        evaluator: LLMProvider,
        evaluator_name: str,
        prompt: str,
        responses: Dict[str, Response],
        other_assessments: Dict[str, str],
    ) -> str:
        """Get updated assessment after seeing others' views."""
        responses_text = "\n\n".join([
            f"=== Response from {name} ===\n{resp.text}"
            for name, resp in responses.items()
            if name != evaluator_name
        ])

        others_text = "\n\n".join([
            f"=== Assessment from {name} ===\n{assessment}"
            for name, assessment in other_assessments.items()
        ])

        discussion_prompt = f"""You are participating in a collaborative evaluation. You've seen
other evaluators' assessments.

ORIGINAL PROMPT:
{prompt}

RESPONSES BEING EVALUATED:
{responses_text}

OTHER EVALUATORS' ASSESSMENTS:
{others_text}

Based on the other assessments:
1. What points do you agree with?
2. What points do you disagree with and why?
3. Has your opinion changed on any response?
4. What consensus is emerging?

Update your assessment considering these perspectives.

Your updated assessment:"""

        result = evaluator.generate(discussion_prompt)
        return result.text

    def _identify_insights(
        self,
        prompt: str,
        responses: Dict[str, Response],
        discussions: List[Discussion],
    ) -> Dict[str, List[Any]]:
        """Identify shared insights from discussions."""
        # Use one of the providers to synthesize insights
        synthesizer = list(self.providers.values())[0]

        discussions_text = "\n\n".join([
            f"=== {d.participant} ===\n{d.content}"
            for d in discussions
        ])

        synthesis_prompt = f"""Analyze the following evaluator discussions and identify:

DISCUSSIONS:
{discussions_text}

Extract the following:

1. SHARED STRENGTHS: Points that multiple evaluators agree are strengths
   (format each as: "description | supporting evaluators | confidence 0-1")

2. SHARED WEAKNESSES: Issues that multiple evaluators identify
   (format each as: "description | supporting evaluators | confidence 0-1")

3. DISAGREEMENTS: Points where evaluators differ
   (format each as: "topic | evaluator1: position1 | evaluator2: position2")

OUTPUT FORMAT:
SHARED_STRENGTHS:
- strength description | evaluator1, evaluator2 | 0.9
- another strength | evaluator1, evaluator3 | 0.8

SHARED_WEAKNESSES:
- weakness description | evaluator1, evaluator2 | 0.85
- another weakness | evaluator2, evaluator3 | 0.7

DISAGREEMENTS:
- topic | evaluator1: their position | evaluator2: their position"""

        result = synthesizer.generate(synthesis_prompt)
        return self._parse_insights(result.text)

    def _parse_insights(self, response_text: str) -> Dict[str, List[Any]]:
        """Parse insights from synthesis response."""
        insights = {
            "strengths": [],
            "weaknesses": [],
            "disagreements": [],
        }

        current_section = None
        lines = response_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            line_upper = line.upper()
            if "SHARED_STRENGTHS" in line_upper:
                current_section = "strengths"
                continue
            elif "SHARED_WEAKNESSES" in line_upper:
                current_section = "weaknesses"
                continue
            elif "DISAGREEMENTS" in line_upper:
                current_section = "disagreements"
                continue

            if line.startswith("-") and current_section:
                content = line[1:].strip()

                if current_section in ("strengths", "weaknesses"):
                    parts = content.split("|")
                    if len(parts) >= 2:
                        description = parts[0].strip()
                        supporters = [s.strip() for s in parts[1].split(",")]
                        confidence = 0.8
                        if len(parts) >= 3:
                            try:
                                confidence = float(parts[2].strip())
                            except ValueError:
                                pass
                        insights[current_section].append(Insight(
                            description=description,
                            supporting_providers=supporters,
                            confidence=confidence,
                        ))

                elif current_section == "disagreements":
                    parts = content.split("|")
                    if len(parts) >= 2:
                        topic = parts[0].strip()
                        positions = {}
                        for part in parts[1:]:
                            if ":" in part:
                                evaluator, position = part.split(":", 1)
                                positions[evaluator.strip()] = position.strip()
                        if positions:
                            insights["disagreements"].append(Disagreement(
                                topic=topic,
                                positions=positions,
                            ))

        return insights

    def _get_vote(
        self,
        evaluator: LLMProvider,
        evaluator_name: str,
        prompt: str,
        responses: Dict[str, Response],
        provider_names: List[str],
    ) -> Vote:
        """Get final ranking vote from an evaluator."""
        responses_text = "\n\n".join([
            f"=== Response from {name} ===\n{resp.text}"
            for name, resp in responses.items()
        ])

        vote_prompt = f"""Based on your evaluation, provide your final ranking of the responses.

ORIGINAL PROMPT:
{prompt}

RESPONSES:
{responses_text}

AVAILABLE RESPONSES TO RANK: {', '.join(provider_names)}

Rank all responses from best to worst.

OUTPUT FORMAT (you must follow this exactly):
RANKINGS: [comma-separated list of provider names, best first]
CONFIDENCE: [0.0 to 1.0 - how confident are you in this ranking]
REASONING: [brief explanation of your ranking]"""

        result = evaluator.generate(vote_prompt)
        return self._parse_vote(result.text, evaluator_name, provider_names)

    def _parse_vote(
        self,
        response_text: str,
        voter: str,
        valid_providers: List[str],
    ) -> Vote:
        """Parse vote from response text."""
        vote = Vote(
            voter=voter,
            rankings=[],
            confidence=0.5,
            reasoning="",
        )

        lines = response_text.strip().split("\n")

        for line in lines:
            line_upper = line.upper().strip()

            if line_upper.startswith("RANKINGS:"):
                content = line.split(":", 1)[1].strip() if ":" in line else ""
                # Parse comma-separated rankings
                rankings = [r.strip().lower() for r in content.split(",")]
                # Map to valid provider names
                vote.rankings = [
                    p for p in valid_providers
                    if p.lower() in rankings or any(
                        p.lower() in r for r in rankings
                    )
                ]
                # If parsing failed, use order from response
                if not vote.rankings:
                    vote.rankings = valid_providers.copy()

            elif line_upper.startswith("CONFIDENCE:"):
                try:
                    conf_text = line.split(":", 1)[1].strip()
                    vote.confidence = float(conf_text)
                    vote.confidence = max(0.0, min(1.0, vote.confidence))
                except (ValueError, IndexError):
                    pass

            elif line_upper.startswith("REASONING:"):
                vote.reasoning = line.split(":", 1)[1].strip() if ":" in line else ""

        # Ensure all providers are ranked
        for provider in valid_providers:
            if provider not in vote.rankings:
                vote.rankings.append(provider)

        return vote

    def _generate_synthesis(self, results: ConsensusResults) -> str:
        """Generate a synthesis of the consensus process."""
        strengths = ", ".join([s.description for s in results.shared_strengths[:3]])
        weaknesses = ", ".join([w.description for w in results.shared_weaknesses[:3]])

        rankings = results.get_rankings()
        if rankings:
            top = rankings[0][0]
            synthesis = f"Consensus indicates {top} provided the best response. "
            synthesis += f"Key agreed strengths: {strengths}. "
            synthesis += f"Identified issues: {weaknesses}."
        else:
            synthesis = "Unable to reach consensus on rankings."

        return synthesis
