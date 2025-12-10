"""Adversarial debate evaluation - D3-inspired debate between LLMs."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..providers.base import LLMProvider, Response
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DebateRound:
    """A single round of debate."""
    round_number: int
    advocate_argument: str
    challenger_argument: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "round_number": self.round_number,
            "advocate_argument": self.advocate_argument,
            "challenger_argument": self.challenger_argument,
        }


@dataclass
class DebateVerdict:
    """Judge's verdict on a debate."""
    score: float
    reasoning: str
    validated_strengths: List[str]
    confirmed_weaknesses: List[str]
    argument_quality: Dict[str, float]  # advocate, challenger scores

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "score": self.score,
            "reasoning": self.reasoning,
            "validated_strengths": self.validated_strengths,
            "confirmed_weaknesses": self.confirmed_weaknesses,
            "argument_quality": self.argument_quality,
        }


@dataclass
class Debate:
    """Complete debate for a single response."""
    response_id: str
    advocate: str
    challenger: str
    judge: str
    rounds: List[DebateRound] = field(default_factory=list)
    verdict: Optional[DebateVerdict] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "response_id": self.response_id,
            "advocate": self.advocate,
            "challenger": self.challenger,
            "judge": self.judge,
            "rounds": [r.to_dict() for r in self.rounds],
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AdversarialResults:
    """Results from adversarial debate phase."""
    prompt: str
    debates: List[Debate] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)

    def get_debate_for(self, provider: str) -> Optional[Debate]:
        """Get debate for a specific provider's response."""
        for debate in self.debates:
            if debate.response_id == provider:
                return debate
        return None

    def calculate_scores(self) -> None:
        """Calculate final scores from debate verdicts."""
        for debate in self.debates:
            if debate.verdict:
                self.scores[debate.response_id] = debate.verdict.score

    def get_rankings(self) -> List[tuple]:
        """Get providers ranked by debate score."""
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "debates": [d.to_dict() for d in self.debates],
            "scores": self.scores,
        }


class AdversarialDebate:
    """Orchestrates adversarial debate evaluation."""

    def __init__(
        self,
        providers: Dict[str, LLMProvider],
        num_rounds: int = 2,
    ):
        """
        Initialize the adversarial debate evaluator.

        Args:
            providers: Dictionary of provider name to provider instance
            num_rounds: Number of debate rounds (default 2)
        """
        self.providers = providers
        self.num_rounds = num_rounds

    def evaluate(
        self,
        prompt: str,
        responses: Dict[str, Response],
    ) -> AdversarialResults:
        """
        Conduct adversarial debates for all responses.

        For each response, assign:
        - Advocate: Defends the response
        - Challenger: Attacks the response
        - Judge: Evaluates the debate

        Args:
            prompt: Original prompt
            responses: Dictionary of provider name to response

        Returns:
            AdversarialResults with all debates
        """
        results = AdversarialResults(prompt=prompt)
        provider_names = list(responses.keys())

        if len(provider_names) < 3:
            logger.warning(
                "Adversarial debate requires at least 3 providers. "
                "Using available providers in multiple roles."
            )

        for response_provider in provider_names:
            # Assign roles (different providers for each role)
            roles = self._assign_roles(response_provider, provider_names)

            logger.info(
                f"Debate for {response_provider}: "
                f"advocate={roles['advocate']}, "
                f"challenger={roles['challenger']}, "
                f"judge={roles['judge']}"
            )

            try:
                debate = self._conduct_debate(
                    prompt=prompt,
                    response=responses[response_provider],
                    response_provider=response_provider,
                    advocate=self.providers[roles["advocate"]],
                    advocate_name=roles["advocate"],
                    challenger=self.providers[roles["challenger"]],
                    challenger_name=roles["challenger"],
                    judge=self.providers[roles["judge"]],
                    judge_name=roles["judge"],
                )
                results.debates.append(debate)

            except Exception as e:
                logger.error(f"Debate failed for {response_provider}: {e}")

        results.calculate_scores()
        return results

    def _assign_roles(
        self,
        response_provider: str,
        all_providers: List[str],
    ) -> Dict[str, str]:
        """
        Assign advocate, challenger, and judge roles.

        Args:
            response_provider: Provider whose response is being debated
            all_providers: List of all available providers

        Returns:
            Dictionary with role assignments
        """
        # Get providers other than the one being evaluated
        others = [p for p in all_providers if p != response_provider]

        if len(others) >= 3:
            # Ideal case: 3+ other providers
            advocate = others[0]
            challenger = others[1]
            judge = others[2]
        elif len(others) == 2:
            # 3 total providers
            advocate = others[0]
            challenger = others[1]
            judge = others[0]  # Advocate also serves as judge
        elif len(others) == 1:
            # Only 2 total providers
            advocate = others[0]
            challenger = others[0]
            judge = others[0]
        else:
            # Single provider - self-debate (not ideal)
            advocate = response_provider
            challenger = response_provider
            judge = response_provider

        return {
            "advocate": advocate,
            "challenger": challenger,
            "judge": judge,
        }

    def _conduct_debate(
        self,
        prompt: str,
        response: Response,
        response_provider: str,
        advocate: LLMProvider,
        advocate_name: str,
        challenger: LLMProvider,
        challenger_name: str,
        judge: LLMProvider,
        judge_name: str,
    ) -> Debate:
        """
        Conduct a complete debate for one response.

        Args:
            prompt: Original prompt
            response: Response being debated
            response_provider: Name of response provider
            advocate: Advocate provider
            advocate_name: Name of advocate
            challenger: Challenger provider
            challenger_name: Name of challenger
            judge: Judge provider
            judge_name: Name of judge

        Returns:
            Debate with all rounds and verdict
        """
        debate = Debate(
            response_id=response_provider,
            advocate=advocate_name,
            challenger=challenger_name,
            judge=judge_name,
        )

        # Conduct debate rounds
        advocate_args = []
        challenger_args = []

        for round_num in range(1, self.num_rounds + 1):
            logger.debug(f"Debate round {round_num} for {response_provider}")

            # Get advocate argument
            if round_num == 1:
                advocate_prompt = self._build_advocate_opening(
                    prompt, response.text
                )
            else:
                advocate_prompt = self._build_advocate_rebuttal(
                    prompt, response.text,
                    challenger_args[-1] if challenger_args else ""
                )

            advocate_response = advocate.generate(advocate_prompt)
            advocate_args.append(advocate_response.text)

            # Get challenger argument
            if round_num == 1:
                challenger_prompt = self._build_challenger_opening(
                    prompt, response.text, advocate_args[-1]
                )
            else:
                challenger_prompt = self._build_challenger_rebuttal(
                    prompt, response.text,
                    advocate_args[-1]
                )

            challenger_response = challenger.generate(challenger_prompt)
            challenger_args.append(challenger_response.text)

            debate.rounds.append(DebateRound(
                round_number=round_num,
                advocate_argument=advocate_args[-1],
                challenger_argument=challenger_args[-1],
            ))

        # Get judge verdict
        verdict = self._get_verdict(
            judge=judge,
            prompt=prompt,
            response_text=response.text,
            advocate_args=advocate_args,
            challenger_args=challenger_args,
        )
        debate.verdict = verdict

        logger.info(
            f"Debate verdict for {response_provider}: score={verdict.score:.2f}"
        )

        return debate

    def _build_advocate_opening(self, prompt: str, response_text: str) -> str:
        """Build opening argument prompt for advocate."""
        return f"""You are the ADVOCATE for the following response. Your job is to present
the strongest case for why this is a good response.

ORIGINAL PROMPT:
{prompt}

RESPONSE YOU ARE DEFENDING:
{response_text}

Present your opening argument highlighting:
1. Key strengths of this response
2. How well it addresses the prompt
3. Quality of reasoning or information provided
4. Any unique value it provides

Be persuasive but truthful - do not fabricate strengths. Present concrete evidence
from the response to support your points.

Your argument:"""

    def _build_advocate_rebuttal(
        self,
        prompt: str,
        response_text: str,
        challenger_arg: str
    ) -> str:
        """Build rebuttal prompt for advocate."""
        return f"""You are the ADVOCATE defending a response. The challenger has raised objections.

ORIGINAL PROMPT:
{prompt}

RESPONSE YOU ARE DEFENDING:
{response_text}

CHALLENGER'S ARGUMENT:
{challenger_arg}

Present your rebuttal:
1. Address the challenger's specific criticisms
2. Defend against unfair or inaccurate attacks
3. Acknowledge valid points while maintaining overall defense
4. Provide additional evidence for the response's quality

Your rebuttal:"""

    def _build_challenger_opening(
        self,
        prompt: str,
        response_text: str,
        advocate_arg: str
    ) -> str:
        """Build opening argument prompt for challenger."""
        return f"""You are the CHALLENGER against the following response. Your job is to
identify weaknesses and issues.

ORIGINAL PROMPT:
{prompt}

RESPONSE YOU ARE CHALLENGING:
{response_text}

ADVOCATE'S ARGUMENT:
{advocate_arg}

Present your challenge highlighting:
1. Weaknesses or gaps in the response
2. Factual errors or questionable claims
3. Missing important information
4. Areas where it fails to address the prompt
5. Counter the advocate's points where appropriate

Be critical but fair - acknowledge valid strengths while focusing on legitimate issues.

Your challenge:"""

    def _build_challenger_rebuttal(
        self,
        prompt: str,
        response_text: str,
        advocate_arg: str
    ) -> str:
        """Build rebuttal prompt for challenger."""
        return f"""You are the CHALLENGER attacking a response. The advocate has defended it.

ORIGINAL PROMPT:
{prompt}

RESPONSE YOU ARE CHALLENGING:
{response_text}

ADVOCATE'S DEFENSE:
{advocate_arg}

Present your counter-argument:
1. Address the advocate's defense points
2. Maintain your key criticisms
3. Introduce any additional weaknesses not yet discussed
4. Summarize why the response falls short

Your counter-argument:"""

    def _get_verdict(
        self,
        judge: LLMProvider,
        prompt: str,
        response_text: str,
        advocate_args: List[str],
        challenger_args: List[str],
    ) -> DebateVerdict:
        """
        Get judge's verdict on the debate.

        Args:
            judge: Judge provider
            prompt: Original prompt
            response_text: Response being debated
            advocate_args: List of advocate arguments
            challenger_args: List of challenger arguments

        Returns:
            DebateVerdict with final judgment
        """
        # Format debate transcript
        debate_transcript = ""
        for i, (adv, chl) in enumerate(zip(advocate_args, challenger_args), 1):
            debate_transcript += f"\n=== Round {i} ===\n"
            debate_transcript += f"\nADVOCATE:\n{adv}\n"
            debate_transcript += f"\nCHALLENGER:\n{chl}\n"

        judge_prompt = f"""You are the JUDGE evaluating a debate about an AI response.

ORIGINAL PROMPT:
{prompt}

RESPONSE BEING EVALUATED:
{response_text}

DEBATE TRANSCRIPT:
{debate_transcript}

Evaluate the response based on the debate. Consider:
1. Which arguments were most compelling?
2. Were the advocate's claimed strengths valid?
3. Were the challenger's criticisms fair and accurate?
4. Overall, how good is this response?

OUTPUT FORMAT (you must follow this exactly):
ANALYSIS: [Your evaluation of the debate and response]
VALIDATED_STRENGTHS: [Comma-separated list of confirmed strengths]
CONFIRMED_WEAKNESSES: [Comma-separated list of valid criticisms]
ADVOCATE_SCORE: [0-10 score for argument quality]
CHALLENGER_SCORE: [0-10 score for argument quality]
FINAL_SCORE: [0-10 overall score for the response]
REASONING: [Brief explanation of final score]"""

        result = judge.generate(judge_prompt)
        return self._parse_verdict(result.text)

    def _parse_verdict(self, response_text: str) -> DebateVerdict:
        """Parse judge's verdict from response text."""
        verdict = DebateVerdict(
            score=5.0,
            reasoning="",
            validated_strengths=[],
            confirmed_weaknesses=[],
            argument_quality={"advocate": 5.0, "challenger": 5.0},
        )

        lines = response_text.strip().split("\n")

        for line in lines:
            line_upper = line.upper().strip()

            if line_upper.startswith("VALIDATED_STRENGTHS:"):
                content = line.split(":", 1)[1].strip() if ":" in line else ""
                verdict.validated_strengths = [
                    s.strip() for s in content.split(",") if s.strip()
                ]

            elif line_upper.startswith("CONFIRMED_WEAKNESSES:"):
                content = line.split(":", 1)[1].strip() if ":" in line else ""
                verdict.confirmed_weaknesses = [
                    s.strip() for s in content.split(",") if s.strip()
                ]

            elif line_upper.startswith("ADVOCATE_SCORE:"):
                try:
                    score_text = line.split(":", 1)[1].strip()
                    verdict.argument_quality["advocate"] = float(score_text.split("/")[0])
                except (ValueError, IndexError):
                    pass

            elif line_upper.startswith("CHALLENGER_SCORE:"):
                try:
                    score_text = line.split(":", 1)[1].strip()
                    verdict.argument_quality["challenger"] = float(score_text.split("/")[0])
                except (ValueError, IndexError):
                    pass

            elif line_upper.startswith("FINAL_SCORE:"):
                try:
                    score_text = line.split(":", 1)[1].strip()
                    verdict.score = float(score_text.split("/")[0])
                    verdict.score = max(0, min(10, verdict.score))
                except (ValueError, IndexError):
                    pass

            elif line_upper.startswith("REASONING:"):
                verdict.reasoning = line.split(":", 1)[1].strip() if ":" in line else ""

        return verdict
