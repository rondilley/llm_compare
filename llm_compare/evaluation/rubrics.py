"""Evaluation rubrics for scoring LLM responses."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any


@dataclass
class Rubric:
    """A single evaluation rubric."""
    name: str
    description: str
    scale_min: int = 0
    scale_max: int = 10
    weight: float = 1.0
    scoring_guidance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rubric":
        return cls(
            name=data["name"],
            description=data["description"],
            scale_min=data.get("scale_min", 0),
            scale_max=data.get("scale_max", 10),
            weight=data.get("weight", 1.0),
            scoring_guidance=data.get("scoring_guidance", ""),
        )


@dataclass
class RubricSet:
    """A collection of evaluation rubrics."""
    rubrics: List[Rubric] = field(default_factory=list)

    def total_weight(self) -> float:
        return sum(r.weight for r in self.rubrics)

    def normalize_weights(self) -> None:
        total = self.total_weight()
        if total > 0:
            for rubric in self.rubrics:
                rubric.weight /= total

    def to_dict(self) -> Dict[str, Any]:
        return {"rubrics": [r.to_dict() for r in self.rubrics]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RubricSet":
        return cls(rubrics=[Rubric.from_dict(r) for r in data.get("rubrics", [])])

    def __iter__(self):
        return iter(self.rubrics)

    def __len__(self):
        return len(self.rubrics)


# Default evaluation rubrics
default_rubrics = RubricSet(rubrics=[
    Rubric(
        name="accuracy",
        description="Factual correctness and precision of information provided in the response",
        scale_min=0,
        scale_max=10,
        weight=0.25,
        scoring_guidance="""Score based on factual accuracy:
- 10: Completely accurate, no factual errors
- 8-9: Mostly accurate with very minor issues
- 6-7: Generally accurate but with some notable errors
- 4-5: Mixed accuracy, significant errors present
- 2-3: Mostly inaccurate
- 0-1: Completely wrong or misleading""",
    ),
    Rubric(
        name="completeness",
        description="Coverage of all relevant aspects of the prompt and thoroughness of the response",
        scale_min=0,
        scale_max=10,
        weight=0.20,
        scoring_guidance="""Score based on completeness:
- 10: Addresses all aspects thoroughly and comprehensively
- 8-9: Covers most aspects well, minor gaps
- 6-7: Adequate coverage but missing some relevant points
- 4-5: Partial coverage, notable omissions
- 2-3: Major gaps in coverage
- 0-1: Fails to address the prompt""",
    ),
    Rubric(
        name="clarity",
        description="Clear, understandable communication with good organization and structure",
        scale_min=0,
        scale_max=10,
        weight=0.20,
        scoring_guidance="""Score based on clarity:
- 10: Crystal clear, excellently organized, easy to follow
- 8-9: Very clear with good structure
- 6-7: Generally clear but some confusing parts
- 4-5: Somewhat unclear or poorly organized
- 2-3: Difficult to understand
- 0-1: Incomprehensible or incoherent""",
    ),
    Rubric(
        name="relevance",
        description="Direct applicability and focus on the prompt without unnecessary tangents",
        scale_min=0,
        scale_max=10,
        weight=0.20,
        scoring_guidance="""Score based on relevance:
- 10: Perfectly relevant throughout, directly addresses the prompt
- 8-9: Highly relevant with minimal tangents
- 6-7: Mostly relevant but some off-topic content
- 4-5: Partially relevant, significant tangents
- 2-3: Mostly irrelevant
- 0-1: Completely off-topic""",
    ),
    Rubric(
        name="reasoning",
        description="Quality of logic, argumentation, analysis, and depth of thought",
        scale_min=0,
        scale_max=10,
        weight=0.15,
        scoring_guidance="""Score based on reasoning quality:
- 10: Excellent logical structure, sophisticated analysis
- 8-9: Strong reasoning with minor gaps
- 6-7: Adequate reasoning, some logical issues
- 4-5: Weak reasoning, noticeable logical flaws
- 2-3: Poor logic, major reasoning errors
- 0-1: No coherent reasoning present""",
    ),
])
