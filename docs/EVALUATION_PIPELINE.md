# Evaluation Pipeline Documentation

This document details the evaluation pipeline used to assess and rank LLM responses.

## Pipeline Overview

The evaluation pipeline consists of five distinct phases, each contributing to the final ranking with configurable weights.

```mermaid
flowchart LR
    subgraph Weights["Default Weights"]
        W1[Pointwise: 30%]
        W2[Pairwise: 30%]
        W3[Adversarial: 25%]
        W4[Collaborative: 15%]
    end

    P1[Phase 1<br/>Pointwise] --> AGG[Aggregation]
    P2[Phase 2<br/>Pairwise] --> AGG
    P3[Phase 3<br/>Adversarial] --> AGG
    P4[Phase 4<br/>Collaborative] --> AGG
    AGG --> P5[Phase 5<br/>Final Ranking]

    W1 -.-> P1
    W2 -.-> P2
    W3 -.-> P3
    W4 -.-> P4
```

---

## Phase 1: Pointwise Evaluation

### Purpose
Score each response independently against defined rubrics using G-Eval methodology.

### Method: G-Eval with Chain-of-Thought

Based on research from [G-Eval: NLG Evaluation using GPT-4](https://arxiv.org/abs/2303.16634), this phase uses structured Chain-of-Thought prompting for consistent scoring.

```mermaid
flowchart TD
    subgraph Input
        RESP[Response to Evaluate]
        RUB[Rubric Definition]
        PROMPT[Original Prompt]
    end

    subgraph GEval["G-Eval Process"]
        TASK[1. Define Evaluation Task]
        COT[2. Generate Chain-of-Thought]
        SCORE[3. Assign Score with Reasoning]
    end

    subgraph Output
        NUM[Numeric Score 0-10]
        REASON[Reasoning Explanation]
    end

    Input --> TASK --> COT --> SCORE --> Output
```

### Default Rubrics

| Rubric | Description | Weight |
|--------|-------------|--------|
| Accuracy | Factual correctness and precision | 25% |
| Completeness | Coverage of all aspects of the prompt | 20% |
| Clarity | Clear, understandable communication | 20% |
| Relevance | Direct applicability to the prompt | 20% |
| Reasoning | Quality of logic and argumentation | 15% |

### Cross-Evaluation Matrix

Each LLM evaluates all responses except its own to prevent self-bias.

```mermaid
flowchart TB
    subgraph Evaluators
        E_OAI[OpenAI as Evaluator]
        E_CLD[Claude as Evaluator]
        E_GEM[Gemini as Evaluator]
        E_XAI[xAI as Evaluator]
    end

    subgraph Responses
        R_OAI[OpenAI Response]
        R_CLD[Claude Response]
        R_GEM[Gemini Response]
        R_XAI[xAI Response]
    end

    E_OAI -->|evaluates| R_CLD & R_GEM & R_XAI
    E_CLD -->|evaluates| R_OAI & R_GEM & R_XAI
    E_GEM -->|evaluates| R_OAI & R_CLD & R_XAI
    E_XAI -->|evaluates| R_OAI & R_CLD & R_GEM
```

### Score Aggregation

```
Final_Pointwise_Score(Response_X) =
    mean(scores from all evaluators) * rubric_weights
```

---

## Phase 2: Pairwise Comparison

### Purpose
Direct head-to-head comparisons to capture relative quality that absolute scores may miss.

### Method: Tournament-Style Comparison

Based on [Chatbot Arena](https://lmsys.org/blog/2023-12-07-leaderboard/) methodology.

```mermaid
flowchart TD
    subgraph Tournament["Round Robin Tournament"]
        M1[Match 1: A vs B]
        M2[Match 2: A vs C]
        M3[Match 3: A vs D]
        M4[Match 4: B vs C]
        M5[Match 5: B vs D]
        M6[Match 6: C vs D]
    end

    subgraph Results["Outcome Recording"]
        WIN[Win: +1]
        LOSE[Loss: 0]
        TIE[Tie: +0.5]
    end

    Tournament --> Results --> MAT[Win Matrix]
```

### Position Bias Mitigation

To prevent order bias, each pair is presented in both orders to different judges:

```mermaid
flowchart LR
    subgraph Pair["A vs B Comparison"]
        ORDER1[Judge 1 sees: A then B]
        ORDER2[Judge 2 sees: B then A]
    end

    ORDER1 --> AGG[Aggregate Verdicts]
    ORDER2 --> AGG
```

### Judge Assignment

Judges are LLMs not involved in the comparison being judged:

| Comparison | Judges |
|------------|--------|
| OpenAI vs Claude | Gemini, xAI |
| OpenAI vs Gemini | Claude, xAI |
| OpenAI vs xAI | Claude, Gemini |
| Claude vs Gemini | OpenAI, xAI |
| Claude vs xAI | OpenAI, Gemini |
| Gemini vs xAI | OpenAI, Claude |

### Win Matrix Output

```
          OpenAI  Claude  Gemini  xAI
OpenAI      -      0.5     1.0    1.0
Claude     0.5      -      1.0    0.5
Gemini     0.0     0.0      -     0.5
xAI        0.0     0.5     0.5     -
```

---

## Phase 3: Adversarial Debate

### Purpose
Use structured argumentation to surface non-obvious strengths and weaknesses.

### Method: D3 Framework (Debate, Deliberate, Decide)

Based on [D3: Adversarial Multi-Agent Evaluation](https://arxiv.org/abs/2410.04663).

```mermaid
sequenceDiagram
    participant A as Advocate
    participant C as Challenger
    participant J as Judge
    participant Y as Jury (optional)

    Note over A,Y: Debate for Response X

    rect rgb(220, 240, 220)
        Note right of A: Opening Arguments
        A->>J: Present case for Response X
        Note right of A: - Key strengths<br/>- Why it answers prompt well<br/>- Quality of reasoning
        C->>J: Present case against Response X
        Note right of C: - Weaknesses and gaps<br/>- Factual issues<br/>- Missing elements
    end

    rect rgb(240, 220, 220)
        Note right of A: Rebuttal Round
        A->>J: Address challenger's points
        Note right of A: - Refute criticisms<br/>- Provide evidence
        C->>J: Counter advocate's defense
        Note right of C: - Maintain objections<br/>- Introduce new issues
    end

    rect rgb(220, 220, 240)
        Note right of J: Deliberation
        J->>J: Evaluate argument quality
        J->>J: Assess evidence presented
        J->>J: Consider both sides fairly
    end

    J->>Y: Present verdict
    Y->>Y: Confirm or override (if enabled)
```

### Role Assignment Strategy

For each response being debated:

```mermaid
flowchart TD
    RESP[Response X from Provider P] --> ASSIGN[Role Assignment]

    ASSIGN --> ADV[Advocate: Different provider than P]
    ASSIGN --> CHL[Challenger: Another different provider]
    ASSIGN --> JDG[Judge: Remaining provider]

    subgraph Example["Example: Evaluating OpenAI Response"]
        E_ADV[Advocate: Claude]
        E_CHL[Challenger: Gemini]
        E_JDG[Judge: xAI]
    end
```

### Debate Scoring Criteria

| Criterion | Description |
|-----------|-------------|
| Argument Strength | Quality of reasoning and evidence |
| Responsiveness | How well rebuttals address points |
| Completeness | Coverage of relevant aspects |
| Fairness | Objectivity and acknowledgment of valid counterpoints |

### Output Structure

```json
{
  "response_debated": "openai",
  "advocate": "claude",
  "challenger": "gemini",
  "judge": "xai",
  "rounds": [
    {
      "round": 1,
      "advocate_argument": "...",
      "challenger_argument": "..."
    },
    {
      "round": 2,
      "advocate_rebuttal": "...",
      "challenger_rebuttal": "..."
    }
  ],
  "verdict": {
    "score": 7.5,
    "reasoning": "...",
    "key_strengths_validated": ["..."],
    "key_weaknesses_confirmed": ["..."]
  }
}
```

---

## Phase 4: Collaborative Consensus

### Purpose
Use multi-model collaboration to identify emergent insights and build consensus.

### Method: ReConcile-Style Consensus Building

Based on [ReConcile: Multi-Model Consensus](https://arxiv.org/abs/2309.13007).

```mermaid
flowchart TD
    subgraph Round1["Round 1: Initial Assessment"]
        ALL1[All LLMs independently assess all responses]
        SHARE1[Share assessments with confidence scores]
    end

    subgraph Round2["Round 2: Discussion"]
        DISC[Each LLM reviews others' assessments]
        UPDATE[Update own assessment based on new perspectives]
        SHARE2[Share updated assessments]
    end

    subgraph Round3["Round 3: Consensus"]
        VOTE[Confidence-weighted voting]
        SYNTH[Synthesize shared insights]
    end

    Round1 --> Round2 --> Round3 --> OUT[Consensus Report]
```

### Confidence-Weighted Voting

Each LLM provides assessments with confidence levels:

```mermaid
flowchart LR
    subgraph Votes["Individual Votes with Confidence"]
        V1["OpenAI: Response A is best<br/>Confidence: 0.85"]
        V2["Claude: Response B is best<br/>Confidence: 0.72"]
        V3["Gemini: Response A is best<br/>Confidence: 0.90"]
        V4["xAI: Response A is best<br/>Confidence: 0.65"]
    end

    Votes --> WEIGHT[Weight by Confidence]
    WEIGHT --> RESULT["Weighted Result:<br/>A: 2.40, B: 0.72"]
```

### Consensus Outputs

1. **Shared Strengths**: Qualities all evaluators agree are positive
2. **Shared Weaknesses**: Issues all evaluators identify
3. **Points of Disagreement**: Areas where evaluators differ
4. **Synthesis**: Combined insight from multiple perspectives

---

## Phase 5: Final Ranking

### Purpose
Aggregate all evaluation signals into a definitive ranking.

### Method: Bradley-Terry Model

Based on [Bradley-Terry statistical model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) used by Chatbot Arena.

```mermaid
flowchart TD
    subgraph Inputs["Aggregated Inputs"]
        IN1[Pointwise Scores<br/>weight: 30%]
        IN2[Pairwise Win Rates<br/>weight: 30%]
        IN3[Debate Scores<br/>weight: 25%]
        IN4[Consensus Scores<br/>weight: 15%]
    end

    subgraph BT["Bradley-Terry Computation"]
        NORM[Normalize all scores to 0-1]
        WEIGHT[Apply weights]
        MLE[Maximum Likelihood Estimation]
        ITER[Iterative optimization]
    end

    subgraph Output["Final Output"]
        RANK[Ordered Ranking]
        SCORES[Numeric Scores]
        CI[95% Confidence Intervals]
    end

    Inputs --> NORM --> WEIGHT --> MLE --> ITER --> Output
```

### Bradley-Terry Formula

For items i and j, the probability that i beats j:

```
P(i > j) = exp(beta_i) / (exp(beta_i) + exp(beta_j))
```

Where beta values are estimated via maximum likelihood from observed comparisons.

### Final Score Calculation

```python
final_score = (
    0.30 * normalized_pointwise_score +
    0.30 * normalized_pairwise_winrate +
    0.25 * normalized_debate_score +
    0.15 * normalized_consensus_score
)

# Bradley-Terry refinement
bt_scores = bradley_terry_mle(pairwise_matrix)
final_ranking = weighted_merge(final_score, bt_scores)
```

### Output Format

```json
{
  "rankings": [
    {
      "rank": 1,
      "provider": "claude",
      "score": 8.45,
      "confidence_interval": [8.12, 8.78],
      "breakdown": {
        "pointwise": 8.2,
        "pairwise_winrate": 0.83,
        "debate": 8.7,
        "consensus": 8.5
      }
    },
    {
      "rank": 2,
      "provider": "openai",
      "score": 8.21,
      "confidence_interval": [7.89, 8.53],
      "breakdown": {...}
    }
  ]
}
```

---

## Evaluation Prompts

### Pointwise Evaluation Prompt Template

```
You are evaluating an AI response to the following prompt:

ORIGINAL PROMPT:
{prompt}

RESPONSE TO EVALUATE:
{response}

EVALUATION RUBRIC - {rubric_name}:
{rubric_description}

Please evaluate this response on a scale of 0-10 for {rubric_name}.

First, think through your evaluation step by step:
1. What aspects of {rubric_name} does this response demonstrate well?
2. What aspects are lacking or could be improved?
3. How does it compare to an ideal response?

Then provide your final score and a brief justification.

OUTPUT FORMAT:
REASONING: [Your step-by-step analysis]
SCORE: [0-10]
JUSTIFICATION: [Brief explanation of score]
```

### Pairwise Comparison Prompt Template

```
You are comparing two AI responses to the same prompt.

ORIGINAL PROMPT:
{prompt}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Which response better answers the prompt? Consider:
- Accuracy and correctness
- Completeness and depth
- Clarity and coherence
- Relevance to the question

Provide your analysis and verdict.

OUTPUT FORMAT:
ANALYSIS: [Your comparison analysis]
VERDICT: [A, B, or TIE]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Brief explanation of verdict]
```

### Adversarial Debate Prompt Templates

**Advocate Prompt:**
```
You are the ADVOCATE for the following response. Your job is to present
the strongest case for why this is a good response.

ORIGINAL PROMPT: {prompt}
RESPONSE YOU ARE DEFENDING: {response}

Present your opening argument highlighting:
1. Key strengths of this response
2. How well it addresses the prompt
3. Quality of reasoning or information provided
4. Any unique value it provides

Be persuasive but truthful - do not fabricate strengths.
```

**Challenger Prompt:**
```
You are the CHALLENGER against the following response. Your job is to
identify weaknesses and issues.

ORIGINAL PROMPT: {prompt}
RESPONSE YOU ARE CHALLENGING: {response}
ADVOCATE'S ARGUMENT: {advocate_argument}

Present your challenge highlighting:
1. Weaknesses or gaps in the response
2. Factual errors or questionable claims
3. Missing important information
4. Areas where it fails to address the prompt

Be critical but fair - acknowledge valid strengths while focusing on issues.
```

**Judge Prompt:**
```
You are the JUDGE evaluating a debate about the following response.

ORIGINAL PROMPT: {prompt}
RESPONSE BEING EVALUATED: {response}

ADVOCATE'S ARGUMENTS:
{advocate_arguments}

CHALLENGER'S ARGUMENTS:
{challenger_arguments}

Evaluate the response based on the debate. Consider:
1. Which arguments were most compelling?
2. Were the advocate's claimed strengths valid?
3. Were the challenger's criticisms fair and accurate?
4. Overall, how good is this response?

OUTPUT FORMAT:
ANALYSIS: [Your evaluation of the debate]
VALIDATED_STRENGTHS: [List of confirmed strengths]
CONFIRMED_WEAKNESSES: [List of valid criticisms]
SCORE: [0-10]
REASONING: [Explanation of final judgment]
```
