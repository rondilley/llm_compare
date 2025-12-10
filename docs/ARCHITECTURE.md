# LLM Compare - Architecture Documentation

## System Overview

LLM Compare is a multi-AI evaluation tool that orchestrates responses from multiple LLM providers and uses collaborative and adversarial strategies to evaluate, score, and rank the responses.

```mermaid
flowchart TB
    subgraph Input
        USER[User] --> CLI[CLI Interface]
        KEYS[API Key Files<br/>*.key.txt] --> DISC[Provider Discovery]
    end

    subgraph Core["Core System"]
        CLI --> SM[Session Manager]
        DISC --> PM[Provider Manager]
        PM --> SM
        SM --> EVAL[Evaluation Engine]
        EVAL --> RANK[Ranking Engine]
    end

    subgraph Providers["LLM Providers"]
        PM --> OAI[OpenAI<br/>GPT-4]
        PM --> CLD[Anthropic<br/>Claude]
        PM --> GEM[Google<br/>Gemini]
        PM --> XAI[xAI<br/>Grok]
    end

    subgraph Output
        RANK --> STORE[Session Storage]
        STORE --> RPT[Report Generator]
        RPT --> PDF[PDF Report]
        STORE --> JSON[JSON Data]
    end
```

---

## Component Architecture

### High-Level Components

```mermaid
classDiagram
    class CLI {
        +parse_args()
        +run_interactive()
        +run_direct(prompt)
    }

    class SessionManager {
        -session_id: str
        -providers: List~Provider~
        -storage: Storage
        +create_session(prompt)
        +execute_pipeline()
        +get_results()
    }

    class ProviderManager {
        -providers: Dict~str, Provider~
        +discover_providers()
        +get_provider(name)
        +get_all_active()
    }

    class EvaluationEngine {
        -rubrics: List~Rubric~
        +run_pointwise(responses)
        +run_pairwise(responses)
        +run_adversarial(responses)
        +run_collaborative(responses)
    }

    class RankingEngine {
        +bradley_terry(comparisons)
        +aggregate_scores(evaluations)
        +compute_final_ranking()
    }

    class ReportGenerator {
        +generate_pdf(session_data)
        +format_responses()
        +format_evaluations()
    }

    CLI --> SessionManager
    SessionManager --> ProviderManager
    SessionManager --> EvaluationEngine
    SessionManager --> RankingEngine
    SessionManager --> ReportGenerator
```

---

## Provider System

### Provider Discovery Flow

```mermaid
sequenceDiagram
    participant App as Application
    participant Disc as Discovery
    participant FS as File System
    participant Prov as Provider

    App->>Disc: discover_providers()
    Disc->>FS: scan for *.key.txt
    FS-->>Disc: [openai.key.txt, claude.key.txt, ...]

    loop For each key file
        Disc->>FS: read key file
        FS-->>Disc: API key content
        Disc->>Prov: create_provider(name, key)
        Prov->>Prov: validate_key()
        alt Key Valid
            Prov-->>Disc: Provider instance
        else Key Invalid
            Prov-->>Disc: None + warning
        end
    end

    Disc-->>App: List of active providers
```

### Provider Class Hierarchy

```mermaid
classDiagram
    class LLMProvider {
        <<abstract>>
        #api_key: str
        #client: Any
        +name: str*
        +model_id: str*
        +generate(prompt, kwargs) Response*
        +evaluate(response, rubric) Evaluation*
        #_handle_error(error)
        #_retry_with_backoff(func)
    }

    class OpenAIProvider {
        +name = "openai"
        +model_id = "gpt-4-turbo"
        -client: OpenAI
        +generate(prompt, kwargs)
        +evaluate(response, rubric)
    }

    class ClaudeProvider {
        +name = "claude"
        +model_id = "claude-3-opus"
        -client: Anthropic
        +generate(prompt, kwargs)
        +evaluate(response, rubric)
    }

    class GeminiProvider {
        +name = "gemini"
        +model_id = "gemini-pro"
        -client: GenerativeModel
        +generate(prompt, kwargs)
        +evaluate(response, rubric)
    }

    class XAIProvider {
        +name = "xai"
        +model_id = "grok-beta"
        -client: OpenAI
        +generate(prompt, kwargs)
        +evaluate(response, rubric)
    }

    LLMProvider <|-- OpenAIProvider
    LLMProvider <|-- ClaudeProvider
    LLMProvider <|-- GeminiProvider
    LLMProvider <|-- XAIProvider
```

---

## Evaluation Pipeline

### Complete Pipeline Flow

```mermaid
flowchart TD
    subgraph Phase0["Phase 0: Response Collection"]
        P0A[Send prompt to all providers] --> P0B[Collect responses]
        P0B --> P0C[Store raw responses]
    end

    subgraph Phase1["Phase 1: Pointwise Evaluation"]
        P1A[Load evaluation rubrics] --> P1B[Each LLM scores other responses]
        P1B --> P1C[Aggregate scores per rubric]
        P1C --> P1D[Normalize and weight scores]
    end

    subgraph Phase2["Phase 2: Pairwise Comparison"]
        P2A[Generate all response pairs] --> P2B[Each LLM judges pairs]
        P2B --> P2C[Record win/loss/tie]
        P2C --> P2D[Build comparison matrix]
    end

    subgraph Phase3["Phase 3: Adversarial Debate"]
        P3A[Assign roles: Advocate, Challenger, Judge] --> P3B[Round 1: Initial arguments]
        P3B --> P3C[Round 2: Rebuttals]
        P3C --> P3D[Judge evaluates debate]
        P3D --> P3E[Record debate outcomes]
    end

    subgraph Phase4["Phase 4: Collaborative Consensus"]
        P4A[Multi-model discussion] --> P4B[Identify shared strengths]
        P4B --> P4C[Identify common weaknesses]
        P4C --> P4D[Confidence-weighted voting]
        P4D --> P4E[Generate synthesis]
    end

    subgraph Phase5["Phase 5: Final Ranking"]
        P5A[Aggregate all signals] --> P5B[Apply evaluation weights]
        P5B --> P5C[Bradley-Terry ranking]
        P5C --> P5D[Compute confidence intervals]
        P5D --> P5E[Output final rankings]
    end

    Phase0 --> Phase1
    Phase1 --> Phase2
    Phase2 --> Phase3
    Phase3 --> Phase4
    Phase4 --> Phase5
```

### Pointwise Evaluation Detail

```mermaid
flowchart LR
    subgraph Rubrics
        R1[Accuracy]
        R2[Completeness]
        R3[Clarity]
        R4[Relevance]
        R5[Reasoning]
    end

    subgraph Responses
        RESP_A[Response A<br/>OpenAI]
        RESP_B[Response B<br/>Claude]
        RESP_C[Response C<br/>Gemini]
        RESP_D[Response D<br/>xAI]
    end

    subgraph Evaluators["Evaluator Assignment"]
        E1[Claude evaluates A, C, D]
        E2[OpenAI evaluates B, C, D]
        E3[Gemini evaluates A, B, D]
        E4[xAI evaluates A, B, C]
    end

    Rubrics --> Evaluators
    Responses --> Evaluators
    Evaluators --> AGG[Score Aggregation]
    AGG --> NORM[Normalized Scores]
```

### Pairwise Comparison Matrix

```mermaid
flowchart TB
    subgraph Pairs["All Pairwise Comparisons"]
        P1[A vs B]
        P2[A vs C]
        P3[A vs D]
        P4[B vs C]
        P5[B vs D]
        P6[C vs D]
    end

    subgraph Judging["Each pair judged by non-participants"]
        P1 --> J1[Gemini & xAI judge]
        P2 --> J2[Claude & xAI judge]
        P3 --> J3[Claude & Gemini judge]
        P4 --> J4[OpenAI & xAI judge]
        P5 --> J5[OpenAI & Gemini judge]
        P6 --> J6[OpenAI & Claude judge]
    end

    J1 & J2 & J3 & J4 & J5 & J6 --> MAT[Win/Loss Matrix]
    MAT --> BT[Bradley-Terry Input]
```

### Adversarial Debate Structure

```mermaid
sequenceDiagram
    participant ADV as Advocate
    participant CHL as Challenger
    participant JDG as Judge

    Note over ADV,JDG: Debate for Response X

    rect rgb(200, 230, 200)
        Note right of ADV: Round 1
        ADV->>JDG: Present strengths of Response X
        CHL->>JDG: Present weaknesses of Response X
    end

    rect rgb(230, 200, 200)
        Note right of ADV: Round 2
        ADV->>JDG: Rebut challenger's points
        CHL->>JDG: Counter advocate's defense
    end

    rect rgb(200, 200, 230)
        Note right of JDG: Judgment
        JDG->>JDG: Evaluate argument quality
        JDG->>JDG: Score response based on debate
    end
```

---

## Data Flow

### Session Data Structure

```mermaid
erDiagram
    SESSION ||--o{ RESPONSE : contains
    SESSION ||--o{ EVALUATION : contains
    SESSION ||--|| RANKING : produces
    SESSION ||--|| REPORT : generates

    SESSION {
        string session_id PK
        datetime timestamp
        string prompt
        json config
    }

    RESPONSE {
        string provider_name PK
        string model_id
        string text
        int latency_ms
        int token_count
    }

    EVALUATION {
        string eval_type PK
        json pointwise_scores
        json pairwise_results
        json debate_transcript
        json consensus_data
    }

    RANKING {
        json final_order
        json scores
        json confidence_intervals
    }

    REPORT {
        string pdf_path
        datetime generated_at
    }
```

### File System Structure

```mermaid
flowchart TD
    ROOT[llm_compare/] --> EVAL_DIR[evaluations/]
    ROOT --> DOCS[docs/]
    ROOT --> SRC[source code]

    EVAL_DIR --> SESS1[session_abc123/]
    EVAL_DIR --> SESS2[session_def456/]

    SESS1 --> DATA1[data.json]
    SESS1 --> RPT1[report.pdf]
    SESS1 --> LOG1[session.log]

    SESS2 --> DATA2[data.json]
    SESS2 --> RPT2[report.pdf]
    SESS2 --> LOG2[session.log]
```

---

## API Integration Patterns

### Request/Response Flow

```mermaid
sequenceDiagram
    participant SM as Session Manager
    participant PM as Provider Manager
    participant API as LLM API
    participant RETRY as Retry Handler

    SM->>PM: generate_all(prompt)

    par Parallel API Calls
        PM->>API: OpenAI request
        PM->>API: Claude request
        PM->>API: Gemini request
        PM->>API: xAI request
    end

    alt Success
        API-->>PM: Response
        PM-->>SM: Collected responses
    else Rate Limited
        API-->>RETRY: 429 Error
        RETRY->>RETRY: Exponential backoff
        RETRY->>API: Retry request
    else Timeout
        API-->>RETRY: Timeout
        RETRY->>RETRY: Retry with extended timeout
        RETRY->>API: Retry request
    else Provider Down
        API-->>PM: Connection error
        PM->>PM: Mark provider unavailable
        PM-->>SM: Partial results + warning
    end
```

### Error Handling Strategy

```mermaid
flowchart TD
    REQ[API Request] --> TRY{Try Request}

    TRY -->|Success| RESP[Return Response]
    TRY -->|Error| ERR{Error Type?}

    ERR -->|Rate Limit 429| BACK[Exponential Backoff]
    ERR -->|Timeout| RETRY_TO[Retry with 2x timeout]
    ERR -->|Auth Error 401| FAIL_AUTH[Disable Provider]
    ERR -->|Server Error 5xx| RETRY_SRV[Retry up to 3x]
    ERR -->|Connection Error| RETRY_CONN[Retry with delay]

    BACK --> COUNT{Retry Count < 5?}
    RETRY_TO --> COUNT
    RETRY_SRV --> COUNT
    RETRY_CONN --> COUNT

    COUNT -->|Yes| TRY
    COUNT -->|No| FAIL[Log Error, Continue Without]

    FAIL_AUTH --> FAIL
```

---

## Report Generation

### Report Structure

```mermaid
flowchart TD
    subgraph Cover["Cover Page"]
        C1[Session ID]
        C2[Timestamp]
        C3[Providers Used]
    end

    subgraph Prompt["Section 1: Prompt"]
        S1[Original User Prompt]
    end

    subgraph Responses["Section 2: Responses"]
        S2A[OpenAI Response]
        S2B[Claude Response]
        S2C[Gemini Response]
        S2D[xAI Response]
    end

    subgraph Pointwise["Section 3: Pointwise Scores"]
        S3A[Score Tables by Rubric]
        S3B[Aggregated Scores]
    end

    subgraph Pairwise["Section 4: Pairwise Results"]
        S4A[Comparison Matrix]
        S4B[Win/Loss Summary]
    end

    subgraph Debate["Section 5: Adversarial Debate"]
        S5A[Debate Transcripts]
        S5B[Argument Analysis]
    end

    subgraph Consensus["Section 6: Collaborative Consensus"]
        S6A[Discussion Summary]
        S6B[Identified Strengths]
        S6C[Identified Weaknesses]
    end

    subgraph Rankings["Section 7: Final Rankings"]
        S7A[Ranked List 1st to Last]
        S7B[Score Breakdown]
        S7C[Confidence Intervals]
    end

    Cover --> Prompt --> Responses --> Pointwise --> Pairwise --> Debate --> Consensus --> Rankings
```

---

## Configuration System

### Configuration Hierarchy

```mermaid
flowchart TD
    DEF[Default Config] --> FILE[config.yaml]
    FILE --> ENV[Environment Variables]
    ENV --> CLI_ARG[CLI Arguments]
    CLI_ARG --> FINAL[Final Configuration]

    subgraph Defaults
        D1[timeout: 120s]
        D2[retries: 3]
        D3[debate_rounds: 2]
        D4[weights: 30/30/25/15]
    end

    subgraph Overrides
        O1[--timeout 180]
        O2[--providers openai,claude]
        O3[--debate-rounds 3]
    end

    Defaults --> DEF
    Overrides --> CLI_ARG
```

---

## Module Dependencies

```mermaid
flowchart BT
    subgraph Utils["utils/"]
        LOG[logging.py]
        RETRY[retry.py]
    end

    subgraph Providers["providers/"]
        BASE[base.py]
        OAI[openai_provider.py]
        CLD[claude_provider.py]
        GEM[gemini_provider.py]
        XAI_P[xai_provider.py]
        DISC[discovery.py]
    end

    subgraph Evaluation["evaluation/"]
        RUB[rubrics.py]
        POINT[pointwise.py]
        PAIR[pairwise.py]
        ADV[adversarial.py]
        COLL[collaborative.py]
        RANK_E[ranking.py]
    end

    subgraph Session["session/"]
        MGR[manager.py]
        STOR[storage.py]
    end

    subgraph Report["report/"]
        GEN[generator.py]
        TMPL[templates.py]
    end

    MAIN[main.py] --> MGR
    MGR --> DISC
    MGR --> POINT & PAIR & ADV & COLL
    MGR --> RANK_E
    MGR --> GEN
    MGR --> STOR

    DISC --> BASE
    OAI & CLD & GEM & XAI_P --> BASE
    BASE --> LOG & RETRY

    POINT & PAIR & ADV & COLL --> RUB
    POINT & PAIR & ADV & COLL --> BASE

    GEN --> TMPL
    GEN --> STOR
```

---

## Deployment View

```mermaid
flowchart TB
    subgraph Local["Local Machine"]
        CLI_APP[CLI Application]
        KEYS[API Key Files]
        OUT[Output Directory]
    end

    subgraph External["External APIs"]
        OAI_API[OpenAI API]
        ANT_API[Anthropic API]
        GOO_API[Google AI API]
        XAI_API[xAI API]
    end

    CLI_APP --> |HTTPS| OAI_API
    CLI_APP --> |HTTPS| ANT_API
    CLI_APP --> |HTTPS| GOO_API
    CLI_APP --> |HTTPS| XAI_API

    KEYS --> CLI_APP
    CLI_APP --> OUT
```
