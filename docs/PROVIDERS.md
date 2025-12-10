# Provider Integration Documentation

This document details how LLM providers are discovered, configured, and used within the system.

## Provider Discovery

### Automatic Detection

Providers are automatically discovered based on API key files in the project root and local GGUF model files.

```mermaid
flowchart TD
    START[Application Start] --> SCAN[Scan for *.key.txt files]
    SCAN --> LOCAL[Scan for local GGUF models]
    LOCAL --> FOUND{Providers Found?}

    FOUND -->|No| ERROR[Error: No providers available]
    FOUND -->|Yes| PARSE[Parse file names]

    PARSE --> MAP[Map to provider classes]

    subgraph Cloud["Cloud API Mapping"]
        F1[openai.key.txt] --> P1[OpenAIProvider]
        F2[claude.key.txt] --> P2[ClaudeProvider]
        F3[gemini.key.txt] --> P3[GeminiProvider]
        F4[xai.key.txt] --> P4[XAIProvider]
    end

    subgraph Local["Local Model Mapping"]
        F5[*.gguf files] --> P5[LlamaCppProvider]
        F6[llamacpp.config.json] --> P5
    end

    MAP --> LOAD[Load API keys / models]
    LOAD --> VALIDATE[Validate keys / files]
    VALIDATE --> READY[Providers Ready]
```

### Key File Format

Each key file contains the API key on the first line:

```
sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

| File Name | Provider | API Endpoint |
|-----------|----------|--------------|
| `openai.key.txt` | OpenAI | api.openai.com |
| `claude.key.txt` | Anthropic | api.anthropic.com |
| `gemini.key.txt` | Google | generativelanguage.googleapis.com |
| `xai.key.txt` | xAI | api.x.ai |

### Local Model Configuration

Local GGUF models are discovered via:
1. **Auto-discovery**: Scanning `model_dirs` for `*.gguf` files
2. **Explicit config**: Models defined in `llamacpp.config.json`

See [llama.cpp Provider](#llamacpp-provider-local-models) section for details.

---

## Provider Interface

### Abstract Base Class

```mermaid
classDiagram
    class LLMProvider {
        <<abstract>>
        #api_key: str
        #client: Any
        #config: ProviderConfig
        +name: str*
        +model_id: str*
        +generate(prompt: str, **kwargs) Response*
        +evaluate(response: str, rubric: Rubric) Evaluation*
        +get_token_count(text: str) int
        +estimate_cost(tokens: int) float
        #_create_client() Any
        #_handle_error(error: Exception) None
        #_retry_with_backoff(func: Callable) Any
    }

    class Response {
        +text: str
        +model: str
        +provider: str
        +latency_ms: int
        +input_tokens: int
        +output_tokens: int
        +finish_reason: str
    }

    class Evaluation {
        +scores: Dict~str, float~
        +reasoning: Dict~str, str~
        +evaluator: str
        +evaluated: str
    }

    class ProviderConfig {
        +timeout: int
        +max_retries: int
        +temperature: float
        +max_tokens: int
    }

    LLMProvider ..> Response : creates
    LLMProvider ..> Evaluation : creates
    LLMProvider --> ProviderConfig : uses
```

### Interface Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `generate(prompt, **kwargs)` | Generate response to prompt | `Response` |
| `evaluate(response, rubric)` | Score a response against rubric | `Evaluation` |
| `get_token_count(text)` | Count tokens in text | `int` |
| `estimate_cost(tokens)` | Estimate API cost | `float` |

---

## Provider Implementations

### OpenAI Provider

```mermaid
flowchart LR
    subgraph OpenAI["OpenAI Provider"]
        CLIENT[OpenAI Client]
        MODEL[gpt-4-turbo]
    end

    subgraph Features
        F1[Function calling]
        F2[JSON mode]
        F3[Vision support]
    end

    subgraph Config
        C1[timeout: 120s]
        C2[max_tokens: 4096]
        C3[temperature: 0.7]
    end

    Config --> OpenAI
    OpenAI --> Features
```

**Configuration:**
```python
class OpenAIProvider(LLMProvider):
    name = "openai"
    model_id = "gpt-4-turbo"

    default_config = {
        "timeout": 120,
        "max_tokens": 4096,
        "temperature": 0.7,
    }
```

**API Details:**
- Endpoint: `https://api.openai.com/v1/chat/completions`
- Authentication: Bearer token in header
- Rate limits: Varies by tier (TPM/RPM)

### Claude Provider

```mermaid
flowchart LR
    subgraph Claude["Claude Provider"]
        CLIENT[Anthropic Client]
        MODEL[claude-3-opus-20240229]
    end

    subgraph Features
        F1[Long context 200k]
        F2[Tool use]
        F3[Vision support]
    end

    subgraph Config
        C1[timeout: 120s]
        C2[max_tokens: 4096]
        C3[temperature: 0.7]
    end

    Config --> Claude
    Claude --> Features
```

**Configuration:**
```python
class ClaudeProvider(LLMProvider):
    name = "claude"
    model_id = "claude-3-opus-20240229"

    default_config = {
        "timeout": 120,
        "max_tokens": 4096,
        "temperature": 0.7,
    }
```

**API Details:**
- Endpoint: `https://api.anthropic.com/v1/messages`
- Authentication: `x-api-key` header
- Rate limits: Varies by tier

### Gemini Provider

```mermaid
flowchart LR
    subgraph Gemini["Gemini Provider"]
        CLIENT[GenerativeModel]
        MODEL[gemini-pro]
    end

    subgraph Features
        F1[Multimodal]
        F2[Long context]
        F3[Grounding]
    end

    subgraph Config
        C1[timeout: 120s]
        C2[max_tokens: 4096]
        C3[temperature: 0.7]
    end

    Config --> Gemini
    Gemini --> Features
```

**Configuration:**
```python
class GeminiProvider(LLMProvider):
    name = "gemini"
    model_id = "gemini-pro"

    default_config = {
        "timeout": 120,
        "max_output_tokens": 4096,
        "temperature": 0.7,
    }
```

**API Details:**
- Uses `google-generativeai` SDK
- Authentication: API key in SDK configuration
- Different parameter naming (max_output_tokens vs max_tokens)

### xAI Provider

```mermaid
flowchart LR
    subgraph xAI["xAI Provider"]
        CLIENT[OpenAI-compatible Client]
        MODEL[grok-beta]
    end

    subgraph Features
        F1[Real-time data]
        F2[OpenAI compatible]
    end

    subgraph Config
        C1[timeout: 120s]
        C2[max_tokens: 4096]
        C3[temperature: 0.7]
        C4[base_url: api.x.ai]
    end

    Config --> xAI
    xAI --> Features
```

**Configuration:**
```python
class XAIProvider(LLMProvider):
    name = "xai"
    model_id = "grok-beta"

    default_config = {
        "timeout": 120,
        "max_tokens": 4096,
        "temperature": 0.7,
        "base_url": "https://api.x.ai/v1",
    }
```

**API Details:**
- Uses OpenAI client with custom base URL
- OpenAI-compatible API format
- Authentication: Bearer token

### llama.cpp Provider (Local Models)

```mermaid
flowchart LR
    subgraph LlamaCpp["llama.cpp Provider"]
        CLIENT[llama-cpp-python]
        MODEL[GGUF Model File]
    end

    subgraph Features
        F1[Local inference]
        F2[GPU acceleration]
        F3[Quantized models]
        F4[No API costs]
    end

    subgraph Config
        C1[n_ctx: 4096]
        C2[n_gpu_layers: 0-35]
        C3[chat_format: auto]
        C4[temperature: 0.7]
    end

    Config --> LlamaCpp
    LlamaCpp --> Features
```

**Configuration:**
```python
class LlamaCppProvider(LLMProvider):
    name = "llamacpp-{model_name}"
    model_id = "{gguf_filename}"

    default_config = {
        "timeout": 300,
        "max_tokens": 4096,
        "temperature": 0.7,
        "n_ctx": 4096,
        "n_gpu_layers": 0,
    }
```

**Setup Requirements:**
- Install: `pip install llama-cpp-python`
- GPU support: `CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python`
- Place GGUF model files in configured directories

**Model Discovery:**

```mermaid
flowchart TD
    START[Provider Discovery] --> CONFIG{llamacpp.config.json exists?}

    CONFIG -->|Yes| LOAD[Load configuration]
    CONFIG -->|No| DEFAULTS[Use default settings]

    LOAD --> EXPLICIT[Load explicitly configured models]
    DEFAULTS --> SCAN[Scan model_dirs for *.gguf]

    EXPLICIT --> SCAN
    SCAN --> FOUND{GGUF files found?}

    FOUND -->|Yes| CREATE[Create LlamaCppProvider instances]
    FOUND -->|No| SKIP[Skip local models]

    CREATE --> INFER[Infer chat format from filename]
    INFER --> READY[Providers ready]
```

**Chat Format Inference:**

| Filename Pattern | Chat Format |
|------------------|-------------|
| `*llama-2*`, `*llama2*` | llama-2 |
| `*llama-3*`, `*llama3*` | llama-3 |
| `*mistral*` | mistral-instruct |
| `*qwen*`, `*phi*` | chatml |
| `*gemma*` | gemma |
| `*vicuna*` | vicuna |
| `*alpaca*` | alpaca |

**Configuration File (`llamacpp.config.json`):**
```json
{
  "model_dirs": ["./models", "~/models"],
  "default_n_ctx": 4096,
  "default_n_gpu_layers": 0,
  "models": {
    "llama3-8b": {
      "path": "./models/llama-3-8b-instruct.Q4_K_M.gguf",
      "n_ctx": 8192,
      "n_gpu_layers": 35,
      "chat_format": "llama-3"
    }
  }
}
```

**Key Differences from Cloud Providers:**
- No API key required
- Runs entirely locally
- No per-token costs
- Performance depends on hardware (CPU/GPU)
- Lazy loading: llama-cpp-python only imported when models are used

---

## Error Handling

### Error Categories

```mermaid
flowchart TD
    ERR[API Error] --> TYPE{Error Type}

    TYPE --> AUTH[Authentication<br/>401/403]
    TYPE --> RATE[Rate Limit<br/>429]
    TYPE --> SERVER[Server Error<br/>5xx]
    TYPE --> TIMEOUT[Timeout]
    TYPE --> CONN[Connection<br/>Error]

    AUTH --> DISABLE[Disable Provider]
    RATE --> BACKOFF[Exponential Backoff]
    SERVER --> RETRY[Retry 3x]
    TIMEOUT --> EXTEND[Extend Timeout + Retry]
    CONN --> RETRY_CONN[Retry with Delay]
```

### Retry Strategy

```mermaid
sequenceDiagram
    participant APP as Application
    participant RETRY as Retry Handler
    participant API as Provider API

    APP->>RETRY: Request
    RETRY->>API: Attempt 1

    alt Success
        API-->>RETRY: Response
        RETRY-->>APP: Response
    else Retryable Error
        API-->>RETRY: Error
        RETRY->>RETRY: Wait 1s
        RETRY->>API: Attempt 2

        alt Success
            API-->>RETRY: Response
            RETRY-->>APP: Response
        else Error
            RETRY->>RETRY: Wait 2s
            RETRY->>API: Attempt 3

            alt Success
                API-->>RETRY: Response
                RETRY-->>APP: Response
            else Final Error
                API-->>RETRY: Error
                RETRY-->>APP: Raise Exception
            end
        end
    end
```

### Backoff Configuration

```python
retry_config = {
    "max_retries": 3,
    "initial_delay": 1.0,      # seconds
    "max_delay": 60.0,         # seconds
    "exponential_base": 2,
    "jitter": True,            # Add randomness to prevent thundering herd
}
```

---

## Concurrent Execution

### Parallel Request Pattern

```mermaid
flowchart TD
    PROMPT[User Prompt] --> DISPATCH[Dispatch to All Providers]

    subgraph Cloud["Cloud API Requests"]
        REQ1[OpenAI Request]
        REQ2[Claude Request]
        REQ3[Gemini Request]
        REQ4[xAI Request]
    end

    subgraph Local["Local Model Inference"]
        REQ5[llama.cpp Model 1]
        REQ6[llama.cpp Model 2]
    end

    DISPATCH --> REQ1 & REQ2 & REQ3 & REQ4
    DISPATCH --> REQ5 & REQ6

    REQ1 --> COLLECT[Response Collector]
    REQ2 --> COLLECT
    REQ3 --> COLLECT
    REQ4 --> COLLECT
    REQ5 --> COLLECT
    REQ6 --> COLLECT

    COLLECT --> TIMEOUT{All Complete or Timeout?}
    TIMEOUT -->|All Complete| SUCCESS[All Responses]
    TIMEOUT -->|Timeout| PARTIAL[Partial Results + Warnings]
```

### Concurrency Control

```python
async def generate_all(prompt: str) -> Dict[str, Response]:
    semaphore = asyncio.Semaphore(4)  # Max concurrent requests

    async def bounded_generate(provider):
        async with semaphore:
            return await provider.generate(prompt)

    tasks = [bounded_generate(p) for p in self.providers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        p.name: r for p, r in zip(self.providers, results)
        if not isinstance(r, Exception)
    }
```

---

## Cost Tracking

### Token Counting

```mermaid
flowchart LR
    TEXT[Input/Output Text] --> TOKENIZE[Provider Tokenizer]
    TOKENIZE --> COUNT[Token Count]
    COUNT --> COST[Cost Calculation]

    subgraph Pricing["Pricing per 1M tokens"]
        P1[GPT-4-turbo: $10/$30]
        P2[Claude-3-opus: $15/$75]
        P3[Gemini-pro: $0.50/$1.50]
        P4[Grok: TBD]
        P5[llama.cpp: $0 local]
    end

    Pricing --> COST
    COST --> TOTAL[Session Total]
```

### Cost Estimation

```python
pricing = {
    "openai": {
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},  # per 1M tokens
    },
    "claude": {
        "claude-3-opus": {"input": 15.00, "output": 75.00},
    },
    "gemini": {
        "gemini-pro": {"input": 0.50, "output": 1.50},
    },
    "llamacpp": {
        "*": {"input": 0.00, "output": 0.00},  # Local inference - no API cost
    },
}

def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    if provider.startswith("llamacpp"):
        return 0.0  # Local models have no API cost
    rates = pricing[provider][model]
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
```

---

## Provider Health Monitoring

### Health Check Flow

```mermaid
stateDiagram-v2
    [*] --> Healthy
    Healthy --> Degraded : Slow responses
    Healthy --> Unhealthy : Errors
    Degraded --> Healthy : Normal response
    Degraded --> Unhealthy : Continued issues
    Unhealthy --> Degraded : Successful retry
    Unhealthy --> Disabled : Too many failures
    Disabled --> Unhealthy : Manual re-enable
```

### Metrics Tracked

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Latency | Response time | > 30s |
| Error Rate | Failed requests / total | > 10% |
| Timeout Rate | Timeouts / total | > 5% |
| Availability | Successful / attempted | < 95% |

---

## Adding New Providers

### Steps to Add a Cloud Provider

1. Create key file pattern (e.g., `newprovider.key.txt`)
2. Implement provider class extending `LLMProvider`
3. Register in provider discovery mapping
4. Add default configuration
5. Implement error handling specific to provider
6. Add cost tracking data

### Steps to Add a Local Provider

For local inference engines (like llama.cpp):
1. Create configuration file format (e.g., `engine.config.json`)
2. Implement provider class with lazy imports for optional dependencies
3. Add model discovery logic to scan for model files
4. Handle hardware-specific configuration (GPU layers, context size)
5. No API key or cost tracking needed

### Provider Template

```python
class NewProvider(LLMProvider):
    name = "newprovider"
    model_id = "model-name"

    default_config = {
        "timeout": 120,
        "max_tokens": 4096,
        "temperature": 0.7,
    }

    def _create_client(self):
        # Initialize API client
        pass

    def generate(self, prompt: str, **kwargs) -> Response:
        # Implement generation logic
        pass

    def evaluate(self, response: str, rubric: Rubric) -> Evaluation:
        # Implement evaluation logic (usually uses generate internally)
        pass
```
