# LLM Compare

A multi-AI comparison tool that evaluates responses from multiple LLM providers using collaborative and adversarial evaluation strategies.

## Overview

LLM Compare sends the same prompt to all available AI providers, then orchestrates a comprehensive evaluation pipeline where the AIs review, critique, score, and rank each other's responses. The tool generates a detailed PDF report with the full evaluation trace.

## Features

- **Automatic Provider Discovery:** Detects available LLM providers via `*.key.txt` files
- **Dynamic Model Selection:** Automatically queries each provider's API to select the best available model
- **Multi-Provider Support:** OpenAI, Anthropic Claude, Google Gemini, xAI Grok
- **Local Model Support:** Run local GGUF models via llama.cpp alongside cloud APIs
- **Hybrid Evaluation Pipeline:**
  - Pointwise scoring against defined rubrics
  - Pairwise head-to-head comparisons
  - Adversarial debate rounds
  - Collaborative consensus building
- **Statistical Ranking:** Bradley-Terry model for final response ordering
- **PDF Reports:** Comprehensive reports with proper markdown rendering (headers, code blocks, lists, inline formatting)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm_compare

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## API Key Setup

Place API keys in the project root with the naming convention `{provider}.key.txt`:

| Provider | Key File |
|----------|----------|
| OpenAI | `openai.key.txt` |
| Anthropic | `claude.key.txt` |
| Google | `gemini.key.txt` |
| xAI | `xai.key.txt` |

Each file should contain only the API key on the first line.

**Note:** These files are excluded from git via `.gitignore`.

## Local Model Setup (llama.cpp)

To use local GGUF models alongside cloud APIs:

1. Install llama-cpp-python:
```bash
pip install llama-cpp-python
# For GPU acceleration (CUDA):
# CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
```

2. Place GGUF model files in `./models/` or configure custom paths

3. Create `llamacpp.config.json` (see `llamacpp.config.json.example`):
```json
{
  "model_dirs": ["./models"],
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

Local models will be discovered automatically and participate in evaluations alongside cloud APIs.

## Usage

```bash
# Basic usage - interactive prompt
python -m llm_compare

# Direct prompt
python -m llm_compare --prompt "Explain quantum entanglement"

# Specify providers
python -m llm_compare --providers openai,claude --prompt "Your prompt"

# Custom output directory
python -m llm_compare --output ./my_evaluations --prompt "Your prompt"
```

## Evaluation Pipeline

### Phase 1: Pointwise Evaluation
Each response scored individually against rubrics:
- Accuracy/Correctness
- Completeness
- Clarity/Coherence
- Relevance
- Reasoning Quality

### Phase 2: Pairwise Comparison
Tournament-style head-to-head comparisons between all responses.

### Phase 3: Adversarial Debate
D3-inspired debate with:
- Advocates defending responses
- Challengers attacking weaknesses
- Independent judges evaluating arguments

### Phase 4: Collaborative Consensus
Multi-model discussion to identify strengths, weaknesses, and synthesize insights.

### Phase 5: Final Ranking
Bradley-Terry aggregation of all evaluation signals.

## Output Structure

```
evaluations/
  {session-id}/
    data.json      # Full evaluation data
    report.pdf     # Formatted PDF report
```

## Configuration

See `config.py` for customizable settings:
- API timeouts and retry behavior
- Evaluation weights
- Rubric definitions
- Report formatting options

## Requirements

- Python 3.10+
- API keys for at least 2 LLM providers (for meaningful comparison)

## Documentation

All detailed documentation is in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture with Mermaid diagrams |
| [EVALUATION_PIPELINE.md](docs/EVALUATION_PIPELINE.md) | Detailed evaluation methodology |
| [PROVIDERS.md](docs/PROVIDERS.md) | Provider integration and API details |
| [DATA_MODELS.md](docs/DATA_MODELS.md) | Data structures and JSON schemas |
| [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Implementation roadmap |

Project files:
- `CLAUDE.md` - Quick architecture reference
- `VIBE_HISTORY.md` - Development history and decisions

## License

[License TBD]
