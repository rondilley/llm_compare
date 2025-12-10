# LLM Compare Documentation

## Overview

This directory contains comprehensive technical documentation for the LLM Compare tool.

## Documents

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture, component diagrams, data flow |
| [EVALUATION_PIPELINE.md](EVALUATION_PIPELINE.md) | Detailed evaluation methodology and prompts |
| [PROVIDERS.md](PROVIDERS.md) | LLM provider integration and API details |
| [DATA_MODELS.md](DATA_MODELS.md) | Data structures, schemas, and storage |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Implementation roadmap and task breakdown |

## Quick Links

### Architecture Highlights

- [System Overview](ARCHITECTURE.md#system-overview)
- [Component Architecture](ARCHITECTURE.md#component-architecture)
- [Module Dependencies](ARCHITECTURE.md#module-dependencies)

### Evaluation Pipeline

- [Phase 1: Pointwise Evaluation](EVALUATION_PIPELINE.md#phase-1-pointwise-evaluation)
- [Phase 2: Pairwise Comparison](EVALUATION_PIPELINE.md#phase-2-pairwise-comparison)
- [Phase 3: Adversarial Debate](EVALUATION_PIPELINE.md#phase-3-adversarial-debate)
- [Phase 4: Collaborative Consensus](EVALUATION_PIPELINE.md#phase-4-collaborative-consensus)
- [Phase 5: Final Ranking](EVALUATION_PIPELINE.md#phase-5-final-ranking)
- [Evaluation Prompts](EVALUATION_PIPELINE.md#evaluation-prompts)

### Provider Integration

- [Provider Discovery](PROVIDERS.md#provider-discovery)
- [Provider Interface](PROVIDERS.md#provider-interface)
- [Error Handling](PROVIDERS.md#error-handling)
- [Adding New Providers](PROVIDERS.md#adding-new-providers)

### Data Models

- [Session Model](DATA_MODELS.md#session-model)
- [Response Model](DATA_MODELS.md#response-model)
- [Evaluation Models](DATA_MODELS.md#evaluation-models)
- [Ranking Model](DATA_MODELS.md#ranking-model)

## Diagrams

All diagrams use [Mermaid](https://mermaid.js.org/) syntax and can be rendered by:
- GitHub (native support)
- VS Code with Mermaid extension
- Any Mermaid-compatible viewer

## Research References

The evaluation methodology is based on published research:

- **LLM-as-a-Judge**: [arXiv:2411.15594](https://arxiv.org/abs/2411.15594), [arXiv:2412.05579](https://arxiv.org/abs/2412.05579)
- **D3 Framework**: [arXiv:2410.04663](https://arxiv.org/abs/2410.04663)
- **Chatbot Arena / Bradley-Terry**: [LMSYS Blog](https://lmsys.org/blog/2023-12-07-leaderboard/)
- **Multi-Agent Consensus**: [arXiv:2501.06322](https://arxiv.org/html/2501.06322v1)
