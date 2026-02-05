# Vibe AIGC

**A New Paradigm for Content Generation via Agentic Orchestration**

Based on [arXiv:2602.04575](https://arxiv.org/abs/2602.04575)

## Overview

Vibe AIGC introduces a paradigm shift from model-centric to agentic orchestration for content generation. Instead of traditional prompt engineering, users provide a **Vibe** — a high-level representation encompassing aesthetic preferences, functional logic, and intent.

## Key Concepts

### The Intent-Execution Gap
The fundamental disparity between a creator's high-level intent and the stochastic, black-box nature of current single-shot models.

### Meta-Planner
A centralized system architect that deconstructs a user's "Vibe" into executable, verifiable, and adaptive agentic pipelines.

### Hierarchical Multi-Agent Workflows
Autonomous synthesis of agent workflows that bridge human imagination and machine execution.

## Architecture

```
User Vibe → Meta-Planner → Agentic Pipeline → Execution → Result
     ↑                           ↓
     └──── Feedback Loop ────────┘
```

## Installation

```bash
pip install -e .
```

## Usage

```python
from vibe_aigc import MetaPlanner, Vibe

# Define your vibe
vibe = Vibe(
    description="Create a cinematic sci-fi scene",
    style="dark, atmospheric",
    constraints=["no violence", "PG-13"]
)

# Plan and execute
planner = MetaPlanner()
result = await planner.execute(vibe)
```

## License

MIT
