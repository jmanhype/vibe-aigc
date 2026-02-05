# Vibe AIGC

**A New Paradigm for Content Generation via Agentic Orchestration**

[![CI](https://github.com/jmanhype/vibe-aigc/actions/workflows/ci.yml/badge.svg)](https://github.com/jmanhype/vibe-aigc/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/vibe-aigc.svg)](https://pypi.org/project/vibe-aigc/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is Vibe AIGC?

Vibe AIGC introduces a paradigm shift from model-centric to **agentic orchestration** for content generation. Based on the research paper [arXiv:2602.04575](https://arxiv.org/abs/2602.04575), it bridges the **Intent-Execution Gap** between what creators want and what AI systems produce.

Instead of traditional prompt engineering, users provide a **Vibe** — a high-level representation encompassing aesthetic preferences, functional logic, and intent. A **Meta-Planner** then decomposes this into executable, adaptive workflows.

```python
from vibe_aigc import MetaPlanner, Vibe

# Express your intent as a Vibe
vibe = Vibe(
    description="Create a cinematic sci-fi trailer",
    style="dark, atmospheric, Blade Runner aesthetic",
    constraints=["under 60 seconds", "no dialogue"]
)

# Let the Meta-Planner handle the rest
planner = MetaPlanner()
result = await planner.execute(vibe)
```

## Key Features

<div class="grid cards" markdown>

-   :dart: **Vibe-based Planning**

    ---

    Decompose high-level creative intent into executable workflows automatically

-   :zap: **Parallel Execution**

    ---

    Independent nodes execute concurrently for faster results

-   :arrows_counterclockwise: **Adaptive Replanning**

    ---

    Automatic workflow adaptation based on execution feedback

-   :floppy_disk: **Checkpoint/Resume**

    ---

    Save and restore workflow state for long-running tasks

-   :chart_with_upwards_trend: **Progress Tracking**

    ---

    Real-time callbacks and visualization during execution

-   :art: **Workflow Visualization**

    ---

    ASCII and Mermaid diagram generation

</div>

## Architecture

```
User Vibe → MetaPlanner → Agentic Pipeline → Execution → Result
     ↑                           ↓
     └──── Feedback Loop ────────┘
```

The system follows a closed-loop architecture where:

1. **User** provides a high-level Vibe (intent + constraints)
2. **MetaPlanner** decomposes it into a hierarchical workflow
3. **Executor** runs nodes in parallel where possible
4. **Feedback** flows back for adaptive replanning on failures

## Quick Install

```bash
pip install vibe-aigc
```

## Next Steps

- [Installation Guide](getting-started/installation.md) - Detailed setup instructions
- [Quick Start](getting-started/quickstart.md) - Your first Vibe workflow
- [Core Concepts](guide/concepts.md) - Understanding the architecture
- [API Reference](api/models.md) - Complete API documentation
