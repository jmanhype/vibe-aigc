# Vibe AIGC

[![CI](https://github.com/jmanhype/vibe-aigc/actions/workflows/ci.yml/badge.svg)](https://github.com/jmanhype/vibe-aigc/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jmanhype/vibe-aigc/branch/main/graph/badge.svg)](https://codecov.io/gh/jmanhype/vibe-aigc)
[![PyPI](https://img.shields.io/pypi/v/vibe-aigc.svg)](https://pypi.org/project/vibe-aigc/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://jmanhype.github.io/vibe-aigc)

**A New Paradigm for Content Generation via Agentic Orchestration**

Based on [arXiv:2602.04575](https://arxiv.org/abs/2602.04575)

📚 **[Documentation](https://jmanhype.github.io/vibe-aigc)** | 🚀 **[Quick Start](https://jmanhype.github.io/vibe-aigc/getting-started/quickstart/)** | 📖 **[API Reference](https://jmanhype.github.io/vibe-aigc/api/models/)**

---

## What is Vibe AIGC?

Vibe AIGC bridges the **Intent-Execution Gap** in AI content generation. Instead of prompt engineering, you provide a **Vibe** — a high-level representation of your creative intent — and the system automatically decomposes it into executable workflows.

```python
from vibe_aigc import MetaPlanner, Vibe

# Express your intent
vibe = Vibe(
    description="Create a cinematic sci-fi trailer",
    style="dark, atmospheric, Blade Runner aesthetic",
    constraints=["under 60 seconds", "no dialogue"]
)

# Let the Meta-Planner handle the rest
planner = MetaPlanner()
result = await planner.execute(vibe)
```

## Architecture (Paper Section 5)

The implementation follows the paper's three-part architecture:

| Component | Purpose | Module |
|-----------|---------|--------|
| **MetaPlanner** | Decomposes Vibes into workflows | `vibe_aigc.planner` |
| **KnowledgeBase** | Domain expertise for intent understanding | `vibe_aigc.knowledge` |
| **ToolRegistry** | Atomic tools for content generation | `vibe_aigc.tools` |

```python
from vibe_aigc import MetaPlanner, Vibe, create_knowledge_base, create_default_registry

# The full architecture
kb = create_knowledge_base()  # Film, writing, design, music knowledge
tools = create_default_registry()  # LLM, templates, combine tools

planner = MetaPlanner(knowledge_base=kb, tool_registry=tools)

# Query knowledge for "Hitchcockian suspense" → technical specs
result = kb.query("Hitchcockian suspense")
# Returns: camera techniques, lighting specs, editing patterns
```

## Features

- 🎯 **Vibe-based Planning** — High-level intent → executable workflows
- 🧠 **Domain Knowledge** — Built-in expertise for film, writing, design, music
- 🔧 **Tool Library** — Pluggable tools for actual content generation
- ⚡ **Parallel Execution** — Independent nodes run concurrently
- 🔄 **Adaptive Replanning** — Automatic recovery from failures
- 💾 **Checkpoint/Resume** — Save and restore workflow state
- 📊 **Progress Tracking** — Real-time callbacks and visualization
- 🎨 **Workflow Visualization** — ASCII and Mermaid diagrams

## Installation

```bash
pip install vibe-aigc
```

## CLI Usage

```bash
# Generate a workflow plan
vibe-aigc plan "Create a blog post about AI" --style "informative" --format ascii

# Execute a vibe
vibe-aigc execute "Design a landing page" --visualize --checkpoint

# Manage checkpoints
vibe-aigc checkpoints --list
```

## Quick Example

```python
import asyncio
from vibe_aigc import Vibe, MetaPlanner

async def main():
    vibe = Vibe(
        description="Write a blog post about AI agents",
        style="informative, engaging",
        constraints=["under 1000 words"]
    )
    
    planner = MetaPlanner()
    result = await planner.execute_with_visualization(vibe)
    
    print(f"Status: {result.get_summary()['status']}")

asyncio.run(main())
```

## Architecture

```
User Vibe → MetaPlanner → Agentic Pipeline → Execution → Result
     ↑                           ↓
     └──── Feedback Loop ────────┘
```

## Documentation

- [Installation Guide](https://jmanhype.github.io/vibe-aigc/getting-started/installation/)
- [Quick Start Tutorial](https://jmanhype.github.io/vibe-aigc/getting-started/quickstart/)
- [Core Concepts](https://jmanhype.github.io/vibe-aigc/guide/concepts/)
- [API Reference](https://jmanhype.github.io/vibe-aigc/api/models/)
- [Examples](https://jmanhype.github.io/vibe-aigc/examples/)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{vibe-aigc-2025,
  title={Vibe AIGC: A New Paradigm for Content Generation via Agentic Orchestration},
  journal={arXiv preprint arXiv:2602.04575},
  year={2025}
}
```
