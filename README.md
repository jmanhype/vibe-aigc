# Vibe AIGC

[![CI](https://github.com/jmanhype/vibe-aigc/actions/workflows/ci.yml/badge.svg)](https://github.com/jmanhype/vibe-aigc/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A New Paradigm for Content Generation via Agentic Orchestration**

Based on [arXiv:2602.04575](https://arxiv.org/abs/2602.04575)

## Overview

Vibe AIGC introduces a paradigm shift from model-centric to agentic orchestration for content generation. Instead of traditional prompt engineering, users provide a **Vibe** — a high-level representation encompassing aesthetic preferences, functional logic, and intent.

## Key Features

- 🎯 **Vibe-based Planning** — Decompose high-level intent into executable workflows
- ⚡ **Parallel Execution** — Independent nodes execute concurrently for faster results
- 🔄 **Adaptive Replanning** — Automatic workflow adaptation based on execution feedback
- 💾 **Checkpoint/Resume** — Save and restore workflow state for long-running tasks
- 📊 **Progress Tracking** — Real-time callbacks and visualization
- 🎨 **Workflow Visualization** — ASCII and Mermaid diagram generation

## Architecture

```
User Vibe → MetaPlanner → Agentic Pipeline → Execution → Result
     ↑                           ↓
     └──── Feedback Loop ────────┘
```

## Installation

```bash
pip install -e .
```

## Quick Start

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

## Advanced Usage

### With Progress Callbacks

```python
def on_progress(event):
    print(f"[{event.event_type}] {event.message}")

planner = MetaPlanner(progress_callback=on_progress)
result = await planner.execute_with_visualization(vibe)
```

### With Checkpoint/Resume

```python
# Enable automatic checkpointing
planner = MetaPlanner(checkpoint_interval=5)

# Execute with checkpoints
result = await planner.execute_with_resume(vibe)

# Resume from a checkpoint
checkpoints = planner.list_checkpoints()
result = await planner.execute_with_resume(vibe, checkpoint_id=checkpoints[0]["checkpoint_id"])
```

### With Adaptive Replanning

```python
# Automatically adapt workflow on failures
result = await planner.execute_with_adaptation(vibe)

# Check adaptation history
print(result["adaptation_info"])
```

## API Reference

### Vibe

```python
Vibe(
    description: str,      # What you want to create
    style: str = None,     # Aesthetic preferences
    constraints: list = [],# Boundaries and requirements
    domain: str = None,    # Content domain
    metadata: dict = {}    # Additional context
)
```

### MetaPlanner

| Method | Description |
|--------|-------------|
| `plan(vibe)` | Generate workflow plan without execution |
| `execute(vibe)` | Plan and execute workflow |
| `execute_with_adaptation(vibe)` | Execute with automatic replanning |
| `execute_with_visualization(vibe)` | Execute with progress visualization |
| `execute_with_resume(vibe, checkpoint_id)` | Execute with checkpoint support |
| `list_checkpoints()` | List available checkpoints |
| `get_checkpoint(id)` | Load a specific checkpoint |
| `delete_checkpoint(id)` | Remove a checkpoint |

## Testing

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_planner.py -v

# Run with coverage
pytest --cov=vibe_aigc
```

## Project Structure

```
vibe_aigc/
├── __init__.py        # Package exports
├── models.py          # Vibe, WorkflowPlan, WorkflowNode
├── llm.py             # LLM client for plan generation
├── planner.py         # MetaPlanner orchestration
├── executor.py        # Workflow execution engine
├── persistence.py     # Checkpoint management
└── visualization.py   # Diagram generation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

MIT

## Citation

```bibtex
@article{vibe-aigc-2025,
  title={Vibe AIGC: A New Paradigm for Content Generation via Agentic Orchestration},
  journal={arXiv preprint arXiv:2602.04575},
  year={2025}
}
```
