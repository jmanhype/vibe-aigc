## Vibe AIGC v0.1.0

**A New Paradigm for Content Generation via Agentic Orchestration**

Based on [arXiv:2602.04575](https://arxiv.org/abs/2602.04575)

### Features

- 🎯 **Vibe-based Planning** - Express high-level intent, get executable workflows
- ⚡ **Parallel Execution** - Independent nodes run concurrently
- 🔄 **Adaptive Replanning** - Automatic recovery from failures
- 💾 **Checkpoint/Resume** - Save and restore workflow state
- 📊 **Progress Tracking** - Real-time callbacks and visualization
- 🎨 **Workflow Visualization** - ASCII and Mermaid diagrams
- 🖥️ **CLI Tool** - Command-line interface for quick operations

### Installation

```bash
pip install vibe-aigc
```

### Quick Start

```python
from vibe_aigc import MetaPlanner, Vibe

vibe = Vibe(
    description="Create a blog post about AI",
    style="informative, engaging"
)

planner = MetaPlanner()
result = await planner.execute(vibe)
```

### Links

- 📚 [Documentation](https://jmanhype.github.io/vibe-aigc)
- 📦 [PyPI](https://pypi.org/project/vibe-aigc/)
- 📄 [Paper](https://arxiv.org/abs/2602.04575)
