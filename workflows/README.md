# Workflow Templates

This directory contains ComfyUI workflow templates that serve as **atomic tools** in the vibe-aigc system.

## Philosophy (Paper Section 5.4)

> "Traverse atomic tool library, select optimal ensemble of components"

Each workflow here is a **reusable capability**:
- Exported from ComfyUI (graph or API format)
- Parameterizable (prompt, seed, resolution)
- Discoverable by the WorkflowRegistry
- Selectable by the MetaPlanner

## Workflow Format

Workflows can be in either format:
1. **Graph format** - Exported from ComfyUI UI (includes positions, links)
2. **API format** - Prompt dict (minimal, just nodes and inputs)

## Adding New Workflows

1. Create your workflow in ComfyUI
2. Export/save as JSON
3. Place in this directory
4. Optionally add `_vibe_metadata`:

```json
{
  "_vibe_metadata": {
    "name": "my_workflow",
    "description": "What this workflow does",
    "capabilities": ["text_to_video", "image_to_video"],
    "parameters": {
      "prompt": {"node": "6", "input": "text"},
      "seed": {"node": "9", "input": "noise_seed"}
    },
    "tags": ["wan", "video", "svi"]
  },
  "nodes": [...]
}
```

## Available Workflows

| Workflow | Capability | Description |
|----------|------------|-------------|
| wan22_i2v_svi_pro.json | image_to_video | Wan 2.2 I2V with SVI Pro LoRAs |

## Usage

```python
from vibe_aigc.workflow_registry import create_registry, WorkflowRunner

# Discover workflows
registry = create_registry(comfyui_url="http://192.168.1.143:8188")
registry.discover()

# Get workflow for capability
workflows = registry.get_for_capability(WorkflowCapability.IMAGE_TO_VIDEO)
workflow = workflows[0]

# Run with parameters
runner = WorkflowRunner("http://192.168.1.143:8188")
result = await runner.run(
    workflow,
    prompt="cyberpunk samurai in neon rain",
    seed=42
)
```
