# Quick Start

This guide walks you through creating your first Vibe workflow.

## Basic Usage

### 1. Define a Vibe

A Vibe captures your high-level creative intent:

```python
from vibe_aigc import Vibe

vibe = Vibe(
    description="Create a blog post about AI agents",
    style="informative, engaging, accessible",
    constraints=["under 1000 words", "include code examples"],
    domain="content"
)
```

### 2. Create a MetaPlanner

The MetaPlanner decomposes your Vibe into executable workflows:

```python
from vibe_aigc import MetaPlanner

planner = MetaPlanner()
```

### 3. Plan and Execute

```python
import asyncio

async def main():
    # Plan only (inspect the workflow)
    plan = await planner.plan(vibe)
    print(f"Generated {len(plan.root_nodes)} workflow nodes")
    
    # Or plan and execute in one step
    result = await planner.execute(vibe)
    print(f"Execution status: {result.status}")

asyncio.run(main())
```

## With Progress Tracking

Monitor execution in real-time:

```python
def on_progress(event):
    print(f"[{event.event_type}] {event.node_id}: {event.message}")

planner = MetaPlanner(progress_callback=on_progress)
result = await planner.execute(vibe)
```

## With Visualization

Generate workflow diagrams:

```python
# Execute with ASCII visualization
result = await planner.execute_with_visualization(vibe)

# Output includes:
# Workflow Plan: plan-001
# ==================================================
# Source Vibe: Create a blog post about AI agents
# 
# ├── ✅ [analyze] Research topic (research)
# ├── ✅ [generate] Create outline (outline)
# └── ✅ [generate] Write content (write)
```

## With Checkpoints

Save progress for long-running tasks:

```python
# Enable automatic checkpointing every 5 nodes
planner = MetaPlanner(checkpoint_interval=5)

# Execute with checkpoint support
result = await planner.execute_with_resume(vibe)

# List available checkpoints
checkpoints = planner.list_checkpoints()

# Resume from a specific checkpoint
result = await planner.execute_with_resume(
    vibe, 
    checkpoint_id=checkpoints[0]["checkpoint_id"]
)
```

## With Adaptive Replanning

Automatically recover from failures:

```python
# Execute with automatic replanning on failures
result = await planner.execute_with_adaptation(vibe)

# Check adaptation history
if result.adaptation_info:
    print(f"Adaptations made: {result.adaptation_info['adaptations']}")
```

## Complete Example

```python
import asyncio
from vibe_aigc import Vibe, MetaPlanner

async def main():
    # Define creative intent
    vibe = Vibe(
        description="Design a landing page for a SaaS product",
        style="modern, minimalist, professional",
        constraints=[
            "mobile-responsive",
            "fast loading",
            "clear call-to-action"
        ],
        domain="web design",
        metadata={
            "target_audience": "developers",
            "color_scheme": "dark mode"
        }
    )
    
    # Create planner with all features
    def on_progress(event):
        print(f"  → {event.message}")
    
    planner = MetaPlanner(
        progress_callback=on_progress,
        checkpoint_interval=3
    )
    
    # Execute with visualization and adaptation
    print("Starting workflow...")
    result = await planner.execute_with_visualization(vibe)
    
    # Check results
    summary = result.get_summary()
    print(f"\nCompleted: {summary['completed']}/{summary['total_nodes']} nodes")
    print(f"Status: {summary['status']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Next Steps

- [Core Concepts](../guide/concepts.md) - Understand the architecture
- [Workflows](../guide/workflows.md) - Deep dive into workflow creation
- [API Reference](../api/models.md) - Complete API documentation
