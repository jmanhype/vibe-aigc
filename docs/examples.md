# Examples

Practical examples of using Vibe AIGC.

## Basic Content Generation

```python
import asyncio
from vibe_aigc import Vibe, MetaPlanner

async def create_blog_post():
    vibe = Vibe(
        description="Write a blog post about the future of AI",
        style="thought-provoking, accessible",
        constraints=["800-1200 words", "include real-world examples"],
        domain="tech blog"
    )
    
    planner = MetaPlanner()
    result = await planner.execute(vibe)
    
    print(f"Status: {result.get_summary()['status']}")
    return result

asyncio.run(create_blog_post())
```

## With Progress Monitoring

```python
import asyncio
from vibe_aigc import Vibe, MetaPlanner

async def monitored_execution():
    vibe = Vibe(
        description="Design a marketing campaign",
        style="creative, bold",
        constraints=["social media focused", "budget-friendly"]
    )
    
    def on_progress(event):
        status_emoji = {
            "started": "🚀",
            "completed": "✅",
            "failed": "❌"
        }.get(event.event_type, "📍")
        print(f"{status_emoji} [{event.node_id}] {event.message}")
    
    planner = MetaPlanner(progress_callback=on_progress)
    result = await planner.execute(vibe)
    
    return result

asyncio.run(monitored_execution())
```

## Workflow Visualization

```python
import asyncio
from vibe_aigc import Vibe, MetaPlanner
from vibe_aigc.visualization import WorkflowVisualizer, VisualizationFormat

async def visualize_workflow():
    vibe = Vibe(
        description="Create a product launch plan",
        style="organized, comprehensive"
    )
    
    planner = MetaPlanner()
    
    # Generate plan without executing
    plan = await planner.plan(vibe)
    
    # ASCII visualization
    print("=== ASCII Diagram ===")
    ascii_viz = WorkflowVisualizer.generate_diagram(
        plan, 
        format=VisualizationFormat.ASCII
    )
    print(ascii_viz)
    
    # Mermaid visualization
    print("\n=== Mermaid Diagram ===")
    mermaid_viz = WorkflowVisualizer.generate_diagram(
        plan,
        format=VisualizationFormat.MERMAID
    )
    print(mermaid_viz)

asyncio.run(visualize_workflow())
```

## Checkpoint and Resume

```python
import asyncio
from vibe_aigc import Vibe, MetaPlanner

async def resumable_workflow():
    vibe = Vibe(
        description="Generate comprehensive documentation",
        style="technical, detailed",
        constraints=["include code examples", "API reference"]
    )
    
    # Enable checkpoints every 3 nodes
    planner = MetaPlanner(checkpoint_interval=3)
    
    try:
        # First attempt
        result = await planner.execute_with_resume(vibe)
    except Exception as e:
        print(f"Execution interrupted: {e}")
        
        # List available checkpoints
        checkpoints = planner.list_checkpoints()
        if checkpoints:
            print(f"Found {len(checkpoints)} checkpoint(s)")
            
            # Resume from latest
            result = await planner.execute_with_resume(
                vibe,
                checkpoint_id=checkpoints[0]["checkpoint_id"]
            )
    
    return result

asyncio.run(resumable_workflow())
```

## Adaptive Replanning

```python
import asyncio
from vibe_aigc import Vibe, MetaPlanner

async def adaptive_workflow():
    vibe = Vibe(
        description="Build a web scraping pipeline",
        style="robust, fault-tolerant",
        constraints=["handle rate limits", "retry on failure"]
    )
    
    planner = MetaPlanner()
    
    # Execute with automatic replanning on failures
    result = await planner.execute_with_adaptation(vibe)
    
    # Check if any adaptations were made
    summary = result.get_summary()
    print(f"Final status: {summary['status']}")
    print(f"Nodes completed: {summary['completed']}/{summary['total_nodes']}")
    
    if hasattr(result, 'adaptation_history'):
        print(f"Adaptations made: {len(result.adaptation_history)}")

asyncio.run(adaptive_workflow())
```

## Custom LLM Configuration

```python
import asyncio
from vibe_aigc import Vibe, MetaPlanner
from vibe_aigc.llm import LLMConfig

async def custom_llm():
    # Use a different model or endpoint
    config = LLMConfig(
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",  # Or your endpoint
        model="gpt-4o",
        temperature=0.7,
        max_tokens=4000
    )
    
    vibe = Vibe(
        description="Write creative fiction",
        style="imaginative, vivid",
        constraints=["short story format"]
    )
    
    planner = MetaPlanner(llm_config=config)
    result = await planner.execute(vibe)
    
    return result

asyncio.run(custom_llm())
```

## Manual Workflow Creation

```python
from vibe_aigc import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.visualization import WorkflowVisualizer, VisualizationFormat

# Create nodes manually for full control
nodes = [
    WorkflowNode(
        id="gather",
        type=WorkflowNodeType.ANALYZE,
        description="Gather requirements"
    ),
    WorkflowNode(
        id="design",
        type=WorkflowNodeType.GENERATE,
        description="Create system design",
        dependencies=["gather"]
    ),
    WorkflowNode(
        id="implement",
        type=WorkflowNodeType.GENERATE,
        description="Implement solution",
        dependencies=["design"]
    ),
    WorkflowNode(
        id="test",
        type=WorkflowNodeType.VALIDATE,
        description="Test implementation",
        dependencies=["implement"]
    ),
    WorkflowNode(
        id="document",
        type=WorkflowNodeType.GENERATE,
        description="Write documentation",
        dependencies=["implement"]
    ),
    WorkflowNode(
        id="deploy",
        type=WorkflowNodeType.TRANSFORM,
        description="Deploy to production",
        dependencies=["test", "document"]
    )
]

vibe = Vibe(description="Build and deploy a feature")
plan = WorkflowPlan(
    id="manual-plan",
    source_vibe=vibe,
    root_nodes=nodes
)

# Visualize
print(WorkflowVisualizer.generate_diagram(plan, format=VisualizationFormat.ASCII))
```
