# Workflows

Deep dive into creating and managing workflows.

## Automatic Workflow Generation

The MetaPlanner automatically generates workflows from Vibes:

```python
from vibe_aigc import MetaPlanner, Vibe

vibe = Vibe(
    description="Create a technical tutorial",
    style="clear, step-by-step",
    constraints=["beginner-friendly"]
)

planner = MetaPlanner()
plan = await planner.plan(vibe)

# Inspect the generated plan
print(f"Plan ID: {plan.id}")
print(f"Nodes: {len(plan.root_nodes)}")
for node in plan.root_nodes:
    print(f"  - {node.type.value}: {node.description}")
```

## Manual Workflow Creation

For full control, create workflows manually:

```python
from vibe_aigc import WorkflowPlan, WorkflowNode, WorkflowNodeType, Vibe

# Define nodes
research = WorkflowNode(
    id="research",
    type=WorkflowNodeType.ANALYZE,
    description="Research the topic thoroughly"
)

outline = WorkflowNode(
    id="outline",
    type=WorkflowNodeType.GENERATE,
    description="Create content outline",
    dependencies=["research"]
)

write = WorkflowNode(
    id="write",
    type=WorkflowNodeType.GENERATE,
    description="Write the content",
    dependencies=["outline"]
)

review = WorkflowNode(
    id="review",
    type=WorkflowNodeType.VALIDATE,
    description="Review and refine",
    dependencies=["write"]
)

# Create plan
vibe = Vibe(description="Create content")
plan = WorkflowPlan(
    id="manual-plan-001",
    source_vibe=vibe,
    root_nodes=[research, outline, write, review]
)
```

## Hierarchical Workflows

Nodes can contain children for complex decomposition:

```python
# Parent node with children
generate_content = WorkflowNode(
    id="generate",
    type=WorkflowNodeType.COMPOSITE,
    description="Generate all content",
    children=[
        WorkflowNode(
            id="intro",
            type=WorkflowNodeType.GENERATE,
            description="Write introduction"
        ),
        WorkflowNode(
            id="body",
            type=WorkflowNodeType.GENERATE,
            description="Write main content"
        ),
        WorkflowNode(
            id="conclusion",
            type=WorkflowNodeType.GENERATE,
            description="Write conclusion"
        )
    ]
)
```

## Execution Options

### Basic Execution

```python
result = await planner.execute(vibe)
```

### With Visualization

```python
result = await planner.execute_with_visualization(vibe)
# Prints ASCII diagram during execution
```

### With Adaptation

```python
result = await planner.execute_with_adaptation(vibe)
# Automatically replans on failures
```

### With Checkpoints

```python
result = await planner.execute_with_resume(vibe)
# Saves checkpoints for recovery
```

## Handling Results

```python
result = await planner.execute(vibe)

# Get summary
summary = result.get_summary()
print(f"Status: {summary['status']}")
print(f"Completed: {summary['completed']}/{summary['total_nodes']}")

# Access individual node results
for node_id, node_result in result.node_results.items():
    print(f"{node_id}: {node_result.status.value}")
    if node_result.output:
        print(f"  Output: {node_result.output[:100]}...")
```

## Custom Executors

Extend the executor for custom behavior:

```python
from vibe_aigc.executor import WorkflowExecutor

class CustomExecutor(WorkflowExecutor):
    async def execute_node(self, node, context):
        # Custom execution logic
        print(f"Executing: {node.description}")
        result = await super().execute_node(node, context)
        # Post-processing
        return result
```
