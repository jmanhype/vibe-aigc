# Checkpoints

Save and resume workflow execution state.

## Why Checkpoints?

Long-running workflows can fail midway due to:

- API rate limits
- Network issues
- System interruptions
- Temporary failures

Checkpoints let you resume from where you left off instead of starting over.

## Enabling Checkpoints

### Automatic Checkpointing

Save checkpoints at regular intervals:

```python
from vibe_aigc import MetaPlanner

# Checkpoint every 5 completed nodes
planner = MetaPlanner(checkpoint_interval=5)
result = await planner.execute_with_resume(vibe)
```

### Manual Checkpoints

Create checkpoints programmatically:

```python
# Create a checkpoint
checkpoint_id = planner.create_checkpoint(
    plan_id="plan-001",
    execution_state=current_state
)
print(f"Saved checkpoint: {checkpoint_id}")
```

## Managing Checkpoints

### List Checkpoints

```python
checkpoints = planner.list_checkpoints()
for cp in checkpoints:
    print(f"ID: {cp['checkpoint_id']}")
    print(f"Created: {cp['created_at']}")
    print(f"Plan: {cp['plan_id']}")
    print(f"Progress: {cp['completed_nodes']}/{cp['total_nodes']}")
    print()
```

### Get Checkpoint Details

```python
checkpoint = planner.get_checkpoint(checkpoint_id)
print(f"Workflow state: {checkpoint['state']}")
print(f"Node results: {checkpoint['node_results']}")
```

### Delete Checkpoint

```python
planner.delete_checkpoint(checkpoint_id)
```

## Resuming Execution

### Resume from Latest

```python
# Automatically uses most recent checkpoint
result = await planner.execute_with_resume(vibe)
```

### Resume from Specific Checkpoint

```python
result = await planner.execute_with_resume(
    vibe,
    checkpoint_id="cp-abc123"
)
```

## Checkpoint Storage

By default, checkpoints are stored in `.vibe_checkpoints/`:

```
.vibe_checkpoints/
├── plan-001/
│   ├── cp-abc123.json
│   └── cp-def456.json
└── plan-002/
    └── cp-ghi789.json
```

### Custom Storage Location

```python
from vibe_aigc.persistence import WorkflowPersistenceManager

manager = WorkflowPersistenceManager(
    checkpoint_dir="/path/to/checkpoints"
)
planner = MetaPlanner(persistence_manager=manager)
```

## Checkpoint Contents

Each checkpoint contains:

```json
{
  "checkpoint_id": "cp-abc123",
  "plan_id": "plan-001",
  "created_at": "2026-02-05T10:30:00Z",
  "vibe": {
    "description": "...",
    "style": "...",
    "constraints": []
  },
  "workflow_plan": { ... },
  "completed_nodes": ["node1", "node2"],
  "node_results": {
    "node1": { "status": "completed", "output": "..." },
    "node2": { "status": "completed", "output": "..." }
  },
  "pending_nodes": ["node3", "node4"]
}
```

## Best Practices

1. **Set appropriate intervals**: Too frequent = overhead, too sparse = lost progress
2. **Clean up old checkpoints**: Delete after successful completion
3. **Use meaningful plan IDs**: Easier to find relevant checkpoints
4. **Handle resume failures**: Implement fallback to fresh start

```python
try:
    result = await planner.execute_with_resume(vibe)
except CheckpointCorruptError:
    # Fall back to fresh execution
    planner.delete_checkpoint(checkpoint_id)
    result = await planner.execute(vibe)
```
