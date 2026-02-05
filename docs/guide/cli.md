# Command-Line Interface

Vibe AIGC includes a CLI for quick workflow operations without writing Python code.

## Installation

The CLI is included when you install the package:

```bash
pip install vibe-aigc
```

## Commands

### Plan

Generate a workflow plan from a vibe description:

```bash
# Basic usage
vibe-aigc plan "Create a blog post about AI agents"

# With options
vibe-aigc plan "Design a mobile app" \
  --style "modern, minimal" \
  --constraints "iOS-first" "under 5 screens" \
  --domain "mobile design" \
  --format ascii

# Output formats
vibe-aigc plan "..." --format ascii    # ASCII diagram (default)
vibe-aigc plan "..." --format mermaid  # Mermaid diagram
vibe-aigc plan "..." --format json     # JSON structure

# Save to file
vibe-aigc plan "..." --output plan.txt
```

### Execute

Plan and execute a vibe workflow:

```bash
# Basic execution
vibe-aigc execute "Write technical documentation"

# With visualization
vibe-aigc execute "Create a marketing campaign" --visualize

# With checkpointing
vibe-aigc execute "Generate comprehensive report" \
  --checkpoint \
  --checkpoint-interval 3

# With adaptive replanning
vibe-aigc execute "Complex multi-step task" --adapt
```

### Checkpoints

Manage workflow checkpoints:

```bash
# List all checkpoints
vibe-aigc checkpoints --list

# Delete a specific checkpoint
vibe-aigc checkpoints --delete cp-abc123

# Clear all checkpoints
vibe-aigc checkpoints --clear
```

### Resume

Resume execution from a checkpoint:

```bash
vibe-aigc resume cp-abc123
```

## Examples

### Generate and Visualize a Plan

```bash
$ vibe-aigc plan "Create a product launch video" --style "cinematic" --format ascii

Planning workflow for: Create a product launch video

Workflow Plan: plan-a1b2c3
==================================================
Source Vibe: Create a product launch video

├── ⏳ [analyze] Research product features (research)
├── ⏳ [generate] Write script (script)
│   └─ Depends on: research
├── ⏳ [generate] Create storyboard (storyboard)
│   └─ Depends on: script
└── ⏳ [generate] Produce video (produce)
    └─ Depends on: storyboard
```

### Execute with Progress Tracking

```bash
$ vibe-aigc execute "Write a technical tutorial" --visualize

Executing workflow for: Write a technical tutorial
🚀 [research] Starting research phase
✅ [research] Research complete
🚀 [outline] Creating outline
✅ [outline] Outline complete
🚀 [write] Writing content
✅ [write] Content written
🚀 [review] Reviewing
✅ [review] Review complete

==================================================
Status: completed
Completed: 4/4 nodes
```

### Long-Running Task with Checkpoints

```bash
# Start a long task with checkpointing
$ vibe-aigc execute "Generate comprehensive documentation" --checkpoint --checkpoint-interval 2

# If interrupted, check available checkpoints
$ vibe-aigc checkpoints --list

Found 3 checkpoint(s):

  ID: cp-abc123
  Created: 2026-02-05T10:30:00Z
  Progress: 4/10 nodes

# Resume from checkpoint
$ vibe-aigc resume cp-abc123
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for LLM operations |
| `VIBE_CHECKPOINT_DIR` | Custom checkpoint directory |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Execution failed or error |

## Tips

- Use `--visualize` during development to see real-time progress
- Enable `--checkpoint` for long-running tasks to allow recovery
- Use `--adapt` when working with unreliable external services
- Pipe Mermaid output to documentation: `vibe-aigc plan "..." --format mermaid >> docs/workflow.md`
