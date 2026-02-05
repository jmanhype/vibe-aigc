"""Command-line interface for Vibe AIGC."""

import argparse
import asyncio
import json
import sys
from typing import Optional

from .models import Vibe
from .planner import MetaPlanner
from .visualization import WorkflowVisualizer, VisualizationFormat


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="vibe-aigc",
        description="Vibe AIGC - Content Generation via Agentic Orchestration",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Plan command
    plan_parser = subparsers.add_parser("plan", help="Generate a workflow plan from a vibe")
    plan_parser.add_argument("description", help="Main description of what you want to create")
    plan_parser.add_argument("--style", "-s", help="Style preferences")
    plan_parser.add_argument("--constraints", "-c", nargs="*", default=[], help="Constraints")
    plan_parser.add_argument("--domain", "-d", help="Domain context")
    plan_parser.add_argument("--format", "-f", choices=["ascii", "mermaid", "json"], default="ascii")
    plan_parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    # Execute command
    exec_parser = subparsers.add_parser("execute", help="Plan and execute a vibe")
    exec_parser.add_argument("description", help="Main description of what you want to create")
    exec_parser.add_argument("--style", "-s", help="Style preferences")
    exec_parser.add_argument("--constraints", "-c", nargs="*", default=[], help="Constraints")
    exec_parser.add_argument("--domain", "-d", help="Domain context")
    exec_parser.add_argument("--checkpoint", action="store_true", help="Enable checkpointing")
    exec_parser.add_argument("--checkpoint-interval", type=int, default=5, help="Checkpoint interval")
    exec_parser.add_argument("--visualize", "-v", action="store_true", help="Show visualization")
    exec_parser.add_argument("--adapt", "-a", action="store_true", help="Enable adaptive replanning")

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume from a checkpoint")
    resume_parser.add_argument("checkpoint_id", help="Checkpoint ID to resume from")

    # Checkpoints command
    cp_parser = subparsers.add_parser("checkpoints", help="List or manage checkpoints")
    cp_parser.add_argument("--list", "-l", action="store_true", help="List all checkpoints")
    cp_parser.add_argument("--delete", "-d", help="Delete a checkpoint by ID")
    cp_parser.add_argument("--clear", action="store_true", help="Delete all checkpoints")

    return parser


def create_vibe(args) -> Vibe:
    """Create a Vibe from CLI arguments."""
    return Vibe(
        description=args.description,
        style=args.style,
        constraints=args.constraints or [],
        domain=args.domain,
    )


async def cmd_plan(args) -> int:
    """Handle plan command."""
    vibe = create_vibe(args)
    planner = MetaPlanner()

    print(f"Planning workflow for: {vibe.description}", file=sys.stderr)
    plan = await planner.plan(vibe)

    if args.format == "json":
        output = plan.model_dump_json(indent=2)
    elif args.format == "mermaid":
        output = WorkflowVisualizer.generate_diagram(plan, format=VisualizationFormat.MERMAID)
    else:
        output = WorkflowVisualizer.generate_diagram(plan, format=VisualizationFormat.ASCII)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0


async def cmd_execute(args) -> int:
    """Handle execute command."""
    vibe = create_vibe(args)

    checkpoint_interval = args.checkpoint_interval if args.checkpoint else None
    planner = MetaPlanner(checkpoint_interval=checkpoint_interval)

    def on_progress(event):
        status = {"started": "🚀", "completed": "✅", "failed": "❌"}.get(event.event_type, "📍")
        print(f"{status} [{event.node_id}] {event.message}", file=sys.stderr)

    if args.visualize:
        planner = MetaPlanner(
            checkpoint_interval=checkpoint_interval,
            progress_callback=on_progress
        )

    print(f"Executing workflow for: {vibe.description}", file=sys.stderr)

    if args.adapt:
        result = await planner.execute_with_adaptation(vibe)
    elif args.checkpoint:
        result = await planner.execute_with_resume(vibe)
    elif args.visualize:
        result = await planner.execute_with_visualization(vibe)
    else:
        result = await planner.execute(vibe)

    summary = result.get_summary()
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Status: {summary['status']}", file=sys.stderr)
    print(f"Completed: {summary['completed']}/{summary['total_nodes']} nodes", file=sys.stderr)

    return 0 if summary['status'] == 'completed' else 1


async def cmd_checkpoints(args) -> int:
    """Handle checkpoints command."""
    planner = MetaPlanner()

    if args.delete:
        planner.delete_checkpoint(args.delete)
        print(f"Deleted checkpoint: {args.delete}")
    elif args.clear:
        checkpoints = planner.list_checkpoints()
        for cp in checkpoints:
            planner.delete_checkpoint(cp["checkpoint_id"])
        print(f"Deleted {len(checkpoints)} checkpoint(s)")
    else:
        checkpoints = planner.list_checkpoints()
        if not checkpoints:
            print("No checkpoints found.")
        else:
            print(f"Found {len(checkpoints)} checkpoint(s):\n")
            for cp in checkpoints:
                print(f"  ID: {cp['checkpoint_id']}")
                print(f"  Created: {cp.get('created_at', 'unknown')}")
                print(f"  Progress: {cp.get('completed_nodes', '?')}/{cp.get('total_nodes', '?')} nodes")
                print()

    return 0


async def cmd_resume(args) -> int:
    """Handle resume command."""
    planner = MetaPlanner()

    print(f"Resuming from checkpoint: {args.checkpoint_id}", file=sys.stderr)

    # Need to get the vibe from checkpoint
    checkpoint = planner.get_checkpoint(args.checkpoint_id)
    if not checkpoint:
        print(f"Checkpoint not found: {args.checkpoint_id}", file=sys.stderr)
        return 1

    vibe = Vibe(**checkpoint.get("vibe", {}))
    result = await planner.execute_with_resume(vibe, checkpoint_id=args.checkpoint_id)

    summary = result.get_summary()
    print(f"\nStatus: {summary['status']}", file=sys.stderr)
    print(f"Completed: {summary['completed']}/{summary['total_nodes']} nodes", file=sys.stderr)

    return 0 if summary['status'] == 'completed' else 1


async def async_main(args) -> int:
    """Async main entry point."""
    if args.command == "plan":
        return await cmd_plan(args)
    elif args.command == "execute":
        return await cmd_execute(args)
    elif args.command == "checkpoints":
        return await cmd_checkpoints(args)
    elif args.command == "resume":
        return await cmd_resume(args)
    else:
        print("Use --help for usage information.", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
