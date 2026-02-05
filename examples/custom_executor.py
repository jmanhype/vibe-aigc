"""
Example: Custom executor with external tool integration.

This shows how to extend the WorkflowExecutor to integrate
with external APIs or tools (image generation, web search, etc.)
"""

import asyncio
from typing import Any, Dict, Optional

from vibe_aigc import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import WorkflowExecutor, ExecutionStatus
from vibe_aigc.visualization import WorkflowVisualizer, VisualizationFormat


class CustomExecutor(WorkflowExecutor):
    """
    Extended executor that can integrate with external tools.
    
    In a real implementation, you would add API clients for:
    - Image generation (DALL-E, Midjourney, Stable Diffusion)
    - Text generation (OpenAI, Anthropic, local models)
    - Web search (Brave, Google, Bing)
    - Code execution (sandboxed environments)
    """

    def __init__(self, tools: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.tools = tools or {}
        self.execution_log = []

    async def execute_node(self, node: WorkflowNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Custom node execution with tool integration.
        
        Override this method to add custom behavior for different node types.
        """
        self.execution_log.append(f"Executing: {node.id} ({node.type.value})")

        # Route based on node type
        if node.type == WorkflowNodeType.ANALYZE:
            return await self._handle_analyze(node, context)
        elif node.type == WorkflowNodeType.GENERATE:
            return await self._handle_generate(node, context)
        elif node.type == WorkflowNodeType.TRANSFORM:
            return await self._handle_transform(node, context)
        elif node.type == WorkflowNodeType.VALIDATE:
            return await self._handle_validate(node, context)
        else:
            return await super().execute_node(node, context)

    async def _handle_analyze(self, node: WorkflowNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analysis nodes - could integrate with search APIs."""
        # Example: integrate with web search
        if "search" in self.tools:
            # search_client = self.tools["search"]
            # results = await search_client.search(node.description)
            pass

        # Simulate analysis
        await asyncio.sleep(0.1)
        return {
            "status": ExecutionStatus.COMPLETED,
            "output": f"Analysis complete for: {node.description}",
            "artifacts": {"analyzed": True}
        }

    async def _handle_generate(self, node: WorkflowNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generation nodes - could integrate with LLM/image APIs."""
        # Example: integrate with image generation
        if "image_gen" in self.tools and "image" in node.description.lower():
            # image_client = self.tools["image_gen"]
            # image_url = await image_client.generate(node.description)
            pass

        # Simulate generation
        await asyncio.sleep(0.2)
        return {
            "status": ExecutionStatus.COMPLETED,
            "output": f"Generated content for: {node.description}",
            "artifacts": {"generated": True, "type": "text"}
        }

    async def _handle_transform(self, node: WorkflowNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle transformation nodes."""
        await asyncio.sleep(0.1)
        return {
            "status": ExecutionStatus.COMPLETED,
            "output": f"Transformed: {node.description}",
            "artifacts": {"transformed": True}
        }

    async def _handle_validate(self, node: WorkflowNode, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle validation nodes."""
        await asyncio.sleep(0.05)
        return {
            "status": ExecutionStatus.COMPLETED,
            "output": f"Validated: {node.description}",
            "artifacts": {"valid": True}
        }


async def main():
    """Demo custom executor with a manually constructed workflow."""

    # Create a multi-modal content workflow
    nodes = [
        WorkflowNode(
            id="research",
            type=WorkflowNodeType.ANALYZE,
            description="Research trending topics in AI"
        ),
        WorkflowNode(
            id="outline",
            type=WorkflowNodeType.GENERATE,
            description="Generate content outline",
            dependencies=["research"]
        ),
        WorkflowNode(
            id="write_intro",
            type=WorkflowNodeType.GENERATE,
            description="Write introduction section",
            dependencies=["outline"]
        ),
        WorkflowNode(
            id="write_body",
            type=WorkflowNodeType.GENERATE,
            description="Write main content body",
            dependencies=["outline"]
        ),
        WorkflowNode(
            id="generate_image",
            type=WorkflowNodeType.GENERATE,
            description="Generate header image for article",
            dependencies=["outline"]
        ),
        WorkflowNode(
            id="combine",
            type=WorkflowNodeType.TRANSFORM,
            description="Combine all sections into final article",
            dependencies=["write_intro", "write_body", "generate_image"]
        ),
        WorkflowNode(
            id="review",
            type=WorkflowNodeType.VALIDATE,
            description="Review and validate final content",
            dependencies=["combine"]
        )
    ]

    vibe = Vibe(description="Create an AI article with images")
    plan = WorkflowPlan(
        id="custom-workflow-001",
        source_vibe=vibe,
        root_nodes=nodes
    )

    # Show the workflow
    print("=" * 60)
    print("Custom Executor Demo")
    print("=" * 60)
    print("\nWorkflow Plan:")
    print(WorkflowVisualizer.generate_diagram(plan, format=VisualizationFormat.ASCII))

    # Execute with custom executor
    print("\nExecuting with custom executor...")
    print("-" * 40)

    # In production, you'd pass real tool clients here
    tools = {
        # "search": BraveSearchClient(api_key="..."),
        # "image_gen": OpenAIImageClient(api_key="..."),
        # "llm": AnthropicClient(api_key="..."),
    }

    executor = CustomExecutor(tools=tools)
    result = await executor.execute_plan(plan)

    # Show results
    print("\n" + "-" * 40)
    print("Execution Log:")
    for entry in executor.execution_log:
        print(f"  • {entry}")

    print("\nResults:")
    summary = result.get_summary()
    print(f"  Status: {summary['status']}")
    print(f"  Completed: {summary['completed']}/{summary['total_nodes']} nodes")

    # Show with status
    print("\nFinal Status:")
    print(WorkflowVisualizer.generate_diagram(
        plan,
        execution_result=result,
        format=VisualizationFormat.ASCII
    ))


if __name__ == "__main__":
    asyncio.run(main())
