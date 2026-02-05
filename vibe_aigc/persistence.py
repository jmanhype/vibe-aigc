"""Workflow persistence and resume capabilities."""

import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib

from .models import WorkflowPlan, WorkflowNode, Vibe, WorkflowNodeType
from .executor import ExecutionResult, NodeResult, ExecutionStatus

class WorkflowCheckpoint:
    """Represents a workflow execution checkpoint."""

    def __init__(self, plan: WorkflowPlan, execution_result: ExecutionResult,
                 checkpoint_id: Optional[str] = None):
        self.checkpoint_id = checkpoint_id or self._generate_checkpoint_id(plan.id)
        self.plan = plan
        self.execution_result = execution_result
        self.created_at = datetime.now().isoformat()
        self.schema_version = "1.0"

    def _generate_checkpoint_id(self, plan_id: str) -> str:
        """Generate unique checkpoint ID."""
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:8]  # Use UUID for guaranteed uniqueness
        return f"{plan_id}_{timestamp}_{unique_suffix}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at,
            "plan": self._serialize_plan(self.plan),
            "execution_result": self._serialize_execution_result(self.execution_result)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowCheckpoint':
        """Deserialize checkpoint from dictionary."""

        # Version compatibility check
        if data.get("schema_version") != "1.0":
            raise ValueError(f"Unsupported checkpoint schema version: {data.get('schema_version')}")

        # Deserialize plan and execution result
        plan = cls._deserialize_plan(data["plan"])
        execution_result = cls._deserialize_execution_result(data["execution_result"])

        checkpoint = cls(plan, execution_result, data["checkpoint_id"])
        checkpoint.created_at = data["created_at"]
        return checkpoint

    def _serialize_plan(self, plan: WorkflowPlan) -> Dict[str, Any]:
        """Serialize WorkflowPlan to dict."""
        return {
            "id": plan.id,
            "source_vibe": {
                "description": plan.source_vibe.description,
                "style": plan.source_vibe.style,
                "constraints": plan.source_vibe.constraints,
                "domain": plan.source_vibe.domain,
                "metadata": plan.source_vibe.metadata
            },
            "root_nodes": [self._serialize_node(node) for node in plan.root_nodes],
            "estimated_total_duration": plan.estimated_total_duration,
            "created_at": plan.created_at
        }

    def _serialize_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """Serialize WorkflowNode to dict."""
        return {
            "id": node.id,
            "type": node.type.value,
            "description": node.description,
            "parameters": node.parameters,
            "dependencies": node.dependencies,
            "children": [self._serialize_node(child) for child in node.children],
            "estimated_duration": node.estimated_duration
        }

    def _serialize_execution_result(self, result: ExecutionResult) -> Dict[str, Any]:
        """Serialize ExecutionResult to dict."""
        return {
            "plan_id": result.plan_id,
            "status": result.status.value,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
            "total_duration": result.total_duration,
            "node_results": {
                node_id: {
                    "node_id": node_result.node_id,
                    "status": node_result.status.value,
                    "result": node_result.result,
                    "error": node_result.error,
                    "duration": node_result.duration,
                    "started_at": node_result.started_at
                }
                for node_id, node_result in result.node_results.items()
            },
            # Include extended fields if they exist
            "parallel_efficiency": getattr(result, 'parallel_efficiency', 0.0),
            "execution_groups": getattr(result, 'execution_groups', []),
            "feedback_data": getattr(result, 'feedback_data', {}),
            "replan_suggestions": getattr(result, 'replan_suggestions', [])
        }

    @classmethod
    def _deserialize_plan(cls, data: Dict[str, Any]) -> WorkflowPlan:
        """Deserialize WorkflowPlan from dict."""

        vibe_data = data["source_vibe"]
        source_vibe = Vibe(
            description=vibe_data["description"],
            style=vibe_data.get("style"),
            constraints=vibe_data.get("constraints", []),
            domain=vibe_data.get("domain"),
            metadata=vibe_data.get("metadata", {})
        )

        root_nodes = [cls._deserialize_node(node_data) for node_data in data["root_nodes"]]

        return WorkflowPlan(
            id=data["id"],
            source_vibe=source_vibe,
            root_nodes=root_nodes,
            estimated_total_duration=data.get("estimated_total_duration"),
            created_at=data.get("created_at")
        )

    @classmethod
    def _deserialize_node(cls, data: Dict[str, Any]) -> WorkflowNode:
        """Deserialize WorkflowNode from dict."""

        children = [cls._deserialize_node(child_data) for child_data in data.get("children", [])]

        return WorkflowNode(
            id=data["id"],
            type=WorkflowNodeType(data["type"]),
            description=data["description"],
            parameters=data.get("parameters", {}),
            dependencies=data.get("dependencies", []),
            children=children,
            estimated_duration=data.get("estimated_duration")
        )

    @classmethod
    def _deserialize_execution_result(cls, data: Dict[str, Any]) -> ExecutionResult:
        """Deserialize ExecutionResult from dict."""
        result = ExecutionResult(data["plan_id"])
        result.status = ExecutionStatus(data["status"])
        result.started_at = data["started_at"]
        result.completed_at = data["completed_at"]
        result.total_duration = data["total_duration"]

        # Deserialize node results
        for node_id, node_data in data["node_results"].items():
            node_result = NodeResult(
                node_data["node_id"],
                ExecutionStatus(node_data["status"]),
                node_data.get("result"),
                node_data.get("error"),
                node_data.get("duration", 0.0)
            )
            node_result.started_at = node_data.get("started_at", datetime.now().isoformat())
            result.node_results[node_id] = node_result

        # Restore extended fields
        if "parallel_efficiency" in data:
            result.parallel_efficiency = data["parallel_efficiency"]
        if "execution_groups" in data:
            result.execution_groups = data["execution_groups"]
        if "feedback_data" in data:
            result.feedback_data = data["feedback_data"]
        if "replan_suggestions" in data:
            result.replan_suggestions = data["replan_suggestions"]

        return result


class WorkflowPersistenceManager:
    """Manages workflow checkpoint persistence."""

    def __init__(self, checkpoint_dir: str = ".vibe_checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> str:
        """Save checkpoint to disk."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint.checkpoint_id}.json")

        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, default=str)
            return checkpoint_path
        except Exception as e:
            raise RuntimeError(f"Failed to save checkpoint {checkpoint.checkpoint_id}: {e}") from e

    def load_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint:
        """Load checkpoint from disk."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            return WorkflowCheckpoint.from_dict(data)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_id}: {e}") from e

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints with metadata."""
        checkpoints = []

        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.json'):
                checkpoint_id = filename[:-5]  # Remove .json extension

                try:
                    checkpoint_path = os.path.join(self.checkpoint_dir, filename)
                    with open(checkpoint_path, 'r') as f:
                        data = json.load(f)

                    checkpoints.append({
                        "checkpoint_id": checkpoint_id,
                        "plan_id": data.get("plan", {}).get("id"),
                        "created_at": data.get("created_at"),
                        "status": data.get("execution_result", {}).get("status"),
                        "vibe_description": data.get("plan", {}).get("source_vibe", {}).get("description", "")[:50]
                    })
                except Exception:
                    # Skip corrupted checkpoints
                    continue

        return sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from disk."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                return True
            except Exception as e:
                raise RuntimeError(f"Failed to delete checkpoint {checkpoint_id}: {e}") from e

        return False