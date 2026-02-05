# Core Vibe Model and Meta-Planner Architecture Implementation Plan

## Implementation Plan Title

Foundational Vibe AIGC System: Core Data Models, Meta-Planner, and Execution Engine

## Overview

Implementing a completely new Python package from scratch that transforms high-level creative intent (Vibe) into executable hierarchical agent workflows. This establishes the foundational architecture for the Vibe AIGC system based on arXiv:2602.04575, focusing on the Intent-Execution Gap bridge through intelligent decomposition and agentic orchestration.

## Current State

**Complete greenfield project** - Only documentation and project infrastructure exist:
- `README.md:39-50` - Defines expected API surface with `from vibe_aigc import MetaPlanner, Vibe`
- `README.md:42-46` - Shows Vibe constructor with `description`, `style`, `constraints` fields
- `README.md:49-50` - Shows `MetaPlanner().execute(vibe)` as primary interface
- `.wreckit/items/001-vibe-model-meta-planner/item.json:17-21` - Specifies success criteria including WorkflowPlan and WorkflowNode structures
- No Python package structure, dependencies, or source code exists

### Key Discoveries:

- **API Contract Defined**: `README.md:39` imports suggest clean separation between `MetaPlanner` and `Vibe` classes
- **Async Pattern Required**: `README.md:50` shows `await planner.execute(vibe)` indicating async/await architecture
- **Hierarchical Structure**: `item.json:19` requires WorkflowNode for hierarchical task decomposition
- **LLM Integration**: `item.json:20` specifies "LLM-based decomposition of Vibe into concrete tasks"
- **Pydantic Validation**: `item.json:25` mandates Pydantic for data validation
- **Python 3.12+**: `item.json:24` sets modern Python version requirement

## Desired End State

A fully functional Python package that enables the exact usage pattern shown in `README.md:39-50`:

```python
from vibe_aigc import MetaPlanner, Vibe

vibe = Vibe(
    description="Create a cinematic sci-fi scene",
    style="dark, atmospheric",
    constraints=["no violence", "PG-13"]
)

planner = MetaPlanner()
result = await planner.execute(vibe)
```

**Success Verification:**
- Package imports cleanly with `from vibe_aigc import MetaPlanner, Vibe`
- Vibe objects validate input fields using Pydantic
- MetaPlanner decomposes Vibe into structured WorkflowPlan via LLM
- Basic execution engine runs WorkflowPlan and returns results
- All operations use async/await patterns
- Comprehensive test suite validates end-to-end functionality

## What We're NOT Doing

- **Complex agent implementations** - Focus on orchestration framework, not specific agent capabilities
- **Advanced UI/visualization** - Command-line/API interface only
- **Production deployment** - Development-ready package without scaling concerns
- **Multiple LLM providers** - Start with OpenAI client, design for future extensibility
- **Persistent state management** - In-memory execution for initial implementation
- **Advanced workflow features** - No conditional branching, parallel execution, or retry logic initially
- **Performance optimization** - Prioritize correctness and clean architecture over speed

## Implementation Approach

**Bottom-up layered architecture** starting with data models, then planning logic, finally execution:

1. **Foundation Layer**: Establish Python package structure, dependencies, and core Pydantic models
2. **Planning Layer**: Implement MetaPlanner with LLM integration for Vibe decomposition
3. **Execution Layer**: Build basic async workflow execution engine
4. **Integration Layer**: Wire components together and validate end-to-end functionality

**Key Design Principles:**
- **Immutable data structures** using Pydantic for type safety and validation
- **Clean separation of concerns** between intent representation, planning, and execution
- **Async-first architecture** for all I/O operations including LLM calls
- **Extensible abstractions** that support future enhancement without breaking changes

---

## Phases

### Phase 1: Package Foundation and Core Models

#### Overview

Establish the Python package structure and implement the core Pydantic models that define the system's data contracts. This creates the foundation for all subsequent development and enables early validation of the API design.

#### Changes Required:

##### 1. Python Package Structure

**File**: `C:\Users\strau\clawd\vibe-aigc\pyproject.toml`
**Changes**: Create modern Python packaging configuration

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vibe-aigc"
version = "0.1.0"
description = "A New Paradigm for Content Generation via Agentic Orchestration"
authors = [{name = "Vibe AIGC Contributors"}]
license = {text = "MIT"}
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0.0",
    "openai>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]
```

**File**: `C:\Users\strau\clawd\vibe-aigc\vibe_aigc\__init__.py`
**Changes**: Package exports matching README.md:39

```python
"""Vibe AIGC: A New Paradigm for Content Generation via Agentic Orchestration."""

from .models import Vibe, WorkflowPlan, WorkflowNode
from .planner import MetaPlanner

__version__ = "0.1.0"
__all__ = ["Vibe", "WorkflowPlan", "WorkflowNode", "MetaPlanner"]
```

##### 2. Core Data Models

**File**: `C:\Users\strau\clawd\vibe-aigc\vibe_aigc\models.py`
**Changes**: Implement Pydantic models for Vibe, WorkflowPlan, and WorkflowNode

```python
"""Core data models for Vibe AIGC system."""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class Vibe(BaseModel):
    """High-level representation of user's creative intent and aesthetic preferences."""

    description: str = Field(..., description="Primary description of the desired outcome")
    style: Optional[str] = Field(None, description="Aesthetic style preferences")
    constraints: List[str] = Field(default_factory=list, description="Limitations and requirements")
    domain: Optional[str] = Field(None, description="Domain context (e.g., 'visual', 'text', 'audio')")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context and parameters")


class WorkflowNodeType(str, Enum):
    """Types of workflow nodes for different operation categories."""

    ANALYZE = "analyze"
    GENERATE = "generate"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    COMPOSITE = "composite"


class WorkflowNode(BaseModel):
    """Individual task node in hierarchical workflow decomposition."""

    id: str = Field(..., description="Unique identifier for this node")
    type: WorkflowNodeType = Field(..., description="Category of operation this node performs")
    description: str = Field(..., description="Human-readable description of the task")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task-specific parameters")
    dependencies: List[str] = Field(default_factory=list, description="IDs of nodes that must complete first")
    children: List['WorkflowNode'] = Field(default_factory=list, description="Sub-tasks for hierarchical decomposition")
    estimated_duration: Optional[int] = Field(None, description="Estimated execution time in seconds")


class WorkflowPlan(BaseModel):
    """Complete execution plan generated from a Vibe."""

    id: str = Field(..., description="Unique identifier for this plan")
    source_vibe: Vibe = Field(..., description="Original vibe that generated this plan")
    root_nodes: List[WorkflowNode] = Field(..., description="Top-level workflow nodes")
    estimated_total_duration: Optional[int] = Field(None, description="Total estimated execution time")
    created_at: Optional[str] = Field(None, description="ISO timestamp of plan creation")


# Enable forward references for WorkflowNode.children
WorkflowNode.model_rebuild()
```

##### 3. Basic Testing Infrastructure

**File**: `C:\Users\strau\clawd\vibe-aigc\tests\__init__.py`
**Changes**: Empty file to make tests a package

```python
```

**File**: `C:\Users\strau\clawd\vibe-aigc\tests\test_models.py`
**Changes**: Unit tests for core data models

```python
"""Tests for core data models."""

import pytest
from vibe_aigc.models import Vibe, WorkflowNode, WorkflowPlan, WorkflowNodeType


class TestVibe:
    """Test Vibe model validation and behavior."""

    def test_basic_vibe_creation(self):
        """Test creating a basic Vibe with required fields."""
        vibe = Vibe(
            description="Create a cinematic sci-fi scene",
            style="dark, atmospheric",
            constraints=["no violence", "PG-13"]
        )

        assert vibe.description == "Create a cinematic sci-fi scene"
        assert vibe.style == "dark, atmospheric"
        assert vibe.constraints == ["no violence", "PG-13"]
        assert vibe.domain is None
        assert vibe.metadata == {}

    def test_vibe_with_optional_fields(self):
        """Test Vibe creation with all optional fields."""
        vibe = Vibe(
            description="Test description",
            domain="visual",
            metadata={"quality": "high", "format": "4K"}
        )

        assert vibe.domain == "visual"
        assert vibe.metadata == {"quality": "high", "format": "4K"}

    def test_vibe_validation_requires_description(self):
        """Test that description field is required."""
        with pytest.raises(ValueError):
            Vibe()


class TestWorkflowNode:
    """Test WorkflowNode model validation and hierarchical structure."""

    def test_basic_node_creation(self):
        """Test creating a basic WorkflowNode."""
        node = WorkflowNode(
            id="task-001",
            type=WorkflowNodeType.GENERATE,
            description="Generate initial concept"
        )

        assert node.id == "task-001"
        assert node.type == WorkflowNodeType.GENERATE
        assert node.description == "Generate initial concept"
        assert node.parameters == {}
        assert node.dependencies == []
        assert node.children == []

    def test_hierarchical_node_structure(self):
        """Test creating nodes with children for hierarchical decomposition."""
        child1 = WorkflowNode(
            id="subtask-001",
            type=WorkflowNodeType.ANALYZE,
            description="Analyze requirements"
        )
        child2 = WorkflowNode(
            id="subtask-002",
            type=WorkflowNodeType.GENERATE,
            description="Generate content",
            dependencies=["subtask-001"]
        )

        parent = WorkflowNode(
            id="task-001",
            type=WorkflowNodeType.COMPOSITE,
            description="Complete content creation",
            children=[child1, child2]
        )

        assert len(parent.children) == 2
        assert parent.children[0].id == "subtask-001"
        assert parent.children[1].dependencies == ["subtask-001"]


class TestWorkflowPlan:
    """Test WorkflowPlan model and plan structure."""

    def test_basic_plan_creation(self):
        """Test creating a WorkflowPlan with minimal structure."""
        vibe = Vibe(description="Test vibe")
        node = WorkflowNode(
            id="task-001",
            type=WorkflowNodeType.GENERATE,
            description="Test task"
        )

        plan = WorkflowPlan(
            id="plan-001",
            source_vibe=vibe,
            root_nodes=[node]
        )

        assert plan.id == "plan-001"
        assert plan.source_vibe.description == "Test vibe"
        assert len(plan.root_nodes) == 1
        assert plan.root_nodes[0].id == "task-001"
```

#### Success Criteria:

##### Automated Verification:

- [ ] Package installs: `pip install -e .`
- [ ] Tests pass: `pytest tests/`
- [ ] Type checking passes: `mypy vibe_aigc/`
- [ ] Code formatting: `black --check vibe_aigc/`
- [ ] Linting passes: `ruff check vibe_aigc/`

##### Manual Verification:

- [ ] Can import core classes: `from vibe_aigc import MetaPlanner, Vibe`
- [ ] Vibe model validates input according to README.md:42-46 example
- [ ] WorkflowNode supports hierarchical structure with children
- [ ] All Pydantic models serialize/deserialize correctly
- [ ] Package structure follows Python best practices

**Note**: Complete all automated verification, then pause for manual confirmation before proceeding to next phase.

---

### Phase 2: MetaPlanner and LLM Integration

#### Overview

Implement the MetaPlanner class with LLM integration that can decompose a Vibe into a structured WorkflowPlan. This is the core intelligence of the system that transforms high-level intent into executable task hierarchies.

#### Changes Required:

##### 1. LLM Client Abstraction

**File**: `C:\Users\strau\clawd\vibe-aigc\vibe_aigc\llm.py`
**Changes**: Create abstraction for LLM interactions with OpenAI client

```python
"""LLM client abstraction for Vibe decomposition."""

import asyncio
import json
from typing import Any, Dict, Optional
from openai import AsyncOpenAI
from pydantic import BaseModel

from .models import Vibe, WorkflowPlan


class LLMConfig(BaseModel):
    """Configuration for LLM client."""

    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: Optional[str] = None


class LLMClient:
    """Async client for LLM-based Vibe decomposition."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.client = AsyncOpenAI(api_key=self.config.api_key)

    async def decompose_vibe(self, vibe: Vibe) -> Dict[str, Any]:
        """Decompose a Vibe into structured workflow plan data."""

        system_prompt = """You are a Meta-Planner that decomposes high-level creative intent (Vibes) into executable workflow plans.

Given a Vibe, create a hierarchical breakdown of tasks needed to achieve the user's intent.

Respond with a JSON object containing:
- id: unique plan identifier
- root_nodes: array of top-level tasks, each with:
  - id: unique task identifier
  - type: one of "analyze", "generate", "transform", "validate", "composite"
  - description: clear task description
  - parameters: task-specific configuration
  - dependencies: array of task IDs that must complete first
  - children: array of sub-tasks (same structure)
  - estimated_duration: estimated seconds to complete

Focus on logical decomposition and clear dependencies. Keep tasks atomic and executable."""

        user_prompt = f"""Decompose this Vibe into a workflow plan:

Description: {vibe.description}
Style: {vibe.style or 'Not specified'}
Constraints: {', '.join(vibe.constraints) if vibe.constraints else 'None'}
Domain: {vibe.domain or 'General'}

Additional context: {vibe.metadata}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            return json.loads(content)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}")
```

##### 2. MetaPlanner Implementation

**File**: `C:\Users\strau\clawd\vibe-aigc\vibe_aigc\planner.py`
**Changes**: Implement MetaPlanner class with async Vibe decomposition

```python
"""MetaPlanner: Core orchestration and Vibe decomposition."""

import uuid
from datetime import datetime
from typing import Any, Dict, List

from .models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from .llm import LLMClient, LLMConfig


class MetaPlanner:
    """Central system architect that decomposes Vibes into executable workflows."""

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_client = LLMClient(llm_config)

    async def plan(self, vibe: Vibe) -> WorkflowPlan:
        """Generate a WorkflowPlan from a Vibe using LLM decomposition."""

        # Get structured decomposition from LLM
        plan_data = await self.llm_client.decompose_vibe(vibe)

        # Convert LLM response to structured WorkflowPlan
        plan_id = plan_data.get("id", f"plan-{uuid.uuid4().hex[:8]}")

        root_nodes = [
            self._build_workflow_node(node_data)
            for node_data in plan_data.get("root_nodes", [])
        ]

        # Calculate total estimated duration
        total_duration = sum(
            self._calculate_node_duration(node)
            for node in root_nodes
        )

        return WorkflowPlan(
            id=plan_id,
            source_vibe=vibe,
            root_nodes=root_nodes,
            estimated_total_duration=total_duration,
            created_at=datetime.now().isoformat()
        )

    async def execute(self, vibe: Vibe) -> Dict[str, Any]:
        """Plan and execute a Vibe workflow (basic implementation)."""

        # Generate execution plan
        plan = await self.plan(vibe)

        # For now, return the plan structure as the "result"
        # Phase 3 will implement actual execution
        return {
            "status": "completed",
            "plan_id": plan.id,
            "vibe_description": vibe.description,
            "total_tasks": len(plan.root_nodes),
            "estimated_duration": plan.estimated_total_duration,
            "execution_summary": "Plan generated successfully (execution engine pending)"
        }

    def _build_workflow_node(self, node_data: Dict[str, Any]) -> WorkflowNode:
        """Convert LLM response data to WorkflowNode structure."""

        # Validate and normalize node type
        node_type_str = node_data.get("type", "generate").lower()
        try:
            node_type = WorkflowNodeType(node_type_str)
        except ValueError:
            node_type = WorkflowNodeType.GENERATE

        # Recursively build children nodes
        children = [
            self._build_workflow_node(child_data)
            for child_data in node_data.get("children", [])
        ]

        return WorkflowNode(
            id=node_data.get("id", f"task-{uuid.uuid4().hex[:8]}"),
            type=node_type,
            description=node_data.get("description", "Untitled task"),
            parameters=node_data.get("parameters", {}),
            dependencies=node_data.get("dependencies", []),
            children=children,
            estimated_duration=node_data.get("estimated_duration")
        )

    def _calculate_node_duration(self, node: WorkflowNode) -> int:
        """Calculate total estimated duration for a node and its children."""

        node_duration = node.estimated_duration or 0
        children_duration = sum(
            self._calculate_node_duration(child)
            for child in node.children
        )

        return node_duration + children_duration
```

##### 3. Integration Tests

**File**: `C:\Users\strau\clawd\vibe-aigc\tests\test_planner.py`
**Changes**: Integration tests for MetaPlanner functionality

```python
"""Tests for MetaPlanner and LLM integration."""

import pytest
import json
from unittest.mock import AsyncMock, patch

from vibe_aigc.models import Vibe, WorkflowNodeType
from vibe_aigc.planner import MetaPlanner
from vibe_aigc.llm import LLMClient, LLMConfig


@pytest.mark.asyncio
class TestLLMClient:
    """Test LLM client functionality with mocked responses."""

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_decompose_vibe_success(self, mock_openai):
        """Test successful Vibe decomposition."""

        # Mock LLM response
        mock_response = {
            "id": "plan-test-001",
            "root_nodes": [
                {
                    "id": "task-001",
                    "type": "analyze",
                    "description": "Analyze scene requirements",
                    "parameters": {"detail_level": "high"},
                    "dependencies": [],
                    "children": [],
                    "estimated_duration": 30
                }
            ]
        }

        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = json.dumps(mock_response)
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        # Test decomposition
        client = LLMClient(LLMConfig())
        vibe = Vibe(description="Create a cinematic sci-fi scene")

        result = await client.decompose_vibe(vibe)

        assert result["id"] == "plan-test-001"
        assert len(result["root_nodes"]) == 1
        assert result["root_nodes"][0]["type"] == "analyze"

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_decompose_vibe_invalid_json(self, mock_openai):
        """Test handling of invalid JSON response."""

        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = "Invalid JSON response"
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        client = LLMClient(LLMConfig())
        vibe = Vibe(description="Test")

        with pytest.raises(ValueError, match="Invalid JSON response from LLM"):
            await client.decompose_vibe(vibe)


@pytest.mark.asyncio
class TestMetaPlanner:
    """Test MetaPlanner core functionality."""

    @patch('vibe_aigc.planner.LLMClient')
    async def test_plan_generation(self, mock_llm_client):
        """Test WorkflowPlan generation from Vibe."""

        # Mock LLM client response
        mock_client_instance = AsyncMock()
        mock_client_instance.decompose_vibe.return_value = {
            "id": "plan-test-001",
            "root_nodes": [
                {
                    "id": "task-001",
                    "type": "generate",
                    "description": "Generate sci-fi concept",
                    "parameters": {"style": "cinematic"},
                    "dependencies": [],
                    "children": [
                        {
                            "id": "subtask-001",
                            "type": "analyze",
                            "description": "Analyze style requirements",
                            "parameters": {},
                            "dependencies": [],
                            "children": [],
                            "estimated_duration": 15
                        }
                    ],
                    "estimated_duration": 60
                }
            ]
        }
        mock_llm_client.return_value = mock_client_instance

        # Test plan generation
        planner = MetaPlanner()
        vibe = Vibe(
            description="Create a cinematic sci-fi scene",
            style="dark, atmospheric",
            constraints=["no violence", "PG-13"]
        )

        plan = await planner.plan(vibe)

        assert plan.id == "plan-test-001"
        assert plan.source_vibe.description == "Create a cinematic sci-fi scene"
        assert len(plan.root_nodes) == 1
        assert plan.root_nodes[0].type == WorkflowNodeType.GENERATE
        assert len(plan.root_nodes[0].children) == 1
        assert plan.estimated_total_duration == 75  # 60 + 15

    @patch('vibe_aigc.planner.LLMClient')
    async def test_execute_basic_workflow(self, mock_llm_client):
        """Test basic execute method (Phase 2 implementation)."""

        mock_client_instance = AsyncMock()
        mock_client_instance.decompose_vibe.return_value = {
            "root_nodes": [
                {
                    "id": "task-001",
                    "type": "generate",
                    "description": "Test task",
                    "estimated_duration": 30
                }
            ]
        }
        mock_llm_client.return_value = mock_client_instance

        planner = MetaPlanner()
        vibe = Vibe(description="Test vibe")

        result = await planner.execute(vibe)

        assert result["status"] == "completed"
        assert result["vibe_description"] == "Test vibe"
        assert result["total_tasks"] == 1
        assert result["estimated_duration"] == 30
```

##### 4. Update Package Exports

**File**: `C:\Users\strau\clawd\vibe-aigc\vibe_aigc\__init__.py`
**Changes**: Add LLM-related exports

```python
"""Vibe AIGC: A New Paradigm for Content Generation via Agentic Orchestration."""

from .models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from .planner import MetaPlanner
from .llm import LLMClient, LLMConfig

__version__ = "0.1.0"
__all__ = [
    "Vibe", "WorkflowPlan", "WorkflowNode", "WorkflowNodeType",
    "MetaPlanner", "LLMClient", "LLMConfig"
]
```

#### Success Criteria:

##### Automated Verification:

- [ ] Tests pass: `pytest tests/`
- [ ] Type checking passes: `mypy vibe_aigc/`
- [ ] Code formatting: `black --check vibe_aigc/`
- [ ] LLM integration tests pass (with mocked responses)
- [ ] MetaPlanner can generate WorkflowPlan from Vibe

##### Manual Verification:

- [ ] Can create MetaPlanner instance: `planner = MetaPlanner()`
- [ ] Can call `plan = await planner.plan(vibe)` and get valid WorkflowPlan
- [ ] Generated WorkflowPlan contains hierarchical WorkflowNodes
- [ ] LLM client properly handles API errors and invalid responses
- [ ] Execute method returns structured result (even without full execution)

**Note**: Complete all automated verification, then pause for manual confirmation before proceeding to next phase.

---

### Phase 3: Basic Execution Engine

#### Overview

Implement a basic async execution engine that can run generated WorkflowPlans. This phase focuses on the fundamental execution patterns without advanced features like parallel processing or error recovery.

#### Changes Required:

##### 1. Execution Engine Core

**File**: `C:\Users\strau\clawd\vibe-aigc\vibe_aigc\executor.py`
**Changes**: Implement basic async workflow execution engine

```python
"""Workflow execution engine for running WorkflowPlans."""

import asyncio
from typing import Any, Dict, List, Set
from datetime import datetime
from enum import Enum

from .models import WorkflowPlan, WorkflowNode, WorkflowNodeType


class ExecutionStatus(str, Enum):
    """Status of workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeResult:
    """Result of executing a single WorkflowNode."""

    def __init__(self, node_id: str, status: ExecutionStatus,
                 result: Any = None, error: str = None, duration: float = 0.0):
        self.node_id = node_id
        self.status = status
        self.result = result
        self.error = error
        self.duration = duration
        self.started_at = datetime.now().isoformat()


class ExecutionResult:
    """Complete result of WorkflowPlan execution."""

    def __init__(self, plan_id: str):
        self.plan_id = plan_id
        self.status = ExecutionStatus.PENDING
        self.node_results: Dict[str, NodeResult] = {}
        self.started_at = datetime.now().isoformat()
        self.completed_at: str = None
        self.total_duration: float = 0.0

    def add_node_result(self, result: NodeResult):
        """Add result for a completed node."""
        self.node_results[result.node_id] = result

    def is_complete(self) -> bool:
        """Check if all nodes have completed (successfully or failed)."""
        return all(
            result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.SKIPPED]
            for result in self.node_results.values()
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        completed = sum(1 for r in self.node_results.values() if r.status == ExecutionStatus.COMPLETED)
        failed = sum(1 for r in self.node_results.values() if r.status == ExecutionStatus.FAILED)
        total = len(self.node_results)

        return {
            "plan_id": self.plan_id,
            "status": self.status.value,
            "total_nodes": total,
            "completed": completed,
            "failed": failed,
            "total_duration": self.total_duration,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


class WorkflowExecutor:
    """Basic execution engine for WorkflowPlans."""

    def __init__(self):
        self.node_handlers = {
            WorkflowNodeType.ANALYZE: self._execute_analyze,
            WorkflowNodeType.GENERATE: self._execute_generate,
            WorkflowNodeType.TRANSFORM: self._execute_transform,
            WorkflowNodeType.VALIDATE: self._execute_validate,
            WorkflowNodeType.COMPOSITE: self._execute_composite
        }

    async def execute_plan(self, plan: WorkflowPlan) -> ExecutionResult:
        """Execute a complete WorkflowPlan."""

        result = ExecutionResult(plan.id)
        result.status = ExecutionStatus.RUNNING

        try:
            # Execute root nodes (for now, sequentially)
            for node in plan.root_nodes:
                await self._execute_node_tree(node, result)

            result.status = ExecutionStatus.COMPLETED
            result.completed_at = datetime.now().isoformat()

        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.completed_at = datetime.now().isoformat()

            # Add error result for any nodes that haven't been executed
            for node in plan.root_nodes:
                self._mark_tree_failed(node, result, str(e))

        # Calculate total duration
        result.total_duration = sum(r.duration for r in result.node_results.values())

        return result

    async def _execute_node_tree(self, node: WorkflowNode, result: ExecutionResult):
        """Execute a node and its children, respecting dependencies."""

        # Check if dependencies are satisfied
        if not self._dependencies_satisfied(node, result):
            result.add_node_result(NodeResult(
                node.id, ExecutionStatus.SKIPPED,
                error="Dependencies not satisfied"
            ))
            return

        start_time = asyncio.get_event_loop().time()

        try:
            # Execute the node
            handler = self.node_handlers.get(node.type, self._execute_default)
            node_result = await handler(node)

            duration = asyncio.get_event_loop().time() - start_time
            result.add_node_result(NodeResult(
                node.id, ExecutionStatus.COMPLETED,
                result=node_result, duration=duration
            ))

            # Execute children sequentially
            for child in node.children:
                await self._execute_node_tree(child, result)

        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            result.add_node_result(NodeResult(
                node.id, ExecutionStatus.FAILED,
                error=str(e), duration=duration
            ))
            raise

    def _dependencies_satisfied(self, node: WorkflowNode, result: ExecutionResult) -> bool:
        """Check if all dependencies for a node have completed successfully."""

        for dep_id in node.dependencies:
            dep_result = result.node_results.get(dep_id)
            if not dep_result or dep_result.status != ExecutionStatus.COMPLETED:
                return False
        return True

    def _mark_tree_failed(self, node: WorkflowNode, result: ExecutionResult, error: str):
        """Mark a node and its children as failed."""

        if node.id not in result.node_results:
            result.add_node_result(NodeResult(
                node.id, ExecutionStatus.FAILED, error=error
            ))

        for child in node.children:
            self._mark_tree_failed(child, result, error)

    # Basic node type handlers (mock implementations for Phase 3)

    async def _execute_analyze(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute analysis task."""
        await asyncio.sleep(0.1)  # Simulate work
        return {
            "type": "analysis",
            "description": node.description,
            "result": f"Analysis completed for: {node.description}",
            "parameters": node.parameters
        }

    async def _execute_generate(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute generation task."""
        await asyncio.sleep(0.2)  # Simulate work
        return {
            "type": "generation",
            "description": node.description,
            "result": f"Generated content for: {node.description}",
            "parameters": node.parameters
        }

    async def _execute_transform(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute transformation task."""
        await asyncio.sleep(0.1)  # Simulate work
        return {
            "type": "transformation",
            "description": node.description,
            "result": f"Transformed content for: {node.description}",
            "parameters": node.parameters
        }

    async def _execute_validate(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute validation task."""
        await asyncio.sleep(0.1)  # Simulate work
        return {
            "type": "validation",
            "description": node.description,
            "result": f"Validation passed for: {node.description}",
            "parameters": node.parameters
        }

    async def _execute_composite(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute composite task (children handle the work)."""
        return {
            "type": "composite",
            "description": node.description,
            "result": f"Composite task organized: {node.description}",
            "child_count": len(node.children)
        }

    async def _execute_default(self, node: WorkflowNode) -> Dict[str, Any]:
        """Default handler for unknown node types."""
        await asyncio.sleep(0.1)
        return {
            "type": "default",
            "description": node.description,
            "result": f"Default execution for: {node.description}"
        }
```

##### 2. Enhanced MetaPlanner with Full Execution

**File**: `C:\Users\strau\clawd\vibe-aigc\vibe_aigc\planner.py`
**Changes**: Update MetaPlanner to use the execution engine

```python
"""MetaPlanner: Core orchestration and Vibe decomposition."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from .llm import LLMClient, LLMConfig
from .executor import WorkflowExecutor, ExecutionResult


class MetaPlanner:
    """Central system architect that decomposes Vibes into executable workflows."""

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_client = LLMClient(llm_config)
        self.executor = WorkflowExecutor()

    async def plan(self, vibe: Vibe) -> WorkflowPlan:
        """Generate a WorkflowPlan from a Vibe using LLM decomposition."""

        # Get structured decomposition from LLM
        plan_data = await self.llm_client.decompose_vibe(vibe)

        # Convert LLM response to structured WorkflowPlan
        plan_id = plan_data.get("id", f"plan-{uuid.uuid4().hex[:8]}")

        root_nodes = [
            self._build_workflow_node(node_data)
            for node_data in plan_data.get("root_nodes", [])
        ]

        # Calculate total estimated duration
        total_duration = sum(
            self._calculate_node_duration(node)
            for node in root_nodes
        )

        return WorkflowPlan(
            id=plan_id,
            source_vibe=vibe,
            root_nodes=root_nodes,
            estimated_total_duration=total_duration,
            created_at=datetime.now().isoformat()
        )

    async def execute(self, vibe: Vibe) -> Dict[str, Any]:
        """Plan and execute a Vibe workflow with full execution engine."""

        # Generate execution plan
        plan = await self.plan(vibe)

        # Execute the plan
        execution_result = await self.executor.execute_plan(plan)

        # Format result for API compatibility
        summary = execution_result.get_summary()

        return {
            "status": summary["status"],
            "plan_id": summary["plan_id"],
            "vibe_description": vibe.description,
            "execution_summary": {
                "total_nodes": summary["total_nodes"],
                "completed": summary["completed"],
                "failed": summary["failed"],
                "total_duration": summary["total_duration"],
                "started_at": summary["started_at"],
                "completed_at": summary["completed_at"]
            },
            "node_results": {
                node_id: {
                    "status": result.status.value,
                    "result": result.result,
                    "error": result.error,
                    "duration": result.duration
                }
                for node_id, result in execution_result.node_results.items()
            }
        }

    async def plan_and_execute(self, vibe: Vibe) -> tuple[WorkflowPlan, ExecutionResult]:
        """Get both the plan and execution result (for detailed analysis)."""

        plan = await self.plan(vibe)
        execution_result = await self.executor.execute_plan(plan)
        return plan, execution_result

    def _build_workflow_node(self, node_data: Dict[str, Any]) -> WorkflowNode:
        """Convert LLM response data to WorkflowNode structure."""

        # Validate and normalize node type
        node_type_str = node_data.get("type", "generate").lower()
        try:
            node_type = WorkflowNodeType(node_type_str)
        except ValueError:
            node_type = WorkflowNodeType.GENERATE

        # Recursively build children nodes
        children = [
            self._build_workflow_node(child_data)
            for child_data in node_data.get("children", [])
        ]

        return WorkflowNode(
            id=node_data.get("id", f"task-{uuid.uuid4().hex[:8]}"),
            type=node_type,
            description=node_data.get("description", "Untitled task"),
            parameters=node_data.get("parameters", {}),
            dependencies=node_data.get("dependencies", []),
            children=children,
            estimated_duration=node_data.get("estimated_duration")
        )

    def _calculate_node_duration(self, node: WorkflowNode) -> int:
        """Calculate total estimated duration for a node and its children."""

        node_duration = node.estimated_duration or 0
        children_duration = sum(
            self._calculate_node_duration(child)
            for child in node.children
        )

        return node_duration + children_duration
```

##### 3. Execution Engine Tests

**File**: `C:\Users\strau\clawd\vibe-aigc\tests\test_executor.py`
**Changes**: Comprehensive tests for workflow execution

```python
"""Tests for workflow execution engine."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from vibe_aigc.models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from vibe_aigc.executor import WorkflowExecutor, ExecutionStatus, ExecutionResult


@pytest.mark.asyncio
class TestWorkflowExecutor:
    """Test workflow execution engine functionality."""

    async def test_basic_node_execution(self):
        """Test execution of a single WorkflowNode."""

        executor = WorkflowExecutor()

        node = WorkflowNode(
            id="test-001",
            type=WorkflowNodeType.GENERATE,
            description="Test generation task"
        )

        vibe = Vibe(description="Test vibe")
        plan = WorkflowPlan(
            id="plan-001",
            source_vibe=vibe,
            root_nodes=[node]
        )

        result = await executor.execute_plan(plan)

        assert result.plan_id == "plan-001"
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 1
        assert result.node_results["test-001"].status == ExecutionStatus.COMPLETED
        assert "Generated content" in result.node_results["test-001"].result["result"]

    async def test_hierarchical_execution(self):
        """Test execution of hierarchical workflow with children."""

        executor = WorkflowExecutor()

        child1 = WorkflowNode(
            id="child-001",
            type=WorkflowNodeType.ANALYZE,
            description="Analyze requirements"
        )

        child2 = WorkflowNode(
            id="child-002",
            type=WorkflowNodeType.GENERATE,
            description="Generate based on analysis",
            dependencies=["child-001"]
        )

        parent = WorkflowNode(
            id="parent-001",
            type=WorkflowNodeType.COMPOSITE,
            description="Complete workflow",
            children=[child1, child2]
        )

        vibe = Vibe(description="Test hierarchical vibe")
        plan = WorkflowPlan(
            id="plan-hierarchical",
            source_vibe=vibe,
            root_nodes=[parent]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 3  # parent + 2 children
        assert result.node_results["parent-001"].status == ExecutionStatus.COMPLETED
        assert result.node_results["child-001"].status == ExecutionStatus.COMPLETED
        assert result.node_results["child-002"].status == ExecutionStatus.COMPLETED

    async def test_dependency_handling(self):
        """Test that dependencies are respected in execution order."""

        executor = WorkflowExecutor()

        # Create nodes with dependencies
        node1 = WorkflowNode(
            id="step-001",
            type=WorkflowNodeType.ANALYZE,
            description="First step"
        )

        node2 = WorkflowNode(
            id="step-002",
            type=WorkflowNodeType.GENERATE,
            description="Second step depends on first",
            dependencies=["step-001"]
        )

        vibe = Vibe(description="Test dependency vibe")
        plan = WorkflowPlan(
            id="plan-deps",
            source_vibe=vibe,
            root_nodes=[node1, node2]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.node_results) == 2

        # Check that both nodes completed
        assert result.node_results["step-001"].status == ExecutionStatus.COMPLETED
        assert result.node_results["step-002"].status == ExecutionStatus.COMPLETED

        # Verify dependency was satisfied (timing-based, approximate)
        step1_started = result.node_results["step-001"].started_at
        step2_started = result.node_results["step-002"].started_at
        assert step2_started >= step1_started

    async def test_execution_failure_handling(self):
        """Test handling of node execution failures."""

        executor = WorkflowExecutor()

        # Override handler to simulate failure
        original_handler = executor._execute_generate

        async def failing_handler(node):
            if node.id == "fail-node":
                raise RuntimeError("Simulated failure")
            return await original_handler(node)

        executor._execute_generate = failing_handler

        failing_node = WorkflowNode(
            id="fail-node",
            type=WorkflowNodeType.GENERATE,
            description="This node will fail"
        )

        vibe = Vibe(description="Test failure handling")
        plan = WorkflowPlan(
            id="plan-failure",
            source_vibe=vibe,
            root_nodes=[failing_node]
        )

        result = await executor.execute_plan(plan)

        assert result.status == ExecutionStatus.FAILED
        assert result.node_results["fail-node"].status == ExecutionStatus.FAILED
        assert "Simulated failure" in result.node_results["fail-node"].error


@pytest.mark.asyncio
class TestEndToEndExecution:
    """Test complete end-to-end execution flow."""

    @patch('vibe_aigc.planner.LLMClient')
    async def test_complete_vibe_execution(self, mock_llm_client):
        """Test complete flow from Vibe to execution results."""

        from vibe_aigc.planner import MetaPlanner

        # Mock LLM response
        mock_client_instance = AsyncMock()
        mock_client_instance.decompose_vibe.return_value = {
            "id": "plan-e2e-001",
            "root_nodes": [
                {
                    "id": "analyze-scene",
                    "type": "analyze",
                    "description": "Analyze scene requirements",
                    "parameters": {"detail_level": "high"},
                    "dependencies": [],
                    "children": [],
                    "estimated_duration": 30
                },
                {
                    "id": "generate-scene",
                    "type": "generate",
                    "description": "Generate cinematic scene",
                    "parameters": {"style": "dark, atmospheric"},
                    "dependencies": ["analyze-scene"],
                    "children": [],
                    "estimated_duration": 60
                }
            ]
        }
        mock_llm_client.return_value = mock_client_instance

        # Test complete execution
        planner = MetaPlanner()
        vibe = Vibe(
            description="Create a cinematic sci-fi scene",
            style="dark, atmospheric",
            constraints=["no violence", "PG-13"]
        )

        result = await planner.execute(vibe)

        assert result["status"] == "completed"
        assert result["plan_id"] == "plan-e2e-001"
        assert result["vibe_description"] == "Create a cinematic sci-fi scene"

        execution_summary = result["execution_summary"]
        assert execution_summary["total_nodes"] == 2
        assert execution_summary["completed"] == 2
        assert execution_summary["failed"] == 0

        node_results = result["node_results"]
        assert "analyze-scene" in node_results
        assert "generate-scene" in node_results
        assert node_results["analyze-scene"]["status"] == "completed"
        assert node_results["generate-scene"]["status"] == "completed"
```

##### 4. Update Package Exports

**File**: `C:\Users\strau\clawd\vibe-aigc\vibe_aigc\__init__.py`
**Changes**: Add executor exports

```python
"""Vibe AIGC: A New Paradigm for Content Generation via Agentic Orchestration."""

from .models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from .planner import MetaPlanner
from .llm import LLMClient, LLMConfig
from .executor import WorkflowExecutor, ExecutionStatus, ExecutionResult

__version__ = "0.1.0"
__all__ = [
    "Vibe", "WorkflowPlan", "WorkflowNode", "WorkflowNodeType",
    "MetaPlanner", "LLMClient", "LLMConfig",
    "WorkflowExecutor", "ExecutionStatus", "ExecutionResult"
]
```

#### Success Criteria:

##### Automated Verification:

- [ ] Tests pass: `pytest tests/`
- [ ] Type checking passes: `mypy vibe_aigc/`
- [ ] End-to-end tests pass with mocked LLM responses
- [ ] Execution engine handles hierarchical workflows correctly
- [ ] Dependency resolution works for sequential tasks

##### Manual Verification:

- [ ] Can execute complete workflow: `result = await planner.execute(vibe)`
- [ ] Execution returns structured results with node-level details
- [ ] Hierarchical workflows execute children after parents
- [ ] Failed nodes don't prevent other nodes from executing
- [ ] Execution timing and duration tracking work correctly

**Note**: Complete all automated verification, then pause for manual confirmation before proceeding to next phase.

---

### Phase 4: Integration Testing and Documentation

#### Overview

Validate the complete end-to-end system matches the README.md usage example, add comprehensive testing, and ensure the package is ready for use. This phase focuses on integration validation and polishing the implementation.

#### Changes Required:

##### 1. End-to-End Integration Tests

**File**: `C:\Users\strau\clawd\vibe-aigc\tests\test_integration.py`
**Changes**: Complete integration tests matching README.md usage

```python
"""End-to-end integration tests for the complete Vibe AIGC system."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from vibe_aigc import MetaPlanner, Vibe


@pytest.mark.asyncio
class TestReadmeUsageExample:
    """Test that the exact README.md usage example works."""

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_readme_example_exact_match(self, mock_openai):
        """Test the exact code example from README.md:39-50."""

        # Mock OpenAI client to return valid workflow plan
        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = '''
        {
            "id": "plan-cinematic-scifi",
            "root_nodes": [
                {
                    "id": "concept-development",
                    "type": "analyze",
                    "description": "Develop core sci-fi concept respecting constraints",
                    "parameters": {
                        "style": "dark, atmospheric",
                        "constraints": ["no violence", "PG-13"],
                        "domain": "cinematic"
                    },
                    "dependencies": [],
                    "children": [
                        {
                            "id": "mood-analysis",
                            "type": "analyze",
                            "description": "Analyze dark atmospheric mood requirements",
                            "parameters": {"mood": "dark, atmospheric"},
                            "dependencies": [],
                            "children": [],
                            "estimated_duration": 15
                        },
                        {
                            "id": "constraint-validation",
                            "type": "validate",
                            "description": "Ensure PG-13 rating and no violence",
                            "parameters": {"constraints": ["no violence", "PG-13"]},
                            "dependencies": ["mood-analysis"],
                            "children": [],
                            "estimated_duration": 10
                        }
                    ],
                    "estimated_duration": 45
                },
                {
                    "id": "scene-generation",
                    "type": "generate",
                    "description": "Generate cinematic sci-fi scene with validated concept",
                    "parameters": {
                        "style": "cinematic",
                        "genre": "sci-fi",
                        "mood": "dark, atmospheric"
                    },
                    "dependencies": ["concept-development"],
                    "children": [],
                    "estimated_duration": 90
                },
                {
                    "id": "final-review",
                    "type": "validate",
                    "description": "Final review of generated scene against original vibe",
                    "parameters": {"review_type": "comprehensive"},
                    "dependencies": ["scene-generation"],
                    "children": [],
                    "estimated_duration": 20
                }
            ]
        }
        '''
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        # Execute the exact README example
        # from vibe_aigc import MetaPlanner, Vibe  # Already imported

        # Define your vibe
        vibe = Vibe(
            description="Create a cinematic sci-fi scene",
            style="dark, atmospheric",
            constraints=["no violence", "PG-13"]
        )

        # Plan and execute
        planner = MetaPlanner()
        result = await planner.execute(vibe)

        # Verify the result structure and content
        assert result["status"] == "completed"
        assert result["vibe_description"] == "Create a cinematic sci-fi scene"

        execution_summary = result["execution_summary"]
        assert execution_summary["total_nodes"] == 6  # 3 root + 3 children
        assert execution_summary["completed"] == 6
        assert execution_summary["failed"] == 0

        # Verify specific workflow nodes were executed
        node_results = result["node_results"]
        assert "concept-development" in node_results
        assert "scene-generation" in node_results
        assert "final-review" in node_results
        assert "mood-analysis" in node_results  # child node
        assert "constraint-validation" in node_results  # child node

        # Verify all nodes completed successfully
        for node_id, node_result in node_results.items():
            assert node_result["status"] == "completed"
            assert node_result["result"] is not None
            assert node_result["error"] is None
            assert node_result["duration"] >= 0


@pytest.mark.asyncio
class TestRobustnessAndEdgeCases:
    """Test system robustness and edge case handling."""

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_minimal_vibe_execution(self, mock_openai):
        """Test with minimal Vibe configuration."""

        # Mock minimal LLM response
        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = '''
        {
            "id": "plan-minimal",
            "root_nodes": [
                {
                    "id": "simple-task",
                    "type": "generate",
                    "description": "Handle minimal vibe request",
                    "parameters": {},
                    "dependencies": [],
                    "children": [],
                    "estimated_duration": 30
                }
            ]
        }
        '''
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        # Test minimal vibe (only required field)
        vibe = Vibe(description="Simple test")
        planner = MetaPlanner()

        result = await planner.execute(vibe)

        assert result["status"] == "completed"
        assert result["vibe_description"] == "Simple test"
        assert result["execution_summary"]["total_nodes"] == 1
        assert result["execution_summary"]["completed"] == 1

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_complex_hierarchical_workflow(self, mock_openai):
        """Test deeply nested hierarchical workflow."""

        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = '''
        {
            "id": "plan-complex",
            "root_nodes": [
                {
                    "id": "phase1",
                    "type": "composite",
                    "description": "Phase 1: Analysis and Planning",
                    "parameters": {},
                    "dependencies": [],
                    "children": [
                        {
                            "id": "deep-analysis",
                            "type": "analyze",
                            "description": "Deep requirement analysis",
                            "parameters": {},
                            "dependencies": [],
                            "children": [
                                {
                                    "id": "user-intent",
                                    "type": "analyze",
                                    "description": "Analyze user intent",
                                    "parameters": {},
                                    "dependencies": [],
                                    "children": [],
                                    "estimated_duration": 10
                                },
                                {
                                    "id": "technical-constraints",
                                    "type": "analyze",
                                    "description": "Analyze technical constraints",
                                    "parameters": {},
                                    "dependencies": ["user-intent"],
                                    "children": [],
                                    "estimated_duration": 15
                                }
                            ],
                            "estimated_duration": 30
                        },
                        {
                            "id": "strategy-planning",
                            "type": "generate",
                            "description": "Generate execution strategy",
                            "parameters": {},
                            "dependencies": ["deep-analysis"],
                            "children": [],
                            "estimated_duration": 25
                        }
                    ],
                    "estimated_duration": 70
                }
            ]
        }
        '''
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        vibe = Vibe(
            description="Complex multi-phase project",
            style="systematic, thorough",
            constraints=["high quality", "detailed analysis"]
        )
        planner = MetaPlanner()

        result = await planner.execute(vibe)

        assert result["status"] == "completed"

        # Verify all levels of hierarchy executed
        node_results = result["node_results"]
        assert "phase1" in node_results  # Root composite
        assert "deep-analysis" in node_results  # Level 2 composite
        assert "strategy-planning" in node_results  # Level 2 leaf
        assert "user-intent" in node_results  # Level 3 leaf
        assert "technical-constraints" in node_results  # Level 3 leaf

        # Verify execution order respected dependencies
        assert all(result["status"] == "completed" for result in node_results.values())

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_llm_error_handling(self, mock_openai):
        """Test handling of LLM API errors."""

        # Mock LLM client to raise an exception
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        vibe = Vibe(description="Test error handling")
        planner = MetaPlanner()

        # Should raise RuntimeError with descriptive message
        with pytest.raises(RuntimeError, match="LLM request failed"):
            await planner.execute(vibe)

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_malformed_llm_response(self, mock_openai):
        """Test handling of malformed LLM responses."""

        # Mock LLM to return invalid JSON
        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = "This is not valid JSON"
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        vibe = Vibe(description="Test malformed response")
        planner = MetaPlanner()

        # Should raise ValueError for invalid JSON
        with pytest.raises(ValueError, match="Invalid JSON response from LLM"):
            await planner.execute(vibe)


@pytest.mark.asyncio
class TestPlanningWithoutExecution:
    """Test the planning phase independently."""

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_plan_only_workflow(self, mock_openai):
        """Test generating a plan without executing it."""

        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = '''
        {
            "id": "plan-only-test",
            "root_nodes": [
                {
                    "id": "planning-task",
                    "type": "analyze",
                    "description": "Planning task for testing",
                    "parameters": {"mode": "planning"},
                    "dependencies": [],
                    "children": [],
                    "estimated_duration": 40
                }
            ]
        }
        '''
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        vibe = Vibe(description="Test planning without execution")
        planner = MetaPlanner()

        # Test plan() method directly
        plan = await planner.plan(vibe)

        assert plan.id == "plan-only-test"
        assert plan.source_vibe.description == "Test planning without execution"
        assert len(plan.root_nodes) == 1
        assert plan.root_nodes[0].description == "Planning task for testing"
        assert plan.estimated_total_duration == 40
        assert plan.created_at is not None

    @patch('vibe_aigc.llm.AsyncOpenAI')
    async def test_plan_and_execute_separation(self, mock_openai):
        """Test plan_and_execute method for getting both plan and results."""

        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices[0].message.content = '''
        {
            "id": "plan-execute-test",
            "root_nodes": [
                {
                    "id": "separation-task",
                    "type": "validate",
                    "description": "Test plan/execute separation",
                    "parameters": {},
                    "dependencies": [],
                    "children": [],
                    "estimated_duration": 20
                }
            ]
        }
        '''
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        vibe = Vibe(description="Test plan and execute separation")
        planner = MetaPlanner()

        plan, execution_result = await planner.plan_and_execute(vibe)

        # Verify plan structure
        assert plan.id == "plan-execute-test"
        assert len(plan.root_nodes) == 1

        # Verify execution result
        assert execution_result.plan_id == "plan-execute-test"
        assert execution_result.status.value == "completed"
        assert len(execution_result.node_results) == 1
        assert "separation-task" in execution_result.node_results
```

##### 2. Package Configuration Improvements

**File**: `C:\Users\strau\clawd\vibe-aigc\pyproject.toml`
**Changes**: Enhanced package metadata and development tools

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "vibe-aigc"
version = "0.1.0"
description = "A New Paradigm for Content Generation via Agentic Orchestration"
readme = "README.md"
authors = [{name = "Vibe AIGC Contributors"}]
license = {text = "MIT"}
requires-python = ">=3.12"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
dependencies = [
    "pydantic>=2.0.0",
    "openai>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.urls]
"Homepage" = "https://github.com/vibe-aigc/vibe-aigc"
"Bug Tracker" = "https://github.com/vibe-aigc/vibe-aigc/issues"

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --cov=vibe_aigc --cov-report=term-missing"
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.black]
line-length = 100
target-version = ["py312"]

[tool.ruff]
line-length = 100
target-version = "py312"
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

##### 3. Enhanced .gitignore

**File**: `C:\Users\strau\clawd\vibe-aigc\.gitignore`
**Changes**: Comprehensive Python package .gitignore

```gitignore
# Wreckit local config (may contain secrets)
.wreckit/config.local.json

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# OpenAI API keys and secrets
.env.local
*.key
```

##### 4. Development Scripts

**File**: `C:\Users\strau\clawd\vibe-aigc\scripts\dev_test.py`
**Changes**: Development testing script for manual validation

```python
#!/usr/bin/env python3
"""Development testing script for manual validation of Vibe AIGC system."""

import asyncio
import os
from vibe_aigc import MetaPlanner, Vibe


async def test_readme_example():
    """Test the exact README.md example."""

    print("🚀 Testing README.md Example")
    print("=" * 50)

    # Define your vibe (from README.md:42-46)
    vibe = Vibe(
        description="Create a cinematic sci-fi scene",
        style="dark, atmospheric",
        constraints=["no violence", "PG-13"]
    )

    print(f"Vibe: {vibe.description}")
    print(f"Style: {vibe.style}")
    print(f"Constraints: {vibe.constraints}")
    print()

    # Plan and execute (from README.md:49-50)
    planner = MetaPlanner()

    try:
        print("🧠 Planning workflow...")
        result = await planner.execute(vibe)

        print("✅ Execution completed!")
        print(f"Status: {result['status']}")
        print(f"Plan ID: {result['plan_id']}")

        summary = result['execution_summary']
        print(f"Total nodes: {summary['total_nodes']}")
        print(f"Completed: {summary['completed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Duration: {summary['total_duration']:.2f}s")

        print("\nNode Results:")
        for node_id, node_result in result['node_results'].items():
            status_emoji = "✅" if node_result['status'] == 'completed' else "❌"
            print(f"  {status_emoji} {node_id}: {node_result['result']['description'] if node_result['result'] else 'No result'}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have set OPENAI_API_KEY environment variable")


async def test_planning_only():
    """Test planning without execution."""

    print("\n🎯 Testing Planning Only")
    print("=" * 50)

    vibe = Vibe(
        description="Design a futuristic user interface",
        style="minimalist, clean",
        domain="ui/ux"
    )

    planner = MetaPlanner()

    try:
        plan = await planner.plan(vibe)

        print(f"Generated plan: {plan.id}")
        print(f"Source vibe: {plan.source_vibe.description}")
        print(f"Estimated duration: {plan.estimated_total_duration}s")
        print(f"Root nodes: {len(plan.root_nodes)}")

        print("\nWorkflow Structure:")
        for i, node in enumerate(plan.root_nodes, 1):
            print(f"  {i}. {node.description} ({node.type.value})")
            for j, child in enumerate(node.children, 1):
                print(f"     {i}.{j} {child.description} ({child.type.value})")

    except Exception as e:
        print(f"❌ Planning error: {e}")


async def test_minimal_vibe():
    """Test with minimal Vibe configuration."""

    print("\n🔧 Testing Minimal Vibe")
    print("=" * 50)

    vibe = Vibe(description="Hello world")
    planner = MetaPlanner()

    try:
        result = await planner.execute(vibe)
        print(f"Minimal vibe executed successfully: {result['status']}")
        print(f"Nodes executed: {result['execution_summary']['total_nodes']}")

    except Exception as e:
        print(f"❌ Minimal test error: {e}")


async def main():
    """Run all development tests."""

    print("🧪 Vibe AIGC Development Testing")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. LLM tests will fail.")
        print("   Set your API key: export OPENAI_API_KEY='your-key-here'")
        print()

    await test_readme_example()
    await test_planning_only()
    await test_minimal_vibe()

    print("\n🎉 Development testing complete!")
    print("   Run 'pytest tests/' for comprehensive test suite")


if __name__ == "__main__":
    asyncio.run(main())
```

##### 5. Type Import Fixes

**File**: `C:\Users\strau\clawd\vibe-aigc\vibe_aigc\planner.py`
**Changes**: Add missing Optional import

```python
"""MetaPlanner: Core orchestration and Vibe decomposition."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import Vibe, WorkflowPlan, WorkflowNode, WorkflowNodeType
from .llm import LLMClient, LLMConfig
from .executor import WorkflowExecutor, ExecutionResult

# ... rest of file remains the same
```

#### Success Criteria:

##### Automated Verification:

- [ ] All tests pass: `pytest tests/` (including new integration tests)
- [ ] Type checking passes: `mypy vibe_aigc/`
- [ ] Code formatting: `black --check vibe_aigc/`
- [ ] Linting passes: `ruff check vibe_aigc/`
- [ ] Package builds: `python -m build`
- [ ] Coverage report shows adequate test coverage

##### Manual Verification:

- [ ] Development script runs without errors: `python scripts/dev_test.py`
- [ ] README.md example works exactly as shown
- [ ] Package installs cleanly: `pip install -e .`
- [ ] All imports work: `from vibe_aigc import MetaPlanner, Vibe`
- [ ] Can create and execute complex hierarchical workflows
- [ ] Error handling gracefully handles LLM API issues

**Note**: Complete all automated verification, then pause for manual confirmation before considering the implementation complete.

---

## Testing Strategy

### Unit Tests:

- **Model validation**: Pydantic models validate input correctly and handle edge cases
- **Node type handling**: WorkflowNodeType enum validation and conversion
- **Hierarchical structure**: WorkflowNode children and dependency relationships
- **LLM client**: Mocked LLM responses and error handling (API failures, malformed JSON)
- **Workflow execution**: Individual node execution, dependency resolution, failure handling
- **Duration calculation**: Accurate timing for nodes and hierarchical structures

### Integration Tests:

- **End-to-end Vibe execution**: Complete flow from Vibe creation to results
- **README.md usage example**: Exact code from documentation works as expected
- **Complex hierarchical workflows**: Multi-level decomposition and execution
- **Plan and execute separation**: Independent planning vs. execution functionality
- **Error propagation**: LLM errors surface correctly through the system
- **Performance characteristics**: Execution completes in reasonable time

### Manual Testing Steps:

1. **Basic Import Test**: Verify `from vibe_aigc import MetaPlanner, Vibe` works
2. **README Example**: Run exact code from README.md:39-50 with real OpenAI API
3. **Planning Validation**: Create complex Vibe and inspect generated WorkflowPlan structure
4. **Execution Monitoring**: Watch node execution order and dependency resolution
5. **Error Handling**: Test with invalid API keys, network issues, malformed responses
6. **Performance Check**: Verify reasonable execution times for typical workflows

## Migration Notes

No migration required - this is a new package. Installation steps:

1. **Clone Repository**: `git clone <repo-url>`
2. **Install Package**: `cd vibe-aigc && pip install -e .`
3. **Set API Key**: `export OPENAI_API_KEY="your-openai-api-key"`
4. **Verify Installation**: `python scripts/dev_test.py`

For development:
1. **Install Dev Dependencies**: `pip install -e ".[dev]"`
2. **Run Tests**: `pytest tests/`
3. **Code Quality**: `black vibe_aigc/ && ruff check vibe_aigc/ && mypy vibe_aigc/`

## References

- Research: `C:\Users\strau\clawd\vibe-aigc\.wreckit\items\001-vibe-model-meta-planner\research.md`
- README.md Usage Example: `C:\Users\strau\clawd\vibe-aigc\README.md:39-50`
- Success Criteria: `C:\Users\strau\clawd\vibe-aigc\.wreckit\items\001-vibe-model-meta-planner\item.json:17-21`
- Technical Constraints: `C:\Users\strau\clawd\vibe-aigc\.wreckit\items\001-vibe-model-meta-planner\item.json:24-27`