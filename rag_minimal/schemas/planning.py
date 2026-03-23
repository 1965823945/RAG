"""Planning Agent schemas for task decomposition, chain of thought, and self-reflection."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .base import ToolOutput


# ─────────────────────────────────────────────────────────────
# Task Status & Priority
# ─────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"  # 待执行
    IN_PROGRESS = "in_progress"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    BLOCKED = "blocked"  # 被阻塞
    CANCELLED = "cancelled"  # 已取消


class TaskPriority(str, Enum):
    """Task priority levels."""

    CRITICAL = "critical"  # 关键
    HIGH = "high"  # 高
    MEDIUM = "medium"  # 中
    LOW = "low"  # 低


# ─────────────────────────────────────────────────────────────
# Task Decomposition Schemas
# ─────────────────────────────────────────────────────────────


class SubTask(BaseModel):
    """A subtask decomposed from the main task."""

    id: str = Field(..., description="Unique subtask ID")
    description: str = Field(..., description="What this subtask does")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status")
    priority: TaskPriority = Field(
        default=TaskPriority.MEDIUM, description="Task priority"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="IDs of subtasks this depends on"
    )
    result: Optional[str] = Field(default=None, description="Execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    max_retries: int = Field(default=2, description="Maximum retry attempts")


class TaskDecomposition(BaseModel):
    """Result of task decomposition."""

    original_task: str = Field(..., description="The original user task")
    subtasks: List[SubTask] = Field(
        default_factory=list, description="Decomposed subtasks"
    )
    total_steps: int = Field(default=0, description="Total number of steps")
    complexity: str = Field(
        default="medium", description="Estimated complexity: simple/medium/complex"
    )
    estimated_time: Optional[str] = Field(
        default=None, description="Estimated completion time"
    )


# ─────────────────────────────────────────────────────────────
# Chain of Thought Schemas
# ─────────────────────────────────────────────────────────────


class ThoughtStep(BaseModel):
    """A single step in the chain of thought."""

    step_number: int = Field(..., description="Step number in the chain")
    thought: str = Field(..., description="The reasoning thought")
    action: Optional[str] = Field(default=None, description="Action to take")
    action_input: Optional[Dict[str, Any]] = Field(
        default=None, description="Input for the action"
    )
    observation: Optional[str] = Field(default=None, description="Result of the action")
    timestamp: Optional[str] = Field(
        default=None, description="When this step occurred"
    )


class ChainOfThoughtResult(BaseModel):
    """Complete chain of thought reasoning."""

    question: str = Field(..., description="The question being reasoned about")
    thoughts: List[ThoughtStep] = Field(
        default_factory=list, description="Chain of thought steps"
    )
    final_answer: Optional[str] = Field(default=None, description="Final conclusion")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in the answer"
    )
    total_steps: int = Field(default=0, description="Total reasoning steps")


# ─────────────────────────────────────────────────────────────
# Self-Reflection Schemas
# ─────────────────────────────────────────────────────────────


class ReflectionType(str, Enum):
    """Types of self-reflection."""

    QUALITY_CHECK = "quality_check"  # 质量检查
    ERROR_ANALYSIS = "error_analysis"  # 错误分析
    IMPROVEMENT = "improvement"  # 改进建议
    COMPLETENESS = "completeness"  # 完整性检查
    CONSISTENCY = "consistency"  # 一致性检查


class ReflectionItem(BaseModel):
    """A single reflection insight."""

    reflection_type: ReflectionType = Field(..., description="Type of reflection")
    aspect: str = Field(..., description="What aspect is being reflected on")
    assessment: str = Field(..., description="Assessment of this aspect")
    score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Score for this aspect"
    )
    issues: List[str] = Field(default_factory=list, description="Issues identified")
    suggestions: List[str] = Field(
        default_factory=list, description="Improvement suggestions"
    )


class SelfReflectionResult(BaseModel):
    """Result of self-reflection process."""

    context: str = Field(..., description="What is being reflected upon")
    reflections: List[ReflectionItem] = Field(
        default_factory=list, description="List of reflections"
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall quality score"
    )
    should_retry: bool = Field(default=False, description="Whether to retry the task")
    retry_reason: Optional[str] = Field(
        default=None, description="Reason for retry if applicable"
    )
    improvements: List[str] = Field(
        default_factory=list, description="Suggested improvements"
    )


# ─────────────────────────────────────────────────────────────
# Planning Agent State & Output
# ─────────────────────────────────────────────────────────────


class PlanningAgentState(BaseModel):
    """State of the planning agent execution."""

    task_id: str = Field(..., description="Unique task execution ID")
    original_query: str = Field(..., description="Original user query")

    # Task decomposition
    decomposition: Optional[TaskDecomposition] = Field(
        default=None, description="Task decomposition result"
    )

    # Chain of thought
    reasoning: Optional[ChainOfThoughtResult] = Field(
        default=None, description="Chain of thought reasoning"
    )

    # Self reflection
    reflection: Optional[SelfReflectionResult] = Field(
        default=None, description="Self reflection result"
    )

    # Execution state
    current_step: int = Field(default=0, description="Current execution step")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Overall status")
    iterations: int = Field(default=0, description="Number of iterations")
    max_iterations: int = Field(default=3, description="Maximum iterations allowed")

    # Results
    final_answer: Optional[str] = Field(default=None, description="Final answer")
    execution_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="History of all executions"
    )

    # Timing
    start_time: Optional[str] = Field(default=None, description="Execution start time")
    end_time: Optional[str] = Field(default=None, description="Execution end time")


class PlanningAgentOutput(ToolOutput):
    """Output from the planning agent."""

    task_id: str = Field(default="", description="Task execution ID")
    query: str = Field(default="", description="Original query")
    answer: str = Field(default="", description="Final answer")

    # Planning details
    subtasks_completed: int = Field(default=0, description="Subtasks completed")
    subtasks_total: int = Field(default=0, description="Total subtasks")
    reasoning_steps: int = Field(default=0, description="Reasoning steps taken")
    iterations: int = Field(default=0, description="Improvement iterations")

    # Quality metrics
    confidence: float = Field(default=0.0, description="Answer confidence")
    quality_score: float = Field(default=0.0, description="Quality score")

    # Detailed state (optional)
    state: Optional[PlanningAgentState] = Field(
        default=None, description="Full agent state"
    )
