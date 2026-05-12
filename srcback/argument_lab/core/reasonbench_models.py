from enum import Enum
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    TASK_1_LOGIC = "task_1_logic"
    TASK_2_STRATEGY = "task_2_strategy"
    TASK_3_TRADEOFF = "task_3_tradeoff"


class ReasonBenchResponse(BaseModel):
    """Shared output schema for all models across all ReasonBench tasks."""

    strategy_or_answer: str = Field(description="Final answer or plan")
    rationale: str = Field(description="Step-by-step reasoning")
    assumptions: list[str] = Field(description="Explicit assumptions made")
    opponent_model: str = Field(
        description="What the model believes about the opponent (if applicable)"
    )
    risks: list[str] = Field(description="Failure modes or weaknesses")
    conditions: list[str] = Field(description="When the answer/strategy would change")


# ---------------------------------------------------------------------------
# Task 1: Deterministic Logic (Constraint Puzzle)
# ---------------------------------------------------------------------------


class Task1Score(BaseModel):
    correctness: int = Field(
        ge=0, le=2, description="0: incorrect, 1: partially correct, 2: fully correct"
    )
    logical_consistency: int = Field(
        ge=0, le=2, description="0: contradictory, 1: minor issues, 2: fully consistent"
    )
    completeness: int = Field(
        ge=0, le=2, description="0: incomplete, 1: partial, 2: fully explains all boxes"
    )
    responsiveness: int = Field(
        ge=0,
        le=2,
        description="0: ignores critique, 1: partially integrates, 2: fully integrates",
    )


class Task1Evaluation(BaseModel):
    proponent_score: Task1Score
    opponent_score: Task1Score
    explanation: str = Field(description="Judge's explanation for the assigned scores.")


# ---------------------------------------------------------------------------
# Task 2: Strategic Reasoning (Asymmetric Game)
# ---------------------------------------------------------------------------


class Task2Score(BaseModel):
    opponent_modeling: int = Field(
        ge=0, le=2, description="0: ignores, 1: partial, 2: uses strategically"
    )
    strategic_coherence: int = Field(
        ge=0, le=2, description="0: inconsistent, 1: partial, 2: structured plan"
    )
    risk_awareness: int = Field(
        ge=0, le=2, description="0: ignores, 1: partial, 2: balances risk/reward"
    )
    conditional_reasoning: int = Field(
        ge=0, le=2, description="0: static, 1: partial, 2: adaptive plan"
    )
    responsiveness: int = Field(
        ge=0,
        le=2,
        description="0: ignores critique, 1: partially integrates, 2: fully integrates",
    )


class Task2Evaluation(BaseModel):
    proponent_score: Task2Score
    opponent_score: Task2Score
    explanation: str = Field(description="Judge's explanation for the assigned scores.")


# ---------------------------------------------------------------------------
# Task 3: Constrained Tradeoff Reasoning
# ---------------------------------------------------------------------------


class Task3Score(BaseModel):
    constraint_utilization: int = Field(
        ge=0, le=2, description="0: ignores, 1: partial, 2: deeply used"
    )
    tradeoff_specificity: int = Field(
        ge=0, le=2, description="0: generic, 1: partial, 2: contextual"
    )
    assumptions_quality: int = Field(
        ge=0, le=2, description="0: implicit, 1: partial, 2: explicit"
    )
    risk_analysis: int = Field(
        ge=0, le=2, description="0: vague, 1: partial, 2: concrete"
    )
    conditional_reasoning: int = Field(
        ge=0, le=2, description="0: static, 1: partial, 2: adaptive"
    )
    responsiveness: int = Field(
        ge=0,
        le=2,
        description="0: ignores critique, 1: partially integrates, 2: fully integrates",
    )


class Task3Evaluation(BaseModel):
    proponent_score: Task3Score
    opponent_score: Task3Score
    explanation: str = Field(description="Judge's explanation for the assigned scores.")
