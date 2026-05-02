from typing import Annotated, Literal, TypedDict
import operator

from argument_lab.core.models import Argument, Claim, JudgeEvaluation

MAX_ROUNDS = 3

def union_sets(a: set[str] | None, b: set[str] | None) -> set[str]:
    return (a or set()) | (b or set())

def merge_dicts(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}

def max_round(a: int | None, b: int | None) -> int:
    return max(a or 0, b or 0)

def merge_status(a: str | None, b: str | None) -> str:
    priority = ["terminated", "stalemate", "converged", "in_progress"]
    a_val = a if a in priority else "in_progress"
    b_val = b if b in priority else "in_progress"
    return a_val if priority.index(a_val) < priority.index(b_val) else b_val

class DebateState(TypedDict):
    proposition: str
    current_round: Annotated[int, max_round]
    arguments: Annotated[list[Argument], operator.add]
    claims_registry: Annotated[dict[str, Claim], merge_dicts]
    addressed_claims: Annotated[set[str], union_sets]
    ignored_claims: Annotated[set[str], union_sets]
    agent_positions: Annotated[dict[str, list[float]], merge_dicts]
    repetition_flags: Annotated[list[str], operator.add]
    status: Annotated[Literal["in_progress", "converged", "stalemate", "terminated"], merge_status]
    hallucination_flags: Annotated[list[str], operator.add]
    contradiction_flags: Annotated[list[str], operator.add]
    scores: Annotated[list[JudgeEvaluation], operator.add]
