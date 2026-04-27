from typing import Literal
from pydantic import BaseModel, Field

class EvidenceRef(BaseModel):
    source_id: str
    excerpt: str
    reliability_score: float = Field(ge=0.0, le=1.0)

class Argument(BaseModel):
    id: str
    round: int
    agent: Literal["proponent", "opponent"]
    claim: str
    evidence: list[EvidenceRef] = Field(
        min_length=1, 
        description="Must contain \u22651 retrieved source"
    )
    assumptions: list[str]
    counterpoints_addressed: list[str] = Field(
        default_factory=list,
        description="Claim IDs of opponent's prior points"
    )
    confidence_score: float = Field(ge=0.0, le=1.0)

class Claim(BaseModel):
    id: str
    text: str
    agent: str
    round: int

class ArgumentScore(BaseModel):
    logical_consistency: float = Field(ge=0.0, le=1.0)
    evidence_support: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    
class JudgeEvaluation(BaseModel):
    proponent_score: ArgumentScore
    opponent_score: ArgumentScore
    convergence_detected: bool
    explanation: str
    hallucination_flags: list[str] = Field(default_factory=list,
        description="Claim IDs where evidence grounding failed")
    contradiction_flags: list[str] = Field(default_factory=list)
