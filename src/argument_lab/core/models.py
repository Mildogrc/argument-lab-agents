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
        description="Must contain ≥1 retrieved source"
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

    @property
    def composite(self) -> float:
        """Weighted composite per architecture spec."""
        return round(
            self.logical_consistency * 0.30
            + self.evidence_support * 0.30
            + self.relevance * 0.20
            + self.completeness * 0.20,
            4,
        )


class JudgeEvaluation(BaseModel):
    round: int
    proponent_score: ArgumentScore
    opponent_score: ArgumentScore
    convergence_detected: bool = False
    stalemate_detected: bool = False
    explanation: str


# ---------------------------------------------------------------------------
# Hallucination checker output
# ---------------------------------------------------------------------------

class HallucinationFlag(BaseModel):
    claim_id: str
    reason: str
    severity: Literal["low", "medium", "high"]


class HallucinationReport(BaseModel):
    flags: list[HallucinationFlag] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Contradiction checker output
# ---------------------------------------------------------------------------

class ContradictionFlag(BaseModel):
    claim_id: str        # current claim that contradicts a prior one
    prior_claim_id: str  # the earlier claim it contradicts
    contradiction_type: Literal[
        "direct_negation",
        "weakened_commitment",
        "shifted_evidence_basis",
        "ignored_own_prior_claim",
    ]
    explanation: str


class ContradictionReport(BaseModel):
    flags: list[ContradictionFlag] = Field(default_factory=list)
