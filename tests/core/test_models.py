import pytest
from pydantic import ValidationError
from argument_lab.core.models import Argument, EvidenceRef

def test_argument_requires_evidence():
    with pytest.raises(ValidationError) as exc_info:
        Argument(
            id="arg_1",
            round=1,
            agent="proponent",
            claim="AI is good.",
            evidence=[],
            assumptions=[],
            counterpoints_addressed=[],
            confidence_score=0.9
        )
    assert "at least 1" in str(exc_info.value).lower() or "min_length" in str(exc_info.value).lower()

def test_valid_argument():
    ev = EvidenceRef(source_id="doc_1", excerpt="AI helps.", reliability_score=0.8)
    arg = Argument(
        id="arg_1",
        round=1,
        agent="proponent",
        claim="AI is good.",
        evidence=[ev],
        assumptions=[],
        counterpoints_addressed=[],
        confidence_score=0.9
    )
    assert arg.evidence[0].source_id == "doc_1"
