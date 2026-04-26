import pytest
from argument_lab.core.agents import _get_prior_opponent_claim_ids, _update_state_from_argument
from argument_lab.core.models import Argument

def test_get_prior_opponent_claim_ids():
    state = {
        "current_round": 2,
        "arguments": [
            Argument.model_construct(id="a1", round=1, agent="proponent", claim="p1"),
            Argument.model_construct(id="a2", round=1, agent="opponent", claim="o1"),
        ]
    }
    ids = _get_prior_opponent_claim_ids(state, "proponent")
    assert ids == ["a2"]
    
    ids = _get_prior_opponent_claim_ids(state, "opponent")
    assert ids == ["a1"]

def test_update_state_from_argument():
    arg = Argument.model_construct(
        id="a2", round=2, agent="proponent", claim="p2", 
        counterpoints_addressed=["o1"], confidence_score=0.9
    )
    
    state = {
        "arguments": [
            Argument.model_construct(id="o1", round=1, agent="opponent"),
            Argument.model_construct(id="o2", round=1, agent="opponent")
        ],
        "ignored_claims": set(),
        "agent_positions": {"proponent": [0.8]}
    }
    
    updates = _update_state_from_argument(arg, state)
    
    assert "a2" in updates["claims_registry"]
    assert updates["addressed_claims"] == {"o1"}
    assert updates["ignored_claims"] == {"o2"}
    assert updates["agent_positions"]["proponent"] == [0.8, 0.9]
