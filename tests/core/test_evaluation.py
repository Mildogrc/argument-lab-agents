import pytest
from argument_lab.core.evaluation import _get_current_round_args, EvaluationError
from argument_lab.core.models import Argument

def test_get_current_round_args():
    p_arg = Argument.model_construct(agent="proponent", round=2)
    o_arg = Argument.model_construct(agent="opponent", round=2)
    old_p_arg = Argument.model_construct(agent="proponent", round=1)

    state = {
        "current_round": 2,
        "arguments": [old_p_arg, p_arg, o_arg]
    }

    p, o = _get_current_round_args(state)
    assert p is p_arg
    assert o is o_arg

def test_get_current_round_args_missing():
    state = {
        "current_round": 2,
        "arguments": []
    }
    p, o = _get_current_round_args(state)
    assert p is None
    assert o is None
