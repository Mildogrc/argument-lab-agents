import pytest
from argument_lab.core.state import union_sets, merge_dicts, max_round, merge_status

def test_union_sets():
    assert union_sets({"a"}, {"b"}) == {"a", "b"}
    assert union_sets(None, {"b"}) == {"b"}
    assert union_sets({"a"}, None) == {"a"}
    assert union_sets(None, None) == set()

def test_merge_dicts():
    assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}  # right hand wins
    assert merge_dicts(None, {"b": 2}) == {"b": 2}

def test_max_round():
    assert max_round(1, 2) == 2
    assert max_round(3, 1) == 3
    assert max_round(None, 2) == 2

def test_merge_status():
    assert merge_status("in_progress", "converged") == "converged"
    assert merge_status("stalemate", "converged") == "stalemate"
    assert merge_status("terminated", "in_progress") == "terminated"
    assert merge_status(None, "converged") == "converged"
