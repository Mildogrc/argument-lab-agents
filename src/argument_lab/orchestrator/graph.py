"""
argument_lab/core/graph.py

Builds and compiles the LangGraph debate workflow.

Agent nodes are constructed via factories (make_proponent_node,
make_opponent_node) that close over a shared Retriever instance.
Pass a configured Retriever to build_graph() at startup.
"""

from langgraph.graph import StateGraph, START, END

from argument_lab.core.agents import make_proponent_node, make_opponent_node
from argument_lab.core.retriever import Retriever
from argument_lab.core.state import DebateState


def start_round(state: DebateState) -> dict:
    return {}


def judge_node(state: DebateState) -> dict:
    return {}


def hallucination_check(state: DebateState) -> dict:
    return {}


def contradiction_check(state: DebateState) -> dict:
    return {}


def graph_update(state: DebateState) -> dict:
    return {}


def route_round(state: DebateState) -> str:
    if state.get("status") in ["converged", "stalemate", "terminated"]:
        return END
    if state.get("current_round", 1) > 3:
        return END
    return "start_round"


def build_graph(retriever: Retriever):
    """
    Compile the debate workflow. Call once at application startup and
    reuse the compiled graph across debate sessions.

    Usage:
        retriever = Retriever(index=your_faiss_index)
        debate_graph = build_graph(retriever)
        result = debate_graph.invoke({
            "proposition": "...",
            "current_round": 1,
            "arguments": [],
            "claims_registry": {},
            "addressed_claims": set(),
            "ignored_claims": set(),
            "agent_positions": {},
            "repetition_flags": [],
            "status": "in_progress",
            "hallucination_flags": [],
            "contradiction_flags": [],
            "scores": [],
        })
    """
    workflow = StateGraph(DebateState)

    # --- Node registration ---
    workflow.add_node("start_round", start_round)
    workflow.add_node("proponent", make_proponent_node(retriever))
    workflow.add_node("opponent", make_opponent_node(retriever))
    workflow.add_node("start_evaluation", lambda state: {})
    workflow.add_node("judge", judge_node)
    workflow.add_node("hallucination_check", hallucination_check)
    workflow.add_node("contradiction_check", contradiction_check)
    workflow.add_node("graph_update", graph_update)

    # --- Edge wiring ---

    # Entry
    workflow.add_edge(START, "start_round")

    # Fan-out: both agents run in parallel each round
    workflow.add_edge("start_round", "proponent")
    workflow.add_edge("start_round", "opponent")

    # Fan-in: both agents must complete before evaluation starts
    workflow.add_edge("proponent", "start_evaluation")
    workflow.add_edge("opponent", "start_evaluation")

    # Fan-out: judge, hallucination, and contradiction run in parallel
    workflow.add_edge("start_evaluation", "judge")
    workflow.add_edge("start_evaluation", "hallucination_check")
    workflow.add_edge("start_evaluation", "contradiction_check")

    # Fan-in: all evaluation nodes complete before graph_update
    workflow.add_edge("judge", "graph_update")
    workflow.add_edge("hallucination_check", "graph_update")
    workflow.add_edge("contradiction_check", "graph_update")

    # Conditional routing: continue or terminate
    workflow.add_conditional_edges("graph_update", route_round)

    return workflow.compile()