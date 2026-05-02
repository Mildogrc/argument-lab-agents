"""
argument_lab/orchestrator/graph.py

Builds and compiles the LangGraph debate workflow.

Agent nodes are constructed via factories (make_proponent_node,
make_opponent_node) that close over a shared Retriever instance.
Evaluation nodes (judge, hallucination_check, contradiction_check) are
imported directly from argument_lab.core.evaluation.

Pass a configured Retriever to build_graph() at startup — it is the only
external dependency required to compile the graph.
"""

from langgraph.graph import StateGraph, START, END

from argument_lab.core.agents import make_proponent_node, make_opponent_node
from argument_lab.core.evaluation import judge_node, hallucination_check, contradiction_check
from argument_lab.core.retriever import Retriever
from argument_lab.core.state import DebateState, MAX_ROUNDS


# ---------------------------------------------------------------------------
# Non-LLM nodes
# ---------------------------------------------------------------------------

def start_round(state: DebateState) -> dict:
    """
    Passthrough node that acts as the fan-out point at the start of each
    round. Returns the current round to satisfy LangGraph's update requirement.
    """
    return {"current_round": state.get("current_round", 1)}


def graph_update(state: DebateState) -> dict:
    """
    Fan-in point after all three evaluation nodes complete.
    Returns the current round to satisfy LangGraph's update requirement.
    """
    return {"current_round": state.get("current_round", 1)}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_round(state: DebateState) -> str:
    """
    Decides whether to loop back for another round or terminate.

    Termination conditions (in priority order):
      1. Judge detected convergence or stalemate → status already set
      2. current_round has been incremented past MAX_ROUNDS by judge_node
      3. Status was explicitly set to "terminated" by any node

    Note: current_round is incremented by judge_node at the END of each
    round. So after Round 3 completes, current_round becomes 4, which
    triggers the > MAX_ROUNDS guard here.
    """
    status = state.get("status", "in_progress")
    if status in ("converged", "stalemate", "terminated"):
        return END
    if state.get("current_round", 1) > MAX_ROUNDS:
        return END
    return "start_round"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(retriever: Retriever):
    """
    Compile the debate workflow. Call once at application startup and
    reuse the compiled graph across all debate sessions.

    Args:
        retriever: A configured Retriever instance wrapping a FAISS,
                   ChromaDB, or OpenSearch index.

    Returns:
        A compiled LangGraph CompiledStateGraph ready to invoke.

    Usage:
        retriever = Retriever(index=your_faiss_index)
        debate_graph = build_graph(retriever)

        final_state = debate_graph.invoke({
            "proposition": "Companies should replace legacy infrastructure with AI-driven systems.",
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

    # Passthrough fan-in/fan-out between agent round and evaluation round
    workflow.add_node("start_evaluation", lambda state: {"current_round": state.get("current_round", 1)})

    # Evaluation nodes — all three run in parallel
    workflow.add_node("judge", judge_node)
    workflow.add_node("hallucination_check", hallucination_check)
    workflow.add_node("contradiction_check", contradiction_check)

    # Final fan-in before routing decision
    workflow.add_node("graph_update", graph_update)

    # --- Edge wiring ---

    # Entry point
    workflow.add_edge(START, "start_round")

    # Fan-out: both agents run in parallel each round
    workflow.add_edge("start_round", "proponent")
    workflow.add_edge("start_round", "opponent")

    # Fan-in: both agents must complete before any evaluation runs
    workflow.add_edge("proponent", "start_evaluation")
    workflow.add_edge("opponent", "start_evaluation")

    # Fan-out: judge, hallucination check, and contradiction check run in parallel
    workflow.add_edge("start_evaluation", "judge")
    workflow.add_edge("start_evaluation", "hallucination_check")
    workflow.add_edge("start_evaluation", "contradiction_check")

    # Fan-in: all three evaluation nodes must complete before graph_update
    workflow.add_edge("judge", "graph_update")
    workflow.add_edge("hallucination_check", "graph_update")
    workflow.add_edge("contradiction_check", "graph_update")

    # Conditional routing: loop or terminate
    workflow.add_conditional_edges("graph_update", route_round)

    return workflow.compile()
