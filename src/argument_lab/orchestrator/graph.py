from langgraph.graph import StateGraph, START, END
from argument_lab.core.state import DebateState

def start_round(state: DebateState):
    return state

def proponent_node(state: DebateState):
    return {"arguments": []}

def opponent_node(state: DebateState):
    return {"arguments": []}

def judge_node(state: DebateState):
    return state

def hallucination_check(state: DebateState):
    return state

def contradiction_check(state: DebateState):
    return state

def graph_update(state: DebateState):
    return state

def route_round(state: DebateState):
    if state.get("status") in ["converged", "stalemate", "terminated"]:
        return END
    return "start_round"

def build_graph():
    workflow = StateGraph(DebateState)
    
    workflow.add_node("start_round", start_round)
    workflow.add_node("proponent", proponent_node)
    workflow.add_node("opponent", opponent_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("hallucination_check", hallucination_check)
    workflow.add_node("contradiction_check", contradiction_check)
    workflow.add_node("graph_update", graph_update)
    
    workflow.add_node("start_evaluation", lambda state: state)
    
    workflow.add_edge(START, "start_round")
    
    workflow.add_edge("start_round", "proponent")
    workflow.add_edge("start_round", "opponent")
    
    workflow.add_edge("proponent", "start_evaluation")
    workflow.add_edge("opponent", "start_evaluation")
    
    workflow.add_edge("start_evaluation", "judge")
    workflow.add_edge("start_evaluation", "hallucination_check")
    workflow.add_edge("start_evaluation", "contradiction_check")
    
    workflow.add_edge("judge", "graph_update")
    workflow.add_edge("hallucination_check", "graph_update")
    workflow.add_edge("contradiction_check", "graph_update")
    
    workflow.add_conditional_edges("graph_update", route_round)
    
    return workflow.compile()
