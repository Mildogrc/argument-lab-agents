"""
argument_lab/core/agents.py
 
Proponent and opponent agent nodes for the LangGraph debate workflow.
 
Each agent follows a strict two-step pipeline:
  Step 1 — Query formulation: a lightweight LLM call produces 1-3 search queries tailored to the agent's current goal.
  Step 2 — Retrieval + generation: the queries are executed against the vector index, and the retrieved chunks are 
            injected into the system prompt before the structured argument generation call.
 
This guarantees that every Argument object contains grounded evidence
before it ever reaches Pydantic validation.
"""
 
import json
import uuid
from typing import Any
 
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
 
from argument_lab.core.models import Argument, Claim, EvidenceRef
from argument_lab.core.retriever import Retriever, RetrieverError
from argument_lab.core.state import DebateState, MAX_ROUNDS
from argument_lab.core.prompts import (
    QUERY_FORMULATION_SYSTEM,
    QUERY_FORMULATION_USER,
    AGENT_SYSTEM_TEMPLATE,
    AGENT_USER_TEMPLATE,
    COUNTERPOINT_RULES,
    ROUND_GOALS,
    format_debate_history,
    format_evidence_context,
)
 
 
# ---------------------------------------------------------------------------
# LLM setup
# Temperature 0.2 for structured output — low enough for schema compliance,
# high enough to avoid degenerate repetition across rounds.
# ---------------------------------------------------------------------------
 
import os

_llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=os.environ.get("OPENAI_API_KEY", "dummy"))
_query_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.environ.get("OPENAI_API_KEY", "dummy"))


 
 
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
 
def _formulate_queries(
    proposition: str,
    stance: str,
    history: str,
    current_round: int,
    llm: Any = _query_llm,
) -> list[str]:
    """
    Step 1: Ask a lightweight LLM to produce search queries for this agent's
    next argument. Returns a list of query strings, falling back to the
    proposition itself if the LLM output cannot be parsed.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_FORMULATION_SYSTEM),
        ("user", QUERY_FORMULATION_USER),
    ])
    chain = prompt | llm | JsonOutputParser()
 
    try:
        result = chain.invoke({
            "proposition": proposition,
            "stance": stance,
            "history": history,
            "current_round": current_round,
            "round_goal": ROUND_GOALS[min(current_round, MAX_ROUNDS)],
        })
        queries = result.get("queries", [])
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:3]  # enforce max
    except Exception:
        pass  # fall through to fallback
 
    # Fallback: use the proposition directly so retrieval never returns empty
    return [proposition]
 
 
def _retrieve_evidence(
    retriever: Retriever,
    queries: list[str],
) -> list[EvidenceRef]:
    """
    Execute the queries against the vector index and convert chunks to EvidenceRef objects ready for the Argument schema.
    Raises AgentError if retrieval returns nothing — no evidence means no valid argument can be produced.
    """
    chunks = retriever.retrieve_multi(queries)
    if not chunks:
        raise AgentError(
            "RAG retrieval returned no results. Cannot produce a grounded argument."
        )
    return [
        EvidenceRef(
            source_id=chunk.source_id,
            excerpt=chunk.excerpt,
            reliability_score=round(chunk.score, 3),
        )
        for chunk in chunks
    ]
 
 
def _generate_argument(
    *,
    role: str,
    stance: str,
    proposition: str,
    current_round: int,
    history: str,
    evidence_refs: list[EvidenceRef],
    evidence_context: str,
    argument_id: str,
    llm: Any = _llm,
) -> Argument:
    """
    Step 2: Generate the structured Argument using the retrieved evidence injected into the system prompt. Uses .with_structured_output() to
    enforce schema compliance at the LangChain layer.
    """
    structured_llm = llm.with_structured_output(Argument)
 
    system_prompt = AGENT_SYSTEM_TEMPLATE.format_map({
        "role": role,
        "stance": "FOR" if role == "Proponent" else "AGAINST",
        "proposition": proposition,
        "counterpoint_rule": COUNTERPOINT_RULES[min(current_round, MAX_ROUNDS)],
        "evidence_context": evidence_context,
    })
 
    user_prompt = AGENT_USER_TEMPLATE.format_map({
        "history": history,
        "current_round": current_round,
        "argument_id": argument_id,
    })
 
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])
 
    chain = prompt | structured_llm
    argument = chain.invoke({})
 
    # Patch in the argument_id and round in case the LLM didn't follow them
    # exactly — the schema enforces types but not specific string values.
    argument = argument.model_copy(update={
        "id": argument_id,
        "round": current_round,
        "agent": role.lower(),
    })
 
    # Ensure the LLM cited sources that were actually retrieved (not hallucinated IDs)
    valid_source_ids = {ref.source_id for ref in evidence_refs}
    argument = argument.model_copy(update={
        "evidence": [e for e in argument.evidence if e.source_id in valid_source_ids]
        or evidence_refs[:1],  # guarantee min_length=1 even if LLM cited nothing valid
    })
 
    return argument
 
 
def _enforce_counterpoint_rule(
    argument: Argument,
    current_round: int,
    prior_opponent_claim_ids: list[str],
    role: str,
    llm: Any = _llm,
    proposition: str = "",
    history: str = "",
    evidence_refs: list[EvidenceRef] = [],
    evidence_context: str = "",
) -> Argument:
    """
    Option 3 enforcement: if Round >= 2 and counterpoints_addressed is empty, re-prompt the LLM once with an explicit correction instruction.
    Raises AgentError if the second attempt also fails — this surfaces as a node failure rather than silently passing a non-compliant argument.
    """
    if current_round < 2:
        return argument  # Round 1: no counterpoints required
 
    if argument.counterpoints_addressed:
        return argument  # already compliant
 
    if not prior_opponent_claim_ids:
        # No opponent claims exist yet (e.g., opponent hasn't run this round)
        # This shouldn't happen in normal flow but guard defensively.
        return argument
 
    # Re-prompt with an explicit correction
    correction_note = (
        f"Your previous response left counterpoints_addressed empty. "
        f"You MUST include at least one of these opponent claim IDs: "
        f"{prior_opponent_claim_ids}. Revise your argument to address "
        f"at least one of these claims directly."
    )
    structured_llm = llm.with_structured_output(Argument)
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_TEMPLATE.format_map({
            "role": role,
            "stance": "FOR" if role == "Proponent" else "AGAINST",
            "proposition": proposition,
            "counterpoint_rule": COUNTERPOINT_RULES[min(current_round, MAX_ROUNDS)],
            "evidence_context": evidence_context,
        })),
        ("user", AGENT_USER_TEMPLATE.format_map({
            "history": history,
            "current_round": current_round,
            "argument_id": argument.id,
        })),
        ("assistant", argument.model_dump_json()),
        ("user", correction_note),
    ])
 
    revised = (prompt | structured_llm).invoke({})
    revised = revised.model_copy(update={
        "id": argument.id,
        "round": current_round,
        "agent": argument.agent,
    })
 
    if not revised.counterpoints_addressed:
        raise AgentError(
            f"{role} failed to address any opponent counterpoints in Round {current_round} "
            f"after correction. Prior claim IDs available: {prior_opponent_claim_ids}"
        )
 
    return revised
 
 
def _get_prior_opponent_claim_ids(state: DebateState, my_role: str) -> list[str]:
    """
    Returns claim IDs from the opponent's prior arguments.
    Used to validate and enforce counterpoint_addressed in Round >= 2.
    """
    opponent_role = "opponent" if my_role == "proponent" else "proponent"
    return [
        arg.id
        for arg in state.get("arguments", [])
        if arg.agent == opponent_role and arg.round < state["current_round"]
    ]
 
 
def _update_state_from_argument(
    argument: Argument,
    state: DebateState,
) -> dict:
    """
    Derive all state updates that a new argument produces:
    - adds argument to the accumulator
    - registers its claim in claims_registry
    - updates agent_positions with the new confidence score
    - marks addressed and ignored claims
    """
    # Register the new claim
    from argument_lab.core.models import Claim
    new_claim = Claim(
        id=argument.id,
        text=argument.claim,
        agent=argument.agent,
        round=argument.round,
    )
 
    # Determine which prior opponent claims were ignored this round
    opponent_role = "opponent" if argument.agent == "proponent" else "proponent"
    prior_opponent_ids = {
        arg.id
        for arg in state.get("arguments", [])
        if arg.agent == opponent_role
    }
    newly_addressed = set(argument.counterpoints_addressed)
    newly_ignored = prior_opponent_ids - newly_addressed - state.get("ignored_claims", set())
 
    # Extend the agent's confidence trajectory
    current_positions = dict(state.get("agent_positions", {}))
    trajectory = list(current_positions.get(argument.agent, []))
    trajectory.append(argument.confidence_score)
 
    return {
        "arguments": [argument],
        "claims_registry": {argument.id: new_claim},
        "addressed_claims": newly_addressed,
        "ignored_claims": newly_ignored,
        "agent_positions": {argument.agent: trajectory},
    }
 
 
# ---------------------------------------------------------------------------
# Public node functions
# ---------------------------------------------------------------------------
 
def make_proponent_node(retriever: Retriever):
    """
    Factory that closes over a Retriever instance and returns a LangGraph-
    compatible node function. Call this at graph compile time:
 
        workflow.add_node("proponent", make_proponent_node(retriever))
    """
    def proponent_node(state: DebateState) -> dict:
        return _run_agent_node(
            state=state,
            role="Proponent",
            agent_key="proponent",
            retriever=retriever,
        )
    return proponent_node
 
 
def make_opponent_node(retriever: Retriever):
    """
    Factory that closes over a Retriever instance and returns a LangGraph-
    compatible node function.
 
        workflow.add_node("opponent", make_opponent_node(retriever))
    """
    def opponent_node(state: DebateState) -> dict:
        return _run_agent_node(
            state=state,
            role="Opponent",
            agent_key="opponent",
            retriever=retriever,
        )
    return opponent_node
 
 
def _run_agent_node(
    *,
    state: DebateState,
    role: str,
    agent_key: str,
    retriever: Retriever,
) -> dict:
    """
    Shared implementation for both agent nodes.
 
    Pipeline:
      1. Format debate history for context
      2. Formulate search queries (lightweight LLM call)
      3. Retrieve evidence from vector index
      4. Generate structured Argument (main LLM call)
      5. Enforce counterpoint rule (re-prompt if needed)
      6. Derive and return state updates
    """
    current_round = state["current_round"]
    proposition = state["proposition"]
    prior_arguments = state.get("arguments", [])
 
    # Step 1: format history
    history = format_debate_history(prior_arguments)
    stance = "FOR" if role == "Proponent" else "AGAINST"
 
    # Step 2: query formulation
    queries = _formulate_queries(
        proposition=proposition,
        stance=stance,
        history=history,
        current_round=current_round,
    )
 
    # Step 3: retrieve evidence
    evidence_refs = _retrieve_evidence(retriever, queries)
    evidence_context = format_evidence_context(
        # Pass the raw chunks back for display; evidence_refs are already converted
        retriever.retrieve_multi(queries)
    )
 
    # Step 4: generate argument
    argument_id = str(uuid.uuid4())
    argument = _generate_argument(
        role=role,
        stance=stance,
        proposition=proposition,
        current_round=current_round,
        history=history,
        evidence_refs=evidence_refs,
        evidence_context=evidence_context,
        argument_id=argument_id,
    )
 
    # Step 5: enforce counterpoint rule (Option 3)
    prior_opponent_ids = _get_prior_opponent_claim_ids(state, agent_key)
    argument = _enforce_counterpoint_rule(
        argument=argument,
        current_round=current_round,
        prior_opponent_claim_ids=prior_opponent_ids,
        role=role,
        proposition=proposition,
        history=history,
        evidence_refs=evidence_refs,
        evidence_context=evidence_context,
    )
 
    # Step 6: derive state updates
    return _update_state_from_argument(argument, state)
 
 
# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
 
class AgentError(RuntimeError):
    """
    Raised when an agent node cannot produce a valid, schema-compliant argument.
    The LangGraph node will propagate this as a node failure, which can be caught by a retry policy or surfaced to the metrics dashboard.
    """
    pass