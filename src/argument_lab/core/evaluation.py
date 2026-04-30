"""
argument_lab/core/evaluation.py

The three evaluation nodes that run in parallel after each agent round:

  judge_node          — Scores both arguments on four rubric dimensions,
                        detects convergence/stalemate, increments current_round.

  hallucination_check — Verifies that cited evidence actually supports each
                        claim; appends failing claim IDs to hallucination_flags.

  contradiction_check — Detects internal inconsistencies within each agent's
                        own argument history; appends offending claim IDs to
                        contradiction_flags.

All three read from state["arguments"] filtered to the current round and
write independent, non-overlapping keys — safe for parallel fan-in.
"""

import os
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from argument_lab.core.models import (
    Argument,
    JudgeEvaluation,
    HallucinationReport,
    ContradictionReport,
)
from argument_lab.core.state import DebateState, MAX_ROUNDS
from argument_lab.core.eval_prompts import (
    JUDGE_SYSTEM,
    JUDGE_USER,
    HALLUCINATION_SYSTEM,
    HALLUCINATION_USER,
    CONTRADICTION_SYSTEM,
    CONTRADICTION_USER,
    format_argument_for_eval,
    format_prior_scores,
    format_prior_args_for_agent,
)


# ---------------------------------------------------------------------------
# LLM setup
#
# Judge uses temperature=0.1 — scoring needs to be near-deterministic but
# not fully frozen so the composite explanation stays coherent.
#
# Hallucination and contradiction checkers use temperature=0.0 — these are
# strict fact-checking tasks where any randomness risks missed flags or
# false positives.
# ---------------------------------------------------------------------------

_judge_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
)

_checker_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _get_current_round_args(
    state: DebateState,
) -> tuple[Argument | None, Argument | None]:
    """
    Returns (proponent_arg, opponent_arg) for the current round.
    Either may be None if the agent hasn't submitted yet — callers must
    guard against this, though in normal graph flow both will be present
    by the time start_evaluation fans out.
    """
    current_round = state["current_round"]
    all_args = state.get("arguments", [])

    proponent_arg = next(
        (a for a in all_args if a.agent == "proponent" and a.round == current_round),
        None,
    )
    opponent_arg = next(
        (a for a in all_args if a.agent == "opponent" and a.round == current_round),
        None,
    )
    return proponent_arg, opponent_arg


# ---------------------------------------------------------------------------
# 1. Judge node
# ---------------------------------------------------------------------------

def judge_node(state: DebateState) -> dict:
    """
    Scores both agents' current-round arguments and determines the next
    debate status.

    State updates returned:
      scores          — appends the new JudgeEvaluation
      status          — "converged" | "stalemate" | "in_progress"
      current_round   — incremented by 1 (via max_round reducer)
    """
    current_round = state["current_round"]
    proposition = state["proposition"]
    prior_scores = state.get("scores", [])

    proponent_arg, opponent_arg = _get_current_round_args(state)

    if proponent_arg is None or opponent_arg is None:
        raise EvaluationError(
            f"Judge node called but current round {current_round} arguments are incomplete. "
            f"Proponent present: {proponent_arg is not None}, "
            f"Opponent present: {opponent_arg is not None}."
        )

    # Build prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM),
        ("user", JUDGE_USER),
    ])

    # JudgeEvaluation minus the `round` field — the LLM doesn't need to
    # infer it; we patch it in after.
    structured_llm = _judge_llm.with_structured_output(JudgeEvaluation)
    chain = prompt | structured_llm

    evaluation: JudgeEvaluation = chain.invoke({
        "proposition": proposition,
        "current_round": current_round,
        "prior_scores": format_prior_scores(prior_scores),
        "proponent_arg": format_argument_for_eval(proponent_arg),
        "opponent_arg": format_argument_for_eval(opponent_arg),
    })

    # Patch round in — the LLM may not have set it correctly
    evaluation = evaluation.model_copy(update={"round": current_round})

    # Derive status from the evaluation result
    if evaluation.convergence_detected:
        new_status = "converged"
    elif evaluation.stalemate_detected:
        new_status = "stalemate"
    elif current_round >= MAX_ROUNDS:
        new_status = "terminated"
    else:
        new_status = "in_progress"

    return {
        "scores": [evaluation],
        "status": new_status,
        # max_round reducer means this only takes effect if it's larger than
        # the current value — safe to write from judge without racing opponents
        "current_round": current_round + 1,
    }


# ---------------------------------------------------------------------------
# 2. Hallucination checker
# ---------------------------------------------------------------------------

def hallucination_check(state: DebateState) -> dict:
    """
    Verifies that each claim in the current round's arguments is explicitly
    supported by the evidence the agent cited.

    Runs independently for each agent and aggregates flags into a single list.

    State updates returned:
      hallucination_flags — list of claim IDs that failed grounding check
    """
    proposition = state["proposition"]
    proponent_arg, opponent_arg = _get_current_round_args(state)

    flagged_ids: list[str] = []

    for arg in filter(None, [proponent_arg, opponent_arg]):
        report = _check_hallucinations_for_arg(arg, proposition)
        flagged_ids.extend(flag.claim_id for flag in report.flags)

    return {"hallucination_flags": flagged_ids}


def _check_hallucinations_for_arg(
    arg: Argument,
    proposition: str,
    llm: Any = _checker_llm,
) -> HallucinationReport:
    """
    Runs the hallucination check for a single argument. Returns a
    HallucinationReport with zero or more flags.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", HALLUCINATION_SYSTEM),
        ("user", HALLUCINATION_USER),
    ])
    structured_llm = llm.with_structured_output(HallucinationReport)
    chain = prompt | structured_llm

    try:
        return chain.invoke({
            "proposition": proposition,
            "argument_block": format_argument_for_eval(arg),
        })
    except Exception as exc:
        raise EvaluationError(
            f"Hallucination check failed for claim {arg.id}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# 3. Contradiction checker
# ---------------------------------------------------------------------------

def contradiction_check(state: DebateState) -> dict:
    """
    Compares each agent's current-round argument against all of their
    prior arguments to detect internal inconsistencies.

    Runs independently for each agent and aggregates flags into a single list.

    State updates returned:
      contradiction_flags — list of claim IDs where a contradiction was found
    """
    proposition = state["proposition"]
    current_round = state["current_round"]
    all_args = state.get("arguments", [])

    proponent_arg, opponent_arg = _get_current_round_args(state)

    flagged_ids: list[str] = []

    for arg in filter(None, [proponent_arg, opponent_arg]):
        # Prior args = all args from the same agent in earlier rounds
        prior_args = [
            a for a in all_args
            if a.agent == arg.agent and a.round < current_round
        ]
        # Nothing to compare in Round 1
        if not prior_args:
            continue

        report = _check_contradictions_for_agent(
            current_arg=arg,
            prior_args=prior_args,
            proposition=proposition,
            current_round=current_round,
        )
        flagged_ids.extend(flag.claim_id for flag in report.flags)

    return {"contradiction_flags": flagged_ids}


def _check_contradictions_for_agent(
    current_arg: Argument,
    prior_args: list[Argument],
    proposition: str,
    current_round: int,
    llm: Any = _checker_llm,
) -> ContradictionReport:
    """
    Runs the contradiction check for a single agent's current argument
    against their full prior argument history.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", CONTRADICTION_SYSTEM),
        ("user", CONTRADICTION_USER),
    ])
    structured_llm = llm.with_structured_output(ContradictionReport)
    chain = prompt | structured_llm

    try:
        return chain.invoke({
            "agent": current_arg.agent.upper(),
            "proposition": proposition,
            "current_round": current_round,
            "current_arg": format_argument_for_eval(current_arg),
            "prior_args": format_prior_args_for_agent(prior_args, current_arg.agent),
        })
    except Exception as exc:
        raise EvaluationError(
            f"Contradiction check failed for agent {current_arg.agent}, "
            f"claim {current_arg.id}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class EvaluationError(RuntimeError):
    """
    Raised when an evaluation node cannot complete due to missing state,
    LLM failure, or schema validation errors. Surfaces as a node failure
    in LangGraph and can be caught by a retry policy or the metrics dashboard.
    """
    pass
