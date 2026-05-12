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

from argument_lab.core.models import (
    Argument,
    JudgeEvaluation,
    HallucinationReport,
    ContradictionReport,
    HallucinationFlag,
    ContradictionFlag,
    Verdict,
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
    VERDICT_SYSTEM,
    VERDICT_USER,
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

_judge_llm: Any | None = None
_checker_llm: Any | None = None


def _make_chat_openai(*, model: str, temperature: float) -> Any:
    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:
        raise EvaluationError(
            "langchain_openai is required for LLM-backed evaluation execution. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
    )


def _get_judge_llm() -> Any:
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = _make_chat_openai(model="gpt-4o", temperature=0.1)
    return _judge_llm


def _get_checker_llm() -> Any:
    global _checker_llm
    if _checker_llm is None:
        _checker_llm = _make_chat_openai(model="gpt-4o", temperature=0.0)
    return _checker_llm


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


def _apply_penalties(
    score: JudgeEvaluation,
    hallucination_flags: list[HallucinationFlag],
    contradiction_flags: list[ContradictionFlag],
) -> JudgeEvaluation:
    """
    Applies programmatic penalties to agent scores based on detected flags.
    - Hallucinations: 0.1 per low, 0.2 per medium, 0.3 per high severity.
    - Contradictions: 0.2 flat penalty per contradiction.
    Penalties are capped at 0.5 total deduction per agent per round.
    """
    return score


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
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JUDGE_SYSTEM),
            ("user", JUDGE_USER),
        ]
    )

    # JudgeEvaluation minus the `round` field — the LLM doesn't need to
    # infer it; we patch it in after.
    structured_llm = _get_judge_llm().with_structured_output(JudgeEvaluation)
    chain = prompt | structured_llm

    evaluation: JudgeEvaluation = chain.invoke(
        {
            "proposition": proposition,
            "current_round": current_round,
            "prior_scores": format_prior_scores(prior_scores),
            "proponent_arg": format_argument_for_eval(proponent_arg),
            "opponent_arg": format_argument_for_eval(opponent_arg),
        }
    )

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

    # Apply penalties before returning
    hallucination_flags = state.get("hallucination_flags", [])
    contradiction_flags = state.get("contradiction_flags", [])

    # Penalties for proponent
    p_h = [f for f in hallucination_flags if f.claim_id == proponent_arg.id]
    p_c = [f for f in contradiction_flags if f.claim_id == proponent_arg.id]

    p_h_penalty = sum(
        0.1 if f.severity == "low" else 0.2 if f.severity == "medium" else 0.3
        for f in p_h
    )
    p_c_penalty = len(p_c) * 0.2

    evaluation.proponent_score.hallucination_penalty = p_h_penalty
    evaluation.proponent_score.contradiction_penalty = p_c_penalty

    # Penalties for opponent
    o_h = [f for f in hallucination_flags if f.claim_id == opponent_arg.id]
    o_c = [f for f in contradiction_flags if f.claim_id == opponent_arg.id]

    o_h_penalty = sum(
        0.1 if f.severity == "low" else 0.2 if f.severity == "medium" else 0.3
        for f in o_h
    )
    o_c_penalty = len(o_c) * 0.2

    evaluation.opponent_score.hallucination_penalty = o_h_penalty
    evaluation.opponent_score.contradiction_penalty = o_c_penalty

    return {
        "scores": [evaluation],
        "status": new_status,
        "current_round": current_round + 1,
    }


# ---------------------------------------------------------------------------
# 4. Verdict generator
# ---------------------------------------------------------------------------


def verdict_generator(state: DebateState) -> dict:
    """
    Terminal node that synthesises the entire debate history into a final
    verdict object. Called only when status is converged, stalemate, or terminated.
    """
    proposition = state["proposition"]
    arguments = state.get("arguments", [])
    scores = state.get("scores", [])

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", VERDICT_SYSTEM),
            ("user", VERDICT_USER),
        ]
    )

    structured_llm = _get_judge_llm().with_structured_output(Verdict)
    chain = prompt | structured_llm

    history = "\n".join(
        [
            f"Round {arg.round} {arg.agent.upper()}: {arg.claim}"
            for arg in sorted(arguments, key=lambda a: (a.round, a.agent))
        ]
    )

    score_history = "\n".join(
        [
            f"Round {s.round}: PROP {s.proponent_score.composite:.3f} | OPP {s.opponent_score.composite:.3f}"
            for s in sorted(scores, key=lambda s: s.round)
        ]
    )

    verdict = chain.invoke(
        {
            "proposition": proposition,
            "history": history,
            "scores": score_history,
        }
    )

    return {"verdict": verdict}


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
        flagged_ids.extend(report.flags)

    return {"hallucination_flags": flagged_ids}


def _check_hallucinations_for_arg(
    arg: Argument,
    proposition: str,
    llm: Any | None = None,
) -> HallucinationReport:
    """
    Runs the hallucination check for a single argument. Returns a
    HallucinationReport with zero or more flags.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", HALLUCINATION_SYSTEM),
            ("user", HALLUCINATION_USER),
        ]
    )
    llm = llm or _get_checker_llm()
    structured_llm = llm.with_structured_output(HallucinationReport)
    chain = prompt | structured_llm

    try:
        return chain.invoke(
            {
                "proposition": proposition,
                "argument_block": format_argument_for_eval(arg),
            }
        )
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
            a for a in all_args if a.agent == arg.agent and a.round < current_round
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
        flagged_ids.extend(report.flags)

    return {"contradiction_flags": flagged_ids}


def _check_contradictions_for_agent(
    current_arg: Argument,
    prior_args: list[Argument],
    proposition: str,
    current_round: int,
    llm: Any | None = None,
) -> ContradictionReport:
    """
    Runs the contradiction check for a single agent's current argument
    against their full prior argument history.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTRADICTION_SYSTEM),
            ("user", CONTRADICTION_USER),
        ]
    )
    llm = llm or _get_checker_llm()
    structured_llm = llm.with_structured_output(ContradictionReport)
    chain = prompt | structured_llm

    try:
        return chain.invoke(
            {
                "agent": current_arg.agent.upper(),
                "proposition": proposition,
                "current_round": current_round,
                "current_arg": format_argument_for_eval(current_arg),
                "prior_args": format_prior_args_for_agent(
                    prior_args, current_arg.agent
                ),
            }
        )
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
