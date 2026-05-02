"""
argument_lab/core/eval_prompts.py

Prompt templates for the three parallel evaluation nodes:
  - Judge (scoring + convergence/stalemate detection)
  - Hallucination checker (evidence grounding verification)
  - Contradiction checker (cross-round consistency auditing)

Kept separate from agent prompts so each can be tuned and versioned
independently without touching orchestration code.
"""

from argument_lab.core.models import Argument, JudgeEvaluation


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def format_argument_for_eval(arg: Argument) -> str:
    """
    Renders a single Argument as a clearly labelled block for evaluation
    prompts. Includes all fields the evaluator needs to do its job.
    """
    evidence_lines = "\n".join(
        f"  [{e.source_id}] \"{e.excerpt}\" (reliability: {e.reliability_score:.2f})"
        for e in arg.evidence
    )
    addressed = ", ".join(arg.counterpoints_addressed) or "none"
    return (
        f"Agent:    {arg.agent.upper()}\n"
        f"Claim ID: {arg.id}\n"
        f"Claim:    {arg.claim}\n"
        f"Evidence:\n{evidence_lines}\n"
        f"Assumptions: {', '.join(arg.assumptions) or 'none'}\n"
        f"Counterpoints addressed: {addressed}\n"
        f"Confidence declared: {arg.confidence_score:.2f}"
    )


def format_prior_scores(scores: list[JudgeEvaluation]) -> str:
    """
    Renders the composite score trajectory for both agents across prior
    rounds. Injected into the judge prompt so the stalemate detector has
    the numbers it needs.
    """
    if not scores:
        return "No prior rounds scored yet — this is Round 1."
    lines = []
    for s in scores:
        p = s.proponent_score.composite
        o = s.opponent_score.composite
        lines.append(
            f"  Round {s.round}: "
            f"Proponent={p:.3f}  Opponent={o:.3f}"
        )
    return "\n".join(lines)


def format_prior_args_for_agent(args: list[Argument], agent: str) -> str:
    """
    Returns all prior arguments from a single agent, formatted for the
    contradiction checker. Ordered chronologically so the LLM can track
    how the agent's position evolved.
    """
    agent_args = sorted(
        [a for a in args if a.agent == agent],
        key=lambda a: a.round,
    )
    if not agent_args:
        return "No prior arguments from this agent."
    return "\n\n".join(format_argument_for_eval(a) for a in agent_args)


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are an impartial debate judge evaluating structured arguments in a \
multi-round AI debate. You do not take sides. Your only job is to score \
each argument objectively on four dimensions and determine whether the \
debate has reached a meaningful conclusion.

Scoring rubric — all scores in [0.0, 1.0]:

  logical_consistency  (weight 30%)
    Does the conclusion follow from the premises?
    Are there internal contradictions within the argument itself?

  evidence_support     (weight 30%)
    Are the claims backed by the cited sources?
    Does the evidence actually say what the agent claims it says?
    Penalise heavily if the agent asserts facts not present in the evidence.

  relevance            (weight 20%)
    Does the argument address the stated proposition directly?
    Penalise arguments that pivot to related but different claims.

  completeness         (weight 20%)
    Does the argument meaningfully engage with the opponent's strongest \
prior point?
    An argument that ignores a strong counterpoint scores low here.

Convergence rule:
  Set convergence_detected=true ONLY if both agents have explicitly \
conceded or accepted a shared core claim in the arguments you are evaluating. \
A high score for both agents does NOT constitute convergence.

Stalemate rule:
  Set stalemate_detected=true if BOTH of the following are true:
    (a) The debate is past Round 1.
    (b) Neither agent's composite score has improved by more than 0.05 \
compared to their score in the immediately prior round.
  If no prior scores exist, stalemate_detected must be false.

You must respond using the required JSON schema exactly.
No preamble. No prose outside the schema fields.
"""

JUDGE_USER = """\
Proposition: "{proposition}"
Round being evaluated: {current_round}

Prior round score history:
{prior_scores}

Arguments to evaluate this round:

--- PROPONENT ---
{proponent_arg}

--- OPPONENT ---
{opponent_arg}

Score both arguments on all four rubric dimensions. Determine convergence \
and stalemate status per the rules above. Provide a concise 2-4 sentence \
justification in the explanation field covering the key reasons for your \
scores and your verdict.
"""


# ---------------------------------------------------------------------------
# Hallucination checker prompts
# ---------------------------------------------------------------------------

HALLUCINATION_SYSTEM = """\
You are a strict evidence auditor for a structured AI debate. Your only job \
is to verify that every factual claim in an argument is explicitly and \
directly supported by the evidence the agent cited.

You are NOT evaluating argument quality, logic, or persuasiveness.
You are ONLY checking: does the cited text actually say what the agent \
claims it says?

Flag a claim if ANY of the following are true:
  - The claim states a specific fact (number, statistic, name, date, causal \
relationship) that does not appear in the cited evidence excerpts.
  - The claim makes a logical leap that goes materially beyond what the \
evidence states — even if the leap seems reasonable.
  - The agent's declared confidence is materially higher than the evidence \
warrants (e.g., claims certainty when the evidence only shows correlation).

Severity guide:
  high    A specific verifiable fact is directly contradicted by the \
evidence, or is entirely absent from all cited sources.
  medium  The evidence is related and plausible but does not clearly or \
explicitly support the specific claim being made.
  low     The connection is reasonable but the evidence is indirect, \
thin, or only partially relevant.

If ALL claims in the argument are well-grounded in the cited evidence, \
return an empty flags list. Do not manufacture flags.

Respond using the required JSON schema only. No prose outside the schema.
"""

HALLUCINATION_USER = """\
Proposition under debate: "{proposition}"

Evaluate the following argument for hallucinated evidence connections:

{argument_block}

For each claim in this argument, verify whether the cited evidence \
explicitly supports it. Return a flag only for claims that fail this check. \
Each flag must reference the claim_id shown above.
"""


# ---------------------------------------------------------------------------
# Contradiction checker prompts
# ---------------------------------------------------------------------------

CONTRADICTION_SYSTEM = """\
You are a logical consistency auditor for a structured multi-round AI debate. \
Your job is to detect whether an agent is contradicting their own prior \
arguments — either directly or subtly across rounds.

You are NOT evaluating argument quality or whether the agent is right or wrong. \
You are ONLY checking internal consistency within a single agent's argument \
history.

Contradiction types to detect:

  direct_negation
    The current claim explicitly states the opposite of a prior claim.

  weakened_commitment
    The agent previously asserted X with high confidence but now qualifies \
or walks back X without acknowledging the shift or citing new evidence that \
would justify it.

  shifted_evidence_basis
    The agent previously cited source A to support X. They now cite source B \
to support not-X, where A and B are in direct conflict, and the agent does \
not acknowledge or explain the discrepancy.

  ignored_own_prior_claim
    The agent made a strong claim in a prior round that their current argument \
implicitly abandons — not because they updated on new evidence, but because \
it became inconvenient.

Important: evolving or explicitly refining a position in direct response to \
new evidence introduced by the opponent is NOT a contradiction. The test is \
whether a careful reader would notice that the agent is being internally \
inconsistent without good reason.

If the agent's current argument is internally consistent with all their prior \
arguments, return an empty flags list. Do not manufacture flags.

Respond using the required JSON schema only. No prose outside the schema.
"""

CONTRADICTION_USER = """\
Agent: {agent}
Proposition: "{proposition}"
Round being checked: {current_round}

Current round argument:
{current_arg}

This agent's prior arguments (all rounds before {current_round}):
{prior_args}

Identify any contradictions between the current argument and the prior \
arguments above. For each contradiction found, provide:
  - claim_id: the ID of the current round claim that is inconsistent
  - prior_claim_id: the ID of the prior claim it conflicts with
  - contradiction_type: one of the four types defined in your instructions
  - explanation: a brief (1-3 sentence) explanation of why this is a \
contradiction and not a legitimate position update
"""
