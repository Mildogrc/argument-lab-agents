"""
argument_lab/core/prompts.py

All prompt strings live here, outside the node logic. Keeping them
separate makes it easy to iterate on phrasing without touching orchestration
code, and makes prompt versioning straightforward.

Templates use Python str.format_map() so they're readable without a
third-party templating library.
"""

# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def format_argument(arg) -> str:
    """Render a prior Argument object as a readable block for debate history."""
    evidence_lines = "\n".join(
        f"    [{e.source_id}] \"{e.excerpt}\" (reliability: {e.reliability_score:.2f})"
        for e in arg.evidence
    )
    addressed = ", ".join(arg.counterpoints_addressed) if arg.counterpoints_addressed else "none"
    return (
        f"[{arg.agent.upper()} — Round {arg.round} — claim_id: {arg.id}]\n"
        f"Claim: {arg.claim}\n"
        f"Evidence:\n{evidence_lines}\n"
        f"Assumptions: {', '.join(arg.assumptions) or 'none'}\n"
        f"Counterpoints addressed: {addressed}\n"
        f"Confidence: {arg.confidence_score:.2f}"
    )


def format_debate_history(arguments: list) -> str:
    if not arguments:
        return "No prior arguments."
    return "\n\n".join(format_argument(a) for a in arguments)


def format_evidence_context(chunks: list) -> str:
    """Render retrieved RAG chunks for injection into the generation prompt."""
    if not chunks:
        return "No evidence retrieved."
    return "\n".join(
        f"[{c.source_id}] (similarity: {c.score:.2f})\n\"{c.excerpt}\""
        for c in chunks
    )


# ---------------------------------------------------------------------------
# Query formulation prompts
# Lightweight prompt used in Step 1 to get search queries from the LLM
# before the main argument generation call.
# ---------------------------------------------------------------------------

QUERY_FORMULATION_SYSTEM = """\
You are a research assistant for a structured debate. Your only job is to \
formulate precise search queries that will retrieve the most relevant evidence \
for the debater's next argument.

Return a JSON object with a single key "queries" containing a list of 1-3 \
short, specific search queries (each under 12 words). Do not explain. \
Do not argue. Only return the JSON.
"""

QUERY_FORMULATION_USER = """\
Proposition under debate: {proposition}

The debater you are helping argues: {stance}

Debate history so far:
{history}

Round {current_round} goal: {round_goal}

Formulate search queries to find evidence for this debater's next argument.
"""


# ---------------------------------------------------------------------------
# Agent generation prompts
# Used in Step 2 after evidence has been retrieved and injected.
# ---------------------------------------------------------------------------

AGENT_SYSTEM_TEMPLATE = """\
You are the {role} in a structured multi-round debate.

Your position: You argue {stance} the following proposition.
Proposition: "{proposition}"

Rules of engagement:
1. Every claim you make MUST be grounded in the provided evidence. \
Do not assert facts that are not present in the retrieved sources.
2. {counterpoint_rule}
3. Assign a confidence_score between 0.0 and 1.0 reflecting how strongly \
the evidence supports your claim (not how strongly you personally believe it).
4. List any unstated premises your argument depends on in the assumptions field.
5. Your response must conform exactly to the required JSON schema. \
No preamble. No explanation outside the schema.

Retrieved evidence you MAY cite (you must cite at least one):
{evidence_context}
"""

AGENT_USER_TEMPLATE = """\
Debate history:
{history}

Construct your Round {current_round} argument using the required schema. \
Your argument id should be: "{argument_id}"
"""


# ---------------------------------------------------------------------------
# Round-specific rule strings (injected into AGENT_SYSTEM_TEMPLATE)
# ---------------------------------------------------------------------------

COUNTERPOINT_RULES = {
    1: (
        "This is Round 1. No rebuttals are required. Focus on establishing "
        "your strongest top-level case for your position. Leave counterpoints_addressed empty."
    ),
    2: (
        "This is Round 2. You MUST address at least one specific claim from "
        "your opponent's Round 1 argument. Include its claim_id in counterpoints_addressed. "
        "Failing to address a prior claim will be penalized in scoring."
    ),
    3: (
        "This is Round 3. You MUST address at least one claim from your opponent's "
        "prior arguments. You may also update your confidence_score to reflect "
        "any new evidence introduced in Round 2. Summarize your strongest remaining position."
    ),
}

ROUND_GOALS = {
    1: "Establish your strongest top-level case for your position.",
    2: "Rebut your opponent's Round 1 claims with specific evidence.",
    3: "Refine your position based on all prior evidence and finalize your case.",
}