#!/usr/bin/env python3
"""
scripts/run_debate.py

Entry point for running a full ArgumentLab debate from the command line.

Streams each node's output as it arrives via graph.stream(), printing
arguments and judge scores round-by-round. At the end, prints a structured
summary table and exports the full results to JSON + Markdown.

Usage:
    python setup/debate.py \
        --proposition "Companies should replace legacy infrastructure with AI-driven systems." \
        --session-id my_debate_001

    # Optional flags:
    --index-path  local_data/faiss_index   (default)
    --output-dir  local_data/results        (default)
    --top-k       4                          (chunks retrieved per query)

Prerequisites:
    1. export OPENAI_API_KEY=sk-...
    2. python setup/ingest_corpus.py --sample
"""

import argparse
import os
import sys
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without pip install
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from argument_lab.core.faiss_index import FaissIndex
from argument_lab.core.retriever import Retriever
from argument_lab.core.state import DebateState, MAX_ROUNDS
from argument_lab.core.exporter import export_debate
from argument_lab.orchestrator.graph import build_graph


# ---------------------------------------------------------------------------
# ANSI colour helpers (degrade gracefully on Windows without colorama)
# ---------------------------------------------------------------------------

def _supports_colour() -> bool:
    return sys.stdout.isatty() and os.name != "nt"


RESET  = "\033[0m"  if _supports_colour() else ""
BOLD   = "\033[1m"  if _supports_colour() else ""
DIM    = "\033[2m"  if _supports_colour() else ""
CYAN   = "\033[36m" if _supports_colour() else ""
GREEN  = "\033[32m" if _supports_colour() else ""
YELLOW = "\033[33m" if _supports_colour() else ""
RED    = "\033[31m" if _supports_colour() else ""
BLUE   = "\033[34m" if _supports_colour() else ""


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Streaming printer — handles each node update as it arrives
# ---------------------------------------------------------------------------

def _print_agent_update(node_name: str, update: dict) -> None:
    """
    Called when a proponent or opponent node emits a state update.
    Prints the new argument in a readable format.
    """
    new_args = update.get("arguments", [])
    if not new_args:
        return

    arg = new_args[-1]  # the argument just produced
    role_colour = CYAN if arg.agent == "proponent" else YELLOW
    label = f"{role_colour}{BOLD}{arg.agent.upper()}{RESET}"

    print(f"\n  {label} — Round {arg.round}")
    print(f"  {DIM}{_hr('·', 68)}{RESET}")

    # Wrap claim text for readability
    claim_lines = textwrap.wrap(arg.claim, width=64)
    print(f"  {BOLD}Claim{RESET} (confidence: {arg.confidence_score:.2f})")
    for line in claim_lines:
        print(f"    {line}")

    print(f"  {BOLD}Evidence{RESET}")
    for e in arg.evidence:
        print(f"    [{e.source_id}] (reliability: {e.reliability_score:.2f})")
        excerpt_lines = textwrap.wrap(e.excerpt, width=60)
        for line in excerpt_lines:
            print(f"      {DIM}{line}{RESET}")

    if arg.counterpoints_addressed:
        ids = ", ".join(arg.counterpoints_addressed)
        print(f"  {BOLD}Addresses:{RESET} {DIM}{ids}{RESET}")

    if arg.assumptions:
        print(f"  {BOLD}Assumptions:{RESET}")
        for a in arg.assumptions:
            print(f"    • {a}")


def _print_judge_update(update: dict) -> None:
    """
    Called when the judge node emits a state update.
    Prints the score table for the round just evaluated.
    """
    scores = update.get("scores", [])
    if not scores:
        return

    score = scores[-1]
    p = score.proponent_score
    o = score.opponent_score

    verdict = ""
    if score.convergence_detected:
        verdict = f"  {GREEN}{BOLD}✅ CONVERGENCE DETECTED{RESET}"
    elif score.stalemate_detected:
        verdict = f"  {RED}{BOLD}⚖️  STALEMATE DETECTED{RESET}"

    print(f"\n  {BLUE}{BOLD}JUDGE — Round {score.round}{RESET}")
    print(f"  {DIM}{_hr('·', 68)}{RESET}")
    print(f"  {'Dimension':<24} {'Proponent':>10} {'Opponent':>10}")
    print(f"  {DIM}{_hr('·', 46)}{RESET}")

    dims = [
        ("Logical Consistency", p.logical_consistency, o.logical_consistency),
        ("Evidence Support",    p.evidence_support,    o.evidence_support),
        ("Relevance",           p.relevance,           o.relevance),
        ("Completeness",        p.completeness,        o.completeness),
    ]
    for name, pv, ov in dims:
        print(f"  {name:<24} {pv:>10.2f} {ov:>10.2f}")

    print(f"  {DIM}{_hr('·', 46)}{RESET}")
    print(f"  {'Composite (weighted)':<24} {p.composite:>10.3f} {o.composite:>10.3f}")

    if verdict:
        print(verdict)

    print(f"\n  {BOLD}Judge's note:{RESET}")
    for line in textwrap.wrap(score.explanation, width=64):
        print(f"    {line}")


def _print_hallucination_update(update: dict) -> None:
    flags = update.get("hallucination_flags", [])
    if flags:
        print(f"\n  {RED}⚠  Hallucination flags:{RESET} {', '.join(flags)}")


def _print_contradiction_update(update: dict) -> None:
    flags = update.get("contradiction_flags", [])
    if flags:
        print(f"\n  {RED}⚠  Contradiction flags:{RESET} {', '.join(flags)}")


# ---------------------------------------------------------------------------
# Summary table (printed at the end of all rounds)
# ---------------------------------------------------------------------------

def _print_summary(final_state: DebateState) -> None:
    scores = final_state.get("scores", [])
    status = final_state.get("status", "unknown")

    status_display = {
        "converged":   f"{GREEN}✅ Converged{RESET}",
        "stalemate":   f"{YELLOW}⚖️  Stalemate{RESET}",
        "terminated":  f"{BLUE}🏁 Terminated (max rounds){RESET}",
        "in_progress": f"{DIM}⏳ Still in progress{RESET}",
    }.get(status, status)

    print(f"\n{BOLD}{_hr('═')}{RESET}")
    print(f"{BOLD}  DEBATE SUMMARY{RESET}")
    print(f"{BOLD}{_hr('═')}{RESET}\n")
    print(f"  Status:  {status_display}")
    print(f"  Rounds:  {len(scores)} / {MAX_ROUNDS} completed\n")

    if scores:
        print(f"  {'Round':<8} {'Proponent':>12} {'Opponent':>12}  {'Verdict'}")
        print(f"  {DIM}{_hr('·', 52)}{RESET}")
        for s in sorted(scores, key=lambda x: x.round):
            verdict = ""
            if s.convergence_detected:
                verdict = f"{GREEN}Converged{RESET}"
            elif s.stalemate_detected:
                verdict = f"{YELLOW}Stalemate{RESET}"
            print(
                f"  {s.round:<8} "
                f"{s.proponent_score.composite:>12.3f} "
                f"{s.opponent_score.composite:>12.3f}  "
                f"{verdict}"
            )

    h_count = len(final_state.get("hallucination_flags", []))
    c_count = len(final_state.get("contradiction_flags", []))
    i_count = len(final_state.get("ignored_claims", []))

    print(f"\n  {BOLD}Evaluation flags{RESET}")
    print(f"  {'Hallucinations:':<22} {h_count}")
    print(f"  {'Contradictions:':<22} {c_count}")
    print(f"  {'Ignored claims:':<22} {i_count}")

    # Agent confidence drift
    print(f"\n  {BOLD}Confidence trajectories{RESET}")
    for agent, positions in final_state.get("agent_positions", {}).items():
        trajectory = " → ".join(f"{v:.2f}" for v in positions)
        print(f"  {agent.capitalize():<14} {trajectory}")

    print(f"\n{BOLD}{_hr('═')}{RESET}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an ArgumentLab structured debate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python setup/debate.py \
            --proposition "Companies should replace legacy infrastructure with AI."

          python setup/debate.py \
            --proposition "Remote work improves engineering productivity." \
            --session-id remote_work_001 \
            --top-k 6
        """),
    )
    parser.add_argument(
        "--proposition",
        required=True,
        help="The debate proposition. Agents will argue FOR and AGAINST this.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Unique session identifier for output filenames. "
             "Defaults to a timestamp-based ID.",
    )
    parser.add_argument(
        "--index-path",
        default="local_data/faiss_index",
        help="Path to the FAISS index directory (default: local_data/faiss_index)",
    )
    parser.add_argument(
        "--output-dir",
        default="local_data/results",
        help="Directory for JSON + Markdown output (default: local_data/results)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of evidence chunks to retrieve per query (default: 4)",
    )
    args = parser.parse_args()

    # ── Pre-flight checks ──────────────────────────────────────────────────
    if not os.environ.get("OPENAI_API_KEY"):
        print(f"{RED}Error: OPENAI_API_KEY is not set.{RESET}")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    session_id = args.session_id or (
        "debate_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )

    # ── Header ────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{_hr('═')}{RESET}")
    print(f"{BOLD}  ARGUMENTLAB{RESET}")
    print(f"{BOLD}{_hr('═')}{RESET}")
    print(f"\n  {BOLD}Proposition:{RESET}")
    for line in textwrap.wrap(args.proposition, width=64):
        print(f"    {line}")
    print(f"\n  {BOLD}Session:{RESET}  {session_id}")
    print(f"  {BOLD}Rounds:{RESET}   {MAX_ROUNDS}")
    print(f"  {BOLD}Top-k:{RESET}    {args.top_k}\n")
    print(f"{BOLD}{_hr('═')}{RESET}\n")

    # ── Load index ─────────────────────────────────────────────────────────
    print(f"  Loading FAISS index from {args.index_path}...")
    try:
        faiss_index = FaissIndex.load(args.index_path)
    except FileNotFoundError as e:
        print(f"\n{RED}Error:{RESET} {e}")
        sys.exit(1)

    retriever = Retriever(index=faiss_index, top_k=args.top_k)

    # ── Build graph ────────────────────────────────────────────────────────
    print(f"  Building debate graph...\n")
    debate_graph = build_graph(retriever)

    # ── Initial state ──────────────────────────────────────────────────────
    initial_state: DebateState = {
        "proposition":      args.proposition,
        "current_round":    1,
        "arguments":        [],
        "claims_registry":  {},
        "addressed_claims": set(),
        "ignored_claims":   set(),
        "agent_positions":  {},
        "repetition_flags": [],
        "status":           "in_progress",
        "hallucination_flags": [],
        "contradiction_flags": [],
        "scores":           [],
    }

    # ── Stream the debate ──────────────────────────────────────────────────
    current_round_printed = 0
    final_state: DebateState | None = None

    for node_name, update in debate_graph.stream(initial_state):
        # Print round header when we first see a new round's arguments
        new_args = update.get("arguments", [])
        if new_args:
            round_num = new_args[-1].round
            if round_num != current_round_printed:
                current_round_printed = round_num
                print(f"\n{BOLD}{_hr()}{RESET}")
                print(f"{BOLD}  ROUND {round_num}{RESET}")
                print(f"{BOLD}{_hr()}{RESET}")

        # Dispatch to per-node printers
        if node_name in ("proponent", "opponent"):
            _print_agent_update(node_name, update)
        elif node_name == "judge":
            _print_judge_update(update)
        elif node_name == "hallucination_check":
            _print_hallucination_update(update)
        elif node_name == "contradiction_check":
            _print_contradiction_update(update)

        # Accumulate the last known full state
        # LangGraph's stream() yields (node_name, state_delta) tuples;
        # the final full state is available via invoke() but we reconstruct
        # it from the last graph_update node output which holds the full state.
        if node_name == "graph_update":
            final_state = update  # graph_update passthrough holds full state

    # Fallback: run invoke() to guarantee we have the final state
    if final_state is None or "proposition" not in final_state:
        print(f"\n  {DIM}Retrieving final state...{RESET}")
        final_state = debate_graph.invoke(initial_state)

    # ── Summary ────────────────────────────────────────────────────────────
    _print_summary(final_state)

    # ── Export ─────────────────────────────────────────────────────────────
    print(f"  Exporting results...")
    json_path, md_path = export_debate(
        state=final_state,
        session_id=session_id,
        output_dir=args.output_dir,
    )
    print(f"\n  {GREEN}✓{RESET} JSON:     {json_path}")
    print(f"  {GREEN}✓{RESET} Markdown: {md_path}")
    print(f"\n{BOLD}{_hr('═')}{RESET}\n")


if __name__ == "__main__":
    main()