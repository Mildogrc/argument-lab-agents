#!/usr/bin/env python3
"""
setup/debate.py

CLI runner for the ArgumentLab debate engine.
"""

import argparse
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# Path setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from argument_lab.core.state import DebateState
from argument_lab.core.retriever import Retriever
from argument_lab.core.faiss_index import FaissIndex
from argument_lab.orchestrator.graph import build_graph
from argument_lab.core.exporter import export_debate


# ANSI color helpers
def _supports_color() -> bool:
    return sys.stdout.isatty() and os.name != "nt"


RESET = "\033[0m" if _supports_color() else ""
BOLD = "\033[1m" if _supports_color() else ""
CYAN = "\033[36m" if _supports_color() else ""
YELLOW = "\033[33m" if _supports_color() else ""
GREEN = "\033[32m" if _supports_color() else ""
RED = "\033[31m" if _supports_color() else ""


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def _print_agent_update(node_name: str, update: dict) -> None:
    if "arguments" not in update:
        return
    arg = update["arguments"][-1]
    role = node_name.capitalize()
    color = CYAN if node_name == "proponent" else YELLOW

    print(f"\n  {color}{BOLD}{role}{RESET}")
    print(
        f"  {textwrap.fill(arg.claim, width=68, initial_indent='  ', subsequent_indent='  ')}"
    )
    print(f"  {BOLD}Confidence:{RESET} {arg.confidence_score:.2f}")


def _print_judge_update(update: dict) -> None:
    if "scores" not in update:
        return
    score = update["scores"][-1]
    print(f"\n  {BOLD}JUDGE{RESET}")
    print(
        f"  {textwrap.fill(score.explanation, width=68, initial_indent='  ', subsequent_indent='  ')}"
    )


def _merge_stream_update(state: DebateState, update: dict) -> DebateState:
    merged = dict(state)

    for key in (
        "arguments",
        "repetition_flags",
        "hallucination_flags",
        "contradiction_flags",
        "scores",
    ):
        if key in update:
            merged[key] = [*merged.get(key, []), *update[key]]

    for key in ("addressed_claims", "ignored_claims"):
        if key in update:
            merged[key] = set(merged.get(key, set())) | set(update[key])

    for key in ("claims_registry", "agent_positions"):
        if key in update:
            merged[key] = {**merged.get(key, {}), **update[key]}

    if "current_round" in update:
        merged["current_round"] = max(
            merged.get("current_round", 0),
            update["current_round"],
        )

    if "status" in update:
        merged["status"] = update["status"]

    if "verdict" in update:
        merged["verdict"] = update["verdict"]

    return merged


def main():
    parser = argparse.ArgumentParser(description="Run an ArgumentLab debate.")
    parser.add_argument("--proposition", required=True, help="The debate topic")
    parser.add_argument("--session-id", default=None, help="Session identifier")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print(f"{RED}Error: OPENAI_API_KEY not set.{RESET}")
        sys.exit(1)

    index_path = Path("local_data/faiss_index")
    if not index_path.exists():
        print(
            f"{RED}Error: FAISS index not found at {index_path}. Run ingest_corpus.py first.{RESET}"
        )
        sys.exit(1)

    index = FaissIndex.load(index_path)
    retriever = Retriever(index=index)
    debate_graph = build_graph(retriever)

    session_id = (
        args.session_id
        or f"debate_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )

    print(f"\n{BOLD}Proposition:{RESET} {args.proposition}\n")

    initial_state: DebateState = {
        "proposition": args.proposition,
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
        "verdict": None,
    }

    final_state = initial_state
    for chunk in debate_graph.stream(initial_state):
        for node_name, update in chunk.items():
            if node_name in ("proponent", "opponent"):
                _print_agent_update(node_name, update)
            elif node_name == "judge":
                _print_judge_update(update)
            final_state = _merge_stream_update(final_state, update)

    print(f"\n{BOLD}Debate Finished.{RESET} Status: {final_state['status']}")

    if final_state.get("verdict"):
        print(
            f"\n{BOLD}FINAL VERDICT ({final_state['verdict'].verdict_type.upper()}):{RESET}"
        )
        print(textwrap.fill(final_state["verdict"].summary, width=72))

    export_debate(final_state, session_id)


if __name__ == "__main__":
    main()
