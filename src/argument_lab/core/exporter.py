"""
argument_lab/core/exporter.py

Converts a completed DebateState into two outputs:
  1. A structured JSON file — the source of truth for the dashboard,
     argument graph renderer, and any downstream tooling.
  2. A human-readable Markdown report — auto-rendered from the JSON,
     suitable for reading debate results and evaluating agent reasoning.

Usage:
    from argument_lab.core.exporter import export_debate

    export_debate(
        state=final_state,
        session_id="debate_001",
        output_dir="local_data/results/",
    )
    # Writes:
    #   local_data/results/debate_001.json
    #   local_data/results/debate_001.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from argument_lab.core.models import Argument, JudgeEvaluation
from argument_lab.core.state import DebateState


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def export_debate(
    state: DebateState,
    session_id: str,
    output_dir: str | Path = "local_data/results",
) -> tuple[Path, Path]:
    """
    Serialises the final DebateState to JSON and Markdown.

    Args:
        state:       The final state returned by debate_graph.invoke().
        session_id:  A unique identifier for this debate session.
                     Used as the filename stem.
        output_dir:  Directory to write output files into.
                     Created if it doesn't exist.

    Returns:
        (json_path, md_path) — paths to the two written files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _build_json_payload(state, session_id)

    json_path = output_dir / f"{session_id}.json"
    md_path   = output_dir / f"{session_id}.md"

    _write_json(payload, json_path)
    _write_markdown(payload, md_path)

    return json_path, md_path


# ---------------------------------------------------------------------------
# JSON payload builder
# ---------------------------------------------------------------------------

def _build_json_payload(state: DebateState, session_id: str) -> dict:
    """
    Converts the DebateState into a clean, serialisable dict. All Pydantic
    models are expanded to dicts; sets are converted to sorted lists so
    the JSON is deterministic and diffable.
    """
    arguments = state.get("arguments", [])
    scores    = state.get("scores", [])

    return {
        "session_id":   session_id,
        "exported_at":  datetime.now(timezone.utc).isoformat(),
        "proposition":  state["proposition"],
        "status":       state.get("status", "unknown"),
        "rounds_completed": _rounds_completed(arguments),

        # ── Per-round debate transcript ──────────────────────────────
        "rounds": _build_rounds(arguments, scores),

        # ── Evaluation summary ───────────────────────────────────────
        "evaluation": {
            "hallucination_flags":  sorted(state.get("hallucination_flags", [])),
            "contradiction_flags":  sorted(state.get("contradiction_flags", [])),
            "hallucination_count":  len(state.get("hallucination_flags", [])),
            "contradiction_count":  len(state.get("contradiction_flags", [])),
        },

        # ── Score trajectories (for dashboard charts) ────────────────
        "score_trajectories": _build_score_trajectories(scores),

        # ── Agent confidence drift ───────────────────────────────────
        "agent_positions": {
            agent: positions
            for agent, positions in state.get("agent_positions", {}).items()
        },

        # ── Claim graph data ─────────────────────────────────────────
        "claim_graph": _build_claim_graph(arguments),

        # ── Ignored claims (penalised in scoring) ────────────────────
        "ignored_claims":  sorted(state.get("ignored_claims", [])),
        "addressed_claims": sorted(state.get("addressed_claims", [])),
    }


def _rounds_completed(arguments: list[Argument]) -> int:
    if not arguments:
        return 0
    return max(a.round for a in arguments)


def _build_rounds(
    arguments: list[Argument],
    scores: list[JudgeEvaluation],
) -> list[dict]:
    """
    Groups arguments by round and merges in the judge scores for that round.
    """
    max_round = _rounds_completed(arguments)
    score_by_round = {s.round: s for s in scores}
    rounds = []

    for r in range(1, max_round + 1):
        round_args = [a for a in arguments if a.round == r]
        proponent  = next((a for a in round_args if a.agent == "proponent"), None)
        opponent   = next((a for a in round_args if a.agent == "opponent"),  None)
        score      = score_by_round.get(r)

        rounds.append({
            "round": r,
            "proponent": _serialise_argument(proponent) if proponent else None,
            "opponent":  _serialise_argument(opponent)  if opponent  else None,
            "judge": _serialise_score(score, r)         if score     else None,
        })

    return rounds


def _serialise_argument(arg: Argument) -> dict:
    return {
        "id":             arg.id,
        "claim":          arg.claim,
        "evidence":       [
            {
                "source_id":        e.source_id,
                "excerpt":          e.excerpt,
                "reliability_score": e.reliability_score,
            }
            for e in arg.evidence
        ],
        "assumptions":             arg.assumptions,
        "counterpoints_addressed": arg.counterpoints_addressed,
        "confidence_score":        arg.confidence_score,
    }


def _serialise_score(score: JudgeEvaluation, round_num: int) -> dict:
    return {
        "round": round_num,
        "proponent": {
            "logical_consistency": score.proponent_score.logical_consistency,
            "evidence_support":    score.proponent_score.evidence_support,
            "relevance":           score.proponent_score.relevance,
            "completeness":        score.proponent_score.completeness,
            "composite":           score.proponent_score.composite,
        },
        "opponent": {
            "logical_consistency": score.opponent_score.logical_consistency,
            "evidence_support":    score.opponent_score.evidence_support,
            "relevance":           score.opponent_score.relevance,
            "completeness":        score.opponent_score.completeness,
            "composite":           score.opponent_score.composite,
        },
        "convergence_detected": score.convergence_detected,
        "stalemate_detected":   score.stalemate_detected,
        "explanation":          score.explanation,
    }


def _build_score_trajectories(scores: list[JudgeEvaluation]) -> dict:
    """
    Flattens per-round scores into chart-friendly arrays, one value per round.
    """
    sorted_scores = sorted(scores, key=lambda s: s.round)
    return {
        "rounds": [s.round for s in sorted_scores],
        "proponent_composite": [s.proponent_score.composite for s in sorted_scores],
        "opponent_composite":  [s.opponent_score.composite  for s in sorted_scores],
        "proponent_breakdown": [
            {
                "logical_consistency": s.proponent_score.logical_consistency,
                "evidence_support":    s.proponent_score.evidence_support,
                "relevance":           s.proponent_score.relevance,
                "completeness":        s.proponent_score.completeness,
            }
            for s in sorted_scores
        ],
        "opponent_breakdown": [
            {
                "logical_consistency": s.opponent_score.logical_consistency,
                "evidence_support":    s.opponent_score.evidence_support,
                "relevance":           s.opponent_score.relevance,
                "completeness":        s.opponent_score.completeness,
            }
            for s in sorted_scores
        ],
    }


def _build_claim_graph(arguments: list[Argument]) -> dict:
    """
    Exports the minimal claim graph data needed by the frontend D3 renderer.
    Full NetworkX graph construction lives in graph_update (v2), but this
    gives the dashboard enough to draw nodes and edges now.

    Nodes: one per argument (claim)
    Edges: counterpoints_addressed → "challenged_by" edges
    """
    nodes = []
    edges = []

    for arg in arguments:
        nodes.append({
            "id":     arg.id,
            "agent":  arg.agent,
            "round":  arg.round,
            "claim":  arg.claim,
            "confidence": arg.confidence_score,
        })
        for prior_id in arg.counterpoints_addressed:
            edges.append({
                "source": arg.id,
                "target": prior_id,
                "type":   "challenged_by",
            })

    return {"nodes": nodes, "edges": edges}


def _write_json(payload: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[exporter] JSON written to {path}")


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def _write_markdown(payload: dict, path: Path) -> None:
    lines = _render_markdown(payload)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[exporter] Markdown written to {path}")


def _render_markdown(p: dict) -> list[str]:
    status_emoji = {
        "converged":   "✅ Converged",
        "stalemate":   "⚖️  Stalemate",
        "terminated":  "🏁 Terminated (max rounds)",
        "in_progress": "⏳ In Progress",
    }.get(p["status"], p["status"])

    lines: list[str] = []
    w = lines.append  # shorthand

    # ── Header ────────────────────────────────────────────────────────────
    w(f"# ArgumentLab Debate Report")
    w(f"")
    w(f"**Session:** `{p['session_id']}`  ")
    w(f"**Exported:** {p['exported_at']}  ")
    w(f"**Status:** {status_emoji}  ")
    w(f"**Rounds completed:** {p['rounds_completed']}  ")
    w(f"")
    w(f"---")
    w(f"")
    w(f"## Proposition")
    w(f"")
    w(f"> {p['proposition']}")
    w(f"")

    # ── Score Summary ──────────────────────────────────────────────────────
    w(f"---")
    w(f"")
    w(f"## Score Summary")
    w(f"")
    traj = p.get("score_trajectories", {})
    rounds_list       = traj.get("rounds", [])
    prop_composites   = traj.get("proponent_composite", [])
    opp_composites    = traj.get("opponent_composite", [])

    if rounds_list:
        w(f"| Round | Proponent (composite) | Opponent (composite) | Verdict |")
        w(f"|---|---|---|---|")
        for r, pc, oc, round_data in zip(
            rounds_list,
            prop_composites,
            opp_composites,
            p.get("rounds", []),
        ):
            judge = round_data.get("judge") or {}
            verdict = ""
            if judge.get("convergence_detected"):
                verdict = "✅ Converged"
            elif judge.get("stalemate_detected"):
                verdict = "⚖️ Stalemate"
            w(f"| {r} | {pc:.3f} | {oc:.3f} | {verdict} |")
        w(f"")

    # ── Evaluation flags ───────────────────────────────────────────────────
    w(f"---")
    w(f"")
    w(f"## Evaluation Flags")
    w(f"")
    eval_data = p.get("evaluation", {})
    w(f"| Metric | Count |")
    w(f"|---|---|")
    w(f"| Hallucination flags | {eval_data.get('hallucination_count', 0)} |")
    w(f"| Contradiction flags | {eval_data.get('contradiction_count', 0)} |")
    w(f"| Ignored claims      | {len(p.get('ignored_claims', []))} |")
    w(f"| Addressed claims    | {len(p.get('addressed_claims', []))} |")
    w(f"")

    if eval_data.get("hallucination_flags"):
        w(f"**Hallucinated claim IDs:** `{'`, `'.join(eval_data['hallucination_flags'])}`")
        w(f"")
    if eval_data.get("contradiction_flags"):
        w(f"**Contradicted claim IDs:** `{'`, `'.join(eval_data['contradiction_flags'])}`")
        w(f"")

    # ── Round transcripts ──────────────────────────────────────────────────
    w(f"---")
    w(f"")
    w(f"## Debate Transcript")
    w(f"")

    for round_data in p.get("rounds", []):
        r = round_data["round"]
        w(f"### Round {r}")
        w(f"")

        for role in ("proponent", "opponent"):
            arg = round_data.get(role)
            if not arg:
                continue
            label = role.capitalize()
            confidence = arg["confidence_score"]
            w(f"#### {label}")
            w(f"")
            w(f"**Claim** *(confidence: {confidence:.2f})*")
            w(f"> {arg['claim']}")
            w(f"")

            if arg.get("evidence"):
                w(f"**Evidence cited**")
                for e in arg["evidence"]:
                    w(f"- `[{e['source_id']}]` (reliability: {e['reliability_score']:.2f})")
                    w(f"  > {e['excerpt']}")
                w(f"")

            if arg.get("assumptions"):
                w(f"**Assumptions**")
                for assumption in arg["assumptions"]:
                    w(f"- {assumption}")
                w(f"")

            if arg.get("counterpoints_addressed"):
                w(f"**Counterpoints addressed:** `{'`, `'.join(arg['counterpoints_addressed'])}`")
                w(f"")

        # Judge evaluation for this round
        judge = round_data.get("judge")
        if judge:
            w(f"#### Judge Evaluation — Round {r}")
            w(f"")
            w(f"| Dimension | Proponent | Opponent |")
            w(f"|---|---|---|")
            p_b = judge["proponent"]
            o_b = judge["opponent"]
            for dim in ("logical_consistency", "evidence_support", "relevance", "completeness"):
                label = dim.replace("_", " ").title()
                w(f"| {label} | {p_b[dim]:.2f} | {o_b[dim]:.2f} |")
            w(f"| **Composite** | **{p_b['composite']:.3f}** | **{o_b['composite']:.3f}** |")
            w(f"")
            w(f"**Judge's note:** {judge['explanation']}")
            w(f"")

        w(f"---")
        w(f"")

    # ── Confidence trajectories ────────────────────────────────────────────
    w(f"## Agent Confidence Trajectories")
    w(f"")
    for agent, positions in p.get("agent_positions", {}).items():
        trajectory = " → ".join(f"{v:.2f}" for v in positions)
        w(f"- **{agent.capitalize()}:** {trajectory}")
    w(f"")

    # ── Claim graph summary ────────────────────────────────────────────────
    graph = p.get("claim_graph", {})
    node_count = len(graph.get("nodes", []))
    edge_count = len(graph.get("edges", []))
    w(f"---")
    w(f"")
    w(f"## Argument Graph")
    w(f"")
    w(f"- **Total claims (nodes):** {node_count}")
    w(f"- **Challenged-by edges:** {edge_count}")
    w(f"- **Ignored claims:** {', '.join(p.get('ignored_claims', [])) or 'none'}")
    w(f"")
    w(f"*Full interactive graph available in the ArgumentLab dashboard.*")
    w(f"")

    return lines