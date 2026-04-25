# ArgumentLab — Architecture Document

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Layer 1: Core Reasoning Engine](#layer-1-core-reasoning-engine)
4. [Layer 2: Debate System](#layer-2-debate-system)
5. [Layer 3: Evaluation and Reliability](#layer-3-evaluation-and-reliability)
6. [Data Models](#data-models)
7. [Component Interaction & Data Flow](#component-interaction--data-flow)
8. [Tech Stack](#tech-stack)
9. [Deployment Architecture](#deployment-architecture)
10. [Key Design Decisions](#key-design-decisions)
11. [MVP vs. Future Scope](#mvp-vs-future-scope)

---

## System Overview

ArgumentLab is a multi-agent reasoning framework that runs structured AI-to-AI debates and quantitatively evaluates argument quality, consistency, and hallucination under adversarial conditions.

The system is organized into three stacked layers:

![System Overview](/docs/images/systemoverview.png)

Each layer is independently testable and builds on the outputs of the layer below it.

---

## High-Level Architecture

![High-Level Architecture](/docs/images/highlevelarchitecture.png)

---

## Layer 1: Core Reasoning Engine

### Agents

The system uses four agents with distinct, non-overlapping roles:

| Agent | Role | Required |
|---|---|---|
| **Proponent** | Argues FOR the proposition | Yes |
| **Opponent** | Argues AGAINST the proposition | Yes |
| **Judge** | Evaluates quality, detects convergence, produces verdict | Yes |
| **Moderator** | Enforces structure, prevents topic drift | Optional (v2+) |

Each agent is implemented as a stateless function that receives a context object and returns a structured argument. State is held externally in the Shared Debate State Manager.

### Structured Argument Schema

Every agent output is a typed JSON object — no free-form prose:

```json
{
  "round": 2,
  "agent": "proponent",
  "claim": "string — the primary assertion being made",
  "evidence": [
    {
      "source_id": "doc_42_chunk_7",
      "excerpt": "string — relevant passage",
      "reliability_score": 0.91
    }
  ],
  "assumptions": ["string — unstated premises this argument depends on"],
  "counterpoints_addressed": ["claim_id_from_prior_round"],
  "confidence_score": 0.82,
  "metadata": {
    "timestamp": "ISO 8601",
    "tokens_used": 412,
    "retrieval_latency_ms": 230
  }
}
```

This schema is enforced at the orchestrator level. Arguments that fail schema validation are rejected and the agent is re-prompted.

### Debate Loop

Debates run across three rounds with increasing specificity requirements:

```
Round 1 ──► Initial arguments; top-level claims; no rebuttal required
Round 2 ──► Targeted rebuttals; each agent must reference ≥1 prior claim by ID
Round 3 ──► Position refinement; agents may update confidence scores
             and must acknowledge new evidence introduced in Round 2
```

The orchestrator enforces sequencing: agents within a round may run in parallel, but Round N cannot begin until all agents have submitted valid Round N-1 outputs.

### Shared Debate State Manager

A central state object persists across rounds and is passed to each agent:

```python
DebateState = {
    "proposition": str,
    "rounds": List[Round],            # All prior round outputs
    "claims_registry": Dict[str, Claim],  # claim_id → claim object
    "addressed_claims": Set[str],     # Claims that received rebuttals
    "ignored_claims": Set[str],       # Claims that were not engaged
    "agent_positions": Dict[str, List[float]],  # Confidence drift per agent
    "repetition_flags": List[str],    # Claim IDs flagged as near-duplicates
}
```

Position drift (change in `confidence_score` across rounds) is tracked as a signal for convergence detection and contradiction analysis.

---

## Layer 2: Debate System

### Topic Framing

Raw user input is pre-processed by the Topic Framer before any agent receives it:

1. **Proposition normalization** — converts questions to binary propositions ("Should X?" → "X is preferable to Y under conditions Z")
2. **Scope constraints** — limits the proposition to a defined domain to prevent topic drift
3. **Ambiguity flagging** — identifies underspecified terms that could cause agents to argue past each other

Output is a `DebateProposition` object shared with all agents.

### Evidence Retrieval (RAG)

Agents cannot assert facts without grounding them in retrieved evidence. The RAG pipeline:

![Evidence Retrieval](/docs/images/evidenceretrieval.png)

Source reliability is tracked as a first-class metric. Sources are scored on: domain authority (if web), internal consistency, and citation frequency across rounds.

### Argument Graph

The debate is represented as a directed graph, not a linear transcript. This is the primary data structure for the debate:

![Argument Graph](/docs/images/argumentgraph.png)

**Node types:** `claim`, `rebuttal`, `evidence`, `concession`, `agreement_zone`

**Edge types:** `supports`, `challenged_by`, `ignored_by`, `conceded_by`, `cited_by`

The graph is stored in NetworkX (backend) and serialized to JSON for frontend rendering with D3.js. The Evaluation Layer queries this graph directly for contradiction and convergence analysis.

### Convergence Detector

The Judge agent monitors three convergence signals after each round:

| Signal | Definition | Action |
|---|---|---|
| **Agreement zone** | Both agents assign high confidence to the same claim | Mark as resolved; remove from active debate |
| **Unresolved conflict** | Both agents maintain opposing high-confidence positions | Flag for final verdict |
| **Stalemate** | No argument quality score improvement across a full round | Trigger early termination |

At termination, the Judge outputs either a `ConsensusVerdict` or a `BestArgumentVerdict` with a structured explanation of what remained unresolved and why.

---

## Layer 3: Evaluation and Reliability

### Argument Quality Scorer

Each structured argument is scored independently by the Judge agent using a rubric-prompted LLM call:

| Dimension | Weight | What It Measures |
|---|---|---|
| **Logical Consistency** | 30% | Premises → conclusion validity; no internal contradictions |
| **Evidence Support** | 30% | Fraction of claims backed by retrieved sources |
| **Relevance** | 20% | Argument addresses the stated proposition, not a related but different claim |
| **Completeness** | 20% | Engages with the opponent's strongest prior point |

Scores are in [0, 1]. The weighted composite score is stored per agent per round and fed to the Metrics Dashboard.

### Hallucination Detector

Runs as a post-processing step on every argument:

```
For each claim in argument:
  1. Check that every cited source_id exists in the vector index
  2. Retrieve the cited chunk; verify the claim is semantically entailed
  3. Extract named entities (names, numbers, dates) from the claim
  4. Verify each entity appears verbatim in the cited source
  5. Flag mismatches as hallucinations with a severity score
```

Hallucination rate is tracked as `hallucinations / total_claims` per agent per round.

### Contradiction Detector

Compares each agent's Round N output against all prior rounds for the same agent:

```
For each (Round N claim, Round K claim) where K < N:
  1. Embed both claims
  2. If cosine similarity > threshold_similar:
     → Check for logical contradiction using LLM-as-judge
  3. If contradiction confirmed:
     → Flag with round numbers, claim IDs, and contradiction type
     → Apply penalty to agent's Round N quality score
```

Contradiction types: `direct_negation`, `weakened_commitment`, `shifted_evidence_basis`, `ignored_own_prior_claim`.

### Adversarial Testing Mode

A controlled evaluation mode that injects degraded inputs before the debate begins:

| Injection Type | Implementation | Purpose |
|---|---|---|
| **Misleading data** | Replace k% of RAG chunks with plausible-but-false variants | Test hallucination resistance |
| **Incomplete context** | Remove chunks covering key sub-topics | Test inference under gaps |
| **Conflicting evidence** | Inject sources contradicting existing corpus | Test conflict resolution |

Adversarial sessions produce a `BehavioralProfile` comparing quality scores, hallucination rates, and contradiction frequency against a clean baseline run on the same proposition.

### Metrics Dashboard

Tracked per session, surfaced in real-time (Streamlit or React + Recharts):

```
Per-agent, per-round:
  - Argument quality score (composite + per-dimension breakdown)
  - Hallucination rate
  - Contradiction count
  - Evidence citation rate
  - Confidence score trajectory

Per-session:
  - Convergence round (or "did not converge")
  - Disagreement persistence (% of claims still unresolved at termination)
  - Argument graph topology (depth, branching factor, ignored-claim rate)
  - Adversarial delta (if applicable)
```

---

## Data Models

### Core Types (Python / Pydantic)

```python
class Argument(BaseModel):
    id: str                          # UUID
    round: int
    agent: Literal["proponent", "opponent", "judge"]
    claim: str
    evidence: List[EvidenceRef]
    assumptions: List[str]
    counterpoints_addressed: List[str]  # List of claim IDs
    confidence_score: float          # [0, 1]

class EvidenceRef(BaseModel):
    source_id: str
    excerpt: str
    reliability_score: float

class DebateSession(BaseModel):
    id: str
    proposition: str
    status: Literal["in_progress", "converged", "stalemate", "terminated"]
    rounds: List[List[Argument]]    # rounds[i] = all arguments in round i
    argument_graph: GraphData
    metrics: SessionMetrics
    verdict: Optional[Verdict]

class Verdict(BaseModel):
    type: Literal["consensus", "best_argument"]
    winner: Optional[Literal["proponent", "opponent"]]
    consensus_claims: List[str]
    unresolved_claims: List[str]
    explanation: str
```

---

## Component Interaction & Data Flow

![Component Interaction](/docs/images/dataflow.png)

---

## Tech Stack

| Component | Technology | Notes |
|---|---|---|
| Agent orchestration | LangGraph / custom loop | LangGraph for state management; custom loop for strict round enforcement |
| LLM backbone | Claude (Anthropic API) / GPT-4o | Swappable via provider abstraction |
| Embedding model | OpenAI `text-embedding-3-small` or equivalent | Used for RAG and contradiction detection |
| Vector index | FAISS (local) / ChromaDB (persistent) | FAISS for MVP; ChromaDB for v2 with session history |
| Argument graph | NetworkX (backend), D3.js (frontend) | Graph exported as JSON for frontend |
| Scoring | LLM-as-judge with rubric prompting | Judge agent uses structured output schema |
| Metrics dashboard | Streamlit (MVP) / React + Recharts (v2) | Streamlit for rapid iteration |
| Backend API | FastAPI | REST endpoints + WebSocket for real-time dashboard |
| Frontend | React (v2) | D3.js for argument graph; Recharts for metrics |

---

## Deployment Architecture

![Deployment Architecture](/docs/images/deployment.png)

---

## Key Design Decisions

**Structured output over free text.** All agent outputs are JSON objects validated against a schema. This prevents the "chatty LLM" failure mode and makes every output programmatically evaluable.

**Claim IDs as the unit of discourse.** Every claim receives a UUID. Rebuttals must reference prior claim IDs. This enforces specificity and makes it possible to detect when agents ignore arguments rather than counter them.

**External state management.** Agents are stateless functions. All debate context lives in the Shared Debate State Manager. This makes agents independently testable and allows replaying any round with a modified state.

**Graph over transcript.** Representing the debate as a directed graph (rather than a linear conversation) exposes the actual reasoning topology — which arguments were addressed, which were ignored, and which drove convergence.

**LLM-as-judge with rubric.** The Judge agent uses structured prompting with an explicit scoring rubric rather than free-form evaluation. This makes scores reproducible and auditable.

**Hallucination as a first-class metric.** Most debate systems track persuasiveness. ArgumentLab treats factual grounding as equally important — an argument that cites non-existent evidence is penalized regardless of its logical structure.

---

## MVP vs. Future Scope

### v1 (MVP)
- Proponent + Opponent + Judge agents
- 3-round debate loop with structured argument format
- Basic quality scoring (logical consistency + evidence support)
- Hallucination detection (source existence check)
- CLI or minimal web UI
- FAISS vector index over user-provided documents

### v2
- Argument graph visualization (D3.js)
- Full hallucination pipeline (semantic entailment + entity grounding)
- Contradiction detector with cross-round comparison
- Metrics dashboard (React + Recharts)
- Convergence detection with stalemate handling
- Persistent sessions (ChromaDB + Postgres)

### Stretch Goals
- Human-in-the-loop intervention (pause, redirect, inject evidence)
- Adversarial injection test suite
- Strategy modes: `aggressive`, `evidence-first`, `exploratory`
- Multi-agent expansion: domain expert, skeptic, data-driven agents
- Longitudinal session tracking (improvement across multiple debates on the same topic)
