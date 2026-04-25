# ArgumentLab
 
**A multi-agent reasoning system that conducts structured debates and quantitatively evaluates argument quality, consistency, and hallucination under adversarial conditions.**
 
Most AI systems are built to answer questions. ArgumentLab is built to *interrogate them* — forcing agents to construct structured arguments, defend them across debate rounds, cite real evidence, and withstand adversarial pressure. The goal is not a better chatbot. It is a rigorous framework for studying how AI systems reason, disagree, and fail.
 
---
 
## Why This Is Hard
 
Getting an LLM to argue a position is trivial. Getting it to:
 
- maintain logical consistency across multiple rounds
- cite grounded, verifiable evidence
- respond specifically to an opponent's claims (not just re-assert its own)
- detect when it is contradicting itself
- converge toward a defensible conclusion under adversarial input
...is not. ArgumentLab treats each of these as a measurable engineering problem.
 
---
 
## Architecture Overview
 
Refer to [Architecture document](/docs/architecture.md)
 
---
 
## Layer 1: Core Reasoning Engine
 
### Multi-Agent Architecture
 
Four agents drive the system:
 
| Agent | Role |
|---|---|
| **Proponent** | Argues FOR the proposition |
| **Opponent** | Argues AGAINST the proposition |
| **Judge** | Evaluates argument quality and detects convergence |
| **Moderator** *(optional)* | Enforces debate structure and prevents drift |
 
### Structured Argument Format
 
Agents do not produce free text. Every argument is a structured object:
 
```json
{
  "claim": "...",
  "evidence": ["source_1", "source_2"],
  "assumptions": ["..."],
  "counterpoints_addressed": ["..."],
  "confidence_score": 0.82
}
```
 
This eliminates the "chatty LLM" failure mode and makes every output machine-evaluable.
 
### Iterative Debate Loop
 
Debates run across three rounds with increasing specificity:
 
- **Round 1** — Initial arguments, top-level claims
- **Round 2** — Targeted rebuttals; agents must address specific prior claims
- **Round 3** — Refinement; agents update positions based on accumulated evidence
Each agent receives the full prior-round context and is penalized (in scoring) for ignoring it.
 
### Memory and Context Tracking
 
A shared debate state tracks:
- all claims made across rounds
- which claims have been addressed vs. ignored
- agent position drift over time
- repetition detection
---
 
## Layer 2: Debate System
 
### Topic Input and Framing
 
The user provides a question and optional supporting documents. The system converts this into a formal debate proposition with scoped constraints, ensuring agents argue the same thing rather than talking past each other.
 
### Evidence Integration
 
This is the key technical differentiator. Agents are not allowed to assert facts — they must retrieve them. Evidence is pulled via RAG over user-provided documents or curated corpora. Each claim is linked to a source, and source reliability is tracked as a first-class metric.
 
### Convergence Detection
 
The Judge agent continuously monitors for:
- **Agreement zones** — claims both agents accept
- **Unresolved conflicts** — positions neither agent concedes
- **Stalemates** — rounds where neither argument quality score improves
At termination, the Judge produces either a synthesized consensus or a ranked "best argument so far" with an explanation of what remained unresolved.
 
---
 
## Layer 3: Evaluation and Reliability
 
This layer is what separates ArgumentLab from a demo.
 
### Argument Quality Scoring
 
Each argument is scored across four dimensions:
 
| Dimension | What It Measures |
|---|---|
| **Logical Consistency** | Does the argument follow from its premises? |
| **Evidence Support** | Are claims backed by retrieved sources? |
| **Relevance** | Does the argument address the actual proposition? |
| **Completeness** | Does it engage with the opponent's strongest points? |
 
### Hallucination Detection
 
The system checks whether cited sources exist, whether their content supports the stated claim, and whether specific facts (names, numbers, dates) are grounded in the retrieved evidence. Hallucination rate is tracked as a per-agent, per-round metric.
 
### Contradiction Detection
 
Agents are compared against their own prior arguments. If an agent's Round 3 position is logically inconsistent with its Round 1 claims, this is flagged, scored, and surfaced in the metrics dashboard. Ignoring an opponent's argument is also penalized.
 
### Adversarial Testing
 
ArgumentLab includes a controlled evaluation mode that injects:
- **Misleading data** — plausible but false evidence
- **Incomplete context** — information gaps that require inference
- **Conflicting evidence** — sources that contradict each other
This stress-tests agent robustness and produces behavioral profiles under degraded conditions.
 
### Metrics Dashboard
 
Tracked across every debate session:
 
- Argument quality score per agent per round
- Hallucination rate
- Contradiction frequency
- Convergence round (or failure to converge)
- Evidence citation rate
- Disagreement persistence across rounds
---
 
## Demo Flow
 
1. Input a real-world question (e.g., *"Should companies replace legacy infrastructure with AI-driven systems?"*)
2. Watch structured debate rounds with per-argument scoring
3. Explore the argument graph — claims, rebuttals, ignored threads
4. Review the metrics dashboard — quality trends, hallucination flags, contradiction alerts
5. Read the Judge's synthesis — consensus reached, or best-argument verdict with unresolved conflicts
---
 
## Tech Stack
 
| Component | Technology |
|---|---|
| Agent orchestration | LangGraph / custom agent loop |
| LLM backbone | OpenAI GPT-4o / Claude (via API) |
| Evidence retrieval | RAG with FAISS or ChromaDB |
| Argument scoring | Structured LLM-as-judge with rubric prompting |
| Argument graph | NetworkX (backend), D3.js (frontend) |
| Metrics dashboard | Streamlit or React + Recharts |
| Backend | FastAPI |
 
---
 
## MVP Scope
 
**Must-have (v1):**
- Proponent + Opponent + Judge agents
- Structured argument format (claim, evidence, confidence)
- 3-round debate loop with context tracking
- Basic scoring — logical consistency + evidence support
- CLI or minimal web UI
**v2 additions:**
- Argument graph visualization
- Hallucination detection pipeline
- Metrics dashboard
- RAG-based evidence integration
**Stretch goals:**
- Human-in-the-loop intervention
- Adversarial injection testing suite
- Strategy modes (aggressive / evidence-first / exploratory)
- Multi-agent expansion (domain expert, skeptic, data-driven agents)
- Session history and longitudinal improvement tracking
---
 
## How This Differs from Kialo
 
[Kialo](https://www.kialo.com) is a platform for human-generated, community-refined argument trees — effectively structured Wikipedia for reasoning. It is a valuable tool for its purpose.
 
ArgumentLab is a different category entirely:
 
| | Kialo | ArgumentLab |
|---|---|---|
| Arguments | Human-written | AI-generated |
| Debate | Static tree | Multi-round, iterative |
| Self-critique | No | Yes |
| Evidence grounding | No | Yes (RAG + citation) |
| Scoring / evaluation | No | Yes (multi-dimensional) |
| Hallucination detection | No | Yes |
| Adversarial testing | No | Yes |
| Convergence mechanism | No | Yes |
 
> Kialo organizes human arguments. ArgumentLab studies how AI systems reason, fail, and improve.
 
---
 
## Research Connections
 
ArgumentLab sits at the intersection of several active research directions:
 
- **LLM-as-Judge** — using language models as evaluators of reasoning quality
- **Multi-agent debate** — Du et al. (2023), *Improving Factuality and Reasoning in Language Models through Multiagent Debate*
- **Constitutional AI / self-critique** — agents that evaluate and revise their own outputs
- **Adversarial robustness** — measuring agent behavior under distributional shift
---
 
## Project Status
 
🚧 In development — contributions and feedback welcome.
 
---
 
## Author
 
**Milind C** — MS Computer Science (Artifical Intelligence), Georgia Institute of Technology
[LinkedIn](https://linkedin.com/in/milind-chandramohan) · [GitHub](https://github.com/mildogrc)
 
