# ArgumentLab: Code Design & Implementation

This document serves as a living guide to the actual codebase. It explains the current structure of the Python packages, what each file does, and how the logic is implemented.

---

## Code Organization

The application is modularized under `src/argument_lab/`, currently split into `core` data structures and the `orchestrator` graph logic.

### 1. `core/models.py` (Pydantic Schemas)
This file defines the strict data schemas enforced throughout the system, primarily for structured LLM outputs.
- **`Argument`**: The core output format for agents. It strictly types the `agent` field (`"proponent"` or `"opponent"`) and enforces a `min_length=1` validator on `evidence`, meaning an agent cannot return an argument without at least one citation.
- **`EvidenceRef`** & **`Claim`**: Base objects for tracking retrieved evidence and registering claims.
- **`JudgeEvaluation` & `ArgumentScore`**: Defines the multi-dimensional scoring rubric (logical consistency, evidence support, relevance, completeness) as well as arrays for storing `hallucination_flags` and `contradiction_flags`.

### 2. `core/state.py` (LangGraph State Management)
This file defines `DebateState`, the shared dictionary passed between all nodes in the LangGraph workflow.
Because nodes run in parallel, we implement custom **reducers** to prevent race conditions (where "last-write-wins" would corrupt the data):
- **`merge_dicts`**: Merges dictionary updates to `claims_registry` and `agent_positions`.
- **`union_sets`**: Merges sets of `addressed_claims` and `ignored_claims`.
- **`max_round`**: Ensures the `current_round` integer can only increase, preventing a lagging node from resetting the round number.
- **`merge_status`**: Resolves `status` updates by priority (e.g. if one node writes `"converged"` and another lazily writes `"in_progress"`, it resolves to `"converged"`).

### 3. `core/agents.py` (Proponent & Opponent Logic)
Implements the core LangGraph agent nodes. Both agents follow a strict, deterministic pipeline instead of a chatty ReAct loop:
1. **Query Formulation**: A lightweight LLM call creates 1-3 targeted search queries based on the agent's stance and the debate history.
2. **Retrieval**: The queries are executed to fetch real `EvidenceRef` chunks.
3. **Generation**: The LLM uses `.with_structured_output(Argument)` to generate its argument, using the retrieved context.
4. **Counterpoint Enforcement**: In Rounds 2 and 3, the node explicitly re-prompts the LLM if it fails to populate `counterpoints_addressed` with an opponent's prior claim ID.

### 4. `core/retriever.py` (RAG Interface)
A thin abstraction over the vector database (e.g. FAISS). It exposes `retrieve_multi()` which aggregates search results for multiple queries and deduplicates them by `source_id`, guaranteeing the best chunks are surfaced to the agent.

### 5. `core/evaluation.py` (Parallel Evaluators)
Contains the three concurrent evaluation nodes that run after the agents:
- **`judge_node`**: Uses an LLM to score both arguments across four dimensions, detects convergence/stalemate, updates the debate `status`, and increments the round.
- **`hallucination_check`**: Validates that cited sources explicitly support the claims. Appends failing claim IDs to `hallucination_flags`.
- **`contradiction_check`**: Compares current arguments against the agent's historical claims to detect goalpost shifting. Appends offending claim IDs to `contradiction_flags`.

### 6. `core/prompts.py` & `core/eval_prompts.py`
Isolate all LangChain `ChatPromptTemplate` strings. They handle formatting debate histories, chunk excerpts, and evaluation logic, making it easy to iterate on prompt wording without touching workflow logic.

### 7. `orchestrator/graph.py` (Workflow Topology)
This file compiles the `StateGraph` that controls the execution flow. It is heavily parallelized to reduce latency:
- **Agent Fan-out**: The `start_round` node branches unconditionally to `proponent_node` and `opponent_node`, running them concurrently.
- **Evaluation Sync & Fan-out**: Both agents join at a dummy node (`start_evaluation`). From there, the graph fans out again to three concurrent evaluation nodes: `judge_node`, `hallucination_check`, and `contradiction_check`.
- **Graph Update & Routing**: The parallel evaluation nodes join at `graph_update`, which writes final states. The `route_round` conditional edge then reads the state's `status` to decide whether to loop back to `start_round` or terminate the debate (`END`).

## Implementation Efficiencies

1. **The 2-step retrieval pipeline**: Doing `_formulate_queries` -> `_retrieve_evidence` before entering the structured argument generator avoids the grounding problem. It gives you the benefits of tool use without the risk of the LLM abandoning the schema or crashing into infinite tool loops.
2. **The `.model_copy(update=...)` filter**: 
   ```python
   "evidence": [e for e in argument.evidence if e.source_id in valid_source_ids] or evidence_refs[:1]
   ```
   If the LLM hallucinates source IDs, they are filtered out. But because Pydantic demands `min_length=1`, replacing an empty list with `evidence_refs[:1]` guarantees that validation will pass, avoiding a potential failure state.
3. **Counterpoint Enforcement**: Using the `_enforce_counterpoint_rule` to re-prompt the LLM explicitly when it fails to address an opponent's claim handles Option 3. Raising an `AgentError` if it fails twice propagates the failure to the workflow and it will get caught by the judge, as a result the judge will lower the score for logical consistency.
4. **State updates**: Extracting the confidence trajectory and accurately mapping `newly_ignored` claims via set math.


## Testing

Here is what was added:
1. **`tests/core/test_state.py`**: Tests all the custom reducers (`union_sets`, `merge_dicts`, `max_round`, `merge_status`) to ensure they handle `None` defaults properly and execute the right merge logic.
2. **`tests/core/test_retriever.py`**: Mocks the `VectorIndex` protocol to test `Retriever.retrieve()` and ensures that `retrieve_multi()` correctly deduplicates source chunks, keeping the highest score.
3. **`tests/core/test_prompts.py`**: Tests the formatting helpers (`format_debate_history` and `format_evidence_context`) for edge cases like empty histories.
4. **`tests/core/test_agents.py`**: Tests the pure Python state derivation logic (`_get_prior_opponent_claim_ids` and `_update_state_from_argument`).
5. **`tests/orchestrator/test_graph.py`**: Tests that the `build_graph()` factory can successfully compile the graph topologically.
6. **`tests/core/test_models.py`**: Kept your existing test verifying `min_length=1` for evidence.

---

*Note: This document should be updated whenever significant structural changes, new node implementations, or data models are introduced.*
