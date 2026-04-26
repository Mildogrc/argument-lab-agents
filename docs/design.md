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

### 3. `orchestrator/graph.py` (Workflow Topology)
This file compiles the `StateGraph` that controls the execution flow. It is heavily parallelized to reduce latency:
- **Agent Fan-out**: The `start_round` node branches unconditionally to `proponent_node` and `opponent_node`, running them concurrently.
- **Evaluation Sync & Fan-out**: Both agents join at a dummy node (`start_evaluation`). From there, the graph fans out again to three concurrent evaluation nodes: `judge_node`, `hallucination_check`, and `contradiction_check`.
- **Graph Update & Routing**: The parallel evaluation nodes join at `graph_update`, which writes final states. The `route_round` conditional edge then reads the state's `status` to decide whether to loop back to `start_round` or terminate the debate (`END`).

---

*Note: This document should be updated whenever significant structural changes, new node implementations, or data models are introduced.*
