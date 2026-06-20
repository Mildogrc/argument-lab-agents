# ReasonBench Code Map


> The current codebase is a working **ArgumentLab debate engine**, with the new README now repositioning it toward **ReasonBench as an SLM-focused benchmark**. In other words: the docs/roadmap have moved toward local SLM reasoning improvement, but the actual code still mostly implements the earlier OpenAI-backed multi-agent debate loop.

## V0.1 ArgumentLab With LLMs 

The runtime path is:

```text
setup/debate.py
  -> loads FAISS index
  -> wraps it in Retriever
  -> builds LangGraph workflow
  -> runs proponent + opponent debate rounds
  -> runs hallucination/contradiction/judge evaluation
  -> generates final verdict
  -> exports JSON/Markdown report
```

The main package lives under `src/argument_lab`.

### Core Files

- The file `setup/debate.py` is the CLI entry point. It requires `OPENAI_API_KEY`, expects `local_data/faiss_index`, builds the graph, streams node updates, and exports results.

- The file `src/argument_lab/orchestrator/graph.py` defines the LangGraph topology. Each round fans out to `proponent` and `opponent`, then evaluation runs through hallucination and contradiction checks before the judge scores and routes to the next round or final verdict.

- The file `src/argument_lab/core/state.py` defines `DebateState`, the shared LangGraph state. The important thing here is the custom reducers: lists append, sets union, dicts merge, round number only increases, and status resolves by priority.

- The file `src/argument_lab/core/models.py` defines the main Pydantic objects: `Argument`, `EvidenceRef`, `Claim`, `ArgumentScore`, `JudgeEvaluation`, hallucination/contradiction flags, and final `Verdict`.

- The file `src/argument_lab/core/agents.py` contains the *proponent*/*opponent* logic. Each agent formulates retrieval queries, fetches evidence, generates a structured `Argument`, filters hallucinated source IDs, and enforces that later rounds address opponent claims.

- The file `src/argument_lab/core/retriever.py` is a small adapter around any vector index. It has `retrieve()` and `retrieve_multi()`, with deduplication by `source_id`.

- The file `src/argument_lab/core/faiss_index.py` is the concrete FAISS/OpenAI embeddings backend. It builds, saves, loads, and searches the local vector index.

- The file `src/argument_lab/core/evaluation.py` runs judge scoring, hallucination checks, contradiction checks, and final verdict generation. These currently use `gpt-4o` through `langchain_openai`.

- The file `src/argument_lab/core/reasonbench_models.py` and  `src/argument_lab/core/reasonbench_eval.py` are early ReasonBench-specific schemas/eval helpers for the three task types, but they are not yet fully integrated into the main graph.

### What Works Today
The current implementation can run a structured debate over a proposition, assuming:
- dependencies are installed,
- `OPENAI_API_KEY` is set,
- a FAISS index exists from `setup/ingest_corpus.py`.

It supports:
- proponent/opponent agents,
- 3-round debate flow,
- RAG-grounded argument generation,
- structured Pydantic outputs,
- LLM judge scoring,
- hallucination and contradiction checker prompts,
- final JSON/Markdown export.

### Important Gap
Though the README now says “local-first SLM benchmark,” but the code is still mostly:
- OpenAI-backed,
- debate-first,
- FAISS/OpenAI embeddings based,
- not yet organized into `reasonbench-core`, `reasonbench-tasks`, `reasonbench-evals`, etc.

### Tests
The tests are focused and lightweight:
- state reducer behavior,
- retriever dedup/failure behavior,
- model validation,
- prompt formatting,
- agent state-update helpers,
- graph compilation.

## V0.2 - With SLM Capability

### New Additions

The class `model_adapters.py` is the abstraction layer for talking to different local/open model backends through one common interface. Its purpose is to let the rest of ReasonBench say “generate text from this model” without caring whether the model is served by Ollama, llama.cpp, or HuggingFace. It defines:

- `GenerationRequest`: prompt, optional system message, temperature, max tokens.
- `GenerationResponse`: generated text plus model/provider metadata.
- `ModelAdapter`: the shared protocol all adapters follow.
- `OllamaAdapter`: calls Ollama’s local `/api/chat`.
- `LlamaCppAdapter`: calls llama.cpp’s OpenAI-compatible `/v1/chat/completions`.
- `HuggingFaceAdapter`: runs a local HuggingFace `transformers` pipeline, imported lazily so it stays optional.

The file `protocols.py` is the benchmark protocol layer. Its purpose is to run a task through one of the ReasonBench evaluation modes and produce a structured trace of what happened. It defines:

- `baseline`: model answers once.
- `self_critique`: model answers, critiques itself, then revises.
- `debate`: subject model answers, opponent challenges, subject revises.
- `debate_judge`: debate plus judge scoring.
- `BenchmarkTask`: task input schema.
- `ProtocolTurn`: each generated step in the trace.
- `JudgeScores`: scoring fields, including initial/final score.
- `ProtocolResult`: final structured result for one protocol run.
- `ProtocolRunner`: the class that coordinates models through these modes.

Together, they move the repo from “hardcoded debate engine” toward “local-first ReasonBench benchmark framework.” `model_adapters.py` answers “how do we call models?” and `protocols.py` answers “what experimental protocol do we run over those models?”

### Judging SLMs

The class `OpenAIChatAdapter` in `model_adapters.py` is to allow LLM to judge, and `OllamaAdapter` is to create `proponent` and `opponent` entities; so the intended setup is now:

```python
runner = ProtocolRunner(
    subject_model=OllamaAdapter("qwen2.5:3b"),      # SLM
    opponent_model=OllamaAdapter("gemma2:2b"),      # SLM
    judge_model=OpenAIChatAdapter("gpt-4o"),        # stronger LLM judge
)
```

Then:

```python
result = runner.run("debate_judge", task)
```

The SLMs do the debate/revision work, while the LLM only evaluates `initial_response` vs `final_response` and returns structured scores. I also tightened the judge system prompt so it is framed as an external evaluator, not another debater.

Verification:
- `python -m ruff check .` passed
- `python -m black --check .` passed
- `python -m pytest` passed: 24 tests