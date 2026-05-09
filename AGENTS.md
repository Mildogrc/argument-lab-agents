# AGENTS.md

## Project overview
**ArgumentLab**: *Multi-Agent Debate and Reasoning Engine* 
**Core Tech**: Python, LLMs, LangGraph, Retrieval, Evals

- Built a multi-agent reasoning system where generator, critic, and judge agents iteratively construct, evaluate, and refine arguments on complex queries.
- Implemented adversarial critique loops and structured reasoning traces (tree-of-thought) to improve logical consistency and depth of generated responses.
- Developed evaluation metrics for reasoning quality, including coherence, factual grounding, and inter-agent agreement, enabling quantitative analysis of outputs.
- Integrated retrieval over external knowledge sources to ground arguments in evidence, reducing hallucination rates and improving factual accuracy.
- Designed experimentation framework to compare single-agent vs multi-agent performance, demonstrating measurable gains in answer quality and robustness.


## Tech stack
- Current implementation: Python, LangGraph, LangChain, FAISS, pytest
- Planned backend: FastAPI
- Planned frontend: React, TypeScript
- Planned database: PostgreSQL
- LLM orchestration: LangGraph
- Tests: pytest; Vitest only after a frontend is added

## Rules for AI coding agents
- Do not make large rewrites unless explicitly requested.
- Prefer small, reviewable changes.
- Before coding, summarize the intended approach.
- After coding, run tests and explain what changed.
- Do not add new dependencies without explaining why.
- Do not modify secrets, credentials, or deployment settings.

## Commands
- Install backend: `pip install -r requirements.txt`
- Run backend tests: `pytest`
- Run frontend tests: `npm test` after `package.json` exists
- Run full verification: `./scripts/verify.sh`

## Repository map
- Architecture overview: `docs/architecture.md`
- Current code design: `docs/design.md`
- Development environment: `docs/dev_environment.md`
- Testing and verification: `docs/testing.md`
- Agent workflow: `docs/agent_workflow.md`
- Harness readiness checklist: `docs/harness_readiness.md`

## After making changes

1. Run ./scripts/verify.sh
2. If failures occur:
   - Fix lint issues first
   - Then fix test failures
3. Do not stop until verify.sh passes

## Definition of done
- Code compiles
- Tests pass
- Lint passes
- New behavior is covered by tests
- Diff has been reviewed for risky changes
