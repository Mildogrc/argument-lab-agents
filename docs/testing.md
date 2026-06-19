# Testing and Verification

This document is the source of truth for how humans and coding agents should validate ArgumentLab changes.

## Current Verification Commands

Run the full repository verification before handing off a code change:

```bash
./scripts/verify.sh
```

The verification script currently runs:

```bash
./scripts/setup.sh
./scripts/lint.sh
./scripts/test.sh
```

For focused work, use the smaller checks:

```bash
./scripts/lint.sh
./scripts/test.sh
pytest
```

## Current Harness Limitations

These are known gaps. Agents should not treat this project as fully unattended until they are fixed.

- The setup flow depends on `python` and `pip` being available on PATH.
- There is no `pyproject.toml`, lockfile, or pinned Python runtime file.
- `scripts/lint.sh` skips Python lint/format checks if `ruff` or `black` are missing.
- `scripts/test.sh` skips Python tests if `pytest` is missing.
- Frontend commands are documented in `AGENTS.md`, but this repository currently has no `package.json`.
- LLM-backed flows require `OPENAI_API_KEY`; unit tests should stay offline unless explicitly marked otherwise.

## Expected Test Layers

Use these categories when adding or organizing tests:

| Layer | Purpose | Default command |
|---|---|---|
| Unit | Pure Python logic, schemas, reducers, prompt formatting, retrieval adapters with fakes | `pytest tests/core tests/orchestrator` |
| Integration | Multiple modules working together without live LLM calls | `pytest tests -m integration` |
| LLM | Tests that call external model APIs | `pytest tests -m llm` |
| Smoke | Minimal end-to-end confidence check for import, graph build, sample corpus, and CLI wiring | `./scripts/smoke.sh` once added |

Until markers are configured in `pytest.ini`, prefer clear test names and module placement over ad hoc environment checks.

## Rules for New Tests

- New behavior should include a focused test unless the change is documentation-only.
- Prefer deterministic fakes over live model calls.
- Do not require real API keys for default CI.
- Keep test fixtures small and local to `tests/` unless they are shared sample data.
- If a test needs generated local data, document how to regenerate it.
- If a failure is flaky, fix the source of nondeterminism before widening timeouts.

## Data and Fixtures

The `local_data/` directory currently contains sample corpus, FAISS index files, and prior debate outputs. Treat it carefully:

- Use `local_data/sample_corpus.json` as sample input.
- Do not assume prior files in `local_data/results/` are authoritative fixtures unless a test references them explicitly.
- Avoid committing new generated debate outputs unless they are intentionally curated examples.
- If FAISS artifacts are regenerated, include the command used and why the binary diff is needed.

## Recommended Harness Improvements

These improvements should be handled as small PRs:

1. Add a pinned Python runtime file such as `.python-version`.
2. Add `pyproject.toml` with runtime and dev dependencies.
3. Make `scripts/setup.sh`, `scripts/lint.sh`, and `scripts/test.sh` fail when required tools are unavailable.
4. Add pytest markers for `integration`, `llm`, and `slow`.
5. Add a fast `scripts/smoke.sh` that works without network access.
6. Update CI to run the same strict commands agents run locally.
