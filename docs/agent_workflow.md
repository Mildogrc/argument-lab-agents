# Agent Workflow

This document describes how coding agents should take work from request to handoff in this repository.

## Operating Model

ArgumentLab should be prepared for an Antigravity plus Codespaces workflow. Antigravity is the primary agentic coding surface; GitHub Codespaces provides the reproducible environment; Codex is used for targeted review, debugging, and second opinions.

A good task contains enough context for an agent to work in an isolated workspace, produce Antigravity artifacts, run verification, and submit a reviewable diff without needing hidden knowledge from chat.

## Before Coding

1. Read `AGENTS.md`.
2. Read the task ticket and confirm the requested outcome.
3. Start from the Codespaces/devcontainer environment when possible.
4. Inspect the relevant source and tests before editing.
5. Summarize the intended approach in the task thread or Antigravity plan artifact.
6. Keep the change small unless the ticket explicitly asks for a larger refactor.

## During Coding

- Prefer existing project patterns over new abstractions.
- Keep prompts in `src/argument_lab/core/prompts.py` or `src/argument_lab/core/eval_prompts.py`.
- Keep state-shape changes aligned with `src/argument_lab/core/state.py` and `src/argument_lab/core/models.py`.
- Do not add dependencies without explaining why the existing stack cannot solve the problem.
- Do not modify secrets, credentials, deployment settings, or generated local outputs unless the task asks for it.
- Update docs when behavior, commands, architecture, or setup changes.

## Verification

Run the narrowest useful check while developing, then run the full check before handoff:

```bash
./scripts/verify.sh
```

If verification fails:

1. Fix lint/format failures first.
2. Fix test failures next.
3. Re-run the failing command.
4. Re-run full verification before final handoff.

Known verification limitations are tracked in `docs/testing.md`.

## Handoff Format

Every agent handoff should include:

- What changed.
- Why it changed.
- Antigravity artifacts produced, if applicable.
- Tests or checks run.
- Any checks that could not be run.
- Residual risks or follow-up tasks.

## Ticket Readiness Checklist

A ticket is ready for agent work when it has:

- A concrete problem statement.
- Expected behavior or acceptance criteria.
- Relevant files, commands, logs, or reproduction steps.
- Explicit scope boundaries.
- A verification command.
- Expected artifact types, such as plan, diff, test report, or browser recording.
- Notes about API keys, network access, or data requirements.

## PR Readiness Checklist

A PR is ready for review when:

- The diff is small enough to review.
- New behavior is covered by tests or a clear reason is given.
- `./scripts/verify.sh` has passed, or the failure is documented.
- Documentation was updated when user-facing behavior or commands changed.
- No secrets or private local paths were added.
- Generated artifacts were avoided unless explicitly needed.

## Restricted Areas

Agents should treat these as sensitive:

- `.env` and any credential files.
- Deployment settings and production credentials.
- Large generated files in `local_data/`.
- Binary index artifacts unless the ticket specifically covers retrieval fixture updates.
- Git history rewrites or destructive cleanup commands.

## Good First Agent Tasks

- Harden `scripts/lint.sh` and `scripts/test.sh` so missing tools fail.
- Add pytest markers for `integration`, `llm`, and `slow`.
- Add a no-network smoke test.
- Align `AGENTS.md` and `README.md` with the repo's current implemented scope.
- Add a pinned Python runtime and packaging metadata.
- Validate the `.devcontainer` in GitHub Codespaces.
