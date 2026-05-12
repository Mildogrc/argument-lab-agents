# Development Environment

ArgumentLab targets a low-cost agentic workflow:

1. Google Antigravity for agentic coding and visual verification.
2. GitHub Codespaces for a reproducible Linux development environment.
3. ChatGPT Plus / Codex for design review, debugging, and targeted code help.
4. OpenAI API only inside ReasonBench-style model benchmark runs or explicit LLM-backed experiments.

## GitHub Codespaces

This repository includes a devcontainer at `.devcontainer/devcontainer.json`.

When a Codespace starts, it should:

- Use Python 3.11.
- Install `requirements.txt`.
- Provide GitHub CLI.
- Enable pytest and Ruff extensions in VS Code-compatible editors.
- Set `ARGUMENT_LAB_OFFLINE_MODE=true` by default.

After the Codespace starts, run:

```bash
./scripts/verify.sh
```

If `verify.sh` fails because the current scripts are too permissive or environment-dependent, fix the harness first before assigning broad agent work.

## Google Antigravity

Use Antigravity as the primary agentic coding surface. Prefer tasks that ask the agent to produce reviewable artifacts:

- implementation plan
- task checklist
- code diff
- test output
- browser recording or screenshot when UI exists
- final handoff summary

For this repository, most work is currently backend and CLI-focused, so useful Antigravity artifacts are plans, diffs, terminal output, and test reports. Browser artifacts will matter more after the planned frontend exists.

Antigravity availability and quota terms may change. Treat the free individual preview as a good starting point, not as a permanent infrastructure guarantee.

## ChatGPT Plus / Codex

Use Codex as a second reviewer or specialist helper:

- architecture/design review
- debugging a failing test
- reviewing an Antigravity-produced diff
- writing focused tests
- checking risky refactors

Do not use Codex reviews as a substitute for running the project verification commands.

## OpenAI API Use

Default development and CI should not require live model calls.

Use `OPENAI_API_KEY` only for:

- manual debate runs
- explicit LLM integration tests
- benchmark/evaluation runs
- ReasonBench-style model-vs-model comparisons

Keep live API usage out of default unit tests.
