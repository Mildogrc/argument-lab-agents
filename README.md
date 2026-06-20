# ReasonBench

**Benchmarking reasoning improvement in small language models through critique and debate.**

ReasonBench evaluates how much reasoning capability can be extracted from small language models through structured critique, self-reflection, and adversarial debate. Debate is not the product; it is a stress mechanism for reasoning. The goal is to measure whether smaller local/open models can improve their answers after being challenged, revised, or judged.

## Core Research Question

Do small language models improve their reasoning quality after critique, self-reflection, or adversarial interaction?

ReasonBench replaces the question "Which model wins the debate?" with a more useful benchmark target: how much a model's reasoning improves from its initial answer to its final answer under controlled protocols.

## Main Motivation

- SLMs are increasingly capable and popular.
- Local inference makes large-scale testing cheap.
- Only judging may require a stronger LLM API call.
- This enables many trials, seeds, ablations, and model comparisons.
- The result is more interesting than another GPT/Claude benchmark.

## Benchmark Protocols

ReasonBench should support four protocol modes over the same task set:

1. **Baseline:** The model answers once with no critique or revision.
2. **Self-critique:** The model answers, critiques its own reasoning, then revises.
3. **Debate:** Two agents argue opposing views, then each revises.
4. **Debate + judge:** Debate is followed by external judge scoring and structured feedback.

Each mode should capture the initial answer, intermediate critique or interaction trace, final answer, and scoring metadata so improvement can be measured directly.

## Models

ReasonBench should be local-first, with model adapters for Ollama, llama.cpp, and HuggingFace. API-backed frontier models can still be used as optional judges or comparison baselines, but they are no longer the primary benchmark target.

Suggested model families:

- Qwen small models
- Gemma small models
- Phi models
- Llama 8B-class models
- Mistral/Ministral-class models

## Tasks

The benchmark keeps the original three task categories, adapted for SLM-scale reasoning evaluation:

- **Deterministic logic:** Constraint puzzles, symbolic consistency, and answer-verifiable reasoning.
- **Strategic reasoning:** Asymmetric games, opponent modeling, conditional plans, and risk-sensitive choices.
- **Tradeoff reasoning:** Multi-constraint decisions where the model must expose assumptions, compare options, and justify tradeoffs.

Future categories:

- contradiction detection
- uncertainty calibration
- planning under constraints
- prompt perturbation stability

## Metrics

ReasonBench should score both the first answer and the final answer, then report improvement.

Tracked metrics:

- initial reasoning quality
- final reasoning quality
- improvement delta
- contradiction rate
- consistency
- risk awareness
- responsiveness to critique
- structured JSON compliance

Main metric:

```text
Reasoning Improvement Rate = final_score - initial_score
```

The benchmark should also report per-model distributions across tasks, seeds, protocols, and prompt variants rather than relying on single-run scores.

## Cost Strategy

Most computation should run through local SLM inference, making large batches of trials, seeds, ablations, and model comparisons inexpensive. Stronger LLM judge calls are optional and should use minimal prompts, compact scoring rubrics, and cached results keyed by task, model output, protocol, and judge version.

ReasonBench should also support human or evaluator rubrics so API judging is not required. This allows fully local benchmark runs, mixed human/LLM evaluation, and reproducible score audits.

## Repository Plan

Planned project structure:

- **reasonbench-core:** orchestration, model adapters, agents, and benchmark protocols
- **reasonbench-tasks:** public benchmark tasks, task schemas, fixtures, and prompt variants
- **reasonbench-evals:** scoring rubrics, judge prompts, metric calculators, and score reports
- **reasonbench-ui:** optional visualization dashboard for protocol traces and score comparisons
- **private advanced evals:** hidden tests, adversarial tasks, and leaderboard protection

## Implementation Roadmap

1. Define shared task, response, protocol trace, and score schemas.
2. Add local model adapters for Ollama first, then llama.cpp and HuggingFace.
3. Implement baseline and self-critique runners.
4. Adapt the existing debate loop into a protocol module, keeping debate as a measurement stressor.
5. Add judge scoring with cacheable outputs and human-rubric fallback.
6. Build batch evaluation across models, seeds, tasks, and protocols.
7. Generate machine-readable reports with improvement deltas and protocol comparisons.
8. Add optional dashboard views for traces, score distributions, and failure examples.

## Definition of Done for MVP

- Run all three core task categories across at least two local SLMs.
- Compare baseline, self-critique, debate, and debate + judge protocols.
- Produce initial score, final score, and Reasoning Improvement Rate for every run.
- Track contradiction rate, consistency, risk awareness, responsiveness to critique, and JSON compliance.
- Export reproducible JSON/CSV reports suitable for README results tables and resume discussion.

## Resume Positioning

Built ReasonBench, a local-first benchmark framework for measuring reasoning improvement in small language models under critique, self-reflection, and adversarial debate.

