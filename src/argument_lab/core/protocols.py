from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from argument_lab.core.model_adapters import (
    GenerationRequest,
    GenerationResponse,
    ModelAdapter,
)


class ProtocolMode(str, Enum):
    BASELINE = "baseline"
    SELF_CRITIQUE = "self_critique"
    DEBATE = "debate"
    DEBATE_JUDGE = "debate_judge"


class BenchmarkTask(BaseModel):
    id: str
    category: str
    prompt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProtocolTurn(BaseModel):
    role: str
    phase: str
    model: str
    provider: str
    content: str


class JudgeScores(BaseModel):
    initial_score: float | None = Field(default=None, ge=0.0, le=1.0)
    final_score: float | None = Field(default=None, ge=0.0, le=1.0)
    contradiction_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    consistency: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_awareness: float | None = Field(default=None, ge=0.0, le=1.0)
    responsiveness_to_critique: float | None = Field(default=None, ge=0.0, le=1.0)
    json_compliance: float | None = Field(default=None, ge=0.0, le=1.0)
    explanation: str = ""

    @property
    def reasoning_improvement_rate(self) -> float | None:
        if self.initial_score is None or self.final_score is None:
            return None
        return round(self.final_score - self.initial_score, 4)


class ProtocolResult(BaseModel):
    task_id: str
    mode: ProtocolMode
    subject_model: str
    initial_response: str
    final_response: str
    turns: list[ProtocolTurn]
    judge_scores: JudgeScores | None = None

    @property
    def reasoning_improvement_rate(self) -> float | None:
        if self.judge_scores is None:
            return None
        return self.judge_scores.reasoning_improvement_rate


class ProtocolRunner:
    """
    Runs ReasonBench protocols over a single task and model adapter.

    Debate is treated as a stress mechanism: the benchmark target remains the
    subject model's initial-to-final reasoning change.
    """

    def __init__(
        self,
        subject_model: ModelAdapter,
        opponent_model: ModelAdapter | None = None,
        judge_model: ModelAdapter | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> None:
        self.subject_model = subject_model
        self.opponent_model = opponent_model or subject_model
        self.judge_model = judge_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, mode: ProtocolMode | str, task: BenchmarkTask) -> ProtocolResult:
        mode = ProtocolMode(mode)
        if mode == ProtocolMode.BASELINE:
            return self.run_baseline(task)
        if mode == ProtocolMode.SELF_CRITIQUE:
            return self.run_self_critique(task)
        if mode == ProtocolMode.DEBATE:
            return self.run_debate(task)
        if mode == ProtocolMode.DEBATE_JUDGE:
            return self.run_debate_judge(task)
        raise ValueError(f"Unsupported protocol mode: {mode}")

    def run_baseline(self, task: BenchmarkTask) -> ProtocolResult:
        answer = self._generate(
            self.subject_model,
            role="subject",
            phase="initial_answer",
            system=_SYSTEM_REASONER,
            prompt=_answer_prompt(task),
        )
        return ProtocolResult(
            task_id=task.id,
            mode=ProtocolMode.BASELINE,
            subject_model=self.subject_model.model_name,
            initial_response=answer.text,
            final_response=answer.text,
            turns=[_turn("subject", "initial_answer", answer)],
        )

    def run_self_critique(self, task: BenchmarkTask) -> ProtocolResult:
        initial = self._generate(
            self.subject_model,
            role="subject",
            phase="initial_answer",
            system=_SYSTEM_REASONER,
            prompt=_answer_prompt(task),
        )
        critique = self._generate(
            self.subject_model,
            role="subject",
            phase="self_critique",
            system=_SYSTEM_CRITIC,
            prompt=_self_critique_prompt(task, initial.text),
        )
        final = self._generate(
            self.subject_model,
            role="subject",
            phase="revision",
            system=_SYSTEM_REASONER,
            prompt=_revision_prompt(task, initial.text, critique.text),
        )
        return ProtocolResult(
            task_id=task.id,
            mode=ProtocolMode.SELF_CRITIQUE,
            subject_model=self.subject_model.model_name,
            initial_response=initial.text,
            final_response=final.text,
            turns=[
                _turn("subject", "initial_answer", initial),
                _turn("subject", "self_critique", critique),
                _turn("subject", "revision", final),
            ],
        )

    def run_debate(self, task: BenchmarkTask) -> ProtocolResult:
        initial = self._generate(
            self.subject_model,
            role="proponent",
            phase="initial_answer",
            system=_SYSTEM_REASONER,
            prompt=_debate_answer_prompt(task, stance="proponent"),
        )
        opponent = self._generate(
            self.opponent_model,
            role="opponent",
            phase="opposing_argument",
            system=_SYSTEM_DEBATER,
            prompt=_opponent_prompt(task, initial.text),
        )
        final = self._generate(
            self.subject_model,
            role="proponent",
            phase="revision_after_debate",
            system=_SYSTEM_REASONER,
            prompt=_debate_revision_prompt(task, initial.text, opponent.text),
        )
        return ProtocolResult(
            task_id=task.id,
            mode=ProtocolMode.DEBATE,
            subject_model=self.subject_model.model_name,
            initial_response=initial.text,
            final_response=final.text,
            turns=[
                _turn("proponent", "initial_answer", initial),
                _turn("opponent", "opposing_argument", opponent),
                _turn("proponent", "revision_after_debate", final),
            ],
        )

    def run_debate_judge(self, task: BenchmarkTask) -> ProtocolResult:
        result = self.run_debate(task)
        result.mode = ProtocolMode.DEBATE_JUDGE

        if self.judge_model is None:
            raise ProtocolError("debate_judge requires a judge_model adapter.")

        judge = self._generate(
            self.judge_model,
            role="judge",
            phase="judge_scoring",
            system=_SYSTEM_JUDGE,
            prompt=_judge_prompt(task, result.initial_response, result.final_response),
        )
        result.turns.append(_turn("judge", "judge_scoring", judge))
        result.judge_scores = _parse_judge_scores(judge.text)
        return result

    def _generate(
        self,
        adapter: ModelAdapter,
        *,
        role: str,
        phase: str,
        system: str,
        prompt: str,
    ) -> GenerationResponse:
        try:
            return adapter.generate(
                GenerationRequest(
                    prompt=prompt,
                    system=system,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            )
        except Exception as exc:
            raise ProtocolError(
                f"{role} model failed during {phase} for protocol execution: {exc}"
            ) from exc


def _turn(role: str, phase: str, response: GenerationResponse) -> ProtocolTurn:
    return ProtocolTurn(
        role=role,
        phase=phase,
        model=response.model,
        provider=response.provider,
        content=response.text,
    )


def _parse_judge_scores(text: str) -> JudgeScores:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return JudgeScores(explanation=text)

    allowed = set(JudgeScores.model_fields)
    filtered = {key: value for key, value in payload.items() if key in allowed}
    return JudgeScores(**filtered)


def _answer_prompt(task: BenchmarkTask) -> str:
    return (
        f"Task category: {task.category}\n"
        f"Task:\n{task.prompt}\n\n"
        "Answer with clear reasoning, assumptions, risks, and final answer."
    )


def _debate_answer_prompt(task: BenchmarkTask, stance: str) -> str:
    return (
        f"Task category: {task.category}\n"
        f"Task:\n{task.prompt}\n\n"
        f"You are the {stance}. Give your strongest initial reasoning."
    )


def _self_critique_prompt(task: BenchmarkTask, initial: str) -> str:
    return (
        f"Task:\n{task.prompt}\n\n"
        f"Initial answer:\n{initial}\n\n"
        "Critique the reasoning. Focus on contradictions, missed constraints, "
        "unsupported assumptions, risk awareness, and JSON/structure issues."
    )


def _revision_prompt(task: BenchmarkTask, initial: str, critique: str) -> str:
    return (
        f"Task:\n{task.prompt}\n\n"
        f"Initial answer:\n{initial}\n\n"
        f"Critique:\n{critique}\n\n"
        "Revise the answer. Keep what was correct, fix what was weak, and make "
        "the final reasoning explicit."
    )


def _opponent_prompt(task: BenchmarkTask, initial: str) -> str:
    return (
        f"Task:\n{task.prompt}\n\n"
        f"Proponent initial answer:\n{initial}\n\n"
        "Argue the opposing view. Identify weak reasoning, contradictions, "
        "missed constraints, and risks."
    )


def _debate_revision_prompt(task: BenchmarkTask, initial: str, opponent: str) -> str:
    return (
        f"Task:\n{task.prompt}\n\n"
        f"Initial answer:\n{initial}\n\n"
        f"Opposing argument:\n{opponent}\n\n"
        "Revise your reasoning after the adversarial challenge. Address the "
        "strongest objections directly."
    )


def _judge_prompt(task: BenchmarkTask, initial: str, final: str) -> str:
    return (
        f"Task category: {task.category}\n"
        f"Task:\n{task.prompt}\n\n"
        f"Initial answer:\n{initial}\n\n"
        f"Final answer:\n{final}\n\n"
        "Return JSON with these optional 0.0-1.0 fields: initial_score, "
        "final_score, contradiction_rate, consistency, risk_awareness, "
        "responsiveness_to_critique, json_compliance, and explanation."
    )


_SYSTEM_REASONER = (
    "You are a small language model being evaluated for reasoning quality. "
    "Expose concise reasoning, avoid unsupported claims, and revise when evidence "
    "or critique warrants it."
)

_SYSTEM_CRITIC = (
    "You are a strict self-critique pass. Find concrete reasoning failures and "
    "actionable revisions. Do not praise the answer unless it helps identify "
    "what should be preserved."
)

_SYSTEM_DEBATER = (
    "You are an adversarial debate agent. Debate is a stress test for reasoning, "
    "not a performance. Challenge weak assumptions and missed constraints."
)

_SYSTEM_JUDGE = (
    "You are a strong external LLM judge scoring reasoning improvement in small "
    "language models. Do not join the debate. Be strict, concise, and return "
    "only valid JSON."
)


class ProtocolError(RuntimeError):
    pass
