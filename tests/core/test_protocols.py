from argument_lab.core.model_adapters import GenerationRequest, GenerationResponse
from argument_lab.core.protocols import (
    BenchmarkTask,
    ProtocolMode,
    ProtocolRunner,
)


class FakeAdapter:
    provider = "fake"

    def __init__(self, model_name: str, responses: list[str]):
        self.model_name = model_name
        self.responses = list(responses)
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        text = self.responses.pop(0)
        return GenerationResponse(
            text=text,
            model=self.model_name,
            provider=self.provider,
        )


def _task() -> BenchmarkTask:
    return BenchmarkTask(
        id="logic_001",
        category="deterministic_logic",
        prompt="A is before B. B is before C. What is first?",
    )


def test_baseline_returns_single_answer_as_initial_and_final():
    model = FakeAdapter("qwen-test", ["A is first."])
    runner = ProtocolRunner(subject_model=model)

    result = runner.run(ProtocolMode.BASELINE, _task())

    assert result.mode == ProtocolMode.BASELINE
    assert result.initial_response == "A is first."
    assert result.final_response == "A is first."
    assert [turn.phase for turn in result.turns] == ["initial_answer"]


def test_self_critique_runs_answer_critique_revision():
    model = FakeAdapter(
        "gemma-test",
        ["Initial answer", "Critique", "Revised answer"],
    )
    runner = ProtocolRunner(subject_model=model)

    result = runner.run("self_critique", _task())

    assert result.initial_response == "Initial answer"
    assert result.final_response == "Revised answer"
    assert [turn.phase for turn in result.turns] == [
        "initial_answer",
        "self_critique",
        "revision",
    ]
    assert "Initial answer" in model.requests[1].prompt
    assert "Critique" in model.requests[2].prompt


def test_debate_uses_opponent_then_subject_revision():
    subject = FakeAdapter("phi-test", ["Initial", "Final"])
    opponent = FakeAdapter("mistral-test", ["Challenge"])
    runner = ProtocolRunner(subject_model=subject, opponent_model=opponent)

    result = runner.run_debate(_task())

    assert result.initial_response == "Initial"
    assert result.final_response == "Final"
    assert [turn.role for turn in result.turns] == [
        "proponent",
        "opponent",
        "proponent",
    ]
    assert "Initial" in opponent.requests[0].prompt
    assert "Challenge" in subject.requests[1].prompt


def test_debate_judge_parses_reasoning_improvement_rate():
    subject = FakeAdapter("llama-test", ["Initial", "Final"])
    opponent = FakeAdapter("qwen-test", ["Challenge"])
    judge = FakeAdapter(
        "judge-test",
        [
            '{"initial_score": 0.4, "final_score": 0.7, '
            '"responsiveness_to_critique": 0.8, "explanation": "better"}'
        ],
    )
    runner = ProtocolRunner(
        subject_model=subject,
        opponent_model=opponent,
        judge_model=judge,
    )

    result = runner.run_debate_judge(_task())

    assert result.mode == ProtocolMode.DEBATE_JUDGE
    assert result.judge_scores is not None
    assert result.reasoning_improvement_rate == 0.3
    assert result.judge_scores.explanation == "better"
    assert result.turns[-1].role == "judge"
