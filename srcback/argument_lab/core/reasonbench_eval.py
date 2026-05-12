import os
from typing import Any, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from argument_lab.core.reasonbench_models import (
    TaskType,
    ReasonBenchResponse,
    Task1Evaluation,
    Task2Evaluation,
    Task3Evaluation,
)

REASONBENCH_JUDGE_SYSTEM = """You are a strict, expert judge evaluating an adversarial debate between two AI models on a complex reasoning task.
Your goal is to evaluate their reasoning quality based on a specific scoring rubric, not just correctness. 
For each model, score them on the provided dimensions from 0 to 2.
- 0 indicates failure or ignoring the dimension.
- 1 indicates partial success or minor issues.
- 2 indicates mastery or full integration.

You must return your evaluation strictly in the requested JSON format."""

REASONBENCH_JUDGE_USER = """Evaluate the models based on their performance in the current round. 

Task Type: {task_type}
Problem Description: {problem}
Current Round: {current_round}

Prior context (for measuring responsiveness):
{prior_context}

--- Proponent Response ---
{proponent_response}

--- Opponent Response ---
{opponent_response}

Provide your scores and a brief explanation."""

_judge_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
)


def evaluate_reasonbench_round(
    task_type: TaskType,
    problem: str,
    current_round: int,
    proponent_response: ReasonBenchResponse,
    opponent_response: ReasonBenchResponse,
    prior_context: str = "None (Round 1)",
    llm: Any = _judge_llm,
) -> Union[Task1Evaluation, Task2Evaluation, Task3Evaluation]:

    # Select the correct output schema
    if task_type == TaskType.TASK_1_LOGIC:
        schema = Task1Evaluation
    elif task_type == TaskType.TASK_2_STRATEGY:
        schema = Task2Evaluation
    elif task_type == TaskType.TASK_3_TRADEOFF:
        schema = Task3Evaluation
    else:
        raise ValueError(f"Unknown TaskType: {task_type}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REASONBENCH_JUDGE_SYSTEM),
            ("user", REASONBENCH_JUDGE_USER),
        ]
    )

    structured_llm = llm.with_structured_output(schema)
    chain = prompt | structured_llm

    result = chain.invoke(
        {
            "task_type": task_type.value,
            "problem": problem,
            "current_round": current_round,
            "prior_context": prior_context,
            "proponent_response": proponent_response.model_dump_json(indent=2),
            "opponent_response": opponent_response.model_dump_json(indent=2),
        }
    )

    return result
