from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    system: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    model: str
    provider: str
    raw: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(Protocol):
    """Minimal interface shared by local/open model backends."""

    model_name: str
    provider: str

    def generate(self, request: GenerationRequest) -> GenerationResponse: ...


class OllamaAdapter:
    """Adapter for local Ollama chat models."""

    provider = "ollama"

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        messages = _chat_messages(request)
        options: dict[str, Any] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": options,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("message", {}).get("content", "")
        return GenerationResponse(
            text=text,
            model=self.model_name,
            provider=self.provider,
            raw=payload,
        )


class LlamaCppAdapter:
    """Adapter for a llama.cpp server exposing an OpenAI-compatible API."""

    provider = "llama.cpp"

    def __init__(
        self,
        model_name: str = "local-model",
        base_url: str = "http://localhost:8080",
        timeout: float = 120.0,
        api_key: str = "not-needed",
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": _chat_messages(request),
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return GenerationResponse(
            text=text,
            model=self.model_name,
            provider=self.provider,
            raw=body,
        )


class OpenAIChatAdapter:
    """Adapter for API-backed judge models using OpenAI's chat completions API."""

    provider = "openai"

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        timeout: float = 120.0,
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.timeout = timeout
        if client is not None:
            self._client = client
            return

        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise AdapterError(
                "OpenAIChatAdapter requires `openai`. Install project "
                "dependencies with `pip install -r requirements.txt`."
            ) from exc

        self._client = OpenAI(timeout=timeout)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": _chat_messages(request),
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        response = self._client.chat.completions.create(**payload)
        choice = response.choices[0]
        text = choice.message.content or ""

        raw: dict[str, Any]
        if hasattr(response, "model_dump"):
            raw = response.model_dump()
        else:
            raw = {"response": response}

        return GenerationResponse(
            text=text,
            model=self.model_name,
            provider=self.provider,
            raw=raw,
        )


class HuggingFaceAdapter:
    """
    Adapter for local HuggingFace text-generation pipelines.

    `transformers` is intentionally imported lazily so the project can keep
    HuggingFace support optional until local SLM runs need it.
    """

    provider = "huggingface"

    def __init__(
        self,
        model_name: str,
        task: str = "text-generation",
        pipeline_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            from transformers import pipeline
        except ModuleNotFoundError as exc:
            raise AdapterError(
                "HuggingFaceAdapter requires `transformers`. Install it only "
                "when you need local HuggingFace model execution."
            ) from exc

        self.model_name = model_name
        self._pipeline = pipeline(
            task,
            model=model_name,
            **(pipeline_kwargs or {}),
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        prompt = _plain_prompt(request)
        kwargs: dict[str, Any] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            kwargs["max_new_tokens"] = request.max_tokens

        output = self._pipeline(prompt, **kwargs)
        first = output[0] if output else {}
        text = first.get("generated_text", "")
        if text.startswith(prompt):
            text = text[len(prompt) :].strip()

        return GenerationResponse(
            text=text,
            model=self.model_name,
            provider=self.provider,
            raw={"output": output},
        )


def _chat_messages(request: GenerationRequest) -> list[dict[str, str]]:
    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})
    return messages


def _plain_prompt(request: GenerationRequest) -> str:
    if not request.system:
        return request.prompt
    return f"System:\n{request.system}\n\nUser:\n{request.prompt}"


class AdapterError(RuntimeError):
    pass
