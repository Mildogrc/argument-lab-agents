import httpx
import pytest

from argument_lab.core.model_adapters import (
    AdapterError,
    GenerationRequest,
    LlamaCppAdapter,
    OllamaAdapter,
    OpenAIChatAdapter,
)


def test_ollama_adapter_posts_chat_payload(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return httpx.Response(
            200,
            json={"message": {"content": "ollama answer"}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    adapter = OllamaAdapter(model_name="qwen2.5:3b", base_url="http://ollama")

    response = adapter.generate(
        GenerationRequest(
            prompt="Solve it",
            system="Reason carefully",
            temperature=0.1,
            max_tokens=64,
        )
    )

    assert response.text == "ollama answer"
    assert captured["url"] == "http://ollama/api/chat"
    assert captured["json"]["model"] == "qwen2.5:3b"
    assert captured["json"]["messages"][0]["role"] == "system"
    assert captured["json"]["options"]["num_predict"] == 64


def test_llama_cpp_adapter_posts_openai_compatible_payload(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "llama answer"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    adapter = LlamaCppAdapter(model_name="local", base_url="http://llama")

    response = adapter.generate(GenerationRequest(prompt="Solve it"))

    assert response.text == "llama answer"
    assert captured["url"] == "http://llama/v1/chat/completions"
    assert captured["json"]["messages"][0]["role"] == "user"
    assert captured["headers"]["Authorization"] == "Bearer not-needed"


def test_openai_chat_adapter_uses_chat_completions_client():
    class FakeMessage:
        content = '{"initial_score": 0.3, "final_score": 0.8}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

        def model_dump(self):
            return {"id": "resp_1"}

    class FakeCompletions:
        def __init__(self):
            self.payload = None

        def create(self, **payload):
            self.payload = payload
            return FakeResponse()

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    client = FakeClient()
    adapter = OpenAIChatAdapter(model_name="gpt-4o", client=client)

    response = adapter.generate(
        GenerationRequest(
            prompt="Score this",
            system="Return JSON",
            temperature=0.0,
            max_tokens=128,
        )
    )

    assert response.text == '{"initial_score": 0.3, "final_score": 0.8}'
    assert response.provider == "openai"
    assert client.chat.completions.payload["model"] == "gpt-4o"
    assert client.chat.completions.payload["messages"][0]["role"] == "system"
    assert client.chat.completions.payload["max_tokens"] == 128


def test_huggingface_adapter_is_optional():
    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        from argument_lab.core.model_adapters import HuggingFaceAdapter

        with pytest.raises(AdapterError):
            HuggingFaceAdapter("missing-model")
