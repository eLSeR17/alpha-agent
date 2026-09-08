"""Integration tests — verify defaults and wiring without network calls.

These tests validate that the project is configured correctly for Ollama
function-calling without requiring a live LLM endpoint.
"""

from __future__ import annotations

from alpha_agent.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaClient


class TestDefaultModel:
    """Ensure the default model supports Ollama function calling.

    qwen2.5:7b returns tool_calls in ``message.tool_calls`` (correct).
    qwen2.5-coder:7b returns tool calls as plain text in ``message.content``
    with an empty ``tool_calls`` array — which breaks the agent loop.
    """

    def test_default_model_is_qwen25(self) -> None:
        """OllamaClient must default to qwen2.5:7b, NOT qwen2.5-coder:7b."""
        assert DEFAULT_MODEL == "qwen2.5:7b"

    def test_default_base_url_is_docker_network(self) -> None:
        """Must point to Ollama inside the Docker network, not localhost."""
        assert DEFAULT_BASE_URL == "http://ollama:11434"

    def test_client_inherits_default_model(self) -> None:
        """A fresh OllamaClient() should use the project default model."""
        client = OllamaClient()
        assert client.model == "qwen2.5:7b"
        client.close()

    def test_client_custom_model_override(self) -> None:
        """Users can still override the model via constructor."""
        client = OllamaClient(model="llama3:8b")
        assert client.model == "llama3:8b"
        client.close()

    def test_client_custom_base_url_override(self) -> None:
        """Users can still override the base URL via constructor."""
        client = OllamaClient(base_url="http://custom-host:11434")
        assert client.base_url == "http://custom-host:11434"
        client.close()
