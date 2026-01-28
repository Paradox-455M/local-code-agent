"""Abstract LLM backend interface and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM backend."""
    
    text: str
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt text.
            model: Optional model name override.
            **kwargs: Additional backend-specific parameters.
        
        Returns:
            LLMResponse with generated text.
        """
        pass
    
    @abstractmethod
    def stream(self, prompt: str, model: Optional[str] = None, **kwargs) -> Iterator[str]:
        """
        Stream a response from the LLM.
        
        Args:
            prompt: The prompt text.
            model: Optional model name override.
            **kwargs: Additional backend-specific parameters.
        
        Yields:
            Chunks of the response text.
        """
        pass
    
    @abstractmethod
    def list_models(self) -> list[str]:
        """
        List available models.
        
        Returns:
            List of available model names.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass


class OllamaBackend(LLMBackend):
    """Ollama LLM backend implementation."""
    
    def __init__(self, host: str = "http://127.0.0.1:11434", timeout: float = 300.0):
        """
        Initialize Ollama backend.
        
        Args:
            host: Ollama server host URL.
            timeout: Request timeout in seconds.
        """
        self.host = host
        self.timeout = timeout
        self._client = None
    
    @property
    def name(self) -> str:
        return "ollama"
    
    def _get_client(self):
        """Get or create HTTP client."""
        import httpx
        if self._client is None:
            self._client = httpx.Client(base_url=self.host, timeout=self.timeout)
        return self._client
    
    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate response using Ollama /api/generate endpoint."""
        import httpx
        from core.config import config
        from core.exceptions import LLMError
        
        client = self._get_client()
        payload = {
            "model": model or config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": kwargs.get("temperature", 0.1)},
        }
        
        try:
            response = client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            reply = data.get("response")
            if reply is None:
                raise RuntimeError("Ollama response missing 'response' field")
            
            return LLMResponse(
                text=reply,
                model=model or config.model,
                tokens_used=data.get("eval_count"),
            )
        except httpx.HTTPStatusError as exc:
            # Try /api/chat as fallback
            if exc.response.status_code == 404:
                return self._generate_chat(prompt, model, **kwargs)
            raise LLMError(f"Ollama request failed: {exc}") from exc
    
    def _generate_chat(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate response using Ollama /api/chat endpoint."""
        import httpx
        from core.config import config
        from core.exceptions import LLMError
        
        client = self._get_client()
        payload = {
            "model": model or config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": kwargs.get("temperature", 0.1)},
        }
        
        try:
            response = client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            message = data.get("message") or {}
            reply = message.get("content") if isinstance(message, dict) else None
            if not reply:
                reply = data.get("response")
            if not reply:
                raise RuntimeError("Ollama chat response missing content")
            
            return LLMResponse(
                text=reply,
                model=model or config.model,
                tokens_used=data.get("eval_count"),
            )
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Ollama chat request failed: {exc}") from exc
    
    def stream(self, prompt: str, model: Optional[str] = None, **kwargs) -> Iterator[str]:
        """Stream response using Ollama."""
        import httpx
        import json
        from core.config import config
        
        client = self._get_client()
        payload = {
            "model": model or config.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": kwargs.get("temperature", 0.1)},
        }
        
        try:
            with client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="replace")
                        if isinstance(line, str) and line.startswith("data: "):
                            line = line[6:]
                        if line.strip() == "[DONE]":
                            break
                        chunk = json.loads(line)
                        text = chunk.get("response")
                        if text:
                            yield text
                        if chunk.get("done"):
                            break
                    except Exception:
                        pass
        except httpx.HTTPStatusError as exc:
            # Try /api/chat as fallback
            if exc.response.status_code == 404:
                yield from self._stream_chat(prompt, model, **kwargs)
            else:
                raise
    
    def _stream_chat(self, prompt: str, model: Optional[str] = None, **kwargs) -> Iterator[str]:
        """Stream response using Ollama /api/chat endpoint."""
        import httpx
        import json
        from core.config import config
        
        client = self._get_client()
        payload = {
            "model": model or config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {"temperature": kwargs.get("temperature", 0.1)},
        }
        
        with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if isinstance(line, str) and line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        break
                    chunk = json.loads(line)
                    message = chunk.get("message") or {}
                    text = message.get("content")
                    if text:
                        yield text
                    if chunk.get("done"):
                        break
                except Exception:
                    pass
    
    def list_models(self) -> list[str]:
        """List available Ollama models."""
        import httpx
        
        try:
            client = self._get_client()
            response = client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            return [m.get("name", "") for m in models if m.get("name")]
        except Exception:
            return []


class OpenAICompatibleBackend(LLMBackend):
    """OpenAI-compatible API backend implementation."""
    
    def __init__(self, api_key: str, base_url: str, default_model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI-compatible backend.
        
        Args:
            api_key: API key for authentication.
            base_url: Base URL for the API.
            default_model: Default model name.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self._client = None
    
    @property
    def name(self) -> str:
        return "openai-compatible"
    
    def _get_client(self):
        """Get or create HTTP client."""
        import httpx
        if self._client is None:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=300.0)
        return self._client
    
    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate response using OpenAI-compatible API."""
        import httpx
        from core.exceptions import LLMError
        
        client = self._get_client()
        payload = {
            "model": model or self.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.1),
        }
        
        try:
            response = client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("OpenAI response missing choices")
            
            message = choices[0].get("message", {})
            content = message.get("content", "")
            
            return LLMResponse(
                text=content,
                model=model or self.default_model,
                tokens_used=data.get("usage", {}).get("total_tokens"),
                finish_reason=choices[0].get("finish_reason"),
            )
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"OpenAI-compatible API request failed: {exc}") from exc
    
    def stream(self, prompt: str, model: Optional[str] = None, **kwargs) -> Iterator[str]:
        """Stream response using OpenAI-compatible API."""
        import httpx
        import json
        
        client = self._get_client()
        payload = {
            "model": model or self.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": kwargs.get("temperature", 0.1),
        }
        
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        break
                    chunk = json.loads(line)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            yield text
                except Exception:
                    pass
    
    def list_models(self) -> list[str]:
        """List available models (if API supports it)."""
        import httpx
        
        try:
            client = self._get_client()
            response = client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            return [m.get("id", "") for m in models if m.get("id")]
        except Exception:
            return [self.default_model]


# Backend registry
_backends: dict[str, LLMBackend] = {}
_default_backend: Optional[LLMBackend] = None


def register_backend(name: str, backend: LLMBackend, default: bool = False) -> None:
    """
    Register an LLM backend.
    
    Args:
        name: Backend name.
        backend: Backend instance.
        default: Whether to set as default backend.
    """
    global _default_backend
    _backends[name] = backend
    if default or _default_backend is None:
        _default_backend = backend


def get_backend(name: Optional[str] = None) -> LLMBackend:
    """
    Get an LLM backend by name, or the default backend.
    
    Args:
        name: Backend name. If None, returns default backend.
    
    Returns:
        LLMBackend instance.
    
    Raises:
        ValueError: If backend not found.
    """
    if name is None:
        if _default_backend is None:
            # Auto-register Ollama as default
            ollama = OllamaBackend()
            register_backend("ollama", ollama, default=True)
            return ollama
        return _default_backend
    
    if name not in _backends:
        raise ValueError(f"Backend '{name}' not found. Available: {list(_backends.keys())}")
    
    return _backends[name]


def list_backends() -> list[str]:
    """List registered backend names."""
    return list(_backends.keys())


# Auto-register Ollama on import
try:
    ollama_backend = OllamaBackend()
    register_backend("ollama", ollama_backend, default=True)
except Exception:
    pass
