from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Final, Iterator, TypeVar, Callable

import httpx

try:
    from .config import config
    from .exceptions import LLMError
    from .rate_limiter import get_rate_limiter
    from .security import sanitize_prompt
except ImportError:
    # Allow running as a standalone script by adding repo root to sys.path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from core.config import config
    from core.exceptions import LLMError
    try:
        from core.rate_limiter import get_rate_limiter
        from core.security import sanitize_prompt
    except ImportError:
        get_rate_limiter = None  # type: ignore
        sanitize_prompt = None  # type: ignore

OLLAMA_HOST: Final[str] = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
REQUEST_TIMEOUT: Final[float] = 300.0  # Increased timeout for longer generations
LOW_TEMPERATURE: Final[float] = 0.1
MAX_RETRIES: Final[int] = int(os.getenv("LCA_MAX_RETRIES", "3"))
RETRY_BACKOFF_BASE: Final[float] = 1.5  # Exponential backoff multiplier
RETRYABLE_STATUS_CODES: Final[set[int]] = {429, 500, 502, 503, 504}  # Retry on these HTTP status codes

T = TypeVar("T")


def _retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = MAX_RETRIES,
    backoff_base: float = RETRY_BACKOFF_BASE,
    retryable_check: Callable[[Exception], bool] | None = None,
) -> T:
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry (should be a callable that takes no arguments).
        max_retries: Maximum number of retry attempts.
        backoff_base: Base multiplier for exponential backoff.
        retryable_check: Optional function to check if an exception is retryable.
                        If None, retries on httpx.HTTPStatusError with retryable status codes.

    Returns:
        Result of the function call.

    Raises:
        Last exception if all retries are exhausted.
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except httpx.HTTPStatusError as exc:
            last_exception = exc
            # Don't retry on model missing errors (404 with model not found)
            if _is_model_missing(exc):
                raise LLMError(_missing_model_message(config.model)) from exc

            # Check if this is a retryable error
            is_retryable = (
                retryable_check(exc) if retryable_check
                else exc.response.status_code in RETRYABLE_STATUS_CODES
            )

            if not is_retryable or attempt >= max_retries:
                if attempt >= max_retries:
                    raise LLMError(
                        f"Request failed after {max_retries + 1} attempts. "
                        f"Last error: HTTP {exc.response.status_code} - {exc.response.text[:200]}"
                    ) from exc
                raise

            # Wait before retrying (exponential backoff)
            wait_time = backoff_base ** attempt
            time.sleep(wait_time)
        except httpx.TimeoutException as exc:
            last_exception = exc
            if attempt >= max_retries:
                raise LLMError(
                    f"Request timed out after {max_retries + 1} attempts. "
                    "The Ollama server may be overloaded or the model may be too slow. "
                    "Try reducing the prompt size or using a faster model."
                ) from exc
            wait_time = backoff_base ** attempt
            time.sleep(wait_time)
        except httpx.HTTPError as exc:
            last_exception = exc
            if attempt >= max_retries:
                raise LLMError(
                    f"HTTP error after {max_retries + 1} attempts: {exc}. "
                    "Check that Ollama is running and accessible."
                ) from exc
            wait_time = backoff_base ** attempt
            time.sleep(wait_time)
        except Exception as exc:
            # For non-HTTP exceptions, don't retry unless explicitly retryable
            if retryable_check and retryable_check(exc):
                last_exception = exc
                if attempt >= max_retries:
                    raise
                wait_time = backoff_base ** attempt
                time.sleep(wait_time)
            else:
                raise

    # Should never reach here, but satisfy type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed unexpectedly")


def ask(prompt: str, model: str | None = None) -> str:
    """
    Send a prompt to the local Ollama server and return the response text.

    Automatically retries on transient errors (5xx, 429) with exponential backoff.
    Includes rate limiting and prompt sanitization.

    Args:
        prompt: The prompt text to send to the LLM.
        model: Optional model name override. Uses config.model if not provided.

    Returns:
        The LLM response text.

    Raises:
        LLMError: If the request fails after all retries or encounters a non-retryable error.
    """
    # Sanitize prompt (only for actual LLM calls, not stubs)
    # Note: We sanitize here but tests might use stub functions that bypass this
    if sanitize_prompt is not None:
        try:
            prompt = sanitize_prompt(prompt)
        except ValueError as e:
            raise LLMError(f"Invalid prompt: {e}") from e
    
    # Rate limiting (only for actual LLM calls)
    # Note: Rate limiting is skipped for stub/test functions
    if get_rate_limiter is not None:
        try:
            limiter = get_rate_limiter()
            limiter.wait_if_needed()
        except RuntimeError as e:
            raise LLMError(f"Rate limit exceeded: {e}") from e
    
    def _make_request() -> str:
        with httpx.Client(base_url=OLLAMA_HOST, timeout=REQUEST_TIMEOUT) as client:
            # Prefer /api/generate; fall back to /api/chat only when the endpoint is missing.
            try:
                return _call_generate(client, prompt, model_override=model)
            except httpx.HTTPStatusError as exc:
                # If /api/generate is unavailable (older/limited Ollama builds), try /api/chat.
                if _is_missing_generate_endpoint(exc):
                    try:
                        return _call_chat(client, prompt, model_override=model)
                    except httpx.HTTPStatusError as chat_exc:
                        _raise_if_model_missing(chat_exc)
                        raise

                _raise_if_model_missing(exc)
                raise

    result = _retry_with_backoff(_make_request)
    
    # Record successful request for rate limiting
    if get_rate_limiter is not None:
        try:
            limiter = get_rate_limiter()
            limiter.record_request()
        except Exception:
            pass
    
    return result


def ask_stream(prompt: str, model: str | None = None) -> Iterator[str]:
    """
    Stream a response from the local Ollama server.

    Note: Streaming requests are not retried automatically due to their stateful nature.
    For retry logic, use ask() instead.

    Args:
        prompt: The prompt text to send to the LLM.
        model: Optional model name override. Uses config.model if not provided.

    Yields:
        Chunks of the LLM response as they arrive.

    Raises:
        LLMError: If the request fails or encounters an error.
    """
    try:
        with httpx.Client(base_url=OLLAMA_HOST, timeout=REQUEST_TIMEOUT) as client:
            try:
                yield from _call_generate_stream(client, prompt, model_override=model)
            except httpx.HTTPStatusError as exc:
                if _is_missing_generate_endpoint(exc):
                    try:
                        yield from _call_chat_stream(client, prompt, model_override=model)
                    except httpx.HTTPStatusError as chat_exc:
                        _raise_if_model_missing(chat_exc)
                        raise
                else:
                    _raise_if_model_missing(exc)
                    raise
    except httpx.TimeoutException as exc:
        raise LLMError(
            "Ollama request timed out while streaming. "
            "The model may be too slow or the server overloaded. "
            "Try using ask() instead for automatic retries."
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMError(
            f"Ollama HTTP error while streaming: {exc}. "
            "Check that Ollama is running and accessible."
        ) from exc


def _call_generate(client: httpx.Client, prompt: str, model_override: str | None) -> str:
    payload = {
        "model": model_override or config.model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": LOW_TEMPERATURE},
    }
    response = client.post("/api/generate", json=payload)
    response.raise_for_status()
    data = response.json()
    reply = data.get("response")
    if reply is None:
        raise RuntimeError("Ollama /api/generate response missing 'response' field")
    return reply


def _call_generate_stream(client: httpx.Client, prompt: str, model_override: str | None) -> Iterator[str]:
    payload = {
        "model": model_override or config.model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": LOW_TEMPERATURE},
    }
    with client.stream("POST", "/api/generate", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            try:
                import json
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


def _call_chat(client: httpx.Client, prompt: str, model_override: str | None) -> str:
    payload = {
        "model": model_override or config.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": LOW_TEMPERATURE},
    }
    response = client.post("/api/chat", json=payload)
    response.raise_for_status()
    data = response.json()
    # Chat responses typically include message.content
    message = data.get("message") or {}
    reply = message.get("content") if isinstance(message, dict) else None
    if not reply:
        reply = data.get("response")
    if not reply:
        raise RuntimeError("Ollama /api/chat response missing content")
    return reply


def _call_chat_stream(client: httpx.Client, prompt: str, model_override: str | None) -> Iterator[str]:
    payload = {
        "model": model_override or config.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"temperature": LOW_TEMPERATURE},
    }
    with client.stream("POST", "/api/chat", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            try:
                import json
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


def _is_model_missing(exc: httpx.HTTPStatusError) -> bool:
    if exc.response.status_code != 404:
        return False
    try:
        body = exc.response.json()
        message = body.get("error") or body.get("message") or ""
    except Exception:
        message = exc.response.text
    return "model" in message.lower() and "found" in message.lower()


def _is_missing_generate_endpoint(exc: httpx.HTTPStatusError) -> bool:
    """Detect when /api/generate itself is not available."""
    if exc.response.status_code != 404:
        return False
    try:
        path = exc.request.url.path
    except Exception:
        path = ""
    if not path.endswith("/api/generate"):
        return False

    try:
        body = exc.response.json()
        message = body.get("error") or body.get("message") or ""
    except Exception:
        message = exc.response.text or ""

    lower_msg = message.lower()
    # Treat as missing endpoint when message says generic "not found" without mentioning models.
    return "not found" in lower_msg and "model" not in lower_msg


def _raise_if_model_missing(exc: httpx.HTTPStatusError) -> None:
    if _is_model_missing(exc):
        raise LLMError(_missing_model_message(config.model)) from exc


def _missing_model_message(model_name: str) -> str:
    return (
        f"Model '{model_name}' is not available. "
        f"Run `ollama pull {model_name}` or set LCA_MODEL to an installed model."
    )


if __name__ == "__main__":
    print(ask("Say hello briefly in one sentence."))
