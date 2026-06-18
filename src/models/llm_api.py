"""LLM-based baseline using an OpenAI-compatible chat completions API.

Uses the ``openai`` Python package with structured JSON output
(``response_format={"type": "json_object"}``).  Supports both reasoning
and classic models.  Inference presets are loaded from a TOML config file
(default: ``config/llm_presets.toml``).
"""

from __future__ import annotations

import json
import os
import random
import time
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import (
    APIStatusError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from .base import NameFixer

# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """\
You are an expert code reviewer specialized in fixing typographical \
errors in Python identifiers (variable names, function names, class names). \
You correct spelling mistakes like "calcualte"→"calculate", "nubmer"→"number", \
"requset"→"request".  Never suggest semantic renames — only fix spelling typos.

You must respond with a JSON object."""

_USER_PROMPT_TEMPLATE = """\
Find and fix typographical errors in the identifiers of the following Python \
code.  Some identifiers may already be correct — leave those unchanged.

```python
{code}
```

Identifiers found in the code: {names}

Return a JSON object mapping each **misspelled** identifier to its \
corrected form.  Omit identifiers that are already correctly spelled.  \
Output ONLY the JSON object, nothing else.

Example of expected output:
{{"calcualte": "calculate", "nubmer": "number"}}"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _extract_json_object(text: str) -> Optional[Dict[str, str]]:
    """Try to extract a ``{str: str, ...}`` JSON object from *text*.

    The LLM may wrap the JSON in markdown fences or add explanatory text.
    This function searches for the outermost ``{...}`` and parses it.
    """
    start = text.find("{")
    if start == -1:
        return None
    # Walk forward balancing braces to find the matching close.
    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in obj.items()
    ):
        return obj  # type: ignore[return-value]
    return None


def _load_presets(config_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load inference presets from a TOML file.

    Returns a dict mapping preset names to their parameter dictionaries.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Preset config not found: {path}")
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    return {k: dict(v) for k, v in raw.items() if isinstance(v, dict)}


# --------------------------------------------------------------------------- #
# LLM API model
# --------------------------------------------------------------------------- #


class LLMAPIFixer(NameFixer):
    """Fix identifiers by calling a local or remote LLM via OpenAI-compatible API.

    Uses the ``openai`` Python package with ``response_format={"type":
    "json_object"}`` to request structured JSON output.

    Parameters
    ----------
    base_url:
        API base URL (e.g. ``http://127.0.0.1:12434/v1``).
    model:
        Model name as known to the API (e.g. ``"gemma4-26b-Q6K"``).
    api_key_env:
        Environment variable name for the API key, or ``None`` for no auth.
    max_tokens:
        Maximum completion tokens (must cover reasoning + JSON output).
    temperature:
        Sampling temperature (0 = deterministic).
    system_prompt:
        System-level instruction string.
    max_parallel_requests:
        Maximum concurrent API requests.  Set to > 1 for cloud APIs
        (OpenAI, etc.) to saturate rate limits.  Default 1 (serial).
    max_retries:
        Maximum retry attempts for transient errors (429, 5xx, timeout,
        connection).  0 = no retry.  Default 5.
    retry_base_delay:
        Base delay in seconds for exponential backoff on retry.
        Default 1.0 (1 s, 2 s, 4 s, 8 s, 16 s).
    timeout:
        HTTP request timeout in seconds.
    """

    name = "llm_api"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        max_parallel_requests: int = 1,
        max_retries: int = 5,
        retry_base_delay: float = 1.0,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self.max_parallel_requests = max_parallel_requests

        api_key = None
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"API key environment variable {api_key_env!r} is not set."
                )

        self._client = OpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key or "not-needed",
            timeout=timeout,
        )

    # ------------------------------------------------------------------ #
    # NameFixer interface
    # ------------------------------------------------------------------ #

    def fix_names(self, code: str, names: List[str]) -> Dict[str, str]:
        """Ask the LLM to correct typos in *names* given *code* context.

        Only entries for identifiers the model considers misspelled are
        returned.  Names not in the result are treated as already correct.

        Rate-limit (429), server (5xx), timeout, and connection errors
        are retried with exponential backoff (up to ``self._max_retries``
        attempts).
        """
        if not names:
            return {}

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            code=code.strip(),
            names=json.dumps(sorted(names)),
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = self._call_with_retry(messages)

        content = self._extract_content(completion)
        if content is None:
            return {}

        result = _extract_json_object(content)
        if result is None:
            return {}

        # Filter: keep only fixes for names that actually appear in the
        # requested list, and skip no-op corrections.
        filtered: Dict[str, str] = {}
        for corrupted, fixed in result.items():
            if corrupted not in names:
                continue
            if corrupted == fixed:
                continue
            filtered[corrupted] = fixed
        return filtered

    # ------------------------------------------------------------------ #
    # Retry logic
    # ------------------------------------------------------------------ #

    # Errors that warrant a retry (transient).
    _RETRYABLE: tuple = (
        RateLimitError,         # 429
        InternalServerError,    # 500+
        APITimeoutError,        # timeout
        APIConnectionError,     # connection refused / DNS
    )

    def _call_with_retry(self, messages: List[Dict[str, str]]) -> Any:
        """Call the chat completions API with exponential backoff retry."""
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
                    stream=False,
                )
            except self._RETRYABLE as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    raise
                delay = self._retry_delay(attempt)
                # If the server sent a Retry-After header, use it instead.
                retry_after = self._get_retry_after(exc)
                if retry_after is not None:
                    delay = max(delay, retry_after)
                time.sleep(delay)
        # Should be unreachable — the loop always raises or returns.
        if last_exc is not None:
            raise last_exc

    def _retry_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter.

        Delay = base_delay * 2^attempt, capped at 60 s, with ±25 % jitter.
        """
        base = self._retry_base_delay * (2 ** attempt)
        capped = min(base, 60.0)
        jitter = capped * random.uniform(-0.25, 0.25)
        return max(0.1, capped + jitter)

    @staticmethod
    def _get_retry_after(exc: APIStatusError) -> Optional[float]:
        """Extract ``Retry-After`` header value if present, in seconds."""
        try:
            headers = getattr(exc, "headers", None) or {}
            value = headers.get("retry-after")
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_content(completion: Any) -> Optional[str]:
        """Extract the assistant text from an OpenAI chat completion.

        For reasoning models, the final answer is in ``message.content``
        (``reasoning_content`` holds the chain-of-thought).  Falls back to
        ``reasoning_content`` if ``content`` is empty.
        """
        try:
            choice = completion.choices[0]
            msg = choice.message
        except (IndexError, AttributeError):
            return None
        # Primary: message.content (final answer).
        if msg.content:
            return msg.content.strip()
        # Fallback: reasoning_content (chain-of-thought, may contain answer).
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            return str(reasoning).strip()
        return None

    # ------------------------------------------------------------------ #
    # Factory helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        config_path: str | Path = "config/llm_presets.toml",
        **overrides: Any,
    ) -> "LLMAPIFixer":
        """Create an instance from a named preset in the TOML config file.

        Any keyword arguments in *overrides* take precedence over the preset
        values (useful for one-off experiments).
        """
        presets = _load_presets(config_path)
        if preset_name not in presets:
            available = ", ".join(sorted(presets.keys()))
            raise ValueError(
                f"Unknown preset {preset_name!r}.  Available: {available}"
            )
        cfg: Dict[str, Any] = dict(presets[preset_name])
        cfg.update(overrides)
        return cls(**cfg)
