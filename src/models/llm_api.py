"""LLM-based baseline using an OpenAI-compatible chat completions API.

Uses the ``openai`` Python package with structured JSON output
(``response_format={"type": "json_object"}``).  Supports both reasoning
and classic models.  Inference presets are loaded from a TOML config file
(default: ``config/llm_presets.toml``).
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

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
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self._timeout = timeout

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
        """
        if not names:
            return {}

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            code=code.strip(),
            names=json.dumps(sorted(names)),
        )

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            response_format={"type": "json_object"},
            stream=False,
        )

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
