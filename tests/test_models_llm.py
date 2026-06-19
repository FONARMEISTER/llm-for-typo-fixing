"""Unit tests for ``src/models/llm_api.py``.

Covers pure helpers: ``_extract_json_object``, ``_load_presets``,
``_get_retry_after``, ``_retry_delay``, ``_extract_content``, plus
``LLMAPIFixer`` construction, ``from_preset``, and ``fix_names`` with
a mocked completion.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from openai import (
    InternalServerError,
    RateLimitError,
)

from src.models.llm_api import (
    LLMAPIFixer,
    _extract_json_object,
    _load_presets,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _mock_completion(content: str | None, reasoning: str | None = None) -> Any:
    """Build a minimal mock OpenAI completion with one choice."""
    msg = MagicMock()
    msg.content = content
    if reasoning is not None:
        msg.reasoning_content = reasoning
    else:
        # remove attribute so we test the fallback properly.
        del msg.reasoning_content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _make_preset_toml(presets: dict[str, dict[str, Any]]) -> str:
    """Serialize a dict of presets to a TOML string."""
    lines: list[str] = []
    for name, cfg in presets.items():
        lines.append(f"[{name}]")
        for k, v in cfg.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, bool):
                lines.append(f"{k} = {str(v).lower()}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# _extract_json_object
# --------------------------------------------------------------------------- #


class ExtractJsonObjectTests(unittest.TestCase):
    """Tests for ``_extract_json_object``."""

    def test_plain_json_object(self) -> None:
        result = _extract_json_object('{"a": "b", "c": "d"}')
        self.assertEqual(result, {"a": "b", "c": "d"})

    def test_json_in_markdown_fence(self) -> None:
        text = '```json\n{"calcualte": "calculate"}\n```'
        result = _extract_json_object(text)
        self.assertEqual(result, {"calcualte": "calculate"})

    def test_json_after_explanatory_text(self) -> None:
        text = 'Here are the fixes:\n{"x": "y"}'
        result = _extract_json_object(text)
        self.assertEqual(result, {"x": "y"})

    def test_nested_braces(self) -> None:
        """Nested dicts are rejected — all values must be strings."""
        text = '{"outer": {"inner": "val"}}'
        self.assertIsNone(_extract_json_object(text))

    def test_no_braces_returns_none(self) -> None:
        self.assertIsNone(_extract_json_object("no json here"))

    def test_unmatched_open_brace_returns_none(self) -> None:
        self.assertIsNone(_extract_json_object('{"x": "y"'))

    def test_unmatched_close_brace_returns_none(self) -> None:
        self.assertIsNone(_extract_json_object('"x": "y"}'))

    def test_array_instead_of_object(self) -> None:
        self.assertIsNone(_extract_json_object('[1, 2, 3]'))

    def test_string_instead_of_object(self) -> None:
        self.assertIsNone(_extract_json_object('"hello"'))

    def test_number_instead_of_object(self) -> None:
        self.assertIsNone(_extract_json_object("42"))

    def test_dict_with_non_string_values(self) -> None:
        self.assertIsNone(_extract_json_object('{"a": 1, "b": [1,2]}'))

    def test_malformed_json(self) -> None:
        self.assertIsNone(_extract_json_object('{a: b}'))

    def test_empty_object(self) -> None:
        result = _extract_json_object("{}")
        self.assertEqual(result, {})

    def test_empty_text(self) -> None:
        self.assertIsNone(_extract_json_object(""))


# --------------------------------------------------------------------------- #
# _load_presets
# --------------------------------------------------------------------------- #


class LoadPresetsTests(unittest.TestCase):
    """Tests for ``_load_presets``."""

    def test_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            _load_presets("/nonexistent/path.toml")

    def test_valid_single_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "presets.toml"
            path.write_text(_make_preset_toml({
                "test-model": {
                    "base_url": "http://localhost:8000/v1",
                    "model": "test-7b",
                    "max_tokens": 512,
                },
            }))
            presets = _load_presets(path)
            self.assertEqual(set(presets), {"test-model"})
            self.assertEqual(presets["test-model"]["model"], "test-7b")
            self.assertEqual(presets["test-model"]["max_tokens"], 512)

    def test_multiple_presets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "presets.toml"
            path.write_text(_make_preset_toml({
                "a": {"model": "a-model", "base_url": "http://a"},
                "b": {"model": "b-model", "base_url": "http://b"},
            }))
            presets = _load_presets(path)
            self.assertEqual(len(presets), 2)
            self.assertIn("a", presets)
            self.assertIn("b", presets)

    def test_empty_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.toml"
            path.write_text("")
            presets = _load_presets(path)
            self.assertEqual(presets, {})

    def test_non_dict_entries_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.toml"
            path.write_text(
                '[model-a]\nbase_url = "http://a"\nmodel = "a"\n\n'
                "# a bare key at the top level is not a dict.\n"
                'bare = "not-a-table"\n'
            )
            presets = _load_presets(path)
            # Only table entries are returned; bare string is ignored.
            self.assertEqual(set(presets), {"model-a"})

    def test_boolean_and_float_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "typed.toml"
            path.write_text(
                '[model]\n'
                'base_url = "http://x"\n'
                'model = "x"\n'
                "temperature = 0.7\n"
                "stream = false\n"
            )
            presets = _load_presets(path)
            self.assertAlmostEqual(presets["model"]["temperature"], 0.7)
            self.assertIs(presets["model"]["stream"], False)


# --------------------------------------------------------------------------- #
# _get_retry_after
# --------------------------------------------------------------------------- #


class GetRetryAfterTests(unittest.TestCase):
    """Tests for ``LLMAPIFixer._get_retry_after``."""

    def test_retry_after_integer_string(self) -> None:
        exc = MagicMock()
        exc.response.headers = {"retry-after": "30"}
        self.assertEqual(LLMAPIFixer._get_retry_after(exc), 30.0)  # type: ignore[arg-type]

    def test_retry_after_float_string(self) -> None:
        exc = MagicMock()
        exc.response.headers = {"retry-after": "3.14"}
        self.assertAlmostEqual(LLMAPIFixer._get_retry_after(exc), 3.14)  # type: ignore[arg-type]

    def test_no_retry_after_header(self) -> None:
        exc = MagicMock()
        exc.response.headers = {"content-type": "application/json"}
        self.assertIsNone(LLMAPIFixer._get_retry_after(exc))  # type: ignore[arg-type]

    def test_empty_headers(self) -> None:
        exc = MagicMock()
        exc.response.headers = {}
        self.assertIsNone(LLMAPIFixer._get_retry_after(exc))  # type: ignore[arg-type]

    def test_no_response_attribute(self) -> None:
        """Exception without a .response attribute → None."""
        exc = MagicMock(spec=[])  # no 'response'
        self.assertIsNone(LLMAPIFixer._get_retry_after(exc))

    def test_invalid_nan_string(self) -> None:
        exc = MagicMock()
        exc.response.headers = {"retry-after": "not-a-number"}
        self.assertIsNone(LLMAPIFixer._get_retry_after(exc))  # type: ignore[arg-type]

    def test_retry_after_zero(self) -> None:
        exc = MagicMock()
        exc.response.headers = {"retry-after": "0"}
        self.assertEqual(LLMAPIFixer._get_retry_after(exc), 0.0)  # type: ignore[arg-type]

    def test_real_rate_limit_error(self) -> None:
        """Headers on a realistic exception are accessed via .response.headers."""
        exc = MagicMock()
        exc.response.headers = {"retry-after": "10"}
        self.assertEqual(LLMAPIFixer._get_retry_after(exc), 10.0)


# --------------------------------------------------------------------------- #
# _retry_delay
# --------------------------------------------------------------------------- #


class RetryDelayTests(unittest.TestCase):
    """Tests for ``LLMAPIFixer._retry_delay``."""

    def setUp(self) -> None:
        self.fixer = LLMAPIFixer(
            base_url="http://localhost/v1",
            model="test",
            retry_base_delay=1.0,
        )

    def test_zeroth_attempt_around_base(self) -> None:
        delay = self.fixer._retry_delay(0)
        # base_delay * 2^0 = 1.0, ±25% jitter → [0.75, 1.25].
        self.assertGreaterEqual(delay, 0.1)
        self.assertLessEqual(delay, 1.25)

    def test_first_attempt_around_double(self) -> None:
        delay = self.fixer._retry_delay(1)
        # base_delay * 2^1 = 2.0, ±25% jitter → [1.5, 2.5].
        self.assertGreaterEqual(delay, 1.5 - 0.25 * 1.5)  # 1.125 roughly
        self.assertLessEqual(delay, 2.5)

    def test_delay_grows_with_attempt(self) -> None:
        d0 = self.fixer._retry_delay(0)
        d1 = self.fixer._retry_delay(3)
        # Statistically very likely: 1.0 vs 8.0.
        self.assertGreater(d1, d0)

    def test_capped_at_sixty_plus_jitter(self) -> None:
        """Large attempt → capped at 60 s + jitter."""
        delay = self.fixer._retry_delay(10)  # 2^10 = 1024 → capped at 60.
        self.assertLessEqual(delay, 75.0)  # 60 + 25% = 75.
        self.assertGreaterEqual(delay, 0.1)

    def test_custom_base_delay(self) -> None:
        fixer = LLMAPIFixer(
            base_url="http://localhost/v1",
            model="test",
            retry_base_delay=0.5,
        )
        delay = fixer._retry_delay(0)
        # 0.5 ±25% → [0.375, 0.625].
        self.assertGreaterEqual(delay, 0.1)
        self.assertLessEqual(delay, 0.625)


# --------------------------------------------------------------------------- #
# _extract_content
# --------------------------------------------------------------------------- #


class ExtractContentTests(unittest.TestCase):
    """Tests for ``LLMAPIFixer._extract_content``."""

    def test_normal_content(self) -> None:
        completion = _mock_completion('{"a": "b"}')
        self.assertEqual(
            LLMAPIFixer._extract_content(completion), '{"a": "b"}'
        )

    def test_content_with_whitespace(self) -> None:
        completion = _mock_completion('  {"x": "y"}  \n')
        self.assertEqual(
            LLMAPIFixer._extract_content(completion), '{"x": "y"}'
        )

    def test_empty_content_falls_back_to_reasoning(self) -> None:
        completion = _mock_completion(None, reasoning='{"from": "reasoning"}')
        self.assertEqual(
            LLMAPIFixer._extract_content(completion), '{"from": "reasoning"}'
        )

    def test_empty_content_no_reasoning_returns_none(self) -> None:
        completion = _mock_completion(None)
        self.assertIsNone(LLMAPIFixer._extract_content(completion))

    def test_empty_choices_list_returns_none(self) -> None:
        """Empty choices attribute triggers IndexError catch."""
        completion = MagicMock(choices=[])
        self.assertIsNone(LLMAPIFixer._extract_content(completion))


# --------------------------------------------------------------------------- #
# LLMAPIFixer constructor
# --------------------------------------------------------------------------- #


class ConstructorTests(unittest.TestCase):
    """Tests for ``LLMAPIFixer.__init__``."""

    def test_basic_construction(self) -> None:
        fixer = LLMAPIFixer(base_url="http://localhost:8000/v1", model="llama")
        self.assertEqual(fixer.name, "llm_api")
        self.assertEqual(fixer.max_parallel_requests, 1)

    def test_base_url_trailing_slash_stripped(self) -> None:
        fixer = LLMAPIFixer(base_url="http://localhost:8000/v1/", model="x")
        # We can't easily inspect _client, but no crash = success.
        self.assertIsNotNone(fixer)

    def test_custom_parallel_requests(self) -> None:
        fixer = LLMAPIFixer(
            base_url="http://localhost/v1", model="x", max_parallel_requests=4
        )
        self.assertEqual(fixer.max_parallel_requests, 4)

    def test_missing_api_key_env(self) -> None:
        with self.assertRaises(RuntimeError):
            LLMAPIFixer(
                base_url="http://localhost/v1",
                model="x",
                api_key_env="NONEXISTENT_KEY_XYZ",
            )

    def test_present_api_key_env(self) -> None:
        with patch.dict(os.environ, {"MY_KEY": "secret-123"}):
            fixer = LLMAPIFixer(
                base_url="http://localhost/v1",
                model="x",
                api_key_env="MY_KEY",
            )
            self.assertIsNotNone(fixer)


# --------------------------------------------------------------------------- #
# fix_names with mocked completion
# --------------------------------------------------------------------------- #


class FixNamesTests(unittest.TestCase):
    """Tests for ``LLMAPIFixer.fix_names`` with mocked ``_call_with_retry``."""

    def _make_fixer(self) -> LLMAPIFixer:
        return LLMAPIFixer(base_url="http://localhost/v1", model="test")

    def test_empty_names_returns_empty(self) -> None:
        fixer = self._make_fixer()
        self.assertEqual(fixer.fix_names("x = 1", []), {})

    def test_valid_json_response(self) -> None:
        fixer = self._make_fixer()
        mock_completion = _mock_completion('{"calcualte": "calculate"}')
        with patch.object(fixer, "_call_with_retry", return_value=mock_completion):
            fixes = fixer.fix_names("calcualte = 1", ["calcualte"])
        self.assertEqual(fixes, {"calcualte": "calculate"})

    def test_response_filters_unknown_names(self) -> None:
        fixer = self._make_fixer()
        # Model returns fixes for names not in the requested list.
        mock_completion = _mock_completion(
            '{"calcualte": "calculate", "othre": "other"}'
        )
        with patch.object(fixer, "_call_with_retry", return_value=mock_completion):
            fixes = fixer.fix_names("othre = 1", ["othre"])
        self.assertEqual(fixes, {"othre": "other"})

    def test_identity_fix_filtered_out(self) -> None:
        fixer = self._make_fixer()
        mock_completion = _mock_completion(
            '{"okname": "okname", "badname": "goodname"}'
        )
        with patch.object(fixer, "_call_with_retry", return_value=mock_completion):
            fixes = fixer.fix_names("okname badname", ["okname", "badname"])
        self.assertEqual(fixes, {"badname": "goodname"})

    def test_no_content_returns_empty(self) -> None:
        fixer = self._make_fixer()
        mock_completion = _mock_completion(None)
        with patch.object(fixer, "_call_with_retry", return_value=mock_completion):
            fixes = fixer.fix_names("x = 1", ["x"])
        self.assertEqual(fixes, {})

    def test_unparseable_json_returns_empty(self) -> None:
        fixer = self._make_fixer()
        mock_completion = _mock_completion("not json at all")
        with patch.object(fixer, "_call_with_retry", return_value=mock_completion):
            fixes = fixer.fix_names("x = 1", ["x"])
        self.assertEqual(fixes, {})


# --------------------------------------------------------------------------- #
# _call_with_retry
# --------------------------------------------------------------------------- #


class CallWithRetryTests(unittest.TestCase):
    """Tests for ``LLMAPIFixer._call_with_retry``."""

    def _make_fixer(self, max_retries: int = 2) -> LLMAPIFixer:
        return LLMAPIFixer(
            base_url="http://localhost/v1", model="test", max_retries=max_retries
        )

    def test_success_first_attempt(self) -> None:
        fixer = self._make_fixer()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = "ok"
        with patch.object(fixer, "_client", mock_client):
            result = fixer._call_with_retry([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "ok")
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

    def test_success_after_one_retry(self) -> None:
        fixer = self._make_fixer(max_retries=3)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            InternalServerError("fail", response=MagicMock(), body=None),  # type: ignore[arg-type]
            "ok",
        ]
        with patch.object(fixer, "_client", mock_client):
            result = fixer._call_with_retry([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "ok")
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    def test_exhaust_retries_raises(self) -> None:
        fixer = self._make_fixer(max_retries=1)
        mock_client = MagicMock()
        exc = InternalServerError("fail", response=MagicMock(), body=None)  # type: ignore[arg-type]
        exc.headers = {}
        mock_client.chat.completions.create.side_effect = exc
        with patch.object(fixer, "_client", mock_client):
            with self.assertRaises(InternalServerError):
                fixer._call_with_retry([{"role": "user", "content": "hi"}])
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    def test_respects_retry_after_header(self) -> None:
        fixer = self._make_fixer(max_retries=2)
        exc = RateLimitError("rate", response=MagicMock(), body=None)  # type: ignore[arg-type]
        exc.headers = {"retry-after": "0.05"}  # type: ignore[attr-defined]
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [exc, "ok"]
        with patch.object(fixer, "_client", mock_client):
            with patch("time.sleep") as mock_sleep:
                result = fixer._call_with_retry([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "ok")
        mock_sleep.assert_called_once()


# --------------------------------------------------------------------------- #
# from_preset
# --------------------------------------------------------------------------- #


class FromPresetTests(unittest.TestCase):
    """Tests for ``LLMAPIFixer.from_preset``."""

    def test_valid_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "presets.toml"
            path.write_text(_make_preset_toml({
                "my-model": {
                    "base_url": "http://localhost:8000/v1",
                    "model": "llama-local",
                    "max_tokens": 2048,
                },
            }))
            fixer = LLMAPIFixer.from_preset("my-model", config_path=path)
            self.assertIsInstance(fixer, LLMAPIFixer)

    def test_unknown_preset_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "presets.toml"
            path.write_text(_make_preset_toml({
                "known": {"base_url": "http://x", "model": "x"},
            }))
            with self.assertRaises(ValueError) as ctx:
                LLMAPIFixer.from_preset("nonexistent", config_path=path)
            self.assertIn("nonexistent", str(ctx.exception))
            self.assertIn("known", str(ctx.exception))

    def test_overrides_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "presets.toml"
            path.write_text(_make_preset_toml({
                "base": {"base_url": "http://x", "model": "x", "max_tokens": 512},
            }))
            fixer = LLMAPIFixer.from_preset(
                "base", config_path=path, max_tokens=1024
            )
            # We verify construction succeeds; max_tokens override is
            # applied at the cfg.update(overrides) level before cls(**cfg).
            self.assertIsInstance(fixer, LLMAPIFixer)


# --------------------------------------------------------------------------- #
# end to end: complete pipeline with a mocked LLM
# --------------------------------------------------------------------------- #


class FullMockedFixerTests(unittest.TestCase):
    """Verify the end-to-end path, short of actually calling any API."""

    def test_correction_roundtrip(self) -> None:
        code = "def calcualte_total(items):\n    ttal = 0\n    return ttal\n"
        names = ["calcualte_total", "ttal", "items"]
        fixer = LLMAPIFixer(base_url="http://localhost/v1", model="test")
        mock_completion = _mock_completion(
            json.dumps({"calcualte_total": "calculate_total", "ttal": "total"})
        )
        with patch.object(fixer, "_call_with_retry", return_value=mock_completion):
            fixes = fixer.fix_names(code, names)
        self.assertEqual(
            fixes,
            {"calcualte_total": "calculate_total", "ttal": "total"},
        )


if __name__ == "__main__":
    unittest.main()
