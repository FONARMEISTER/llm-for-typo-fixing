"""Shared tokenizer cache — avoid reloading the same HuggingFace tokenizer
across different test files."""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerFast


_cache: dict[str, Any] = {}


def _get_cached(key: str, model_name: str, **kwargs: Any) -> PreTrainedTokenizerFast:
    if key not in _cache:
        _cache[key] = AutoTokenizer.from_pretrained(model_name, **kwargs)
    return _cache[key]


def get_codebert_tokenizer() -> PreTrainedTokenizerFast:
    """CodeBERT tokenizer — loaded once across all tests."""
    return _get_cached("codebert-base", "microsoft/codebert-base")


def get_codet5_tokenizer() -> PreTrainedTokenizerFast:
    """CodeT5-small tokenizer — loaded once across all tests."""
    return _get_cached(
        "codet5-small", "Salesforce/codet5-small", additional_special_tokens=[]
    )


def get_roberta_tokenizer() -> PreTrainedTokenizerFast:
    """RoBERTa tokenizer — loaded once across all tests."""
    return _get_cached(
        "roberta-base", "roberta-base", additional_special_tokens=[]
    )
