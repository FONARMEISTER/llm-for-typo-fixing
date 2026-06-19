"""Model implementations for identifier name-fixing."""

from .base import NameFixer
from .llm_api import LLMAPIFixer
from .spellcheck import SpellCheckerFixer
from .typos import TyposFixer

__all__ = [
    "NameFixer",
    "LLMAPIFixer",
    "SpellCheckerFixer",
    "TyposFixer",
    "MODEL_REGISTRY",
    "make_model",
]


def _make_gector(**kw: object):
    """Lazy factory for ``gector`` model — avoids importing torch at top level."""
    from ..gector.fixer import GECToRFixer
    return GECToRFixer(**kw)  # type: ignore[arg-type]


def _make_llm_api(**kw: object) -> LLMAPIFixer:
    """Factory for ``llm_api`` model — supports ``preset`` and ``config_path``."""
    preset = kw.pop("preset", None)
    config_path = kw.pop("config_path", "config/llm_presets.toml")
    if preset is not None:
        return LLMAPIFixer.from_preset(str(preset), config_path=str(config_path), **kw)
    # Direct construction — requires base_url and model kwargs.
    return LLMAPIFixer(**kw)  # type: ignore[arg-type]


MODEL_REGISTRY = {
    "spellcheck": lambda **kw: SpellCheckerFixer(**kw),
    "typos": lambda **kw: TyposFixer(**kw),
    "gector": _make_gector,
    "llm_api": _make_llm_api,
}


def make_model(name: str, **kwargs):
    """Create a model by name from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}.  Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)