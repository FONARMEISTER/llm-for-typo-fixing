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
    # GECToR: requires --gector-model <checkpoint_dir> kwarg.
    # Registered as a factory so the checkpoint path can be passed at
    # runtime without importing torch at module load time.
    "gector": lambda **kw: _make_gector(**kw),
    "llm_api": _make_llm_api,
}


def _make_gector(**kwargs):
    """Lazy factory for GECToRFixer — avoids importing torch at import time."""
    from ..gector.fixer import GECToRFixer
    model_dir = kwargs.pop("model_dir", None)
    if model_dir is None:
        raise ValueError(
            "GECToRFixer requires model_dir=<checkpoint_dir>.  "
            "Pass it via make_model('gector', model_dir='models/gector/best') "
            "or use --gector-model on the CLI."
        )
    return GECToRFixer(model_dir=model_dir, **kwargs)


def make_model(name: str, **kwargs):
    """Create a model by name from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}.  Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)
