"""Model implementations for identifier name-fixing."""

from .base import NameFixer
from .spellcheck import SpellCheckerFixer
from .typos import TyposFixer

__all__ = [
    "NameFixer",
    "SpellCheckerFixer",
    "TyposFixer",
    "MODEL_REGISTRY",
    "make_model",
]

MODEL_REGISTRY = {
    "spellcheck": lambda **kw: SpellCheckerFixer(**kw),
    "typos": lambda **kw: TyposFixer(**kw),
}


def make_model(name: str, **kwargs):
    """Create a model by name from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}.  Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)
