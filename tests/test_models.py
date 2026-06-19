"""Tests for `src.models` — model registry and `make_model()` factory."""

import unittest

from src.models.base import NameFixer
from src.models.spellcheck import SpellCheckerFixer
from src.models.typos import TyposFixer
from src.models import MODEL_REGISTRY, make_model


class ModelRegistryTests(unittest.TestCase):
    """Tests for model registry and `make_model()`."""

    def test_registry_contains_all_models(self) -> None:
        """All expected model names are registered."""
        self.assertIn("spellcheck", MODEL_REGISTRY)
        self.assertIn("typos", MODEL_REGISTRY)
        self.assertIn("gector", MODEL_REGISTRY)
        self.assertIn("llm_api", MODEL_REGISTRY)

    def test_make_model_unknown_raises(self) -> None:
        """Unknown model name raises ValueError with helpful message."""
        with self.assertRaises(ValueError) as ctx:
            make_model("no_such_model")
        self.assertIn("no_such_model", str(ctx.exception))
        self.assertIn("spellcheck", str(ctx.exception))

    def test_make_model_spellcheck(self) -> None:
        """`make_model('spellcheck')` returns SpellCheckerFixer."""
        model = make_model("spellcheck")
        self.assertIsInstance(model, SpellCheckerFixer)
        self.assertIsInstance(model, NameFixer)

    def test_make_model_typos(self) -> None:
        """`make_model('typos')` returns TyposFixer."""
        model = make_model("typos")
        self.assertIsInstance(model, TyposFixer)
        self.assertIsInstance(model, NameFixer)


if __name__ == "__main__":
    unittest.main()
