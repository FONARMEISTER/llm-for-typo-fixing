"""Tests for :class:`src.models.typos.TyposFixer`."""

from __future__ import annotations

from src.models.typos import TyposFixer
from src.identifier_utils import extract_renameable_identifiers


def test_typos_creates():
    fixer = TyposFixer()
    assert fixer.name == "typos"


def test_typos_fixes_calcluate():
    fixer = TyposFixer()
    code = "def calcluate(x):\n    return x + 1\n"
    names = list(extract_renameable_identifiers(code).keys())
    fixes = fixer.fix_names(code, names)
    assert fixes == {"calcluate": "calculate"}


def test_typos_fixes_multiple():
    fixer = TyposFixer()
    code = """def implment_logic(x):
    reslut = x + 1
    return reslut
"""
    names = list(extract_renameable_identifiers(code).keys())
    fixes = fixer.fix_names(code, names)
    assert fixes.get("implment_logic") == "implement_logic"
    assert fixes.get("reslut") == "result"


def test_typos_noop_on_correct():
    fixer = TyposFixer()
    code = "def function(x):\n    return x + 1\n"
    names = list(extract_renameable_identifiers(code).keys())
    fixes = fixer.fix_names(code, names)
    assert fixes == {}


def test_typos_handles_empty():
    fixer = TyposFixer()
    fixes = fixer.fix_names("x = 1", [])
    assert fixes == {}


def test_typos_fixes_camelcase_identifier():
    fixer = TyposFixer()
    code = "class MyClas:\n    pass\n"
    names = list(extract_renameable_identifiers(code).keys())
    fixes = fixer.fix_names(code, names)
    assert fixes.get("MyClas") == "MyClass"


def test_typos_fixes_with_correct_context():
    """typos uses surrounding code — neighbouring correct names shouldn't confuse it."""
    fixer = TyposFixer()
    code = """class Vehicle:
    def intialize_engine(self):
        pass
"""
    names = list(extract_renameable_identifiers(code).keys())
    fixes = fixer.fix_names(code, names)
    assert fixes == {"intialize_engine": "initialize_engine"}


def test_typos_fixes_method_name():
    """typos detects and fixes typos in method definitions."""
    fixer = TyposFixer()
    code = "def lenght(self):\n    return 42\n"
    names = list(extract_renameable_identifiers(code).keys())
    fixes = fixer.fix_names(code, names)
    assert fixes == {"lenght": "length"}
