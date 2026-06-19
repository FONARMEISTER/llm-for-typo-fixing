"""Tests for :class:`src.models.spellcheck.SpellCheckerFixer`."""

from __future__ import annotations

from src.models.spellcheck import SpellCheckerFixer


def test_spellcheck_creates():
    fixer = SpellCheckerFixer()
    assert fixer.name == "spellcheck"


def test_spellcheck_fixes_simple_typo_in_identifier():
    fixer = SpellCheckerFixer()
    fixes = fixer.fix_names("x = 1", ["abailable"])  # missing 'v' in 'available'.
    assert fixes.get("abailable") == "available"


def test_spellcheck_preserves_case():
    fixer = SpellCheckerFixer()
    fixes = fixer.fix_names("", ["myVaraible"])
    assert "myVaraible" in fixes
    assert fixes["myVaraible"] in ("myVariable", "myVariably")  # may vary.


def test_spellcheck_skips_short():
    fixer = SpellCheckerFixer()
    fixes = fixer.fix_names("", ["ab", "cd"])
    assert fixes == {}


def test_spellcheck_noop_on_correct():
    fixer = SpellCheckerFixer()
    fixes = fixer.fix_names("", ["function", "variable"])
    assert fixes == {}


def test_spellcheck_handles_empty():
    fixer = SpellCheckerFixer()
    fixes = fixer.fix_names("", [])
    assert fixes == {}


def test_spellcheck_handles_empty_name_string():
    """Empty identifier string is returned as-is (no words to spellcheck)."""
    fixer = SpellCheckerFixer()
    fixes = fixer.fix_names("", [""])
    assert fixes == {}


def test_spellcheck_skips_non_alpha_word():
    """Words containing non-alpha chars (underscores, digits) are left untouched."""
    fixer = SpellCheckerFixer()
    fixes = fixer.fix_names("", ["_leading", "file2", "my_var"])
    # All words contain non-alpha chars → no corrections.
    assert fixes == {}


def test_spellcheck_unknown_word_no_suggestion():
    """Word with no known suggestion → returned as-is (no fix)."""
    fixer = SpellCheckerFixer()
    fixes = fixer.fix_names("", ["zzzzzzzzzz"])
    assert fixes == {}
