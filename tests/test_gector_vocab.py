"""Unit tests for :class:`~src.gector.vocab.TagVocab` and tag helpers.

Tests cover:
- ``compute_char_edit_tag`` for all five operation types.
- ``apply_char_edit`` for all five operation types.
- ``TagVocab.build_char_edit`` vocabulary size.
- ``TagVocab.build_from_jsonl`` with min_freq and max_replace_tags.
- TagVocab save/load roundtrip.
- Edge cases: identical strings, multi-char diffs, MAX_CHAR_POS boundary.
- ``is_char_edit_tag`` / ``is_replace_tag`` discrimination.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from src.gector.vocab import (
    TagVocab,
    KEEP_TAG, DELETE_TAG, UNK_TAG,
    KEEP_IDX, DELETE_IDX, UNK_IDX,
    MAX_CHAR_POS, EDIT_ALPHABET,
    compute_char_edit_tag,
    apply_char_edit,
    replace_tag,
    is_replace_tag,
    is_char_edit_tag,
    replacement_token,
    _SPECIAL_TAGS,
)


# ------------------------------------------------------------------ #
# Shared helpers
# ------------------------------------------------------------------ #

def _make_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# ------------------------------------------------------------------ #
# compute_char_edit_tag
# ------------------------------------------------------------------ #

class ComputeCharEditTagTests(unittest.TestCase):
    """Tests for :func:`compute_char_edit_tag`."""

    # -- Swap --

    def test_swap_adjacent(self) -> None:
        """Adjacent character transposition → $SWAP_."""
        self.assertEqual(compute_char_edit_tag("calcluate", "calculate"), "$SWAP_4")
        self.assertEqual(compute_char_edit_tag("nubmer", "number"), "$SWAP_2")
        self.assertEqual(compute_char_edit_tag("hte", "the"), "$SWAP_0")

    def test_swap_non_adjacent(self) -> None:
        """Adjacent swap at pos 1 → $SWAP_1."""
        self.assertEqual(compute_char_edit_tag("abxde", "axbde"), "$SWAP_1")

    # -- Substitution --

    def test_sub_one_char(self) -> None:
        """Single character substitution → $SUB_.*."""
        self.assertEqual(compute_char_edit_tag("varname", "varnome"), "$SUB_4_o")
        self.assertEqual(compute_char_edit_tag("c0unt", "count"), "$SUB_1_o")

    # -- Case flip --

    def test_case_flip(self) -> None:
        """Single-case difference → $CASE_.*."""
        self.assertEqual(compute_char_edit_tag("myVar", "myvar"), "$CASE_2")

    def test_case_flip_multi_char(self) -> None:
        """Multi-case diff (>2) → no single edit."""
        self.assertIsNone(compute_char_edit_tag("MYVAR", "myvar"))  # 5 diffs.

    # -- Deletion --

    def test_delete_one_char(self) -> None:
        """One extra char → $DEL_.*."""
        self.assertEqual(compute_char_edit_tag("numbber", "number"), "$DEL_3")
        self.assertEqual(compute_char_edit_tag("ahhead", "ahead"), "$DEL_1")

    # -- Insertion --

    def test_insert_one_char(self) -> None:
        """One missing char → $INS_.*.*."""
        self.assertEqual(compute_char_edit_tag("numbr", "number"), "$INS_4_e")
        self.assertEqual(compute_char_edit_tag("teh", "tech"), "$INS_2_c")

    # -- Edge cases --

    def test_identical_strings(self) -> None:
        """Identical strings → None (no diff)."""
        self.assertIsNone(compute_char_edit_tag("abc", "abc"))

    def test_empty_strings(self) -> None:
        """Empty corrupted vs non-empty original."""
        self.assertEqual(compute_char_edit_tag("", "a"), "$INS_0_a")
        self.assertEqual(compute_char_edit_tag("a", ""), "$DEL_0")

    def test_length_diff_two(self) -> None:
        """Length difference > 1 → None."""
        self.assertIsNone(compute_char_edit_tag("abc", "abcde"))
        self.assertIsNone(compute_char_edit_tag("abcde", "abc"))

    def test_position_at_max_char_pos(self) -> None:
        """Position exactly at MAX_CHAR_POS → returns None."""
        long_c = "x" * (MAX_CHAR_POS + 1) + "ab"
        long_o = "x" * (MAX_CHAR_POS + 1) + "ba"
        self.assertIsNone(compute_char_edit_tag(long_c, long_o),
                          "Swap at MAX_CHAR_POS+1 should be out of range.")

    def test_sub_char_not_in_alphabet(self) -> None:
        """Substitution char check looks at *original* char (the target), not corrupted."""
        # 'ab$' → 'abc': original[2] = 'c' is in EDIT_ALPHABET → valid SUB tag.
        self.assertEqual(compute_char_edit_tag("ab$", "abc"), "$SUB_2_c")

    def test_ins_char_not_in_alphabet(self) -> None:
        """Insertion char not in EDIT_ALPHABET → None."""
        self.assertIsNone(compute_char_edit_tag("abc", "ab$"))


# ------------------------------------------------------------------ #
# apply_char_edit
# ------------------------------------------------------------------ #

class ApplyCharEditTests(unittest.TestCase):
    """Tests for :func:`apply_char_edit`."""

    def test_swap(self) -> None:
        self.assertEqual(apply_char_edit("calcluate", "$SWAP_4"), "calculate")
        self.assertEqual(apply_char_edit("nubmer", "$SWAP_2"), "number")

    def test_swap_out_of_range(self) -> None:
        self.assertIsNone(apply_char_edit("abc", "$SWAP_2"), "pos 2 + 1 >= len 3")
        self.assertIsNone(apply_char_edit("ab", "$SWAP_1"), "pos 1 + 1 >= len 2")

    def test_del_char(self) -> None:
        self.assertEqual(apply_char_edit("numbber", "$DEL_4"), "number")
        self.assertEqual(apply_char_edit("ahhead", "$DEL_1"), "ahead")

    def test_del_out_of_range(self) -> None:
        self.assertIsNone(apply_char_edit("ab", "$DEL_2"))
        self.assertIsNone(apply_char_edit("", "$DEL_0"))

    def test_ins_char(self) -> None:
        self.assertEqual(apply_char_edit("numbr", "$INS_4_e"), "number")
        self.assertEqual(apply_char_edit("teh", "$INS_2_c"), "tech")
        # Insert at position 0.
        self.assertEqual(apply_char_edit("elp", "$INS_0_h"), "help")
        # Insert at end.
        self.assertEqual(apply_char_edit("hel", "$INS_3_p"), "help")

    def test_ins_out_of_range(self) -> None:
        self.assertIsNone(apply_char_edit("ab", "$INS_5_x"))

    def test_sub_char(self) -> None:
        self.assertEqual(apply_char_edit("varname", "$SUB_4_o"), "varnome")
        self.assertEqual(apply_char_edit("c0unt", "$SUB_1_o"), "count")

    def test_sub_out_of_range(self) -> None:
        self.assertIsNone(apply_char_edit("ab", "$SUB_2_x"))

    def test_case_flip(self) -> None:
        self.assertEqual(apply_char_edit("myVar", "$CASE_2"), "myvar")
        self.assertEqual(apply_char_edit("hello", "$CASE_0"), "Hello")

    def test_case_out_of_range(self) -> None:
        self.assertIsNone(apply_char_edit("ab", "$CASE_2"))

    def test_malformed_tag(self) -> None:
        self.assertIsNone(apply_char_edit("abc", "$SWAP_"))
        self.assertIsNone(apply_char_edit("abc", "$SWAP_xyz"))
        self.assertIsNone(apply_char_edit("abc", "$UNKNOWN_0"))
        self.assertIsNone(apply_char_edit("abc", "not_a_tag"))

    def test_ins_tag_multi_char_suffix(self) -> None:
        """INS tag with multi-char suffix → None (our parser uses sep on '_')."""
        self.assertIsNone(apply_char_edit("abc", "$INS_0_ab"))


# ------------------------------------------------------------------ #
# Tag discrimination
# ------------------------------------------------------------------ #

class TagDiscriminationTests(unittest.TestCase):
    """Tests for :func:`is_replace_tag` and :func:`is_char_edit_tag`."""

    def test_is_replace_tag(self) -> None:
        self.assertTrue(is_replace_tag("$REPLACE_hello"))
        self.assertFalse(is_replace_tag("$KEEP"))
        self.assertFalse(is_replace_tag("$SWAP_3"))

    def test_is_char_edit_tag(self) -> None:
        self.assertTrue(is_char_edit_tag("$SWAP_3"))
        self.assertTrue(is_char_edit_tag("$DEL_7"))
        self.assertTrue(is_char_edit_tag("$INS_0_a"))
        self.assertTrue(is_char_edit_tag("$SUB_5_z"))
        self.assertTrue(is_char_edit_tag("$CASE_0"))
        self.assertFalse(is_char_edit_tag("$KEEP"))
        self.assertFalse(is_char_edit_tag("$REPLACE_hello"))

    def test_replacement_token(self) -> None:
        self.assertEqual(replacement_token("$REPLACE_hello"), "hello")
        self.assertEqual(replacement_token("$REPLACE_hello_world"), "hello_world")


# ------------------------------------------------------------------ #
# TagVocab (char-edit)
# ------------------------------------------------------------------ #

class TagVocabCharEditTests(unittest.TestCase):
    """Tests for :class:`TagVocab` in char-edit mode."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.vocab = TagVocab.build_char_edit()

    def test_special_tags_first(self) -> None:
        self.assertEqual(self.vocab.tags[:3], _SPECIAL_TAGS)
        self.assertEqual(self.vocab.tags[0], KEEP_TAG)
        self.assertEqual(self.vocab.tags[1], DELETE_TAG)
        self.assertEqual(self.vocab.tags[2], UNK_TAG)

    def test_is_char_edit(self) -> None:
        self.assertTrue(self.vocab.is_char_edit)

    def test_vocab_size_reasonable(self) -> None:
        """Char-edit vocab should have ~4000 tags."""
        self.assertGreater(self.vocab.size, 3000)
        self.assertLess(self.vocab.size, 6000)

    def test_tag2idx_keep(self) -> None:
        self.assertEqual(self.vocab.tag2idx(KEEP_TAG), KEEP_IDX)
        self.assertEqual(self.vocab.tag2idx(DELETE_TAG), DELETE_IDX)
        self.assertEqual(self.vocab.tag2idx(UNK_TAG), UNK_IDX)

    def test_tag2idx_unknown(self) -> None:
        self.assertEqual(self.vocab.tag2idx("$REPLACE_nonexistent"), UNK_IDX)
        self.assertEqual(self.vocab.tag2idx("garbage"), UNK_IDX)

    def test_idx2tag(self) -> None:
        self.assertEqual(self.vocab.idx2tag(0), KEEP_TAG)
        self.assertEqual(self.vocab.idx2tag(1), DELETE_TAG)
        self.assertEqual(self.vocab.idx2tag(2), UNK_TAG)

    def test_idx2tag_out_of_range(self) -> None:
        self.assertEqual(self.vocab.idx2tag(99999), UNK_TAG)

    def test_contains(self) -> None:
        self.assertIn(KEEP_TAG, self.vocab)
        self.assertIn("$SWAP_0", self.vocab)
        self.assertNotIn("$REPLACE_foo", self.vocab)

    def test_tag_for_edit_char_mode(self) -> None:
        idx = self.vocab.tag_for_edit("nubmer", "number")
        self.assertNotEqual(idx, UNK_IDX)
        tag = self.vocab.idx2tag(idx)
        self.assertEqual(tag, "$SWAP_2")

    def test_tag_for_edit_unk_fallback(self) -> None:
        """Three diffs → no single char edit → UNK_IDX."""
        idx = self.vocab.tag_for_edit("abc", "xyz")
        self.assertEqual(idx, UNK_IDX)

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab.txt"
            self.vocab.save(path)
            loaded = TagVocab.load(path)
            self.assertEqual(loaded.size, self.vocab.size)
            self.assertEqual(loaded.tags, self.vocab.tags)
            self.assertTrue(loaded.is_char_edit)


# ------------------------------------------------------------------ #
# TagVocab (replace mode)
# ------------------------------------------------------------------ #

class TagVocabReplaceTests(unittest.TestCase):
    """Tests for :class:`TagVocab` in replace mode."""

    def test_build_from_jsonl(self) -> None:
        data = [
            {
                "code": "def nubmer(x):\n    return x",
                "fixed": "def number(x):\n    return x",
                "has_errors": True,
                "edits": [{"corrupted_name": "nubmer", "original_name": "number"}],
            },
            {
                "code": "def calcluate(x):\n    return x",
                "fixed": "def calculate(x):\n    return x",
                "has_errors": True,
                "edits": [{"corrupted_name": "calcluate", "original_name": "calculate"}],
            },
            {
                "code": "def clean(x):\n    return x",
                "fixed": "def clean(x):\n    return x",
                "has_errors": False,
                "edits": [],
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            _make_jsonl(str(path), data)
            vocab = TagVocab.build_from_jsonl([path])
            self.assertFalse(vocab.is_char_edit)
            self.assertIn(replace_tag("number"), vocab)
            self.assertIn(replace_tag("calculate"), vocab)
            self.assertNotIn(replace_tag("clean"), vocab)  # no error samples for 'clean'.

    def test_build_with_min_freq(self) -> None:
        data = [
            {
                "code": "def nubmer(x): pass",
                "fixed": "def number(x): pass",
                "has_errors": True,
                "edits": [{"corrupted_name": "nubmer", "original_name": "number"}],
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            _make_jsonl(str(path), data)
            vocab = TagVocab.build_from_jsonl([path], min_freq=2)
            # 'number' appears only once → not in vocab.
            self.assertNotIn(replace_tag("number"), vocab)
            # But special tags always present.
            self.assertEqual(vocab.tags[:3], _SPECIAL_TAGS)

    def test_build_with_max_replace_tags(self) -> None:
        data = [
            {
                "code": f"def name{i}(x): pass",
                "fixed": f"def name{i}(x): pass",
                "has_errors": True,
                "edits": [{"corrupted_name": f"name{i}", "original_name": f"name{i}"}],
            }
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            _make_jsonl(str(path), data)
            vocab = TagVocab.build_from_jsonl([path], max_replace_tags=3)
            # Count REPLACE tags (exclude special tags).
            replace_tags = [t for t in vocab.tags if is_replace_tag(t)]
            self.assertEqual(len(replace_tags), 3)

    def test_tag_for_edit_replace_mode(self) -> None:
        data = [
            {
                "code": "def nubmer(x): pass",
                "fixed": "def number(x): pass",
                "has_errors": True,
                "edits": [{"corrupted_name": "nubmer", "original_name": "number"}],
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            _make_jsonl(str(path), data)
            vocab = TagVocab.build_from_jsonl([path])
            idx = vocab.tag_for_edit("nubmer", "number")
            self.assertNotEqual(idx, UNK_IDX)
            self.assertEqual(vocab.idx2tag(idx), "$REPLACE_number")

    def test_tag_for_edit_replace_unk(self) -> None:
        """Unknown original_name → UNK_IDX in replace mode."""
        data = [
            {
                "code": "def known(x): pass",
                "fixed": "def known(x): pass",
                "has_errors": True,
                "edits": [{"corrupted_name": "known", "original_name": "known"}],
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            _make_jsonl(str(path), data)
            vocab = TagVocab.build_from_jsonl([path])
            idx = vocab.tag_for_edit("unknown", "unknown")
            self.assertEqual(idx, UNK_IDX)

    def test_save_load_roundtrip_replace(self) -> None:
        data = [
            {
                "code": "def nubmer(x): pass",
                "fixed": "def number(x): pass",
                "has_errors": True,
                "edits": [{"corrupted_name": "nubmer", "original_name": "number"}],
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            _make_jsonl(str(path), data)
            vocab = TagVocab.build_from_jsonl([path])
            vocab_path = Path(tmpdir) / "vocab.txt"
            vocab.save(vocab_path)
            loaded = TagVocab.load(vocab_path)
            self.assertEqual(loaded.size, vocab.size)
            self.assertIn("$REPLACE_number", loaded)


# ------------------------------------------------------------------ #
# TagVocab.minimal
# ------------------------------------------------------------------ #

class TagVocabMinimalTests(unittest.TestCase):
    """Tests for :meth:`TagVocab.minimal`."""

    def test_minimal_size(self) -> None:
        v = TagVocab.minimal()
        self.assertEqual(v.size, 3)
        self.assertEqual(v.tags, _SPECIAL_TAGS)

    def test_minimal_tag_for_edit(self) -> None:
        v = TagVocab.minimal()
        self.assertFalse(v.is_char_edit)
        idx = v.tag_for_edit("nubmer", "number")
        self.assertEqual(idx, UNK_IDX)  # no REPLACE tags in minimal vocab.


if __name__ == "__main__":
    unittest.main()
