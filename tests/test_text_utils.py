"""Tests for `src.text_utils` — identifier splitting and reassembly."""

import unittest

from src.text_utils import _restore_case, reassemble_identifier, split_identifier


class SplitIdentifierTests(unittest.TestCase):
    """Tests for `split_identifier()`."""

    def test_simple_snake_case(self) -> None:
        """snake_case → constituent words."""
        self.assertEqual(split_identifier("my_variable"), ["my", "variable"])

    def test_camel_case(self) -> None:
        """CamelCase → constituent words."""
        self.assertEqual(split_identifier("myVariable"), ["my", "Variable"])

    def test_pascal_case(self) -> None:
        """PascalCase → constituent words."""
        self.assertEqual(split_identifier("HttpResponse"), ["Http", "Response"])

    def test_mixed_camel_and_snake(self) -> None:
        """camelCase_snakeCase → mixed split."""
        self.assertEqual(
            split_identifier("camelCase_snakeCase"),
            ["camel", "Case", "snake", "Case"],
        )

    def test_leading_single_underscore(self) -> None:
        """Leading underscore is a separate word."""
        self.assertEqual(split_identifier("_private"), ["_", "private"])

    def test_leading_double_underscore(self) -> None:
        """Double leading underscore: one "_" word per underscore char."""
        self.assertEqual(split_identifier("__hidden"), ["_", "_", "hidden"])

    def test_empty_string(self) -> None:
        """Empty identifier → empty list."""
        self.assertEqual(split_identifier(""), [])

    def test_only_underscore(self) -> None:
        """A single underscore is the entire identifier."""
        self.assertEqual(split_identifier("_"), ["_"])

    def test_only_double_underscore(self) -> None:
        """Double underscore only: two "_" words."""
        self.assertEqual(split_identifier("__"), ["_", "_"])

    def test_double_underscore_in_middle(self) -> None:
        """Double underscore between words: a__b preserves separator."""
        result = split_identifier("a__b")
        self.assertEqual(result, ["a", "_", "b"])

    def test_all_caps_acronym(self) -> None:
        """UPPERCASE acronym: HTTP → single word."""
        self.assertEqual(split_identifier("HTTP"), ["HTTP"])

    def test_acronym_then_word(self) -> None:
        """HTTPError → two words."""
        self.assertEqual(split_identifier("HTTPError"), ["HTTP", "Error"])

    def test_plain_lowercase_word(self) -> None:
        """A single lowercase word stays one token (no CamelCase match)."""
        self.assertEqual(split_identifier("abc"), ["abc"])

    def test_number_suffix(self) -> None:
        """Identifier ending with digits: digits are stripped (alpha-only split)."""
        self.assertEqual(split_identifier("file2"), ["file"])

    def test_triple_underscore(self) -> None:
        """Triple underscore between words."""
        result = split_identifier("a___b")
        self.assertEqual(result, ["a", "_", "_", "b"])

    def test_number_after_acronym(self) -> None:
        """Acronym followed by digit: digits stripped (alpha-only split)."""
        self.assertEqual(split_identifier("HTTP2"), ["HTTP"])


class ReassembleIdentifierTests(unittest.TestCase):
    """Tests for `reassemble_identifier()`."""

    def test_simple_reassembly(self) -> None:
        """Reassemble snake_case with corrected words."""
        result = reassemble_identifier("my_variable", ["my", "var"])
        self.assertEqual(result, "my_var")

    def test_camel_case_reassembly(self) -> None:
        """Reassemble CamelCase with corrected words."""
        result = reassemble_identifier("myVariable", ["my", "Var"])
        self.assertEqual(result, "myVar")

    def test_case_preservation_upper(self) -> None:
        """Uppercase original word → corrected word goes uppercase."""
        result = reassemble_identifier("HTTP_ERROR", ["http", "error"])
        self.assertEqual(result, "HTTP_ERROR")

    def test_case_preservation_capitalize(self) -> None:
        """Capitalized original word → corrected word gets capitalized."""
        result = reassemble_identifier("HttpError", ["http", "error"])
        self.assertEqual(result, "HttpError")

    def test_case_preservation_lower(self) -> None:
        """Lowercase original word → corrected word stays lowercase."""
        result = reassemble_identifier("my_variable", ["My", "Variable"])
        self.assertEqual(result, "my_variable")

    def test_word_count_mismatch_fallback(self) -> None:
        """If corrected word count differs, return original unchanged."""
        result = reassemble_identifier("my_variable", ["only", "one", "word"])
        # "my_variable" splits to 2 words, but we gave 3 → fallback.
        self.assertEqual(result, "my_variable")
        # Correct count mismatch: too few.
        result2 = reassemble_identifier("my_variable", ["one"])
        self.assertEqual(result2, "my_variable")

    def test_leading_underscore_preserved(self) -> None:
        """Leading underscore is preserved in reassembly."""
        result = reassemble_identifier("_private", ["_", "secret"])
        self.assertEqual(result, "_secret")

    def test_leading_double_underscore_reassembly(self) -> None:
        """Reassembling __private maps two "_" words correctly."""
        # With split_identifier now returning ["_", "_", "private"],
        # reassembly must handle the leading double underscore.
        result = reassemble_identifier("__private", ["_", "_", "secret"])
        self.assertEqual(result, "__secret")

    def test_leading_double_underscore_identity(self) -> None:
        """Reassembly identity round-trip for __private."""
        name = "__private"
        words = split_identifier(name)
        self.assertEqual(reassemble_identifier(name, words), name)

    def test_double_underscore_in_reassembly(self) -> None:
        """Double underscore is preserved correctly."""
        result = reassemble_identifier("a__b", ["a", "_", "bb"])
        self.assertEqual(result, "a__bb")

    def test_mixed_underscore_and_camel(self) -> None:
        """Mixed snake_case and CamelCase parts."""
        result = reassemble_identifier(
            "my_snakeAndCamel",
            ["my", "snake", "And", "Camel"],
        )
        self.assertEqual(result, "my_snakeAndCamel")


class RestoreCaseTests(unittest.TestCase):
    """Tests for `_restore_case()` — private case-restoration helper."""

    def test_upper_original(self) -> None:
        """ALL_CAPS original → corrected word uppercased."""
        result = _restore_case("HTTP", "http")
        self.assertEqual(result, "HTTP")

    def test_capitalize_original(self) -> None:
        """Capitalized original → corrected word capitalized."""
        result = _restore_case("Http", "http")
        self.assertEqual(result, "Http")

    def test_lower_original(self) -> None:
        """Lowercase original → corrected word lowercased."""
        result = _restore_case("variable", "VARIABLE")
        self.assertEqual(result, "variable")

    def test_mixed_case_fallback(self) -> None:
        """Mixed-case original (e.g., 'mIxEd') → original preserved."""
        result = _restore_case("mIxEd", "mixed")
        self.assertEqual(result, "mIxEd")


if __name__ == "__main__":
    unittest.main()
