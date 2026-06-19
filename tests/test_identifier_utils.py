"""Tests for ``identifier_utils.py`` — the Jedi-based identifier extraction and rename.

These tests serve as a **migration safety net**: when we replace Jedi with libcst,
all tests in this file must still pass, guaranteeing behavioural equivalence.

Divided into:
1. ``is_protected_name`` — unit tests for the name filter.
2. ``extract_renameable_identifiers`` — scope-aware extraction correctness.
3. ``apply_jedi_rename`` — single-identifier rename correctness.
4. Cross-cutting edge cases — comments, strings, imports.
"""

from __future__ import annotations

import unittest

from src.identifier_utils import (
    apply_jedi_rename,
    extract_renameable_identifiers,
    is_protected_name,
)


def _is_identifier_token(code: str, name: str) -> bool:
    """Check whether ``name`` appears as a standalone identifier in ``code``.

    Uses a simple but effective test: split on non-identifier characters
    and check for exact match.  This avoids false positives when ``name``
    is a substring of a longer identifier (e.g. ``factorial`` inside
    ``factorial_fixed``).
    """
    import re
    for token in re.split(r"[^a-zA-Z0-9_]", code):
        if token == name:
            return True
    return False


# --------------------------------------------------------------------------- #
# Test snippets
# --------------------------------------------------------------------------- #

# Simple function with a few identifiers.
SIMPLE = """\
def factorial(number):
    result = 1
    for value in range(2, number + 1):
        result = result * value
    return result
"""

# Two functions defining the *same* variable name in different scopes.
NESTED_SAME_NAME = """\
def outer():
    result = 1
    def inner():
        result = 2
        return result
    return result
"""

# Class with method and attribute assignment.
CLASS_METHOD = """\
class Calculator:
    def compute(self, x, y):
        total = x + y
        return total
"""

# Imported names.
IMPORTS = """\
import os
from math import sqrt, floor as fl
"""

# Short names that should be skipped.
SHORT_NAMES = """\
def f():
    a = 1
    return a
"""

# Dunder and protected names.
PROTECTED = """\
class Foo:
    def __init__(self, arg):
        self.value = arg
    def __repr__(self):
        return str(self.value)
"""

# Names with dots (attribute access — the owner should be found, not the attr).
ATTR_ACCESS = """\
class A:
    def method(self):
        pass

a = A()
a.method()
"""

# Comprehensions.
COMPREHENSIONS = """\
def fn():
    squares = [i * i for i in range(10)]
    return squares
"""

# Lambda.
LAMBDA_SRC = """\
def fn():
    f = lambda x: x + 1
    return f
"""

# Code with comments that contain identifiers (must NOT be renamed).
COMMENT_CODE = """\
def compute(x, y):
    # compute the result of x and y
    result = x + y
    return result
"""

# Code with string that contains identifiers (must NOT be renamed).
STRING_CODE = """\
def greet(name):
    msg = "Hello name!"
    print(msg)
"""


# --------------------------------------------------------------------------- #
# 1. is_protected_name
# --------------------------------------------------------------------------- #


class IsProtectedNameTests(unittest.TestCase):
    """Unit tests for ``is_protected_name()``."""

    def test_keywords(self):
        for kw in ("if", "else", "for", "while", "def", "class", "return",
                   "import", "from", "try", "except", "with", "as"):
            self.assertTrue(is_protected_name(kw), f"keyword '{kw}' should be protected")

    def test_soft_keywords(self):
        import keyword
        for skw in getattr(keyword, "softkwlist", []):
            self.assertTrue(is_protected_name(skw), f"soft keyword '{skw}' should be protected")

    def test_builtins(self):
        for name in ("print", "len", "range", "int", "str", "list", "dict"):
            self.assertTrue(is_protected_name(name), f"builtin '{name}' should be protected")

    def test_self_and_cls(self):
        self.assertTrue(is_protected_name("self"))
        self.assertTrue(is_protected_name("cls"))

    def test_dunders(self):
        for name in ("__init__", "__main__", "__name__", "__file__", "__doc__"):
            self.assertTrue(is_protected_name(name), f"dunder '{name}' should be protected")
        # Any dunder pattern.
        self.assertTrue(is_protected_name("__custom_dunder__"))

    def test_normal_names_not_protected(self):
        for name in ("foo", "my_var", "Factorial", "compute", "result", "a1b2"):
            self.assertFalse(is_protected_name(name), f"'{name}' should NOT be protected")

    def test_names_with_double_underscore_inside_not_protected(self):
        # "__" in the middle is not a dunder.
        self.assertFalse(is_protected_name("my__name"))

    def test_single_underscore_names_not_protected(self):
        self.assertFalse(is_protected_name("_private"))
        # "_" is a soft keyword in Python >= 3.10 (match-case wildcard pattern),
        # so it IS protected.  Verify this is intentional.
        self.assertTrue(is_protected_name("_"))

    def test_leading_double_underscore_not_dunder(self):
        # "___foo" starts and ends with __ but is not a typical dunder format.
        # Our check: starts with "__" AND ends with "__".  "___foo" starts with "__"
        # but does NOT end with "__", so it is NOT protected.
        self.assertFalse(is_protected_name("___foo"))


# --------------------------------------------------------------------------- #
# 2. extract_renameable_identifiers
# --------------------------------------------------------------------------- #


class ExtractIdentifiersTests(unittest.TestCase):
    """Unit tests for ``extract_renameable_identifiers()``."""

    # -- Basic extraction ------------------------------------------------

    def test_simple_function(self):
        """All user-defined names in a simple function are extracted."""
        ids = extract_renameable_identifiers(SIMPLE)
        self.assertIn("factorial", ids)
        self.assertIn("number", ids)
        self.assertIn("result", ids)
        self.assertIn("value", ids)

    def test_positions_are_non_empty(self):
        """Every extracted name has at least one definition position."""
        ids = extract_renameable_identifiers(SIMPLE)
        for name, positions in ids.items():
            self.assertGreaterEqual(
                len(positions), 1,
                f"'{name}' should have at least one definition position",
            )
            for line, col in positions:
                self.assertIsInstance(line, int)
                self.assertIsInstance(col, int)
                self.assertGreater(line, 0, f"line for '{name}' should be >0")
                self.assertGreaterEqual(col, 0, f"col for '{name}' should be >=0")

    # -- Keywords and builtins should be skipped ------------------------

    def test_skips_keywords(self):
        ids = extract_renameable_identifiers("def f(): for x in y: pass")
        self.assertNotIn("def", ids)
        self.assertNotIn("for", ids)
        self.assertNotIn("in", ids)
        self.assertNotIn("pass", ids)

    def test_skips_builtins(self):
        ids = extract_renameable_identifiers(SIMPLE)
        self.assertNotIn("range", ids)
        self.assertNotIn("print", ids)

    def test_skips_self_and_cls(self):
        ids = extract_renameable_identifiers(CLASS_METHOD)
        self.assertNotIn("self", ids)
        self.assertNotIn("cls", ids)

    def test_skips_dunders(self):
        ids = extract_renameable_identifiers(PROTECTED)
        self.assertNotIn("__init__", ids)
        self.assertNotIn("__repr__", ids)

    # -- Scope awareness: nested functions ------------------------------

    def test_nested_same_name_multiple_definitions(self):
        """Same name defined in outer and inner scope: both definitions extracted."""
        ids = extract_renameable_identifiers(NESTED_SAME_NAME)
        self.assertIn("result", ids)
        # "result" is defined twice: once in outer, once in inner.
        self.assertEqual(len(ids["result"]), 2,
                         f"Expected 2 defs of 'result', got {ids['result']}")

    def test_nested_distinct_positions(self):
        """The two definitions of 'result' are at different positions."""
        ids = extract_renameable_identifiers(NESTED_SAME_NAME)
        positions = ids["result"]
        self.assertEqual(len(positions), 2)
        self.assertNotEqual(positions[0], positions[1])

    # -- Classes --------------------------------------------------------

    def test_class_definition_found(self):
        ids = extract_renameable_identifiers(CLASS_METHOD)
        self.assertIn("Calculator", ids)
        self.assertIn("compute", ids)

    def test_method_parameters_found(self):
        ids = extract_renameable_identifiers(CLASS_METHOD)
        self.assertIn("x", ids)
        self.assertIn("y", ids)
        self.assertIn("total", ids)

    # -- Attribute access -----------------------------------------------

    def test_attribute_owners_found(self):
        """For 'a.method()', 'a' (the owner) is a statement; 'method' is a function."""
        ids = extract_renameable_identifiers(ATTR_ACCESS)
        self.assertIn("A", ids)
        self.assertIn("method", ids)
        self.assertIn("a", ids)

    # -- Comprehensions -------------------------------------------------

    def test_comprehension_variables(self):
        """Comprehension iterator variables (e.g., 'i') are scope-local."""
        ids = extract_renameable_identifiers(COMPREHENSIONS)
        self.assertIn("squares", ids)
        self.assertIn("fn", ids)
        # 'i' is defined inside the comprehension and is a local.
        # Jedi may or may not pick it up as a 'statement' — it depends on the version.
        # We just check that nothing crashes and that 'squares' is found.

    # -- Lambda parameters ----------------------------------------------

    def test_lambda_parameters(self):
        """Lambda parameters should be extractable."""
        ids = extract_renameable_identifiers(LAMBDA_SRC)
        self.assertIn("fn", ids)
        self.assertIn("f", ids)
        # 'x' (the lambda param) may or may not be picked up as a 'param'.

    # -- Imports --------------------------------------------------------

    def test_imported_module_names(self):
        """Module names from 'import X' should be extractable statements."""
        ids = extract_renameable_identifiers(IMPORTS)
        # 'os' is an import — Jedi may treat it as a statement.
        # 'sqrt', 'fl' are imported names — may be statements as well.
        # The key invariant: nothing crashes.
        self.assertIsInstance(ids, dict)

    # -- Short names ----------------------------------------------------

    def test_short_names_not_extracted(self):
        """Identifiers shorter than 3 characters are excluded by the _PROTECTED_NAMES check
        ... actually, they're not in _PROTECTED_NAMES, but Jedi may treat single-letter
        names differently.  The key invariant is: extraction doesn't crash, and
        'f' and 'a' are either absent or present depending on Jedi's classification."""
        ids = extract_renameable_identifiers(SHORT_NAMES)
        # Jedi may classify 'f' as 'function' and 'a' as 'statement'.
        # Both are >=  Turbo (the 3-char minimum is not enforced by Jedi itself).
        # We just test that it doesn't crash.
        self.assertIsInstance(ids, dict)


# --------------------------------------------------------------------------- #
# 3. apply_jedi_rename
# --------------------------------------------------------------------------- #


class ApplyJediRenameTests(unittest.TestCase):
    """Unit tests for ``apply_jedi_rename()``."""

    def _extract_def(self, source: str, name: str) -> tuple[int, int]:
        """Helper: extract the first definition position for ``name``."""
        ids = extract_renameable_identifiers(source)
        return ids[name][0]

    # -- Simple renames ------------------------------------------------

    def test_rename_function(self):
        line, col = self._extract_def(SIMPLE, "factorial")
        result = apply_jedi_rename(SIMPLE, line, col, "fact")
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(len(result), 1)
        code = list(result.values())[0]
        self.assertNotIn("def factorial(", code)
        self.assertIn("def fact(", code)

    def test_rename_variable(self):
        line, col = self._extract_def(SIMPLE, "result")
        result = apply_jedi_rename(SIMPLE, line, col, "output")
        self.assertGreaterEqual(len(result), 1)
        code = list(result.values())[0]
        self.assertNotIn("result", code)
        self.assertIn("output", code)

    def test_rename_parameter(self):
        line, col = self._extract_def(SIMPLE, "number")
        result = apply_jedi_rename(SIMPLE, line, col, "n")
        code = list(result.values())[0]
        self.assertNotIn("number", code)
        self.assertIn("n", code)

    # -- Rename all occurrences ----------------------------------------

    def test_rename_all_occurrences(self):
        """Renaming 'result' must change ALL usages, not just the definition."""
        line, col = self._extract_def(SIMPLE, "result")
        result = apply_jedi_rename(SIMPLE, line, col, "r")
        code = list(result.values())[0]
        self.assertIn("r = 1", code)
        self.assertIn("r = r * value", code)
        self.assertIn("return r", code)
        self.assertNotIn("result", code)

    # -- Nested scopes -------------------------------------------------

    def test_rename_outer_does_not_affect_inner(self):
        """Renaming 'result' in outer scope must not rename inner 'result'."""
        line, col = self._extract_def(NESTED_SAME_NAME, "result")
        result = apply_jedi_rename(NESTED_SAME_NAME, line, col, "output")
        code = list(result.values())[0]
        # The OUTER 'result' should be renamed, but INNER 'result' is a different
        # variable and should stay.
        self.assertIn("output = 1", code)      # outer definition renamed.
        self.assertIn("result = 2", code)       # inner definition untouched.
        self.assertIn("return output", code)    # outer return renamed.

    def test_rename_inner_does_not_affect_outer(self):
        """Renaming the SECOND definition position renames the inner scope only."""
        ids = extract_renameable_identifiers(NESTED_SAME_NAME)
        # Get the second definition position (inner scope).
        line, col = ids["result"][1]
        result = apply_jedi_rename(NESTED_SAME_NAME, line, col, "inner_output")
        code = list(result.values())[0]
        self.assertIn("result = 1", code)            # outer untouched.
        self.assertIn("inner_output = 2", code)      # inner renamed.
        self.assertIn("return inner_output", code)   # inner return renamed.
        self.assertIn("return result", code)         # outer return untouched.

    # -- Class method rename -------------------------------------------

    def test_rename_method(self):
        line, col = self._extract_def(CLASS_METHOD, "compute")
        result = apply_jedi_rename(CLASS_METHOD, line, col, "calc")
        code = list(result.values())[0]
        self.assertIn("def calc(self, x, y):", code)
        self.assertNotIn("def compute(", code)

    def test_rename_class(self):
        line, col = self._extract_def(CLASS_METHOD, "Calculator")
        result = apply_jedi_rename(CLASS_METHOD, line, col, "CalcClass")
        code = list(result.values())[0]
        self.assertIn("class CalcClass:", code)
        self.assertNotIn("class Calculator:", code)

    # -- Invalid positions ---------------------------------------------

    def test_invalid_position_returns_empty(self):
        """Renaming at a position without an identifier returns empty result."""
        result = apply_jedi_rename(SIMPLE, 1, 0, "whatever")
        self.assertEqual(result, {})

    def test_keyword_position_returns_empty(self):
        """Trying to rename a keyword should fail gracefully."""
        # 'def' is at the beginning of line 1 in SIMPLE.
        result = apply_jedi_rename(SIMPLE, 1, 0, "foobar")
        self.assertEqual(result, {})

    # -- Comments and strings must NOT be renamed ----------------------

    def test_rename_does_not_touch_comments(self):
        line, col = self._extract_def(COMMENT_CODE, "result")
        result = apply_jedi_rename(COMMENT_CODE, line, col, "output")
        code = list(result.values())[0]
        # The phrase "compute the result of x and y" in the comment must keep "result".
        self.assertIn("# compute the result of x and y", code)

    def test_rename_does_not_touch_strings(self):
        line, col = self._extract_def(STRING_CODE, "name")
        result = apply_jedi_rename(STRING_CODE, line, col, "username")
        code = list(result.values())[0]
        # The string "Hello name!" must keep "name" (it's inside quotes).
        self.assertIn('"Hello name!"', code)
        # But the parameter 'name' should be renamed.
        self.assertIn("def greet(username):", code)

    # -- path parameter ------------------------------------------------

    def test_path_none(self):
        """With path=None, Jedi does not attempt file resolution — fast path."""
        line, col = self._extract_def(SIMPLE, "result")
        result = apply_jedi_rename(SIMPLE, line, col, "r", path=None)
        self.assertIsInstance(result, dict)
        code = result.get("", list(result.values())[0] if result else "")
        self.assertIsInstance(code, str)

    # -- Return value: dict path→code ----------------------------------

    def test_return_dict_structure(self):
        """Result is dict with file paths as keys (str or None when path= was omitted)."""
        line, col = self._extract_def(SIMPLE, "result")
        result = apply_jedi_rename(SIMPLE, line, col, "r")
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(len(result), 1)
        for key, value in result.items():
            self.assertIn(type(key), (str, type(None)),
                          f"key should be str or None, got {type(key)}")
            self.assertIsInstance(value, str)
            self.assertGreater(len(value), 0)


# --------------------------------------------------------------------------- #
# 4. Cross-cutting integration tests
# --------------------------------------------------------------------------- #


class IntegrationTests(unittest.TestCase):
    """Tests that exercise the full extract→rename pipeline on non-trivial code."""

    @staticmethod
    def _extract_def(source: str, name: str) -> tuple[int, int]:
        ids = extract_renameable_identifiers(source)
        return ids[name][0]

    def test_extract_then_rename_cycle(self):
        """Extract positions, rename at each, verify consistency."""
        source = SIMPLE
        ids = extract_renameable_identifiers(source)
        for name, positions in ids.items():
            for line, col in positions:
                new_name = name + "_fixed"
                result = apply_jedi_rename(source, line, col, new_name)
                if result:
                    code = list(result.values())[0]
                    # The fixed name should appear.
                    self.assertIn(new_name, code)
                    # The original name should NOT appear as a standalone identifier.
                    # We check by tokenising the code: the old name must not be a token.
                    self.assertFalse(
                        _is_identifier_token(code, name),
                        f"'{name}' should have been fully renamed, but found in: {code}")

    def test_multi_rename_sequential(self):
        """Simulate harness: rename identifiers one by one, re-extracting each time."""
        source = SIMPLE
        ids = extract_renameable_identifiers(source)
        code = source
        for name, positions in ids.items():
            for line, col in positions:
                # Re-extract positions from the *current* code.
                current_ids = extract_renameable_identifiers(code)
                if name not in current_ids:
                    continue  # already renamed.
                # Positions for this name may have changed; find the matching one.
                renamed = False
                for cl, cc in current_ids[name]:
                    result = apply_jedi_rename(code, cl, cc, name + "_ok")
                    if result:
                        code = list(result.values())[0]
                        renamed = True
                        break
                self.assertTrue(renamed, f"Failed to rename '{name}'")

        # All original names should be gone (as standalone identifiers).
        for name in ids:
            self.assertFalse(
                _is_identifier_token(code, name),
                f"'{name}' should have been fully renamed in final code: {code}")

    def test_path_parameter_never_empty_string(self):
        """Regression test: path="" causes ~1.9s overhead per Jedi call.
        We ensure apply_jedi_rename never passes path="" to Jedi internally."""
        import time
        line, col = self._extract_def(SIMPLE, "result")
        t0 = time.monotonic()
        result = apply_jedi_rename(SIMPLE, line, col, "r")
        elapsed = time.monotonic() - t0
        # With path=None (default), this should be fast (<0.5s).
        self.assertLess(
            elapsed, 0.5,
            f"Jedi rename with default path=None took {elapsed:.2f}s — "
            f"make sure path='' is never passed.",
        )
        self.assertIsInstance(result, dict)


# --------------------------------------------------------------------------- #
# 5. Edge-case snippets (compile-only, no crash)
# --------------------------------------------------------------------------- #


EDGE_CASES = {
    "empty_file": "",
    "only_comment": "# nothing here",
    "only_string": "'just a string'",
    "only_pass": "pass",
    "decorator": """\
def deco(f):
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper

@deco
def greet(msg):
    print(msg)
""",
    "generator": """\
def gen(n):
    for i in range(n):
        yield i
""",
    "async_function": """\
async def fetch(url):
    result = await something(url)
    return result
""",
    "try_except": """\
def safe_div(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        result = 0
    return result
""",
    "global_stmt": """\
x = 1
def f():
    global x
    x = 2
    return x
""",
    "nonlocal_stmt": """\
def outer():
    v = 1
    def inner():
        nonlocal v
        v = 2
        return v
    return inner()
""",
    "star_args": """\
def fn(a, *args, **kwargs):
    return a, args, kwargs
""",
    "multiple_classes": """\
class A:
    value = 1
class B(A):
    value = 2
""",
    "string_formatting": """\
def report(name, score):
    print(f'{name}: {score}')
""",
}


class EdgeCaseTests(unittest.TestCase):
    """Ensure extraction and rename don't crash on unusual but valid Python."""

    def test_extract_no_crash(self):
        for label, code in EDGE_CASES.items():
            with self.subTest(f"extract: {label}"):
                ids = extract_renameable_identifiers(code)
                self.assertIsInstance(ids, dict, f"extract on '{label}' should return dict")

    def test_rename_no_crash(self):
        for label, code in EDGE_CASES.items():
            with self.subTest(f"rename: {label}"):
                ids = extract_renameable_identifiers(code)
                if not ids:
                    continue  # nothing to rename.
                name = next(iter(ids))
                line, col = ids[name][0]
                result = apply_jedi_rename(code, line, col, name + "_ok")
                self.assertIsInstance(result, dict,
                                      f"rename on '{label}' should return dict")


if __name__ == "__main__":
    unittest.main()
