"""Run the ``identifier_utils`` test suite against the libcst implementation.

This verifies behavioural equivalence: every test in ``test_identifier_utils.py``
that passes with Jedi must also pass with libcst.
"""

from __future__ import annotations

import sys
import unittest

# Redirect imports from identifier_utils to identifier_utils_libcst.
# All tests in test_identifier_utils import from ``src.identifier_utils``;
# we override that module with the libcst version.
import src.identifier_utils_libcst as _libcst_impl
sys.modules["src.identifier_utils"] = _libcst_impl
sys.modules["src.identifier_utils"].__name__ = "src.identifier_utils"

# Also patch the harness module which imports from identifier_utils.
# But the test file tests the functions directly, so we just run them.

from tests.test_identifier_utils import (
    ApplyJediRenameTests,
    EdgeCaseTests,
    ExtractIdentifiersTests,
    IntegrationTests,
    IsProtectedNameTests,
)


def load_tests(loader, standard_tests, pattern):
    """Collect all test classes from test_identifier_utils."""
    suite = unittest.TestSuite()
    for test_class in (
        IsProtectedNameTests,
        ExtractIdentifiersTests,
        ApplyJediRenameTests,
        IntegrationTests,
        EdgeCaseTests,
    ):
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    return suite


if __name__ == "__main__":
    unittest.main()
