"""Identifier extraction and rename utilities.

Uses libcst's ``ScopeProvider`` for scope-aware analysis and
``CSTTransformer`` for single-pass multi-identifier rename.

Shared by :mod:`typo_injector` (dataset generation) and :mod:`harness`
(model evaluation).

Thread-safe — no shared mutable state, no file cache.

Raises
------
:class:`UnparseableCodeError`
    When libcst cannot parse the source code (Python 2 syntax, merge
    conflict markers, etc.).  Callers should skip such samples.
"""

from __future__ import annotations

import builtins
import keyword
from typing import Dict, List, Set, Tuple

import libcst as cst
from libcst.metadata import (
    MetadataWrapper,
    PositionProvider,
    ScopeProvider,
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UnparseableCodeError(Exception):
    """Raised when libcst cannot parse the source code.

    This happens with Python 2 syntax, merge conflict markers, or other
    invalid Python 3 constructs.  Callers should skip the offending sample.
    """


# ---------------------------------------------------------------------------
# Name protection
# ---------------------------------------------------------------------------

_PROTECTED_NAMES: set[str] = (
    set(keyword.kwlist)
    | set(getattr(keyword, "softkwlist", []))
    | set(dir(builtins))
    | {"self", "cls", "__init__", "__name__", "__main__", "__file__", "__doc__"}
)


def is_protected_name(name: str) -> bool:
    """Return ``True`` for keywords, builtins, ``self``/``cls``, or dunders."""
    return name in _PROTECTED_NAMES or (name.startswith("__") and name.endswith("__"))


# ---------------------------------------------------------------------------
# CST helpers
# ---------------------------------------------------------------------------


def _get_name_node(node: cst.CSTNode) -> cst.CSTNode:
    """Extract the ``Name`` CST node from a ``FunctionDef``, ``ClassDef``,
    ``Param``, or return the node itself if it is already a ``Name``."""
    if isinstance(node, (cst.FunctionDef, cst.ClassDef, cst.Param)):
        return node.name
    return node


def _get_name_position(
    node: cst.CSTNode,
    positions: Dict[cst.CSTNode, cst.metadata.CodeRange],
) -> Tuple[int, int]:
    """Return ``(line, column)`` of the identifier within ``node``."""
    target = _get_name_node(node)
    cr = positions.get(target)
    if cr is None:
        return (0, 0)
    return (cr.start.line, cr.start.column)


def _is_import_or_builtin(assignment) -> bool:
    """Exclude ``ImportAssignment`` and ``BuiltinAssignment`` scope objects."""
    name = type(assignment).__name__
    return name.endswith("ImportAssignment") or name.endswith("BuiltinAssignment")


def _is_valid_identifier(name: str) -> bool:
    """Check whether *name* is a valid Python identifier.

    Uses libcst's own validation — names with hyphens, leading digits,
    or other invalid characters will be rejected by :class:`cst.Name`.
    """
    try:
        cst.Name(name)
    except cst.CSTValidationError:
        return False
    return True


# ---------------------------------------------------------------------------
# extract_renameable_identifiers
# ---------------------------------------------------------------------------


def extract_renameable_identifiers(
    source: str,
) -> Dict[str, List[Tuple[int, int]]]:
    """Return ``{name: [(line, col), ...]}`` of definition positions.

    A name may appear multiple times if defined in different scopes
    (e.g. ``result`` in both ``outer()`` and ``inner()``).

    Includes functions, classes, variables, and parameters.

    Filters out keywords, builtins, ``self``/``cls``, dunders, and
    import-originated names.

    Returns an empty dict if the source cannot be parsed (e.g., Python 2
    syntax, merge conflict markers).

    Raises
    ------
    UnparseableCodeError
        If the source cannot be parsed by libcst.
    """
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError:
        raise UnparseableCodeError(
            f"Cannot parse source: {cst.ParserSyntaxError}"
        ) from None
    wrapper = MetadataWrapper(module)
    scopes_map = wrapper.resolve(ScopeProvider)
    positions = wrapper.resolve(PositionProvider)

    result: Dict[str, List[Tuple[int, int]]] = {}
    seen_scopes: Set[int] = set()

    for scope in scopes_map.values():
        if scope is None or id(scope) in seen_scopes:
            continue
        seen_scopes.add(id(scope))

        for assignment in scope.assignments:
            name = assignment.name
            if is_protected_name(name) or _is_import_or_builtin(assignment):
                continue
            line, col = _get_name_position(assignment.node, positions)
            if line == 0 and col == 0:
                continue
            result.setdefault(name, []).append((line, col))

    return result


# ---------------------------------------------------------------------------
# apply_rename — batch, single-pass
# ---------------------------------------------------------------------------


class _MultiRenameTransformer(cst.CSTTransformer):
    """Renames a pre-computed set of ``Name`` nodes in one pass."""

    def __init__(
        self,
        rename_map: Dict[str, str],
        nodes_to_rename: Set[cst.CSTNode],
    ) -> None:
        super().__init__()
        self._rename_map = rename_map
        self._nodes = nodes_to_rename

    def leave_Name(
        self,
        original_node: cst.Name,
        updated_node: cst.Name,
    ) -> cst.Name:
        if original_node in self._nodes:
            new_value = self._rename_map.get(original_node.value)
            if new_value is not None:
                return updated_node.with_changes(value=new_value)
        return updated_node


def apply_rename(
    source: str,
    rename_map: Dict[str, str],
) -> str:
    """Rename all identifiers in ``rename_map`` in a single CST pass.

    For each ``corrupted_name → fixed_name`` mapping, finds every
    assignment (definition) of that name across all scopes, and renames
    both the definition and all of its references.

    If a name is defined in two scopes, both are renamed.

    Targets that are not valid Python identifiers (e.g., names with
    hyphens) are silently skipped.

    Raises
    ------
    UnparseableCodeError
        If the source cannot be parsed by libcst.
    """
    if not rename_map:
        return source

    # Filter out targets that are not valid Python identifiers.
    # Models sometimes hallucinate names with hyphens or other invalid
    # characters.
    rename_map = {
        k: v for k, v in rename_map.items()
        if _is_valid_identifier(v)
    }
    if not rename_map:
        return source

    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError:
        raise UnparseableCodeError(
            f"Cannot parse source: {cst.ParserSyntaxError}"
        ) from None
    wrapper = MetadataWrapper(module)
    scopes_map = wrapper.resolve(ScopeProvider)

    nodes_to_rename: Set[cst.CSTNode] = set()
    seen_scopes: Set[int] = set()

    for scope in scopes_map.values():
        if scope is None or id(scope) in seen_scopes:
            continue
        seen_scopes.add(id(scope))

        for assignment in scope.assignments:
            name = assignment.name
            if name not in rename_map:
                continue
            # Definition site.
            nodes_to_rename.add(_get_name_node(assignment.node))
            # All references.
            for access in assignment.references:
                if isinstance(access.node, cst.Name):
                    nodes_to_rename.add(access.node)

    if not nodes_to_rename:
        return source

    transformer = _MultiRenameTransformer(rename_map, nodes_to_rename)
    return wrapper.module.visit(transformer).code


# ---------------------------------------------------------------------------
# apply_rename_single — position-based rename (used by typo_injector)
# ---------------------------------------------------------------------------


def apply_rename_single(
    source: str,
    line: int,
    col: int,
    new_name: str,
) -> str:
    """Rename the identifier at ``(line, col)`` to ``new_name``.

    Finds the assignment at the given position and renames all definitions
    of the same name in its scope plus all of their references.

    Returns the modified source, or the original if no assignment was
    found at ``(line, col)`` or if the source cannot be parsed.

    Raises
    ------
    UnparseableCodeError
        If the source cannot be parsed by libcst.
    """
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError:
        raise UnparseableCodeError(
            f"Cannot parse source: {cst.ParserSyntaxError}"
        ) from None
    wrapper = MetadataWrapper(module)
    scopes_map = wrapper.resolve(ScopeProvider)
    positions = wrapper.resolve(PositionProvider)

    # Find the assignment at (line, col).
    target_assignment = None
    target_name = ""
    seen_scopes: Set[int] = set()
    for scope in scopes_map.values():
        if scope is None or id(scope) in seen_scopes:
            continue
        seen_scopes.add(id(scope))
        for assignment in scope.assignments:
            if is_protected_name(assignment.name) or _is_import_or_builtin(assignment):
                continue
            name_node = _get_name_node(assignment.node)
            cr = positions.get(name_node)
            if cr is not None and cr.start.line == line and cr.start.column == col:
                target_assignment = assignment
                target_name = assignment.name
                break
        if target_assignment:
            break

    if target_assignment is None:
        return source

    # Find all assignments of the same name in the same scope.
    nodes_to_rename: Set[cst.CSTNode] = set()
    seen: Set[int] = set()
    for scope in scopes_map.values():
        if scope is None or id(scope) in seen:
            continue
        seen.add(id(scope))
        scope_list = list(scope.assignments)
        if target_assignment not in scope_list:
            continue
        for assignment in scope_list:
            if assignment.name != target_name:
                continue
            nodes_to_rename.add(_get_name_node(assignment.node))
            for access in assignment.references:
                if isinstance(access.node, cst.Name):
                    nodes_to_rename.add(access.node)
        break  # only one scope.

    if not nodes_to_rename:
        return source

    transformer = _MultiRenameTransformer({target_name: new_name}, nodes_to_rename)
    return wrapper.module.visit(transformer).code
