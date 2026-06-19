"""LibCST-based identifier utilities — drop-in replacement for the Jedi-based
:mod:`identifier_utils`.

Advantages over Jedi:
- **Thread-safe**: no shared mutable state, no file cache.
- **Single-pass multi-rename**: all identifiers renamed in one CST traversal.
- **No filesystem**: works on strings directly.
- **Preserves formatting**: Concrete Syntax Tree keeps whitespace and comments.

API compatibility with :mod:`identifier_utils`:
- ``is_protected_name(name)`` — unchanged (pure string logic).
- ``extract_renameable_identifiers(source)`` — same return type
  ``{name: [(line, col), ...]}``.
- ``apply_rename(source, rename_map)`` — batch rename: ``{corrupted_name: fixed_name}``
  → modified code string.

The final step is to replace imports in the codebase:
``from <identifier_utils> import extract_renameable_identifiers, apply_...``
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
# Name protection — identical to identifier_utils.py
# ---------------------------------------------------------------------------

_PROTECTED_NAMES: set[str] = (
    set(keyword.kwlist)
    | set(getattr(keyword, "softkwlist", []))
    | set(dir(builtins))
    | {"self", "cls", "__init__", "__name__", "__main__", "__file__", "__doc__"}
)


def is_protected_name(name: str) -> bool:
    """Return ``True`` if ``name`` is a keyword, builtin, ``self``/``cls``,
    or a dunder."""
    return name in _PROTECTED_NAMES or (name.startswith("__") and name.endswith("__"))


# ---------------------------------------------------------------------------
# CST helpers
# ---------------------------------------------------------------------------


def _get_name_position(
    node: cst.CSTNode, positions: Dict[cst.CSTNode, cst.metadata.CodeRange],
) -> Tuple[int, int]:
    """Return the ``(line, column)`` of the *identifier* within ``node``.

    - ``FunctionDef`` / ``ClassDef`` → position of ``node.name``.
    - ``Param`` → position of ``node.name``.
    - ``Name`` → position of the node itself.
    - Anything else → position of the node (best-effort).
    """
    if isinstance(node, (cst.FunctionDef, cst.ClassDef)):
        target = node.name
    elif isinstance(node, cst.Param):
        target = node.name
    elif isinstance(node, cst.Name):
        target = node
    else:
        target = node
    cr = positions.get(target)
    if cr is None:
        return (0, 0)
    return (cr.start.line, cr.start.column)


def _get_assign_name_node(node: cst.CSTNode) -> cst.CSTNode:
    """Extract the ``Name`` CST node that carries the identifier value.

    - ``FunctionDef`` / ``ClassDef`` → ``node.name``.
    - ``Param`` → ``node.name``.
    - ``Name`` (variable) → ``node``.
    """
    if isinstance(node, (cst.FunctionDef, cst.ClassDef)):
        return node.name
    elif isinstance(node, cst.Param):
        return node.name
    return node


# ---------------------------------------------------------------------------
# extract_renameable_identifiers (LibCST)
# ---------------------------------------------------------------------------


def extract_renameable_identifiers(
    source: str,
) -> Dict[str, List[Tuple[int, int]]]:
    """Return ``name -> [(line, col), ...]`` of definition positions.

    Uses libcst's ``ScopeProvider`` for scope-aware extraction.  A name
    may appear multiple times if it is defined in different scopes.

    Includes function, class, variable, and parameter definitions.

    Filters out keywords, builtins, ``self``/``cls``, dunders, and
    import-assigned names (which are typically not renameable in our
    typo-fixing context).
    """
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    scopes_map = wrapper.resolve(ScopeProvider)
    positions = wrapper.resolve(PositionProvider)

    result: Dict[str, List[Tuple[int, int]]] = {}

    # Deduplicate scopes — the same scope object is attached to many CST nodes.
    seen_scopes: Set[int] = set()
    for scope in scopes_map.values():
        if scope is None or id(scope) in seen_scopes:
            continue
        seen_scopes.add(id(scope))

        for assignment in scope.assignments:
            name = assignment.name
            if is_protected_name(name):
                continue

            # Note: no explicit length filter.  Jedi's implementation also does
            # NOT filter by length — only by _PROTECTED_NAMES and type.

            # Exclude import-originated names — these are module/symbol aliases
            # that we don't want to corrupt or fix.
            if _is_import_assignment(assignment):
                continue

            line, col = _get_name_position(assignment.node, positions)
            if line == 0 and col == 0:
                continue

            result.setdefault(name, []).append((line, col))

    return result


def _is_import_assignment(assignment) -> bool:
    """Return ``True`` if the assignment originates from an ``import`` statement.

    ``ScopeProvider`` creates ``ImportAssignment`` objects for names brought
    in via ``import X`` or ``from Y import Z``.  ``BuiltinAssignment`` similarly
    marks builtins (``range``, ``print``, etc.).  We exclude both.
    """
    cls_name = type(assignment).__name__
    return cls_name.endswith("ImportAssignment") or cls_name.endswith("BuiltinAssignment")


# ---------------------------------------------------------------------------
# apply_rename (LibCST) — batch, single-pass
# ---------------------------------------------------------------------------


class _MultiRenameTransformer(cst.CSTTransformer):
    """CST transformer that renames a pre-computed set of ``Name`` nodes."""

    def __init__(self, rename_map: Dict[str, str], nodes_to_rename: Set[cst.CSTNode]) -> None:
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
    """Rename all identifiers specified in ``rename_map`` in one CST pass.

    Parameters:
        source: The Python source code.
        rename_map: ``{corrupted_name: fixed_name, ...}`` — every occurrence
                    of each ``corrupted_name`` in the relevant scope will be
                    renamed to the corresponding ``fixed_name``.

    Returns:
        The modified source code.

    Scope awareness: for each ``corrupted_name`` we find its
    assignments (definitions) and rename both the definition and all of
    its references.  A name defined in *two* scopes will have both
    assignment groups renamed — this matches Jedi's per-scope rename
    behaviour.
    """
    if not rename_map:
        return source

    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    scopes_map = wrapper.resolve(ScopeProvider)

    # Collect every CST ``Name`` node that must be renamed.
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

            # Add the definition-site Name node.
            nodes_to_rename.add(_get_assign_name_node(assignment.node))

            # Add all reference (access) Name nodes.
            for access in assignment.references:
                if isinstance(access.node, cst.Name):
                    nodes_to_rename.add(access.node)

    if not nodes_to_rename:
        return source

    transformer = _MultiRenameTransformer(rename_map, nodes_to_rename)
    modified = wrapper.module.visit(transformer)
    return modified.code


# ---------------------------------------------------------------------------
# Backward-compatible wrapper for test harness (single-identifier rename)
# ---------------------------------------------------------------------------


def apply_jedi_rename(
    source: str, line: int, col: int, new_name: str,
    path: object = None,
) -> dict:
    """Backward-compatible wrapper that mimics the old ``apply_jedi_rename`` API.

    Finds the assignment at ``(line, col)`` and renames only that scope's
    occurrences.  Callers should migrate to ``apply_rename()`` for batch
    multi-identifier renames.

    Returns ``{"": modified_source}`` on success, ``{}`` on failure.
    """
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    scopes_map = wrapper.resolve(ScopeProvider)
    positions = wrapper.resolve(PositionProvider)

    # Find the assignment whose name sits exactly at (line, col).
    target_assignment = None
    seen_scopes: Set[int] = set()
    for scope in scopes_map.values():
        if scope is None or id(scope) in seen_scopes:
            continue
        seen_scopes.add(id(scope))
        for assignment in scope.assignments:
            if is_protected_name(assignment.name):
                continue
            name_node = _get_assign_name_node(assignment.node)
            cr = positions.get(name_node)
            if cr is not None and cr.start.line == line and cr.start.column == col:
                target_assignment = assignment
                break
        if target_assignment:
            break

    if target_assignment is None:
        return {}

    # Jedi's ``rename()`` at *any* definition position renames ALL definitions
    # of the same name within the same scope.  Find all such assignments.
    #
    # Note: ``scope.assignments`` is an ``Assignments`` collection whose
    # ``__contains__`` accepts *strings* (by name), not Assignment objects.
    # We use ``list(scope.assignments)`` for identity checks.
    target_scope = None
    _target_list: List[cst.metadata.BaseAssignment] = []
    seen_scopes2: Set[int] = set()
    for scope in scopes_map.values():
        if scope is None or id(scope) in seen_scopes2:
            continue
        seen_scopes2.add(id(scope))
        scope_list = list(scope.assignments)
        if target_assignment in scope_list:
            target_scope = scope
            _target_list = scope_list
            break

    nodes_to_rename: Set[cst.CSTNode] = set()
    seen_assignment_ids: Set[int] = set()

    if target_scope is not None and _target_list:
        for assignment in _target_list:
            if assignment.name != target_assignment.name:
                continue
            if id(assignment) in seen_assignment_ids:
                continue
            seen_assignment_ids.add(id(assignment))
            nodes_to_rename.add(_get_assign_name_node(assignment.node))
            for access in assignment.references:
                if isinstance(access.node, cst.Name):
                    nodes_to_rename.add(access.node)
    else:
        # Fallback: just the matched assignment.
        nodes_to_rename.add(_get_assign_name_node(target_assignment.node))
        for access in target_assignment.references:
            if isinstance(access.node, cst.Name):
                nodes_to_rename.add(access.node)

    if not nodes_to_rename:
        return {}

    rename_map = {target_assignment.name: new_name}
    transformer = _MultiRenameTransformer(rename_map, nodes_to_rename)
    modified = wrapper.module.visit(transformer)
    return {"": modified.code}
