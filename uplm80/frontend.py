"""uplox-driven PL/M-80 front-end.

Pipeline per source file:

1. :func:`preprocess.preprocess` — uplm80's own preprocessor: high-bit
   strip, recursive ``$INCLUDE``, ``$cond`` / ``$if`` conditional
   compilation.
2. :func:`preprocess.macro_pass` — block-scoped LITERALLY substitution
   + case fold to upper. PL/M LITERALLYs are scoped to the enclosing
   ``DO`` / ``PROCEDURE`` block, so this stays in uplm80 rather than
   moving into uplox's stateless preprocess.
3. :func:`uplm80._plm_parser.parse` — uplox-generated typed parser+AST.
   The grammar lives in ``../uplox/examples/plm_full.uplox`` and is
   annotated with ``%ast=`` tags so reduction builds a typed dataclass
   tree directly (no CST-to-AST lowering pass). Regenerate via
   ``scripts/regen_parser.sh``.

The returned AST root is :class:`uplm80._plm_parser.Module`. Downstream
(codegen, optimizer) walks the typed nodes via ``isinstance``.

The parser sees one text, the file with its ``$INCLUDE`` files spliced
in, and numbers its lines.  Every node's position is given the file and
line it really came from (:func:`note_origins`), which is what a
diagnostic names (:func:`source_location`).
"""

from __future__ import annotations

import re

from . import _plm_parser
from . import _plm_parser as P
from ._plm_parser import Module
from .errors import ParserError, SourceLocation
from .preprocess import macro_pass, preprocess as uplm_preprocess


def parse_source(
    source: str,
    filename: str = "<input>",
    defines: list[str] | None = None,
    include_paths: list[str] | None = None,
) -> Module:
    """Run the full PL/M-80 front-end pipeline and return the typed
    :class:`Module` produced by the uplox-generated parser."""
    line_map: list[tuple[str, int]] = []
    pre1 = uplm_preprocess(source, filename, defines=defines, include_paths=include_paths,
                           line_map=line_map)
    src = macro_pass(pre1)
    try:
        tree = _plm_parser.parse(src, filename=filename)
    except Exception as e:  # ScanError or ParseError
        raise _syntax_error(e, line_map, filename) from e
    note_origins(tree, line_map)
    tree.uplm80_file = filename
    return tree


def _origin(line_map: list[tuple[str, int]], line: int, filename: str) -> tuple[str, int]:
    """(file, line) of line ``line`` of the preprocessed text."""
    if 1 <= line <= len(line_map):
        return line_map[line - 1]
    return filename, line


def _syntax_error(e: Exception, line_map, filename: str) -> ParserError:
    """A scanner or parser error, placed where its text came from.

    uplox reports a line of the preprocessed text, which past an
    ``$INCLUDE`` is not a line of the file being compiled.
    """
    token = getattr(e, "token", None)
    line = getattr(token, "line", None) or getattr(e, "line", None)
    column = getattr(token, "column", None) or getattr(e, "column", None)
    message = str(e)
    if not line:
        return ParserError(message, SourceLocation(1, 1, filename))
    # The message says where in uplox's terms; the location says it right.
    message = re.sub(r"\s*at line \d+, column \d+", "", message)
    message = re.sub(r"^[^\s:]*:\d+:\d+:\s*", "", message)
    file, orig = _origin(line_map, line, filename)
    return ParserError(message, SourceLocation(orig, column or 1, file))


def note_origins(tree, line_map: list[tuple[str, int]]) -> None:
    """Give every node position of ``tree`` the (file, line) it came from,
    as ``pos.origin``.

    The optimizer carries a node's position over to what it rewrites the
    node into, so what code generation reports on still knows its origin.
    """
    if not line_map:
        return
    seen: set[int] = set()
    stack: list = [tree]
    while stack:
        n = stack.pop()
        if isinstance(n, (list, tuple)):
            stack.extend(n)
            continue
        fields = getattr(n, "__dataclass_fields__", None)
        if not fields or id(n) in seen:
            continue
        seen.add(id(n))
        pos = getattr(n, "pos", None)
        if isinstance(n, P.AssignStmt) and n.targets:
            # The grammar starts an assignment's span at its `=': the
            # targets before it are a list, which has no position.
            first = getattr(n.targets[0], "pos", None)
            if first is not None and first.start_line:
                pos.start_line, pos.start_column = first.start_line, first.start_column
        if pos is not None and getattr(pos, "start_line", 0):
            pos.origin = _origin(line_map, pos.start_line, "")
        stack.extend(getattr(n, f, None) for f in fields if f != "pos")


def source_location(node) -> SourceLocation | None:
    """Where ``node`` (or a node position) is in the source, if known."""
    pos = getattr(node, "pos", node)
    line = getattr(pos, "start_line", 0)
    if not line:
        return None
    origin = getattr(pos, "origin", None)
    if origin is not None:
        return SourceLocation(origin[1], pos.start_column, origin[0])
    return SourceLocation(line, pos.start_column)
