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
"""

from __future__ import annotations

from . import _plm_parser
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
    pre1 = uplm_preprocess(source, filename, defines=defines, include_paths=include_paths)
    src = macro_pass(pre1)
    try:
        tree = _plm_parser.parse(src, filename=filename)
    except Exception as e:  # ScanError or ParseError
        raise ParserError(str(e), SourceLocation(1, 1, filename)) from e
    return tree
