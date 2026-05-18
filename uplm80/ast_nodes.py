"""Vestigial shim — the hand-built PL/M-80 AST is gone.

After the uplox migration, parsing produces typed dataclasses from
:mod:`uplm80._plm_parser` directly. Codegen and the AST optimizer
consume those via helpers in :mod:`uplm80.ast_view`.

What survives here is only what the symbol table genuinely needs:

* :class:`DataType` — re-exported from :mod:`uplm80.ast_view`. The
  enum's member set is the boundary contract with :mod:`uplm80.symbols`.
* :class:`StructMember` — small structural record used by the symbol
  table to remember a struct's per-field layout. The typed AST itself
  carries struct shape on ``P.StructMember`` / ``P.StructMemberUntyped``
  nodes; this dataclass is a flattened denormalised view that the
  symbol-table consumers expect.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ast_view import DataType  # noqa: F401  re-exported

__all__ = ["DataType", "StructMember"]


@dataclass
class StructMember:
    """Symbol-table record for one field of a structure type."""

    name: str
    data_type: DataType
    dimension: int | None = None
