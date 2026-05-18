"""Stateless walker helpers over the uplox-generated typed PL/M-80 AST.

Codegen and the AST optimizer consume the new typed nodes directly
(no CST converter); these helpers handle the shape-unpacking that
the old hand-built AST used to bake in at parse time. They're pure
functions over :mod:`uplm80._plm_parser` nodes — no caching, no
side effects, safe to call repeatedly on the same node.

Convention: helpers named ``<kind>_<field>`` extract a single named
piece; ``iter_<plural>`` yield items; ``is_<flag>`` answer a boolean
question. Operator-kind enums (:class:`BinaryOpKind` / :class:`UnaryOpKind`)
mirror the legacy ``BinaryOp`` / ``UnaryOp`` enums so existing codegen
dispatch on ``op_kind(expr) == BinaryOpKind.ADD`` reads naturally.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator, Optional

from . import _plm_parser as P


# ---- value enums (mirror the legacy ast_nodes shape) -----------------------


class DataType(Enum):
    """PL/M-80 data types — kept as an enum for codegen dispatch."""

    BYTE = auto()
    ADDRESS = auto()
    LABEL = auto()
    PROCEDURE = auto()


class BinaryOpKind(Enum):
    """Operator kind for :class:`uplm80._plm_parser.BinaryOp` nodes.

    The grammar carries the operator as a Token; map to this enum at
    consumption time via :func:`binop_kind`.
    """

    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    AND = auto()
    OR = auto()
    XOR = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    PLUS = auto()   # carry-aware
    MINUS = auto()  # carry-aware


class UnaryOpKind(Enum):
    """Operator kind for :class:`uplm80._plm_parser.UnaryOp` nodes."""

    NEG = auto()
    NOT = auto()


_BINOP_TOKEN_TO_KIND = {
    "OP_PLUS": BinaryOpKind.ADD,
    "OP_MINUS": BinaryOpKind.SUB,
    "STAR": BinaryOpKind.MUL,
    "SLASH": BinaryOpKind.DIV,
    "KW_MOD": BinaryOpKind.MOD,
    "KW_AND": BinaryOpKind.AND,
    "KW_OR": BinaryOpKind.OR,
    "KW_XOR": BinaryOpKind.XOR,
    "EQ": BinaryOpKind.EQ,
    "NE": BinaryOpKind.NE,
    "LT": BinaryOpKind.LT,
    "GT": BinaryOpKind.GT,
    "LE": BinaryOpKind.LE,
    "GE": BinaryOpKind.GE,
    "KW_PLUS": BinaryOpKind.PLUS,
    "KW_MINUS": BinaryOpKind.MINUS,
}


def binop_kind(expr: P.BinaryOp) -> BinaryOpKind:
    """Map a ``BinaryOp`` node's operator token to its enum kind."""
    return _BINOP_TOKEN_TO_KIND[expr.op.name]


def unop_kind(expr: P.UnaryOp) -> UnaryOpKind:
    """Map a ``UnaryOp`` node's operator token to its enum kind."""
    name = expr.op.name
    if name == "OP_MINUS":
        return UnaryOpKind.NEG
    if name == "KW_NOT":
        return UnaryOpKind.NOT
    raise ValueError(f"unknown unary operator token: {name}")


# ---- identifier text normalisation ----------------------------------------


def ident_text(tok) -> str:
    """Strip the PL/M ``$`` break-character from an identifier token.

    PL/M-80 allows ``PRINT$CHAR`` and ``PRINTCHAR`` to refer to the
    same name. Canonicalise on the form with the ``$`` removed.
    """
    return tok.text.replace("$", "")


# ---- numeric literal parsing ----------------------------------------------


def parse_plm_number(text: str) -> int:
    """Parse a PL/M-80 numeric literal token text to int.

    Accepts decimal, hex (``H`` / ``h``), octal (``O`` / ``Q``),
    and binary (``B``) suffixes. ``$`` separators are stripped.
    """
    s = text.replace("$", "")
    if not s:
        return 0
    suffix = s[-1].upper()
    if suffix == "H":
        return int(s[:-1], 16)
    if suffix in ("O", "Q"):
        return int(s[:-1], 8)
    if suffix == "B":
        return int(s[:-1], 2)
    if suffix == "D":
        return int(s[:-1], 10)
    return int(s, 10)


def number_value(node: P.NumberLiteral) -> int:
    """Lift a ``NumberLiteral`` to its Python int value."""
    return parse_plm_number(node.value.text)


# ---- string literal parsing -----------------------------------------------


def string_value(node: P.StringLiteral) -> str:
    """Decode a ``STRING`` token's text into the underlying string.

    PL/M strings are surrounded by single quotes; the escape for a
    literal quote is ``''``. The grammar regex preserves the quotes
    in the token text.
    """
    raw = node.value.text
    if raw.startswith("'") and raw.endswith("'"):
        raw = raw[1:-1]
    return raw.replace("''", "'")


def string_bytes(node: P.StringLiteral) -> list[int]:
    """ASCII byte values for a ``StringLiteral`` (no NUL terminator)."""
    return [ord(c) for c in string_value(node)]


# ---- module shape ----------------------------------------------------------


@dataclass
class ModuleShape:
    """Codegen-friendly view onto a :class:`P.Module` after legacy
    unwrap rules: a top-level ``NAME: DO; ... END NAME;`` block is
    treated as the module's body, with ``NAME`` becoming the module
    name and the body's items becoming the module's items. A leading
    ``NUMBER:`` (``100H:``) becomes the module origin.
    """

    name: str
    origin: Optional[int]
    decls: list  # ProcDecl / DeclItem / DeclItemBasedGroup / LiterallyDecl
    stmts: list  # everything else


def module_shape(m: P.Module, default_name: str = "<input>") -> ModuleShape:
    items = list(m.items)
    name = default_name
    # Unwrap a single top-level labeled DO block.
    if len(items) == 1 and isinstance(items[0], P.LabeledStmt) and isinstance(
        items[0].stmt, P.DoBlock
    ):
        name = ident_text(items[0].label)
        items = list(items[0].stmt.items)

    # Origin literal: a leading bare `NUMBER:` becomes the module origin.
    origin: Optional[int] = None
    if items and isinstance(items[0], P.AddressLiteral):
        origin = parse_plm_number(items[0].value.text)
        items = items[1:]

    # Split body items: declarations vs statements. Declarations are
    # the ones the legacy code carried in Module.decls (ProcDecl,
    # LiterallyDecl, plus any items inside a DeclareStmt).
    decls: list = []
    stmts: list = []
    for it in items:
        if isinstance(it, P.ProcDecl):
            decls.append(it)
        elif isinstance(it, P.DeclareStmt):
            decls.extend(it.declarations)
        else:
            stmts.append(it)
    return ModuleShape(name=name, origin=origin, decls=decls, stmts=stmts)


# ---- procedure attribute unpacking ----------------------------------------


@dataclass
class ProcAttrs:
    """Flattened view of a procedure's signature attribute clauses."""

    is_external: bool = False
    is_public: bool = False
    is_reentrant: bool = False
    interrupt_num: Optional[int] = None


def proc_attrs(proc: P.ProcDecl) -> ProcAttrs:
    """Walk ``proc.signature.attrs`` into a flat attribute view."""
    out = ProcAttrs()
    for attr in proc.signature.attrs or []:
        if isinstance(attr, P.ProcAttrExternal):
            out.is_external = True
        elif isinstance(attr, P.ProcAttrPublic):
            out.is_public = True
        elif isinstance(attr, P.ProcAttrReentrant):
            out.is_reentrant = True
        elif isinstance(attr, P.ProcAttrInterrupt):
            out.interrupt_num = parse_plm_number(attr.number.text)
    return out


def proc_name(proc: P.ProcDecl) -> str:
    return ident_text(proc.name)


def proc_param_names(proc: P.ProcDecl) -> list[str]:
    """List of parameter names, in declaration order, ``$`` stripped."""
    params_node = proc.signature.params
    if params_node is None:
        return []
    return [ident_text(n.name) for n in (params_node.names or [])]


def proc_return_type(proc: P.ProcDecl) -> Optional[DataType]:
    """``DataType.BYTE`` / ``DataType.ADDRESS`` / ``None`` for an
    untyped procedure."""
    rt = proc.signature.return_type
    if rt is None:
        return None
    kw = rt.kw.name
    if kw == "KW_BYTE":
        return DataType.BYTE
    if kw == "KW_ADDRESS":
        return DataType.ADDRESS
    return None


def proc_body_items(proc: P.ProcDecl) -> list:
    """Items inside the procedure body in source order."""
    return list(proc.body.items)


def proc_end_label(proc: P.ProcDecl) -> Optional[str]:
    el = proc.body.end_label
    return ident_text(el.name) if el is not None else None


# ---- DECLARE statement / decl_item helpers --------------------------------


def iter_declare_items(stmt: P.DeclareStmt) -> Iterator:
    """Yield each top-level declaration in a ``DECLARE`` statement."""
    for d in stmt.declarations or []:
        yield d


def decl_item_names(item: P.DeclItem) -> list[str]:
    """Names declared by a single ``DeclItem`` (``A`` or ``(A, B, C)``)."""
    names = item.names
    if isinstance(names, P.DeclName):
        return [ident_text(names.name)]
    if isinstance(names, P.DeclNames):
        return [ident_text(n.name) for n in (names.names or [])]
    raise TypeError(f"unexpected decl_names node: {type(names).__name__}")


def array_size_value(node) -> Optional[int]:
    """``None`` for scalar; an int for fixed-size arrays; ``-1`` for
    the implicit ``(*)`` form. ``size`` may be a ``SizeIdent`` (a
    LITERALLY-expanded macro name) — return ``None`` and let the
    caller resolve through the symbol table."""
    if node is None:
        return None
    if isinstance(node, P.ArraySizeStar):
        return -1
    if isinstance(node, P.ArraySize):
        sz = node.size
        if isinstance(sz, P.SizeNumber):
            return parse_plm_number(sz.value.text)
        # SizeIdent: unresolved at the syntax level
        return None
    return None
