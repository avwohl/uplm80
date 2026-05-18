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


def proc_local_decls_stmts(proc: P.ProcDecl) -> tuple[list, list]:
    """Split a procedure body into ``(local_decls, statements)``.

    Mirrors :func:`module_shape`'s declaration/statement split for a
    procedure body. Declarations include nested :class:`P.ProcDecl`
    nodes plus the flattened contents of any inner ``DECLARE`` statement
    (:class:`P.DeclItem` / :class:`P.DeclItemBasedGroup` /
    :class:`P.LiterallyDecl`). Everything else is treated as a
    statement and returned in source order.
    """
    decls: list = []
    stmts: list = []
    for it in proc.body.items:
        if isinstance(it, P.ProcDecl):
            decls.append(it)
        elif isinstance(it, P.DeclareStmt):
            decls.extend(it.declarations)
        else:
            stmts.append(it)
    return decls, stmts


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


# ---- declaration attribute / shape unpacking ------------------------------


@dataclass
class DeclAttrs:
    """Flattened view of a declaration's attribute clauses.

    ``initial_values`` and ``data_values`` are the raw typed expression
    nodes from the AST; ``at_location`` is the typed expression
    addressed by an ``AT(...)`` clause, or ``None``.
    """

    is_public: bool = False
    is_external: bool = False
    initial_values: Optional[list] = None
    data_values: Optional[list] = None
    at_location = None  # typed expression node | None


def _scan_attrs(attrs, out: DeclAttrs) -> None:
    """Apply each attribute clause's effect to ``out``."""
    for attr in attrs or []:
        if isinstance(attr, P.AttrExternal):
            out.is_external = True
        elif isinstance(attr, P.AttrPublic):
            out.is_public = True
        elif isinstance(attr, P.AttrInitial):
            out.initial_values = list(attr.values or [])
        elif isinstance(attr, P.AttrAt):
            out.at_location = attr.address


def decl_attrs(item) -> DeclAttrs:
    """Flatten a ``DeclItem`` / ``DeclItemBasedGroup`` tail's attribute
    clauses into a :class:`DeclAttrs` view.

    Walks both the type-tail's ``attrs`` and the DATA-tail's leading +
    trailing attribute lists, so the caller doesn't need to know which
    tail variant carries the attributes.
    """
    out = DeclAttrs()
    tail = getattr(item, "tail", None)
    if tail is None:
        return out
    if isinstance(tail, P.DeclTailType):
        _scan_attrs(tail.attrs, out)
    elif isinstance(tail, P.DeclTailTypeData):
        _scan_attrs(tail.leading_attrs, out)
        out.data_values = list(tail.data_values or [])
        _scan_attrs(tail.trailing_attrs, out)
    elif isinstance(tail, P.DeclTailData):
        _scan_attrs(tail.leading_attrs, out)
        out.data_values = list(tail.data_values or [])
        _scan_attrs(tail.trailing_attrs, out)
    elif isinstance(tail, P.DeclTailStructure):
        _scan_attrs(tail.attrs, out)
    elif isinstance(tail, P.DeclTailStructureData):
        _scan_attrs(tail.leading_attrs, out)
        out.data_values = list(tail.data_values or [])
        _scan_attrs(tail.trailing_attrs, out)
    return out


def decl_item_type(item) -> tuple[Optional[DataType], Optional[int]]:
    """Return ``(view DataType | None, dimension | None)`` for a typed
    ``DeclItem``. ``dimension`` is ``None`` for scalars, an int for
    fixed-size arrays, or ``-1`` for the ``(*)`` form. STRUCTURE /
    user-defined types return ``None`` for the data type — the caller
    consults :func:`decl_item_struct_members` to detect them."""
    dt: Optional[DataType] = None
    tail = getattr(item, "tail", None)
    type_node = getattr(tail, "type", None) if tail is not None else None
    if isinstance(type_node, (P.TypeByte, P.TypeByteSized)):
        dt = DataType.BYTE
    elif isinstance(type_node, (P.TypeAddress, P.TypeAddressSized)):
        dt = DataType.ADDRESS
    elif isinstance(type_node, P.TypeLabel):
        dt = DataType.LABEL
    dim = array_size_value(getattr(item, "array_size", None))
    return dt, dim


def decl_item_struct_members(item) -> Optional[list]:
    """Return the list of typed ``StructMember`` / ``StructMemberUntyped``
    nodes for a STRUCTURE declaration, else ``None``."""
    tail = getattr(item, "tail", None)
    if isinstance(tail, (P.DeclTailStructure, P.DeclTailStructureData)):
        return list(tail.members or [])
    return None


def dotted_ident_parts(node) -> list[str]:
    """Walk a ``DottedIdent`` / ``DottedMember`` chain into a list of
    identifier name strings, base first."""
    parts: list[str] = []
    while True:
        if isinstance(node, P.DottedIdent):
            parts.append(ident_text(node.name))
            return list(reversed(parts))
        if isinstance(node, P.DottedMember):
            parts.append(ident_text(node.member))
            node = node.base
            continue
        return list(reversed(parts))


def decl_item_based(item) -> tuple[Optional[str], Optional[str]]:
    """Return ``(base_name, member_name)`` for a ``BASED`` declaration's
    base reference, or ``(None, None)`` if the item is not based.

    ``base.member`` chains become ``("base", "member")``; deeper chains
    collapse the trailing members into a dotted string, since codegen
    currently expects a single member name."""
    based = getattr(item, "based", None)
    if based is None:
        return None, None
    parts = dotted_ident_parts(based.base)
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], ".".join(parts[1:])


def struct_member_names(m) -> list[str]:
    """Names declared by a single ``StructMember`` (one or many)."""
    names = m.names
    if isinstance(names, P.DeclName):
        return [ident_text(names.name)]
    if isinstance(names, P.DeclNames):
        return [ident_text(n.name) for n in (names.names or [])]
    raise TypeError(f"unexpected struct member names node: {type(names).__name__}")


def struct_member_type(m) -> DataType:
    """Type of a struct member; untyped members default to BYTE per
    PL/M-80 semantics."""
    if isinstance(m, P.StructMemberUntyped):
        return DataType.BYTE
    t = m.type
    if isinstance(t, (P.TypeByte, P.TypeByteSized)):
        return DataType.BYTE
    if isinstance(t, (P.TypeAddress, P.TypeAddressSized)):
        return DataType.ADDRESS
    # TypeLabel / TypeUserDefined: not legal in struct context; fall back to BYTE
    return DataType.BYTE


def struct_member_dim(m) -> Optional[int]:
    """Array dimension of a struct member (``None`` for scalars)."""
    return array_size_value(getattr(m, "array_size", None))


def literally_value(decl: P.LiterallyDecl) -> str:
    """Unquote a ``LITERALLY`` macro's replacement text."""
    text = decl.value.text
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    return text
