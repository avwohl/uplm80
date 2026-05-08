"""plox-driven PL/M-80 front-end.

Pipeline per source file:

1. ``preprocess.preprocess`` — uplm80's own preprocessor: high-bit
   strip, recursive ``$INCLUDE``, ``$cond/$if/...`` conditional
   compilation.
2. ``plox.preprocess.plm.preprocess`` — LITERALLY substitution,
   EQU/LIT alias bootstrap, harmless ``$``-directive stripping,
   case folding to upper.
3. plox plm_full LR(1) parse — produces a ``ParseNode`` tree.
4. ``_convert_module`` — lower the ``ParseNode`` tree into uplm80's
   ``ast_nodes`` dataclasses (``Module``, ``ProcDecl``, ``DeclareStmt``,
   ``IfStmt``, ``DoBlock``, ``BinaryExpr``, ...).

The plm_full LR table is loaded once at import time from the JSON bundle
in ``uplm80/data/plm_full.json`` (built by ``plox build``). Building from
the grammar source takes ~2 s; loading the JSON takes ~70 ms.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import cast

from plox.lex.scanner import Scanner, Token
from plox.parse.runtime import HookRegistry, ParseNode, parse as _plox_parse
from plox.tables import balanced_from_json, dfa_from_json, table_from_json

from .ast_nodes import (
    AssignStmt,
    BinaryExpr,
    BinaryOp,
    CallExpr,
    CallStmt,
    ConstListExpr,
    DataType,
    Declaration,
    DeclareStmt,
    DisableStmt,
    DoBlock,
    DoCaseBlock,
    DoIterBlock,
    DoWhileBlock,
    EmbeddedAssignExpr,
    EnableStmt,
    Expr,
    GotoStmt,
    HaltStmt,
    Identifier,
    IfStmt,
    LabelDecl,
    LabeledStmt,
    LiterallyDecl,
    LocationExpr,
    MemberExpr,
    Module,
    NullStmt,
    NumberLiteral,
    ProcDecl,
    ReturnStmt,
    SourceSpan,
    Stmt,
    StringLiteral,
    StructMember,
    SubscriptExpr,
    UnaryExpr,
    UnaryOp,
    VarDecl,
)
from .errors import ParserError, SourceLocation
from .preprocess import macro_pass, preprocess as uplm_preprocess


_BUNDLE_PATH = Path(__file__).parent / "data" / "plm_full.json"


@lru_cache(maxsize=1)
def _load_plm_full() -> tuple[Scanner, object]:
    """Load the prebuilt plm_full lexer + LR table from the JSON bundle.
    Cached for the lifetime of the process."""
    with open(_BUNDLE_PATH) as f:
        bundle = json.load(f)
    dfa, _tokens, skip_list = dfa_from_json(bundle["lex"])
    balanced = balanced_from_json(bundle["lex"].get("balanced", {}))
    scanner = Scanner(dfa=dfa, skip_tokens=frozenset(skip_list), balanced=balanced)
    table = table_from_json(bundle["parse"])
    return scanner, table


def parse_source(
    source: str,
    filename: str = "<input>",
    defines: list[str] | None = None,
    include_paths: list[str] | None = None,
) -> Module:
    """Run the full PL/M-80 front-end pipeline and return an uplm80
    :class:`Module`."""
    pre1 = uplm_preprocess(source, filename, defines=defines, include_paths=include_paths)
    # Block-scoped LITERALLY substitution + case folding + harmless
    # ``$``-directive line stripping. Lives in uplm80 (not plox)
    # because PL/M LITERALLYs are scoped to the enclosing
    # ``DO``/``PROCEDURE`` block — PIP.PLM defines ``M LITERALLY '20'``
    # inside one block then reuses ``M`` as a variable name in another.
    src = macro_pass(pre1)
    scanner, table = _load_plm_full()
    try:
        tree = _plox_parse(
            table,
            scanner.scan(src),
            hooks=HookRegistry(ignore_missing=True),
        )
    except Exception as e:  # ScanError or ParseError
        raise ParserError(str(e), SourceLocation(1, 1, filename)) from e
    if not isinstance(tree, ParseNode):
        raise ParserError("plm_full parse returned non-ParseNode", SourceLocation(1, 1, filename))
    return _convert_module(tree, filename)


# ---------------------------------------------------------------------------
# ParseNode -> uplm80 AST conversion
#
# plox emits a verbose CST (one ParseNode per production); we lower it
# to uplm80's hand-written AST. The conversion is rule-driven: each
# function below maps one or two related grammar non-terminals.
# ---------------------------------------------------------------------------


def _kind(n: ParseNode | Token) -> str:
    return n.kind if isinstance(n, ParseNode) else n.name


def _is_token(n: ParseNode | Token, name: str | None = None) -> bool:
    if not isinstance(n, Token):
        return False
    return name is None or n.name == name


def _tok_text(n: ParseNode | Token) -> str:
    assert isinstance(n, Token), f"expected token, got {type(n).__name__}: {n}"
    return n.text


def _ident_text(n: ParseNode | Token) -> str:
    """IDENT name with PL/M ``$`` break-characters stripped.

    PL/M-80 lets ``$`` be sprinkled into identifiers as a readability
    separator (``PRINT$CHAR`` == ``PRINTCHAR``). The legacy lexer
    normalised at tokenize time; we do it here so the rest of the
    compiler sees a single canonical spelling per identifier."""
    return _tok_text(n).replace("$", "")


def _span_of(node: ParseNode | Token, filename: str) -> SourceSpan | None:
    """Build a SourceSpan covering ``node``. Walks to leftmost / rightmost
    tokens. Returns None if the subtree is empty."""
    first = _first_token(node)
    last = _last_token(node)
    if first is None or last is None:
        return None
    return SourceSpan(
        start_line=first.line,
        start_col=first.column,
        end_line=last.line,
        end_col=last.column + len(last.text),
        filename=filename,
    )


def _first_token(node: ParseNode | Token) -> Token | None:
    if isinstance(node, Token):
        return node
    for c in node.children:
        t = _first_token(c)
        if t is not None:
            return t
    return None


def _last_token(node: ParseNode | Token) -> Token | None:
    if isinstance(node, Token):
        return node
    for c in reversed(node.children):
        t = _last_token(c)
        if t is not None:
            return t
    return None


# --- module / items --------------------------------------------------------


def _flatten_items(node: ParseNode) -> list[ParseNode | Token]:
    """Flatten the left-recursive ``<items>`` chain into a list of
    ``<item>`` nodes."""
    out: list[ParseNode | Token] = []
    stack = [node]
    while stack:
        n = stack.pop()
        if not isinstance(n, ParseNode):
            continue
        if n.kind == "items":
            # items : items item | item
            for c in reversed(n.children):
                stack.append(c)
            continue
        if n.kind == "item":
            out.append(n)
            continue
    return out


def _flatten_body_items(node: ParseNode) -> list[ParseNode | Token]:
    """Flatten the left-recursive ``<body_items>`` chain into a list of
    ``<body_item>`` nodes."""
    out: list[ParseNode | Token] = []
    stack = [node]
    while stack:
        n = stack.pop()
        if not isinstance(n, ParseNode):
            continue
        if n.kind == "body_items":
            for c in reversed(n.children):
                stack.append(c)
            continue
        if n.kind == "body_item":
            out.append(n)
            continue
    return out


def _convert_module(root: ParseNode, filename: str) -> Module:
    """Lower ``<module>`` to uplm80's :class:`Module`.

    plm_full returns a flat list of items. We synthesise the module-level
    metadata: a leading address literal becomes ``origin``; a single
    top-level labeled DO block (``NAME: DO; ... END NAME;``) is unwrapped
    so its inner declarations and statements live directly on the Module
    and ``NAME`` becomes the module name; otherwise the first procedure
    name (or ``"MODULE"``) is used.
    """
    # module : items | items EOF
    items_node = root.children[0]
    assert isinstance(items_node, ParseNode) and items_node.kind == "items"
    items = _flatten_items(items_node)

    origin: int | None = None
    name = "MODULE"
    decls: list[Declaration] = []
    stmts: list[Stmt] = []

    idx = 0
    if items and _item_is_address_literal(items[0]):
        origin = _convert_address_literal(items[0])
        idx = 1

    # Detect single-wrapper labeled DO at module level: exactly one item
    # remaining whose stmt is `IDENT ':' do_block(DO ; ... END NAME ;)`.
    rest = items[idx:]
    unwrapped = _try_unwrap_module_do(rest)
    if unwrapped is not None:
        name, body_items = unwrapped
        for bi in body_items:
            _emit_body_item(bi, decls, stmts, filename)
        # Module top level: lift ProcDecls (and any other DeclareStmt-
        # wrapped declarations) out of stmts into decls so nested-vs-
        # top scoping matches legacy parse_module.
        flat_stmts: list[Stmt] = []
        for s in stmts:
            if isinstance(s, DeclareStmt):
                decls.extend(s.declarations)
            else:
                flat_stmts.append(s)
        stmts[:] = flat_stmts
    else:
        for it in rest:
            _emit_top_item(it, decls, stmts, filename)
        # If first decl is a ProcDecl, use its name as module name.
        for d in decls:
            if isinstance(d, ProcDecl):
                name = d.name
                break

    return Module(
        name=name,
        origin=origin,
        decls=decls,
        stmts=stmts,
        span=_span_of(root, filename),
    )


def _item_is_address_literal(item: ParseNode | Token) -> bool:
    if not isinstance(item, ParseNode) or item.kind != "item":
        return False
    inner = item.children[0]
    return isinstance(inner, ParseNode) and inner.kind == "address_literal"


def _convert_address_literal(item: ParseNode | Token) -> int:
    assert isinstance(item, ParseNode)
    inner = item.children[0]
    assert isinstance(inner, ParseNode) and inner.kind == "address_literal"
    # address_literal : NUMBER ':'
    num_tok = inner.children[0]
    return _parse_plm_number(_tok_text(num_tok))


def _try_unwrap_module_do(
    items: list[ParseNode | Token],
) -> tuple[str, list[ParseNode | Token]] | None:
    """Recognize the canonical ``NAME: DO; ... END NAME;`` module wrapper.

    Returns ``(name, [body_item, ...])`` if this is a single labeled DO
    block at module scope, else None.
    """
    if len(items) != 1:
        return None
    item = items[0]
    if not isinstance(item, ParseNode) or item.kind != "item":
        return None
    inner = item.children[0]
    if not (isinstance(inner, ParseNode) and inner.kind == "stmt"):
        return None
    # stmt -> matched_stmt -> IDENT ':' matched_stmt -> non_if_stmt -> do_block
    matched = inner.children[0]
    if not (isinstance(matched, ParseNode) and matched.kind == "matched_stmt"):
        return None
    if not (
        len(matched.children) == 3
        and _is_token(matched.children[0], "IDENT")
        and _is_token(matched.children[1], "COLON")
    ):
        return None
    name = _tok_text(matched.children[0])
    inner2 = matched.children[2]
    do_block = _peek_do_block(inner2)
    if do_block is None:
        return None
    # Must be the plain `DO ;` form (5 children + trailer)
    children = do_block.children
    if not (len(children) >= 4 and _is_token(children[0], "KW_DO") and _is_token(children[1], "SEMI")):
        return None
    body_items_node = children[2]
    if not (isinstance(body_items_node, ParseNode) and body_items_node.kind == "body_items"):
        return None
    return name, _flatten_body_items(body_items_node)


def _peek_do_block(node: ParseNode | Token) -> ParseNode | None:
    """Walk through stmt -> matched_stmt -> non_if_stmt -> do_block when
    present; else None."""
    cur: ParseNode | Token = node
    while isinstance(cur, ParseNode):
        if cur.kind == "do_block":
            return cur
        if cur.kind in ("stmt", "matched_stmt", "non_if_stmt") and len(cur.children) == 1:
            cur = cur.children[0]
            continue
        return None
    return None


def _emit_top_item(
    item: ParseNode | Token, decls: list[Declaration], stmts: list[Stmt], filename: str
) -> None:
    assert isinstance(item, ParseNode) and item.kind == "item"
    inner = item.children[0]
    assert isinstance(inner, ParseNode)
    if inner.kind == "address_literal":
        # Already handled at module top, but tolerate inline.
        return
    if inner.kind == "proc_decl":
        decls.append(_convert_proc_decl(inner, filename))
        return
    if inner.kind == "declare_stmt":
        for d in _convert_declare(inner, filename):
            decls.append(d)
        return
    if inner.kind == "stmt":
        stmts.append(_convert_stmt(inner, filename))
        return
    raise ParserError(f"Unexpected top-level item: {inner.kind}")


def _emit_body_item(
    item: ParseNode | Token, decls: list[Declaration], stmts: list[Stmt], filename: str
) -> None:
    """Body-item emit for *nested* contexts (proc body, DO block).

    Nested ProcDecls are wrapped in a single-item ``DeclareStmt`` and
    appended to ``stmts`` so source order between procedure declarations
    and ordinary statements is preserved — the legacy uplm80 parser did
    the same and the codegen relies on that order to emit the
    block-scoped name prefix (``@Bnn$NAME``) for nested procs. Plain
    ``DECLARE`` (variables, labels, LITERALLY) collects into ``decls``.
    """
    assert isinstance(item, ParseNode) and item.kind == "body_item"
    inner = item.children[0]
    assert isinstance(inner, ParseNode)
    if inner.kind == "proc_decl":
        proc = _convert_proc_decl(inner, filename)
        stmts.append(DeclareStmt(declarations=[proc], span=proc.span))
        return
    if inner.kind == "declare_stmt":
        for d in _convert_declare(inner, filename):
            decls.append(d)
        return
    if inner.kind == "stmt":
        stmts.append(_convert_stmt(inner, filename))
        return
    raise ParserError(f"Unexpected body item: {inner.kind}")


# --- procedures ------------------------------------------------------------


def _convert_proc_decl(node: ParseNode, filename: str) -> ProcDecl:
    """proc_decl : IDENT ':' PROCEDURE proc_signature proc_body"""
    name = _ident_text(node.children[0])
    sig = node.children[3]
    body = node.children[4]
    assert isinstance(sig, ParseNode) and sig.kind == "proc_signature"
    assert isinstance(body, ParseNode) and body.kind == "proc_body"

    # proc_signature : proc_params proc_return proc_attrs ';'
    params = _convert_proc_params(sig.children[0])
    return_type = _convert_proc_return(sig.children[1])
    is_public = is_external = is_reentrant = False
    interrupt_num: int | None = None
    attrs = sig.children[2]
    assert isinstance(attrs, ParseNode) and attrs.kind == "proc_attrs"
    for at in _flatten_attrs_list(attrs, "proc_attrs", "proc_attr"):
        kw = at.children[0]
        kn = _kind(kw)
        if kn == "KW_EXTERNAL":
            is_external = True
        elif kn == "KW_PUBLIC":
            is_public = True
        elif kn == "KW_REENTRANT":
            is_reentrant = True
        elif kn == "KW_INTERRUPT":
            interrupt_num = _parse_plm_number(_tok_text(at.children[1]))

    # proc_body : body_items END opt_end_label ';'
    body_items_node = body.children[0]
    assert isinstance(body_items_node, ParseNode) and body_items_node.kind == "body_items"
    decls: list[Declaration] = []
    stmts: list[Stmt] = []
    for bi in _flatten_body_items(body_items_node):
        _emit_body_item(bi, decls, stmts, filename)

    return ProcDecl(
        name=name,
        params=params,
        return_type=return_type,
        is_public=is_public,
        is_external=is_external,
        is_reentrant=is_reentrant,
        interrupt_num=interrupt_num,
        decls=decls,
        stmts=stmts,
        span=_span_of(node, filename),
    )


def _convert_proc_params(node: ParseNode | Token) -> list[str]:
    assert isinstance(node, ParseNode) and node.kind == "proc_params"
    if len(node.children) == 0:
        return []
    # '(' ident_list ')' or '(' ')'
    if len(node.children) == 2:
        return []
    ident_list = node.children[1]
    return _flatten_ident_list(ident_list)


def _convert_proc_return(node: ParseNode | Token) -> DataType | None:
    assert isinstance(node, ParseNode) and node.kind == "proc_return"
    if not node.children:
        return None
    kw = node.children[0]
    return _type_from_kw(_kind(kw))


def _flatten_ident_list(node: ParseNode | Token) -> list[str]:
    out: list[str] = []
    stack: list[ParseNode | Token] = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Token):
            if n.name == "IDENT":
                out.append(_ident_text(n))
            continue
        # ident_list : IDENT | ident_list ',' IDENT
        for c in n.children:
            stack.append(c)
    # Stack pops produce reversed order — re-collect in source order.
    return list(reversed(out))


def _flatten_attrs_list(
    node: ParseNode, list_kind: str, item_kind: str
) -> list[ParseNode]:
    """Flatten left-recursive ``X : X Y |`` into a list of ``Y`` nodes
    in source order."""
    out: list[ParseNode] = []
    stack: list[ParseNode | Token] = [node]
    while stack:
        n = stack.pop()
        if not isinstance(n, ParseNode):
            continue
        if n.kind == list_kind:
            for c in reversed(n.children):
                stack.append(c)
            continue
        if n.kind == item_kind:
            out.append(n)
    return out


# --- declarations ----------------------------------------------------------


def _convert_declare(node: ParseNode, filename: str) -> list[Declaration]:
    """declare_stmt : DECLARE decl_list ';'"""
    decl_list = node.children[1]
    assert isinstance(decl_list, ParseNode) and decl_list.kind == "decl_list"
    return _convert_decl_list(decl_list, filename)


def _convert_decl_list(node: ParseNode, filename: str) -> list[Declaration]:
    """decl_list : decl_list ',' decl_item | decl_item — flatten and convert each."""
    items: list[ParseNode] = []
    stack: list[ParseNode | Token] = [node]
    while stack:
        n = stack.pop()
        if not isinstance(n, ParseNode):
            continue
        if n.kind == "decl_list":
            for c in reversed(n.children):
                stack.append(c)
            continue
        if n.kind == "decl_item":
            items.append(n)
    out: list[Declaration] = []
    for it in items:
        out.extend(_convert_decl_item(it, filename))
    return out


def _convert_decl_item(node: ParseNode, filename: str) -> list[Declaration]:
    """decl_item is one of:
    1. decl_names based_opt array_size_opt decl_tail
    2. '(' based_decl_list ')' array_size_opt decl_tail
    3. IDENT LITERALLY STRING
    """
    children = node.children
    span = _span_of(node, filename)

    # Form 3: IDENT LITERALLY STRING
    if (
        len(children) == 3
        and _is_token(children[0], "IDENT")
        and _is_token(children[1], "KW_LITERALLY")
        and _is_token(children[2], "STRING")
    ):
        name = _ident_text(children[0])
        body = _strip_string(_tok_text(children[2]))
        return [LiterallyDecl(name=name, value=body, span=span)]

    # Form 2: '(' based_decl_list ')' array_size_opt decl_tail
    if _is_token(children[0], "LPAREN") and isinstance(children[1], ParseNode) and children[1].kind == "based_decl_list":
        bdl = children[1]
        array_size = _convert_array_size_opt(children[3])
        decl_tail_info = _convert_decl_tail(children[4], filename)
        names_with_base = _flatten_based_decl_list(bdl)
        decls: list[Declaration] = []
        for nm, base, base_member in names_with_base:
            decls.append(
                _build_var_decl(
                    nm,
                    based_on=base,
                    based_member=base_member,
                    array_size=array_size,
                    tail=decl_tail_info,
                    span=span,
                )
            )
        return decls

    # Form 1: decl_names based_opt array_size_opt decl_tail
    decl_names_node = children[0]
    based_opt = children[1]
    array_size_opt = children[2]
    decl_tail = children[3]
    names = _flatten_decl_names(decl_names_node)
    based_on, based_member = _convert_based_opt(based_opt)
    array_size = _convert_array_size_opt(array_size_opt)
    decl_tail_info = _convert_decl_tail(decl_tail, filename)

    # Special case: LABEL type
    if decl_tail_info.is_label:
        return [
            LabelDecl(
                name=nm,
                is_public=decl_tail_info.is_public,
                is_external=decl_tail_info.is_external,
                span=span,
            )
            for nm in names
        ]

    return [
        _build_var_decl(
            nm,
            based_on=based_on,
            based_member=based_member,
            array_size=array_size,
            tail=decl_tail_info,
            span=span,
        )
        for nm in names
    ]


def _flatten_decl_names(node: ParseNode | Token) -> list[str]:
    """decl_names : IDENT | '(' ident_list ')'"""
    assert isinstance(node, ParseNode) and node.kind == "decl_names"
    if len(node.children) == 1:
        return [_ident_text(node.children[0])]
    return _flatten_ident_list(node.children[1])


def _flatten_based_decl_list(node: ParseNode) -> list[tuple[str, str, str | None]]:
    """based_decl_list : IDENT BASED dotted_ident
                       | based_decl_list ',' IDENT BASED dotted_ident"""
    out: list[tuple[str, str, str | None]] = []
    stack: list[ParseNode | Token] = [node]
    chunks: list[ParseNode] = []
    while stack:
        n = stack.pop()
        if not isinstance(n, ParseNode):
            continue
        if n.kind == "based_decl_list":
            for c in reversed(n.children):
                stack.append(c)
            continue
    # Easier: walk recursively to flatten
    def visit(n: ParseNode) -> None:
        if n.kind != "based_decl_list":
            return
        cs = n.children
        if len(cs) == 3:
            # IDENT BASED dotted_ident
            name = _ident_text(cs[0])
            base, member = _convert_dotted_ident(cs[2])
            out.append((name, base, member))
        elif len(cs) == 5:
            # based_decl_list ',' IDENT BASED dotted_ident
            assert isinstance(cs[0], ParseNode)
            visit(cs[0])
            name = _ident_text(cs[2])
            base, member = _convert_dotted_ident(cs[4])
            out.append((name, base, member))
    visit(node)
    return out


def _convert_dotted_ident(node: ParseNode | Token) -> tuple[str, str | None]:
    """dotted_ident : IDENT | dotted_ident '.' IDENT — returns (head, optional .tail).
    Multi-level dots collapse; uplm80's based_member is single-segment."""
    parts: list[str] = []
    def visit(n: ParseNode | Token) -> None:
        if isinstance(n, Token):
            if n.name == "IDENT":
                parts.append(_ident_text(n))
            return
        for c in n.children:
            visit(c)
    visit(node)
    if not parts:
        return ("", None)
    if len(parts) == 1:
        return (parts[0], None)
    return (parts[0], ".".join(parts[1:]))


def _convert_based_opt(node: ParseNode | Token) -> tuple[str | None, str | None]:
    assert isinstance(node, ParseNode) and node.kind == "based_opt"
    if not node.children:
        return None, None
    # BASED dotted_ident
    base, member = _convert_dotted_ident(node.children[1])
    return base, member


def _convert_array_size_opt(node: ParseNode | Token) -> int | None:
    """array_size_opt : '(' size_expr ')' | '(' '*' ')' |"""
    assert isinstance(node, ParseNode) and node.kind == "array_size_opt"
    if not node.children:
        return None
    if len(node.children) == 3 and _is_token(node.children[1], "STAR"):
        return -1  # implicit
    inner = node.children[1]
    # size_expr : NUMBER | IDENT
    if isinstance(inner, ParseNode):
        first = inner.children[0]
        if _is_token(first, "NUMBER"):
            return _parse_plm_number(_tok_text(first))
        return -2  # unresolved IDENT (legacy sentinel)
    if _is_token(inner, "NUMBER"):
        return _parse_plm_number(_tok_text(inner))
    return -2


# --- decl tail (type + initial / data / structure) ---


from dataclasses import dataclass


@dataclass
class _DeclTail:
    data_type: DataType | None = None
    type_dimension: int | None = None  # BYTE(n) / ADDRESS(n)
    type_alias: str | None = None  # IDENT type (struct alias)
    struct_members: list[StructMember] | None = None
    is_public: bool = False
    is_external: bool = False
    is_label: bool = False
    initial_values: list[Expr] | None = None
    data_values: list[Expr] | None = None
    at_location: Expr | None = None


def _convert_decl_tail(node: ParseNode | Token, filename: str) -> _DeclTail:
    """decl_tail :
    | type_spec attrs_opt
    | type_spec attrs_opt DATA '(' arg_list ')' attrs_opt
    | DATA '(' arg_list ')' attrs_opt
    | STRUCTURE '(' struct_member_list ')' attrs_opt
    | STRUCTURE '(' struct_member_list ')' attrs_opt DATA '(' arg_list ')' attrs_opt
    """
    assert isinstance(node, ParseNode) and node.kind == "decl_tail"
    tail = _DeclTail()
    cs = node.children

    if isinstance(cs[0], ParseNode) and cs[0].kind == "type_spec":
        _apply_type_spec(cs[0], tail)
        i = 1
    elif isinstance(cs[0], ParseNode) and cs[0].kind == "attrs_opt" and len(cs) >= 6 and _is_token(cs[1], "KW_DATA"):
        # attrs_opt DATA '(' arg_list ')' attrs_opt — type is borrowed
        # from a sibling decl_item (e.g. `name public data(...)` after
        # `status$pd process$header, ...` in MP/M sources).
        _apply_attrs_opt(cs[0], tail, filename)
        tail.data_type = DataType.BYTE  # PL/M-80 untyped DATA defaults to BYTE
        tail.data_values = _convert_arg_list(cs[3], filename)
        _apply_attrs_opt(cs[5], tail, filename)
        return tail
    elif _is_token(cs[0], "KW_STRUCTURE"):
        # STRUCTURE '(' struct_member_list ')' attrs_opt [DATA ...]
        tail.struct_members = _convert_struct_member_list(cs[2])
        _apply_attrs_opt(cs[4], tail, filename)
        if len(cs) > 5 and _is_token(cs[5], "KW_DATA"):
            tail.data_values = _convert_arg_list(cs[7], filename)
            _apply_attrs_opt(cs[9], tail, filename)
        return tail
    else:
        raise ParserError(f"Unexpected decl_tail head: {_kind(cs[0])}")

    # type_spec attrs_opt [DATA '(' arg_list ')' attrs_opt]
    _apply_attrs_opt(cs[i], tail, filename)
    i += 1
    if i < len(cs) and _is_token(cs[i], "KW_DATA"):
        tail.data_values = _convert_arg_list(cs[i + 2], filename)
        _apply_attrs_opt(cs[i + 4], tail, filename)
    return tail


def _apply_type_spec(node: ParseNode, tail: _DeclTail) -> None:
    """type_spec : BYTE | ADDRESS | LABEL | BYTE '(' size_expr ')' |
    ADDRESS '(' size_expr ')' | IDENT"""
    cs = node.children
    head = cs[0]
    if _is_token(head, "KW_BYTE"):
        tail.data_type = DataType.BYTE
    elif _is_token(head, "KW_ADDRESS"):
        tail.data_type = DataType.ADDRESS
    elif _is_token(head, "KW_LABEL"):
        tail.is_label = True
        tail.data_type = DataType.LABEL
    elif _is_token(head, "IDENT"):
        tail.type_alias = _ident_text(head)
        tail.data_type = None
    else:
        raise ParserError(f"Unexpected type_spec head: {_kind(head)}")

    if len(cs) >= 4 and _is_token(cs[1], "LPAREN"):
        size = cs[2]
        if isinstance(size, ParseNode):
            inner = size.children[0]
            if _is_token(inner, "NUMBER"):
                tail.type_dimension = _parse_plm_number(_tok_text(inner))


def _apply_attrs_opt(node: ParseNode | Token, tail: _DeclTail, filename: str) -> None:
    """attrs_opt : attrs_opt attr_clause |"""
    if not isinstance(node, ParseNode) or not node.children:
        return
    for at in _flatten_attrs_list(node, "attrs_opt", "attr_clause"):
        head = at.children[0]
        kn = _kind(head)
        if kn == "KW_EXTERNAL":
            tail.is_external = True
        elif kn == "KW_PUBLIC":
            tail.is_public = True
        elif kn == "KW_INITIAL":
            tail.initial_values = _convert_arg_list(at.children[2], filename)
        elif kn == "KW_AT":
            tail.at_location = _convert_expr(at.children[2], filename)


def _convert_arg_list(node: ParseNode | Token, filename: str) -> list[Expr]:
    """arg_list : arg_list ',' expr | expr"""
    out: list[Expr] = []
    def visit(n: ParseNode | Token) -> None:
        if not isinstance(n, ParseNode):
            return
        if n.kind == "arg_list":
            cs = n.children
            if len(cs) == 1:
                out.append(_convert_expr(cs[0], filename))
            else:
                visit(cs[0])
                out.append(_convert_expr(cs[2], filename))
            return
    visit(node)
    return out


def _convert_struct_member_list(node: ParseNode | Token) -> list[StructMember]:
    """struct_member_list : struct_member | struct_member_list ',' struct_member"""
    out: list[StructMember] = []
    def visit(n: ParseNode | Token) -> None:
        if not isinstance(n, ParseNode):
            return
        if n.kind == "struct_member_list":
            cs = n.children
            if len(cs) == 1:
                out.extend(_convert_struct_member(cs[0]))
            else:
                visit(cs[0])
                out.extend(_convert_struct_member(cs[2]))
    visit(node)
    return out


def _convert_struct_member(node: ParseNode | Token) -> list[StructMember]:
    """struct_member : decl_names array_size_opt type_spec
                     | decl_names array_size_opt"""
    assert isinstance(node, ParseNode) and node.kind == "struct_member"
    cs = node.children
    names = _flatten_decl_names(cs[0])
    size = _convert_array_size_opt(cs[1])
    if len(cs) == 3:
        type_node = cs[2]
        # Use a tiny tail to extract the type
        t = _DeclTail()
        _apply_type_spec(type_node, t)
        dt = t.data_type or DataType.BYTE
    else:
        dt = DataType.BYTE
    return [StructMember(name=n, data_type=dt, dimension=size) for n in names]


def _build_var_decl(
    name: str,
    *,
    based_on: str | None,
    based_member: str | None,
    array_size: int | None,
    tail: _DeclTail,
    span: SourceSpan | None,
) -> VarDecl:
    """Combine the parsed components into a VarDecl. ``tail.type_dimension``
    (e.g. ``BYTE(64)``) becomes the dimension when no leading
    ``array_size_opt`` was given."""
    dim = array_size
    if dim is None and tail.type_dimension is not None:
        dim = tail.type_dimension
    # The implicit-size sentinel (`(*)`) is stored as None on VarDecl —
    # legacy convention, codegen uses ``if d.dimension`` checks and a
    # truthy ``-1`` would corrupt array length / LENGTH() / storage
    # calculations. The actual length is recovered later from the
    # ``data_values`` initialiser when one is present.
    if dim == -1:
        dim = None
    return VarDecl(
        name=name,
        data_type=tail.data_type,
        dimension=dim,
        struct_members=tail.struct_members,
        based_on=based_on,
        based_member=based_member,
        at_location=tail.at_location,
        is_public=tail.is_public,
        is_external=tail.is_external,
        initial_values=tail.initial_values,
        data_values=tail.data_values,
        span=span,
    )


def _type_from_kw(kind: str) -> DataType | None:
    if kind == "KW_BYTE":
        return DataType.BYTE
    if kind == "KW_ADDRESS":
        return DataType.ADDRESS
    if kind == "KW_LABEL":
        return DataType.LABEL
    return None


# --- statements ------------------------------------------------------------


def _convert_stmt(node: ParseNode, filename: str) -> Stmt:
    """stmt : matched_stmt | unmatched_stmt"""
    assert node.kind == "stmt"
    return _convert_matched_or_unmatched(node.children[0], filename)


def _convert_matched_or_unmatched(node: ParseNode | Token, filename: str) -> Stmt:
    assert isinstance(node, ParseNode)
    cs = node.children
    if node.kind == "stmt":
        # The unmatched-IF tail (`IF e THEN <stmt>`) reintroduces a `stmt`
        # node here; transparently descend.
        return _convert_stmt(node, filename)
    if node.kind in ("matched_stmt", "unmatched_stmt"):
        if len(cs) == 1:
            return _convert_matched_or_unmatched_inner(cs[0], filename)
        # IDENT ':' (matched|unmatched)_stmt
        label = _ident_text(cs[0])
        inner = _convert_matched_or_unmatched(cs[2], filename)
        return LabeledStmt(label=label, stmt=inner, span=_span_of(node, filename))
    return _convert_matched_or_unmatched_inner(node, filename)


def _convert_matched_or_unmatched_inner(node: ParseNode | Token, filename: str) -> Stmt:
    assert isinstance(node, ParseNode)
    if node.kind == "matched_if":
        # IF expr THEN matched_stmt ELSE matched_stmt
        cond = _convert_expr(node.children[1], filename)
        then_s = _convert_matched_or_unmatched(node.children[3], filename)
        else_s = _convert_matched_or_unmatched(node.children[5], filename)
        return IfStmt(condition=cond, then_stmt=then_s, else_stmt=else_s, span=_span_of(node, filename))
    if node.kind == "unmatched_if":
        cs = node.children
        cond = _convert_expr(cs[1], filename)
        then_s = _convert_matched_or_unmatched(cs[3], filename)
        else_s = None
        if len(cs) > 4:
            else_s = _convert_matched_or_unmatched(cs[5], filename)
        return IfStmt(condition=cond, then_stmt=then_s, else_stmt=else_s, span=_span_of(node, filename))
    if node.kind == "non_if_stmt":
        return _convert_non_if(node.children[0], filename)
    raise ParserError(f"Unexpected stmt kind: {node.kind}")


def _convert_non_if(node: ParseNode | Token, filename: str) -> Stmt:
    assert isinstance(node, ParseNode)
    span = _span_of(node, filename)
    if node.kind == "assignment_stmt":
        # primary_list '=' expr ';'
        targets = _convert_primary_list(node.children[0], filename)
        value = _convert_expr(node.children[2], filename)
        return AssignStmt(targets=targets, value=value, span=span)
    if node.kind == "call_stmt":
        # CALL primary ';'
        prim = _convert_primary(node.children[1], filename)
        if isinstance(prim, CallExpr):
            return CallStmt(callee=prim.callee, args=prim.args, span=span)
        # Bare CALL FOO; — treat as call with no args.
        return CallStmt(callee=prim, args=[], span=span)
    if node.kind == "return_stmt":
        cs = node.children
        if len(cs) == 2:
            return ReturnStmt(value=None, span=span)
        return ReturnStmt(value=_convert_expr(cs[1], filename), span=span)
    if node.kind == "goto_stmt":
        # GOTO IDENT ';' | GO TO IDENT ';'
        cs = node.children
        ident = cs[1] if _is_token(cs[1], "IDENT") else cs[2]
        return GotoStmt(target=_ident_text(ident), span=span)
    if node.kind == "halt_stmt":
        return HaltStmt(span=span)
    if node.kind == "enable_stmt":
        return EnableStmt(span=span)
    if node.kind == "disable_stmt":
        return DisableStmt(span=span)
    if node.kind == "do_block":
        return _convert_do_block(node, filename)
    if node.kind == "null_stmt":
        return NullStmt(span=span)
    raise ParserError(f"Unexpected non_if_stmt kind: {node.kind}")


def _convert_primary_list(node: ParseNode | Token, filename: str) -> list[Expr]:
    """primary_list : primary | primary_list ',' primary"""
    out: list[Expr] = []
    def visit(n: ParseNode | Token) -> None:
        if not isinstance(n, ParseNode):
            return
        if n.kind == "primary_list":
            cs = n.children
            if len(cs) == 1:
                out.append(_convert_primary(cs[0], filename))
            else:
                visit(cs[0])
                out.append(_convert_primary(cs[2], filename))
    visit(node)
    return out


def _convert_do_block(node: ParseNode, filename: str) -> Stmt:
    cs = node.children
    span = _span_of(node, filename)
    # Find the END label (third-from-last is opt_end_label).
    # Patterns:
    #   DO ; body_items END opt_end_label ; (5)
    #   DO WHILE expr ; body_items END opt_end_label ; (7)
    #   DO IDENT = expr TO expr ; body_items END opt_end_label ; (10)
    #   DO IDENT = expr TO expr BY expr ; body_items END opt_end_label ; (12)
    #   DO CASE expr ; body_items END opt_end_label ; (7)
    end_label = _convert_opt_end_label(cs[-2])

    if _is_token(cs[1], "SEMI"):
        # DO ; body_items END opt_end_label ;
        body_items = _flatten_body_items(cs[2])
        decls, stmts = _split_body(body_items, filename)
        return DoBlock(decls=decls, stmts=stmts, end_label=end_label, span=span)
    if _is_token(cs[1], "KW_WHILE"):
        cond = _convert_expr(cs[2], filename)
        body_items = _flatten_body_items(cs[4])
        _, stmts = _split_body(body_items, filename)
        return DoWhileBlock(condition=cond, stmts=stmts, end_label=end_label, span=span)
    if _is_token(cs[1], "KW_CASE"):
        sel = _convert_expr(cs[2], filename)
        body_items = _flatten_body_items(cs[4])
        _, stmts = _split_body(body_items, filename)
        # DoCaseBlock takes a list of case bodies. Each top-level stmt
        # in the body corresponds to one case (the legacy parser
        # accumulated them this way too).
        cases = [[s] for s in stmts]
        return DoCaseBlock(selector=sel, cases=cases, end_label=end_label, span=span)
    if _is_token(cs[1], "IDENT"):
        # DO IDENT = expr TO expr [BY expr] ; body_items END label ;
        idx = Identifier(name=_ident_text(cs[1]))
        start = _convert_expr(cs[3], filename)
        bound = _convert_expr(cs[5], filename)
        if _is_token(cs[6], "KW_BY"):
            step: Expr | None = _convert_expr(cs[7], filename)
            body_items = _flatten_body_items(cs[9])
        else:
            step = None
            body_items = _flatten_body_items(cs[7])
        _, stmts = _split_body(body_items, filename)
        return DoIterBlock(
            index_var=idx, start=start, bound=bound, step=step,
            stmts=stmts, end_label=end_label, span=span,
        )
    raise ParserError(f"Unexpected do_block shape, head={_kind(cs[1])}")


def _convert_opt_end_label(node: ParseNode | Token) -> str | None:
    if isinstance(node, ParseNode) and node.kind == "opt_end_label":
        if not node.children:
            return None
        return _ident_text(node.children[0])
    return None


def _split_body(
    items: list[ParseNode | Token], filename: str
) -> tuple[list[Declaration], list[Stmt]]:
    decls: list[Declaration] = []
    stmts: list[Stmt] = []
    for bi in items:
        _emit_body_item(bi, decls, stmts, filename)
    return decls, stmts


# --- expressions -----------------------------------------------------------


_REL_OP_BY_TOK = {
    "EQ": BinaryOp.EQ,
    "LT": BinaryOp.LT,
    "GT": BinaryOp.GT,
    "LE": BinaryOp.LE,
    "GE": BinaryOp.GE,
    "NE": BinaryOp.NE,
}


def _convert_expr(node: ParseNode | Token, filename: str) -> Expr:
    assert isinstance(node, ParseNode)
    if node.kind == "expr":
        cs = node.children
        if len(cs) == 1:
            return _convert_or_xor(cs[0], filename)
        # primary ':=' expr
        target = _convert_primary(cs[0], filename)
        value = _convert_expr(cs[2], filename)
        return EmbeddedAssignExpr(target=target, value=value, span=_span_of(node, filename))
    raise ParserError(f"Unexpected expr kind: {node.kind}")


def _convert_or_xor(node: ParseNode | Token, filename: str) -> Expr:
    assert isinstance(node, ParseNode) and node.kind == "or_xor"
    cs = node.children
    if len(cs) == 1:
        return _convert_and(cs[0], filename)
    op_tok = cs[1]
    op = BinaryOp.OR if _kind(op_tok) == "KW_OR" else BinaryOp.XOR
    left = _convert_or_xor(cs[0], filename)
    right = _convert_and(cs[2], filename)
    return BinaryExpr(op=op, left=left, right=right, span=_span_of(node, filename))


def _convert_and(node: ParseNode | Token, filename: str) -> Expr:
    assert isinstance(node, ParseNode) and node.kind == "and_e"
    cs = node.children
    if len(cs) == 1:
        return _convert_not(cs[0], filename)
    left = _convert_and(cs[0], filename)
    right = _convert_not(cs[2], filename)
    return BinaryExpr(op=BinaryOp.AND, left=left, right=right, span=_span_of(node, filename))


def _convert_not(node: ParseNode | Token, filename: str) -> Expr:
    assert isinstance(node, ParseNode) and node.kind == "not_e"
    cs = node.children
    if len(cs) == 1:
        return _convert_rel(cs[0], filename)
    operand = _convert_not(cs[1], filename)
    return UnaryExpr(op=UnaryOp.NOT, operand=operand, span=_span_of(node, filename))


def _convert_rel(node: ParseNode | Token, filename: str) -> Expr:
    assert isinstance(node, ParseNode) and node.kind == "rel_e"
    cs = node.children
    if len(cs) == 1:
        return _convert_add(cs[0], filename)
    left = _convert_add(cs[0], filename)
    op_kind = _kind(cs[1])
    op = _REL_OP_BY_TOK.get(op_kind)
    if op is None:
        raise ParserError(f"Unexpected relational op token: {op_kind}")
    right = _convert_add(cs[2], filename)
    return BinaryExpr(op=op, left=left, right=right, span=_span_of(node, filename))


def _convert_add(node: ParseNode | Token, filename: str) -> Expr:
    assert isinstance(node, ParseNode) and node.kind == "add_e"
    cs = node.children
    if len(cs) == 1:
        return _convert_mul(cs[0], filename)
    left = _convert_add(cs[0], filename)
    op_kind = _kind(cs[1])
    op_map = {
        "OP_PLUS": BinaryOp.ADD,
        "OP_MINUS": BinaryOp.SUB,
        "KW_PLUS": BinaryOp.PLUS,
        "KW_MINUS": BinaryOp.MINUS,
    }
    op = op_map[op_kind]
    right = _convert_mul(cs[2], filename)
    return BinaryExpr(op=op, left=left, right=right, span=_span_of(node, filename))


def _convert_mul(node: ParseNode | Token, filename: str) -> Expr:
    assert isinstance(node, ParseNode) and node.kind == "mul_e"
    cs = node.children
    if len(cs) == 1:
        return _convert_unary(cs[0], filename)
    left = _convert_mul(cs[0], filename)
    op_kind = _kind(cs[1])
    op_map = {"STAR": BinaryOp.MUL, "SLASH": BinaryOp.DIV, "KW_MOD": BinaryOp.MOD}
    op = op_map[op_kind]
    right = _convert_unary(cs[2], filename)
    return BinaryExpr(op=op, left=left, right=right, span=_span_of(node, filename))


def _convert_unary(node: ParseNode | Token, filename: str) -> Expr:
    assert isinstance(node, ParseNode) and node.kind == "unary"
    cs = node.children
    if len(cs) == 1:
        return _convert_primary(cs[0], filename)
    operand = _convert_unary(cs[1], filename)
    return UnaryExpr(op=UnaryOp.NEG, operand=operand, span=_span_of(node, filename))


def _convert_primary(node: ParseNode | Token, filename: str) -> Expr:
    assert isinstance(node, ParseNode) and node.kind == "primary"
    cs = node.children
    span = _span_of(node, filename)

    # NUMBER
    if len(cs) == 1 and _is_token(cs[0], "NUMBER"):
        return NumberLiteral(value=_parse_plm_number(_tok_text(cs[0])), span=span)
    # STRING
    if len(cs) == 1 and _is_token(cs[0], "STRING"):
        body = _strip_string(_tok_text(cs[0]))
        return StringLiteral(value=body, bytes_value=[ord(c) for c in body], span=span)
    # qualname
    if len(cs) == 1 and isinstance(cs[0], ParseNode) and cs[0].kind == "qualname":
        return _convert_qualname(cs[0], filename)
    # '(' expr ')'
    if len(cs) == 3 and _is_token(cs[0], "LPAREN"):
        return _convert_expr(cs[1], filename)
    # '.' qualname
    if len(cs) == 2 and _is_token(cs[0], "DOT"):
        if isinstance(cs[1], ParseNode) and cs[1].kind == "qualname":
            return LocationExpr(operand=_convert_qualname(cs[1], filename), span=span)
        if _is_token(cs[1], "STRING"):
            body = _strip_string(_tok_text(cs[1]))
            inner = StringLiteral(value=body, bytes_value=[ord(c) for c in body])
            return LocationExpr(operand=inner, span=span)
    # '.' '(' arg_list ')' - constant list
    if len(cs) == 4 and _is_token(cs[0], "DOT") and _is_token(cs[1], "LPAREN"):
        values = _convert_arg_list(cs[2], filename)
        return ConstListExpr(values=values, span=span)
    raise ParserError(f"Unexpected primary shape: {[_kind(c) for c in cs]}")


def _convert_qualname(node: ParseNode, filename: str) -> Expr:
    """qualname : IDENT
               | qualname '.' IDENT
               | qualname '(' arg_list ')'
               | qualname '(' ')'

    Lower into nested Identifier / MemberExpr / SubscriptExpr (or CallExpr
    when the parens hold what looks like a call). PL/M doesn't
    syntactically distinguish array indexing from procedure call, so we
    use SubscriptExpr for single-arg parens and CallExpr otherwise; the
    semantic pass decides the actual meaning. Matches what the legacy
    parser produced.
    """
    cs = node.children
    if len(cs) == 1 and _is_token(cs[0], "IDENT"):
        return Identifier(name=_ident_text(cs[0]), span=_span_of(node, filename))
    base = _convert_qualname(cast(ParseNode, cs[0]), filename)
    if _is_token(cs[1], "DOT"):
        return MemberExpr(base=base, member=_ident_text(cs[2]), span=_span_of(node, filename))
    if _is_token(cs[1], "LPAREN"):
        # PL/M doesn't syntactically distinguish array indexing from a
        # procedure call. The legacy parser always emitted CallExpr; the
        # codegen pass later converts ``CallExpr`` whose callee is a
        # known array into ``SubscriptExpr``. Mirror that contract so
        # downstream stays unchanged.
        if len(cs) == 3:  # qualname '(' ')'
            return CallExpr(callee=base, args=[], span=_span_of(node, filename))
        args = _convert_arg_list(cs[2], filename)
        return CallExpr(callee=base, args=args, span=_span_of(node, filename))
    raise ParserError(f"Unexpected qualname shape: {[_kind(c) for c in cs]}")


# --- helpers ---------------------------------------------------------------


def _strip_string(s: str) -> str:
    """Strip the surrounding ``'…'`` from a PL/M STRING and unescape
    doubled quotes."""
    if len(s) >= 2 and s.startswith("'") and s.endswith("'"):
        s = s[1:-1]
    return s.replace("''", "'")


def _parse_plm_number(s: str) -> int:
    """Parse a PL/M-style numeric literal (handles ``$`` separators and
    B/H/O/Q/D suffixes)."""
    s = s.upper().replace("$", "")
    if s.endswith("H"):
        return int(s[:-1], 16)
    if s.endswith("B"):
        return int(s[:-1], 2)
    if s.endswith("O") or s.endswith("Q"):
        return int(s[:-1], 8)
    if s.endswith("D"):
        return int(s[:-1], 10)
    return int(s, 10)
