"""Where a SHL of a BYTE loses bits that uplm80 before 0.4.3 kept.

SHL and SHR have the type of their pattern (Programming Manual 9800268B,
11.1.4): ``SHL(b, 4)`` of a BYTE is a BYTE, shifted in eight bits, and the
bits shifted out of it are lost, as Intel's PL/M-80 V3.1 compiles it.
uplm80 before 0.4.3 zero-extended a BYTE pattern and shifted it in sixteen
bits, with an ADDRESS result, and a program written for it may rely on the
bits it kept - 80un built words with ``lo + shl(b, 8)``.

The two give the same low byte; they differ above it, and only where the
shift can move a set bit past bit 7.  :func:`check_byte_shifts` follows
each such SHL out through what it is part of - the operators whose low byte
depends only on their operands' low bytes (``+ - * OR XOR NOT``, and SHL),
but for an AND with a BYTE, which clears what is above - to where the bits
above the low byte are read: a store to an ADDRESS, an ADDRESS argument or
RETURN value, a subscript, a relation, ``/`` and ``MOD``, SHR, SCL, SCR,
HIGH and DOUBLE of it, a DO CASE's selector, the limits of an ADDRESS
index.  Each SHL that reaches one is warned of, once.

The flags differ too: an operation on a SHL of a BYTE that can lose bits
is one of eight bits where it was one of sixteen, whose carry comes out of
bit 7, and which sets ZERO, SIGN and PARITY where a 16-bit addition left
them as they were.  PLUS, MINUS, SCL, SCR and DEC read the carry, and
CARRY, ZERO, SIGN and PARITY the flags, of the operation before them: each
SHL the flags may differ by there is warned of.  They are followed through
the statements after the operation, to a RETURN and from the call, and
from a GOTO to every label; a call of what is not known, an EXTERNAL
procedure or through an address, leaves flags of its own.

A SHL whose pattern's largest value, shifted by its constant count, fits in
eight bits loses nothing: ``shl(dcnt and 11b, 5)``, ``shl(3, 4)``.  The
largest value of a BYTE or ADDRESS variable is what the statements before
the SHL have left it: what was last assigned to it, and a test it has
passed - after ``if n > 32 then return;``, n is at most 32, after ``if (n
and 0e0h) <> 0 then return;`` 31 - where nothing can have changed it in
between: not the test itself, by an embedded assignment or in a procedure
it calls.  That is followed only for a scalar whose address is never
taken, nor AT, BASED, PUBLIC or EXTERNAL, nor assigned by an INTERRUPT
procedure or what it calls, and a call of a procedure forgets it, but for
a local of the procedure the call is in where the procedure called is not
nested in it.  A procedure that ends in a call of MON1 with the function
0, BDOS's system reset, or of such a procedure, and has no RETURN nor a
label on its END, does not return.  A SHR of a BYTE has the same value
either way.

What is not seen: a store that reaches the variable other than by its
name - past the end of an array, through MEMORY or a BASED variable whose
base is not its address; the flags a procedure is entered with, and those
an EXTERNAL procedure, or one called through an address, returns with;
an EXTERNAL MON1 that is not BDOS's entry, whose function 0 returns - MON1
is the name DRI's programs give BDOS's entry, in every mode; and
arithmetic around a SHL that loses nothing, which can overflow eight
bits, with its value, as ``shr(z, 4) - 1``, or its flags, where it was of
sixteen - ZERO, SIGN and PARITY after such a SHL, too.

It runs on the program as the parser gave it, before the optimizer rewrites
anything, so every ``-O`` level warns of the same places.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

from dataclasses import dataclass, field

from . import _plm_parser as P
from .ast_view import (
    BinaryOpKind,
    DataType,
    UnaryOpKind,
    binop_kind,
    decl_attrs,
    decl_item_names,
    decl_item_struct_members,
    decl_item_type,
    expr_text,
    ident_text,
    is_end_of_block,
    iter_block_proc_decls,
    number_value,
    proc_attrs,
    proc_name,
    proc_param_names,
    proc_return_type,
    string_value,
    struct_member_names,
    struct_member_type,
    unop_kind,
    unwrap_paren,
)
from .frontend import source_location
from .plm_types import RELATIONS, binary_type, literal_type, mask

BYTE = DataType.BYTE
ADDRESS = DataType.ADDRESS

# Built-ins that read the bits of their first argument above its low byte.
_READS_HIGH_BITS = frozenset({"HIGH", "DOUBLE", "SHR", "SCL", "SCR"})
_BYTE_RESULT = frozenset({"LOW", "HIGH", "ROL", "ROR", "INPUT", "DEC", "CARRY", "ZERO",
                          "SIGN", "PARITY", "MEMORY"})
_BUILTINS = _BYTE_RESULT | frozenset({"DOUBLE", "SHL", "SHR", "SCL", "SCR", "LENGTH", "LAST",
                                      "SIZE", "MOVE", "TIME", "STACKPTR", "OUTPUT", "CPUTIME"})
# What a test of `x op k' says of x where it holds: its largest value, from k.
_BOUND_IF_TRUE = {BinaryOpKind.LT: -1, BinaryOpKind.LE: 0, BinaryOpKind.EQ: 0}
_BOUND_IF_FALSE = {BinaryOpKind.GT: 0, BinaryOpKind.GE: -1, BinaryOpKind.NE: 0}
_MIRROR = {BinaryOpKind.LT: BinaryOpKind.GT, BinaryOpKind.GT: BinaryOpKind.LT,
           BinaryOpKind.LE: BinaryOpKind.GE, BinaryOpKind.GE: BinaryOpKind.LE,
           BinaryOpKind.EQ: BinaryOpKind.EQ, BinaryOpKind.NE: BinaryOpKind.NE}


@dataclass(eq=False)
class _Sym:  # pylint: disable=too-many-instance-attributes
    """What a declared name is, as far as types go."""
    kind: str                           # "var", "proc" or "other"
    dtype: DataType | None = None       # scalar, element or return type
    dim: int | None = None              # an array's
    members: dict = field(default_factory=dict)     # name -> type
    params: list = field(default_factory=list)      # a procedure's parameter types
    plain: bool = False                 # a scalar whose value can be followed
    owner: _Sym | None = None           # the procedure it is declared in
    key: tuple = ()                     # the same from one walk to the next
    init: int = 0                       # its largest INITIAL value
    param: bool = False
    name: str = ""
    node: P.ProcDecl | None = None      # a procedure's declaration


@dataclass
class _Val:
    """An expression: its type, the largest value it can have, the SHLs of
    a BYTE whose lost bits make it differ, above its low byte, from what
    uplm80 before 0.4.3 computed, and the SHLs of a BYTE that can lose bits
    among its operands, through any operator (``narrow``), which make the
    flags an operation on it sets differ too."""
    dtype: DataType | None = None
    top: int = 0xFFFF
    shifts: tuple = ()
    narrow: tuple = ()


def _item_syms(item, pinned: set[str]) -> dict[str, _Sym]:
    """The names a DeclItem or DeclItemBasedGroup declares; ``pinned``, the
    names whose address the module takes."""
    dt, dim = decl_item_type(item)
    if isinstance(getattr(item, "tail", None), P.DeclTailData):
        dt = BYTE       # an untyped DATA is BYTE
    members = {}
    for m in decl_item_struct_members(item) or []:
        members.update(dict.fromkeys(struct_member_names(m), struct_member_type(m)))
    if dt not in (BYTE, ADDRESS):
        dt = None
    attrs = decl_attrs(item)
    if isinstance(item, P.DeclItemBasedGroup):
        names = [ident_text(bd.name) for bd in item.based_decls or []]
    else:
        names = decl_item_names(item)
    plain = (dt is not None and dim is None and getattr(item, "based", None) is None
             and not isinstance(item, P.DeclItemBasedGroup) and attrs.at_location is None
             and not attrs.is_public and not attrs.is_external and attrs.data_values is None)
    init = max((_const(v) if _const(v) is not None else 0xFFFF
                for v in attrs.initial_values or []), default=0)
    return {n: _Sym("var", dt, dim, members, plain=plain and n not in pinned,
                    key=(id(item), n), init=init, name=n) for n in names}


def _param_types(proc: P.ProcDecl) -> list:
    """The types of a procedure's parameters, in order, as its DECLAREs give them."""
    types: dict[str, DataType | None] = {}
    for it in proc.body.items:
        if isinstance(it, P.DeclareStmt):
            for d in it.declarations:
                if isinstance(d, P.DeclItem):
                    types.update({n: s.dtype for n, s in _item_syms(d, set()).items()})
    return [types.get(n) for n in proc_param_names(proc)]


def _proc_sym(proc: P.ProcDecl) -> _Sym:
    return _Sym("proc", proc_return_type(proc), params=_param_types(proc),
                name=proc_name(proc), node=proc)


def _scope_of(items, pinned: set[str]) -> dict[str, _Sym]:
    """The names a block's declarations introduce."""
    scope: dict[str, _Sym] = {}
    for it in items:
        for d in it.declarations if isinstance(it, P.DeclareStmt) else [it]:
            if isinstance(d, P.ProcDecl):
                scope[proc_name(d)] = _proc_sym(d)
            elif isinstance(d, (P.DeclItem, P.DeclItemBasedGroup)):
                scope.update(_item_syms(d, pinned))
            elif isinstance(d, P.LiterallyDecl):
                scope[ident_text(d.name)] = _Sym("other")
    for proc in iter_block_proc_decls(items):
        scope.setdefault(proc_name(proc), _proc_sym(proc))
    return scope


def _pinned(tree) -> set[str]:
    """The names whose address ``tree`` takes, `.x' or `.x(i).m', or
    that an AT names: a store through a pointer may change them."""
    out: set[str] = set()
    for n in _nodes(tree):
        if isinstance(n, (P.LocationOf, P.AttrAt)):
            root = unwrap_paren(n.operand if isinstance(n, P.LocationOf) else n.address)
            while isinstance(root, (P.Call, P.MemberAccess, P.LocationOf)):
                root = unwrap_paren(root.callee if isinstance(root, P.Call) else
                                    root.base if isinstance(root, P.MemberAccess)
                                    else root.operand)
            if isinstance(root, (P.Identifier, P.DottedIdent)):
                out.add(ident_text(root.name))
    return out


def _assigned(tree, into_procs: bool = True) -> set[str]:
    """The names ``tree`` assigns to, and the indexes of its loops."""
    out: set[str] = set()
    for n in _nodes(tree, into_procs):
        targets = (n.targets if isinstance(n, P.AssignStmt) else
                   [n.target] if isinstance(n, P.EmbeddedAssign) else [])
        for t in targets:
            t = unwrap_paren(t)
            if isinstance(t, P.Identifier):
                out.add(ident_text(t.name))
        if isinstance(n, (P.DoIterBlock, P.DoIterByBlock)):
            out.add(ident_text(n.index))
    return out


def _join(a: dict | None, b: dict | None) -> dict | None:
    """What is known after either of two ways in; None: no way in."""
    if a is None or b is None:
        return b if a is None else a
    return {k: max(v, b[k]) for k, v in a.items() if k in b}


def _const(expr) -> int | None:
    e = unwrap_paren(expr)
    if isinstance(e, P.NumberLiteral):
        return number_value(e) & 0xFFFF
    return None


def _union(*parts: tuple) -> tuple:
    """The SHLs of ``parts``, each once, in order."""
    out: dict[int, object] = {}
    for part in parts:
        for call in part:
            out.setdefault(id(call), call)
    return tuple(out.values())


_FLAG_READERS = frozenset({"CARRY", "ZERO", "SIGN", "PARITY"})


class _Checker:  # pylint: disable=too-many-public-methods,too-many-instance-attributes
    """One walk over the modules of a compilation."""

    def __init__(self, shared: dict[str, _Sym], pinned: set[str]) -> None:
        self.scopes: list[dict[str, _Sym]] = [shared]
        self.procs: list[_Sym | None] = [None]
        self.returns: list[DataType | None] = []
        self.warned: dict[int, tuple] = {}
        self.pinned = pinned
        # The largest value of each plain variable the statement being
        # checked can be reached with; None where it cannot be reached.
        self.facts: dict[_Sym, int] | None = {}
        # The largest value each plain variable but a parameter ever has,
        # by its key, as the last walk found (bounds) and this one (seen).
        self.bounds: dict[tuple, int] = {}
        self.seen: dict[tuple, int] = {}
        # The names each procedure, by its declaration, may assign, itself
        # or through what it calls (_modsets).
        self.modsets: dict[int, frozenset] = {}
        # The SHLs of a BYTE the flags as they stand may differ by: those
        # the last operation had among its operands, and, after an
        # operation of sixteen bits, which sets the carry alone, those of
        # the operations before it.  What the flags may differ by where
        # each procedure returns, by its declaration, and at a GOTO, as
        # this walk and the last found.
        self.flags: tuple = ()
        self.ret_flags: dict[int, tuple] = {}
        self.jump_flags: tuple = ()
        # The procedures that do not return, by their declarations
        # (_no_return).
        self.no_return: set[int] = set()

    # ---- names ---------------------------------------------------------

    def lookup(self, name: str) -> _Sym | None:
        """The declaration ``name`` means here, if the program declares it."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def push(self, items) -> None:
        """Enter a block that declares ``items``."""
        scope = _scope_of(items, self.pinned)
        for sym in scope.values():
            sym.owner = self.procs[-1]
        self.scopes.append(scope)

    def block(self, items) -> None:
        """Check a block's statements, its declarations in scope."""
        self.push(items)
        for it in items:
            self.stmt(it)
        self.scopes.pop()

    def called(self, callee: _Sym | None) -> None:
        """A call of ``callee``, or of what is not known (None): forget what
        it may assign."""
        if self.facts is None:
            return
        names = self.modsets.get(id(callee.node)) if callee is not None else None
        if names is None:
            self.facts = {}
        else:
            self.facts = {k: v for k, v in self.facts.items() if k.name not in names}

    def top(self, sym: _Sym) -> int:
        """The largest value the plain variable ``sym`` can have here."""
        top = mask(sym.dtype)
        if not sym.param:
            top = min(top, self.bounds.get(sym.key, sym.init))
        return min(top, self.facts.get(sym, top)) if self.facts is not None else top

    def returned(self, proc: _Sym) -> int:
        """The largest value the typed procedure ``proc`` returns: what its
        RETURNs give, but for one declared EXTERNAL."""
        if proc.node is None or proc_attrs(proc.node).is_external:
            return mask(proc.dtype)
        return min(mask(proc.dtype), self.bounds.get(("return", id(proc.node)), 0))

    def assign(self, target, v: _Val, after: int | None = None) -> None:
        """What a store of ``v`` to ``target`` says of the target; ``after``,
        the largest value it is left with, if that is larger."""
        t = unwrap_paren(target)
        sym = self.lookup(ident_text(t.name)) if isinstance(t, P.Identifier) else None
        if sym is None or not sym.plain:
            return
        top = min(v.top, mask(sym.dtype)) if v.dtype is not None else mask(sym.dtype)
        if self.facts is not None:
            self.facts[sym] = top
        if not sym.param:
            top = max(top, min(after or 0, mask(sym.dtype)))
            self.seen[sym.key] = max(self.seen.get(sym.key, sym.init), top)

    def refine(self, cond, holds: bool) -> None:
        """What ``cond`` holding (or not, ``holds``) says of a variable: of
        one the condition itself does not assign, by an embedded assignment
        or in a procedure it calls, after or before the test of it."""
        if self.facts is None:
            return
        changed = set(_assigned(cond))
        for name, _ in _callees(cond):
            sym = self.lookup(name)
            if sym is not None and sym.kind == "proc":
                names = self.modsets.get(id(sym.node))
                if names is None:
                    return          # not known: it may assign anything
                changed |= names
            elif sym is None and name.upper() not in _BUILTINS:
                return
        self._refine(cond, holds, changed)

    def _refine(self, cond, holds: bool, changed: set[str]) -> None:
        c = unwrap_paren(cond)
        if isinstance(c, P.UnaryOp) and unop_kind(c) == UnaryOpKind.NOT:
            self._refine(c.operand, not holds, changed)
            return
        if not isinstance(c, P.BinaryOp):
            return
        kind = binop_kind(c)
        if (kind == BinaryOpKind.OR and not holds) or (kind == BinaryOpKind.AND and holds):
            self._refine(c.left, holds, changed)
            self._refine(c.right, holds, changed)
            return
        if kind not in RELATIONS:
            return
        var, k = unwrap_paren(c.left), _const(c.right)
        if k is None:
            var, k, kind = unwrap_paren(c.right), _const(c.left), _MIRROR[kind]
        if isinstance(var, P.BinaryOp) and binop_kind(var) == BinaryOpKind.AND:
            if k == 0 and (kind == BinaryOpKind.EQ) == holds and kind in (
                    BinaryOpKind.EQ, BinaryOpKind.NE):
                self._refine_mask(var, changed)
            return
        name = ident_text(var.name) if isinstance(var, P.Identifier) else None
        sym = self.lookup(name) if name is not None and name not in changed else None
        delta = (_BOUND_IF_TRUE if holds else _BOUND_IF_FALSE).get(kind)
        if k is None or sym is None or not sym.plain or delta is None or k + delta < 0:
            return
        self.facts[sym] = min(self.top(sym), k + delta)

    def _refine_mask(self, e: P.BinaryOp, changed: set[str]) -> None:
        """`x AND m' is 0: x has no bit of m set."""
        var, m = unwrap_paren(e.left), _const(e.right)
        if m is None:
            var, m = unwrap_paren(e.right), _const(e.left)
        name = ident_text(var.name) if isinstance(var, P.Identifier) else None
        sym = self.lookup(name) if name is not None and name not in changed else None
        if m is None or sym is None or not sym.plain:
            return
        top = mask(sym.dtype) & ~m
        self.facts[sym] = min(self.top(sym), top)

    def refine_case(self, s: P.DoCaseBlock) -> None:
        """A DO CASE on a variable: the variable is less than the number of
        its cases."""
        var = unwrap_paren(s.selector)
        sym = self.lookup(ident_text(var.name)) if isinstance(var, P.Identifier) else None
        cases = sum(1 for it in s.items if not is_end_of_block(it))
        if sym is not None and sym.plain and cases and self.facts is not None:
            self.facts[sym] = min(self.top(sym), cases - 1)

    # ---- where the bits above the low byte are read ----------------------

    def read(self, v: _Val) -> None:
        """``v``'s bits above its low byte are read: warn of the SHLs that
        make them differ."""
        for call in v.shifts:
            if id(call) in self.warned:
                continue
            x, n = (expr_text(a) for a in call.args)
            self.warned[id(call)] = (source_location(call), (
                f"SHL({x}, {n}): SHL of a BYTE is a BYTE (Programming Manual 9800268B, "
                "11.1.4), and the bits shifted out of it, which this expression uses, are "
                f"lost; SHL(DOUBLE({x}), {n}) keeps them. uplm80 before 0.4.3 shifted a "
                "BYTE in 16 bits"))

    def read_flags(self, reader: str, flags: tuple | None = None) -> None:
        """``reader`` - PLUS, MINUS, SCL, SCR, DEC, CARRY, ZERO, SIGN or
        PARITY - reads the flags: warn of the SHLs they may differ by."""
        for call in self.flags if flags is None else flags:
            if id(call) in self.warned:
                continue
            x, n = (expr_text(a) for a in call.args)
            self.warned[id(call)] = (source_location(call), (
                f"SHL({x}, {n}): SHL of a BYTE is a BYTE (Programming Manual 9800268B, "
                f"11.1.4), and {reader} reads the flags of an operation of eight bits on it; "
                f"SHL(DOUBLE({x}), {n}) is shifted in 16 bits, as uplm80 before 0.4.3 "
                "shifted a BYTE"))

    def set_flags(self, narrow: tuple, wide: bool) -> None:
        """An operation sets the flags: of ``narrow`` operands, or, of
        sixteen bits (``wide``), the carry alone."""
        self.flags = narrow if narrow else self.flags if wide else ()

    def returned_flags(self) -> None:
        """The procedure being checked returns with the flags as they are."""
        proc = self.procs[-1]
        if proc is not None and proc.node is not None:
            key = id(proc.node)
            self.ret_flags[key] = _union(self.ret_flags.get(key, ()), self.flags)

    def value(self, expr, wide: bool | None) -> _Val:
        """Evaluate ``expr``, used where it is converted to an ADDRESS
        (``wide``), to a BYTE (False), or neither (None: its own type)."""
        v = self.expr(expr)
        if wide or (wide is None and v.dtype is ADDRESS):
            self.read(v)
        return v

    # ---- statements -----------------------------------------------------

    def stmt(self, s) -> None:  # pylint: disable=too-many-branches
        """Check a statement, and follow what it tells of the variables."""
        if self.facts is None:
            self.facts = {}         # reached only by a GOTO, if at all
        if isinstance(s, P.LabeledStmt):
            self.facts = {}         # a GOTO may come here with anything
            self.flags = _union(self.flags, self.jump_flags)
            self.stmt(s.stmt)
        elif isinstance(s, P.ProcDecl):
            self.proc(s)
        elif isinstance(s, P.DeclareStmt):
            for d in s.declarations:
                if isinstance(d, P.ProcDecl):
                    self.proc(d)
        elif isinstance(s, P.AssignStmt):
            v = self.expr(s.value)
            flags = self.flags
            for t in s.targets:
                if self.target(t) is not BYTE:
                    self.read(v)
                self.assign(t, v)
            self.flags = _union(flags, self.flags)
        elif isinstance(s, P.CallStmt):
            self.call_stmt(s)
        elif isinstance(s, (P.ReturnStmtValue, P.ReturnStmt, P.GotoStmt, P.HaltStmt)):
            self.jump(s)
        elif isinstance(s, (P.IfStmt, P.IfStmtElse)):
            self.if_stmt(s)
        elif isinstance(s, (P.DoWhileBlock, P.DoIterBlock, P.DoIterByBlock)):
            self.loop(s)
        elif isinstance(s, P.DoCaseBlock):
            self.case(s)
        elif isinstance(s, P.DoBlock):
            self.block(s.items)

    def call_stmt(self, s: P.CallStmt) -> None:
        """A CALL: of a procedure, which may not return, or through an
        address."""
        self.expr(s.callee)
        callee = unwrap_paren(s.callee)
        callee = unwrap_paren(callee.callee) if isinstance(callee, P.Call) else callee
        sym = self.lookup(ident_text(callee.name)) if isinstance(callee, P.Identifier) else None
        if sym is not None and sym.kind != "proc":
            self.called(None)           # a call through an address
            self.flags = ()
        elif sym is not None and sym.node is not None and (
                id(sym.node) in self.no_return
                or (_system_reset(s) and proc_attrs(sym.node).is_external)):
            self.facts = None           # it does not return

    def jump(self, s) -> None:
        """A RETURN, a GOTO or a HALT: nothing after it is reached from it."""
        if isinstance(s, P.ReturnStmtValue):
            v = self.value(s.value, self.returns[-1] is not BYTE if self.returns else None)
            proc = self.procs[-1]
            if proc is not None and proc.node is not None and proc.dtype is not None:
                key = ("return", id(proc.node))
                top = min(v.top, mask(proc.dtype)) if v.dtype else mask(proc.dtype)
                self.seen[key] = max(self.seen.get(key, 0), top)
        if isinstance(s, (P.ReturnStmtValue, P.ReturnStmt)):
            self.returned_flags()
        elif isinstance(s, P.GotoStmt):
            self.jump_flags = _union(self.jump_flags, self.flags)
        self.facts = None

    def case(self, s: P.DoCaseBlock) -> None:
        """A DO CASE: after it, what the cases that end leave."""
        self.read(self.expr(s.selector))
        self.push(s.items)
        # The selector picks one of the cases (7.3): it is less than
        # their number.
        self.refine_case(s)
        start, out = self.facts, None
        flags, out_flags = self.flags, ()
        for it in s.items:
            self.facts = dict(start) if start is not None else None
            self.flags = flags
            self.stmt(it)
            out = _join(out, self.facts)
            out_flags = _union(out_flags, self.flags)
        self.scopes.pop()
        self.facts, self.flags = out, out_flags

    def if_stmt(self, s) -> None:
        """An IF: each branch knows what its condition says; after it, what
        both branches that end leave."""
        self.expr(s.condition)          # its bit 0 is its truth
        start = dict(self.facts) if self.facts is not None else None
        flags = self.flags
        self.refine(s.condition, True)
        self.stmt(s.then_stmt)
        then, then_flags = self.facts, self.flags
        self.facts, self.flags = start, flags
        self.refine(s.condition, False)
        if isinstance(s, P.IfStmtElse):
            self.stmt(s.else_stmt)
        self.facts = _join(then, self.facts)
        self.flags = _union(then_flags, self.flags)

    def loop(self, s) -> None:
        """A DO WHILE or an iterative DO: what is known at its head holds
        at each pass, and after it.  The flags at its head are those it is
        entered with and those a pass leaves, and after it those of its
        test."""
        changed = _assigned(s)
        if self.facts is not None:
            self.facts = {k: v for k, v in self.facts.items()
                          if not any(k is self.lookup(n) for n in changed)}
            for name, call in _callees(s):
                sym = self.lookup(name)
                if sym is not None and sym.kind == "proc":
                    self.called(sym)
                elif (sym is None and name.upper() not in _BUILTINS) or (
                        sym is not None and call):
                    self.called(None)       # through an address, or not known
        start = dict(self.facts) if self.facts is not None else None
        flags = self.flags
        while True:
            self.facts = dict(start) if start is not None else None
            self.flags = flags
            head, test_flags = self.loop_pass(s)
            more = _union(flags, self.flags)
            if len(more) == len(flags):
                break
            flags = more
        self.facts, self.flags = head, test_flags
        if isinstance(s, P.DoWhileBlock):
            self.refine(s.condition, False)

    def loop_pass(self, s) -> tuple:
        """One pass of a loop: what is known at its head, and the flags
        after its test."""
        index_top = exit_top = None
        if isinstance(s, P.DoWhileBlock):
            self.expr(s.condition)
            head = dict(self.facts) if self.facts is not None else None
            test_flags = self.flags
            self.refine(s.condition, True)
        else:
            index = self.target(P.Identifier(name=s.index, pos=s.pos))
            vals = [self.value(getattr(s, f), index is not BYTE)
                    for f in ("start", "bound", "step") if getattr(s, f, None) is not None]
            tops = [v.top for v in vals]
            head = dict(self.facts) if self.facts is not None else None
            test_flags = self.flags = _union(self.flags, *(v.narrow for v in vals))
            index_top = max(tops[:2])
            # It ends a step past its limit.
            exit_top = tops[1] + (tops[2] if len(tops) > 2 else 1)
        self.push(s.items)
        if index_top is not None:
            self.assign(P.Identifier(name=s.index, pos=s.pos), _Val(ADDRESS, index_top),
                        exit_top)
        for it in s.items:
            self.stmt(it)
        self.scopes.pop()
        if not isinstance(s, P.DoWhileBlock):
            test_flags = _union(test_flags, self.flags)
        return head, test_flags

    def proc(self, p: P.ProcDecl) -> None:
        """A procedure's body, which knows nothing of its callers."""
        if proc_attrs(p).is_external:
            return
        saved, flags = self.facts, self.flags
        self.facts, self.flags = {}, ()
        self.procs.append(self.lookup(proc_name(p)))
        self.returns.append(proc_return_type(p))
        self.push(p.body.items)
        for n in proc_param_names(p):
            if n in self.scopes[-1]:
                self.scopes[-1][n].param = True
        for it in p.body.items:
            self.stmt(it)
        self.returned_flags()           # at its END
        self.scopes.pop()
        self.returns.pop()
        self.procs.pop()
        self.facts, self.flags = saved, flags

    def target(self, t) -> DataType | None:
        """The type of an assignment's target, its subscripts evaluated."""
        t = unwrap_paren(t)
        callee = unwrap_paren(t.callee) if isinstance(t, P.Call) else None
        if isinstance(callee, P.Identifier) and ident_text(callee.name).upper() == "OUTPUT" \
                and self.lookup(ident_text(callee.name)) is None:
            for a in t.args:
                self.value(a, False)        # the port is a BYTE
            return BYTE
        if isinstance(t, P.Identifier):
            sym = self.lookup(ident_text(t.name))
            return sym.dtype if sym is not None and sym.kind == "var" else None
        return self.expr(t).dtype

    # ---- expressions ----------------------------------------------------

    def expr(self, e) -> _Val:  # pylint: disable=too-many-return-statements
        """Evaluate an expression: its type, largest value and lost bits."""
        e = unwrap_paren(e)
        if isinstance(e, P.NumberLiteral):
            n = number_value(e) & 0xFFFF
            return _Val(literal_type(n), n)
        if isinstance(e, P.StringLiteral):
            text = string_value(e)
            return _Val(BYTE, ord(text) & 0xFF) if len(text) == 1 else _Val(ADDRESS)
        if isinstance(e, (P.Identifier, P.CallNoArgs)):
            callee = unwrap_paren(e.callee) if isinstance(e, P.CallNoArgs) else e
            if isinstance(callee, P.Identifier):
                return self.name(ident_text(callee.name))
            return self.expr(callee)
        if isinstance(e, P.Call):
            return self.call(e)
        if isinstance(e, P.MemberAccess):
            return self.member(e, [])
        if isinstance(e, P.BinaryOp):
            return self.binary(e)
        if isinstance(e, P.UnaryOp):
            v = self.expr(e.operand)
            self.set_flags(v.narrow, v.dtype is not BYTE)
            return _Val(v.dtype, 0 if v.top == 0 else mask(v.dtype or ADDRESS), v.shifts,
                        v.narrow)
        if isinstance(e, P.LocationOf):
            self.location(e.operand)
        elif isinstance(e, P.EmbeddedAssign):
            v = self.expr(e.value)
            if self.target(e.target) is not BYTE:
                self.read(v)
            self.assign(e.target, v)
            return v
        return _Val(ADDRESS) if isinstance(
            e, (P.LocationOf, P.LocationOfList, P.LocationOfString)) else _Val()

    def location(self, operand) -> None:
        """The operand of a dot: only its subscripts are evaluated."""
        operand = unwrap_paren(operand)
        while isinstance(operand, (P.Call, P.MemberAccess)):
            if isinstance(operand, P.Call):
                for a in operand.args:
                    self.value(a, True)
                operand = unwrap_paren(operand.callee)
            else:
                operand = unwrap_paren(operand.base)

    def name(self, name: str) -> _Val:
        """A name used without arguments."""
        sym = self.lookup(name)
        if sym is None:
            upper = name.upper()
            if upper in _FLAG_READERS:
                self.read_flags(upper)
            if upper == "STACKPTR":
                return _Val(ADDRESS)
            return _Val(BYTE, 0xFF) if upper in _BYTE_RESULT else _Val()
        if sym.kind == "proc":
            self.called(sym)
            self.flags = self.ret_flags.get(id(sym.node), ())
        if sym.kind not in ("var", "proc") or sym.dtype is None:
            return _Val()
        if sym.kind == "proc":
            return _Val(sym.dtype, self.returned(sym))
        return _Val(sym.dtype, self.top(sym) if sym.plain else mask(sym.dtype))

    def member(self, e: P.MemberAccess, args: list) -> _Val:
        """``s.m``, ``s(i).m`` or, with ``args``, ``s.m(i)``."""
        for a in args:
            self.value(a, True)
        base = unwrap_paren(e.base)
        if isinstance(base, P.Call):
            for a in base.args:
                self.value(a, True)
            base = unwrap_paren(base.callee)
        sym = self.lookup(ident_text(base.name)) if isinstance(base, P.Identifier) else None
        mt = sym.members.get(ident_text(e.member)) if sym is not None else None
        return _Val(mt, mask(mt)) if mt else _Val()

    def call(self, e: P.Call) -> _Val:
        """A call of a procedure or a built-in, or a subscripted variable."""
        callee = unwrap_paren(e.callee)
        if isinstance(callee, P.MemberAccess):
            return self.member(callee, e.args)
        sym = self.lookup(ident_text(callee.name)) if isinstance(callee, P.Identifier) else None
        name = ident_text(callee.name).upper() if isinstance(callee, P.Identifier) else ""
        if sym is None and name in ("SHL", "SHR") and len(e.args) == 2:
            return self.shift(e, name)
        if sym is None and name in _BUILTINS:
            return self.builtin(name, e.args)
        if sym is not None and sym.kind == "proc":
            for i, a in enumerate(e.args):
                self.value(a, (sym.params[i] if i < len(sym.params) else None) is not BYTE)
            self.called(sym)
            self.flags = self.ret_flags.get(id(sym.node), ())
            return _Val(sym.dtype, self.returned(sym)) if sym.dtype else _Val()
        for a in e.args:                    # a subscript
            self.value(a, True)
        if sym is not None and sym.kind == "var" and sym.dtype is not None:
            return _Val(sym.dtype, mask(sym.dtype))
        return _Val()

    def builtin(self, name: str, args: list) -> _Val:  # pylint: disable=too-many-return-statements
        """A built-in but SHL and SHR, which the program does not declare."""
        if name in ("LENGTH", "LAST", "SIZE"):
            return self.extent(name, args)
        # MOVE's arguments are ADDRESSes and MEMORY's a subscript; the rest
        # take BYTEs, but for the first of those that read its high bits.
        vals = [self.value(a, name in ("MOVE", "MEMORY")
                           or (i == 0 and name in _READS_HIGH_BITS))
                for i, a in enumerate(args)]
        first = vals[0] if vals else _Val()
        if name in ("SCL", "SCR", "DEC"):
            self.read_flags(name)       # the carry, and DEC's half carry
        if name in ("MOVE", "TIME"):
            self.flags = ()
        elif name in ("ROL", "ROR", "SCL", "SCR", "DEC"):
            # A rotation sets the carry alone.
            self.flags = _union(self.flags, first.narrow)
        if name == "LOW":
            return _Val(BYTE, min(first.top, 0xFF))
        if name == "HIGH":
            return _Val(BYTE, first.top >> 8 if first.dtype is ADDRESS else 0)
        if name == "DOUBLE":
            return _Val(ADDRESS, first.top)
        if name in ("SCL", "SCR"):
            return _Val(first.dtype, mask(first.dtype or ADDRESS))
        if name in _BYTE_RESULT:
            return _Val(BYTE, 0xFF)
        return _Val(ADDRESS) if name == "STACKPTR" else _Val()

    def extent(self, name: str, args: list) -> _Val:
        """LENGTH, LAST or SIZE, whose argument is not evaluated."""
        arg = unwrap_paren(args[0]) if args else None
        sym = self.lookup(ident_text(arg.name)) if isinstance(arg, P.Identifier) else None
        if name == "SIZE" or sym is None or not sym.dim or sym.dim < 0:
            return _Val(ADDRESS if name == "SIZE" else None)
        n = sym.dim if name == "LENGTH" else sym.dim - 1
        return _Val(literal_type(n), n)

    def shift(self, e: P.Call, name: str) -> _Val:
        """SHL or SHR: of a BYTE, a BYTE (11.1.4).  A SHR reads the bits
        of its pattern above the low byte, which it shifts into it."""
        pat = self.value(e.args[0], True) if name == "SHR" else self.expr(e.args[0])
        count = self.value(e.args[1], False)    # converted to a BYTE
        c = _const(e.args[1])
        most = c & 0xFF if c is not None else min(count.top, 0xFF) if count.dtype else 0xFF
        if name == "SHR":
            self.set_flags(pat.narrow, pat.dtype is not BYTE)
            return _Val(pat.dtype, pat.top >> c if c is not None else pat.top, (), pat.narrow)
        top = pat.top << most
        if pat.dtype is not BYTE:
            self.set_flags(pat.narrow, True)
            return _Val(pat.dtype, min(top, 0xFFFF), pat.shifts, pat.narrow)
        if top <= 0xFF:
            self.set_flags(pat.narrow, False)
            return _Val(BYTE, top, pat.shifts, pat.narrow)
        # Of eight bits where it was of sixteen, which set the carry alone.
        narrow = _union(pat.narrow, (e,))
        self.flags = _union(self.flags, narrow)
        return _Val(BYTE, 0xFF, pat.shifts + (e,), narrow)

    def binary(self, e: P.BinaryOp) -> _Val:
        """An operator of two operands: whether it uses the bits above their
        low bytes, or passes them on (4.2 to 4.4); and the flags it reads,
        PLUS and MINUS the carry, and sets."""
        kind = binop_kind(e)
        left = self.expr(e.left)
        after_left = self.flags
        right = self.expr(e.right)
        if kind in (BinaryOpKind.PLUS, BinaryOpKind.MINUS):
            self.read_flags(kind.name, _union(after_left, self.flags))
        narrow = _union(left.narrow, right.narrow)
        self.set_flags(narrow, left.dtype is not BYTE or right.dtype is not BYTE)
        v = self.combine(e, kind, left, right)
        v.narrow = () if kind in RELATIONS else narrow
        return v

    def combine(self, e: P.BinaryOp, kind, left: _Val, right: _Val) -> _Val:  # pylint: disable=too-many-return-statements
        """The value of an operator of two operands, ``left`` and ``right``."""
        if kind in RELATIONS or kind in (BinaryOpKind.DIV, BinaryOpKind.MOD):
            self.read(left)
            self.read(right)
            c = _const(e.right)
            if kind in RELATIONS:
                return _Val(BYTE, 0xFF)
            return _Val(ADDRESS, min(left.top, c - 1) if kind == BinaryOpKind.MOD and c
                        else left.top)
        t = binary_type(kind, left.dtype, right.dtype)
        shifts = left.shifts + right.shifts
        full = mask(t or ADDRESS)
        if kind == BinaryOpKind.AND:
            # AND with a BYTE clears the bits above the low byte.
            if any(v.dtype is BYTE and not v.shifts for v in (left, right)):
                shifts = ()
            return _Val(t, min(left.top, right.top), shifts)
        if kind == BinaryOpKind.MUL:
            return _Val(ADDRESS, min(0xFFFF, left.top * right.top), shifts)
        if kind in (BinaryOpKind.ADD, BinaryOpKind.PLUS):
            top = left.top + right.top + (kind == BinaryOpKind.PLUS)
            return _Val(t, top if top <= full else full, shifts)
        if kind in (BinaryOpKind.OR, BinaryOpKind.XOR):
            return _Val(t, (1 << max(left.top, right.top).bit_length()) - 1, shifts)
        if kind == BinaryOpKind.SUB and right.top == 0:
            return _Val(t, left.top, shifts)
        return _Val(t, full, shifts)


def _nodes(tree, into_procs: bool = True):
    """Every node of ``tree``; not those of a procedure's body, but for
    ``into_procs``."""
    if isinstance(tree, (list, tuple)):
        for x in tree:
            yield from _nodes(x, into_procs)
        return
    if not hasattr(tree, "__dataclass_fields__") or hasattr(tree, "file_id"):
        return
    yield tree
    if isinstance(tree, P.ProcDecl) and not into_procs:
        return
    for f in tree.__dataclass_fields__:
        if f != "pos":
            yield from _nodes(getattr(tree, f, None), into_procs)


def _callees(tree) -> set[tuple[str, bool]]:
    """(name, CALLed) of what ``tree`` calls, or may: the names of CALLs and
    calls, and every name it uses without arguments, which may be a typed
    procedure's.  Not in the procedures it declares, which it does not run."""
    out: set[tuple[str, bool]] = set()
    for n in _nodes(tree, into_procs=False):
        if isinstance(n, P.Identifier):
            out.add((ident_text(n.name), False))
        elif isinstance(n, P.CallStmt):
            callee = unwrap_paren(n.callee)
            callee = unwrap_paren(callee.callee) if isinstance(callee, P.Call) else callee
            if isinstance(callee, P.Identifier):
                out.add((ident_text(callee.name), True))
    return out


def _declared(items) -> set[str]:
    """The names a procedure's body declares, in it or in a DO block in it."""
    out: set[str] = set()
    for n in _nodes(items, into_procs=False):
        if isinstance(n, P.ProcDecl):
            out.add(proc_name(n))
        elif isinstance(n, P.DeclItem):
            out.update(decl_item_names(n))
        elif isinstance(n, P.DeclItemBasedGroup):
            out.update(ident_text(b.name) for b in n.based_decls or [])
        elif isinstance(n, P.LiterallyDecl):
            out.add(ident_text(n.name))
    return out


def _modsets(modules) -> dict[int, frozenset]:
    """The names each procedure (by id of its declaration) may assign that
    it does not declare, itself or through the procedures it calls, by
    name.  An EXTERNAL procedure may call back any PUBLIC one."""
    procs = [n for n in _nodes(modules) if isinstance(n, P.ProcDecl)]
    by_name: dict[str, list] = {}
    for p in procs:
        by_name.setdefault(proc_name(p), []).append(p)
    local = {id(p): _declared(p.body.items) | set(proc_param_names(p)) for p in procs}
    direct = {id(p): _assigned(p.body.items, into_procs=False) - local[id(p)]
              for p in procs}
    calls = {id(p): {n for n, _ in _callees(p.body.items) if n in by_name} for p in procs}
    public = [p for p in procs if proc_attrs(p).is_public]
    mod: dict[int, frozenset] = {id(p): frozenset(direct[id(p)]) for p in procs}
    while True:
        new = {}
        for p in procs:
            got = set(direct[id(p)])
            for c in calls[id(p)]:
                for q in by_name[c]:
                    got |= mod[id(q)] - local[id(p)]
            if proc_attrs(p).is_external:
                for q in public:
                    got |= mod[id(q)]
            new[id(p)] = frozenset(got)
        if new == mod:
            return mod
        mod = new


def _system_reset(s: P.CallStmt) -> bool:
    """Whether ``s`` is `CALL MON1(0, ...)': BDOS's function 0, system
    reset, which does not return."""
    c = unwrap_paren(s.callee)
    callee = unwrap_paren(c.callee) if isinstance(c, P.Call) else None
    return (isinstance(callee, P.Identifier) and ident_text(callee.name).upper() == "MON1"
            and bool(c.args) and _const(c.args[0]) == 0)


def _no_return(modules) -> set[int]:
    """The procedures, by id of their declarations, that do not return:
    with no RETURN and no label on the END, which a GOTO reaches past the
    last statement, whose last statement is a call of one that does not,
    or of an EXTERNAL MON1 with the function 0, system reset."""
    procs = [n for n in _nodes(modules) if isinstance(n, P.ProcDecl)]
    by_name: dict[str, list] = {}
    for p in procs:
        by_name.setdefault(proc_name(p).upper(), []).append(p)
    external_mon1 = all(proc_attrs(p).is_external for p in by_name.get("MON1", []))
    last: dict[int, P.CallStmt] = {}
    for p in procs:
        stmts = [it for it in p.body.items
                 if not isinstance(it, (P.DeclareStmt, P.ProcDecl)) and not is_end_of_block(it)]
        s = stmts[-1] if stmts else None
        while isinstance(s, P.LabeledStmt):
            s = s.stmt
        if isinstance(s, P.CallStmt) and not proc_attrs(p).is_external and not any(
                isinstance(n, (P.ReturnStmt, P.ReturnStmtValue))
                for n in _nodes(p.body.items, into_procs=False)) and not any(
                    is_end_of_block(it) for it in p.body.items):
            last[id(p)] = s
    out: set[int] = set()
    while True:
        new = set(out)
        for key, s in last.items():
            c = unwrap_paren(s.callee)
            callee = unwrap_paren(c.callee) if isinstance(c, P.Call) else c
            name = ident_text(callee.name).upper() if isinstance(callee, P.Identifier) else ""
            if (_system_reset(s) and external_mon1) or (
                    by_name.get(name) and all(id(q) in out for q in by_name[name])):
                new.add(key)
        if new == out:
            return out
        out = new


def _module_body(m) -> list:
    items = [it for it in m.items if not isinstance(it, P.AddressLiteral)]
    if len(items) == 1 and isinstance(items[0], P.LabeledStmt) and isinstance(
            items[0].stmt, P.DoBlock):
        return list(items[0].stmt.items)
    return items


def _shared(modules) -> dict[str, _Sym]:
    """The PUBLIC names of ``modules``."""
    shared: dict[str, _Sym] = {}
    for m in modules:
        for d in _module_body(m):
            for x in d.declarations if isinstance(d, P.DeclareStmt) else [d]:
                if isinstance(x, P.ProcDecl) and proc_attrs(x).is_public:
                    shared[proc_name(x)] = _proc_sym(x)
                elif isinstance(x, (P.DeclItem, P.DeclItemBasedGroup)) \
                        and decl_attrs(x).is_public:
                    shared.update(_item_syms(x, set()))
    return shared


def check_byte_shifts(modules: list) -> list[tuple]:
    """(location, text) of each SHL of a BYTE in ``modules`` whose lost
    bits are read, or whose flags are (see the module docstring), in the
    order found."""
    shared = _shared(modules)
    modsets = _modsets(modules)
    # What an INTERRUPT procedure assigns, or what it calls, may change at
    # any time: no bound holds of it.
    pinned = _pinned(modules).union(*(
        modsets[id(p)] for p in _nodes(modules)
        if isinstance(p, P.ProcDecl) and proc_attrs(p).interrupt_num is not None))
    bounds: dict[tuple, int] = {}
    changes: dict[tuple, int] = {}
    last = _Checker(shared, pinned)     # what the flags may differ by, so far
    no_return = _no_return(modules)
    while True:
        # Each walk finds what the variables are assigned, taking them to be
        # at most what the last one found; until it finds no more.  One
        # that keeps growing, as `k = k + 1' does, can have any value.  And
        # what the flags may differ by where a procedure returns and at a
        # GOTO, as the last walk found, until it finds no more.
        c = _Checker(shared, pinned)
        c.bounds, c.modsets = bounds, modsets
        c.ret_flags, c.jump_flags = dict(last.ret_flags), last.jump_flags
        c.no_return = no_return
        for m in modules:
            c.block(_module_body(m))
        grown = {k: v for k, v in c.seen.items() if v > bounds.get(k, 0)}
        flags_grew = (len(c.jump_flags) > len(last.jump_flags) or any(
            len(v) > len(last.ret_flags.get(k, ())) for k, v in c.ret_flags.items()))
        last = c
        if not grown and not flags_grew:
            return list(c.warned.values())
        for k, v in grown.items():
            changes[k] = changes.get(k, 0) + 1
            bounds[k] = v if changes[k] < 3 else 0xFFFF
