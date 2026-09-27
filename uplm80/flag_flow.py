"""Where the flags an operation leaves can be read.

CARRY, ZERO, SIGN and PARITY read the flags the operation before them left,
PLUS and MINUS add or subtract its carry, SCL and SCR rotate through it, and
DEC adjusts by it and the half carry (Programming Manual 9800268B, 12.1 to
12.5).  :class:`FlagFlow` follows the flags from where an operation sets
them to each reader control flow can take them to: through the statements
after it, round loops, into both arms of an IF and each case of a DO CASE,
from a GOTO to every label of its name, into a procedure called, which is
entered with them, and from its RETURNs and its END back to the call.  A
CALL through an address may call any procedure whose address the program
takes, and an EXTERNAL procedure's code any PUBLIC one; an INTERRUPT
procedure is entered with whatever flags the code it interrupts has.  What
sets no flag, or only some, passes them on - a load, a store, a call, a
comparison, a statement - and so does whatever the checker cannot tell
apart.

One operation stops them: an addition, subtraction, AND, OR or XOR whose
operands code generation is certain to take for BYTEs
(:meth:`FlagFlow._bytes`), that is the value of an assignment statement,
or DEC of one, or a PLUS or MINUS of BYTEs whose left operand is one
(:meth:`FlagFlow._sets`).  Code generation makes it an 8-bit `add',
`sub', `and', `or' or `xor' in A, the last thing the value's code does,
which sets the carry, zero, sign, parity and half carry; the optimizer
leaves as they are the operations whose flags a reader can read
(:meth:`FlagFlow.live`), and the peephole optimizer changes no flag an
instruction after it reads.  A CALL of procedures each of which ends in
one every way stops them too.

:meth:`FlagFlow.shift_readers` follows the flags of each shift of a BYTE
to the readers they reach, each procedure's body once from its entry, and
what it returns with stands for it at each call.  A reader reads only such
an operation's flags where it is DEC of one, or a PLUS or MINUS of BYTEs
whose left operand is one, of operands that call, shift and read nothing
(:meth:`FlagFlow._kills`); any other is taken to read the flags of any
operation of its own statement, in whatever order the statement's
operands are evaluated.  :meth:`FlagFlow.live`, for the optimizer, goes
back from the readers, a procedure's exit to every call of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import _plm_parser as P
from .ast_view import (
    DOUBLE_MARK,
    BinaryOpKind,
    DataType,
    binop_kind,
    decl_item_names,
    decl_item_struct_members,
    decl_item_type,
    ident_text,
    is_end_of_block,
    iter_block_proc_decls,
    number_value,
    proc_attrs,
    proc_name,
    proc_return_type,
    string_value,
    unwrap_paren,
)
from .plm_types import BYTE_BUILTINS

BYTE = DataType.BYTE
ADDRESS = DataType.ADDRESS

# What reads the flags: the built-ins without arguments, those with, and
# the operators.
FLAG_NAMES = frozenset({"CARRY", "ZERO", "SIGN", "PARITY"})
_READER_CALLS = frozenset({"SCL", "SCR", "DEC"})
_READER_OPS = (BinaryOpKind.PLUS, BinaryOpKind.MINUS)
# The operations of two BYTEs that are certain to set every flag.
_KILL_OPS = (BinaryOpKind.ADD, BinaryOpKind.SUB, BinaryOpKind.AND, BinaryOpKind.OR,
             BinaryOpKind.XOR)
_BUILTIN_ADDRESS = frozenset({"DOUBLE", DOUBLE_MARK, "STACKPTR", "SIZE"})
_PATTERN_TYPED = frozenset({"SHL", "SHR", "SCL", "SCR"})
# Built-ins whose argument is a variable, not evaluated.
_EXTENTS = frozenset({"LENGTH", "LAST", "SIZE"})


@dataclass(eq=False)
class _Sym:
    """What a declared name is, as far as the flags go."""
    kind: str                       # "var", "proc", "label" or "other"
    dtype: DataType | None = None   # a variable's, or a procedure's result
    array: bool = False
    struct: bool = False
    node: P.ProcDecl | None = None


@dataclass(eq=False)
class _Call:
    """A call: of the procedures ``targets``, and, if ``opaque``, of code
    the program does not have (an EXTERNAL procedure's, or what a CALL
    through an address reaches), which may call any PUBLIC procedure."""
    targets: list
    opaque: bool = False
    through: bool = False           # through an address


@dataclass(eq=False)
class _Unit:  # pylint: disable=too-many-instance-attributes
    """A point of the program: a statement's evaluation, or a label, a
    procedure's entry or its exit."""
    succ: list = field(default_factory=list)
    stmt: object = None             # the statement whose code this is
    gens: list = field(default_factory=list)        # the shifts of a BYTE in it
    # (node, name, what it reads the flags of: the nodes of the operation
    # of its operand that sets every flag, or None)
    readers: list = field(default_factory=list)
    calls: list = field(default_factory=list)       # _Call
    nodes: list = field(default_factory=list)       # (node, under an operation that kills)
    kill: bool = False              # it stops the flags it is entered with
    exit_of: object = None          # the _Proc whose exit this is


@dataclass(eq=False)
class _Proc:
    node: P.ProcDecl
    entry: _Unit
    exit: _Unit
    external: bool = False
    interrupt: bool = False
    callers: list = field(default_factory=list)     # the units that call it


def module_body(m) -> list:
    """The items of a module: of its `name: DO; ... END;' block."""
    items = [it for it in m.items if not isinstance(it, P.AddressLiteral)]
    if len(items) == 1 and isinstance(items[0], P.LabeledStmt) and isinstance(
            items[0].stmt, P.DoBlock):
        return list(items[0].stmt.items)
    return items


def _nodes(tree):
    """Every node of ``tree``, procedures' bodies too."""
    if isinstance(tree, (list, tuple)):
        for x in tree:
            yield from _nodes(x)
        return
    if not hasattr(tree, "__dataclass_fields__") or hasattr(tree, "file_id"):
        return
    yield tree
    for f in tree.__dataclass_fields__:
        if f != "pos":
            yield from _nodes(getattr(tree, f, None))


def address_taken(tree, at: bool = False) -> set[str]:
    """The names whose address ``tree`` takes, `.p' or `.a(i).m', and with
    ``at`` those an AT names."""
    out: set[str] = set()
    for n in _nodes(tree):
        if isinstance(n, P.LocationOf) or (at and isinstance(n, P.AttrAt)):
            root = unwrap_paren(n.operand if isinstance(n, P.LocationOf) else n.address)
            while isinstance(root, (P.Call, P.MemberAccess, P.LocationOf)):
                root = unwrap_paren(root.callee if isinstance(root, P.Call) else
                                    root.base if isinstance(root, P.MemberAccess)
                                    else root.operand)
            if isinstance(root, (P.Identifier, P.DottedIdent)):
                out.add(ident_text(root.name))
    return out


def _scope(items) -> dict[str, _Sym]:
    """The names a block's declarations and labels introduce."""
    scope: dict[str, _Sym] = {}
    for it in items:
        for d in it.declarations if isinstance(it, P.DeclareStmt) else [it]:
            if isinstance(d, P.ProcDecl):
                scope[proc_name(d)] = _Sym("proc", proc_return_type(d), node=d)
            elif isinstance(d, (P.DeclItem, P.DeclItemBasedGroup)):
                dt, dim = decl_item_type(d)
                if isinstance(getattr(d, "tail", None), P.DeclTailData):
                    dt = BYTE           # an untyped DATA is BYTE
                names = ([ident_text(b.name) for b in d.based_decls or []]
                         if isinstance(d, P.DeclItemBasedGroup) else decl_item_names(d))
                kind = "label" if dt is DataType.LABEL else "var"
                for n in names:
                    scope[n] = _Sym(kind, dt if dt in (BYTE, ADDRESS) else None,
                                    array=dim is not None,
                                    struct=decl_item_struct_members(d) is not None)
            elif isinstance(d, P.LiterallyDecl):
                scope[ident_text(d.name)] = _Sym("other")
        s = it
        while isinstance(s, P.LabeledStmt):
            scope.setdefault(ident_text(s.label), _Sym("label"))
            s = s.stmt
    for p in iter_block_proc_decls(items):
        scope.setdefault(proc_name(p), _Sym("proc", proc_return_type(p), node=p))
    return scope


class FlagFlow:  # pylint: disable=too-many-instance-attributes
    """The control flow of ``modules``, and what the flags do along it."""

    def __init__(self, modules: list) -> None:
        self.modules = modules
        self.units: list[_Unit] = []
        self.procs: dict[int, _Proc] = {}
        self.labels: dict[str, list[_Unit]] = {}
        self.gotos: list[tuple[_Unit, str]] = []
        self.opaque_calls: list[_Call] = []
        self.scopes: list[dict[str, _Sym]] = []
        self.proc: _Proc | None = None
        self.pending: list[tuple[P.ProcDecl, list]] = []
        for m in modules:
            body = module_body(m)
            self.scopes = [_scope(body)]
            self.proc = None
            self._seq(body, [self._unit()])
        while self.pending:
            self._body(*self.pending.pop())
        self._connect()

    def _connect(self) -> None:
        """Each GOTO to every label of its name, each call through an
        address to every procedure whose address is taken, and each call of
        code the program does not have to every PUBLIC procedure."""
        for unit, name in self.gotos:
            unit.succ.extend(self.labels.get(name, []))
        taken = address_taken(self.modules)
        procs = [q for q in self.procs.values() if not q.external]
        public = [q for q in procs if proc_attrs(q.node).is_public]
        addressed = [q for q in procs if proc_name(q.node) in taken]
        for c in self.opaque_calls:
            c.targets = list(dict.fromkeys(public + (addressed if c.through else [])))
        for u in self.units:
            for c in u.calls:
                for q in c.targets:
                    q.callers.append(u)

    # ---- building ------------------------------------------------------

    def _unit(self, stmt=None) -> _Unit:
        u = _Unit(stmt=stmt)
        self.units.append(u)
        return u

    @staticmethod
    def _link(preds: list, unit: _Unit) -> None:
        for p in preds:
            p.succ.append(unit)

    def _lookup(self, name: str) -> _Sym | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def _proc_of(self, p: P.ProcDecl) -> _Proc:
        q = self.procs.get(id(p))
        if q is None:
            attrs = proc_attrs(p)
            q = _Proc(p, self._unit(), self._unit(), external=attrs.is_external,
                      interrupt=attrs.interrupt_num is not None)
            q.exit.exit_of = q
            self.procs[id(p)] = q
        return q

    def _declare(self, p: P.ProcDecl) -> None:
        """A procedure declared here: its body is built with the scopes it
        sees."""
        self._proc_of(p)
        if not proc_attrs(p).is_external:
            self.pending.append((p, list(self.scopes)))

    def _body(self, p: P.ProcDecl, scopes: list) -> None:
        q = self._proc_of(p)
        self.scopes = scopes + [_scope(p.body.items)]
        self.proc = q
        self._link(self._seq(p.body.items, [q.entry]), q.exit)

    def _seq(self, items, preds: list) -> list:
        for it in items:
            preds = self._stmt(it, preds)
        return preds

    def _block(self, items, preds: list) -> list:
        self.scopes.append(_scope(items))
        out = self._seq(items, preds)
        self.scopes.pop()
        return out

    def _stmt(self, s, preds: list) -> list:  # pylint: disable=too-many-return-statements,too-many-branches
        """Build statement ``s``, entered from ``preds``; the units control
        leaves it from."""
        if isinstance(s, P.LabeledStmt):
            u = self._unit()
            self._link(preds, u)
            self.labels.setdefault(ident_text(s.label).upper(), []).append(u)
            return self._stmt(s.stmt, [u])
        if isinstance(s, P.ProcDecl):
            self._declare(s)
            return preds
        if isinstance(s, P.DeclareStmt):
            for d in s.declarations:
                if isinstance(d, P.ProcDecl):
                    self._declare(d)
            return preds
        if isinstance(s, (P.DeclItem, P.DeclItemBasedGroup, P.LiterallyDecl, P.NullStmt,
                          P.HaltStmt, P.EnableStmt, P.DisableStmt)) or s is None:
            return preds
        if isinstance(s, (P.IfStmt, P.IfStmtElse)):
            c = self._eval(s, [s.condition], preds)
            out = self._stmt(s.then_stmt, [c])
            return out + (self._stmt(s.else_stmt, [c]) if isinstance(s, P.IfStmtElse) else [c])
        if isinstance(s, P.DoBlock):
            return self._block(s.items, preds)
        if isinstance(s, (P.DoWhileBlock, P.DoIterBlock, P.DoIterByBlock)):
            # The head is evaluated before each pass, and the loop is left
            # from it.
            head = [getattr(s, f, None) for f in ("condition", "start", "bound", "step")]
            h = self._eval(s, head, preds)
            self._link(self._block(s.items, [h]), h)
            return [h]
        if isinstance(s, P.DoCaseBlock):
            return self._case(s, preds)
        if isinstance(s, P.GotoStmt):
            u = self._eval(s, [], preds)
            self.gotos.append((u, ident_text(s.label).upper()))
            return []
        if isinstance(s, (P.ReturnStmt, P.ReturnStmtValue)):
            u = self._eval(s, [getattr(s, "value", None)], preds)
            if self.proc is not None:
                u.succ.append(self.proc.exit)
            return []
        if isinstance(s, P.AssignStmt):
            u = self._eval(s, [s.value], preds, places=list(s.targets))
            u.kill = self._sets(s.value)
            return [u]
        if isinstance(s, P.CallStmt):
            return [self._call_stmt(s, preds)]
        return [self._eval(s, [s], preds)]

    def _case(self, s: P.DoCaseBlock, preds: list) -> list:
        sel = self._eval(s, [s.selector], preds)
        self.scopes.append(_scope(s.items))
        out: list = []
        cases = False
        for it in s.items:
            if is_end_of_block(it):
                out = self._stmt(it, out if cases else [sel])
            elif isinstance(it, (P.DeclareStmt, P.ProcDecl)):
                self._stmt(it, [])
            else:
                cases = True
                out += self._stmt(it, [sel])
        self.scopes.pop()
        return out if cases else [sel]

    def _eval(self, stmt, exprs: list, preds: list, places: list | None = None) -> _Unit:
        """The unit of ``stmt`` that evaluates ``exprs`` and the subscripts
        of ``places``."""
        u = self._unit(stmt)
        self._link(preds, u)
        for e in exprs:
            self._expr(e, u, False)
        for t in places or []:
            self._place(t, u, False)
        return u

    def _call_stmt(self, s: P.CallStmt, preds: list) -> _Unit:
        u = self._unit(s)
        self._link(preds, u)
        callee = unwrap_paren(s.callee)
        inner, args = callee, []
        if isinstance(callee, (P.Call, P.CallNoArgs)):
            u.nodes.append((callee, False))
            inner = unwrap_paren(callee.callee)
            args = list(callee.args) if isinstance(callee, P.Call) else []
        for a in args:
            self._expr(a, u, False)
        sym = self._lookup(ident_text(inner.name)) if isinstance(inner, P.Identifier) else None
        if sym is not None and sym.kind == "proc":
            u.calls.append(self._call_of(sym.node))
        elif isinstance(inner, P.Identifier) and sym is None:
            pass                    # a built-in: MOVE, TIME, OUTPUT's port...
        else:
            if not isinstance(inner, P.Identifier):
                self._place(inner, u, False)
            u.calls.append(self._opaque(through=True))
        return u

    def _opaque(self, through: bool = False) -> _Call:
        c = _Call([], opaque=True, through=through)
        self.opaque_calls.append(c)
        return c

    def _call_of(self, p: P.ProcDecl) -> _Call:
        q = self._proc_of(p)
        return self._opaque() if q.external else _Call([q])

    # ---- expressions ----------------------------------------------------

    def _expr(self, e, u: _Unit, k1: bool) -> None:  # pylint: disable=too-many-branches
        """What evaluating ``e`` does, in ``u``; ``k1``: it is an operand of
        an operation that sets every flag after it."""
        if e is None or isinstance(e, (str, int)):
            return
        if isinstance(e, (list, tuple)):
            for x in e:
                self._expr(x, u, k1)
            return
        if not hasattr(e, "__dataclass_fields__") or hasattr(e, "file_id") \
                or isinstance(e, P.ProcDecl):
            return
        u.nodes.append((e, k1))
        if isinstance(e, P.ParenExpr):
            self._expr(e.inner, u, k1)
        elif isinstance(e, (P.Identifier, P.CallNoArgs)):
            name = unwrap_paren(e.callee) if isinstance(e, P.CallNoArgs) else e
            if isinstance(name, P.Identifier):
                self._name(name, e, u)
            else:
                self._expr(name, u, k1)
        elif isinstance(e, P.Call):
            self._call(e, u, k1)
        elif isinstance(e, P.BinaryOp):
            if binop_kind(e) in _READER_OPS:
                u.readers.append((e, binop_kind(e).name,
                                  self._chain(e.left) if self._reads_its_left(e) else None))
            inner = k1 or self._k1(e)
            self._expr(e.left, u, inner)
            self._expr(e.right, u, inner)
        elif isinstance(e, P.LocationOf):
            self._place(e.operand, u, k1)
        elif isinstance(e, P.EmbeddedAssign):
            self._place(e.target, u, k1)
            self._expr(e.value, u, k1)
        elif isinstance(e, P.MemberAccess):
            self._place(e, u, k1)
        else:
            for f in e.__dataclass_fields__:
                if f != "pos":
                    self._expr(getattr(e, f), u, k1)

    def _name(self, ident: P.Identifier, node, u: _Unit) -> None:
        """A name used without arguments: a flag, a typed procedure called,
        or a variable."""
        text = ident_text(ident.name)
        sym = self._lookup(text)
        if sym is None and text.upper() in FLAG_NAMES:
            u.readers.append((node, text.upper(), None))
        elif sym is not None and sym.kind == "proc":
            u.calls.append(self._call_of(sym.node))

    def _call(self, e: P.Call, u: _Unit, k1: bool) -> None:
        callee = unwrap_paren(e.callee)
        if not isinstance(callee, P.Identifier):
            self._place(callee, u, k1)
            self._expr(e.args, u, k1)
            return
        text = ident_text(callee.name)
        sym = self._lookup(text)
        name = text.upper() if sym is None else ""
        if name in ("SHL", "SHR") and len(e.args) == 2 and not self._address(e.args[0]):
            u.gens.append(e)
        elif name in _READER_CALLS:
            u.readers.append((e, name, self._chain(e.args[0]) if name == "DEC" and len(
                e.args) == 1 and self._kills(e.args[0]) else None))
        elif name in _EXTENTS:
            return                  # its argument is not evaluated
        elif sym is not None and sym.kind == "proc":
            u.calls.append(self._call_of(sym.node))
        self._expr(e.args, u, k1)

    def _place(self, t, u: _Unit, k1: bool) -> None:
        """A place stored to, or whose address is taken: only its
        subscripts are evaluated."""
        t = unwrap_paren(t)
        if isinstance(t, P.Call):
            u.nodes.append((t, k1))
            if not isinstance(unwrap_paren(t.callee), P.Identifier):
                self._place(t.callee, u, k1)
            self._expr(t.args, u, k1)
        elif isinstance(t, P.MemberAccess):
            u.nodes.append((t, k1))
            self._place(t.base, u, k1)
        elif not isinstance(t, (P.Identifier, P.DottedIdent, P.DottedMember)):
            self._expr(t, u, k1)

    # ---- what the flags are certain of ------------------------------------

    def _address(self, e) -> bool:  # pylint: disable=too-many-return-statements
        """Whether ``e`` is certain to be an ADDRESS, whatever the rest of
        the program says: a SHL or SHR of anything else may be of a BYTE."""
        e = unwrap_paren(e)
        if isinstance(e, P.NumberLiteral):
            return number_value(e) > 0xFF
        if isinstance(e, P.StringLiteral):
            return len(string_value(e)) == 2
        if isinstance(e, (P.LocationOf, P.LocationOfList, P.LocationOfString)):
            return True
        if isinstance(e, P.UnaryOp):
            return self._address(e.operand)
        if isinstance(e, P.BinaryOp):
            kind = binop_kind(e)
            if kind in (BinaryOpKind.MUL, BinaryOpKind.DIV, BinaryOpKind.MOD):
                return True
            return kind not in _RELATIONS and (self._address(e.left) or self._address(e.right))
        name = e.callee if isinstance(e, (P.Call, P.CallNoArgs)) else e
        name = unwrap_paren(name)
        if not isinstance(name, P.Identifier):
            return False
        sym = self._lookup(ident_text(name.name))
        if sym is None:
            upper = ident_text(name.name).upper()
            if upper in ("SHL", "SHR", "SCL", "SCR") and isinstance(e, P.Call) and e.args:
                return self._address(e.args[0])
            return upper in _BUILTIN_ADDRESS
        if sym.kind == "var" and not sym.struct:
            return sym.dtype is ADDRESS and (sym.array == isinstance(e, P.Call))
        return sym.kind == "proc" and sym.dtype is ADDRESS

    def _byte(self, e) -> bool:  # pylint: disable=too-many-return-statements
        """Whether ``e`` is a BYTE code generation computes in A without a
        call, a shift or a flag read: a BYTE constant or variable, an
        element of a BYTE array, and +, -, AND, OR, XOR, NOT and minus of
        them."""
        e = unwrap_paren(e)
        if isinstance(e, P.NumberLiteral):
            return number_value(e) <= 0xFF
        if isinstance(e, P.StringLiteral):
            return len(string_value(e)) == 1
        if isinstance(e, P.Identifier):
            sym = self._lookup(ident_text(e.name))
            return (sym is not None and sym.kind == "var" and sym.dtype is BYTE
                    and not sym.array and not sym.struct)
        if isinstance(e, P.Call) and isinstance(unwrap_paren(e.callee), P.Identifier):
            sym = self._lookup(ident_text(unwrap_paren(e.callee).name))
            return (sym is not None and sym.kind == "var" and sym.dtype is BYTE and sym.array
                    and not sym.struct and len(e.args) == 1 and self._pure(e.args[0]))
        if isinstance(e, P.BinaryOp):
            return binop_kind(e) in _KILL_OPS and self._byte(e.left) and self._byte(e.right)
        if isinstance(e, P.UnaryOp):
            return self._byte(e.operand)
        return False

    def _pure(self, e) -> bool:
        """Whether evaluating ``e`` calls nothing, shifts nothing and reads no
        flag: constants, variables, subscripts and operators of them."""
        for n in _nodes(e):
            if isinstance(n, P.BinaryOp) and binop_kind(n) in _READER_OPS:
                return False
            if isinstance(n, (P.Identifier, P.CallNoArgs)):
                name = unwrap_paren(n.callee) if isinstance(n, P.CallNoArgs) else n
                if not isinstance(name, P.Identifier):
                    return False
                sym = self._lookup(ident_text(name.name))
                if sym is None and ident_text(name.name).upper() != DOUBLE_MARK.upper() \
                        or sym is not None and sym.kind not in ("var", "other"):
                    return False
        return True

    def _k1(self, e) -> bool:
        """An 8-bit operation of two BYTEs that sets every flag: +, -, AND,
        OR or XOR.  Code generation computes one of constants too, where a
        statement's value, DEC's argument or the left operand of PLUS or
        MINUS is one (`ld a,34h / add a,21h')."""
        e = unwrap_paren(e)
        return (isinstance(e, P.BinaryOp) and binop_kind(e) in _KILL_OPS
                and self._byte(e.left) and self._byte(e.right))

    def _kills(self, e) -> bool:
        """Whether the flags after evaluating ``e`` are certain to be those
        of an operation of it that sets every flag: one of two BYTEs, or DEC
        of it, or a PLUS or MINUS of BYTEs whose left operand is one."""
        e = unwrap_paren(e)
        if self._k1(e):
            return True
        if isinstance(e, P.Call) and isinstance(unwrap_paren(e.callee), P.Identifier):
            text = ident_text(unwrap_paren(e.callee).name)
            return (text.upper() == "DEC" and self._lookup(text) is None and len(e.args) == 1
                    and self._kills(e.args[0]))
        return isinstance(e, P.BinaryOp) and binop_kind(e) in _READER_OPS \
            and self._reads_its_left(e)

    def _bytes(self, e) -> bool:  # pylint: disable=too-many-return-statements
        """Whether code generation is certain to take ``e`` for a BYTE, and
        compute it in A: a BYTE constant, variable, element of a BYTE
        array, typed procedure or built-in, a shift of one, and relations,
        +, -, AND, OR, XOR, PLUS, MINUS, NOT and minus of them."""
        e = unwrap_paren(e)
        if self._byte(e):
            return True
        if isinstance(e, P.BinaryOp):
            kind = binop_kind(e)
            return kind in _RELATIONS or (kind in _KILL_OPS + _READER_OPS and self._bytes(e.left)
                                          and self._bytes(e.right))
        if isinstance(e, P.UnaryOp):
            return self._bytes(e.operand)
        if isinstance(e, P.EmbeddedAssign):
            return self._bytes(e.value)
        name = unwrap_paren(e.callee) if isinstance(e, (P.Call, P.CallNoArgs)) else e
        if not isinstance(name, P.Identifier):
            return False
        sym = self._lookup(ident_text(name.name))
        if sym is None:
            upper = ident_text(name.name).upper()
            args = e.args if isinstance(e, P.Call) else []
            if upper in _PATTERN_TYPED:
                return bool(args) and self._bytes(args[0])
            return upper in BYTE_BUILTINS and (bool(args) or upper in FLAG_NAMES)
        if sym.kind == "var" and not sym.struct:
            return sym.dtype is BYTE and isinstance(e, P.Call) == sym.array
        return sym.kind == "proc" and sym.dtype is BYTE

    def _sets(self, e) -> bool:
        """Whether the flags after evaluating ``e`` are certain to be those
        of an operation of it that sets every flag, whatever its operands
        are: an addition, subtraction, AND, OR or XOR of two BYTEs, DEC of
        it, or a PLUS or MINUS of BYTEs of it."""
        e = unwrap_paren(e)
        if not isinstance(e, (P.BinaryOp, P.Call)):
            return False
        if isinstance(e, P.Call):
            callee = unwrap_paren(e.callee)
            return (isinstance(callee, P.Identifier) and ident_text(callee.name).upper() == "DEC"
                    and self._lookup(ident_text(callee.name)) is None and len(e.args) == 1
                    and self._sets(e.args[0]))
        kind = binop_kind(e)
        both = self._bytes(e.left) and self._bytes(e.right)
        return both and (kind in _KILL_OPS or (kind in _READER_OPS and self._sets(e.left)))

    def _chain(self, e) -> list:
        """The nodes of ``e``, of which :meth:`_kills` holds, whose flags
        are those after it: DEC's, a PLUS's or MINUS's and its left
        operand's, and the operation that sets every flag."""
        out = [e]
        while isinstance(e, P.ParenExpr):
            e = e.inner
            out.append(e)
        if isinstance(e, P.Call):
            out += self._chain(e.args[0])
        elif isinstance(e, P.BinaryOp) and binop_kind(e) in _READER_OPS:
            out += self._chain(e.left)
        return out

    def _reads_its_left(self, e: P.BinaryOp) -> bool:
        """A PLUS or MINUS of BYTEs whose left operand leaves flags of its
        own, which code generation keeps across the right one (`push af /
        pop af') for the `adc' or `sbc' to read."""
        return self._kills(e.left) and self._byte(e.right)

    # ---- the flags of the shifts of a BYTE (the warning) -------------------

    def shift_readers(self) -> list[tuple]:
        """(reader node, reader's name, [SHL and SHR nodes]) of each flag
        reader the flags a shift of a BYTE sets can reach, in the order the
        readers are in the program.

        Each procedure's body is followed from its entry, where the flags
        are a mark of their own, its callers'; what it returns with, of its
        own shifts and of that mark, stands for it at each call, which
        passes on the caller's flags only where the procedure may set none.
        What the mark stands for is what its calls enter it with."""
        gens = list({id(g): g for u in self.units for g in u.gens}.values())
        if not gens:
            return []
        procs = [q for q in self.procs.values() if not q.external]
        index = {id(g): i for i, g in enumerate(gens)}
        mark = {id(q): 1 << (len(gens) + j) for j, q in enumerate(procs)}
        own = {id(u): _mask(u.gens, index) for u in self.units}
        flags, returns = self._spread(own, mark, procs)
        shifts = (1 << len(gens)) - 1
        every = self._entered(procs, shifts, mark,
                              lambda u: self._at_call(u, own, flags, (returns, mark)))
        found: dict[int, list] = {}
        for u in self.units:
            for node, name, chain in u.readers:
                bits = 0 if chain else self._at_call(u, own, flags, (returns, mark))
                for q in procs:
                    if bits & mark[id(q)]:
                        bits |= every[id(q)]
                if bits & shifts:
                    found.setdefault(id(node), [node, name, 0])[2] |= bits & shifts
        order = {id(n): i for i, n in enumerate(_nodes(self.modules))}
        return [(node, name, [g for g in gens if bits >> index[id(g)] & 1])
                for node, name, bits in sorted(found.values(), key=lambda t: order[id(t[0])])]

    def _entered(self, procs: list, shifts: int, mark: dict, at_call) -> dict:
        """The shifts, as bits of ``shifts``, whose flags each procedure may
        be entered with: of its calls, where the flags of a caller's own
        entry stand for what the caller is entered with.  An INTERRUPT
        procedure is entered with those of any."""
        entered = {id(q): 0 for q in procs}
        for u in self.units:
            for c in u.calls:
                for q in c.targets:
                    entered[id(q)] |= at_call(u)
        every = {id(q): shifts if q.interrupt else entered[id(q)] & shifts for q in procs}
        changed = True
        while changed:
            changed = False
            for q in procs:
                bits = every[id(q)]
                for r in procs:
                    if entered[id(q)] & mark[id(r)]:
                        bits |= every[id(r)]
                if bits != every[id(q)]:
                    every[id(q)], changed = bits, True
        return every

    @staticmethod
    def _sets_by_call(u: _Unit, flags: dict, mark: dict) -> bool:
        """Whether ``u`` is a CALL, the last thing it does, of procedures
        none of which the flags it is entered with can leave: each sets
        every flag, or does not return, on every way to its end."""
        return isinstance(u.stmt, P.CallStmt) and bool(u.calls) and all(
            not c.opaque and c.targets and not any(
                flags[id(q.exit)] & mark[id(q)] for q in c.targets)
            for c in u.calls)

    @staticmethod
    def _at_call(u: _Unit, own: dict, flags: dict, returns: tuple) -> int:
        """What the flags may be of in ``u``, at a call or a reader in it:
        what it is entered with, and what it sets itself or its calls
        return with, in whatever order it evaluates them."""
        rets, mark = returns
        out = own[id(u)] | flags[id(u)]
        for c in u.calls:
            for q in c.targets:
                out |= rets.get(id(q), 0) & ~mark[id(q)]
        return out

    def _spread(self, own: dict, mark: dict, procs: list) -> tuple[dict, dict]:
        """What the flags may be of where each unit is entered, and what
        each procedure returns with of its own (its mark, where it may
        return with the flags it is entered with)."""
        flags: dict[int, int] = {id(u): 0 for u in self.units}
        returns: dict[int, int] = {}
        for q in procs:
            flags[id(q.entry)] = mark[id(q)]
        work = list(self.units)
        queued = {id(u) for u in work}

        def reach(unit: _Unit, bits: int) -> None:
            if flags[id(unit)] | bits != flags[id(unit)]:
                flags[id(unit)] |= bits
                if id(unit) not in queued:
                    queued.add(id(unit))
                    work.append(unit)

        while work:
            u = work.pop()
            queued.discard(id(u))
            here = own[id(u)]
            for c in u.calls:
                for q in c.targets:
                    # The procedure's own, and the caller's where it may
                    # return with them, which ``flags`` has.
                    here |= returns.get(id(q), 0) & ~mark[id(q)]
            for s in u.succ:
                reach(s, here | (0 if u.kill or self._sets_by_call(u, flags, mark)
                                 else flags[id(u)]))
            q = u.exit_of
            if q is not None and flags[id(u)] != returns.get(id(q), 0):
                returns[id(q)] = flags[id(u)]
                for c in q.callers:
                    if id(c) not in queued:
                        queued.add(id(c))
                        work.append(c)
        return flags, returns

    # ---- where flags can be read (the optimizer) ---------------------------

    def live(self) -> set[int]:
        """The nodes, expressions and statements, whose flags a reader can
        read, and every reader: the optimizer leaves them as they are."""
        if not any(u.readers for u in self.units):
            return set()
        entry_live: dict[int, bool] = {}
        live_in: dict[int, bool] = {id(u): False for u in self.units}

        def reads(u: _Unit) -> bool:
            # A reader of the flags its operand's operation sets reads no
            # others.
            return any(chain is None for _, _, chain in u.readers) or any(
                entry_live.get(id(q), False) for c in u.calls for q in c.targets)

        def after(u: _Unit) -> bool:
            return any(live_in[id(s)] for s in u.succ)

        changed = True
        while changed:
            changed = False
            for u in reversed(self.units):
                if live_in[id(u)]:
                    continue
                q = u.exit_of
                now = (reads(u) or (after(u) and not u.kill)) if q is None else any(
                    reads(c) or after(c) for c in q.callers)
                if now:
                    live_in[id(u)] = changed = True
            for q in self.procs.values():
                if live_in[id(q.entry)] and not entry_live.get(id(q)):
                    entry_live[id(q)] = changed = True
        if any(q.interrupt and entry_live.get(id(q)) for q in self.procs.values()):
            # Entered with the flags of whatever it interrupts.
            return {id(node) for u in self.units for node, _ in u.nodes} | {
                id(u.stmt) for u in self.units if u.stmt is not None}
        out: set[int] = set()
        for u in self.units:
            for node, _, chain in u.readers:
                out.update(id(n) for n in [node] + (chain or []))
            if reads(u) or after(u):
                if u.stmt is not None:
                    out.add(id(u.stmt))
                out.update(id(node) for node, killed in u.nodes if not killed)
        return out


_RELATIONS = (BinaryOpKind.EQ, BinaryOpKind.NE, BinaryOpKind.LT, BinaryOpKind.GT,
              BinaryOpKind.LE, BinaryOpKind.GE)


def _mask(nodes: list, index: dict) -> int:
    out = 0
    for n in nodes:
        out |= 1 << index[id(n)]
    return out


def _order(node) -> tuple:
    pos = getattr(node, "pos", None)
    return (getattr(pos, "start_line", 0), getattr(pos, "start_column", 0))


def flag_live(modules: list) -> set[int]:
    """The nodes of ``modules`` whose flags a flag reader can read
    (:meth:`FlagFlow.live`)."""
    if not any(_reads_flags(n) for n in _nodes(modules)):
        return set()
    return FlagFlow(modules).live()


def _reads_flags(n) -> bool:
    """Whether node ``n`` may read the flags: a PLUS or MINUS, or a name
    that may be a flag reader's."""
    if isinstance(n, P.BinaryOp):
        return binop_kind(n) in _READER_OPS
    if isinstance(n, P.Identifier):
        return ident_text(n.name).upper() in FLAG_NAMES | _READER_CALLS
    return False
