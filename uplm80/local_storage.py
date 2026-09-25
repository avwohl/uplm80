"""Which of a procedure's local variables may share ``??AUTO``.

PL/M-80 allocates a procedure's variables statically (Programming Manual,
8.1.7): a local keeps its value from one call of the procedure to the next,
and a program may rely on that - a ``first$time`` flag, a running count.
uplm80 saves memory by overlaying the locals of procedures that are never
active together in one block, ``??AUTO``, which is only right for a local
whose old value no call can see.  This module finds those locals.  A local
may share ``??AUTO`` when

* on every path from the procedure's entry it is written before it is read:
  a definite-assignment analysis over the procedure's statements, GOTOs
  included, in which a call that can reach a procedure nested in this one
  and naming the local counts as a read of it at the call;
* its address is never taken - no ``.x``, in a statement or in the INITIAL,
  DATA or AT of any declaration - and it is not reached by subscripting it
  outside its bounds (a subscripted scalar, a constant subscript past its
  end, a variable subscript of a one-element array);
* no procedure nested in its own that names it, or that calls one that
  does, has its address taken, since a call through that address can come
  when the procedure the local belongs to is not active; and
* the same holds for every name declared with it in one factored
  declaration, which the manual (6.2.4) makes contiguous.

DRI lays a procedure's locals out in declaration order, and programs count
on it, so the split keeps that order wherever a program can see it.  The
locals with INITIAL or PUBLIC are static in any case, but they have their
place in that order too.

* A local whose address is taken, or which is reached outside its bounds,
  takes every local declared after it to static storage with it: a pointer
  made from it can reach them.
* An array or structure subscripted by anything but a constant may run on
  into the locals declared after it, as in DRI's layout (and in 0.3.6's
  frame in ``??AUTO``), so it and every local declared after it are all
  static, in declaration order, or all in ``??AUTO``, in the same order:
  if any of them is static, all of them are.  A read through such a
  subscript reads all of those locals.

A subscript is a constant when it folds to one by PL/M-80's rules, as the
optimizer folds it, so every level decides alike.

A BASED variable reads its base each time it is used, so a use of it is a
read of the base.  An array or a structure is followed element by element
(up to ``MAX_COMPONENTS`` of them): it is assigned once every element has
been assigned through constant subscripts, and a read of one element through
constant subscripts needs only that element.  Any other read needs the
whole.  A store through a variable subscript assigns nothing.

Parameters are written by every call, so they share ``??AUTO`` unless
their address is taken - a parameter is as static as any other local, and a
pointer to it may be kept after the call - or a procedure whose address is
taken names one.  They come first in the order that is kept, as the
PROCEDURE statement lists them.  Anything the analysis cannot follow leaves
the local in static storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from . import _plm_parser as P
from .ast_view import (
    DataType,
    binop_kind,
    block_items_split,
    decl_attrs,
    decl_item_based,
    decl_item_names,
    decl_item_struct_members,
    decl_item_type,
    ident_text,
    proc_attrs,
    proc_body_items,
    proc_name,
    proc_param_names,
    struct_member_dim,
    struct_member_names,
    unop_kind,
    unwrap_paren,
)
from .plm_types import fold_binary, fold_builtin, fold_unary, literal_type, typed_const

# An array or structure with more elements than this is not followed element
# by element: any read of it before... anything needs the whole, which no
# store assigns.
MAX_COMPONENTS = 64

# Builtins whose argument is not evaluated: they give its size.
_SIZE_BUILTINS = frozenset({"LENGTH", "LAST", "SIZE"})

_DO_KINDS = (P.DoBlock, P.DoWhileBlock, P.DoIterBlock, P.DoIterByBlock, P.DoCaseBlock)

# Passes over a loop or a body before the analysis gives up on it.
_MAX_PASSES = 64

# A dataflow state: the components definitely assigned, or None where no
# path reaches (the identity of the meet).
State = Optional[frozenset]

# The one component of a local too large to follow element by element; no
# store assigns it.
_NEVER = ("?never",)


def _meet(a: State, b: State) -> State:
    if a is None:
        return b
    if b is None:
        return a
    return a & b


# What a Local is.  Only an AUTO local may share ??AUTO; a FIXED one - with
# INITIAL, or PUBLIC - is static whatever the analysis finds, but it has its
# place in the procedure's storage, next to the locals declared around it.
# A PARAM is in ??AUTO unless its address is taken or a procedure whose
# address is taken names it; the parameters come first, in the order the
# PROCEDURE statement lists them.
AUTO = "auto"
FIXED = "fixed"
PARAM = "param"


@dataclass
class Local:  # pylint: disable=too-many-instance-attributes
    """A variable with storage of the procedure's own, where DRI's layout
    puts it: in declaration order."""

    name: str
    order: int                    # position among the procedure's locals
    group: int                    # which declaration it is in; -1 for a parameter
    dim: Optional[int]            # array dimension, None for a scalar
    members: Optional[dict]       # structure member -> its dimension or None
    kind: str = AUTO
    keys: frozenset = frozenset()  # every component

    def __post_init__(self) -> None:
        keys = self._components()
        self.keys = frozenset(keys) if keys is not None else frozenset({_NEVER})

    def _components(self) -> Optional[list]:
        per_elem: list[tuple] = []
        if self.members is None:
            per_elem.append(())
        else:
            for m, mdim in self.members.items():
                if mdim is None:
                    per_elem.append((m,))
                elif 0 < mdim <= MAX_COMPONENTS:
                    per_elem.extend((m, k) for k in range(mdim))
                else:
                    return None
        if self.dim is None:
            out = [(self.name,) + e for e in per_elem]
        elif 0 < self.dim <= MAX_COMPONENTS:
            out = [(self.name, i) + e for i in range(self.dim) for e in per_elem]
        else:
            return None
        return out if len(out) <= MAX_COMPONENTS else None

    @property
    def followed(self) -> bool:
        """Whether its elements are followed one by one."""
        return _NEVER not in self.keys


@dataclass
class ProcInfo:
    """What the analysis knows of one procedure."""

    full: str
    decl: object
    chain: list                    # scope frames outside the body, outermost first
    locals: dict[str, Local] = field(default_factory=dict)
    labels: set[str] = field(default_factory=set)   # defined in its own body
    frame: dict = field(default_factory=dict)       # its own names -> binding


# A binding in a scope frame is (kind, owner, extra).  kind is "local" (a
# Local of procedure `owner'), "var" (any other variable), "proc", or "based"
# (a BASED variable; extra is its declaration).  owner is None in a block
# or at module level.


def _designator(expr):
    """(root name, [part, ...]) for a variable reference, or None.

    A part is ("idx", args) for a subscript and ("mem", name) for a member.
    :meth:`_Walk._fold_path` replaces each subscript's args with their
    values, None where one is not constant, for :func:`_select`.
    """
    expr = unwrap_paren(expr)
    if isinstance(expr, P.Identifier):
        return ident_text(expr.name), []
    if isinstance(expr, P.Call):
        inner = _designator(expr.callee)
        if inner is None:
            return None
        return inner[0], inner[1] + [("idx", list(expr.args))]
    if isinstance(expr, P.MemberAccess):
        inner = _designator(expr.base)
        if inner is None:
            return None
        return inner[0], inner[1] + [("mem", ident_text(expr.member))]
    return None


def _subscript(part, dim: int) -> tuple[Optional[int], bool]:
    """(constant index or None, whether it reaches outside the object)."""
    values = part[1]
    if part[0] != "idx" or len(values) != 1:
        return None, True
    k = values[0]
    if k is None:
        # A one-element array is PL/M's open-ended array.
        return None, dim <= 1
    return (k, False) if 0 <= k < dim else (None, True)


def _varies(path: list) -> bool:
    """Whether a folded ``path`` has a subscript that is not a constant."""
    return any(part[0] == "idx" and (len(part[1]) != 1 or part[1][0] is None)
               for part in path)


def _is_zero_subscript(rest: list) -> bool:
    """Whether ``rest`` is exactly ``(0)``: x(0) is x itself."""
    return (len(rest) == 1 and rest[0][0] == "idx" and len(rest[0][1]) == 1
            and rest[0][1][0] == 0)


def _path_prefix(local: Local, path: list) -> tuple[Optional[tuple], bool, bool]:  # pylint: disable=too-many-return-statements
    """(component prefix, exact, reaches outside) of a reference into ``local``.

    The reference touches the components that start with the prefix (all of
    them if it is None); `exact' when a store to it assigns all of those.
    """
    rest = list(path)
    prefix: tuple = (local.name,)
    if local.dim is not None:
        if not rest or rest[0][0] != "idx":
            return None, False, False
        k, reach = _subscript(rest.pop(0), local.dim)
        if k is None:
            return None, False, reach
        prefix += (k,)
    if local.members is None:
        if not rest or _is_zero_subscript(rest):
            return prefix, True, False
        return None, False, True
    if not rest:
        return prefix, False, False
    if rest[0][0] != "mem" or rest[0][1] not in local.members:
        return None, False, True
    member = rest.pop(0)[1]
    mdim = local.members[member]
    prefix += (member,)
    if mdim is None:
        if not rest or _is_zero_subscript(rest):
            return prefix, True, False
        return None, False, True
    if not rest:
        # An array member without a subscript is its first element: read
        # the whole member, assign none of it.
        return prefix, False, False
    k, reach = _subscript(rest.pop(0), mdim)
    if reach or rest:
        return None, False, True
    if k is None:
        return prefix, False, False
    return prefix + (k,), True, False


def _select(local: Local, path: list) -> tuple[frozenset, bool, bool]:
    """(components, exact, reaches outside) of a reference into ``local``."""
    prefix, exact, reach = _path_prefix(local, path)
    if prefix is None or not local.followed:
        return local.keys, False, reach
    n = len(prefix)
    return frozenset(c for c in local.keys if c[:n] == prefix), exact, reach


class _Effects:  # pylint: disable=too-few-public-methods
    """What evaluating an expression does to the procedure's own locals."""

    def __init__(self) -> None:
        self.reads: list[tuple[str, frozenset]] = []
        self.writes: list[frozenset] = []
        self.calls: list[str] = []


class LocalStorage:  # pylint: disable=too-many-instance-attributes
    """The analysis over a program's modules; see the module docstring.

    ``resolve_proc(name, current)`` gives the full name of the procedure
    ``name`` means inside procedure ``current``, as the call graph names it,
    and ``call_graph`` maps each procedure to the procedures it calls.
    """

    def __init__(self, resolve_proc: Callable[[str, str], Optional[str]],
                 call_graph: dict[str, set[str]]) -> None:
        self.resolve_proc = resolve_proc
        self.call_graph = call_graph
        self.procs: dict[str, ProcInfo] = {}
        # procedure -> (owner, name) of the outer locals its body names
        self.free_refs: dict[str, set[tuple[str, str]]] = {}
        # (owner, name) -> why that local is static
        self.reasons: dict[tuple[str, str], str] = {}
        # locals whose address is taken or which are reached outside their
        # bounds: they and everything declared after them are static
        self.escapes: set[tuple[str, str]] = set()
        # arrays and structures subscripted by a variable, which may run on
        # into the locals declared after them: they and those locals share
        # one storage class, in declaration order
        self.runs: set[tuple[str, str]] = set()
        self.proc_addr_taken: set[str] = set()
        # INTERRUPT procedures, which run whenever the interrupt comes
        self.interrupts: set[str] = set()
        # PUBLIC procedures, which code outside the module can call
        self.public: set[str] = set()
        # procedures with a CALL through an address (8.2.1)
        self.indirect_callers: set[str] = set()
        # each module's items and the names its declarations bring in
        self.modules: list[tuple[list, dict]] = []
        self._surveyed = False
        # procedure -> labels it jumps to that are not its own
        self.outward_gotos: dict[str, set[str]] = {}
        # (procedure, label) that can be jumped to from anywhere
        self.label_addr_taken: set[tuple[str, str]] = set()
        self._trans_cache: dict[str, set[tuple[str, str]]] = {}

    # ---- collection --------------------------------------------------------

    def add_module(self, items: list) -> None:
        """Find every procedure of one module and the scope it is declared in.

        ``items`` are the module's declarations and statements.
        """
        decls, _ = _module_split(items)
        frame = _frame(decls, None)
        self.modules.append((items, frame))
        self._scan_items(items, None, [(None, frame)])

    def _scan_items(self, items, parent: Optional[str], chain: list) -> None:
        """Walk ``items`` (a body or block) for procedures, keeping the scope."""
        for it in items:
            if isinstance(it, P.ProcDecl):
                self._add_proc(it, parent, chain)
            elif isinstance(it, P.DeclareStmt):
                for inner in it.declarations:
                    if isinstance(inner, P.ProcDecl):
                        self._add_proc(inner, parent, chain)
            elif isinstance(it, _DO_KINDS):
                decls, _ = block_items_split(it.items)
                frame = _frame(decls, None)
                self._scan_items(it.items, parent, chain + [(None, frame)])
            elif isinstance(it, (P.IfStmt, P.IfStmtElse)):
                self._scan_items([it.then_stmt], parent, chain)
                if isinstance(it, P.IfStmtElse):
                    self._scan_items([it.else_stmt], parent, chain)
            elif isinstance(it, P.LabeledStmt):
                self._scan_items([it.stmt], parent, chain)

    def _add_proc(self, decl, parent: Optional[str], chain: list) -> None:
        attrs = proc_attrs(decl)
        if attrs.is_external:
            return
        name = proc_name(decl)
        full = f"{parent}${name}" if parent and not attrs.is_public else name
        params = proc_param_names(decl)
        info = ProcInfo(full=full, decl=decl, chain=list(chain))
        if attrs.interrupt_num is not None:
            self.interrupts.add(full)
        if attrs.is_public:
            self.public.add(full)
        items = proc_body_items(decl)
        decls, stmts = block_items_split(items)
        if not attrs.is_reentrant:
            info.locals = _layout(decls, params)
        for d in decls:
            if (isinstance(d, P.DeclItem) and decl_item_type(d)[0] == DataType.LABEL
                    and decl_attrs(d).is_public):
                self.label_addr_taken.update((full, n) for n in decl_item_names(d))
        info.frame = _frame(decls, full, info.locals)
        for p in params:
            info.frame.setdefault(p, ("local" if p in info.locals else "var", full, None))
        info.labels = _labels(stmts)
        self.procs[full] = info
        self._scan_items(items, full, chain + [(full, info.frame)])

    # ---- the analysis ------------------------------------------------------

    def survey(self) -> None:
        """What needs no dataflow: the outer locals each procedure names,
        the addresses taken - in the procedures and in each module's own
        declarations and statements - and the calls through an address.
        The call graph may be completed with them (see
        CodeGenerator._complete_call_graph) before :meth:`static_locals`."""
        if self._surveyed:
            return
        self._surveyed = True
        for info in self.procs.values():
            _Walk(self, info, dataflow=False).run()
        for items, frame in self.modules:
            main = ProcInfo(full="", decl=None, chain=[], frame=frame)
            main.labels = _labels(_module_split(items)[1])
            _Walk(self, main, dataflow=False).run(items)

    def static_locals(self) -> dict[str, set[str]]:
        """procedure -> the locals of it that must not share ``??AUTO``."""
        self.survey()
        # A call through an address can come when the procedures `f' is
        # nested in are not active, and what they last left in their
        # locals is what `f' finds.  A procedure that `f' calls, and the
        # ones nested in it, are active whenever one of them names its
        # locals, and that procedure's own analysis sees to those.
        for f in self.proc_addr_taken:
            for owner, name in self.trans_refs(f):
                if f.startswith(owner + "$"):
                    self.make_static(owner, name, f"{f}, whose address is taken, names it")
        # Then definite assignment, with the reads the calls make.
        for info in self.procs.values():
            if any(loc.kind == AUTO for loc in info.locals.values()):
                _Walk(self, info, dataflow=True).run()
        return {full: self._close(full, info) for full, info in self.procs.items()
                if info.locals}

    def _close(self, full: str, info: ProcInfo) -> set[str]:
        """The parameters and AUTO locals of ``info`` that are static, once
        declaration order is kept: after a local whose address is taken, or
        which is subscripted outside its bounds, everything is static; a factored
        declaration is all static or not at all (6.2.4); and from the first
        array or structure subscripted by a variable on, the locals are all
        static or all in ??AUTO - so that an overrun reaches the local
        declared after it, as in DRI's layout."""
        locs = sorted(info.locals.values(), key=lambda loc: loc.order)
        static = {loc.name for loc in locs
                  if loc.kind == FIXED or (full, loc.name) in self.reasons}
        escaped = [loc for loc in locs if (full, loc.name) in self.escapes]
        runs = [loc for loc in locs if (full, loc.name) in self.runs]

        def take(names: Iterable[str], why: str) -> bool:
            added = False
            for n in names:
                if n not in static:
                    static.add(n)
                    self.reasons.setdefault((full, n), why)
                    added = True
            return added

        changed = True
        while changed:
            changed = False
            if escaped:
                first = escaped[0]
                changed |= take([loc.name for loc in locs if loc.order > first.order],
                                f"declared after {first.name}")
            groups = {info.locals[n].group for n in static if info.locals[n].kind == AUTO}
            changed |= take([loc.name for loc in locs
                             if loc.kind == AUTO and loc.group in groups],
                            "declared with a static local")
            if runs:
                tail = [loc for loc in locs if loc.order >= runs[0].order]
                if any(loc.name in static for loc in tail):
                    changed |= take([loc.name for loc in tail],
                                    f"{runs[0].name}, subscripted by a variable, "
                                    "runs on into a static local")
        return {n for n in static if info.locals[n].kind != FIXED}

    def make_static(self, owner: str, name: str, why: str) -> None:
        """Keep local ``name`` of ``owner`` out of ``??AUTO``."""
        self.reasons.setdefault((owner, name), why)

    def escape(self, owner: str, name: str, why: str) -> None:
        """Local ``name`` of ``owner`` can be reached through a pointer."""
        self.make_static(owner, name, why)
        self.escapes.add((owner, name))

    def run_on(self, owner: str, local: Local) -> list[tuple[str, frozenset]]:
        """What a read of ``local`` of ``owner`` through a variable subscript
        also reads: every AUTO local declared after it."""
        return [(loc.name, loc.keys) for loc in self.procs[owner].locals.values()
                if loc.kind == AUTO and loc.order > local.order]

    def trans_refs(self, proc: str) -> set[tuple[str, str]]:
        """The outer locals named by ``proc`` or by anything it can call."""
        got = self._trans_cache.get(proc)
        if got is not None:
            return got
        seen = {proc}
        work = [proc]
        refs: set[tuple[str, str]] = set()
        while work:
            p = work.pop()
            refs |= self.free_refs.get(p, set())
            for c in self.call_graph.get(p, ()):
                if c not in seen:
                    seen.add(c)
                    work.append(c)
        self._trans_cache[proc] = refs
        return refs

    def entered_labels(self, info: ProcInfo) -> set[str]:
        """Labels of ``info``'s body that control can reach from elsewhere:
        through their address, or by a GOTO in a procedure nested in it."""
        out = {lab for lab in info.labels if (info.full, lab) in self.label_addr_taken}
        prefix = info.full + "$"
        for p, labels in self.outward_gotos.items():
            if p.startswith(prefix):
                out |= labels & info.labels
        return out


def _layout(decls, params: list[str]) -> dict[str, Local]:
    """A procedure's parameters and its locals with storage of their own,
    in that order: the parameters as the PROCEDURE statement lists them,
    the locals as they are declared."""
    out: dict[str, Local] = {}
    for p in params:
        out[p] = Local(p, len(out), -1, None, None, PARAM)
    for group, d in enumerate(decls):
        if not isinstance(d, P.DeclItem):
            continue
        kind = _storage_kind(d)
        dt, dim = decl_item_type(d)
        if kind is None or dt == DataType.LABEL:
            continue
        for n in decl_item_names(d):
            if n not in params:
                out[n] = Local(n, len(out), group, dim, _members(d), kind)
    return out


def _storage_kind(d) -> Optional[str]:
    """AUTO for a declaration whose storage may be in ??AUTO, FIXED for one
    whose storage is static (INITIAL, PUBLIC), None for one with no storage
    among the procedure's variables (EXTERNAL, BASED, AT, DATA)."""
    a = decl_attrs(d)
    based, _ = decl_item_based(d)
    if a.is_external or based or a.at_location is not None or a.data_values is not None:
        return None
    if a.is_public or a.initial_values is not None:
        return FIXED
    return AUTO


def _members(d) -> Optional[dict]:
    members = decl_item_struct_members(d)
    if members is None:
        return None
    out: dict = {}
    for m in members:
        mdim = struct_member_dim(m)
        for n in struct_member_names(m):
            out[n] = mdim
    return out


def _frame(decls, owner: Optional[str], owned: Iterable[str] = ()) -> dict:
    """The names a list of declarations brings into scope."""
    owned = set(owned)
    frame: dict = {}
    for d in decls:
        if isinstance(d, P.ProcDecl):
            frame[proc_name(d)] = ("proc", owner, None)
        elif isinstance(d, P.DeclItem):
            based = d.based is not None
            for n in decl_item_names(d):
                if n in owned:
                    frame[n] = ("local", owner, None)
                elif based:
                    frame[n] = ("based", owner, d.based.base)
                else:
                    frame[n] = ("var", owner, None)
        elif isinstance(d, P.DeclItemBasedGroup):
            for b in d.based_decls or []:
                frame[ident_text(b.name)] = ("based", owner, b.base)
    return frame


def _module_split(items) -> tuple[list, list]:
    """(declarations, statements) of a module's items, whose declarations
    come unwrapped from their DECLAREs (see ast_view.module_shape)."""
    decls: list = []
    stmts: list = []
    for it in items:
        if isinstance(it, (P.ProcDecl, P.DeclItem, P.DeclItemBasedGroup, P.LiterallyDecl)):
            decls.append(it)
        elif isinstance(it, P.DeclareStmt):
            decls.extend(it.declarations)
        else:
            stmts.append(it)
    return decls, stmts


def _labels(stmts) -> set[str]:
    """Labels defined in a body, not counting nested procedures'."""
    out: set[str] = set()
    work = list(stmts)
    while work:
        s = work.pop()
        if isinstance(s, P.LabeledStmt):
            out.add(ident_text(s.label))
            work.append(s.stmt)
        elif isinstance(s, (P.IfStmt, P.IfStmtElse)):
            work.append(s.then_stmt)
            if isinstance(s, P.IfStmtElse):
                work.append(s.else_stmt)
        elif isinstance(s, _DO_KINDS):
            work.extend(x for x in s.items if not isinstance(x, (P.ProcDecl, P.DeclareStmt)))
    return out


class _Walk:
    """One analysis of one procedure's body.

    Without ``dataflow`` it records what needs none: the outer locals the
    body names, addresses taken, GOTOs out.  With it, it runs the
    definite-assignment analysis of the procedure's own locals and makes
    each one read where it may not have been assigned static.
    """

    def __init__(self, an: LocalStorage, info: ProcInfo, dataflow: bool) -> None:
        self.an = an
        self.info = info
        self.dataflow = dataflow
        self.frames: list = list(info.chain) + [(info.full, info.frame)]
        self.label_in: dict[str, frozenset] = {}
        self.goto_out: dict[str, frozenset] = {}
        self.entered: set[str] = set()

    # ---- driving -----------------------------------------------------------

    def run(self, items: Optional[list] = None) -> None:
        """Walk the body - or ``items``, the main program's - and with
        dataflow, until the labels' states settle."""
        if items is None:
            decls, stmts = block_items_split(proc_body_items(self.info.decl))
        else:
            decls, stmts = _module_split(items)
        if not self.dataflow:
            self._declarations(decls)
            self._walk_stmts(stmts, frozenset())
            return
        self.entered = self.an.entered_labels(self.info)
        for _ in range(_MAX_PASSES):
            self.goto_out = {}
            self._walk_stmts(stmts, frozenset())
            if self.goto_out == self.label_in:
                return
            self.label_in = dict(self.goto_out)
        self._give_up()

    def _give_up(self) -> None:
        for n, loc in self.info.locals.items():
            if loc.kind == AUTO:
                self.an.make_static(self.info.full, n, "the flow analysis did not settle")

    # ---- names -------------------------------------------------------------

    def _lookup(self, name: str, depth: Optional[int] = None):
        """(binding, frame index) of ``name``, searching out from ``depth``."""
        top = len(self.frames) - 1 if depth is None else depth
        for i in range(top, -1, -1):
            b = self.frames[i][1].get(name)
            if b is not None:
                return b, i
        return None, -1

    def _local(self, binding, name: str) -> Optional[Local]:
        if binding is None or binding[0] != "local":
            return None
        owner = self.an.procs.get(binding[1])
        return owner.locals.get(name) if owner else None

    def _callee(self, name: str, binding) -> Optional[str]:
        """The procedure ``name`` calls, if it names one."""
        if binding is not None and binding[0] != "proc":
            return None
        return self.an.resolve_proc(name, self.info.full)

    # ---- declarations ------------------------------------------------------

    def _declarations(self, decls) -> None:
        """Addresses taken in the INITIAL, DATA and AT of declarations."""
        for d in decls:
            if isinstance(d, (P.DeclItem, P.DeclItemBasedGroup)):
                a = decl_attrs(d)
                for v in (a.initial_values or []) + (a.data_values or []):
                    self._effects(v, _Effects())
                if a.at_location is not None:
                    self._effects(a.at_location, _Effects())

    # ---- statements --------------------------------------------------------

    def _walk_stmts(self, stmts, state: State) -> State:
        for s in stmts:
            state = self._stmt(s, state)
        return state

    def _in_block(self, items, fn):
        """Run ``fn(statements)`` with the block's declarations in scope."""
        decls, stmts = block_items_split(items)
        self.frames.append((None, _frame(decls, None)))
        try:
            if not self.dataflow:
                self._declarations(decls)
            return fn(stmts)
        finally:
            self.frames.pop()

    def _stmt(self, s, state: State) -> State:  # pylint: disable=too-many-return-statements
        if isinstance(s, P.AssignStmt):
            eff = _Effects()
            self._effects(s.value, eff)
            for t in s.targets:
                self._reference(t, eff, write=True)
            return self._apply(eff, state)
        if isinstance(s, P.CallStmt):
            eff = _Effects()
            self._effects(s.callee, eff)
            if self.info.full and self._through_address(s.callee):
                self.an.indirect_callers.add(self.info.full)
            return self._apply(eff, state)
        if isinstance(s, (P.IfStmt, P.IfStmtElse)):
            state = self._expr_state(s.condition, state)
            then = self._stmt(s.then_stmt, state)
            other = self._stmt(s.else_stmt, state) if isinstance(s, P.IfStmtElse) else state
            return _meet(then, other)
        if isinstance(s, P.DoBlock):
            return self._in_block(s.items, lambda stmts: self._walk_stmts(stmts, state))
        if isinstance(s, P.DoWhileBlock):
            return self._loop(state, lambda head: self._expr_state(s.condition, head), s.items)
        if isinstance(s, (P.DoIterBlock, P.DoIterByBlock)):
            return self._do_iter(s, state)
        if isinstance(s, P.DoCaseBlock):
            state = self._expr_state(s.selector, state)
            return self._in_block(s.items, lambda stmts: self._cases(stmts, state))
        if isinstance(s, P.LabeledStmt):
            return self._stmt(s.stmt, self._at_label(ident_text(s.label), state))
        if isinstance(s, P.GotoStmt):
            self._goto(ident_text(s.label), state)
            return None
        if isinstance(s, P.ReturnStmtValue):
            self._expr_state(s.value, state)
            return None
        if isinstance(s, P.ReturnStmt):
            return None
        # HALT waits for an interrupt and goes on; ENABLE, DISABLE and the
        # null statement do nothing to a variable.
        return state

    def _through_address(self, call) -> bool:
        """Whether a CALL statement calls the procedure at an address a
        variable holds (8.2.1), not a procedure it names or a built-in."""
        call = unwrap_paren(call)
        target = call.callee if isinstance(call, (P.Call, P.CallNoArgs)) else call
        target = unwrap_paren(target)
        if isinstance(target, P.Identifier):
            binding, _ = self._lookup(ident_text(target.name))
            return binding is not None and binding[0] != "proc"
        return True

    def _cases(self, stmts, state: State) -> State:
        out: State = None
        for c in stmts:
            out = _meet(out, self._stmt(c, state))
        return out if stmts else state

    def _at_label(self, label: str, state: State) -> State:
        if label in self.entered:
            return frozenset()
        if label in self.label_in:
            return _meet(state, self.label_in[label])
        return state

    def _goto(self, label: str, state: State) -> None:
        if label not in self.info.labels:
            self.an.outward_gotos.setdefault(self.info.full, set()).add(label)
        elif state is not None:
            prev = self.goto_out.get(label)
            self.goto_out[label] = state if prev is None else prev & state

    def _do_iter(self, s, state: State) -> State:
        by = isinstance(s, P.DoIterByBlock)
        index = P.Identifier(name=s.index, pos=s.pos)
        eff = _Effects()
        self._effects(s.start, eff)
        self._effects(s.bound, eff)
        if by:
            self._effects(s.step, eff)
        self._reference(index, eff, write=True)
        state = self._apply(eff, state)

        def head_fn(head: State) -> State:
            # The limit and step are evaluated again, and the step reads
            # the index.
            again = _Effects()
            self._effects(s.bound, again)
            if by:
                self._effects(s.step, again)
            self._effects(index, again)
            return self._apply(again, head)

        return self._loop(state, head_fn, s.items)

    def _loop(self, entry: State, head_fn, items) -> State:
        """A loop tested at the top: the state there is what holds on entry
        and at the end of every pass, and the loop leaves from the test."""
        head = entry
        for _ in range(_MAX_PASSES):
            tested = head_fn(head)
            end = self._in_block(items, lambda stmts, t=tested: self._walk_stmts(stmts, t))
            new_head = _meet(entry, end)
            if new_head == head:
                return tested
            head = new_head
        self._give_up()
        return frozenset()

    # ---- expressions -------------------------------------------------------

    def _expr_state(self, expr, state: State) -> State:
        eff = _Effects()
        self._effects(expr, eff)
        return self._apply(eff, state)

    def _apply(self, eff: _Effects, state: State) -> State:
        """Check what a statement reads against ``state``, then assign."""
        if not self.dataflow or state is None:
            return state
        mine = self.info.full
        reads = list(eff.reads)
        for callee in eff.calls:
            for owner, name in self.an.trans_refs(callee):
                loc = self.info.locals.get(name) if owner == mine else None
                if loc is None:
                    continue
                if loc.kind == AUTO:
                    reads.append((name, loc.keys))
                if (owner, name) in self.an.runs:
                    reads.extend(self.an.run_on(owner, loc))
        for name, keys in reads:
            if not keys <= state:
                self.an.make_static(mine, name, "it may be read before it is assigned")
        for keys in eff.writes:
            state = state | keys
        return state

    def _effects(self, expr, eff: _Effects) -> None:  # pylint: disable=too-many-branches
        """Record what evaluating ``expr`` reads, assigns and calls."""
        expr = unwrap_paren(expr)
        if isinstance(expr, (P.Identifier, P.Call, P.CallNoArgs)):
            node = expr if isinstance(expr, P.Identifier) else unwrap_paren(expr.callee)
            args = expr.args if isinstance(expr, P.Call) else []
            if isinstance(node, P.Identifier):
                name = ident_text(node.name)
                binding, _ = self._lookup(name)
                if binding is None and name in _SIZE_BUILTINS and args:
                    return
                callee = self._callee(name, binding)
                if callee is not None or binding is None:
                    # A call, or a builtin: the arguments are values.
                    if callee is not None:
                        eff.calls.append(callee)
                    for a in args:
                        self._effects(a, eff)
                    return
            if isinstance(expr, P.CallNoArgs):
                self._effects(node, eff)
            else:
                self._reference(expr, eff, write=False)
        elif isinstance(expr, P.MemberAccess):
            self._reference(expr, eff, write=False)
        elif isinstance(expr, P.BinaryOp):
            self._effects(expr.left, eff)
            self._effects(expr.right, eff)
        elif isinstance(expr, P.UnaryOp):
            self._effects(expr.operand, eff)
        elif isinstance(expr, P.EmbeddedAssign):
            self._effects(expr.value, eff)
            self._reference(expr.target, eff, write=True)
        elif isinstance(expr, P.LocationOf):
            self._location(expr.operand, eff)
        elif isinstance(expr, P.LocationOfList):
            for v in expr.values or []:
                self._effects(v, eff)

    def _fold(self, expr) -> Optional[tuple[int, DataType]]:  # pylint: disable=too-many-return-statements
        """(value, type) of a constant expression, as PL/M-80 computes it.

        A subscript the optimizer folds - ``a(1+1)``, ``a(3-5)``, ``a(-1)``
        - is a constant here at every level, so ``-O0`` decides what
        ``-O1`` and up decide.  LENGTH and LAST of a local array are its
        extent; a name the program declares is not a built-in.
        """
        e = unwrap_paren(expr)
        if isinstance(e, (P.NumberLiteral, P.StringLiteral)):
            got = typed_const(e)
            return None if got is None else (got[0] & 0xFFFF, got[1])
        if isinstance(e, P.UnaryOp):
            inner = self._fold(e.operand)
            return None if inner is None else fold_unary(unop_kind(e), *inner)
        if isinstance(e, P.BinaryOp):
            left, right = self._fold(e.left), self._fold(e.right)
            if left is None or right is None:
                return None
            return fold_binary(binop_kind(e), left[0], left[1], right[0], right[1])
        if not (isinstance(e, P.Call) and isinstance(unwrap_paren(e.callee), P.Identifier)):
            return None
        name = ident_text(unwrap_paren(e.callee).name)
        if self._lookup(name)[0] is not None:
            return None
        if name.upper() in ("LENGTH", "LAST") and len(e.args) == 1:
            arg = unwrap_paren(e.args[0])
            if not isinstance(arg, P.Identifier):
                return None
            binding, _ = self._lookup(ident_text(arg.name))
            local = self._local(binding, ident_text(arg.name))
            if local is None or local.dim is None or local.dim <= 0:
                return None
            n = local.dim if name.upper() == "LENGTH" else local.dim - 1
            return n, literal_type(n)
        args = [self._fold(a) for a in e.args]
        if not args or any(a is None for a in args):
            return None
        return fold_builtin(name, args)  # type: ignore[arg-type]

    def _fold_path(self, path: list) -> list:
        """``path`` with each subscript's args replaced by their values."""
        out = []
        for part in path:
            if part[0] == "idx":
                values = []
                for a in part[1]:
                    got = self._fold(a)
                    values.append(None if got is None else got[0])
                out.append(("idx", values))
            else:
                out.append(part)
        return out

    def _subscripts(self, path: list, eff: _Effects) -> None:
        for part in path:
            if part[0] == "idx":
                for a in part[1]:
                    self._effects(a, eff)

    def _reference(self, expr, eff: _Effects, write: bool) -> None:
        """A read of - or with ``write`` a store to - a variable reference.

        Its subscripts are read either way.
        """
        des = _designator(expr)
        if des is None:
            # An indirect callee or some other expression: all values.
            if isinstance(expr, P.Call):
                self._effects(expr.callee, eff)
                for a in expr.args:
                    self._effects(a, eff)
            elif isinstance(expr, P.MemberAccess):
                self._effects(expr.base, eff)
            return
        root, path = des
        self._subscripts(path, eff)
        binding, depth = self._lookup(root)
        if binding is not None and binding[0] == "based":
            self._read_base(binding, depth, eff)
            return
        local = self._local(binding, root)
        if local is None:
            return
        owner = binding[1]
        folded = self._fold_path(path)
        keys, exact, reach = _select(local, folded)
        runs_on = not reach and _varies(folded)
        if reach:
            self.an.escape(owner, root, "it is subscripted outside its bounds")
        elif runs_on:
            self.an.runs.add((owner, root))
        if owner != self.info.full:
            self.an.free_refs.setdefault(self.info.full, set()).add((owner, root))
        elif local.kind != AUTO:
            return
        elif write:
            if exact:
                eff.writes.append(keys)
        else:
            eff.reads.append((root, keys))
            if runs_on:
                # An overrun reads the locals declared after it.
                eff.reads.extend(self.an.run_on(owner, local))

    def _read_base(self, binding, depth: int, eff: _Effects) -> None:
        """A use of a BASED variable reads its base, found where it is declared."""
        des = _designator_of_base(binding[2])
        if des is None:
            return
        base, members = des
        base_binding, base_depth = self._lookup(base, depth)
        if base_binding is not None and base_binding[0] == "based":
            self._read_base(base_binding, base_depth, eff)
            return
        local = self._local(base_binding, base)
        if local is None:
            return
        owner = base_binding[1]
        if owner != self.info.full:
            self.an.free_refs.setdefault(self.info.full, set()).add((owner, base))
        elif local.kind == AUTO:
            keys, _, _ = _select(local, [("mem", m) for m in members])
            eff.reads.append((base, keys))

    def _location(self, operand, eff: _Effects) -> None:
        """``.operand``: the address of a variable, a procedure or a label."""
        des = _designator(operand)
        if des is None:
            self._effects(operand, eff)
            return
        root, path = des
        self._subscripts(path, eff)
        binding, depth = self._lookup(root)
        if binding is not None and binding[0] == "based":
            # `.v' of a BASED v is computed from its base.
            self._read_base(binding, depth, eff)
            return
        if self._local(binding, root) is not None:
            self.an.escape(binding[1], root, "its address is taken")
            if binding[1] != self.info.full:
                self.an.free_refs.setdefault(self.info.full, set()).add((binding[1], root))
            return
        callee = self._callee(root, binding)
        if callee is not None and not path:
            self.an.proc_addr_taken.add(callee)
        elif binding is None and not path:
            # Perhaps a label: a jump through its address comes from anywhere.
            self.an.label_addr_taken.add((self.info.full, root))


def _designator_of_base(base) -> Optional[tuple[str, list[str]]]:
    """(name, members) of a BASED declaration's base, ``p`` or ``s.m``."""
    parts: list[str] = []
    node = base
    while isinstance(node, P.DottedMember):
        parts.append(ident_text(node.member))
        node = node.base
    if not isinstance(node, P.DottedIdent):
        return None
    return ident_text(node.name), list(reversed(parts))
