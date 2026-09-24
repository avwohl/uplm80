"""
AST Optimizer for PL/M-80.

Performs high-level optimizations on the typed uplox-generated AST
(see :mod:`uplm80._plm_parser`) before code generation:

- Constant folding and propagation
- Strength reduction
- Dead code elimination
- Common subexpression elimination (CSE)
- Loop-invariant code motion
- Algebraic simplifications

The optimizer consumes and produces :class:`P.Module` nodes; replacement
expression / statement nodes are constructed via the
:func:`ast_view.make_*` synthetic-token builders so codegen (which only
reads ``.text`` off operator / literal tokens) keeps round-tripping
through the folded result.

Every rewrite has to leave an expression with the value AND the type the
unoptimized program gives it, by the rules in :mod:`plm_types`: a BYTE
``x + 0FFH`` wraps where an ADDRESS one carries, so replacing an ADDRESS
subexpression with a BYTE one of the same value changes the program. The
optimizer therefore types what it rewrites (:meth:`ASTOptimizer._type_of`,
from the declarations in scope), folds constants with their types, and writes
an ADDRESS constant below 256 as ``DOUBLE(n)``.
"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

from . import _plm_parser as P
from .runtime import plm_div, plm_mod
from .ast_view import (
    BinaryOpKind,
    UnaryOpKind,
    binop_kind,
    block_items_split,
    DataType,
    decl_attrs,
    decl_item_names,
    decl_item_struct_members,
    decl_item_type,
    ident_text,
    iter_block_proc_decls,
    make_binary,
    make_identifier,
    make_number_literal,
    make_unary,
    number_value,
    proc_attrs,
    proc_body_items,
    proc_local_decls_stmts,
    proc_name,
    proc_param_names,
    proc_return_type,
    unop_kind,
    unwrap_paren,
)
from .plm_types import (
    ADDRESS,
    BYTE,
    BYTE_BUILTINS,
    PATTERN_TYPED_BUILTINS,
    RELATIONS,
    binary_type,
    convert,
    eval_typed,
    fold_binary,
    fold_builtin,
    fold_unary,
    is_derived,
    literal_type,
    make_typed_const,
    typed_const,
    untyped_root,
)


# Pure built-in functions whose calls are safe to CSE / treat as
# loop-invariant when all their arguments are.
_PURE_BUILTINS = {
    "LOW", "HIGH", "DOUBLE", "SHL", "SHR", "ROL", "ROR",
    "LENGTH", "LAST", "SIZE",
}

# Built-ins that DO something: reading a port, writing one, moving memory,
# or consuming time. Dropping a call to one of these changes the program
# even when its value is unused.
_IMPURE_BUILTINS = {
    "INPUT", "OUTPUT", "MOVE", "TIME",
    # flag reads are not side-effecting but they are not values of the
    # surrounding expression either; treat them as unsafe to discard.
    "CARRY", "ZERO", "SIGN", "PARITY", "STACKPTR", "MEMORY",
    # SCL/SCR rotate through carry, so they depend on and set it.
    "SCL", "SCR",
}


@dataclass(eq=False)
class _Decl:
    """What a name denotes in one scope.

    ``plain`` is an ordinary scalar variable: not BASED, not AT, not an
    array or structure. Only a plain variable's value is tracked, because a
    store through anything else may land on it.
    """

    kind: str                          # "var", "proc" or "other"
    dtype: "DataType | None" = None    # scalar / element / return type
    plain: bool = False
    array: bool = False


def _scope_of(items) -> dict:
    """The names a block's (or procedure body's) declarations introduce."""
    scope: dict[str, _Decl] = {}

    def declare(it) -> None:
        if isinstance(it, P.ProcDecl):
            scope[proc_name(it)] = _Decl("proc", proc_return_type(it))
        elif isinstance(it, P.DeclareStmt):
            for d in it.declarations:
                declare(d)
        elif isinstance(it, P.DeclItem):
            dt, dim = decl_item_type(it)
            if dt not in (DataType.BYTE, DataType.ADDRESS) or decl_item_struct_members(it):
                dt = None
            plain = (dt is not None and dim is None and it.based is None
                     and decl_attrs(it).at_location is None)
            for name in decl_item_names(it):
                scope[name] = _Decl("var", dt, plain, dim is not None)
        elif isinstance(it, P.DeclItemBasedGroup):
            dt, dim = decl_item_type(it)
            if dt not in (DataType.BYTE, DataType.ADDRESS):
                dt = None
            for bd in it.based_decls:
                scope[ident_text(bd.name)] = _Decl("var", dt, False, dim is not None)
        elif isinstance(it, P.LiterallyDecl):
            scope[ident_text(it.name)] = _Decl("other")

    for it in items:
        declare(it)
    # A procedure declared in a nested DO block belongs to the enclosing
    # procedure's scope (see iter_block_proc_decls).
    for proc in iter_block_proc_decls(items):
        scope.setdefault(proc_name(proc), _Decl("proc", proc_return_type(proc)))
    return scope


def _is_number(expr) -> bool:
    return isinstance(unwrap_paren(expr), P.NumberLiteral)


def _num_value(expr) -> int:
    """Get the int value of a NumberLiteral (after stripping parens)."""
    return number_value(unwrap_paren(expr))


def _is_ident(expr) -> bool:
    return isinstance(unwrap_paren(expr), P.Identifier)


def _ident_name(expr) -> str:
    return ident_text(unwrap_paren(expr).name)


def _get_expr_vars(expr) -> set[str]:
    """Get all variable names referenced in an expression."""
    result: set[str] = set()
    if expr is None:
        return result
    e = unwrap_paren(expr)
    if isinstance(e, P.Identifier):
        result.add(ident_text(e.name))
    elif isinstance(e, P.BinaryOp):
        result.update(_get_expr_vars(e.left))
        result.update(_get_expr_vars(e.right))
    elif isinstance(e, P.UnaryOp):
        result.update(_get_expr_vars(e.operand))
    elif isinstance(e, P.MemberAccess):
        result.update(_get_expr_vars(e.base))
    elif isinstance(e, P.Call):
        result.update(_get_expr_vars(e.callee))
        for arg in e.args:
            result.update(_get_expr_vars(arg))
    elif isinstance(e, P.CallNoArgs):
        result.update(_get_expr_vars(e.callee))
    elif isinstance(e, P.LocationOf):
        result.update(_get_expr_vars(e.operand))
    elif isinstance(e, P.LocationOfList):
        for v in e.values:
            result.update(_get_expr_vars(v))
    elif isinstance(e, P.EmbeddedAssign):
        result.update(_get_expr_vars(e.target))
        result.update(_get_expr_vars(e.value))
    return result


def _all_names(node) -> set[str]:
    """Every identifier (and DO index) named anywhere in ``node``."""
    names: set[str] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, P.Identifier):
            names.add(ident_text(n.name))
            continue
        if isinstance(n, (P.DoIterBlock, P.DoIterByBlock)):
            names.add(ident_text(n.index))
        if isinstance(n, (list, tuple)):
            stack.extend(n)
            continue
        fields = getattr(n, "__dataclass_fields__", None)
        if not fields:
            continue
        for f in fields:
            if f == "pos":
                continue
            stack.append(getattr(n, f, None))
    return names


def _expr_key(expr) -> str | None:
    """Generate a hashable key for an expression for CSE.

    Returns None for expressions that shouldn't be cached (with side effects).
    """
    if expr is None:
        return None
    e = unwrap_paren(expr)
    if isinstance(e, P.NumberLiteral):
        return f"NUM:{number_value(e)}"
    if isinstance(e, P.StringLiteral):
        return f"STR:{e.value.text}"
    if isinstance(e, P.Identifier):
        return f"ID:{ident_text(e.name)}"
    if isinstance(e, P.BinaryOp):
        left_key = _expr_key(e.left)
        right_key = _expr_key(e.right)
        if left_key is None or right_key is None:
            return None
        return f"BIN:{binop_kind(e).name}:{left_key}:{right_key}"
    if isinstance(e, P.UnaryOp):
        operand_key = _expr_key(e.operand)
        if operand_key is None:
            return None
        return f"UN:{unop_kind(e).name}:{operand_key}"
    if isinstance(e, P.MemberAccess):
        base_key = _expr_key(e.base)
        if base_key is None:
            return None
        return f"MEM:{base_key}:{ident_text(e.member)}"
    if isinstance(e, P.Call):
        # Only pure built-in functions can be CSE'd. PL/M's grammar
        # can't distinguish ``arr(idx)`` from ``func(arg)`` syntactically,
        # so a single-arg Call on an unknown identifier could be a
        # subscript on a non-modified array — we still treat it as
        # opaque (returning None) to be conservative without symbol info.
        if isinstance(e.callee, P.Identifier):
            name = ident_text(e.callee.name)
            if name in _PURE_BUILTINS:
                arg_keys = [_expr_key(a) for a in e.args]
                if all(k is not None for k in arg_keys):
                    return f"CALL:{name}:{':'.join(arg_keys)}"
        return None
    if isinstance(e, P.CallNoArgs):
        return None
    if isinstance(e, P.LocationOf):
        operand_key = _expr_key(e.operand)
        if operand_key is None:
            return None
        return f"LOC:{operand_key}"
    if isinstance(e, P.LocationOfString):
        return f"LOCSTR:{e.value.text}"
    if isinstance(e, P.LocationOfList):
        keys = [_expr_key(v) for v in e.values]
        if any(k is None for k in keys):
            return None
        return f"LOCLST:{':'.join(keys)}"  # type: ignore[arg-type]
    return None


class OptimizeFor(Enum):
    """Optimization target preference."""
    SPEED = "speed"  # Prefer faster code (may increase size)
    SIZE = "size"    # Prefer smaller code (may be slower)
    BALANCED = "balanced"  # Balance between size and speed


@dataclass
class OptimizationStats:
    """Statistics about optimizations performed."""

    constants_folded: int = 0
    strength_reductions: int = 0
    dead_code_eliminated: int = 0
    algebraic_simplifications: int = 0
    cse_eliminations: int = 0
    loop_invariants_moved: int = 0
    boolean_simplifications: int = 0
    copies_propagated: int = 0
    dead_stores_eliminated: int = 0
    loops_unrolled: int = 0
    procedures_inlined: int = 0
    tail_calls_optimized: int = 0


class ASTOptimizer:
    """
    AST optimizer that performs high-level transformations.

    Optimization levels:
    - 0: No optimization
    - 1: Basic (constant folding, simple algebraic)
    - 2: Standard (+ strength reduction, dead code)
    - 3: Aggressive (+ CSE, loop optimizations)
    """

    def __init__(
        self,
        opt_level: int = 2,
        optimize_for: OptimizeFor = OptimizeFor.BALANCED,
    ) -> None:
        self.opt_level = opt_level
        self.optimize_for = optimize_for
        self.stats = OptimizationStats()
        # Known constant values for propagation: name -> (value, type,
        # derived). The value is already converted to the variable's type.
        self.constants: dict[str, tuple[int, DataType, bool]] = {}
        # The declarations in scope, innermost last.
        self.scopes: list[dict[str, _Decl]] = []
        # For each inlinable procedure, the scopes its body's names resolve
        # in; a call site may inline it only where they resolve the same.
        self.inline_scopes: dict[str, list[dict[str, _Decl]]] = {}
        # Track which variables are modified in current scope
        self.modified_vars: set[str] = set()
        # CSE: map from expression key to (temp_var_name, expr) for level 3
        self.cse_cache: dict[str, tuple[str, object]] = {}
        self.cse_counter: int = 0
        # Track variables used in expressions for invalidation
        self.expr_vars: dict[str, set[str]] = {}  # expr_key -> set of var names
        # Copy propagation: x = y means copies[x] = y
        self.copies: dict[str, str] = {}
        # Procedure inlining: track small procedures that can be inlined
        self.inlinable_procs: dict[str, P.ProcDecl] = {}
        # Every procedure name in the module. A bare identifier that
        # names one is a PL/M parameterless call, not a variable read.
        self.proc_names: set[str] = set()
        # Names a store through a pointer may reach: their address is taken
        # (`.x'), they are AT something, or another module sees them.
        self.pointer_reachable: set[str] = set()
        # True while optimizing a region that reads CARRY / ZERO /
        # SIGN / PARITY, or uses PLUS / MINUS, where an arithmetic
        # operation's flag side effect is observable and must not be
        # folded away.
        self.flag_sensitive: bool = False
        # True while optimizing a DATA / INITIAL / AT value: a restricted
        # expression, folded as plain 16-bit numbers.
        self.restricted: bool = False

    def _parse_plm_number(self, s: str) -> int | None:
        """Parse a PL/M-style numeric literal (handles $ separators and B/H/O/Q/D suffixes)."""
        try:
            s = s.upper().replace("$", "").strip()
            if not s:
                return None
            if s.endswith("H"):
                return int(s[:-1], 16)
            elif s.endswith("B"):
                return int(s[:-1], 2)
            elif s.endswith("O") or s.endswith("Q"):
                return int(s[:-1], 8)
            elif s.endswith("D"):
                return int(s[:-1], 10)
            else:
                return int(s, 0)
        except (ValueError, TypeError):
            return None

    # ---- module-level driver ----------------------------------------------

    def _collect_proc_names(self, items) -> None:
        """Record every procedure name reachable from ``items``."""
        for it in items:
            if isinstance(it, P.ProcDecl):
                self.proc_names.add(proc_name(it))
                self._collect_proc_names(proc_body_items(it))
            elif isinstance(it, P.DeclareStmt):
                self._collect_proc_names(it.declarations)
            elif isinstance(it, (P.DoBlock, P.DoWhileBlock, P.DoIterBlock,
                                 P.DoIterByBlock, P.DoCaseBlock)):
                self._collect_proc_names(it.items)
            elif isinstance(it, (P.IfStmt, P.IfStmtElse)):
                self._collect_proc_names([it.then_stmt])
                if isinstance(it, P.IfStmtElse):
                    self._collect_proc_names([it.else_stmt])
            elif isinstance(it, P.LabeledStmt):
                self._collect_proc_names([it.stmt])

    _FLAG_BUILTINS = frozenset({"CARRY", "ZERO", "SIGN", "PARITY"})

    def _reads_a_flag(self, node) -> bool:
        """Whether anything in ``node`` reads a condition flag.

        PL/M-80's CARRY / ZERO / SIGN / PARITY read the flags left by the
        preceding operation, and PLUS / MINUS add in its carry, so in a
        region that uses them an arithmetic expression is not a pure value:
        folding `A + B` to a constant removes the `add` whose carry the next
        operation reads.
        """
        stack = [node]
        while stack:
            n = stack.pop()
            if isinstance(n, P.Identifier):
                if ident_text(n.name).upper() in self._FLAG_BUILTINS:
                    return True
                continue
            # PLUS and MINUS add in the carry the operation before them left.
            if isinstance(n, P.BinaryOp) and binop_kind(n) in (
                    BinaryOpKind.PLUS, BinaryOpKind.MINUS):
                return True
            # A nested procedure is a region of its own (_optimize_proc_decl).
            if isinstance(n, P.ProcDecl):
                continue
            if isinstance(n, (list, tuple)):
                stack.extend(n)
                continue
            fields = getattr(n, "__dataclass_fields__", None)
            if not fields:
                continue
            for f in fields:
                if f == "pos":
                    continue
                stack.append(getattr(n, f, None))
        return False

    def _contains_label(self, node) -> bool:
        """Whether ``node`` holds a labelled statement anywhere inside it.

        Dropping a dead IF arm drops its labels with it, but a GOTO from
        elsewhere in the procedure still names them, and codegen then emits
        a jump to a symbol nothing defines. PL/M-80 labels are procedure
        scoped, so the arm cannot be discarded when it declares one.
        """
        stack = [node]
        while stack:
            n = stack.pop()
            if isinstance(n, P.LabeledStmt):
                return True
            if isinstance(n, (list, tuple)):
                stack.extend(n)
                continue
            fields = getattr(n, "__dataclass_fields__", None)
            if not fields:
                continue
            for f in fields:
                if f == "pos":
                    continue
                stack.append(getattr(n, f, None))
        return False

    # ---- scopes and types -------------------------------------------------

    def _push_scope(self, items) -> None:
        """Enter a block declaring ``items``.

        What was known about an outer variable a declaration here hides says
        nothing about the new one.
        """
        scope = _scope_of(items)
        self.scopes.append(scope)
        self._forget(scope)

    def _pop_scope(self) -> None:
        """Leave a block: what was learned about its own names goes with it."""
        self._forget(self.scopes.pop())

    def _forget(self, names) -> None:
        for name in names:
            self.constants.pop(name, None)
            self._invalidate_cse_for_var(name)
            self._invalidate_copies_for_var(name)

    def _lookup(self, name: str) -> "_Decl | None":
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def _is_builtin(self, name: str) -> bool:
        """``name`` (as written) is the built-in procedure, not something the
        program declared under the same name."""
        return self._lookup(name) is None

    def _plain_var_type(self, name: str) -> "DataType | None":
        """The type of ``name`` when it is a plain scalar variable, else None."""
        d = self._lookup(name)
        if d is not None and d.kind == "var" and d.plain:
            return d.dtype
        return None

    def _type_of(self, expr) -> "DataType | None":
        """The type code generation gives ``expr``; None when not known.

        Mirrors CodeGenerator._get_expr_type, from the declarations in scope.
        """
        e = unwrap_paren(expr)
        tc = typed_const(e)
        if tc is not None:
            return tc[1]
        if isinstance(e, P.StringLiteral):
            return ADDRESS
        if isinstance(e, P.Identifier):
            name = ident_text(e.name)
            d = self._lookup(name)
            if d is None:
                return ADDRESS if name.upper() == "STACKPTR" else None
            if d.kind == "proc":
                return d.dtype
            if d.kind == "var" and not d.array:
                return d.dtype
            return None
        if isinstance(e, P.BinaryOp):
            kind = binop_kind(e)
            return binary_type(kind, self._type_of(e.left), self._type_of(e.right))
        if isinstance(e, P.UnaryOp):
            return self._type_of(e.operand)
        if isinstance(e, (P.LocationOf, P.LocationOfList, P.LocationOfString)):
            return ADDRESS
        if isinstance(e, P.EmbeddedAssign):
            # "The value of the embedded assignment is the same as that of
            # its right half" (4.6.3).
            return self._type_of(e.value)
        if isinstance(e, (P.Call, P.CallNoArgs)):
            callee = unwrap_paren(e.callee)
            if not isinstance(callee, P.Identifier):
                return None
            raw = ident_text(callee.name)
            d = self._lookup(raw)
            if d is not None:
                return d.dtype if d.kind in ("proc", "var") else None
            name = raw.upper()
            if name in BYTE_BUILTINS:
                return BYTE
            if name in ("DOUBLE", "SIZE", "STACKPTR", "TIME", "SHL", "SHR"):
                return ADDRESS
            if name in PATTERN_TYPED_BUILTINS and isinstance(e, P.Call) and e.args:
                return self._type_of(e.args[0])
            return None
        return None

    def _is_plain_store(self, target) -> bool:
        """A store to ``target`` changes one plain variable and nothing else."""
        t = unwrap_paren(target)
        return isinstance(t, P.Identifier) and self._plain_var_type(ident_text(t.name)) is not None

    def _stores_indirectly(self, node) -> bool:
        """Whether ``node`` stores anywhere but into plain variables.

        A store through a BASED variable, an array element, a structure
        member or an AT-located variable may land on any variable, so no
        fact about any of them survives it.
        """
        stack = [node]
        while stack:
            n = stack.pop()
            if isinstance(n, P.AssignStmt):
                if not all(self._is_plain_store(t) for t in n.targets):
                    return True
            elif isinstance(n, P.EmbeddedAssign):
                if not self._is_plain_store(n.target):
                    return True
            if isinstance(n, (list, tuple)):
                stack.extend(n)
                continue
            fields = getattr(n, "__dataclass_fields__", None)
            if not fields:
                continue
            for f in fields:
                if f == "pos":
                    continue
                stack.append(getattr(n, f, None))
        return False

    def _as_address(self, expr, pos):
        """``expr`` widened, where it might not be, to the ADDRESS it replaces.

        PL/M-80 has no BYTE divide: DRI's compiler zero-extends BYTE operands
        and calls its one 16-bit routine, so a quotient or remainder is an
        ADDRESS and the arithmetic around it is 16-bit. A rewrite of one must
        not narrow it. DOUBLE of an ADDRESS generates no code.
        """
        if self._type_of(expr) is ADDRESS:
            return expr
        tc = typed_const(expr)
        if tc is not None:
            return make_typed_const(tc[0], ADDRESS, pos, derived=is_derived(expr))
        return P.Call(callee=make_identifier("DOUBLE", pos=pos), args=[expr], pos=pos)

    def _optimize_value(self, expr, keep_type: bool = False):
        """Optimize an expression a statement evaluates for its value.

        A call can change any variable, and PL/M-80 leaves the order in
        which an expression's operands are evaluated to the compiler, so no
        fact about a variable is used in an expression that makes a call,
        and none survives it. An embedded assignment's target is forgotten
        for the same reason.

        Unless ``keep_type``, the result is converted by whatever uses it
        (stored, passed, tested...), so a constant's own type does not
        matter and the plain literal is returned.
        """
        if expr is None:
            return None
        effectful = self._contains_call(expr)
        self._forget_embedded_targets(expr)
        if effectful:
            self._reset_flow_state()
        out = self._optimize_expr(expr)
        if effectful or self._stores_indirectly(expr):
            self._reset_flow_state()
        return out if keep_type else untyped_root(out)

    def _forget_embedded_targets(self, node) -> None:
        stack = [node]
        while stack:
            n = stack.pop()
            if isinstance(n, P.EmbeddedAssign):
                t = unwrap_paren(n.target)
                if isinstance(t, P.Identifier):
                    self._forget([ident_text(t.name)])
            if isinstance(n, (list, tuple)):
                stack.extend(n)
                continue
            fields = getattr(n, "__dataclass_fields__", None)
            if not fields:
                continue
            for f in fields:
                if f == "pos":
                    continue
                stack.append(getattr(n, f, None))

    def _contains_call(self, node) -> bool:
        """Whether anything in ``node`` can call out.

        A PL/M call may assign any global, so no fact about any variable
        survives it. A bare identifier naming a procedure is a call too.
        """
        stack = [node]
        while stack:
            n = stack.pop()
            if isinstance(n, P.CallStmt):
                return True
            if isinstance(n, P.Identifier):
                if ident_text(n.name) in self.proc_names:
                    return True
                continue
            if isinstance(n, P.Call):
                callee = unwrap_paren(n.callee)
                if isinstance(callee, P.Identifier):
                    raw = ident_text(callee.name)
                    if raw in self.proc_names or raw.upper() in _IMPURE_BUILTINS:
                        return True
                stack.extend(n.args)
                stack.append(n.callee)
                continue
            if isinstance(n, (list, tuple)):
                stack.extend(n)
                continue
            fields = getattr(n, "__dataclass_fields__", None)
            if not fields:
                continue
            for f in fields:
                if f == "pos":
                    continue
                stack.append(getattr(n, f, None))
        return False

    def _snapshot_flow_state(self):
        """Copy the flow-sensitive state so a branch can be optimized from it."""
        return (
            dict(self.constants),
            dict(self.copies),
            dict(self.cse_cache),
            {k: set(v) for k, v in self.expr_vars.items()},
            set(self.modified_vars),
        )

    def _restore_flow_state(self, snap) -> None:
        constants, copies, cse, expr_vars, modified = snap
        self.constants = dict(constants)
        self.copies = dict(copies)
        self.cse_cache = dict(cse)
        self.expr_vars = {k: set(v) for k, v in expr_vars.items()}
        self.modified_vars = set(modified)

    def _invalidate_modified(self, items) -> None:
        """Drop every fact the given statements can falsify.

        Used at a loop, where the body's assignments are visible on the
        back edge, and after an IF, where only one arm runs but either
        may have assigned.

        Constant and copy propagation are flow-insensitive: a fact
        established before the loop is still in scope while the condition
        and the body are optimized, but control re-enters the top with the
        body's assignments applied. Without this,

            n = 0;
            do while n < 3; call pc('0' + n); n = n + 1; end;

        folded the condition to always-true and pinned ``n`` at 0 through
        the whole body, so the loop printed `0` forever.

        A call can assign any global, and a store through a BASED variable,
        a subscript or a member can land on any variable, so a body with
        either clears everything; otherwise only what the body assigns is
        dropped.
        """
        if self._contains_call(items) or self._stores_indirectly(items):
            self._reset_flow_state()
            return
        _, stmts = block_items_split(items)
        for name in self._get_modified_vars_in_stmts(stmts):
            self.constants.pop(name, None)
            self.copies.pop(name, None)
            self._invalidate_cse_for_var(name)
            self._invalidate_copies_for_var(name)
            self.modified_vars.add(name)

    def _reset_flow_state(self) -> None:
        """Drop everything learned about values along one flow of control.

        `constants`, `copies`, `cse_cache`, `expr_vars` and `modified_vars`
        all describe one straight-line region. Carrying them across a
        procedure boundary propagates one procedure's constant into
        another's body, and carrying them across a pass lets a fact
        derived from already-rewritten code feed back in.
        """
        self.constants.clear()
        self.copies.clear()
        self.cse_cache.clear()
        self.expr_vars.clear()
        self.modified_vars.clear()

    def _is_side_effect_free(self, expr) -> bool:
        """Whether ``expr`` can be dropped without changing behaviour.

        PL/M-80 evaluates both operands of every operator, so an
        algebraic identity that discards one (``x AND 0``, ``x XOR x``)
        is only valid when that operand does nothing observable. A bare
        identifier naming a procedure is a parameterless call, so it is
        not free.
        """
        expr = unwrap_paren(expr)
        if isinstance(expr, (P.NumberLiteral, P.StringLiteral)):
            return True
        if isinstance(expr, P.Identifier):
            return ident_text(expr.name) not in self.proc_names
        if isinstance(expr, P.UnaryOp):
            return self._is_side_effect_free(expr.operand)
        if isinstance(expr, P.BinaryOp):
            return (self._is_side_effect_free(expr.left)
                    and self._is_side_effect_free(expr.right))
        if isinstance(expr, P.Call):
            callee = unwrap_paren(expr.callee)
            if isinstance(callee, P.Identifier):
                raw = ident_text(callee.name)
                name = raw.upper()
                if name in _IMPURE_BUILTINS or raw in self.proc_names:
                    return False
                # What is left is a pure built-in or an array subscript;
                # both are free when their arguments are.
                return all(self._is_side_effect_free(a) for a in expr.args)
            return False
        return False

    def optimize(self, module: P.Module) -> P.Module:
        """Optimize an entire typed :class:`P.Module`."""
        if self.opt_level == 0:
            return module

        self.proc_names.clear()
        self._collect_proc_names(module.items)
        self.pointer_reachable = self._pointer_reachable_names(module.items)

        # Multiple passes for iterative improvement
        changed = True
        passes = 0
        max_passes = 5

        while changed and passes < max_passes:
            changed = False
            passes += 1
            self._reset_flow_state()
            self.flag_sensitive = any(self._reads_a_flag(x) for x in module.items)
            self.scopes = [_scope_of(module.items)]
            self.inlinable_procs.clear()
            self.inline_scopes.clear()

            new_items: list = []
            for item in module.items:
                opt_item = self._optimize_module_item(item)
                if opt_item is not None:
                    new_items.append(opt_item)
                    if opt_item is not item:
                        changed = True

            module = P.Module(items=new_items, pos=module.pos)
            self.scopes = []

        return module

    def _optimize_module_item(self, item):
        """Optimize one top-level :class:`P.Module` item.

        Top-level items are: ``ProcDecl``, ``DeclareStmt``,
        ``LabeledStmt`` (typically wrapping the module-level DO block),
        ``AddressLiteral`` (origin), or any other statement form. We
        dispatch into the existing decl / stmt handlers.
        """
        if isinstance(item, P.ProcDecl):
            return self._optimize_proc_decl(item)
        if isinstance(item, P.LiterallyDecl):
            return self._optimize_literally_decl(item)
        if isinstance(item, P.DeclareStmt):
            return self._optimize_declare_stmt(item)
        if isinstance(item, P.AddressLiteral):
            return item
        # Anything else is a statement form (LabeledStmt wrapping a
        # DoBlock for the standard module shape, or a bare statement).
        return self._optimize_stmt(item)

    # ---- declaration handlers ---------------------------------------------

    def _optimize_proc_decl(self, decl: P.ProcDecl) -> P.ProcDecl:
        """Optimize a typed procedure declaration.

        Walks the body's mixed item list and recursively optimizes both
        the inner declarations and statements, then rebuilds a
        :class:`P.ProcBody` with the optimized items in source order
        (declarations first, then statements — matches the legacy split).
        """
        attrs = proc_attrs(decl)
        local_decls, body_stmts = proc_local_decls_stmts(decl)

        # A procedure body is its own flow region, and its own scope.
        self._reset_flow_state()
        outer_flag_sensitive = self.flag_sensitive
        self.flag_sensitive = any(self._reads_a_flag(x) for x in body_stmts)
        enclosing_scopes = list(self.scopes)
        self._push_scope(decl.body.items)

        new_decls: list = []
        for d in local_decls:
            opt_d = self._optimize_decl_in_body(d)
            if opt_d is not None:
                new_decls.append(opt_d)

        new_stmts: list = []
        for s in body_stmts:
            opt_s = self._optimize_stmt(s)
            if opt_s is not None:
                new_stmts.append(opt_s)

        # Eliminate unreachable code after RETURN/GOTO/HALT
        new_stmts = self._eliminate_unreachable(new_stmts)
        # Eliminate dead stores
        new_stmts = self._eliminate_dead_stores(new_stmts)

        # Nothing learned inside this body is valid outside it.
        self._pop_scope()
        self._reset_flow_state()
        self.flag_sensitive = outer_flag_sensitive

        # Rebuild body items: keep nested ProcDecls and LiterallyDecls as
        # standalone items; group the rest into a DeclareStmt as the
        # parser would have emitted. We rewrap top-level DeclItem-shaped
        # decls into a single DeclareStmt to preserve source-equivalent
        # shape; codegen's block_items_split flattens it back out.
        new_items: list = []
        declare_buf: list = []

        def flush_declare() -> None:
            if declare_buf:
                new_items.append(P.DeclareStmt(declarations=list(declare_buf), pos=decl.body.pos))
                declare_buf.clear()

        for d in new_decls:
            if isinstance(d, P.ProcDecl):
                flush_declare()
                new_items.append(d)
            else:
                declare_buf.append(d)
        flush_declare()
        new_items.extend(new_stmts)

        new_body = P.ProcBody(
            items=new_items,
            end_label=decl.body.end_label,
            pos=decl.body.pos,
        )
        optimized_proc = P.ProcDecl(
            name=decl.name,
            signature=decl.signature,
            body=new_body,
            pos=decl.pos,
        )
        # Level 3: Track inlinable procedures, and the scopes their bodies'
        # names are resolved in.
        name = proc_name(optimized_proc)
        self.inlinable_procs.pop(name, None)
        if self.opt_level >= 3 and self._is_inlinable(optimized_proc, attrs, local_decls):
            self.inlinable_procs[name] = optimized_proc
            self.inline_scopes[name] = enclosing_scopes
        return optimized_proc

    def _optimize_decl_in_body(self, decl):
        """Optimize a single declaration found inside a procedure /
        DO block body. May be a :class:`P.ProcDecl`, a typed decl item
        (:class:`P.DeclItem` / :class:`P.DeclItemBasedGroup`), or a
        :class:`P.LiterallyDecl`."""
        if isinstance(decl, P.ProcDecl):
            return self._optimize_proc_decl(decl)
        if isinstance(decl, P.LiterallyDecl):
            return self._optimize_literally_decl(decl)
        if isinstance(decl, (P.DeclItem, P.DeclItemBasedGroup)):
            return self._optimize_decl_item(decl)
        return decl

    def _optimize_literally_decl(self, decl: P.LiterallyDecl) -> P.LiterallyDecl:
        """Track a LITERALLY constant for later constant propagation."""
        name = ident_text(decl.name)
        raw = decl.value.text
        if raw.startswith("'") and raw.endswith("'"):
            raw = raw[1:-1]
        val = self._parse_plm_number(raw)
        if val is not None and val <= 0xFFFF:
            # The name stands for the literal, typed as a literal is.
            self.constants[name] = (val, literal_type(val), False)
        return decl

    def _optimize_declare_stmt(self, stmt: P.DeclareStmt) -> P.DeclareStmt | None:
        """Optimize a top-level DECLARE statement (rewrap surviving items)."""
        new_decls: list = []
        for d in stmt.declarations:
            opt = self._optimize_decl_in_body(d)
            if opt is not None:
                new_decls.append(opt)
        if not new_decls:
            return None
        return P.DeclareStmt(declarations=new_decls, pos=stmt.pos)

    def _optimize_decl_item(self, item):
        """Optimize the value-bearing parts of a :class:`P.DeclItem` /
        :class:`P.DeclItemBasedGroup` in place (initial values, data
        values, ``AT(...)`` address).

        The structural fields (name, type, dimension) aren't touched —
        we only fold the expression sub-nodes inside the tail's
        attribute clauses. The decl is mutated rather than rebuilt so
        codegen's symbol-table lookups by identity stay consistent.
        """
        tail = getattr(item, "tail", None)
        if tail is None:
            return item

        # These are restricted expressions (6.2.8): address arithmetic on
        # constants, evaluated as plain numbers -- "when a restricted
        # expression is used to initialize a BYTE scalar, its value must not
        # be greater than 255" -- not by the BYTE/ADDRESS rules of an
        # executable expression. Code generation evaluates what is left the
        # same way.
        outer, self.restricted = self.restricted, True
        try:
            # Optimize AttrInitial / AttrAt expressions in the attribute lists.
            for attr_list_name in ("attrs", "leading_attrs", "trailing_attrs"):
                attrs = getattr(tail, attr_list_name, None)
                if not attrs:
                    continue
                for attr in attrs:
                    if isinstance(attr, P.AttrInitial):
                        attr.values = [self._optimize_expr(v) for v in (attr.values or [])]
                    elif isinstance(attr, P.AttrAt):
                        attr.address = self._optimize_expr(attr.address)

            # Optimize DATA values list (lives directly on the tail variant).
            if hasattr(tail, "data_values") and tail.data_values:
                tail.data_values = [self._optimize_expr(v) for v in tail.data_values]
        finally:
            self.restricted = outer

        return item

    # ---- reachability / dead-store analysis -------------------------------

    def _is_terminator(self, stmt) -> bool:
        """Check if a statement is a control flow terminator (no fall-through)."""
        if isinstance(stmt, (P.ReturnStmt, P.ReturnStmtValue)):
            return True
        if isinstance(stmt, P.GotoStmt):
            return True
        if isinstance(stmt, P.HaltStmt):
            return True
        if isinstance(stmt, P.LabeledStmt):
            return self._is_terminator(stmt.stmt)
        return False

    def _eliminate_unreachable(self, stmts: list) -> list:
        """Remove statements after terminators (RETURN, GOTO, HALT).

        Preserves labeled statements after terminators since they can be
        reached via GOTO from elsewhere in the code.
        """
        if self.opt_level < 2:
            return stmts

        result: list = []
        in_unreachable = False
        for stmt in stmts:
            if in_unreachable:
                # A label anywhere inside the statement (in a nested DO, an
                # IF arm) can be jumped to, and a GOTO elsewhere still names
                # it.
                if self._contains_label(stmt):
                    in_unreachable = False
                    result.append(stmt)
                else:
                    self.stats.dead_code_eliminated += 1
            else:
                result.append(stmt)
                if self._is_terminator(stmt):
                    in_unreachable = True
        return result

    def _eliminate_dead_stores(self, stmts: list) -> list:
        """Remove assignments that are immediately overwritten without being read.

        A variable assigned and then reassigned in consecutive statements
        without being read between is a dead store -- when dropping the
        first assignment drops nothing else. Its value must do nothing
        (no call, no embedded assignment), the variable must be a plain
        one (a store to a BASED or AT variable is a store to memory
        something else may read), and the second value must not be able to
        read it: no call (the procedure may read the variable), and nothing
        that reads memory by address (an element, a member, a BASED
        variable), which may alias it.
        """
        if self.opt_level < 3:
            return stmts

        result: list = []
        i = 0
        while i < len(stmts):
            stmt = stmts[i]

            if (i + 1 < len(stmts)
                    and self._single_plain_target(stmt) is not None
                    and self._single_plain_target(stmt) == self._single_plain_target(stmts[i + 1])
                    and self._is_side_effect_free(stmt.value)
                    and self._reads_only_plain_scalars(stmts[i + 1].value)
                    and self._single_plain_target(stmt) not in _get_expr_vars(stmts[i + 1].value)):
                self.stats.dead_stores_eliminated += 1
                i += 1
                continue

            result.append(stmt)
            i += 1

        return result

    def _single_plain_target(self, stmt) -> str | None:
        """The one plain variable ``stmt`` assigns, if that is all it is."""
        if not (isinstance(stmt, P.AssignStmt) and len(stmt.targets) == 1):
            return None
        t = unwrap_paren(stmt.targets[0])
        if isinstance(t, P.Identifier) and self._plain_var_type(ident_text(t.name)):
            return ident_text(t.name)
        return None

    def _reads_only_plain_scalars(self, expr) -> bool:
        """``expr`` reads nothing but constants and plain variables."""
        e = unwrap_paren(expr)
        if isinstance(e, (P.NumberLiteral, P.StringLiteral)):
            return True
        if isinstance(e, P.Identifier):
            return self._plain_var_type(ident_text(e.name)) is not None
        if isinstance(e, P.BinaryOp):
            return (self._reads_only_plain_scalars(e.left)
                    and self._reads_only_plain_scalars(e.right))
        if isinstance(e, P.UnaryOp):
            return self._reads_only_plain_scalars(e.operand)
        if isinstance(e, P.Call):
            callee = unwrap_paren(e.callee)
            return (isinstance(callee, P.Identifier)
                    and ident_text(callee.name).upper() in _PURE_BUILTINS
                    and self._is_builtin(ident_text(callee.name))
                    and all(self._reads_only_plain_scalars(a) for a in e.args))
        if isinstance(e, P.LocationOf):
            return isinstance(unwrap_paren(e.operand), P.Identifier)
        return isinstance(e, P.LocationOfString)

    def _pointer_reachable_names(self, items) -> set[str]:
        """Every name whose address is taken, or that is AT something,
        PUBLIC or EXTERNAL, anywhere in ``items``."""
        names: set[str] = set()
        for n in self._walk(items):
            if isinstance(n, P.LocationOf):
                base = unwrap_paren(n.operand)
                while isinstance(base, (P.Call, P.MemberAccess)):
                    base = unwrap_paren(base.callee if isinstance(base, P.Call) else base.base)
                if isinstance(base, P.Identifier):
                    names.add(ident_text(base.name))
            elif isinstance(n, P.DeclItem):
                attrs = decl_attrs(n)
                if attrs.at_location is not None or attrs.is_public or attrs.is_external:
                    names.update(decl_item_names(n))
        return names

    def _body_may_move(self, name: str, stmts) -> bool:
        """Whether running ``stmts`` may change the variable ``name``: they
        assign it, call something (which may assign anything), or store
        through a pointer while ``name`` is one a pointer may reach."""
        if name in self._get_modified_vars_in_stmts(stmts) or self._contains_call(stmts):
            return True
        d = self._lookup(name)
        reachable = d is None or not d.plain or name in self.pointer_reachable
        return reachable and self._stores_indirectly(stmts)

    def _get_modified_vars_in_stmts(self, stmts: list) -> set[str]:
        """Every variable the statements assign, anywhere inside them.

        Statement targets (and the array of a subscripted one), embedded
        assignments -- in a condition, a subscript or an argument as much
        as in a value: `arr(i := i + 1) = x' modifies i as well as arr, and
        missing that left a loop whose only induction step is in a
        subscript looking invariant -- and iterative DO indices.
        """
        modified: set[str] = set()

        def target_name(t) -> str | None:
            t = unwrap_paren(t)
            if isinstance(t, P.Identifier):
                return ident_text(t.name)
            if isinstance(t, P.Call):
                c = unwrap_paren(t.callee)
                if isinstance(c, P.Identifier):
                    return ident_text(c.name)
            return None

        stack: list = list(stmts)
        while stack:
            n = stack.pop()
            if isinstance(n, P.AssignStmt):
                for t in n.targets:
                    name = target_name(t)
                    if name is not None:
                        modified.add(name)
            elif isinstance(n, P.EmbeddedAssign):
                name = target_name(n.target)
                if name is not None:
                    modified.add(name)
            elif isinstance(n, (P.DoIterBlock, P.DoIterByBlock)):
                modified.add(ident_text(n.index))
            if isinstance(n, (list, tuple)):
                stack.extend(n)
                continue
            fields = getattr(n, "__dataclass_fields__", None)
            if not fields:
                continue
            for f in fields:
                if f == "pos":
                    continue
                stack.append(getattr(n, f, None))
        return modified

    def _cache_invariant_exprs(self, expr, modified_vars: set[str]) -> None:
        """Cache loop-invariant subexpressions for CSE to find later."""
        if expr is None:
            return
        if self._is_loop_invariant(expr, modified_vars):
            key = _expr_key(expr)
            if key is not None and key not in self.cse_cache:
                self.cse_cache[key] = (f"??INV{self.cse_counter}", expr)
                self.expr_vars[key] = _get_expr_vars(expr)
                self.cse_counter += 1
                self.stats.loop_invariants_moved += 1

        e = unwrap_paren(expr)
        if isinstance(e, P.BinaryOp):
            self._cache_invariant_exprs(e.left, modified_vars)
            self._cache_invariant_exprs(e.right, modified_vars)
        elif isinstance(e, P.UnaryOp):
            self._cache_invariant_exprs(e.operand, modified_vars)
        elif isinstance(e, P.Call):
            for arg in e.args:
                self._cache_invariant_exprs(arg, modified_vars)

    def _is_loop_invariant(self, expr, modified_vars: set[str]) -> bool:
        """Check if an expression is invariant (not modified) within a loop."""
        if expr is None:
            return False
        e = unwrap_paren(expr)
        if isinstance(e, P.NumberLiteral):
            return True
        if isinstance(e, P.StringLiteral):
            return True
        if isinstance(e, P.Identifier):
            return ident_text(e.name) not in modified_vars
        if isinstance(e, P.BinaryOp):
            return (
                self._is_loop_invariant(e.left, modified_vars)
                and self._is_loop_invariant(e.right, modified_vars)
            )
        if isinstance(e, P.UnaryOp):
            return self._is_loop_invariant(e.operand, modified_vars)
        if isinstance(e, P.Call):
            # Only pure builtins (incl. subscript-shaped calls on
            # unmodified arrays) can be considered loop-invariant.
            if isinstance(e.callee, P.Identifier):
                name = ident_text(e.callee.name)
                if name in _PURE_BUILTINS:
                    return all(self._is_loop_invariant(a, modified_vars) for a in e.args)
                # Subscript-shaped call: if base array is modified, not invariant.
                if name in modified_vars:
                    return False
                # Otherwise treat as opaque (side effect or unknown) — not invariant.
                return False
            return False
        if isinstance(e, P.LocationOf):
            return self._is_loop_invariant(e.operand, modified_vars)
        if isinstance(e, P.MemberAccess):
            return self._is_loop_invariant(e.base, modified_vars)
        return False

    def _invalidate_cse_for_var(self, var_name: str) -> None:
        """Invalidate CSE cache entries that depend on a modified variable."""
        if self.opt_level < 3:
            return
        to_remove = []
        for key, vars_used in self.expr_vars.items():
            if var_name in vars_used:
                to_remove.append(key)
        for key in to_remove:
            self.cse_cache.pop(key, None)
            self.expr_vars.pop(key, None)

    def _invalidate_copies_for_var(self, var_name: str) -> None:
        """Invalidate copy propagation entries when a variable is modified."""
        self.copies.pop(var_name, None)
        to_remove = [k for k, v in self.copies.items() if v == var_name]
        for k in to_remove:
            del self.copies[k]

    def _count_stmts(self, stmts: list) -> int:
        """Count the number of statements (recursively)."""
        count = 0
        for stmt in stmts:
            count += 1
            if isinstance(stmt, P.DoBlock):
                _, body = block_items_split(stmt.items)
                count += self._count_stmts(body)
            elif isinstance(stmt, P.DoWhileBlock):
                _, body = block_items_split(stmt.items)
                count += self._count_stmts(body)
            elif isinstance(stmt, (P.DoIterBlock, P.DoIterByBlock)):
                _, body = block_items_split(stmt.items)
                count += self._count_stmts(body)
            elif isinstance(stmt, P.DoCaseBlock):
                for case in stmt.items:
                    count += self._count_stmts([case])
            elif isinstance(stmt, P.IfStmt):
                if isinstance(stmt.then_stmt, (P.DoBlock, P.DoWhileBlock)):
                    count += self._count_stmts([stmt.then_stmt])
            elif isinstance(stmt, P.IfStmtElse):
                if isinstance(stmt.then_stmt, (P.DoBlock, P.DoWhileBlock)):
                    count += self._count_stmts([stmt.then_stmt])
                if isinstance(stmt.else_stmt, (P.DoBlock, P.DoWhileBlock)):
                    count += self._count_stmts([stmt.else_stmt])
        return count

    def _is_inlinable(self, proc: P.ProcDecl, attrs, local_decls) -> bool:
        """Check if a procedure is suitable for inlining.

        Only a small untyped procedure with no parameters and nothing
        declared in it: its body then means the same wherever it is copied,
        provided the names it uses resolve to the same declarations there
        (checked at the call, see _names_resolve_alike). It must not RETURN
        other than by falling off its end -- a RETURN copied into the caller
        returns from the caller -- nor hold a label, which the copy would
        define twice, nor a GOTO, nor read a flag.
        """
        if attrs.is_external or attrs.is_reentrant or attrs.interrupt_num is not None:
            return False
        if local_decls or proc_param_names(proc) or proc_return_type(proc) is not None:
            return False
        _, body_stmts = proc_local_decls_stmts(proc)
        if self._count_stmts(body_stmts) > 5:
            return False
        body = list(body_stmts)
        if body and isinstance(body[-1], P.ReturnStmt):
            body = body[:-1]
        stack: list = list(body)
        while stack:
            n = stack.pop()
            if isinstance(n, (P.ReturnStmt, P.ReturnStmtValue, P.LabeledStmt,
                              P.GotoStmt, P.DeclareStmt, P.ProcDecl, P.DeclItem)):
                return False
            if isinstance(n, (list, tuple)):
                stack.extend(n)
                continue
            fields = getattr(n, "__dataclass_fields__", None)
            if not fields:
                continue
            for f in fields:
                if f == "pos":
                    continue
                stack.append(getattr(n, f, None))
        return not self._reads_a_flag(body_stmts)

    def _names_resolve_alike(self, proc_name_: str) -> bool:
        """Every name in the procedure's body means here what it means there."""
        home = self.inline_scopes.get(proc_name_)
        if home is None:
            return False
        proc = self.inlinable_procs[proc_name_]

        def resolve(scopes, name):
            for scope in reversed(scopes):
                if name in scope:
                    return scope[name]
            return None

        for name in _all_names(proc.body):
            if resolve(home, name) is not self._lookup(name):
                return False
        return True

    def _inline_procedure(self, proc: P.ProcDecl, args: list, pos):
        """Inline a procedure call, substituting parameters with arguments."""
        params = proc_param_names(proc)
        param_map: dict[str, object] = {}
        for param, arg in zip(params, args):
            param_map[param] = arg

        _, body_stmts = proc_local_decls_stmts(proc)
        inlined_stmts: list = []
        for stmt in body_stmts:
            # Skip empty RETURN statements (void return)
            if isinstance(stmt, P.ReturnStmt):
                continue
            subst = self._substitute_params(deepcopy(stmt), param_map)
            if subst is not None:
                inlined_stmts.append(subst)

        if not inlined_stmts:
            return P.NullStmt(pos=pos)
        if len(inlined_stmts) == 1:
            return inlined_stmts[0]
        return P.DoBlock(items=inlined_stmts, end_label=None, pos=pos)

    def _substitute_params(self, node, param_map: dict[str, object]):
        """Substitute parameter references with argument expressions.

        Walks every node kind that can contain a name reference. Returns
        the (possibly mutated) input node, or a replacement node for
        :class:`P.Identifier` instances whose text matches a parameter.
        """
        if node is None:
            return None

        if isinstance(node, P.Identifier):
            name = ident_text(node.name)
            if name in param_map:
                return deepcopy(param_map[name])
            return node

        if isinstance(node, P.ParenExpr):
            node.inner = self._substitute_params(node.inner, param_map)
            return node

        if isinstance(node, P.BinaryOp):
            node.left = self._substitute_params(node.left, param_map)
            node.right = self._substitute_params(node.right, param_map)
            return node

        if isinstance(node, P.UnaryOp):
            node.operand = self._substitute_params(node.operand, param_map)
            return node

        if isinstance(node, P.Call):
            node.callee = self._substitute_params(node.callee, param_map)
            node.args = [self._substitute_params(a, param_map) for a in node.args]
            return node

        if isinstance(node, P.CallNoArgs):
            node.callee = self._substitute_params(node.callee, param_map)
            return node

        if isinstance(node, P.MemberAccess):
            node.base = self._substitute_params(node.base, param_map)
            return node

        if isinstance(node, P.LocationOf):
            node.operand = self._substitute_params(node.operand, param_map)
            return node

        if isinstance(node, P.LocationOfList):
            node.values = [self._substitute_params(v, param_map) for v in node.values]
            return node

        if isinstance(node, P.EmbeddedAssign):
            node.target = self._substitute_params(node.target, param_map)
            node.value = self._substitute_params(node.value, param_map)
            return node

        if isinstance(node, P.AssignStmt):
            node.targets = [self._substitute_params(t, param_map) for t in node.targets]
            node.value = self._substitute_params(node.value, param_map)
            return node

        if isinstance(node, P.CallStmt):
            node.callee = self._substitute_params(node.callee, param_map)
            return node

        if isinstance(node, P.ReturnStmtValue):
            node.value = self._substitute_params(node.value, param_map)
            return node

        if isinstance(node, P.IfStmt):
            node.condition = self._substitute_params(node.condition, param_map)
            node.then_stmt = self._substitute_params(node.then_stmt, param_map)
            return node

        if isinstance(node, P.IfStmtElse):
            node.condition = self._substitute_params(node.condition, param_map)
            node.then_stmt = self._substitute_params(node.then_stmt, param_map)
            node.else_stmt = self._substitute_params(node.else_stmt, param_map)
            return node

        if isinstance(node, P.DoBlock):
            node.items = [self._substitute_params(s, param_map) for s in node.items]
            return node

        if isinstance(node, P.DoWhileBlock):
            node.condition = self._substitute_params(node.condition, param_map)
            node.items = [self._substitute_params(s, param_map) for s in node.items]
            return node

        if isinstance(node, (P.DoIterBlock, P.DoIterByBlock)):
            node.start = self._substitute_params(node.start, param_map)
            node.bound = self._substitute_params(node.bound, param_map)
            if isinstance(node, P.DoIterByBlock):
                node.step = self._substitute_params(node.step, param_map)
            node.items = [self._substitute_params(s, param_map) for s in node.items]
            return node

        if isinstance(node, P.DoCaseBlock):
            node.selector = self._substitute_params(node.selector, param_map)
            node.items = [self._substitute_params(s, param_map) for s in node.items]
            return node

        if isinstance(node, P.LabeledStmt):
            node.stmt = self._substitute_params(node.stmt, param_map)
            return node

        return node

    # ---- commutative normalization ----------------------------------------

    _COMMUTATIVE = {
        BinaryOpKind.ADD,
        BinaryOpKind.MUL,
        BinaryOpKind.AND,
        BinaryOpKind.OR,
        BinaryOpKind.XOR,
        BinaryOpKind.EQ,
        BinaryOpKind.NE,
    }

    def _normalize_commutative(self, kind: BinaryOpKind, left, right):
        """Normalize operand order for commutative operations to improve CSE.

        Only when neither operand does anything: PL/M-80 leaves the order
        of evaluation open, and a swap changes the order code generation
        picks. A literal moved to the right of a relation is marked derived,
        since it is not where the programmer wrote it (see
        CodeGenerator._check_impossible_comparison).
        """
        if kind not in self._COMMUTATIVE:
            return left, right
        if not (self._is_side_effect_free(left) and self._is_side_effect_free(right)):
            return left, right

        def sort_key(e) -> tuple[int, str]:
            e = unwrap_paren(e)
            if isinstance(e, P.NumberLiteral):
                return (2, f"{number_value(e)}")
            elif isinstance(e, P.Identifier):
                return (0, ident_text(e.name))
            else:
                return (1, _expr_key(e) or "")

        left_key = sort_key(left)
        right_key = sort_key(right)

        if right_key < left_key:
            if kind in RELATIONS:
                # Any constant, not just a literal: `'AB' <> b', or a sum
                # a PLUS in the procedure keeps from being folded, was
                # rejected at -O3 alone. Its value is the relation's
                # operand (the flags of computing it are dead after the
                # compare), so it goes over as that value.
                typed = eval_typed(left)
                if typed is not None:
                    left = make_typed_const(typed[0], typed[1], getattr(left, "pos", None),
                                            derived=True)
            return right, left
        return left, right

    # ---- statement optimization -------------------------------------------

    def _optimize_target(self, target):
        """Optimize inside an assignment target without rewriting the lvalue.

        A target names a place to store to, not a value, so constant and
        copy propagation must not reach it: rewriting the ``A`` of
        ``A = 5`` into the literal ``5`` turns the assignment into a
        store *through address 5* — on CP/M, straight into the BDOS
        entry vector. Only an array element's subscript is a value, and
        only that is optimized here.
        """
        inner = unwrap_paren(target)
        if isinstance(inner, P.Call):
            callee = self._optimize_target(inner.callee)
            opt_args = [untyped_root(self._optimize_expr(a)) for a in inner.args]
            if callee is not inner.callee or any(a is not b for a, b in zip(opt_args, inner.args)):
                return P.Call(callee=callee, args=opt_args, pos=inner.pos)
        elif isinstance(inner, P.MemberAccess):
            base = self._optimize_target(inner.base)
            if base is not inner.base:
                return P.MemberAccess(base=base, member=inner.member, pos=inner.pos)
        return target

    def _optimize_stmt(self, stmt):
        """Optimize a typed statement. Returns ``None`` to remove it."""
        if stmt is None:
            return None

        if isinstance(stmt, P.AssignStmt):
            return self._optimize_assign(stmt)

        if isinstance(stmt, P.CallStmt):
            return self._optimize_call_stmt(stmt)

        if isinstance(stmt, P.ReturnStmt):
            return stmt

        if isinstance(stmt, P.ReturnStmtValue):
            opt_value = self._optimize_value(stmt.value)
            return P.ReturnStmtValue(value=opt_value, pos=stmt.pos)

        if isinstance(stmt, (P.IfStmt, P.IfStmtElse)):
            return self._optimize_if(stmt)

        if isinstance(stmt, P.DoBlock):
            return self._optimize_do_block(stmt)

        if isinstance(stmt, P.DoWhileBlock):
            return self._optimize_do_while(stmt)

        if isinstance(stmt, (P.DoIterBlock, P.DoIterByBlock)):
            return self._optimize_do_iter(stmt)

        if isinstance(stmt, P.DoCaseBlock):
            return self._optimize_do_case(stmt)

        if isinstance(stmt, P.LabeledStmt):
            # A label is a join point: a GOTO anywhere in the procedure can
            # land here, including one that closes a loop, so nothing
            # learned along the fall-through path survives it. Without this,
            #   n = 0;
            #   lp: if n >= 3 then go to fin;
            #       call pc('0' + n); n = n + 1; go to lp;
            # folded the test with n pinned at 0 and looped forever at -O 3.
            self._reset_flow_state()
            opt_inner = self._optimize_stmt(stmt.stmt)
            if opt_inner is None:
                opt_inner = P.NullStmt(pos=stmt.pos)
            # And nothing learned inside the labelled statement holds for
            # code that reaches it by the same back edge.
            self._reset_flow_state()
            return P.LabeledStmt(label=stmt.label, stmt=opt_inner, pos=stmt.pos)

        if isinstance(stmt, P.DeclareStmt):
            return self._optimize_declare_stmt(stmt)

        # P.GotoStmt, P.HaltStmt, P.EnableStmt, P.DisableStmt, P.NullStmt — pass through.
        return stmt

    def _optimize_assign(self, stmt: P.AssignStmt):
        """Optimize an assignment, and record what it makes known."""
        # The value is computed first, then each target's subscript; a call
        # anywhere in either can change any variable (see _optimize_value).
        effectful = self._contains_call(stmt.value) or self._contains_call(stmt.targets)
        self._forget_embedded_targets([stmt.value, stmt.targets])
        if effectful:
            self._reset_flow_state()
        # Stored, the value takes the target's type, so its own type does
        # not matter when it is a constant.
        opt_value = untyped_root(self._optimize_expr(stmt.value))
        if effectful:
            self._reset_flow_state()
        opt_targets = [self._optimize_target(t) for t in stmt.targets]

        for target in opt_targets:
            t = unwrap_paren(target)
            if isinstance(t, P.Identifier):
                name = ident_text(t.name)
                self.modified_vars.add(name)
                self._forget([name])

        if effectful or self._stores_indirectly(stmt):
            self._reset_flow_state()
        elif self.opt_level >= 3 and len(opt_targets) == 1:
            # Level 3: track constants and copies of plain variables, with
            # the value converted to the variable's type.
            t = unwrap_paren(opt_targets[0])
            ttype = (self._plain_var_type(ident_text(t.name))
                     if isinstance(t, P.Identifier) else None)
            if ttype is not None:
                tname = ident_text(t.name)
                tc = typed_const(opt_value)
                v = unwrap_paren(opt_value)
                if tc is not None:
                    self.constants[tname] = (convert(tc[0], ttype), ttype, True)
                elif (isinstance(v, P.Identifier) and ident_text(v.name) != tname
                      and self._plain_var_type(ident_text(v.name)) is ttype):
                    # Only a plain variable of the same type: `w = b' makes
                    # w the zero-extended b, and `w + 0FFH' is not `b +
                    # 0FFH'. A name that is a procedure is a call, and is
                    # not plain.
                    self.copies[tname] = ident_text(v.name)

        return P.AssignStmt(targets=opt_targets, value=opt_value, pos=stmt.pos)

    def _optimize_call_stmt(self, stmt: P.CallStmt):
        """Optimize a CALL statement; inline it at -O3 where that is sound."""
        # Unpack the call-form payload into (callee_expr, args) so
        # we can fold builtin arguments and check for inlining.
        inner = stmt.callee
        if isinstance(inner, P.Call):
            callee_expr = inner.callee
            args = list(inner.args)
            inner_pos = inner.pos
        elif isinstance(inner, P.CallNoArgs):
            callee_expr = inner.callee
            args = []
            inner_pos = inner.pos
        else:
            callee_expr = inner
            args = []
            inner_pos = getattr(inner, "pos", stmt.pos)

        # The arguments are evaluated before the call, so what is known
        # holds in them -- unless one of them makes a call itself.
        self._forget_embedded_targets(args)
        if self._contains_call(args):
            self._reset_flow_state()
        opt_callee = self._optimize_expr(callee_expr)
        # Each argument is converted to its parameter's type.
        opt_args = [untyped_root(self._optimize_expr(a)) for a in args]

        # Level 3: Inline small procedures
        if (
            self.opt_level >= 3
            and self.optimize_for != OptimizeFor.SIZE
            and isinstance(unwrap_paren(opt_callee), P.Identifier)
            and not opt_args
        ):
            name = ident_text(unwrap_paren(opt_callee).name)
            if name in self.inlinable_procs and self._names_resolve_alike(name):
                inlined = self._inline_procedure(self.inlinable_procs[name], [], stmt.pos)
                if inlined is not None:
                    self.stats.procedures_inlined += 1
                    # What the body changes is no longer known.
                    self._invalidate_modified([inlined])
                    return inlined

        # The procedure may change any variable.
        self._reset_flow_state()

        # Repack into the original call shape.
        if opt_args:
            new_inner = P.Call(callee=opt_callee, args=opt_args, pos=inner_pos)
        elif isinstance(inner, P.Call):
            new_inner = P.Call(callee=opt_callee, args=[], pos=inner_pos)
        elif isinstance(inner, P.CallNoArgs):
            new_inner = P.CallNoArgs(callee=opt_callee, pos=inner_pos)
        else:
            new_inner = opt_callee
        return P.CallStmt(callee=new_inner, pos=stmt.pos)

    def _optimize_if(self, stmt):
        """Optimize an IF / IF-ELSE statement.

        Returns a possibly-different shape — folding ``IF 1`` reduces
        to just the then-branch, ``IF 0`` reduces to the else branch
        (or a :class:`P.NullStmt`). Also collapses :class:`P.IfStmtElse`
        whose else-branch optimizes away into :class:`P.IfStmt`.
        """
        opt_cond = self._optimize_value(stmt.condition)

        # Constant condition elimination (level 2+). Not when either arm
        # declares a label: a GOTO elsewhere in the procedure still names it.
        else_stmt = stmt.else_stmt if isinstance(stmt, P.IfStmtElse) else None
        if (
            self.opt_level >= 2
            and isinstance(unwrap_paren(opt_cond), P.NumberLiteral)
            and not self._contains_label(stmt.then_stmt)
            and not (else_stmt is not None and self._contains_label(else_stmt))
        ):
            self.stats.dead_code_eliminated += 1
            # PL/M-80 truth is bit 0 of the value, not non-zero: `IF 4` is
            # false, and `IF NOT TRUE` with `TRUE LITERALLY '1'` folds to
            # 0FEH, which is false.
            if number_value(unwrap_paren(opt_cond)) & 1:
                return self._optimize_stmt(stmt.then_stmt)
            if isinstance(stmt, P.IfStmtElse):
                return self._optimize_stmt(stmt.else_stmt)
            return P.NullStmt(pos=stmt.pos)

        # Each arm starts from the state at the IF, not from whatever the
        # other arm established: only one of them runs. Facts either arm
        # invalidates do not survive the join, because the compiler cannot
        # know which way control went.
        entry = self._snapshot_flow_state()
        opt_then = self._optimize_stmt(stmt.then_stmt)
        if opt_then is None:
            opt_then = P.NullStmt(pos=stmt.pos)
        self._restore_flow_state(entry)

        if isinstance(stmt, P.IfStmtElse):
            opt_else = self._optimize_stmt(stmt.else_stmt)
            self._restore_flow_state(entry)
            self._invalidate_modified([stmt.then_stmt, stmt.else_stmt])
            if opt_else is None:
                return P.IfStmt(condition=opt_cond, then_stmt=opt_then, pos=stmt.pos)
            return P.IfStmtElse(
                condition=opt_cond,
                then_stmt=opt_then,
                else_stmt=opt_else,
                pos=stmt.pos,
            )
        self._invalidate_modified([stmt.then_stmt])
        return P.IfStmt(condition=opt_cond, then_stmt=opt_then, pos=stmt.pos)

    def _optimize_block_items(self, items: list) -> list:
        """Optimize a typed block's mixed (decl + stmt) item list.

        Walks each item with the appropriate handler and returns a flat
        list of survivors. Declarations are kept inline (wrapped back in
        a :class:`P.DeclareStmt` to mirror the parser's emit shape);
        nested procedure decls survive as standalone items.
        """
        new_items: list = []
        decl_buf: list = []

        def flush() -> None:
            if decl_buf:
                new_items.append(
                    P.DeclareStmt(declarations=list(decl_buf), pos=decl_buf[0].pos)
                )
                decl_buf.clear()

        self._push_scope(items)
        for it in items:
            if isinstance(it, P.ProcDecl):
                flush()
                opt = self._optimize_proc_decl(it)
                if opt is not None:
                    new_items.append(opt)
            elif isinstance(it, P.DeclareStmt):
                # Inline-flatten the inner items so they get rewrapped uniformly.
                for d in it.declarations:
                    opt_d = self._optimize_decl_in_body(d)
                    if opt_d is not None:
                        decl_buf.append(opt_d)
            elif isinstance(it, (P.DeclItem, P.DeclItemBasedGroup, P.LiterallyDecl)):
                opt = self._optimize_decl_in_body(it)
                if opt is not None:
                    decl_buf.append(opt)
            else:
                # Statement
                flush()
                opt = self._optimize_stmt(it)
                if opt is not None:
                    new_items.append(opt)
        flush()
        self._pop_scope()
        return new_items

    def _optimize_do_block(self, stmt: P.DoBlock) -> P.DoBlock:
        """Optimize a plain ``DO ... END`` block."""
        new_items = self._optimize_block_items(stmt.items)
        # Eliminate unreachable + dead stores within the statement
        # portion only (block_items_split splits decls vs stmts based
        # on item kind, and unreachable analysis only applies to
        # the statement tail).
        # We re-run the analyses on the full item list by isolating
        # the trailing statement-only suffix.
        new_items = self._eliminate_unreachable_in_items(new_items)
        new_items = self._eliminate_dead_stores_in_items(new_items)
        return P.DoBlock(items=new_items, end_label=stmt.end_label, pos=stmt.pos)

    def _eliminate_unreachable_in_items(self, items: list) -> list:
        """Apply :meth:`_eliminate_unreachable` only to the statement suffix."""
        if self.opt_level < 2:
            return items
        # Find the split: everything that's a decl-shape stays at the front,
        # the statement suffix is the rest. We treat DeclareStmt as a decl.
        decl_prefix: list = []
        stmt_suffix: list = []
        seen_stmt = False
        for it in items:
            if not seen_stmt and isinstance(it, (P.DeclareStmt, P.ProcDecl)):
                decl_prefix.append(it)
            else:
                seen_stmt = True
                stmt_suffix.append(it)
        stmt_suffix = self._eliminate_unreachable(stmt_suffix)
        return decl_prefix + stmt_suffix

    def _eliminate_dead_stores_in_items(self, items: list) -> list:
        """Apply :meth:`_eliminate_dead_stores` only to the statement suffix."""
        if self.opt_level < 3:
            return items
        decl_prefix: list = []
        stmt_suffix: list = []
        seen_stmt = False
        for it in items:
            if not seen_stmt and isinstance(it, (P.DeclareStmt, P.ProcDecl)):
                decl_prefix.append(it)
            else:
                seen_stmt = True
                stmt_suffix.append(it)
        stmt_suffix = self._eliminate_dead_stores(stmt_suffix)
        return decl_prefix + stmt_suffix

    def _optimize_do_while(self, stmt: P.DoWhileBlock):
        """Optimize a ``DO WHILE cond ... END`` block."""
        # Before anything is folded: the condition is re-evaluated on every
        # back edge, so it cannot use a fact the body invalidates.
        self._invalidate_modified(stmt.items)
        opt_cond = self._optimize_value(stmt.condition)

        # A DO WHILE whose condition has bit 0 clear never executes -- but
        # only drop the loop when its body declares no label.
        if (
            self.opt_level >= 2
            and isinstance(unwrap_paren(opt_cond), P.NumberLiteral)
            and not any(self._contains_label(i) for i in stmt.items)
        ):
            if number_value(unwrap_paren(opt_cond)) & 1 == 0:
                self.stats.dead_code_eliminated += 1
                return P.NullStmt(pos=stmt.pos)

        # Level 3: Check for loop-invariant subexpressions in condition.
        if self.opt_level >= 3:
            _, body_stmts = block_items_split(stmt.items)
            modified_vars = self._get_modified_vars_in_stmts(body_stmts)
            self._cache_invariant_exprs(opt_cond, modified_vars)

        new_items = self._optimize_block_items(stmt.items)
        new_items = self._eliminate_unreachable_in_items(new_items)
        # The body may have run any number of times, including none, so
        # what it established does not hold after the loop.
        self._invalidate_modified(stmt.items)
        return P.DoWhileBlock(
            condition=opt_cond,
            items=new_items,
            end_label=stmt.end_label,
            pos=stmt.pos,
        )

    @staticmethod
    def _loop_values(start: int, bound: int, step: int, t: DataType,
                     limit: int) -> "tuple[list[int], int] | None":
        """The index values of ``DO i = start TO bound BY step`` and the
        index's value after it, for an index of type ``t``, as DRI's PL/M-80
        runs it: the index is compared with the limit before each pass and
        the loop ends when the increment carries out of the index's width
        (5.1.4). None when it runs more than ``limit`` times."""
        i, bound, step = convert(start, t), convert(bound, t), convert(step, t)
        values: list[int] = []
        while i <= bound:
            if len(values) >= limit or step == 0:
                return None
            values.append(i)
            i += step
            if i > convert(0xFFFF, t):
                i = convert(i, t)
                break
        return values, i

    def _optimize_do_iter(self, stmt):
        """Optimize a ``DO I = start TO bound [BY step] ... END`` block.

        ``stmt`` is either :class:`P.DoIterBlock` (no BY) or
        :class:`P.DoIterByBlock` (with explicit step). Both share the
        same body / bound layout; the only difference is the presence
        of ``stmt.step``.
        """
        is_by = isinstance(stmt, P.DoIterByBlock)
        index_name = ident_text(stmt.index)
        # The bound is re-tested on every iteration; the index and whatever
        # the body assigns are not constants inside it.
        self._invalidate_modified(stmt.items)
        self._forget([index_name])
        # Start, limit and step are all converted to the index's type.
        opt_start = self._optimize_value(stmt.start)
        opt_bound = self._optimize_value(stmt.bound)
        opt_step = self._optimize_value(stmt.step) if is_by else None

        index_decl = self._lookup(index_name)
        index_type = (index_decl.dtype if index_decl is not None and index_decl.kind == "var"
                      and not index_decl.array else None)
        start_c, bound_c = typed_const(opt_start), typed_const(opt_bound)
        step_c = typed_const(opt_step) if is_by else (1, BYTE)
        bounds_known = index_type is not None and start_c is not None and bound_c is not None
        constant = bounds_known and step_c is not None
        _, body_stmts = block_items_split(stmt.items)

        def assign_index(value: int):
            return P.AssignStmt(targets=[make_identifier(index_name, pos=stmt.pos)],
                                value=make_number_literal(value, pos=stmt.pos),
                                pos=stmt.pos)

        # A loop that never runs still assigns its index the start value.
        if (
            self.opt_level >= 2
            and bounds_known
            and convert(start_c[0], index_type) > convert(bound_c[0], index_type)
            and not any(self._contains_label(i) for i in stmt.items)
        ):
            self.stats.dead_code_eliminated += 1
            return self._optimize_stmt(assign_index(convert(start_c[0], index_type)))

        # Level 3: Loop unrolling for small constant-bound loops: each pass
        # assigns the index its value, and the index is left with the value
        # the loop leaves in it. Not when the body changes the index, or
        # holds a label or a declaration the copies would repeat.
        max_iter = 4 if self.optimize_for == OptimizeFor.SPEED else 2
        run = (self._loop_values(start_c[0], bound_c[0], step_c[0], index_type, max_iter)
               if constant and self.opt_level >= 3
               and self.optimize_for != OptimizeFor.SIZE else None)
        if (
            run is not None
            and run[0]
            and len(body_stmts) <= 3
            and not self._body_may_move(index_name, body_stmts)
            and not any(self._contains_label(i) for i in stmt.items)
            and not any(isinstance(n, (P.DeclareStmt, P.ProcDecl, P.DeclItem))
                        for n in self._walk(stmt.items))
        ):
            values, final = run
            unrolled: list = []
            for val in values:
                unrolled.append(assign_index(val))
                for s in body_stmts:
                    unrolled.append(deepcopy(s))
            unrolled.append(assign_index(final))
            self.stats.loops_unrolled += 1
            block = P.DoBlock(items=unrolled, end_label=stmt.end_label, pos=stmt.pos)
            return self._optimize_stmt(block)

        # Level 3: Cache loop-invariant bound expressions.
        if self.opt_level >= 3:
            modified_vars = self._get_modified_vars_in_stmts(body_stmts)
            modified_vars.add(index_name)
            self._cache_invariant_exprs(opt_bound, modified_vars)
            if opt_step is not None:
                self._cache_invariant_exprs(opt_step, modified_vars)

        new_items = self._optimize_block_items(stmt.items)
        new_items = self._eliminate_unreachable_in_items(new_items)
        # Nothing the body established survives: it may not have run.
        self._invalidate_modified(stmt.items)
        self._forget([index_name])

        if is_by:
            return P.DoIterByBlock(
                index=stmt.index,
                start=opt_start,
                bound=opt_bound,
                step=opt_step,
                items=new_items,
                end_label=stmt.end_label,
                pos=stmt.pos,
            )
        return P.DoIterBlock(
            index=stmt.index,
            start=opt_start,
            bound=opt_bound,
            items=new_items,
            end_label=stmt.end_label,
            pos=stmt.pos,
        )

    @staticmethod
    def _walk(node):
        """Every node inside ``node``."""
        stack = [node]
        while stack:
            n = stack.pop()
            if isinstance(n, (list, tuple)):
                stack.extend(n)
                continue
            fields = getattr(n, "__dataclass_fields__", None)
            if not fields:
                continue
            yield n
            for f in fields:
                if f == "pos":
                    continue
                stack.append(getattr(n, f, None))

    def _optimize_do_case(self, stmt: P.DoCaseBlock):
        """Optimize a ``DO CASE selector ... END`` block."""
        opt_selector = self._optimize_value(stmt.selector)

        # If selector is constant, keep only that case (level 2+) -- unless
        # a discarded case declares a label a GOTO still names.
        if (
            self.opt_level >= 2
            and isinstance(unwrap_paren(opt_selector), P.NumberLiteral)
            and not any(self._contains_label(c) for c in stmt.items)
        ):
            case_idx = number_value(unwrap_paren(opt_selector))
            if 0 <= case_idx < len(stmt.items):
                self.stats.dead_code_eliminated += 1
                return self._optimize_stmt(stmt.items[case_idx])

        # Exactly one case runs, so each is optimized from the state at the
        # DO CASE and nothing any of them establishes survives the join.
        entry = self._snapshot_flow_state()
        new_cases: list = []
        for c in stmt.items:
            self._restore_flow_state(entry)
            new_cases.append(self._optimize_stmt(c))
        self._restore_flow_state(entry)
        self._invalidate_modified(list(stmt.items))
        # Drop None survivors by replacing with NullStmt so positional
        # case indices stay aligned with the source.
        new_cases = [c if c is not None else P.NullStmt(pos=stmt.pos) for c in new_cases]
        return P.DoCaseBlock(
            selector=opt_selector,
            items=new_cases,
            end_label=stmt.end_label,
            pos=stmt.pos,
        )

    # ---- expression optimization ------------------------------------------

    def _optimize_expr(self, expr):
        """Optimize a typed expression node. ``None`` returns ``None``."""
        if expr is None:
            return None

        # Transparently peel ParenExpr — none of the dispatch logic
        # cares about the wrapper, and folding through it lets the
        # synthetic-literal builders return a bare NumberLiteral.
        if isinstance(expr, P.ParenExpr):
            inner = self._optimize_expr(expr.inner)
            # If folding reduced inner to a literal, drop the wrapper.
            if isinstance(inner, (P.NumberLiteral, P.StringLiteral, P.Identifier)):
                return inner
            return P.ParenExpr(inner=inner, pos=expr.pos)

        if isinstance(expr, P.NumberLiteral):
            return expr

        if isinstance(expr, P.StringLiteral):
            return expr

        if isinstance(expr, P.Identifier):
            name = ident_text(expr.name)
            # Constant propagation (level 1+).
            if self.opt_level >= 1 and name in self.constants:
                self.stats.constants_folded += 1
                value, vtype, derived = self.constants[name]
                return make_typed_const(value, vtype, expr.pos, derived=derived)
            # Copy propagation (level 3).
            if self.opt_level >= 3 and name in self.copies:
                self.stats.copies_propagated += 1
                return make_identifier(self.copies[name], pos=expr.pos)
            return expr

        if isinstance(expr, P.BinaryOp):
            return self._optimize_binary(expr)

        if isinstance(expr, P.UnaryOp):
            return self._optimize_unary(expr)

        if isinstance(expr, P.MemberAccess):
            opt_base = self._optimize_expr(expr.base)
            return P.MemberAccess(base=opt_base, member=expr.member, pos=expr.pos)

        if isinstance(expr, P.Call):
            opt_callee = self._optimize_expr(expr.callee)
            callee = unwrap_paren(opt_callee)
            if (isinstance(callee, P.Identifier)
                    and ident_text(callee.name).upper() in ("SIZE", "LENGTH", "LAST")
                    and self._is_builtin(ident_text(callee.name))):
                # The operand names a variable; it is not a value. After
                # `b0 = 5', SIZE(b0) became SIZE(5), which does not compile.
                return P.Call(callee=opt_callee, args=list(expr.args), pos=expr.pos)
            opt_args = [self._optimize_expr(a) for a in expr.args]
            # A subscript, an argument, or a built-in's operand is converted
            # to the type it is used as -- except the pattern of SCL and SCR,
            # whose type is the result's.
            callee = unwrap_paren(opt_callee)
            keep_first = (isinstance(callee, P.Identifier)
                          and ident_text(callee.name).upper() in PATTERN_TYPED_BUILTINS
                          and self._is_builtin(ident_text(callee.name)))
            opt_args = [a if (keep_first and i == 0) else untyped_root(a)
                        for i, a in enumerate(opt_args)]

            # Optimize built-in calls with constant args.
            if self.opt_level >= 1 and isinstance(unwrap_paren(opt_callee), P.Identifier):
                name = ident_text(unwrap_paren(opt_callee).name)
                result = self._optimize_builtin_call(name, opt_args, expr.pos)
                if result is not None:
                    return result

            return P.Call(callee=opt_callee, args=opt_args, pos=expr.pos)

        if isinstance(expr, P.CallNoArgs):
            opt_callee = self._optimize_expr(expr.callee)
            return P.CallNoArgs(callee=opt_callee, pos=expr.pos)

        if isinstance(expr, P.LocationOf):
            # `.x' names a place, as an assignment target does: propagating
            # x's value into it turned `p = .x' after `x = 5' into `p = 5'.
            opt_operand = self._optimize_target(expr.operand)
            return P.LocationOf(operand=opt_operand, pos=expr.pos)

        if isinstance(expr, P.LocationOfString):
            return expr

        if isinstance(expr, P.LocationOfList):
            opt_values = [self._optimize_expr(v) for v in expr.values]
            return P.LocationOfList(values=opt_values, pos=expr.pos)

        if isinstance(expr, P.EmbeddedAssign):
            # The target of an embedded assignment is a place, not a value, so
            # it goes through _optimize_target like any other lvalue.  Folding
            # it rewrote `(k := 7)' after `k = 5' into a store through the
            # literal 5 - on CP/M that is the BDOS entry vector.
            opt_target = self._optimize_target(expr.target)
            opt_value = self._optimize_expr(expr.value)
            # An embedded assignment changes its target just as a statement
            # assignment does, so the facts recorded about it stop being true
            # here.  Without this, `q = (k := 7)' left the table still saying
            # k is 5, and a later `pc(k)' was handed the stale 5.
            # Nothing new is recorded: the rest of the expression may be
            # evaluated before or after the store, in whatever order code
            # generation picks.
            t = unwrap_paren(opt_target)
            if isinstance(t, P.Identifier):
                name = ident_text(t.name)
                self.modified_vars.add(name)
                self._forget([name])
            if not self._is_plain_store(opt_target):
                self._reset_flow_state()
            return P.EmbeddedAssign(target=opt_target, value=opt_value, pos=expr.pos)

        return expr

    # Operations whose carry a flag-reading region can observe.
    _ARITH = frozenset({BinaryOpKind.ADD, BinaryOpKind.SUB, BinaryOpKind.MUL,
                        BinaryOpKind.DIV, BinaryOpKind.MOD,
                        BinaryOpKind.PLUS, BinaryOpKind.MINUS})

    def _optimize_binary(self, expr: P.BinaryOp):
        """Optimize a binary expression."""
        kind = binop_kind(expr)
        left = self._optimize_expr(expr.left)
        right = self._optimize_expr(expr.right)

        # In a region that reads a flag (CARRY, PLUS...), an arithmetic
        # operation's carry is observable, so the operation has to survive
        # as written: no folding, no rewriting.
        frozen = self.flag_sensitive and kind in self._ARITH

        # Constant folding (level 1+), by PL/M-80's typing rules.
        if self.opt_level >= 1 and not frozen:
            folded = self._fold_binary(kind, left, right, expr.pos)
            if folded is not None:
                self.stats.constants_folded += 1
                return folded

        if self.restricted:
            return make_binary(kind, left, right, pos=expr.pos)

        # Strength reduction (level 2+).
        if self.opt_level >= 2 and not frozen:
            reduced = self._strength_reduce(kind, left, right, expr.pos)
            if reduced is not None:
                self.stats.strength_reductions += 1
                return reduced

        # Algebraic simplifications (level 1+).
        if self.opt_level >= 1 and not frozen:
            simplified = self._algebraic_simplify(kind, left, right, expr.pos)
            if simplified is not None:
                self.stats.algebraic_simplifications += 1
                return simplified

        # Boolean/comparison simplifications (level 2+).
        if self.opt_level >= 2 and not self.flag_sensitive:
            bool_simp = self._boolean_simplify(kind, left, right, expr.pos)
            if bool_simp is not None:
                self.stats.boolean_simplifications += 1
                return bool_simp

        # Commutative normalization for better CSE (level 3).
        if self.opt_level >= 3:
            left, right = self._normalize_commutative(kind, left, right)

        result_expr = make_binary(kind, left, right, pos=expr.pos)

        # CSE: check if we've seen this expression before (level 3).
        if self.opt_level >= 3:
            key = _expr_key(result_expr)
            if key is not None:
                if key in self.cse_cache:
                    self.stats.cse_eliminations += 1
                    cached_expr = self.cse_cache[key][1]
                    return deepcopy(cached_expr)
                else:
                    self.cse_cache[key] = (f"??CSE{self.cse_counter}", result_expr)
                    self.expr_vars[key] = _get_expr_vars(result_expr)
                    self.cse_counter += 1

        return result_expr

    def _fold_binary(self, kind: BinaryOpKind, left, right, pos):
        """``left kind right`` as a constant, if both operands are.

        Typed, so `200 + 100' is the BYTE 44 and `7 MOD 0' the ADDRESS 7;
        but a DATA / INITIAL / AT value is a restricted expression, folded
        as a plain 16-bit number (see _optimize_decl_item).
        """
        if self.restricted:
            if not (_is_number(left) and _is_number(right)):
                return None
            value = self._eval_binary_const(kind, _num_value(left), _num_value(right))
            return None if value is None else make_number_literal(value, pos=pos)
        lc, rc = typed_const(left), typed_const(right)
        if lc is None or rc is None:
            return None
        folded = fold_binary(kind, lc[0], lc[1], rc[0], rc[1])
        if folded is None:
            return None
        return make_typed_const(folded[0], folded[1], pos,
                                derived=is_derived(left) or is_derived(right))

    def _optimize_unary(self, expr: P.UnaryOp):
        """Optimize a unary expression."""
        kind = unop_kind(expr)
        operand = self._optimize_expr(expr.operand)

        # Constant folding: `-x' and `NOT x' keep x's type, so NOT 7 is the
        # BYTE 0F8H and -1 the BYTE 0FFH (4.2.2).
        if self.opt_level >= 1:
            if self.restricted:
                if _is_number(operand):
                    self.stats.constants_folded += 1
                    return make_number_literal(
                        self._eval_unary_const(kind, _num_value(operand)), pos=expr.pos)
            else:
                c = typed_const(operand)
                if c is not None:
                    self.stats.constants_folded += 1
                    value, vtype = fold_unary(kind, c[0], c[1])
                    return make_typed_const(value, vtype, expr.pos,
                                            derived=is_derived(operand))

        # Double negation elimination: -(-x) and NOT NOT x are x, in x's
        # own width.
        inner = unwrap_paren(operand)
        if kind == UnaryOpKind.NEG and isinstance(inner, P.UnaryOp):
            if unop_kind(inner) == UnaryOpKind.NEG:
                self.stats.algebraic_simplifications += 1
                return inner.operand

        # NOT NOT elimination.
        if kind == UnaryOpKind.NOT and isinstance(inner, P.UnaryOp):
            if unop_kind(inner) == UnaryOpKind.NOT:
                self.stats.algebraic_simplifications += 1
                return inner.operand

        return make_unary(kind, operand, pos=expr.pos)

    def _eval_binary_const(self, kind: BinaryOpKind, left: int, right: int) -> int | None:
        """A binary operation on constants as plain 16-bit numbers.

        How a restricted expression (a DATA, INITIAL or AT value) is
        folded. An executable expression is folded by type: see
        plm_types.fold_binary.
        """
        mask = 0xFFFF
        if kind == BinaryOpKind.ADD:
            return (left + right) & mask
        if kind == BinaryOpKind.SUB:
            return (left - right) & mask
        if kind == BinaryOpKind.MUL:
            return (left * right) & mask
        # PL/M-80's divide, zero divisor included: x / 0 is 0FFFFH and
        # x MOD 0 is x, which is what ??div16 / ??mod16 give at run time.
        if kind == BinaryOpKind.DIV:
            return plm_div(left, right)
        if kind == BinaryOpKind.MOD:
            return plm_mod(left, right)
        if kind == BinaryOpKind.AND:
            return left & right
        if kind == BinaryOpKind.OR:
            return left | right
        if kind == BinaryOpKind.XOR:
            return left ^ right
        return None

    def _eval_unary_const(self, kind: UnaryOpKind, value: int) -> int:
        """A unary operation on a constant as a plain 16-bit number."""
        if kind == UnaryOpKind.NEG:
            return (-value) & 0xFFFF
        return (~value) & 0xFFFF

    def _strength_reduce(
        self, kind: BinaryOpKind, left, right, pos
    ):
        """Apply strength reduction transformations.

        Power-of-2 multiply / divide / modulo collapse into shift /
        mask forms expressed as builtin calls (``SHL`` / ``SHR``) or
        a bitwise AND. Each is an ADDRESS, as the product, quotient or
        remainder it replaces is even of BYTE operands.
        """
        rc = typed_const(right)
        if rc is None:
            return None
        shift = self._log2_if_power_of_2(rc[0])
        if shift is None:
            return None

        # Multiply by power of 2 -> shift left.
        if kind == BinaryOpKind.MUL:
            if shift == 0:
                return self._as_address(left, pos)
            if shift == 1:
                # x * 2 -> x + x, for an ADDRESS x that can be evaluated
                # twice. A BYTE `x + x' would wrap; code generation doubles
                # a zero-extended BYTE for `x * 2' anyway.
                if self._type_of(left) is ADDRESS and self._is_side_effect_free(left):
                    return make_binary(BinaryOpKind.ADD, left, deepcopy(left), pos=pos)
                return None
            # x * 2^n -> SHL(x, n), which is an ADDRESS (see plm_types).
            return P.Call(
                callee=make_identifier("SHL", pos=pos),
                args=[left, make_number_literal(shift, pos=pos)],
                pos=pos,
            )

        # Divide by power of 2 -> shift right; SHR is an ADDRESS, as the
        # quotient is.
        if kind == BinaryOpKind.DIV:
            if shift == 0:
                return self._as_address(left, pos)
            return self._as_address(P.Call(
                callee=make_identifier("SHR", pos=pos),
                args=[left, make_number_literal(shift, pos=pos)],
                pos=pos,
            ), pos)

        # Modulo by power of 2 -> AND with (2^n - 1), kept ADDRESS: a BYTE
        # `x AND 7' in place of `x MOD 8' would make `(x MOD 8) + 0FFH' wrap.
        if kind == BinaryOpKind.MOD:
            mask = rc[0] - 1
            return self._as_address(make_binary(
                BinaryOpKind.AND,
                left,
                make_number_literal(mask, pos=pos),
                pos=pos,
            ), pos)

        return None

    def _typed_zero(self, t: "DataType | None", pos):
        """The constant 0 of type ``t`` (None when ``t`` is not known)."""
        if t is None:
            return None
        return make_typed_const(0, t, pos, derived=True)

    def _algebraic_simplify(
        self, kind: BinaryOpKind, left, right, pos
    ):
        """Apply algebraic simplifications.

        Folds identities (``x + 0``, ``x * 1``, ``x - x``), constant
        absorption (``x * 0``, ``x AND 0``), and constant re-association
        on nested add/sub chains -- each giving the type the operation
        had: ``b + DOUBLE(0)`` is ``DOUBLE(b)``, not ``b``, and ``w - w`` is
        the ADDRESS 0.
        """
        lc, rc = typed_const(left), typed_const(right)
        result_type = binary_type(kind, self._type_of(left), self._type_of(right))

        def is_value(c, v) -> bool:
            return c is not None and c[0] == v

        def identity(x, c):
            """``x`` combined with an identity element ``c``: x itself,
            unless ``c`` is an ADDRESS that widened it."""
            return self._as_address(x, pos) if c[1] is ADDRESS else x

        same_ident = (_is_ident(left) and _is_ident(right)
                      and _ident_name(left) == _ident_name(right)
                      and self._is_side_effect_free(left))

        # x + 0 = x, 0 + x = x; the same for OR and XOR; x - 0 = x
        if kind in (BinaryOpKind.ADD, BinaryOpKind.OR, BinaryOpKind.XOR, BinaryOpKind.SUB):
            if is_value(rc, 0):
                return identity(left, rc)
            if is_value(lc, 0) and kind != BinaryOpKind.SUB:
                return identity(right, lc)

        # x - x = 0; x XOR x = 0
        if kind in (BinaryOpKind.SUB, BinaryOpKind.XOR) and same_ident:
            return self._typed_zero(result_type, pos)

        # x * 1 = x, 1 * x = x; x * 0 = 0 -- a product is an ADDRESS
        if kind == BinaryOpKind.MUL:
            if is_value(rc, 1):
                return self._as_address(left, pos)
            if is_value(lc, 1):
                return self._as_address(right, pos)
            if is_value(rc, 0) and self._is_side_effect_free(left):
                return self._typed_zero(ADDRESS, pos)
            if is_value(lc, 0) and self._is_side_effect_free(right):
                return self._typed_zero(ADDRESS, pos)

        # x / 1 = x, as an ADDRESS
        if kind == BinaryOpKind.DIV and is_value(rc, 1):
            return self._as_address(left, pos)

        # x AND 0 = 0, x AND 0FFFFH = x (an ADDRESS)
        if kind == BinaryOpKind.AND:
            for x, c in ((left, rc), (right, lc)):
                if is_value(c, 0) and self._is_side_effect_free(x):
                    return self._typed_zero(result_type, pos)
                if is_value(c, 0xFFFF):
                    return self._as_address(x, pos)

        # x OR 0FFFFH = 0FFFFH
        if kind == BinaryOpKind.OR:
            for x, c in ((left, rc), (right, lc)):
                if is_value(c, 0xFFFF) and self._is_side_effect_free(x):
                    return make_typed_const(0xFFFF, ADDRESS, pos, derived=True)

        # x XOR 0FFFFH = NOT x, as an ADDRESS
        if kind == BinaryOpKind.XOR:
            for x, c in ((left, rc), (right, lc)):
                if is_value(c, 0xFFFF):
                    return make_unary(UnaryOpKind.NOT, self._as_address(x, pos), pos=pos)

        # (x + c1) + c2 -> x + (c1 + c2), and the like with SUB.
        if kind in (BinaryOpKind.ADD, BinaryOpKind.SUB) and rc is not None:
            reassociated = self._reassociate(kind, left, rc, pos)
            if reassociated is not None:
                return reassociated

        # x MOD 1 = 0 and 0 MOD x = 0 (0 MOD 0 is 0 too: the remainder of a
        # zero divisor is the dividend), only when the operand dropped does
        # nothing. There is no `0 / x = 0': 0 / 0 is 0FFFFH.
        if kind == BinaryOpKind.MOD:
            dropped = None
            if is_value(rc, 1):
                dropped = left
            elif is_value(lc, 0):
                dropped = right
            if dropped is not None and self._is_side_effect_free(dropped):
                return self._typed_zero(ADDRESS, pos)

        return None

    def _reassociate(self, kind: BinaryOpKind, left, rc, pos):
        """``(x op1 c1) op2 c2`` as ``x op (c)``, when the width is the same
        throughout: a BYTE ``(x + 200) + 100`` wraps twice at 8 bits, which
        ``x + 44`` does too, but if the outer addition is 16-bit the inner
        wrap cannot be folded into it."""
        inner = unwrap_paren(left)
        if not isinstance(inner, P.BinaryOp):
            return None
        ikind = binop_kind(inner)
        if ikind not in (BinaryOpKind.ADD, BinaryOpKind.SUB):
            return None
        ic = typed_const(inner.right)
        if ic is None:
            return None
        x = inner.left
        inner_type = binary_type(ikind, self._type_of(x), ic[1])
        outer_type = binary_type(kind, inner_type, rc[1])
        if inner_type is None or inner_type is not outer_type:
            return None
        # x + d, where d = (+/-c1) + (+/-c2) in the operation's width.
        d = (ic[0] if ikind == BinaryOpKind.ADD else -ic[0])
        d += rc[0] if kind == BinaryOpKind.ADD else -rc[0]
        d = convert(d, outer_type)
        if d == 0:
            return self._as_address(x, pos) if outer_type is ADDRESS else x
        # Prefer the smaller constant: x - 2 rather than x + 0FFFEH.
        neg = convert(-d, outer_type)
        op, c = (BinaryOpKind.ADD, d) if d <= neg else (BinaryOpKind.SUB, neg)
        # `x op c' has to be as wide as the expression it replaces: with an
        # ADDRESS x any literal will do, with a BYTE x the constant carries
        # the width.
        if self._type_of(x) is ADDRESS:
            const = make_number_literal(c, pos=pos)
        else:
            const = make_typed_const(c, outer_type, pos, derived=True)
        return make_binary(op, x, const, pos=pos)

    def _boolean_simplify(
        self, kind: BinaryOpKind, left, right, pos
    ):
        """Apply boolean and comparison simplifications.

        A relation is the BYTE 0FFH or 00H wherever it is used (4.4), so
        `x = x' is the BYTE 0FFH as a value as much as in a condition.
        """
        l_is_id = _is_ident(left)
        r_is_id = _is_ident(right)
        # `x REL x` folds only when evaluating x twice is unobservable: a
        # bare identifier naming a procedure is a parameterless call.
        same_id = (l_is_id and r_is_id
                   and _ident_name(left) == _ident_name(right)
                   and self._is_side_effect_free(left))

        if same_id and kind in RELATIONS:
            holds = kind in (BinaryOpKind.EQ, BinaryOpKind.LE, BinaryOpKind.GE)
            return make_typed_const(0xFF if holds else 0, BYTE, pos, derived=True)

        # (a AND b) AND b -> a AND b (idempotent). Dropping the repeated
        # operand is only sound when evaluating it has no side effect. The
        # type is unchanged: b's type is already in (a AND b)'s.
        if kind in (BinaryOpKind.AND, BinaryOpKind.OR):
            inner = unwrap_paren(left)
            if isinstance(inner, P.BinaryOp) and binop_kind(inner) == kind:
                if r_is_id and self._is_side_effect_free(right):
                    rn = _ident_name(right)
                    if _is_ident(inner.right) and _ident_name(inner.right) == rn:
                        return left
                    if _is_ident(inner.left) and _ident_name(inner.left) == rn:
                        return left

        # x AND x = x; x OR x = x
        if kind in (BinaryOpKind.AND, BinaryOpKind.OR) and same_id:
            return left

        return None

    def _optimize_builtin_call(self, name: str, args: list, pos):
        """Fold a built-in procedure of constant arguments, by type."""
        if len(args) == 0 or not self._is_builtin(name):
            return None
        if self.restricted:
            return self._fold_builtin_untyped(name, args, pos)
        consts = [typed_const(a) for a in args]
        if any(c is None for c in consts):
            return None
        folded = fold_builtin(name, consts)  # type: ignore[arg-type]
        if folded is None:
            return None
        return make_typed_const(folded[0], folded[1], pos,
                                derived=any(is_derived(a) for a in args))

    def _fold_builtin_untyped(self, name: str, args: list, pos):
        """A built-in of constants in a restricted expression, as numbers."""
        if not all(_is_number(a) for a in args):
            return None
        values = [_num_value(a) for a in args]
        name = name.upper()
        if len(values) == 1:
            v = values[0]
            result = {"LOW": v & 0xFF, "HIGH": (v >> 8) & 0xFF, "DOUBLE": v & 0xFFFF}.get(name)
        elif len(values) == 2 and name in ("SHL", "SHR", "ROL", "ROR"):
            folded = fold_builtin(name, [(values[0], ADDRESS if name in ("SHL", "SHR") else BYTE),
                                         (values[1], BYTE)])
            result = None if folded is None else folded[0]
        else:
            result = None
        return None if result is None else make_number_literal(result, pos=pos)

    def _log2_if_power_of_2(self, n: int) -> int | None:
        """Return log2(n) if n is a power of 2, else None."""
        if n <= 0:
            return None
        if n & (n - 1) != 0:
            return None
        return n.bit_length() - 1


def optimize_ast(
    module: P.Module,
    opt_level: int = 2,
    optimize_for: OptimizeFor = OptimizeFor.BALANCED,
) -> P.Module:
    """Convenience function to optimize a module's typed AST."""
    optimizer = ASTOptimizer(opt_level, optimize_for)
    return optimizer.optimize(module)
