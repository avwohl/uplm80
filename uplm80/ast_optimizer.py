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
"""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

from . import _plm_parser as P
from .ast_view import (
    BinaryOpKind,
    UnaryOpKind,
    binop_kind,
    block_items_split,
    ident_text,
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
    unop_kind,
    unwrap_paren,
)


# Pure built-in functions whose calls are safe to CSE / treat as
# loop-invariant when all their arguments are.
_PURE_BUILTINS = {
    "LOW", "HIGH", "DOUBLE", "SHL", "SHR", "ROL", "ROR",
    "LENGTH", "LAST", "SIZE",
}


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
        # Known constant values for propagation
        self.constants: dict[str, int] = {}
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

    def optimize(self, module: P.Module) -> P.Module:
        """Optimize an entire typed :class:`P.Module`."""
        if self.opt_level == 0:
            return module

        # Multiple passes for iterative improvement
        changed = True
        passes = 0
        max_passes = 5

        while changed and passes < max_passes:
            changed = False
            passes += 1

            new_items: list = []
            for item in module.items:
                opt_item = self._optimize_module_item(item)
                if opt_item is not None:
                    new_items.append(opt_item)
                    if opt_item is not item:
                        changed = True

            module = P.Module(items=new_items, pos=module.pos)

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
        # Level 3: Track inlinable procedures
        if self.opt_level >= 3 and self._is_inlinable(optimized_proc, attrs, local_decls):
            self.inlinable_procs[proc_name(optimized_proc)] = optimized_proc
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
        if val is not None:
            self.constants[name] = val
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
                if isinstance(stmt, P.LabeledStmt):
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
        without being read between is a dead store.
        """
        if self.opt_level < 3:
            return stmts

        result: list = []
        i = 0
        while i < len(stmts):
            stmt = stmts[i]

            if isinstance(stmt, P.AssignStmt) and len(stmt.targets) == 1:
                target = unwrap_paren(stmt.targets[0])
                if isinstance(target, P.Identifier):
                    name = ident_text(target.name)
                    if i + 1 < len(stmts):
                        next_stmt = stmts[i + 1]
                        if (
                            isinstance(next_stmt, P.AssignStmt)
                            and len(next_stmt.targets) == 1
                        ):
                            next_t = unwrap_paren(next_stmt.targets[0])
                            if (
                                isinstance(next_t, P.Identifier)
                                and ident_text(next_t.name) == name
                                and name not in _get_expr_vars(next_stmt.value)
                            ):
                                self.stats.dead_stores_eliminated += 1
                                i += 1
                                continue

            result.append(stmt)
            i += 1

        return result

    def _get_modified_vars_in_stmts(self, stmts: list) -> set[str]:
        """Get all variables modified within a list of statements."""
        modified: set[str] = set()

        def visit_stmt(s) -> None:
            if s is None:
                return
            if isinstance(s, P.AssignStmt):
                for target in s.targets:
                    t = unwrap_paren(target)
                    if isinstance(t, P.Identifier):
                        modified.add(ident_text(t.name))
                    elif isinstance(t, P.Call):
                        # PL/M subscript form: ARR(idx) = ...
                        c = unwrap_paren(t.callee)
                        if isinstance(c, P.Identifier):
                            modified.add(ident_text(c.name))
                visit_expr(s.value)
            elif isinstance(s, P.DoBlock):
                _, body = block_items_split(s.items)
                for sub in body:
                    visit_stmt(sub)
            elif isinstance(s, P.DoWhileBlock):
                _, body = block_items_split(s.items)
                for sub in body:
                    visit_stmt(sub)
            elif isinstance(s, (P.DoIterBlock, P.DoIterByBlock)):
                modified.add(ident_text(s.index))
                _, body = block_items_split(s.items)
                for sub in body:
                    visit_stmt(sub)
            elif isinstance(s, P.DoCaseBlock):
                for case in s.items:
                    visit_stmt(case)
            elif isinstance(s, P.IfStmt):
                visit_stmt(s.then_stmt)
            elif isinstance(s, P.IfStmtElse):
                visit_stmt(s.then_stmt)
                visit_stmt(s.else_stmt)
            elif isinstance(s, P.LabeledStmt):
                visit_stmt(s.stmt)
            elif isinstance(s, P.CallStmt):
                # Calls may modify globals - be conservative; just walk args.
                inner = s.callee
                if isinstance(inner, P.Call):
                    for arg in inner.args:
                        visit_expr(arg)

        def visit_expr(e) -> None:
            if e is None:
                return
            e = unwrap_paren(e)
            if isinstance(e, P.EmbeddedAssign):
                t = unwrap_paren(e.target)
                if isinstance(t, P.Identifier):
                    modified.add(ident_text(t.name))
                visit_expr(e.value)
            elif isinstance(e, P.BinaryOp):
                visit_expr(e.left)
                visit_expr(e.right)
            elif isinstance(e, P.UnaryOp):
                visit_expr(e.operand)
            elif isinstance(e, P.Call):
                for arg in e.args:
                    visit_expr(arg)

        for s in stmts:
            visit_stmt(s)

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
        """Check if a procedure is suitable for inlining."""
        if attrs.is_external or attrs.is_reentrant or attrs.interrupt_num is not None:
            return False
        # Don't inline procedures that contain nested procedures
        for d in local_decls:
            if isinstance(d, P.ProcDecl):
                return False
        # Don't inline procedures with local declarations (complex scoping)
        if local_decls:
            return False
        _, body_stmts = proc_local_decls_stmts(proc)
        if self._count_stmts(body_stmts) > 5:
            return False
        if len(proc_param_names(proc)) > 3:
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
        """Normalize operand order for commutative operations to improve CSE."""
        if kind not in self._COMMUTATIVE:
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
            return right, left
        return left, right

    # ---- statement optimization -------------------------------------------

    def _optimize_stmt(self, stmt):
        """Optimize a typed statement. Returns ``None`` to remove it."""
        if stmt is None:
            return None

        if isinstance(stmt, P.AssignStmt):
            opt_value = self._optimize_expr(stmt.value)
            opt_targets = [self._optimize_expr(t) for t in stmt.targets]

            # Track modified variables and invalidate caches
            for target in opt_targets:
                t = unwrap_paren(target)
                if isinstance(t, P.Identifier):
                    name = ident_text(t.name)
                    self.modified_vars.add(name)
                    self.constants.pop(name, None)
                    self._invalidate_cse_for_var(name)
                    self._invalidate_copies_for_var(name)

            # Level 3: Track copies and constants
            if self.opt_level >= 3 and len(opt_targets) == 1:
                t = unwrap_paren(opt_targets[0])
                v = unwrap_paren(opt_value)
                if isinstance(t, P.Identifier):
                    tname = ident_text(t.name)
                    if isinstance(v, P.NumberLiteral):
                        self.constants[tname] = number_value(v)
                    elif isinstance(v, P.Identifier):
                        self.copies[tname] = ident_text(v.name)

            return P.AssignStmt(targets=opt_targets, value=opt_value, pos=stmt.pos)

        if isinstance(stmt, P.CallStmt):
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

            opt_callee = self._optimize_expr(callee_expr)
            opt_args = [self._optimize_expr(a) for a in args]

            # Level 3: Inline small procedures
            if (
                self.opt_level >= 3
                and self.optimize_for != OptimizeFor.SIZE
                and isinstance(unwrap_paren(opt_callee), P.Identifier)
            ):
                name = ident_text(unwrap_paren(opt_callee).name)
                if name in self.inlinable_procs:
                    proc = self.inlinable_procs[name]
                    if len(opt_args) == len(proc_param_names(proc)):
                        inlined = self._inline_procedure(proc, opt_args, stmt.pos)
                        if inlined is not None:
                            self.stats.procedures_inlined += 1
                            return inlined

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

        if isinstance(stmt, P.ReturnStmt):
            return stmt

        if isinstance(stmt, P.ReturnStmtValue):
            opt_value = self._optimize_expr(stmt.value)
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
            opt_inner = self._optimize_stmt(stmt.stmt)
            if opt_inner is None:
                opt_inner = P.NullStmt(pos=stmt.pos)
            return P.LabeledStmt(label=stmt.label, stmt=opt_inner, pos=stmt.pos)

        if isinstance(stmt, P.DeclareStmt):
            return self._optimize_declare_stmt(stmt)

        # P.GotoStmt, P.HaltStmt, P.EnableStmt, P.DisableStmt, P.NullStmt — pass through.
        return stmt

    def _optimize_if(self, stmt):
        """Optimize an IF / IF-ELSE statement.

        Returns a possibly-different shape — folding ``IF 1`` reduces
        to just the then-branch, ``IF 0`` reduces to the else branch
        (or a :class:`P.NullStmt`). Also collapses :class:`P.IfStmtElse`
        whose else-branch optimizes away into :class:`P.IfStmt`.
        """
        opt_cond = self._optimize_expr(stmt.condition)

        # Constant condition elimination (level 2+).
        if self.opt_level >= 2 and isinstance(unwrap_paren(opt_cond), P.NumberLiteral):
            self.stats.dead_code_eliminated += 1
            if number_value(unwrap_paren(opt_cond)) != 0:
                return self._optimize_stmt(stmt.then_stmt)
            if isinstance(stmt, P.IfStmtElse):
                return self._optimize_stmt(stmt.else_stmt)
            return P.NullStmt(pos=stmt.pos)

        opt_then = self._optimize_stmt(stmt.then_stmt)
        if opt_then is None:
            opt_then = P.NullStmt(pos=stmt.pos)

        if isinstance(stmt, P.IfStmtElse):
            opt_else = self._optimize_stmt(stmt.else_stmt)
            if opt_else is None:
                return P.IfStmt(condition=opt_cond, then_stmt=opt_then, pos=stmt.pos)
            return P.IfStmtElse(
                condition=opt_cond,
                then_stmt=opt_then,
                else_stmt=opt_else,
                pos=stmt.pos,
            )
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
        opt_cond = self._optimize_expr(stmt.condition)

        # DO WHILE 0 never executes.
        if self.opt_level >= 2 and isinstance(unwrap_paren(opt_cond), P.NumberLiteral):
            if number_value(unwrap_paren(opt_cond)) == 0:
                self.stats.dead_code_eliminated += 1
                return P.NullStmt(pos=stmt.pos)

        # Level 3: Check for loop-invariant subexpressions in condition.
        if self.opt_level >= 3:
            _, body_stmts = block_items_split(stmt.items)
            modified_vars = self._get_modified_vars_in_stmts(body_stmts)
            self._cache_invariant_exprs(opt_cond, modified_vars)

        new_items = self._optimize_block_items(stmt.items)
        new_items = self._eliminate_unreachable_in_items(new_items)
        return P.DoWhileBlock(
            condition=opt_cond,
            items=new_items,
            end_label=stmt.end_label,
            pos=stmt.pos,
        )

    def _optimize_do_iter(self, stmt):
        """Optimize a ``DO I = start TO bound [BY step] ... END`` block.

        ``stmt`` is either :class:`P.DoIterBlock` (no BY) or
        :class:`P.DoIterByBlock` (with explicit step). Both share the
        same body / bound layout; the only difference is the presence
        of ``stmt.step``.
        """
        is_by = isinstance(stmt, P.DoIterByBlock)
        opt_start = self._optimize_expr(stmt.start)
        opt_bound = self._optimize_expr(stmt.bound)
        opt_step = self._optimize_expr(stmt.step) if is_by else None

        # Check for empty loop (start > bound with positive step).
        if (
            self.opt_level >= 2
            and _is_number(opt_start)
            and _is_number(opt_bound)
        ):
            step_val = 1
            if opt_step is not None and _is_number(opt_step):
                step_val = _num_value(opt_step)
            if step_val > 0 and _num_value(opt_start) > _num_value(opt_bound):
                self.stats.dead_code_eliminated += 1
                return P.NullStmt(pos=stmt.pos)

        # Level 3: Loop unrolling for small constant-bound loops.
        if (
            self.opt_level >= 3
            and self.optimize_for != OptimizeFor.SIZE
            and _is_number(opt_start)
            and _is_number(opt_bound)
        ):
            step_val = 1
            if opt_step is not None and _is_number(opt_step):
                step_val = _num_value(opt_step)
            if step_val > 0:
                start_v = _num_value(opt_start)
                bound_v = _num_value(opt_bound)
                iterations = (bound_v - start_v) // step_val + 1
                max_iter = 4 if self.optimize_for == OptimizeFor.SPEED else 2
                _, body_stmts = block_items_split(stmt.items)
                if 1 <= iterations <= max_iter and len(body_stmts) <= 3:
                    index_name = ident_text(stmt.index)
                    unrolled: list = []
                    for i in range(iterations):
                        val = start_v + i * step_val
                        unrolled.append(P.AssignStmt(
                            targets=[make_identifier(index_name, pos=stmt.pos)],
                            value=make_number_literal(val, pos=stmt.pos),
                            pos=stmt.pos,
                        ))
                        for s in body_stmts:
                            unrolled.append(deepcopy(s))
                    self.stats.loops_unrolled += 1
                    block = P.DoBlock(
                        items=unrolled, end_label=stmt.end_label, pos=stmt.pos
                    )
                    return self._optimize_stmt(block)

        # Level 3: Cache loop-invariant bound expressions.
        if self.opt_level >= 3:
            _, body_stmts = block_items_split(stmt.items)
            modified_vars = self._get_modified_vars_in_stmts(body_stmts)
            modified_vars.add(ident_text(stmt.index))
            self._cache_invariant_exprs(opt_bound, modified_vars)
            if opt_step is not None:
                self._cache_invariant_exprs(opt_step, modified_vars)

        new_items = self._optimize_block_items(stmt.items)
        new_items = self._eliminate_unreachable_in_items(new_items)

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

    def _optimize_do_case(self, stmt: P.DoCaseBlock):
        """Optimize a ``DO CASE selector ... END`` block."""
        opt_selector = self._optimize_expr(stmt.selector)

        # If selector is constant, keep only that case (level 2+).
        if self.opt_level >= 2 and isinstance(unwrap_paren(opt_selector), P.NumberLiteral):
            case_idx = number_value(unwrap_paren(opt_selector))
            if 0 <= case_idx < len(stmt.items):
                self.stats.dead_code_eliminated += 1
                return self._optimize_stmt(stmt.items[case_idx])

        new_cases: list = [self._optimize_stmt(c) for c in stmt.items]
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
                return make_number_literal(self.constants[name], pos=expr.pos)
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
            opt_args = [self._optimize_expr(a) for a in expr.args]

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
            opt_operand = self._optimize_expr(expr.operand)
            return P.LocationOf(operand=opt_operand, pos=expr.pos)

        if isinstance(expr, P.LocationOfString):
            return expr

        if isinstance(expr, P.LocationOfList):
            opt_values = [self._optimize_expr(v) for v in expr.values]
            return P.LocationOfList(values=opt_values, pos=expr.pos)

        if isinstance(expr, P.EmbeddedAssign):
            opt_target = self._optimize_expr(expr.target)
            opt_value = self._optimize_expr(expr.value)
            return P.EmbeddedAssign(target=opt_target, value=opt_value, pos=expr.pos)

        return expr

    def _optimize_binary(self, expr: P.BinaryOp):
        """Optimize a binary expression."""
        kind = binop_kind(expr)
        left = self._optimize_expr(expr.left)
        right = self._optimize_expr(expr.right)

        # Constant folding (level 1+).
        if (
            self.opt_level >= 1
            and _is_number(left)
            and _is_number(right)
        ):
            result = self._eval_binary_const(kind, _num_value(left), _num_value(right))
            if result is not None:
                self.stats.constants_folded += 1
                return make_number_literal(result, pos=expr.pos)

        # Strength reduction (level 2+).
        if self.opt_level >= 2:
            reduced = self._strength_reduce(kind, left, right, expr.pos)
            if reduced is not None:
                self.stats.strength_reductions += 1
                return reduced

        # Algebraic simplifications (level 1+).
        if self.opt_level >= 1:
            simplified = self._algebraic_simplify(kind, left, right, expr.pos)
            if simplified is not None:
                self.stats.algebraic_simplifications += 1
                return simplified

        # Boolean/comparison simplifications (level 2+).
        if self.opt_level >= 2:
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

    def _optimize_unary(self, expr: P.UnaryOp):
        """Optimize a unary expression."""
        kind = unop_kind(expr)
        operand = self._optimize_expr(expr.operand)

        # Constant folding.
        if self.opt_level >= 1 and _is_number(operand):
            result = self._eval_unary_const(kind, _num_value(operand))
            if result is not None:
                self.stats.constants_folded += 1
                return make_number_literal(result, pos=expr.pos)

        # Double negation elimination.
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
        """Evaluate a binary operation on constants (16-bit unsigned PL/M semantics)."""
        mask = 0xFFFF
        try:
            if kind == BinaryOpKind.ADD:
                return (left + right) & mask
            elif kind == BinaryOpKind.SUB:
                return (left - right) & mask
            elif kind == BinaryOpKind.MUL:
                return (left * right) & mask
            elif kind == BinaryOpKind.DIV:
                if right == 0:
                    return None
                return (left // right) & mask
            elif kind == BinaryOpKind.MOD:
                if right == 0:
                    return None
                return (left % right) & mask
            elif kind == BinaryOpKind.AND:
                return left & right
            elif kind == BinaryOpKind.OR:
                return left | right
            elif kind == BinaryOpKind.XOR:
                return left ^ right
            elif kind == BinaryOpKind.EQ:
                return 0xFFFF if left == right else 0
            elif kind == BinaryOpKind.NE:
                return 0xFFFF if left != right else 0
            elif kind == BinaryOpKind.LT:
                return 0xFFFF if left < right else 0
            elif kind == BinaryOpKind.GT:
                return 0xFFFF if left > right else 0
            elif kind == BinaryOpKind.LE:
                return 0xFFFF if left <= right else 0
            elif kind == BinaryOpKind.GE:
                return 0xFFFF if left >= right else 0
            # PLUS / MINUS (carry-aware) — don't fold at AST level; the
            # codegen lowering depends on the runtime carry chain.
        except (ZeroDivisionError, OverflowError):
            return None

        return None

    def _eval_unary_const(self, kind: UnaryOpKind, value: int) -> int | None:
        """Evaluate a unary operation on a constant."""
        mask = 0xFFFF
        if kind == UnaryOpKind.NEG:
            return (-value) & mask
        elif kind == UnaryOpKind.NOT:
            return (~value) & mask
        return None

    def _strength_reduce(
        self, kind: BinaryOpKind, left, right, pos
    ):
        """Apply strength reduction transformations.

        Power-of-2 multiply / divide / modulo collapse into shift /
        mask forms expressed as builtin calls (``SHL`` / ``SHR``) or
        a bitwise AND.
        """
        # Multiply by power of 2 -> shift left.
        if kind == BinaryOpKind.MUL and _is_number(right):
            r_val = _num_value(right)
            shift = self._log2_if_power_of_2(r_val)
            if shift is not None:
                if shift == 0:
                    if _is_number(left) and _num_value(left) == 0:
                        return make_number_literal(0, pos=pos)
                    return left
                if shift == 1:
                    # x * 2 -> x + x
                    return make_binary(BinaryOpKind.ADD, left, deepcopy(left), pos=pos)
                # x * 2^n -> SHL(x, n)
                return P.Call(
                    callee=make_identifier("SHL", pos=pos),
                    args=[left, make_number_literal(shift, pos=pos)],
                    pos=pos,
                )

        # Divide by power of 2 -> shift right.
        if kind == BinaryOpKind.DIV and _is_number(right):
            r_val = _num_value(right)
            shift = self._log2_if_power_of_2(r_val)
            if shift is not None:
                if shift == 0:
                    return left
                return P.Call(
                    callee=make_identifier("SHR", pos=pos),
                    args=[left, make_number_literal(shift, pos=pos)],
                    pos=pos,
                )

        # Modulo by power of 2 -> AND with (2^n - 1).
        if kind == BinaryOpKind.MOD and _is_number(right):
            r_val = _num_value(right)
            shift = self._log2_if_power_of_2(r_val)
            if shift is not None:
                mask = r_val - 1
                return make_binary(
                    BinaryOpKind.AND,
                    left,
                    make_number_literal(mask, pos=pos),
                    pos=pos,
                )

        return None

    def _algebraic_simplify(
        self, kind: BinaryOpKind, left, right, pos
    ):
        """Apply algebraic simplifications.

        Folds identities (``x + 0``, ``x * 1``, ``x - x``), constant
        absorption (``x * 0``, ``x AND 0``), and constant re-association
        on nested add/sub chains.
        """
        # x + 0 = x, 0 + x = x
        if kind == BinaryOpKind.ADD:
            if _is_number(right) and _num_value(right) == 0:
                return left
            if _is_number(left) and _num_value(left) == 0:
                return right

        # x - 0 = x; x - x = 0
        if kind == BinaryOpKind.SUB:
            if _is_number(right) and _num_value(right) == 0:
                return left
            if _is_ident(left) and _is_ident(right) and _ident_name(left) == _ident_name(right):
                return make_number_literal(0, pos=pos)

        # x * 1 = x, 1 * x = x; x * 0 = 0
        if kind == BinaryOpKind.MUL:
            if _is_number(right) and _num_value(right) == 1:
                return left
            if _is_number(left) and _num_value(left) == 1:
                return right
            if _is_number(right) and _num_value(right) == 0:
                return make_number_literal(0, pos=pos)
            if _is_number(left) and _num_value(left) == 0:
                return make_number_literal(0, pos=pos)

        # x / 1 = x
        if kind == BinaryOpKind.DIV:
            if _is_number(right) and _num_value(right) == 1:
                return left

        # x AND 0 = 0, x AND FFFF = x
        if kind == BinaryOpKind.AND:
            if _is_number(right):
                rv = _num_value(right)
                if rv == 0:
                    return make_number_literal(0, pos=pos)
                if rv == 0xFFFF:
                    return left
            if _is_number(left):
                lv = _num_value(left)
                if lv == 0:
                    return make_number_literal(0, pos=pos)
                if lv == 0xFFFF:
                    return right

        # x OR 0 = x, x OR FFFF = FFFF
        if kind == BinaryOpKind.OR:
            if _is_number(right):
                rv = _num_value(right)
                if rv == 0:
                    return left
                if rv == 0xFFFF:
                    return make_number_literal(0xFFFF, pos=pos)
            if _is_number(left):
                lv = _num_value(left)
                if lv == 0:
                    return right
                if lv == 0xFFFF:
                    return make_number_literal(0xFFFF, pos=pos)

        # x XOR 0 = x; x XOR x = 0; x XOR FFFF = NOT x
        if kind == BinaryOpKind.XOR:
            if _is_number(right) and _num_value(right) == 0:
                return left
            if _is_number(left) and _num_value(left) == 0:
                return right
            if _is_ident(left) and _is_ident(right) and _ident_name(left) == _ident_name(right):
                return make_number_literal(0, pos=pos)
            if _is_number(right) and _num_value(right) == 0xFFFF:
                return make_unary(UnaryOpKind.NOT, left, pos=pos)
            if _is_number(left) and _num_value(left) == 0xFFFF:
                return make_unary(UnaryOpKind.NOT, right, pos=pos)

        # (x + c1) + c2 -> x + (c1 + c2); (x - c1) + c2 -> x + (c2 - c1)
        if kind == BinaryOpKind.ADD and _is_number(right):
            inner = unwrap_paren(left)
            if isinstance(inner, P.BinaryOp):
                ikind = binop_kind(inner)
                if ikind == BinaryOpKind.ADD and _is_number(inner.right):
                    new_const = (_num_value(inner.right) + _num_value(right)) & 0xFFFF
                    return make_binary(
                        BinaryOpKind.ADD,
                        inner.left,
                        make_number_literal(new_const, pos=pos),
                        pos=pos,
                    )
                if ikind == BinaryOpKind.SUB and _is_number(inner.right):
                    new_const = (_num_value(right) - _num_value(inner.right)) & 0xFFFF
                    if new_const == 0:
                        return inner.left
                    return make_binary(
                        BinaryOpKind.ADD,
                        inner.left,
                        make_number_literal(new_const, pos=pos),
                        pos=pos,
                    )

        # (x - c1) - c2 -> x - (c1 + c2); (x + c1) - c2 -> ...
        if kind == BinaryOpKind.SUB and _is_number(right):
            inner = unwrap_paren(left)
            if isinstance(inner, P.BinaryOp):
                ikind = binop_kind(inner)
                if ikind == BinaryOpKind.SUB and _is_number(inner.right):
                    new_const = (_num_value(inner.right) + _num_value(right)) & 0xFFFF
                    return make_binary(
                        BinaryOpKind.SUB,
                        inner.left,
                        make_number_literal(new_const, pos=pos),
                        pos=pos,
                    )
                if ikind == BinaryOpKind.ADD and _is_number(inner.right):
                    diff = _num_value(inner.right) - _num_value(right)
                    if diff == 0:
                        return inner.left
                    if diff > 0:
                        return make_binary(
                            BinaryOpKind.ADD,
                            inner.left,
                            make_number_literal(diff & 0xFFFF, pos=pos),
                            pos=pos,
                        )
                    else:
                        return make_binary(
                            BinaryOpKind.SUB,
                            inner.left,
                            make_number_literal((-diff) & 0xFFFF, pos=pos),
                            pos=pos,
                        )

        # x MOD 1 = 0
        if kind == BinaryOpKind.MOD:
            if _is_number(right) and _num_value(right) == 1:
                return make_number_literal(0, pos=pos)

        # 0 / x = 0 (unless x = 0, but compile-time we can't check)
        if kind == BinaryOpKind.DIV:
            if _is_number(left) and _num_value(left) == 0:
                return make_number_literal(0, pos=pos)

        # 0 MOD x = 0
        if kind == BinaryOpKind.MOD:
            if _is_number(left) and _num_value(left) == 0:
                return make_number_literal(0, pos=pos)

        return None

    def _boolean_simplify(
        self, kind: BinaryOpKind, left, right, pos
    ):
        """Apply boolean and comparison simplifications."""
        l_is_id = _is_ident(left)
        r_is_id = _is_ident(right)
        same_id = l_is_id and r_is_id and _ident_name(left) == _ident_name(right)

        if kind == BinaryOpKind.EQ:
            if same_id:
                return make_number_literal(0xFFFF, pos=pos)
            if _is_number(left) and _is_number(right):
                return make_number_literal(
                    0xFFFF if _num_value(left) == _num_value(right) else 0,
                    pos=pos,
                )

        if kind == BinaryOpKind.NE and same_id:
            return make_number_literal(0, pos=pos)
        if kind == BinaryOpKind.LT and same_id:
            return make_number_literal(0, pos=pos)
        if kind == BinaryOpKind.GT and same_id:
            return make_number_literal(0, pos=pos)
        if kind == BinaryOpKind.LE and same_id:
            return make_number_literal(0xFFFF, pos=pos)
        if kind == BinaryOpKind.GE and same_id:
            return make_number_literal(0xFFFF, pos=pos)

        # (a AND b) AND b -> a AND b (idempotent)
        if kind == BinaryOpKind.AND:
            inner = unwrap_paren(left)
            if isinstance(inner, P.BinaryOp) and binop_kind(inner) == BinaryOpKind.AND:
                if r_is_id:
                    rn = _ident_name(right)
                    if _is_ident(inner.right) and _ident_name(inner.right) == rn:
                        return left
                    if _is_ident(inner.left) and _ident_name(inner.left) == rn:
                        return left

        # (a OR b) OR b -> a OR b (idempotent)
        if kind == BinaryOpKind.OR:
            inner = unwrap_paren(left)
            if isinstance(inner, P.BinaryOp) and binop_kind(inner) == BinaryOpKind.OR:
                if r_is_id:
                    rn = _ident_name(right)
                    if _is_ident(inner.right) and _ident_name(inner.right) == rn:
                        return left
                    if _is_ident(inner.left) and _ident_name(inner.left) == rn:
                        return left

        # x AND x = x; x OR x = x
        if kind == BinaryOpKind.AND and same_id:
            return left
        if kind == BinaryOpKind.OR and same_id:
            return left

        return None

    def _optimize_builtin_call(self, name: str, args: list, pos):
        """Optimize calls to built-in functions with constant args."""
        if len(args) == 0:
            return None
        a0 = unwrap_paren(args[0])

        # LOW(const) -> const & 0xFF
        if name == "LOW" and isinstance(a0, P.NumberLiteral):
            return make_number_literal(number_value(a0) & 0xFF, pos=pos)

        # HIGH(const) -> (const >> 8) & 0xFF
        if name == "HIGH" and isinstance(a0, P.NumberLiteral):
            return make_number_literal((number_value(a0) >> 8) & 0xFF, pos=pos)

        # DOUBLE(const) -> zero-extend byte to address
        if name == "DOUBLE" and isinstance(a0, P.NumberLiteral):
            return make_number_literal(number_value(a0) & 0xFFFF, pos=pos)

        if len(args) == 2:
            a1 = unwrap_paren(args[1])
            if isinstance(a0, P.NumberLiteral) and isinstance(a1, P.NumberLiteral):
                v0 = number_value(a0)
                v1 = number_value(a1)
                if name == "SHL":
                    return make_number_literal((v0 << v1) & 0xFFFF, pos=pos)
                if name == "SHR":
                    return make_number_literal((v0 >> v1) & 0xFFFF, pos=pos)
                if name == "ROL":
                    val = v0 & 0xFF
                    count = v1 & 7
                    result = ((val << count) | (val >> (8 - count))) & 0xFF
                    return make_number_literal(result, pos=pos)
                if name == "ROR":
                    val = v0 & 0xFF
                    count = v1 & 7
                    result = ((val >> count) | (val << (8 - count))) & 0xFF
                    return make_number_literal(result, pos=pos)

        return None

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
