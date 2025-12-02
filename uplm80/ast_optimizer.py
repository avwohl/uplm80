"""
AST Optimizer for PL/M-80.

Performs high-level optimizations on the AST before code generation:
- Constant folding and propagation
- Strength reduction
- Dead code elimination
- Common subexpression elimination (CSE)
- Loop-invariant code motion
- Algebraic simplifications
"""

from dataclasses import dataclass, field
from typing import Callable
from copy import deepcopy

from .ast_nodes import (
    ASTNode,
    DataType,
    BinaryOp,
    UnaryOp,
    Expr,
    NumberLiteral,
    StringLiteral,
    Identifier,
    SubscriptExpr,
    MemberExpr,
    CallExpr,
    BinaryExpr,
    UnaryExpr,
    LocationExpr,
    ConstListExpr,
    EmbeddedAssignExpr,
    Stmt,
    AssignStmt,
    CallStmt,
    ReturnStmt,
    GotoStmt,
    HaltStmt,
    EnableStmt,
    DisableStmt,
    NullStmt,
    LabeledStmt,
    IfStmt,
    DoBlock,
    DoWhileBlock,
    DoIterBlock,
    DoCaseBlock,
    Declaration,
    VarDecl,
    LabelDecl,
    LiterallyDecl,
    ProcDecl,
    DeclareStmt,
    Module,
)


def _get_expr_vars(expr: Expr) -> set[str]:
    """Get all variable names referenced in an expression."""
    result: set[str] = set()

    if isinstance(expr, Identifier):
        result.add(expr.name)
    elif isinstance(expr, BinaryExpr):
        result.update(_get_expr_vars(expr.left))
        result.update(_get_expr_vars(expr.right))
    elif isinstance(expr, UnaryExpr):
        result.update(_get_expr_vars(expr.operand))
    elif isinstance(expr, SubscriptExpr):
        result.update(_get_expr_vars(expr.base))
        result.update(_get_expr_vars(expr.index))
    elif isinstance(expr, MemberExpr):
        result.update(_get_expr_vars(expr.base))
    elif isinstance(expr, CallExpr):
        for arg in expr.args:
            result.update(_get_expr_vars(arg))
    elif isinstance(expr, LocationExpr):
        result.update(_get_expr_vars(expr.operand))
    elif isinstance(expr, EmbeddedAssignExpr):
        result.update(_get_expr_vars(expr.target))
        result.update(_get_expr_vars(expr.value))

    return result


def _expr_key(expr: Expr) -> str | None:
    """Generate a hashable key for an expression for CSE.

    Returns None for expressions that shouldn't be cached (with side effects).
    """
    if isinstance(expr, NumberLiteral):
        return f"NUM:{expr.value}"
    if isinstance(expr, StringLiteral):
        return f"STR:{expr.value}"
    if isinstance(expr, Identifier):
        return f"ID:{expr.name}"
    if isinstance(expr, BinaryExpr):
        left_key = _expr_key(expr.left)
        right_key = _expr_key(expr.right)
        if left_key is None or right_key is None:
            return None
        return f"BIN:{expr.op.name}:{left_key}:{right_key}"
    if isinstance(expr, UnaryExpr):
        operand_key = _expr_key(expr.operand)
        if operand_key is None:
            return None
        return f"UN:{expr.op.name}:{operand_key}"
    if isinstance(expr, SubscriptExpr):
        base_key = _expr_key(expr.base)
        idx_key = _expr_key(expr.index)
        if base_key is None or idx_key is None:
            return None
        return f"SUB:{base_key}:{idx_key}"
    if isinstance(expr, MemberExpr):
        base_key = _expr_key(expr.base)
        if base_key is None:
            return None
        return f"MEM:{base_key}:{expr.member}"
    if isinstance(expr, CallExpr):
        # Only pure built-in functions can be CSE'd
        if isinstance(expr.callee, Identifier):
            name = expr.callee.name
            # These built-ins are pure (no side effects)
            pure_builtins = {"LOW", "HIGH", "DOUBLE", "SHL", "SHR", "ROL", "ROR",
                           "LENGTH", "LAST", "SIZE"}
            if name in pure_builtins:
                arg_keys = [_expr_key(a) for a in expr.args]
                if all(k is not None for k in arg_keys):
                    return f"CALL:{name}:{':'.join(arg_keys)}"
        return None  # Function calls with side effects
    if isinstance(expr, LocationExpr):
        operand_key = _expr_key(expr.operand)
        if operand_key is None:
            return None
        return f"LOC:{operand_key}"
    return None


from enum import Enum


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

    def __init__(self, opt_level: int = 2, optimize_for: OptimizeFor = OptimizeFor.BALANCED) -> None:
        self.opt_level = opt_level
        self.optimize_for = optimize_for
        self.stats = OptimizationStats()
        # Known constant values for propagation
        self.constants: dict[str, int] = {}
        # Track which variables are modified in current scope
        self.modified_vars: set[str] = set()
        # CSE: map from expression key to (temp_var_name, expr) for level 3
        self.cse_cache: dict[str, tuple[str, Expr]] = {}
        self.cse_counter: int = 0
        # Track variables used in expressions for invalidation
        self.expr_vars: dict[str, set[str]] = {}  # expr_key -> set of var names
        # Copy propagation: x = y means copies[x] = y
        self.copies: dict[str, str] = {}
        # Procedure inlining: track small procedures that can be inlined
        self.inlinable_procs: dict[str, ProcDecl] = {}

    def _parse_plm_number(self, s: str) -> int | None:
        """Parse a PL/M-style numeric literal (handles $ separators and B/H/O/Q/D suffixes)."""
        try:
            # Remove $ digit separators and convert to uppercase
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
                return int(s, 0)  # Let Python auto-detect base
        except (ValueError, TypeError):
            return None

    def optimize(self, module: Module) -> Module:
        """Optimize an entire module."""
        if self.opt_level == 0:
            return module

        # Multiple passes for iterative improvement
        changed = True
        passes = 0
        max_passes = 5

        while changed and passes < max_passes:
            changed = False
            passes += 1

            # Optimize declarations
            new_decls: list[Declaration] = []
            for decl in module.decls:
                opt_decl = self._optimize_declaration(decl)
                if opt_decl is not None:
                    new_decls.append(opt_decl)
                    if opt_decl is not decl:
                        changed = True

            # Optimize statements
            new_stmts: list[Stmt] = []
            for stmt in module.stmts:
                opt_stmt = self._optimize_stmt(stmt)
                if opt_stmt is not None:
                    new_stmts.append(opt_stmt)
                    if opt_stmt is not stmt:
                        changed = True

            module = Module(
                name=module.name,
                origin=module.origin,
                decls=new_decls,
                stmts=new_stmts,
                span=module.span,
            )

        return module

    def _optimize_declaration(self, decl: Declaration) -> Declaration | None:
        """Optimize a declaration."""
        if isinstance(decl, VarDecl):
            # Optimize initial values
            if decl.initial_values:
                decl.initial_values = [
                    self._optimize_expr(v) for v in decl.initial_values
                ]
            if decl.data_values:
                decl.data_values = [self._optimize_expr(v) for v in decl.data_values]
            if decl.at_location:
                decl.at_location = self._optimize_expr(decl.at_location)
            return decl

        elif isinstance(decl, LiterallyDecl):
            # Track literal for potential constant propagation
            # Try to evaluate if it's a simple number
            val = self._parse_plm_number(decl.value)
            if val is not None:
                self.constants[decl.name] = val
            return decl

        elif isinstance(decl, ProcDecl):
            # Optimize procedure body
            new_decls = [
                d for d in (self._optimize_declaration(d) for d in decl.decls) if d
            ]
            new_stmts = [
                s for s in (self._optimize_stmt(s) for s in decl.stmts) if s
            ]
            # Eliminate unreachable code after RETURN/GOTO/HALT
            new_stmts = self._eliminate_unreachable(new_stmts)
            # Eliminate dead stores
            new_stmts = self._eliminate_dead_stores(new_stmts)
            optimized_proc = ProcDecl(
                name=decl.name,
                params=decl.params,
                return_type=decl.return_type,
                is_public=decl.is_public,
                is_external=decl.is_external,
                is_reentrant=decl.is_reentrant,
                interrupt_num=decl.interrupt_num,
                decls=new_decls,
                stmts=new_stmts,
                span=decl.span,
            )
            # Level 3: Track inlinable procedures
            if self.opt_level >= 3 and self._is_inlinable(optimized_proc):
                self.inlinable_procs[decl.name] = optimized_proc
            return optimized_proc

        return decl

    def _is_terminator(self, stmt: Stmt) -> bool:
        """Check if a statement is a control flow terminator (no fall-through)."""
        if isinstance(stmt, ReturnStmt):
            return True
        if isinstance(stmt, GotoStmt):
            return True
        if isinstance(stmt, HaltStmt):
            return True
        # A labeled statement is a terminator if its inner statement is
        if isinstance(stmt, LabeledStmt):
            return self._is_terminator(stmt.stmt)
        return False

    def _eliminate_unreachable(self, stmts: list[Stmt]) -> list[Stmt]:
        """Remove statements after terminators (RETURN, GOTO, HALT)."""
        if self.opt_level < 2:
            return stmts

        result: list[Stmt] = []
        for stmt in stmts:
            result.append(stmt)
            if self._is_terminator(stmt):
                # Count eliminated statements
                remaining = len(stmts) - len(result)
                if remaining > 0:
                    self.stats.dead_code_eliminated += remaining
                break
        return result

    def _eliminate_dead_stores(self, stmts: list[Stmt]) -> list[Stmt]:
        """Remove assignments that are immediately overwritten without being read.

        This is a simple local analysis - a variable assigned and then reassigned
        in consecutive statements without being read is a dead store.
        """
        if self.opt_level < 3:
            return stmts

        result: list[Stmt] = []
        i = 0
        while i < len(stmts):
            stmt = stmts[i]

            # Check if this is a simple assignment
            if isinstance(stmt, AssignStmt) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, Identifier):
                    # Look ahead - is the next statement an assignment to the same var?
                    if i + 1 < len(stmts):
                        next_stmt = stmts[i + 1]
                        if (isinstance(next_stmt, AssignStmt) and
                                len(next_stmt.targets) == 1 and
                                isinstance(next_stmt.targets[0], Identifier) and
                                next_stmt.targets[0].name == target.name):
                            # Check if target is NOT used in the value of next assignment
                            if target.name not in _get_expr_vars(next_stmt.value):
                                # This is a dead store - skip it
                                self.stats.dead_stores_eliminated += 1
                                i += 1
                                continue

            result.append(stmt)
            i += 1

        return result

    def _get_modified_vars_in_stmts(self, stmts: list[Stmt]) -> set[str]:
        """Get all variables modified within a list of statements."""
        modified: set[str] = set()

        def visit_stmt(s: Stmt) -> None:
            if isinstance(s, AssignStmt):
                for target in s.targets:
                    if isinstance(target, Identifier):
                        modified.add(target.name)
                    elif isinstance(target, SubscriptExpr):
                        if isinstance(target.base, Identifier):
                            modified.add(target.base.name)
                    # Check for embedded assignments in value
                visit_expr(s.value)
            elif isinstance(s, DoBlock):
                for sub in s.stmts:
                    visit_stmt(sub)
            elif isinstance(s, DoWhileBlock):
                for sub in s.stmts:
                    visit_stmt(sub)
            elif isinstance(s, DoIterBlock):
                if isinstance(s.index_var, Identifier):
                    modified.add(s.index_var.name)
                for sub in s.stmts:
                    visit_stmt(sub)
            elif isinstance(s, DoCaseBlock):
                for case in s.cases:
                    for sub in case:
                        visit_stmt(sub)
            elif isinstance(s, IfStmt):
                visit_stmt(s.then_stmt)
                if s.else_stmt:
                    visit_stmt(s.else_stmt)
            elif isinstance(s, LabeledStmt):
                visit_stmt(s.stmt)
            elif isinstance(s, CallStmt):
                # Calls may modify globals - be conservative
                for arg in s.args:
                    visit_expr(arg)

        def visit_expr(e: Expr | None) -> None:
            if e is None:
                return
            if isinstance(e, EmbeddedAssignExpr):
                if isinstance(e.target, Identifier):
                    modified.add(e.target.name)
                visit_expr(e.value)
            elif isinstance(e, BinaryExpr):
                visit_expr(e.left)
                visit_expr(e.right)
            elif isinstance(e, UnaryExpr):
                visit_expr(e.operand)
            elif isinstance(e, CallExpr):
                for arg in e.args:
                    visit_expr(arg)

        for s in stmts:
            visit_stmt(s)

        return modified

    def _cache_invariant_exprs(self, expr: Expr, modified_vars: set[str]) -> None:
        """Cache loop-invariant subexpressions for CSE to find later."""
        if self._is_loop_invariant(expr, modified_vars):
            key = _expr_key(expr)
            if key is not None and key not in self.cse_cache:
                self.cse_cache[key] = (f"??INV{self.cse_counter}", expr)
                self.expr_vars[key] = _get_expr_vars(expr)
                self.cse_counter += 1
                self.stats.loop_invariants_moved += 1

        # Recursively check subexpressions
        if isinstance(expr, BinaryExpr):
            self._cache_invariant_exprs(expr.left, modified_vars)
            self._cache_invariant_exprs(expr.right, modified_vars)
        elif isinstance(expr, UnaryExpr):
            self._cache_invariant_exprs(expr.operand, modified_vars)
        elif isinstance(expr, CallExpr):
            for arg in expr.args:
                self._cache_invariant_exprs(arg, modified_vars)

    def _is_loop_invariant(self, expr: Expr, modified_vars: set[str]) -> bool:
        """Check if an expression is invariant (not modified) within a loop."""
        if isinstance(expr, NumberLiteral):
            return True
        if isinstance(expr, StringLiteral):
            return True
        if isinstance(expr, Identifier):
            return expr.name not in modified_vars
        if isinstance(expr, BinaryExpr):
            return (self._is_loop_invariant(expr.left, modified_vars) and
                    self._is_loop_invariant(expr.right, modified_vars))
        if isinstance(expr, UnaryExpr):
            return self._is_loop_invariant(expr.operand, modified_vars)
        if isinstance(expr, SubscriptExpr):
            # Array access is tricky - if base is modified, we can't assume invariance
            if isinstance(expr.base, Identifier) and expr.base.name in modified_vars:
                return False
            return (self._is_loop_invariant(expr.base, modified_vars) and
                    self._is_loop_invariant(expr.index, modified_vars))
        if isinstance(expr, CallExpr):
            # Only pure builtins can be loop-invariant
            if isinstance(expr.callee, Identifier):
                pure_builtins = {"LOW", "HIGH", "DOUBLE", "SHL", "SHR", "ROL", "ROR",
                               "LENGTH", "LAST", "SIZE"}
                if expr.callee.name in pure_builtins:
                    return all(self._is_loop_invariant(a, modified_vars) for a in expr.args)
            return False
        if isinstance(expr, LocationExpr):
            return self._is_loop_invariant(expr.operand, modified_vars)
        return False

    def _invalidate_cse_for_var(self, var_name: str) -> None:
        """Invalidate CSE cache entries that depend on a modified variable."""
        if self.opt_level < 3:
            return
        # Find and remove all cached expressions that use this variable
        to_remove = []
        for key, vars_used in self.expr_vars.items():
            if var_name in vars_used:
                to_remove.append(key)
        for key in to_remove:
            self.cse_cache.pop(key, None)
            self.expr_vars.pop(key, None)

    def _invalidate_copies_for_var(self, var_name: str) -> None:
        """Invalidate copy propagation entries when a variable is modified."""
        # Remove if var_name is the target of a copy
        self.copies.pop(var_name, None)
        # Remove any copies where var_name is the source
        to_remove = [k for k, v in self.copies.items() if v == var_name]
        for k in to_remove:
            del self.copies[k]

    def _count_stmts(self, stmts: list[Stmt]) -> int:
        """Count the number of statements (recursively)."""
        count = 0
        for stmt in stmts:
            count += 1
            if isinstance(stmt, DoBlock):
                count += self._count_stmts(stmt.stmts)
            elif isinstance(stmt, DoWhileBlock):
                count += self._count_stmts(stmt.stmts)
            elif isinstance(stmt, DoIterBlock):
                count += self._count_stmts(stmt.stmts)
            elif isinstance(stmt, DoCaseBlock):
                for case in stmt.cases:
                    count += self._count_stmts(case)
            elif isinstance(stmt, IfStmt):
                if isinstance(stmt.then_stmt, (DoBlock, DoWhileBlock)):
                    count += self._count_stmts([stmt.then_stmt])
                if stmt.else_stmt and isinstance(stmt.else_stmt, (DoBlock, DoWhileBlock)):
                    count += self._count_stmts([stmt.else_stmt])
        return count

    def _is_inlinable(self, proc: ProcDecl) -> bool:
        """Check if a procedure is suitable for inlining."""
        # Don't inline external, reentrant, or interrupt procedures
        if proc.is_external or proc.is_reentrant or proc.interrupt_num is not None:
            return False
        # Don't inline procedures with nested procedures
        for decl in proc.decls:
            if isinstance(decl, ProcDecl):
                return False
        # Don't inline procedures with too many statements
        stmt_count = self._count_stmts(proc.stmts)
        if stmt_count > 5:
            return False
        # Don't inline procedures with too many parameters
        if len(proc.params) > 3:
            return False
        # Don't inline procedures that have local declarations (complex scoping)
        if proc.decls:
            return False
        return True

    def _inline_procedure(self, proc: ProcDecl, args: list[Expr], span) -> Stmt | None:
        """Inline a procedure call, substituting parameters with arguments."""
        # Build parameter substitution map
        param_map: dict[str, Expr] = {}
        for param, arg in zip(proc.params, args):
            param_map[param.name] = arg

        # Deep copy and substitute parameters in procedure body
        inlined_stmts: list[Stmt] = []
        for stmt in proc.stmts:
            # Skip RETURN statements for void procedures (they just return)
            if isinstance(stmt, ReturnStmt) and stmt.value is None:
                continue
            # Substitute and copy
            subst_stmt = self._substitute_params(deepcopy(stmt), param_map)
            if subst_stmt is not None:
                inlined_stmts.append(subst_stmt)

        if not inlined_stmts:
            return NullStmt(span=span)
        if len(inlined_stmts) == 1:
            return inlined_stmts[0]
        return DoBlock([], inlined_stmts, None, span=span)

    def _substitute_params(self, node: ASTNode, param_map: dict[str, Expr]) -> ASTNode | None:
        """Substitute parameter references with argument expressions."""
        if isinstance(node, Identifier):
            if node.name in param_map:
                return deepcopy(param_map[node.name])
            return node

        if isinstance(node, BinaryExpr):
            node.left = self._substitute_params(node.left, param_map)
            node.right = self._substitute_params(node.right, param_map)
            return node

        if isinstance(node, UnaryExpr):
            node.operand = self._substitute_params(node.operand, param_map)
            return node

        if isinstance(node, CallExpr):
            node.callee = self._substitute_params(node.callee, param_map)
            node.args = [self._substitute_params(a, param_map) for a in node.args]
            return node

        if isinstance(node, SubscriptExpr):
            node.base = self._substitute_params(node.base, param_map)
            node.index = self._substitute_params(node.index, param_map)
            return node

        if isinstance(node, AssignStmt):
            node.targets = [self._substitute_params(t, param_map) for t in node.targets]
            node.value = self._substitute_params(node.value, param_map)
            return node

        if isinstance(node, CallStmt):
            node.callee = self._substitute_params(node.callee, param_map)
            node.args = [self._substitute_params(a, param_map) for a in node.args]
            return node

        if isinstance(node, ReturnStmt):
            if node.value:
                node.value = self._substitute_params(node.value, param_map)
            return node

        if isinstance(node, IfStmt):
            node.condition = self._substitute_params(node.condition, param_map)
            node.then_stmt = self._substitute_params(node.then_stmt, param_map)
            if node.else_stmt:
                node.else_stmt = self._substitute_params(node.else_stmt, param_map)
            return node

        if isinstance(node, DoBlock):
            node.stmts = [self._substitute_params(s, param_map) for s in node.stmts]
            return node

        if isinstance(node, DoWhileBlock):
            node.condition = self._substitute_params(node.condition, param_map)
            node.stmts = [self._substitute_params(s, param_map) for s in node.stmts]
            return node

        if isinstance(node, DoIterBlock):
            node.index_var = self._substitute_params(node.index_var, param_map)
            node.start = self._substitute_params(node.start, param_map)
            node.bound = self._substitute_params(node.bound, param_map)
            if node.step:
                node.step = self._substitute_params(node.step, param_map)
            node.stmts = [self._substitute_params(s, param_map) for s in node.stmts]
            return node

        # For other node types, return as-is
        return node

    def _normalize_commutative(self, op: BinaryOp, left: Expr, right: Expr) -> tuple[Expr, Expr]:
        """Normalize operand order for commutative operations to improve CSE."""
        # Only for commutative operations
        if op not in (BinaryOp.ADD, BinaryOp.MUL, BinaryOp.AND, BinaryOp.OR,
                      BinaryOp.XOR, BinaryOp.EQ, BinaryOp.NE):
            return left, right

        # Get sort keys for expressions
        def sort_key(e: Expr) -> tuple[int, str]:
            if isinstance(e, NumberLiteral):
                return (2, str(e.value))  # Constants last
            elif isinstance(e, Identifier):
                return (0, e.name)  # Identifiers first, sorted by name
            else:
                return (1, _expr_key(e) or "")  # Complex expressions in middle

        left_key = sort_key(left)
        right_key = sort_key(right)

        # Swap if right should come before left
        if right_key < left_key:
            return right, left
        return left, right

    def _optimize_stmt(self, stmt: Stmt) -> Stmt | None:
        """Optimize a statement. Returns None to remove it."""
        if isinstance(stmt, AssignStmt):
            opt_value = self._optimize_expr(stmt.value)
            opt_targets = [self._optimize_expr(t) for t in stmt.targets]

            # Track modified variables and invalidate caches
            for target in opt_targets:
                if isinstance(target, Identifier):
                    self.modified_vars.add(target.name)
                    # Remove from constants if modified
                    self.constants.pop(target.name, None)
                    # Invalidate CSE entries using this variable
                    self._invalidate_cse_for_var(target.name)
                    # Invalidate copy propagation entries
                    self._invalidate_copies_for_var(target.name)

            # Level 3: Track copies and constants
            if self.opt_level >= 3 and len(opt_targets) == 1:
                target = opt_targets[0]
                if isinstance(target, Identifier):
                    if isinstance(opt_value, NumberLiteral):
                        # Constant propagation
                        self.constants[target.name] = opt_value.value
                    elif isinstance(opt_value, Identifier):
                        # Copy propagation: x = y
                        self.copies[target.name] = opt_value.name

            return AssignStmt(opt_targets, opt_value, span=stmt.span)

        elif isinstance(stmt, CallStmt):
            opt_callee = self._optimize_expr(stmt.callee)
            opt_args = [self._optimize_expr(a) for a in stmt.args]

            # Level 3: Inline small procedures
            # Skip if optimizing for size (inlining increases code size)
            if (self.opt_level >= 3
                    and self.optimize_for != OptimizeFor.SIZE
                    and isinstance(opt_callee, Identifier)
                    and opt_callee.name in self.inlinable_procs):
                proc = self.inlinable_procs[opt_callee.name]
                # Only inline if argument count matches parameter count
                if len(opt_args) == len(proc.params):
                    inlined = self._inline_procedure(proc, opt_args, stmt.span)
                    if inlined is not None:
                        self.stats.procedures_inlined += 1
                        return inlined

            return CallStmt(opt_callee, opt_args, span=stmt.span)

        elif isinstance(stmt, ReturnStmt):
            opt_value = self._optimize_expr(stmt.value) if stmt.value else None
            return ReturnStmt(opt_value, span=stmt.span)

        elif isinstance(stmt, IfStmt):
            opt_cond = self._optimize_expr(stmt.condition)

            # Constant condition elimination (level 2+)
            if self.opt_level >= 2 and isinstance(opt_cond, NumberLiteral):
                self.stats.dead_code_eliminated += 1
                if opt_cond.value != 0:
                    # Condition always true
                    return self._optimize_stmt(stmt.then_stmt)
                else:
                    # Condition always false
                    if stmt.else_stmt:
                        return self._optimize_stmt(stmt.else_stmt)
                    return NullStmt(span=stmt.span)

            opt_then = self._optimize_stmt(stmt.then_stmt)
            opt_else = self._optimize_stmt(stmt.else_stmt) if stmt.else_stmt else None

            if opt_then is None:
                opt_then = NullStmt(span=stmt.span)

            return IfStmt(opt_cond, opt_then, opt_else, span=stmt.span)

        elif isinstance(stmt, DoBlock):
            new_decls = [
                d for d in (self._optimize_declaration(d) for d in stmt.decls) if d
            ]
            new_stmts = [
                s for s in (self._optimize_stmt(s) for s in stmt.stmts) if s
            ]
            # Eliminate unreachable code after RETURN/GOTO/HALT
            new_stmts = self._eliminate_unreachable(new_stmts)
            # Eliminate dead stores (consecutive assignments without read)
            new_stmts = self._eliminate_dead_stores(new_stmts)
            return DoBlock(new_decls, new_stmts, stmt.end_label, span=stmt.span)

        elif isinstance(stmt, DoWhileBlock):
            opt_cond = self._optimize_expr(stmt.condition)

            # Check for DO WHILE 0 (never executes)
            if self.opt_level >= 2 and isinstance(opt_cond, NumberLiteral):
                if opt_cond.value == 0:
                    self.stats.dead_code_eliminated += 1
                    return NullStmt(span=stmt.span)

            # Level 3: Check for loop-invariant subexpressions in condition
            if self.opt_level >= 3:
                modified_vars = self._get_modified_vars_in_stmts(stmt.stmts)
                # Cache invariant expressions in condition for CSE
                self._cache_invariant_exprs(opt_cond, modified_vars)

            new_stmts = [
                s for s in (self._optimize_stmt(s) for s in stmt.stmts) if s
            ]
            new_stmts = self._eliminate_unreachable(new_stmts)
            return DoWhileBlock(opt_cond, new_stmts, stmt.end_label, span=stmt.span)

        elif isinstance(stmt, DoIterBlock):
            opt_start = self._optimize_expr(stmt.start)
            opt_bound = self._optimize_expr(stmt.bound)
            opt_step = self._optimize_expr(stmt.step) if stmt.step else None
            opt_index = self._optimize_expr(stmt.index_var)

            # Check for empty loop (start > bound with positive step)
            if (
                self.opt_level >= 2
                and isinstance(opt_start, NumberLiteral)
                and isinstance(opt_bound, NumberLiteral)
            ):
                step_val = 1
                if isinstance(opt_step, NumberLiteral):
                    step_val = opt_step.value
                if step_val > 0 and opt_start.value > opt_bound.value:
                    self.stats.dead_code_eliminated += 1
                    return NullStmt(span=stmt.span)

            # Level 3: Loop unrolling for small constant-bound loops
            # Skip if optimizing for size (unrolling increases code size)
            if (
                self.opt_level >= 3
                and self.optimize_for != OptimizeFor.SIZE
                and isinstance(opt_start, NumberLiteral)
                and isinstance(opt_bound, NumberLiteral)
                and isinstance(opt_index, Identifier)
            ):
                step_val = 1
                if isinstance(opt_step, NumberLiteral):
                    step_val = opt_step.value
                if step_val > 0:
                    iterations = (opt_bound.value - opt_start.value) // step_val + 1
                    # Unroll loops with <= 4 iterations and <= 3 statements
                    # For balanced mode, be more conservative (<=2 iterations)
                    max_iter = 4 if self.optimize_for == OptimizeFor.SPEED else 2
                    if 1 <= iterations <= max_iter and len(stmt.stmts) <= 3:
                        unrolled_stmts: list[Stmt] = []
                        for i in range(iterations):
                            val = opt_start.value + i * step_val
                            # Add assignment: index = val
                            unrolled_stmts.append(AssignStmt(
                                [Identifier(opt_index.name, span=stmt.span)],
                                NumberLiteral(val, span=stmt.span),
                                span=stmt.span
                            ))
                            # Add loop body (deep copy to avoid sharing)
                            for s in stmt.stmts:
                                unrolled_stmts.append(deepcopy(s))
                        self.stats.loops_unrolled += 1
                        # Wrap in a DoBlock and optimize it
                        block = DoBlock([], unrolled_stmts, stmt.end_label, span=stmt.span)
                        return self._optimize_stmt(block)

            # Level 3: Cache loop-invariant bound expressions
            if self.opt_level >= 3:
                modified_vars = self._get_modified_vars_in_stmts(stmt.stmts)
                if isinstance(stmt.index_var, Identifier):
                    modified_vars.add(stmt.index_var.name)
                # The bound is computed every iteration - cache if invariant
                self._cache_invariant_exprs(opt_bound, modified_vars)
                if opt_step:
                    self._cache_invariant_exprs(opt_step, modified_vars)

            new_stmts = [
                s for s in (self._optimize_stmt(s) for s in stmt.stmts) if s
            ]
            new_stmts = self._eliminate_unreachable(new_stmts)

            return DoIterBlock(
                opt_index, opt_start, opt_bound, opt_step, new_stmts,
                stmt.end_label, span=stmt.span
            )

        elif isinstance(stmt, DoCaseBlock):
            opt_selector = self._optimize_expr(stmt.selector)

            # If selector is constant, keep only that case (level 2+)
            if self.opt_level >= 2 and isinstance(opt_selector, NumberLiteral):
                case_idx = opt_selector.value
                if 0 <= case_idx < len(stmt.cases):
                    self.stats.dead_code_eliminated += 1
                    # Return just that case's statements
                    case_stmts = stmt.cases[case_idx]
                    if len(case_stmts) == 1:
                        return self._optimize_stmt(case_stmts[0])
                    else:
                        opt_stmts = [
                            s for s in (self._optimize_stmt(s) for s in case_stmts) if s
                        ]
                        return DoBlock([], opt_stmts, stmt.end_label, span=stmt.span)

            # Optimize all cases
            new_cases: list[list[Stmt]] = []
            for case in stmt.cases:
                opt_case = [s for s in (self._optimize_stmt(s) for s in case) if s]
                opt_case = self._eliminate_unreachable(opt_case)
                new_cases.append(opt_case)

            return DoCaseBlock(opt_selector, new_cases, stmt.end_label, span=stmt.span)

        elif isinstance(stmt, LabeledStmt):
            opt_inner = self._optimize_stmt(stmt.stmt)
            if opt_inner is None:
                opt_inner = NullStmt(span=stmt.span)
            return LabeledStmt(stmt.label, opt_inner, span=stmt.span)

        elif isinstance(stmt, DeclareStmt):
            new_decls = [
                d for d in (self._optimize_declaration(d) for d in stmt.declarations) if d
            ]
            if not new_decls:
                return None
            return DeclareStmt(new_decls, span=stmt.span)

        # Pass through unchanged
        return stmt

    def _optimize_expr(self, expr: Expr | None) -> Expr | None:
        """Optimize an expression."""
        if expr is None:
            return None

        if isinstance(expr, NumberLiteral):
            return expr

        if isinstance(expr, StringLiteral):
            return expr

        if isinstance(expr, Identifier):
            # Constant propagation (level 1+)
            if self.opt_level >= 1 and expr.name in self.constants:
                self.stats.constants_folded += 1
                return NumberLiteral(self.constants[expr.name], span=expr.span)
            # Copy propagation (level 3) - replace x with y if x = y was seen
            if self.opt_level >= 3 and expr.name in self.copies:
                self.stats.copies_propagated += 1
                return Identifier(self.copies[expr.name], span=expr.span)
            return expr

        if isinstance(expr, BinaryExpr):
            return self._optimize_binary(expr)

        if isinstance(expr, UnaryExpr):
            return self._optimize_unary(expr)

        if isinstance(expr, SubscriptExpr):
            opt_base = self._optimize_expr(expr.base)
            opt_index = self._optimize_expr(expr.index)
            return SubscriptExpr(opt_base, opt_index, span=expr.span)

        if isinstance(expr, MemberExpr):
            opt_base = self._optimize_expr(expr.base)
            return MemberExpr(opt_base, expr.member, span=expr.span)

        if isinstance(expr, CallExpr):
            opt_callee = self._optimize_expr(expr.callee)
            opt_args = [self._optimize_expr(a) for a in expr.args]

            # Optimize built-in calls with constant args
            if self.opt_level >= 1 and isinstance(opt_callee, Identifier):
                result = self._optimize_builtin_call(opt_callee.name, opt_args)
                if result is not None:
                    return result

            return CallExpr(opt_callee, opt_args, span=expr.span)

        if isinstance(expr, LocationExpr):
            opt_operand = self._optimize_expr(expr.operand)
            return LocationExpr(opt_operand, span=expr.span)

        if isinstance(expr, ConstListExpr):
            opt_values = [self._optimize_expr(v) for v in expr.values]
            return ConstListExpr(opt_values, span=expr.span)

        if isinstance(expr, EmbeddedAssignExpr):
            opt_target = self._optimize_expr(expr.target)
            opt_value = self._optimize_expr(expr.value)
            return EmbeddedAssignExpr(opt_target, opt_value, span=expr.span)

        return expr

    def _optimize_binary(self, expr: BinaryExpr) -> Expr:
        """Optimize a binary expression."""
        left = self._optimize_expr(expr.left)
        right = self._optimize_expr(expr.right)

        # Constant folding (level 1+)
        if (
            self.opt_level >= 1
            and isinstance(left, NumberLiteral)
            and isinstance(right, NumberLiteral)
        ):
            result = self._eval_binary_const(expr.op, left.value, right.value)
            if result is not None:
                self.stats.constants_folded += 1
                return NumberLiteral(result, span=expr.span)

        # Strength reduction (level 2+)
        if self.opt_level >= 2:
            reduced = self._strength_reduce(expr.op, left, right, expr.span)
            if reduced is not None:
                self.stats.strength_reductions += 1
                return reduced

        # Algebraic simplifications (level 1+)
        if self.opt_level >= 1:
            simplified = self._algebraic_simplify(expr.op, left, right, expr.span)
            if simplified is not None:
                self.stats.algebraic_simplifications += 1
                return simplified

        # Boolean/comparison simplifications (level 2+)
        if self.opt_level >= 2:
            bool_simp = self._boolean_simplify(expr.op, left, right, expr.span)
            if bool_simp is not None:
                self.stats.boolean_simplifications += 1
                return bool_simp

        # Commutative normalization for better CSE (level 3)
        # Put operands in canonical order: constants on right, identifiers sorted
        if self.opt_level >= 3:
            left, right = self._normalize_commutative(expr.op, left, right)

        result_expr = BinaryExpr(expr.op, left, right, span=expr.span)

        # CSE: check if we've seen this expression before (level 3)
        if self.opt_level >= 3:
            key = _expr_key(result_expr)
            if key is not None:
                if key in self.cse_cache:
                    # Return reference to cached value
                    self.stats.cse_eliminations += 1
                    cached_expr = self.cse_cache[key][1]
                    return deepcopy(cached_expr)
                else:
                    # Cache this expression
                    self.cse_cache[key] = (f"??CSE{self.cse_counter}", result_expr)
                    self.expr_vars[key] = _get_expr_vars(result_expr)
                    self.cse_counter += 1

        return result_expr

    def _optimize_unary(self, expr: UnaryExpr) -> Expr:
        """Optimize a unary expression."""
        operand = self._optimize_expr(expr.operand)

        # Constant folding
        if self.opt_level >= 1 and isinstance(operand, NumberLiteral):
            result = self._eval_unary_const(expr.op, operand.value)
            if result is not None:
                self.stats.constants_folded += 1
                return NumberLiteral(result, span=expr.span)

        # Double negation elimination
        if expr.op == UnaryOp.NEG and isinstance(operand, UnaryExpr):
            if operand.op == UnaryOp.NEG:
                self.stats.algebraic_simplifications += 1
                return operand.operand

        # NOT NOT elimination
        if expr.op == UnaryOp.NOT and isinstance(operand, UnaryExpr):
            if operand.op == UnaryOp.NOT:
                self.stats.algebraic_simplifications += 1
                return operand.operand

        return UnaryExpr(expr.op, operand, span=expr.span)

    def _eval_binary_const(self, op: BinaryOp, left: int, right: int) -> int | None:
        """Evaluate a binary operation on constants."""
        # Use 16-bit unsigned arithmetic (PL/M-80 semantics)
        mask = 0xFFFF

        try:
            if op == BinaryOp.ADD:
                return (left + right) & mask
            elif op == BinaryOp.SUB:
                return (left - right) & mask
            elif op == BinaryOp.MUL:
                return (left * right) & mask
            elif op == BinaryOp.DIV:
                if right == 0:
                    return None
                return (left // right) & mask
            elif op == BinaryOp.MOD:
                if right == 0:
                    return None
                return (left % right) & mask
            elif op == BinaryOp.AND:
                return left & right
            elif op == BinaryOp.OR:
                return left | right
            elif op == BinaryOp.XOR:
                return left ^ right
            elif op == BinaryOp.EQ:
                return 0xFFFF if left == right else 0
            elif op == BinaryOp.NE:
                return 0xFFFF if left != right else 0
            elif op == BinaryOp.LT:
                return 0xFFFF if left < right else 0
            elif op == BinaryOp.GT:
                return 0xFFFF if left > right else 0
            elif op == BinaryOp.LE:
                return 0xFFFF if left <= right else 0
            elif op == BinaryOp.GE:
                return 0xFFFF if left >= right else 0
        except (ZeroDivisionError, OverflowError):
            return None

        return None

    def _eval_unary_const(self, op: UnaryOp, value: int) -> int | None:
        """Evaluate a unary operation on a constant."""
        mask = 0xFFFF

        if op == UnaryOp.NEG:
            return (-value) & mask
        elif op == UnaryOp.NOT:
            return (~value) & mask
        elif op == UnaryOp.LOW:
            return value & 0xFF
        elif op == UnaryOp.HIGH:
            return (value >> 8) & 0xFF

        return None

    def _strength_reduce(
        self, op: BinaryOp, left: Expr, right: Expr, span
    ) -> Expr | None:
        """Apply strength reduction transformations."""
        # Multiply by power of 2 -> shift left
        if op == BinaryOp.MUL and isinstance(right, NumberLiteral):
            shift = self._log2_if_power_of_2(right.value)
            if shift is not None:
                if shift == 0:
                    return NumberLiteral(0, span=span) if isinstance(left, NumberLiteral) and left.value == 0 else left
                if shift == 1:
                    # x * 2 -> x + x
                    return BinaryExpr(BinaryOp.ADD, left, deepcopy(left), span=span)
                # x * 2^n -> SHL(x, n)
                return CallExpr(
                    Identifier("SHL", span=span),
                    [left, NumberLiteral(shift, span=span)],
                    span=span,
                )

        # Divide by power of 2 -> shift right
        if op == BinaryOp.DIV and isinstance(right, NumberLiteral):
            shift = self._log2_if_power_of_2(right.value)
            if shift is not None:
                if shift == 0:
                    return left  # x / 1 = x
                return CallExpr(
                    Identifier("SHR", span=span),
                    [left, NumberLiteral(shift, span=span)],
                    span=span,
                )

        # Modulo by power of 2 -> AND with (2^n - 1)
        if op == BinaryOp.MOD and isinstance(right, NumberLiteral):
            shift = self._log2_if_power_of_2(right.value)
            if shift is not None:
                mask = right.value - 1
                return BinaryExpr(BinaryOp.AND, left, NumberLiteral(mask, span=span), span=span)

        return None

    def _algebraic_simplify(
        self, op: BinaryOp, left: Expr, right: Expr, span
    ) -> Expr | None:
        """Apply algebraic simplifications."""
        # x + 0 = x, 0 + x = x
        if op == BinaryOp.ADD:
            if isinstance(right, NumberLiteral) and right.value == 0:
                return left
            if isinstance(left, NumberLiteral) and left.value == 0:
                return right

        # x - 0 = x
        if op == BinaryOp.SUB:
            if isinstance(right, NumberLiteral) and right.value == 0:
                return left
            # x - x = 0 (if same variable)
            if isinstance(left, Identifier) and isinstance(right, Identifier):
                if left.name == right.name:
                    return NumberLiteral(0, span=span)

        # x * 1 = x, 1 * x = x
        if op == BinaryOp.MUL:
            if isinstance(right, NumberLiteral) and right.value == 1:
                return left
            if isinstance(left, NumberLiteral) and left.value == 1:
                return right
            # x * 0 = 0, 0 * x = 0
            if isinstance(right, NumberLiteral) and right.value == 0:
                return NumberLiteral(0, span=span)
            if isinstance(left, NumberLiteral) and left.value == 0:
                return NumberLiteral(0, span=span)

        # x / 1 = x
        if op == BinaryOp.DIV:
            if isinstance(right, NumberLiteral) and right.value == 1:
                return left

        # x AND 0 = 0, x AND FFFF = x
        if op == BinaryOp.AND:
            if isinstance(right, NumberLiteral):
                if right.value == 0:
                    return NumberLiteral(0, span=span)
                if right.value == 0xFFFF:
                    return left
            if isinstance(left, NumberLiteral):
                if left.value == 0:
                    return NumberLiteral(0, span=span)
                if left.value == 0xFFFF:
                    return right

        # x OR 0 = x, x OR FFFF = FFFF
        if op == BinaryOp.OR:
            if isinstance(right, NumberLiteral):
                if right.value == 0:
                    return left
                if right.value == 0xFFFF:
                    return NumberLiteral(0xFFFF, span=span)
            if isinstance(left, NumberLiteral):
                if left.value == 0:
                    return right
                if left.value == 0xFFFF:
                    return NumberLiteral(0xFFFF, span=span)

        # x XOR 0 = x
        if op == BinaryOp.XOR:
            if isinstance(right, NumberLiteral) and right.value == 0:
                return left
            if isinstance(left, NumberLiteral) and left.value == 0:
                return right
            # x XOR x = 0
            if isinstance(left, Identifier) and isinstance(right, Identifier):
                if left.name == right.name:
                    return NumberLiteral(0, span=span)
            # x XOR FFFF = NOT x
            if isinstance(right, NumberLiteral) and right.value == 0xFFFF:
                return UnaryExpr(UnaryOp.NOT, left, span=span)
            if isinstance(left, NumberLiteral) and left.value == 0xFFFF:
                return UnaryExpr(UnaryOp.NOT, right, span=span)

        # (x + c1) + c2 -> x + (c1 + c2)  - constant folding on nested adds
        if op == BinaryOp.ADD:
            if isinstance(right, NumberLiteral) and isinstance(left, BinaryExpr):
                if left.op == BinaryOp.ADD and isinstance(left.right, NumberLiteral):
                    new_const = (left.right.value + right.value) & 0xFFFF
                    return BinaryExpr(BinaryOp.ADD, left.left,
                                     NumberLiteral(new_const, span=span), span=span)
                if left.op == BinaryOp.SUB and isinstance(left.right, NumberLiteral):
                    # (x - c1) + c2 -> x + (c2 - c1)
                    new_const = (right.value - left.right.value) & 0xFFFF
                    if new_const == 0:
                        return left.left
                    return BinaryExpr(BinaryOp.ADD, left.left,
                                     NumberLiteral(new_const, span=span), span=span)

        # (x - c1) - c2 -> x - (c1 + c2)
        if op == BinaryOp.SUB:
            if isinstance(right, NumberLiteral) and isinstance(left, BinaryExpr):
                if left.op == BinaryOp.SUB and isinstance(left.right, NumberLiteral):
                    new_const = (left.right.value + right.value) & 0xFFFF
                    return BinaryExpr(BinaryOp.SUB, left.left,
                                     NumberLiteral(new_const, span=span), span=span)
                if left.op == BinaryOp.ADD and isinstance(left.right, NumberLiteral):
                    # (x + c1) - c2 -> x + (c1 - c2) or x - (c2 - c1)
                    diff = left.right.value - right.value
                    if diff == 0:
                        return left.left
                    if diff > 0:
                        return BinaryExpr(BinaryOp.ADD, left.left,
                                         NumberLiteral(diff & 0xFFFF, span=span), span=span)
                    else:
                        return BinaryExpr(BinaryOp.SUB, left.left,
                                         NumberLiteral((-diff) & 0xFFFF, span=span), span=span)

        # x MOD 1 = 0
        if op == BinaryOp.MOD:
            if isinstance(right, NumberLiteral) and right.value == 1:
                return NumberLiteral(0, span=span)

        # 0 / x = 0 (unless x is 0, but we can't check that)
        if op == BinaryOp.DIV:
            if isinstance(left, NumberLiteral) and left.value == 0:
                return NumberLiteral(0, span=span)

        # 0 MOD x = 0
        if op == BinaryOp.MOD:
            if isinstance(left, NumberLiteral) and left.value == 0:
                return NumberLiteral(0, span=span)

        return None

    def _boolean_simplify(
        self, op: BinaryOp, left: Expr, right: Expr, span
    ) -> Expr | None:
        """Apply boolean and comparison simplifications."""
        # x = x -> FFFF (always true)
        if op == BinaryOp.EQ:
            if isinstance(left, Identifier) and isinstance(right, Identifier):
                if left.name == right.name:
                    return NumberLiteral(0xFFFF, span=span)
            # 0 = 0 is handled by constant folding, but check nested
            if isinstance(left, NumberLiteral) and isinstance(right, NumberLiteral):
                return NumberLiteral(0xFFFF if left.value == right.value else 0, span=span)

        # x <> x -> 0 (always false)
        if op == BinaryOp.NE:
            if isinstance(left, Identifier) and isinstance(right, Identifier):
                if left.name == right.name:
                    return NumberLiteral(0, span=span)

        # x < x -> 0 (always false)
        if op == BinaryOp.LT:
            if isinstance(left, Identifier) and isinstance(right, Identifier):
                if left.name == right.name:
                    return NumberLiteral(0, span=span)

        # x > x -> 0 (always false)
        if op == BinaryOp.GT:
            if isinstance(left, Identifier) and isinstance(right, Identifier):
                if left.name == right.name:
                    return NumberLiteral(0, span=span)

        # x <= x -> FFFF (always true)
        if op == BinaryOp.LE:
            if isinstance(left, Identifier) and isinstance(right, Identifier):
                if left.name == right.name:
                    return NumberLiteral(0xFFFF, span=span)

        # x >= x -> FFFF (always true)
        if op == BinaryOp.GE:
            if isinstance(left, Identifier) and isinstance(right, Identifier):
                if left.name == right.name:
                    return NumberLiteral(0xFFFF, span=span)

        # (a AND b) AND b -> a AND b (idempotent)
        if op == BinaryOp.AND:
            if isinstance(left, BinaryExpr) and left.op == BinaryOp.AND:
                # Check if right matches left.right or left.left
                if (isinstance(right, Identifier) and isinstance(left.right, Identifier)
                        and right.name == left.right.name):
                    return left
                if (isinstance(right, Identifier) and isinstance(left.left, Identifier)
                        and right.name == left.left.name):
                    return left

        # (a OR b) OR b -> a OR b (idempotent)
        if op == BinaryOp.OR:
            if isinstance(left, BinaryExpr) and left.op == BinaryOp.OR:
                if (isinstance(right, Identifier) and isinstance(left.right, Identifier)
                        and right.name == left.right.name):
                    return left
                if (isinstance(right, Identifier) and isinstance(left.left, Identifier)
                        and right.name == left.left.name):
                    return left

        # x AND x -> x (idempotent)
        if op == BinaryOp.AND:
            if isinstance(left, Identifier) and isinstance(right, Identifier):
                if left.name == right.name:
                    return left

        # x OR x -> x (idempotent)
        if op == BinaryOp.OR:
            if isinstance(left, Identifier) and isinstance(right, Identifier):
                if left.name == right.name:
                    return left

        return None

    def _optimize_builtin_call(self, name: str, args: list[Expr]) -> Expr | None:
        """Optimize calls to built-in functions with constant args."""
        if len(args) == 0:
            return None

        # LOW(const) -> const & 0xFF
        if name == "LOW" and isinstance(args[0], NumberLiteral):
            return NumberLiteral(args[0].value & 0xFF, span=args[0].span)

        # HIGH(const) -> (const >> 8) & 0xFF
        if name == "HIGH" and isinstance(args[0], NumberLiteral):
            return NumberLiteral((args[0].value >> 8) & 0xFF, span=args[0].span)

        # DOUBLE(const) -> const (already 16-bit conceptually)
        if name == "DOUBLE" and isinstance(args[0], NumberLiteral):
            return NumberLiteral(args[0].value & 0xFFFF, span=args[0].span)

        # SHL(const, const)
        if name == "SHL" and len(args) == 2:
            if isinstance(args[0], NumberLiteral) and isinstance(args[1], NumberLiteral):
                result = (args[0].value << args[1].value) & 0xFFFF
                return NumberLiteral(result, span=args[0].span)

        # SHR(const, const)
        if name == "SHR" and len(args) == 2:
            if isinstance(args[0], NumberLiteral) and isinstance(args[1], NumberLiteral):
                result = (args[0].value >> args[1].value) & 0xFFFF
                return NumberLiteral(result, span=args[0].span)

        # ROL(const, const)
        if name == "ROL" and len(args) == 2:
            if isinstance(args[0], NumberLiteral) and isinstance(args[1], NumberLiteral):
                val = args[0].value & 0xFF
                count = args[1].value & 7
                result = ((val << count) | (val >> (8 - count))) & 0xFF
                return NumberLiteral(result, span=args[0].span)

        # ROR(const, const)
        if name == "ROR" and len(args) == 2:
            if isinstance(args[0], NumberLiteral) and isinstance(args[1], NumberLiteral):
                val = args[0].value & 0xFF
                count = args[1].value & 7
                result = ((val >> count) | (val << (8 - count))) & 0xFF
                return NumberLiteral(result, span=args[0].span)

        return None

    def _log2_if_power_of_2(self, n: int) -> int | None:
        """Return log2(n) if n is a power of 2, else None."""
        if n <= 0:
            return None
        if n & (n - 1) != 0:
            return None
        return n.bit_length() - 1


def optimize_ast(module: Module, opt_level: int = 2,
                 optimize_for: OptimizeFor = OptimizeFor.BALANCED) -> Module:
    """Convenience function to optimize a module's AST."""
    optimizer = ASTOptimizer(opt_level, optimize_for)
    return optimizer.optimize(module)
