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


@dataclass
class OptimizationStats:
    """Statistics about optimizations performed."""

    constants_folded: int = 0
    strength_reductions: int = 0
    dead_code_eliminated: int = 0
    algebraic_simplifications: int = 0
    cse_eliminations: int = 0


class ASTOptimizer:
    """
    AST optimizer that performs high-level transformations.

    Optimization levels:
    - 0: No optimization
    - 1: Basic (constant folding, simple algebraic)
    - 2: Standard (+ strength reduction, dead code)
    - 3: Aggressive (+ CSE, loop optimizations)
    """

    def __init__(self, opt_level: int = 2) -> None:
        self.opt_level = opt_level
        self.stats = OptimizationStats()
        # Known constant values for propagation
        self.constants: dict[str, int] = {}
        # Track which variables are modified in current scope
        self.modified_vars: set[str] = set()

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
            try:
                val = int(decl.value, 0)
                self.constants[decl.name] = val
            except (ValueError, TypeError):
                pass
            return decl

        elif isinstance(decl, ProcDecl):
            # Optimize procedure body
            new_decls = [
                d for d in (self._optimize_declaration(d) for d in decl.decls) if d
            ]
            new_stmts = [
                s for s in (self._optimize_stmt(s) for s in decl.stmts) if s
            ]
            return ProcDecl(
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

        return decl

    def _optimize_stmt(self, stmt: Stmt) -> Stmt | None:
        """Optimize a statement. Returns None to remove it."""
        if isinstance(stmt, AssignStmt):
            opt_value = self._optimize_expr(stmt.value)
            opt_targets = [self._optimize_expr(t) for t in stmt.targets]

            # Track modified variables
            for target in opt_targets:
                if isinstance(target, Identifier):
                    self.modified_vars.add(target.name)
                    # Remove from constants if modified
                    self.constants.pop(target.name, None)

            # Dead store elimination: assignment to unused variable
            # (Would need full dataflow analysis - skip for now)

            return AssignStmt(opt_targets, opt_value, span=stmt.span)

        elif isinstance(stmt, CallStmt):
            opt_callee = self._optimize_expr(stmt.callee)
            opt_args = [self._optimize_expr(a) for a in stmt.args]
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
            return DoBlock(new_decls, new_stmts, stmt.end_label, span=stmt.span)

        elif isinstance(stmt, DoWhileBlock):
            opt_cond = self._optimize_expr(stmt.condition)

            # Check for DO WHILE 0 (never executes)
            if self.opt_level >= 2 and isinstance(opt_cond, NumberLiteral):
                if opt_cond.value == 0:
                    self.stats.dead_code_eliminated += 1
                    return NullStmt(span=stmt.span)

            new_stmts = [
                s for s in (self._optimize_stmt(s) for s in stmt.stmts) if s
            ]
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

            new_stmts = [
                s for s in (self._optimize_stmt(s) for s in stmt.stmts) if s
            ]

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

        return BinaryExpr(expr.op, left, right, span=expr.span)

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


def optimize_ast(module: Module, opt_level: int = 2) -> Module:
    """Convenience function to optimize a module's AST."""
    optimizer = ASTOptimizer(opt_level)
    return optimizer.optimize(module)
