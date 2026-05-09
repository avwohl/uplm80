"""
Code Generator for PL/M-80.

Generates Z80 assembly code from the optimized AST.
Outputs MACRO-80 compatible .MAC files.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Iterator

from .ast_nodes import (
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
from .symbols import SymbolTable, Symbol, SymbolKind
from .errors import CodeGenError
from .runtime import get_runtime_library


class Target(Enum):
    """Target processor."""

    Z80 = auto()  # Z80 only - 8080 target removed


class Mode(Enum):
    """Runtime environment mode."""

    CPM = auto()   # CP/M program (ORG 100H, stack from BDOS, return to OS)
    BARE = auto()  # Bare metal program (original Intel PL/M style)


class RegState(Enum):
    """State of a register in the allocator."""

    FREE = auto()      # Available for use
    BUSY = auto()      # Contains live value, in use
    SPILLED = auto()   # Value saved to stack, register reused


class RegClass(Enum):
    """Register classes for allocation requests."""

    BYTE = auto()      # Need A register
    ADDR = auto()      # Need HL (primary 16-bit)
    ADDR_ALT = auto()  # Need DE or BC (secondary 16-bit)
    INDEX = auto()     # Need IX or IY


@dataclass
class RegDescriptor:
    """Descriptor tracking state of a single register."""

    state: RegState = RegState.FREE
    owner: str = ""           # Debug: what claimed this register
    spill_depth: int = 0      # Stack depth when spilled (for nested spills)
    contents: str = ""        # Debug: description of contents


@dataclass
class RegisterAllocator:
    """
    Tracks register state and manages allocation.

    This implements demand-driven register allocation with automatic spilling.
    When code needs a register that's busy, it's automatically saved to the
    stack and restored when released.

    Usage:
        # Claim a register (spills if busy)
        self.regs.need_reg('de', 'binary_left', self._emit)

        # Release when done (restores if spilled)
        self.regs.release_reg('de', self._emit)

        # Or use context manager for scoped usage
        with self.regs.with_reg('de', 'binary_left', self._emit):
            # DE is claimed here
            ...
        # DE automatically released
    """

    # Register descriptors
    a: RegDescriptor = field(default_factory=RegDescriptor)
    hl: RegDescriptor = field(default_factory=RegDescriptor)
    de: RegDescriptor = field(default_factory=RegDescriptor)
    bc: RegDescriptor = field(default_factory=RegDescriptor)
    ix: RegDescriptor = field(default_factory=RegDescriptor)

    # Stack tracking for spilled registers
    spill_stack: list[str] = field(default_factory=list)

    # Statistics for debugging/optimization
    stats: dict[str, int] = field(default_factory=dict)

    def get_reg(self, name: str) -> RegDescriptor:
        """Get descriptor by register name."""
        return getattr(self, name.lower())

    def is_busy(self, reg: str) -> bool:
        """Check if a register is currently busy."""
        return self.get_reg(reg).state == RegState.BUSY

    def is_free(self, reg: str) -> bool:
        """Check if a register is currently free."""
        return self.get_reg(reg).state == RegState.FREE

    def need_reg(self, reg_or_class: str | RegClass, owner: str,
                 emit_fn: Callable[[str, str], None]) -> str:
        """
        Request a register. Returns the register name.
        If busy, automatically spills it first.

        Args:
            reg_or_class: Specific register name ('hl', 'de') or RegClass
            owner: Debug string identifying the requester
            emit_fn: Callback to emit assembly (emit_fn('push', 'hl'))

        Returns:
            The allocated register name
        """
        # Resolve class to specific register
        if isinstance(reg_or_class, RegClass):
            reg = self._pick_reg_from_class(reg_or_class)
        else:
            reg = reg_or_class.lower()

        desc = self.get_reg(reg)

        if desc.state == RegState.BUSY:
            # Must spill - save current contents to stack
            self._spill_reg(reg, emit_fn)

        # Mark as busy with new owner
        desc.state = RegState.BUSY
        desc.owner = owner
        self.stats['claims'] = self.stats.get('claims', 0) + 1
        return reg

    def _spill_reg(self, reg: str, emit_fn: Callable[[str, str], None]) -> None:
        """Spill a register to the stack."""
        desc = self.get_reg(reg)
        # For 'a', we need to push af
        push_reg = 'af' if reg == 'a' else reg
        emit_fn("push", push_reg)
        self.spill_stack.append(reg)
        desc.spill_depth = len(self.spill_stack)
        desc.state = RegState.SPILLED
        self.stats['spills'] = self.stats.get('spills', 0) + 1

    def release_reg(self, reg: str, emit_fn: Callable[[str, str], None]) -> None:
        """
        Release a register. If it was spilled, restore it.

        Args:
            reg: Register name to release
            emit_fn: Callback to emit assembly
        """
        reg = reg.lower()
        desc = self.get_reg(reg)

        # Check if we need to restore from spill
        if self.spill_stack and self.spill_stack[-1] == reg:
            # This register was spilled and is top of stack - restore it
            pop_reg = 'af' if reg == 'a' else reg
            emit_fn("pop", pop_reg)
            self.spill_stack.pop()
            self.stats['restores'] = self.stats.get('restores', 0) + 1

        desc.state = RegState.FREE
        desc.owner = ""
        desc.spill_depth = 0

    @contextmanager
    def with_reg(self, reg: str, owner: str,
                 emit_fn: Callable[[str, str], None]) -> Iterator[str]:
        """Context manager for scoped register use."""
        self.need_reg(reg, owner, emit_fn)
        try:
            yield reg
        finally:
            self.release_reg(reg, emit_fn)

    def _pick_reg_from_class(self, cls: RegClass) -> str:
        """Pick best register from class, preferring free ones."""
        candidates = {
            RegClass.BYTE: ['a'],
            RegClass.ADDR: ['hl'],
            RegClass.ADDR_ALT: ['de', 'bc'],
            RegClass.INDEX: ['ix'],
        }

        for reg in candidates[cls]:
            if self.get_reg(reg).state == RegState.FREE:
                return reg

        # All busy - return first (will be spilled)
        return candidates[cls][0]

    def mark_busy(self, reg: str, owner: str = "") -> None:
        """Mark a register as busy without spilling (for tracking existing code)."""
        desc = self.get_reg(reg.lower())
        desc.state = RegState.BUSY
        desc.owner = owner

    def mark_free(self, reg: str) -> None:
        """Mark a register as free (for tracking existing code)."""
        desc = self.get_reg(reg.lower())
        desc.state = RegState.FREE
        desc.owner = ""
        desc.spill_depth = 0

    def reset(self) -> None:
        """Reset all registers to free state."""
        for reg in ['a', 'hl', 'de', 'bc', 'ix']:
            desc = self.get_reg(reg)
            desc.state = RegState.FREE
            desc.owner = ""
            desc.spill_depth = 0
        self.spill_stack.clear()

    def get_status(self) -> str:
        """Get human-readable status of all registers (for debugging)."""
        parts = []
        for reg in ['a', 'hl', 'de', 'bc', 'ix']:
            desc = self.get_reg(reg)
            state = desc.state.name[0]  # F, B, or S
            owner = f":{desc.owner}" if desc.owner else ""
            parts.append(f"{reg.upper()}={state}{owner}")
        return " ".join(parts)


@dataclass
class AsmLine:
    """A single line of assembly output."""

    label: str = ""
    opcode: str = ""
    operands: str = ""
    comment: str = ""

    def __str__(self) -> str:
        parts: list[str] = []
        if self.label:
            parts.append(f"{self.label}:")
        if self.opcode:
            if self.label:
                parts.append("\t")
            else:
                parts.append("\t")
            parts.append(self.opcode)
            if self.operands:
                parts.append(f"\t{self.operands}")
        if self.comment:
            if parts:
                parts.append(f"\t; {self.comment}")
            else:
                parts.append(f"; {self.comment}")
        return "".join(parts)


class CodeGenerator:
    """
    Generates assembly code from PL/M-80 AST.

    The code generator uses a simple stack-based approach for expressions,
    with the accumulator (A) as the primary working register and HL for
    addresses and 16-bit values.
    """

    # Reserved assembler names that conflict with 8080/Z80 registers
    RESERVED_NAMES = {'A', 'B', 'C', 'D', 'E', 'H', 'L', 'M', 'SP', 'PSW',
                      'AF', 'BC', 'DE', 'HL', 'IX', 'IY', 'I', 'R'}

    def __init__(self, target: Target = Target.Z80, mode: Mode = Mode.CPM, warn_trivial_if: bool = True, reg_debug: bool = False) -> None:
        self.target = target
        self.mode = mode
        self.warn_trivial_if = warn_trivial_if  # Warn on IF 0 / IF 1
        self.reg_debug = reg_debug  # Enable register tracking debug output
        self.warnings: list[str] = []  # Collected warnings
        self.symbols = SymbolTable()
        self.output: list[AsmLine] = []
        self.label_counter = 0
        self.string_counter = 0
        self.data_segment: list[AsmLine] = []
        self.code_data_segment: list[AsmLine] = []  # DATA values emitted inline in code
        self.string_literals: list[tuple[str, str]] = []  # (label, value)
        self.current_proc: str | None = None
        self.current_proc_decl: ProcDecl | None = None
        self.loop_stack: list[tuple[str, str]] = []  # (continue_label, break_label)
        self.needs_runtime: set[str] = set()  # Which runtime routines are needed
        self.needs_end_symbol = False  # Whether __END__ (linker symbol) is needed
        self.literal_macros: dict[str, str] = {}  # LITERALLY macro expansions
        self.block_scope_counter = 0  # Counter for unique DO block scopes
        self.emit_data_inline = False  # If True, DATA goes to code segment
        # Call graph for parameter sharing optimization
        self.call_graph: dict[str, set[str]] = {}  # proc -> set of procs it calls
        self.can_be_active_together: dict[str, set[str]] = {}  # proc -> procs that can be on stack with it
        self.param_slots: dict[str, int] = {}  # param_key -> slot number
        self.slot_storage: list[tuple[str, int]] = []  # (label, size) for each slot
        self.proc_params: dict[str, list[tuple[str, str, DataType, int]]] = {}  # proc -> [(name, asm_name, type, size)]
        # For liveness analysis: remaining statements in current scope
        self.pending_stmts: list[Stmt] = []
        # For tracking embedded assignment target for return optimization
        self.embedded_assign_target: str | None = None  # Variable name of last embedded assignment
        # Current IF statement being processed (for embedded assign optimization)
        self.current_if_stmt: IfStmt | None = None
        # Flag: A register contains L (low byte of HL) - for avoiding redundant ld a,L
        self.a_has_l: bool = False
        # Register allocator for automatic spill/restore
        self.regs = RegisterAllocator()

    def _parse_plm_number(self, s: str) -> int:
        """Parse a PL/M-style numeric literal (handles $ separators and B/H/O/Q/D suffixes)."""
        # Remove $ digit separators and convert to uppercase
        s = s.upper().replace("$", "")
        if s.endswith("H"):
            return int(s[:-1], 16)
        elif s.endswith("B"):
            return int(s[:-1], 2)
        elif s.endswith("O") or s.endswith("Q"):
            return int(s[:-1], 8)
        elif s.endswith("D"):
            return int(s[:-1], 10)
        else:
            return int(s, 0)  # Let Python auto-detect base (0x, 0b, 0o prefixes)

    def _mangle_name(self, name: str) -> str:
        """Mangle variable names that conflict with assembler reserved words."""
        if name.upper() in self.RESERVED_NAMES:
            return f"@{name}"
        return name

    def _get_const_byte_value(self, expr: Expr) -> int | None:
        """Extract a constant byte value from an expression if possible.

        Returns the constant value (0-255) or None if not a constant.
        Handles NumberLiteral, StringLiteral (single char), and LITERALLY macros.
        """
        if isinstance(expr, NumberLiteral):
            if expr.value <= 255:
                return expr.value
        elif isinstance(expr, StringLiteral):
            if len(expr.value) == 1:
                return ord(expr.value[0])
        elif isinstance(expr, Identifier):
            if expr.name in self.literal_macros:
                try:
                    val = self._parse_plm_number(self.literal_macros[expr.name])
                    if val <= 255:
                        return val
                except ValueError:
                    pass
        return None

    def _try_eval_const(self, expr: Expr) -> int | None:
        """Try to evaluate an expression as a compile-time constant.

        Returns the integer value or None if not a constant.
        Handles NumberLiteral, StringLiteral, LITERALLY macros, and UnaryExpr(NEG).
        Values are returned as-is (may be negative or > 255).
        """
        if isinstance(expr, NumberLiteral):
            return expr.value
        elif isinstance(expr, StringLiteral):
            if len(expr.value) == 1:
                return ord(expr.value[0])
            return None
        elif isinstance(expr, Identifier):
            if expr.name in self.literal_macros:
                try:
                    return self._parse_plm_number(self.literal_macros[expr.name])
                except ValueError:
                    pass
        elif isinstance(expr, UnaryExpr):
            if expr.op == UnaryOp.NEG:
                operand_val = self._try_eval_const(expr.operand)
                if operand_val is not None:
                    # Return negative value (may be -1, -255, etc.)
                    return -operand_val
            elif expr.op == UnaryOp.NOT:
                operand_val = self._try_eval_const(expr.operand)
                if operand_val is not None:
                    # Bitwise NOT - complement within 16 bits
                    return (~operand_val) & 0xFFFF
        elif isinstance(expr, BinaryExpr):
            left_val = self._try_eval_const(expr.left)
            right_val = self._try_eval_const(expr.right)
            if left_val is not None and right_val is not None:
                if expr.op == BinaryOp.ADD:
                    return (left_val + right_val) & 0xFFFF
                elif expr.op == BinaryOp.SUB:
                    return (left_val - right_val) & 0xFFFF
                elif expr.op == BinaryOp.AND:
                    return left_val & right_val
                elif expr.op == BinaryOp.OR:
                    return left_val | right_val
                elif expr.op == BinaryOp.XOR:
                    return left_val ^ right_val
        return None

    def _check_impossible_comparison(self, left: Expr, right: Expr, op: BinaryOp) -> None:
        """Check for comparisons that can never or always be true and raise an error.

        For BYTE compared to constant outside 0-255:
        - For = and <>, allow truncation only for "negative byte" values (0xFF00-0xFFFF, i.e. -256 to -1)
        - For <, >, <=, >=, these are always true/false so we error
        """
        left_type = self._get_expr_type(left)
        right_val = self._try_eval_const(right)

        if left_type == DataType.BYTE and right_val is not None:
            # For BYTE comparisons, check if value is outside 0-255 range
            if right_val < 0:
                unsigned_val = right_val & 0xFFFF
            else:
                unsigned_val = right_val

            if unsigned_val > 255:
                from .errors import CodeGenError, SourceLocation
                loc = None
                if hasattr(right, 'span') and right.span:
                    loc = SourceLocation(right.span.start_line, right.span.start_col)

                # For = and <>, allow truncation only for "negative byte" values (high byte is 0xFF)
                # This handles BYTE <> -1 (0xFFFF -> 0xFF) but catches BYTE <> 0x123
                if op in (BinaryOp.EQ, BinaryOp.NE):
                    if (unsigned_val & 0xFF00) == 0xFF00:
                        return  # Valid: -256 to -1 range, will truncate to byte
                    # Otherwise, error - constant like 256 or 0x123 shouldn't be compared to BYTE
                    byte_val = unsigned_val & 0xFF
                    if op == BinaryOp.EQ:
                        msg = f"comparison BYTE = {unsigned_val} is always false (BYTE can only hold 0-255; truncating to {byte_val} would change semantics)"
                    else:
                        msg = f"comparison BYTE <> {unsigned_val} is always true (BYTE can only hold 0-255; truncating to {byte_val} would change semantics)"
                    raise CodeGenError(msg, loc)

                # For ordering comparisons, values outside 0-255 give always true/false
                if op == BinaryOp.LT:
                    msg = f"comparison BYTE < {right_val} is always true (BYTE can only hold 0-255)"
                elif op == BinaryOp.LE:
                    msg = f"comparison BYTE <= {right_val} is always true (BYTE can only hold 0-255)"
                elif op == BinaryOp.GT:
                    msg = f"comparison BYTE > {right_val} is always false (BYTE can only hold 0-255)"
                elif op == BinaryOp.GE:
                    msg = f"comparison BYTE >= {right_val} is always false (BYTE can only hold 0-255)"
                else:
                    return  # Unknown comparison operator

                raise CodeGenError(msg, loc)

    def _check_trivial_condition(self, condition: Expr, context: str = "condition") -> None:
        """Check for trivial constant conditions and raise an error.

        Detects cases like:
        - DO WHILE 1 (always true - infinite loop)
        - DO WHILE 0 (never executes)
        """
        const_val = self._try_eval_const(condition)
        if const_val is not None:
            from .errors import CodeGenError, SourceLocation
            loc = None
            if hasattr(condition, 'span') and condition.span:
                loc = SourceLocation(condition.span.start_line, condition.span.start_col)

            if const_val == 0:
                msg = f"{context} is always false (constant 0)"
            else:
                msg = f"{context} is always true (constant {const_val})"

            raise CodeGenError(msg, loc)

    def _warn_trivial_if(self, condition: Expr) -> None:
        """Emit a warning for trivial IF conditions (IF 0, IF 1).

        Unlike DO WHILE, trivial IF conditions don't cause infinite loops
        so they're only warnings, not errors.
        """
        if not self.warn_trivial_if:
            return

        const_val = self._try_eval_const(condition)
        if const_val is not None:
            from .errors import SourceLocation
            loc = None
            if hasattr(condition, 'span') and condition.span:
                loc = SourceLocation(condition.span.start_line, condition.span.start_col)

            if const_val == 0:
                msg = f"IF condition is always false (constant 0)"
            else:
                msg = f"IF condition is always true (constant {const_val})"

            if loc:
                warning = f"{loc}: warning: {msg}"
            else:
                warning = f"warning: {msg}"
            self.warnings.append(warning)

    # ========================================================================
    # Loop Index Usage Analysis
    # ========================================================================

    def _var_used_in_expr(self, var_name: str, expr: Expr) -> bool:
        """Check if variable is referenced in expression."""
        if isinstance(expr, Identifier):
            return expr.name == var_name
        elif isinstance(expr, NumberLiteral) or isinstance(expr, StringLiteral):
            return False
        elif isinstance(expr, BinaryExpr):
            return self._var_used_in_expr(var_name, expr.left) or self._var_used_in_expr(var_name, expr.right)
        elif isinstance(expr, UnaryExpr):
            return self._var_used_in_expr(var_name, expr.operand)
        elif isinstance(expr, CallExpr):
            for arg in expr.args:
                if self._var_used_in_expr(var_name, arg):
                    return True
            if isinstance(expr.callee, Expr):
                return self._var_used_in_expr(var_name, expr.callee)
            return False
        elif isinstance(expr, SubscriptExpr):
            if self._var_used_in_expr(var_name, expr.index):
                return True
            if isinstance(expr.base, Expr):
                return self._var_used_in_expr(var_name, expr.base)
            return False
        elif isinstance(expr, MemberExpr):
            if isinstance(expr.base, Expr):
                return self._var_used_in_expr(var_name, expr.base)
            return False
        elif isinstance(expr, LocationExpr):
            return self._var_used_in_expr(var_name, expr.operand)
        elif isinstance(expr, EmbeddedAssignExpr):
            return self._var_used_in_expr(var_name, expr.target) or self._var_used_in_expr(var_name, expr.value)
        return False

    def _var_used_in_stmt(self, var_name: str, stmt: Stmt) -> bool:
        """Check if variable is referenced in statement."""
        if isinstance(stmt, AssignStmt):
            # Check if var is read (on RHS or in index of LHS)
            if self._var_used_in_expr(var_name, stmt.value):
                return True
            # Check if var is used in index of target (targets is a list)
            for target in stmt.targets:
                if isinstance(target, SubscriptExpr):
                    if self._var_used_in_expr(var_name, target.index):
                        return True
            return False
        elif isinstance(stmt, CallStmt):
            # Check callee and all arguments for variable usage
            if isinstance(stmt.callee, Expr) and self._var_used_in_expr(var_name, stmt.callee):
                return True
            for arg in stmt.args:
                if self._var_used_in_expr(var_name, arg):
                    return True
            return False
        elif isinstance(stmt, ReturnStmt):
            if stmt.value:
                return self._var_used_in_expr(var_name, stmt.value)
            return False
        elif isinstance(stmt, IfStmt):
            if self._var_used_in_expr(var_name, stmt.condition):
                return True
            if self._var_used_in_stmt(var_name, stmt.then_stmt):
                return True
            if stmt.else_stmt and self._var_used_in_stmt(var_name, stmt.else_stmt):
                return True
            return False
        elif isinstance(stmt, DoBlock):
            for s in stmt.stmts:
                if self._var_used_in_stmt(var_name, s):
                    return True
            return False
        elif isinstance(stmt, DoWhileBlock):
            if self._var_used_in_expr(var_name, stmt.condition):
                return True
            for s in stmt.stmts:
                if self._var_used_in_stmt(var_name, s):
                    return True
            return False
        elif isinstance(stmt, DoIterBlock):
            # Don't recurse into nested DO-ITER as inner loop var shadows outer
            if self._var_used_in_expr(var_name, stmt.start):
                return True
            if self._var_used_in_expr(var_name, stmt.bound):
                return True
            if stmt.step and self._var_used_in_expr(var_name, stmt.step):
                return True
            for s in stmt.stmts:
                if self._var_used_in_stmt(var_name, s):
                    return True
            return False
        elif isinstance(stmt, DoCaseBlock):
            if self._var_used_in_expr(var_name, stmt.selector):
                return True
            for case_stmts in stmt.cases:
                for s in case_stmts:
                    if self._var_used_in_stmt(var_name, s):
                        return True
            return False
        elif isinstance(stmt, LabeledStmt):
            return self._var_used_in_stmt(var_name, stmt.stmt)
        return False

    def _index_used_in_body(self, index_var: Expr, stmts: list[Stmt]) -> bool:
        """Check if loop index variable is used in loop body."""
        if isinstance(index_var, Identifier):
            var_name = index_var.name
            for stmt in stmts:
                if self._var_used_in_stmt(var_name, stmt):
                    return True
        return False

    # ========================================================================
    # Register Liveness Analysis
    # ========================================================================

    def _expr_clobbers_a(self, expr: Expr) -> bool:
        """Check if evaluating expression will clobber A register.

        Most expressions clobber A because they compute into A (for BYTE) or use A
        as a scratch register. Only certain simple operations preserve A.
        """
        if isinstance(expr, NumberLiteral):
            return False  # ld hl,const doesn't touch A

        if isinstance(expr, Identifier):
            # Loading a variable clobbers A (for BYTE) or doesn't touch A (for ADDRESS in HL)
            sym = self._lookup_symbol(expr.name)
            if sym and sym.data_type == DataType.BYTE:
                return True  # ld a,(addr) clobbers A
            return False  # ld hl,(addr) doesn't clobber A

        if isinstance(expr, BinaryExpr):
            # Check expression type - ADDRESS operations use HL, not A
            expr_type = self._get_expr_type(expr)
            if expr_type == DataType.ADDRESS:
                # ADDRESS arithmetic uses add hl,rp which doesn't clobber A
                # But we need to check if operands clobber A
                op = expr.op
                if op == BinaryOp.ADD:
                    # ld hl,(addr), add hl,rp preserves A
                    left_clobbers = self._expr_clobbers_a(expr.left)
                    right_clobbers = self._expr_clobbers_a(expr.right)
                    return left_clobbers or right_clobbers
            # BYTE operations and other ADDRESS ops may clobber A
            return True

        # Most other expressions clobber A
        return True

    def _stmt_clobbers_a(self, stmt: Stmt) -> bool:
        """Check if a statement will clobber the A register.

        This is used for liveness analysis to determine if we need to save A
        across an IF block or other control structure.
        """
        if isinstance(stmt, NullStmt):
            return False

        if isinstance(stmt, LabeledStmt):
            return self._stmt_clobbers_a(stmt.stmt)

        if isinstance(stmt, AssignStmt):
            # Assignment to HL-based variable (ADDRESS type) without touching A
            # Check if all targets are ADDRESS type
            for target in stmt.targets:
                if isinstance(target, Identifier):
                    sym = self._lookup_symbol(target.name)
                    if not sym or sym.data_type == DataType.BYTE:
                        return True  # BYTE assignment uses LD (addr),A -> doesn't clobber but value changes
                else:
                    return True  # Complex target likely clobbers A
            # Check if value expression clobbers A
            return self._expr_clobbers_a(stmt.value)

        if isinstance(stmt, CallStmt):
            # Procedure calls clobber A
            return True

        if isinstance(stmt, ReturnStmt):
            # Return may load a value into A
            if stmt.value:
                return True
            return False

        if isinstance(stmt, GotoStmt):
            return False  # jp doesn't clobber A

        if isinstance(stmt, HaltStmt):
            return False  # halt doesn't clobber A

        if isinstance(stmt, EnableStmt) or isinstance(stmt, DisableStmt):
            return False  # EI/DI don't clobber A

        if isinstance(stmt, IfStmt):
            # IF condition evaluation may clobber A
            # But we special-case conditions that don't change A

            # Simple identifier test: or a doesn't change A
            if isinstance(stmt.condition, Identifier):
                cond_type = self._get_expr_type(stmt.condition)
                if cond_type == DataType.BYTE:
                    # ld a,(x); or a - ld a,(x) clobbers A, so this does clobber
                    return True
                # For ADDRESS: ld a,L; or h - this clobbers A
                return True

            # Comparisons: cp n doesn't change A
            if isinstance(stmt.condition, BinaryExpr):
                op = stmt.condition.op
                if op in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.GT, BinaryOp.LE, BinaryOp.GE):
                    # Comparison: cp n doesn't clobber A, but we need to check
                    # if the left side is already in A or requires loading
                    left_type = self._get_expr_type(stmt.condition.left)
                    if left_type == DataType.BYTE:
                        # For byte comparisons, if right is constant, uses cp n which preserves A
                        if isinstance(stmt.condition.right, NumberLiteral):
                            # Check if then/else branches clobber A
                            then_clobbers = self._stmt_clobbers_a(stmt.then_stmt)
                            else_clobbers = stmt.else_stmt and self._stmt_clobbers_a(stmt.else_stmt)
                            return then_clobbers or else_clobbers

            return True  # Conservative: condition evaluation clobbers A

        if isinstance(stmt, (DoBlock, DoWhileBlock, DoIterBlock, DoCaseBlock)):
            # Loop bodies likely clobber A
            return True

        if isinstance(stmt, DeclareStmt):
            return False  # Declarations don't generate code

        # Default: assume clobbers A
        return True

    def _a_survives_stmts(self, stmts: list[Stmt]) -> bool:
        """Check if A register survives through a list of statements.

        Returns True if A is preserved, False if any statement clobbers A.
        """
        for stmt in stmts:
            if self._stmt_clobbers_a(stmt):
                return False
        return True

    def _lookup_symbol(self, name: str) -> Symbol | None:
        """Look up a symbol in the current scope hierarchy."""
        # Check for LITERALLY macro first
        if name in self.literal_macros:
            return None  # Literals are not symbols

        # Look up in scope hierarchy
        sym = None
        if self.current_proc:
            parts = self.current_proc.split('$')
            for i in range(len(parts), 0, -1):
                scoped_name = '$'.join(parts[:i]) + '$' + name
                sym = self.symbols.lookup(scoped_name)
                if sym:
                    break
        if sym is None:
            sym = self.symbols.lookup(name)
        return sym

    # ========================================================================
    # Call Graph Analysis and Storage Sharing
    # ========================================================================

    def _build_call_graph(self, module: Module) -> None:
        """Build call graph by analyzing all procedure bodies."""
        self.call_graph = {}
        self.proc_storage: dict[str, list[tuple[str, int, DataType]]] = {}  # proc -> [(var_name, size, type)]

        # First pass: collect all procedure names
        all_procs: set[str] = set()
        self._collect_proc_names(module.decls, None, all_procs)

        # Initialize call graph
        for proc in all_procs:
            self.call_graph[proc] = set()

        # Second pass: analyze calls in each procedure
        for decl in module.decls:
            if isinstance(decl, ProcDecl) and not decl.is_external:
                self._analyze_proc_calls(decl, None)

    def _collect_proc_names(self, decls: list, parent_proc: str | None, all_procs: set[str]) -> None:
        """Recursively collect all procedure names."""
        for decl in decls:
            if isinstance(decl, ProcDecl):
                if parent_proc and not decl.is_public and not decl.is_external:
                    full_name = f"{parent_proc}${decl.name}"
                else:
                    full_name = decl.name
                all_procs.add(full_name)
                # Recurse into nested procedures
                if decl.decls:
                    self._collect_proc_names(decl.decls, full_name, all_procs)
                # Also check statements for nested procedures
                for stmt in decl.stmts:
                    if isinstance(stmt, DeclareStmt):
                        self._collect_proc_names(stmt.declarations, full_name, all_procs)

    def _analyze_proc_calls(self, decl: ProcDecl, parent_proc: str | None) -> None:
        """Analyze a procedure to find all calls it makes."""
        if parent_proc and not decl.is_public and not decl.is_external:
            full_name = f"{parent_proc}${decl.name}"
        else:
            full_name = decl.name

        if decl.is_external:
            return

        # Find all calls in this procedure's body
        calls: set[str] = set()
        self._find_calls_in_stmts(decl.stmts, full_name, calls)
        self.call_graph[full_name] = calls

        # Collect storage requirements (params + locals)
        storage: list[tuple[str, int, DataType]] = []

        # Parameters
        for param in decl.params:
            param_type = DataType.ADDRESS
            for d in decl.decls:
                if isinstance(d, VarDecl) and d.name == param:
                    param_type = d.data_type or DataType.ADDRESS
                    break
            size = 1 if param_type == DataType.BYTE else 2
            storage.append((param, size, param_type))

        # Local variables (non-parameter VarDecls)
        for d in decl.decls:
            if isinstance(d, VarDecl) and d.name not in decl.params:
                var_type = d.data_type or DataType.ADDRESS
                if d.dimension:
                    elem_size = 1 if var_type == DataType.BYTE else 2
                    size = d.dimension * elem_size
                else:
                    size = 1 if var_type == DataType.BYTE else 2
                storage.append((d.name, size, var_type))

        # Also check inline declarations in statements
        for stmt in decl.stmts:
            if isinstance(stmt, DeclareStmt):
                for inner in stmt.declarations:
                    if isinstance(inner, VarDecl) and inner.name not in decl.params:
                        var_type = inner.data_type or DataType.ADDRESS
                        if inner.dimension:
                            elem_size = 1 if var_type == DataType.BYTE else 2
                            size = inner.dimension * elem_size
                        else:
                            size = 1 if var_type == DataType.BYTE else 2
                        storage.append((inner.name, size, var_type))

        self.proc_storage[full_name] = storage

        # Recurse into nested procedures
        for d in decl.decls:
            if isinstance(d, ProcDecl):
                self._analyze_proc_calls(d, full_name)
        for stmt in decl.stmts:
            if isinstance(stmt, DeclareStmt):
                for inner in stmt.declarations:
                    if isinstance(inner, ProcDecl):
                        self._analyze_proc_calls(inner, full_name)

    def _find_calls_in_stmts(self, stmts: list[Stmt], current_proc: str, calls: set[str]) -> None:
        """Find all procedure calls in a list of statements."""
        for stmt in stmts:
            self._find_calls_in_stmt(stmt, current_proc, calls)

    def _find_calls_in_stmt(self, stmt: Stmt, current_proc: str, calls: set[str]) -> None:
        """Find procedure calls in a statement."""
        if isinstance(stmt, CallStmt):
            if isinstance(stmt.callee, Identifier):
                callee = self._resolve_proc_name(stmt.callee.name, current_proc)
                if callee:
                    calls.add(callee)
            for arg in stmt.args:
                self._find_calls_in_expr(arg, current_proc, calls)
        elif isinstance(stmt, AssignStmt):
            for target in stmt.targets:
                self._find_calls_in_expr(target, current_proc, calls)
            self._find_calls_in_expr(stmt.value, current_proc, calls)
        elif isinstance(stmt, ReturnStmt):
            if stmt.value:
                self._find_calls_in_expr(stmt.value, current_proc, calls)
        elif isinstance(stmt, IfStmt):
            self._find_calls_in_expr(stmt.condition, current_proc, calls)
            self._find_calls_in_stmt(stmt.then_stmt, current_proc, calls)
            if stmt.else_stmt:
                self._find_calls_in_stmt(stmt.else_stmt, current_proc, calls)
        elif isinstance(stmt, DoBlock):
            self._find_calls_in_stmts(stmt.stmts, current_proc, calls)
        elif isinstance(stmt, DoWhileBlock):
            self._find_calls_in_expr(stmt.condition, current_proc, calls)
            self._find_calls_in_stmts(stmt.stmts, current_proc, calls)
        elif isinstance(stmt, DoIterBlock):
            self._find_calls_in_expr(stmt.start, current_proc, calls)
            self._find_calls_in_expr(stmt.bound, current_proc, calls)
            if stmt.step:
                self._find_calls_in_expr(stmt.step, current_proc, calls)
            self._find_calls_in_stmts(stmt.stmts, current_proc, calls)
        elif isinstance(stmt, DoCaseBlock):
            self._find_calls_in_expr(stmt.selector, current_proc, calls)
            for case_stmts in stmt.cases:
                self._find_calls_in_stmts(case_stmts, current_proc, calls)
        elif isinstance(stmt, LabeledStmt):
            self._find_calls_in_stmt(stmt.stmt, current_proc, calls)

    def _find_calls_in_expr(self, expr: Expr, current_proc: str, calls: set[str]) -> None:
        """Find procedure calls in an expression."""
        if isinstance(expr, CallExpr):
            if isinstance(expr.callee, Identifier):
                callee = self._resolve_proc_name(expr.callee.name, current_proc)
                if callee:
                    calls.add(callee)
            for arg in expr.args:
                self._find_calls_in_expr(arg, current_proc, calls)
        elif isinstance(expr, Identifier):
            # In PL/M-80, a bare identifier that refers to a typed procedure
            # is an implicit call (e.g., RESULT = MYFUNC; calls MYFUNC)
            callee = self._resolve_proc_name(expr.name, current_proc)
            if callee:
                calls.add(callee)
        elif isinstance(expr, BinaryExpr):
            self._find_calls_in_expr(expr.left, current_proc, calls)
            self._find_calls_in_expr(expr.right, current_proc, calls)
        elif isinstance(expr, UnaryExpr):
            self._find_calls_in_expr(expr.operand, current_proc, calls)
        elif isinstance(expr, SubscriptExpr):
            self._find_calls_in_expr(expr.array, current_proc, calls)
            self._find_calls_in_expr(expr.index, current_proc, calls)
        elif isinstance(expr, MemberExpr):
            self._find_calls_in_expr(expr.base, current_proc, calls)
        elif isinstance(expr, LocationExpr):
            self._find_calls_in_expr(expr.operand, current_proc, calls)
        elif isinstance(expr, EmbeddedAssignExpr):
            self._find_calls_in_expr(expr.target, current_proc, calls)
            self._find_calls_in_expr(expr.value, current_proc, calls)

    def _resolve_proc_name(self, name: str, current_proc: str) -> str | None:
        """Resolve a procedure name to its full scoped name."""
        # Try scoped names from innermost to outermost
        if current_proc:
            parts = current_proc.split('$')
            for i in range(len(parts), 0, -1):
                scoped = '$'.join(parts[:i]) + '$' + name
                if scoped in self.call_graph:
                    return scoped
        # Try unscoped
        if name in self.call_graph:
            return name
        return None

    def _compute_active_together(self) -> None:
        """Compute which procedures can be active (on stack) at the same time.

        Two procedures can be active together if:
        1. One calls the other (directly or transitively), OR
        2. Both can be called from a common ancestor

        We compute the transitive closure of the call relation.
        """
        self.can_be_active_together = {proc: {proc} for proc in self.call_graph}

        # For each procedure, find all procedures it can reach (callees, transitively)
        reachable: dict[str, set[str]] = {}
        for proc in self.call_graph:
            reachable[proc] = self._get_reachable(proc, set())

        # Two procs can be active together if one is reachable from the other
        # or if they share a common caller (both reachable from same proc)
        for proc in self.call_graph:
            # Add all procs reachable from this one
            self.can_be_active_together[proc].update(reachable[proc])
            # Add this proc to all procs it can reach
            for callee in reachable[proc]:
                self.can_be_active_together[callee].add(proc)

        # Now handle the "common ancestor" case - if A calls B and A calls C,
        # then B and C can be active together (B returns, then A calls C)
        # Actually no - that's NOT "active together" - only one is on stack at a time
        # The key insight: procs are active together only on a single call chain

        # So the current computation is correct: procs on any call path from root to leaf

    def _get_reachable(self, proc: str, visited: set[str]) -> set[str]:
        """Get all procedures reachable from proc via calls."""
        if proc in visited:
            return set()
        visited.add(proc)
        result = set(self.call_graph.get(proc, set()))
        for callee in list(result):
            result.update(self._get_reachable(callee, visited))
        return result

    def _allocate_shared_storage(self) -> None:
        """Allocate shared storage for procedure locals using graph coloring.

        Procedures that cannot be active together can share the same memory.
        We use a simple greedy algorithm: process procedures by total storage size
        (largest first), assign each to the lowest offset that doesn't conflict.
        """
        self.storage_offsets: dict[str, int] = {}  # proc -> base offset
        self.storage_labels: dict[str, dict[str, str]] = {}  # proc -> {var_name -> label}

        # Sort procedures by total storage size (descending) for better packing
        procs_by_size = sorted(
            [(proc, sum(size for _, size, _ in storage))
             for proc, storage in self.proc_storage.items()],
            key=lambda x: -x[1]
        )

        # Track allocated intervals: list of (start, end, proc)
        allocated: list[tuple[int, int, str]] = []

        for proc, total_size in procs_by_size:
            if total_size == 0:
                self.storage_offsets[proc] = 0
                self.storage_labels[proc] = {}
                continue

            # Find lowest offset where this proc doesn't conflict with any
            # proc that can be active together with it
            offset = 0
            while True:
                conflict = False
                for start, end, other_proc in allocated:
                    if other_proc in self.can_be_active_together.get(proc, set()):
                        # Check for overlap
                        if not (offset + total_size <= start or offset >= end):
                            conflict = True
                            # Move past this allocation
                            offset = max(offset, end)
                            break
                if not conflict:
                    break

            self.storage_offsets[proc] = offset
            allocated.append((offset, offset + total_size, proc))

            # Assign labels to each variable
            var_offset = offset
            self.storage_labels[proc] = {}
            for var_name, size, _ in self.proc_storage.get(proc, []):
                self.storage_labels[proc][var_name] = f"??AUTO+{var_offset}"
                var_offset += size

        # Calculate total automatic storage needed
        self.total_auto_storage = max((end for _, end, _ in allocated), default=0)

    def _emit(
        self,
        opcode: str = "",
        operands: str = "",
        label: str = "",
        comment: str = "",
    ) -> None:
        """Emit an assembly line."""
        self.output.append(AsmLine(label, opcode, operands, comment))

        # Track register operations for debugging
        if self.reg_debug:
            self._track_emit(opcode, operands)

    def _track_emit(self, opcode: str, operands: str) -> None:
        """Track register state changes from emitted instructions (debug mode)."""
        op = opcode.lower()
        ops = operands.lower()

        # Track push/pop for manual spill detection
        if op == "push":
            reg = ops.replace("af", "a")  # Normalize af->a
            if reg in ('a', 'hl', 'de', 'bc', 'ix'):
                self.regs.stats['manual_push'] = self.regs.stats.get('manual_push', 0) + 1

        elif op == "pop":
            reg = ops.replace("af", "a")
            if reg in ('a', 'hl', 'de', 'bc', 'ix'):
                self.regs.stats['manual_pop'] = self.regs.stats.get('manual_pop', 0) + 1

        # Track loads that set result registers
        elif op == "ld":
            if ops.startswith("hl,") or ops.startswith("a,"):
                pass  # Result register being set
            elif ops.startswith("de,") or ops.startswith("bc,"):
                pass  # Secondary register being set

        # Track exchange
        elif op == "ex" and ops == "de,hl":
            self.regs.stats['ex_de_hl'] = self.regs.stats.get('ex_de_hl', 0) + 1

    def _check_regs_free(self, context: str) -> None:
        """Assert that all registers are free (debug mode only).

        Called at statement boundaries to detect register leaks.
        """
        if not self.reg_debug:
            return

        # Check if any registers are still marked busy
        busy_regs = []
        for reg in ['a', 'hl', 'de', 'bc']:  # Don't check IX - used for frame
            desc = self.regs.get_reg(reg)
            if desc.state != RegState.FREE:
                busy_regs.append(f"{reg.upper()}({desc.owner})")

        if busy_regs:
            # Log warning but don't fail - existing code doesn't use allocator yet
            import sys
            print(f"[REG DEBUG] {context}: busy registers: {', '.join(busy_regs)}",
                  file=sys.stderr)

    def _reg_debug_log(self, msg: str) -> None:
        """Log a register debug message."""
        if self.reg_debug:
            import sys
            print(f"[REG DEBUG] {msg}", file=sys.stderr)

    def _emit_label(self, label: str) -> None:
        """Emit a label."""
        self.output.append(AsmLine(label=label))

    def _emit_sub16(self) -> None:
        """Emit 16-bit subtract: HL = HL - DE.

        Uses CALL ??SUBDE runtime routine to save code space.
        """
        self.needs_runtime.add("subde")
        self._emit("call", "??subde")

    def _emit_add_hl_const(self, n: int) -> None:
        """Emit HL = HL + constant, optimized for small values.

        For n=1-3, uses repeated INC HL (1 byte, 6 cycles each).
        For larger values, uses LD DE,n; ADD HL,DE (4 bytes, 21 cycles).
        """
        if n == 0:
            return  # No operation needed
        elif n <= 3:
            # Use INC HL for small values (saves 3, 2, or 1 bytes)
            for _ in range(n):
                self._emit("inc", "hl")
        else:
            self._emit("ld", f"de,{self._format_number(n)}")
            self._emit("add", "hl,de")

    def _new_label(self, prefix: str = "L") -> str:
        """Generate a new unique label."""
        self.label_counter += 1
        return f"??{prefix}{self.label_counter:04d}"

    def _new_string_label(self) -> str:
        """Generate a new string literal label."""
        self.string_counter += 1
        return f"??S{self.string_counter:04d}"

    def _format_number(self, n: int) -> str:
        """Format a number for assembly output."""
        if n < 0:
            n = n & 0xFFFF
        if n > 9:
            # Hex numbers must start with a digit for assemblers
            hex_str = f"{n:04X}" if n > 255 else f"{n:02X}"
            if hex_str[0].isalpha():
                hex_str = "0" + hex_str
            return hex_str + "H"
        return str(n)

    # ========================================================================
    # Pass 1: Collect Procedure Declarations
    # ========================================================================

    def _collect_procedures(self, decls: list, parent_proc: str | None, stmts: list | None = None) -> None:
        """
        First pass: collect all procedure declarations into the symbol table.
        This enables forward references - procedures can call each other
        regardless of declaration order.
        """
        for decl in decls:
            if isinstance(decl, ProcDecl):
                self._register_procedure(decl, parent_proc)

        # Also check statements for DeclareStmt containing procedures
        if stmts:
            for stmt in stmts:
                if isinstance(stmt, DeclareStmt):
                    for inner_decl in stmt.declarations:
                        if isinstance(inner_decl, ProcDecl):
                            self._register_procedure(inner_decl, parent_proc)

    def _register_procedure(self, decl: ProcDecl, parent_proc: str | None) -> None:
        """Register a single procedure in the symbol table at module level."""
        # Compute the asm_name for this procedure
        if parent_proc and not decl.is_public and not decl.is_external:
            # Nested procedure - use scoped name
            proc_asm_name = f"@{parent_proc}${decl.name}"
            full_proc_name = f"{parent_proc}${decl.name}"
        else:
            proc_asm_name = decl.name
            full_proc_name = decl.name

        # Extract parameter types from decl.decls
        param_types = []
        for param in decl.params:
            param_type = DataType.ADDRESS  # Default
            for d in decl.decls:
                if isinstance(d, VarDecl) and d.name == param:
                    param_type = d.data_type or DataType.ADDRESS
                    break
            param_types.append(param_type)

        # For non-reentrant procedures with params, pass the LAST param in register
        # Byte params in A, ADDRESS params in HL - saves a store/load pair
        uses_reg_param = (len(decl.params) >= 1 and
                         not decl.is_reentrant and
                         not decl.is_external)

        # Register in symbol table at the GLOBAL level so it's always accessible
        # This allows forward references from anywhere in the module
        # Use full_proc_name as the symbol name to avoid collisions between
        # nested procedures with the same local name (e.g., multiple ZN procs)
        sym = Symbol(
            name=full_proc_name,
            kind=SymbolKind.PROCEDURE,
            return_type=decl.return_type,
            params=decl.params,
            param_types=param_types,
            is_public=decl.is_public,
            is_external=decl.is_external,
            is_reentrant=decl.is_reentrant,
            uses_reg_param=uses_reg_param,
            interrupt_num=decl.interrupt_num,
            asm_name=proc_asm_name,
        )
        # Define at module (root) level - walk up to root scope
        root_scope = self.symbols.current_scope
        while root_scope.parent is not None:
            root_scope = root_scope.parent
        root_scope.define(sym)

        # Recursively collect nested procedures from decls and stmts
        if decl.decls or decl.stmts:
            self._collect_procedures(decl.decls, full_proc_name, decl.stmts)

    # ========================================================================
    # Main Entry Point
    # ========================================================================

    def generate(self, module: Module) -> str:
        """Generate assembly code for a module."""
        self.output = []
        self.data_segment = []
        self.code_data_segment = []
        self.string_literals = []
        self.needs_runtime = set()
        self.needs_end_symbol = False
        self.literal_macros = {}

        # Header
        self._emit(comment=f"PL/M-80 Compiler Output - {module.name}")
        self._emit(comment="Target: Z80")
        self._emit(comment="Generated by uplm80")
        self._emit()

        # Emit .z80 directive for assembler
        self._emit(".z80")
        self._emit()

        # Origin if specified
        if module.origin is not None:
            self._emit("org", self._format_number(module.origin))
            self._emit()

        # First pass: collect LITERALLY macros
        for decl in module.decls:
            if isinstance(decl, LiterallyDecl):
                self.literal_macros[decl.name] = decl.value

        # Separate procedures from other declarations
        procedures: list[ProcDecl] = []
        data_decls: list[VarDecl] = []  # Module-level DATA declarations
        other_decls: list[Declaration] = []
        entry_proc: ProcDecl | None = None

        for decl in module.decls:
            if isinstance(decl, ProcDecl):
                procedures.append(decl)
                # First non-external procedure with same name as module, or first procedure
                if not decl.is_external and entry_proc is None:
                    if decl.name == module.name or len(procedures) == 1:
                        entry_proc = decl
            elif isinstance(decl, VarDecl) and decl.data_values:
                # Module-level DATA declaration - goes at start of code
                data_decls.append(decl)
            else:
                other_decls.append(decl)

        # Pass 1: Pre-register all procedures in symbol table for forward references
        # This allows procedures to call each other regardless of declaration order
        self._collect_procedures(module.decls, parent_proc=None)

        # Pass 2: Build call graph and allocate shared storage for procedure locals
        self._build_call_graph(module)
        self._compute_active_together()
        self._allocate_shared_storage()

        # Emit module-level DATA declarations first (before entry point)
        # This is how PL/M-80 handles the startup jump bootstrap
        self.emit_data_inline = True
        for decl in data_decls:
            self._gen_var_decl(decl)
        # Emit any inline data that was collected
        if self.code_data_segment:
            self.output.extend(self.code_data_segment)
            self.code_data_segment = []
        self.emit_data_inline = False

        # Process non-DATA declarations (allocate storage in data segment)
        for decl in other_decls:
            self._gen_declaration(decl)

        # If there's an entry procedure, jump to it first
        if entry_proc and not module.stmts:
            self._emit()
            self._emit(comment="Entry point")
            if self.mode == Mode.CPM:
                # CP/M: Set stack from BDOS, call main, return to OS
                self._emit("ld", "hl,(6)")
                self._emit("ld", "sp,hl")
                self._emit("call", entry_proc.name)
                self._emit("jp", "0")  # Warm boot to return to CP/M
            else:
                # BARE: Use locally-defined stack, jump to entry
                self._emit("ld", "sp,??STACK")
                self._emit("jp", entry_proc.name)

        # Generate code for module-level statements
        if module.stmts:
            self._emit()
            self._emit(comment="Module initialization code")
            if self.mode == Mode.CPM:
                # CP/M: Set stack from BDOS address at 0006H
                self._emit("ld", "hl,(6)")
                self._emit("ld", "sp,hl")
            else:
                # BARE: Use locally-defined stack
                self._emit("ld", "sp,??STACK")
            for stmt in module.stmts:
                self._gen_stmt(stmt)
            # For CPM mode, add warm boot after module statements
            if self.mode == Mode.CPM:
                self._emit("jp", "0")  # Warm boot to return to CP/M

        # Generate procedures
        for proc in procedures:
            self._gen_declaration(proc)

        # Emit runtime library if needed
        if self.needs_runtime:
            self._emit()
            # Guard against fallthrough from peephole optimization.
            # The optimizer may convert 'call ??move; ret' to 'jp ??move'
            # then eliminate the jp since ??move immediately follows.
            # This guard ensures we never fall through into runtime code.
            self._emit("jp", "??RTEND")
            self._emit(comment="Runtime library")
            runtime = get_runtime_library(self.needs_runtime)
            for line in runtime.split("\n"):
                stripped = line.strip()
                if stripped:
                    if stripped.endswith(":"):
                        # It's a label
                        self._emit_label(stripped[:-1])
                    elif stripped.startswith(";"):
                        # It's a comment
                        self._emit(comment=stripped[1:].strip())
                    else:
                        # It's an instruction
                        parts = stripped.split(None, 1)
                        if len(parts) == 2:
                            self._emit(parts[0], parts[1])
                        else:
                            self._emit(parts[0])
            # End of runtime library label
            self._emit_label("??RTEND")

        # Emit string literals
        if self.string_literals:
            self._emit()
            self._emit(comment="String literals")
            for label, value in self.string_literals:
                self._emit_label(label)
                escaped = self._escape_string(value)
                self._emit("db", escaped)

        # Emit data segment
        if self.data_segment:
            self._emit()
            self._emit(comment="Data segment")
            self.output.extend(self.data_segment)

        # Emit shared automatic storage for procedure locals
        if hasattr(self, 'total_auto_storage') and self.total_auto_storage > 0:
            self._emit()
            self._emit(comment=f"Shared automatic storage ({self.total_auto_storage} bytes)")
            self._emit_label("??AUTO")
            self._emit("ds", str(self.total_auto_storage))

        # Emit stack storage for BARE mode
        if self.mode == Mode.BARE:
            self._emit()
            self._emit(comment="Stack storage (64 bytes)")
            self._emit("ds", "64")
            self._emit_label("??STACK")  # Label after buffer (top of stack)

        # Note: For CPM mode, stack is provided by CP/M (set from BDOS address at 0006H).
        # For BARE mode, stack storage (??STACK) is emitted above.

        # Define __END__ label if program uses .MEMORY built-in
        # __END__ marks the first free byte after all code/data
        if self.needs_end_symbol:
            self._emit()
            self._emit_label("__END__")

        # End directive
        self._emit()
        self._emit("end")

        # Print register statistics in debug mode
        if self.reg_debug and self.regs.stats:
            import sys
            print(f"[REG DEBUG] Statistics for {module.name}:", file=sys.stderr)
            for key, val in sorted(self.regs.stats.items()):
                print(f"  {key}: {val}", file=sys.stderr)

        # Convert to string
        return "\n".join(str(line) for line in self.output)

    def generate_multi(self, modules: list[Module]) -> str:
        """Generate assembly code for multiple modules with unified call graph.

        This allows better local variable storage allocation by analyzing
        call relationships across all modules together.
        """
        if len(modules) == 1:
            return self.generate(modules[0])

        self.output = []
        self.data_segment = []
        self.code_data_segment = []
        self.string_literals = []
        self.needs_runtime = set()
        self.needs_end_symbol = False
        self.literal_macros = {}

        # Header
        module_names = ', '.join(m.name for m in modules)
        self._emit(comment=f"PL/M-80 Compiler Output - {module_names}")
        self._emit(comment="Target: Z80")
        self._emit(comment="Generated by uplm80")
        self._emit()

        # Emit .z80 directive for assembler
        self._emit(".z80")
        self._emit()

        # Use origin from first module if specified
        if modules[0].origin is not None:
            self._emit("org", self._format_number(modules[0].origin))
            self._emit()

        # Collect LITERALLY macros from all modules
        for module in modules:
            for decl in module.decls:
                if isinstance(decl, LiterallyDecl):
                    self.literal_macros[decl.name] = decl.value

        # Pre-register all procedures from all modules for forward references
        for module in modules:
            self._collect_procedures(module.decls, parent_proc=None)

        # Build unified call graph across all modules
        self._build_call_graph_multi(modules)
        self._compute_active_together()
        self._allocate_shared_storage()

        # First pass: collect all module info
        all_procedures: list[tuple[Module, ProcDecl]] = []
        all_data_decls: list[tuple[Module, VarDecl]] = []
        all_other_decls: list[tuple[Module, Declaration]] = []
        entry_proc: ProcDecl | None = None
        first_module_with_stmts: Module | None = None

        for module in modules:
            if module.stmts and first_module_with_stmts is None:
                first_module_with_stmts = module

            for decl in module.decls:
                if isinstance(decl, ProcDecl):
                    all_procedures.append((module, decl))
                    if not decl.is_external and entry_proc is None:
                        entry_proc = decl
                elif isinstance(decl, VarDecl) and decl.data_values:
                    all_data_decls.append((module, decl))
                else:
                    all_other_decls.append((module, decl))

        # Emit module-level DATA declarations first (at start of code segment)
        self.emit_data_inline = True
        for module, decl in all_data_decls:
            self._gen_var_decl(decl)
        if self.code_data_segment:
            self.output.extend(self.code_data_segment)
            self.code_data_segment = []
        self.emit_data_inline = False

        # Process non-DATA declarations (allocate storage)
        for module, decl in all_other_decls:
            self._gen_declaration(decl)

        # Emit initialization/entry code
        if first_module_with_stmts:
            # Has module-level statements - emit init + statements
            self._emit()
            self._emit(comment="Module initialization")
            if self.mode == Mode.CPM:
                self._emit("ld", "hl,(6)")
                self._emit("ld", "sp,hl")
            else:
                self._emit("ld", "sp,??STACK")
            for stmt in first_module_with_stmts.stmts:
                self._gen_stmt(stmt)
            if self.mode == Mode.CPM:
                self._emit("jp", "0")
        elif entry_proc:
            # No statements - call entry procedure
            self._emit()
            self._emit(comment="Entry point")
            if self.mode == Mode.CPM:
                self._emit("ld", "hl,(6)")
                self._emit("ld", "sp,hl")
                self._emit("call", entry_proc.name)
                self._emit("jp", "0")
            else:
                self._emit("ld", "sp,??STACK")
                self._emit("call", entry_proc.name)

        # Generate code for all procedures
        for module, proc in all_procedures:
            if not proc.is_external:
                self._emit()
                self._emit(comment=f"Module: {module.name}")
                self._gen_proc_decl(proc)

        # Emit runtime library if needed
        if self.needs_runtime:
            self._emit()
            # Guard against fallthrough from peephole optimization
            self._emit("jp", "??RTEND")
            self._emit(comment="Runtime library")
            runtime = get_runtime_library(self.needs_runtime)
            for line in runtime.split("\n"):
                stripped = line.strip()
                if stripped:
                    if stripped.endswith(":"):
                        self._emit_label(stripped[:-1])
                    elif stripped.startswith(";"):
                        self._emit(comment=stripped[1:].strip())
                    else:
                        parts = stripped.split(None, 1)
                        if len(parts) == 2:
                            self._emit(parts[0], parts[1])
                        else:
                            self._emit(parts[0])
            # End of runtime library label
            self._emit_label("??RTEND")

        # Emit string literals
        if self.string_literals:
            self._emit()
            self._emit(comment="String literals")
            for label, value in self.string_literals:
                self._emit_label(label)
                escaped = self._escape_string(value)
                self._emit("db", escaped)

        # Emit data segment
        if self.data_segment:
            self._emit()
            self._emit(comment="Data segment")
            self.output.extend(self.data_segment)

        # Emit shared automatic storage
        if hasattr(self, 'total_auto_storage') and self.total_auto_storage > 0:
            self._emit()
            self._emit(comment=f"Shared automatic storage ({self.total_auto_storage} bytes)")
            self._emit_label("??AUTO")
            self._emit("ds", str(self.total_auto_storage))

        # Emit stack storage for BARE mode
        if self.mode == Mode.BARE:
            self._emit()
            self._emit(comment="Stack storage (64 bytes)")
            self._emit("ds", "64")
            self._emit_label("??STACK")

        # Define __END__ label if program uses .MEMORY built-in
        # __END__ marks the first free byte after all code/data
        if self.needs_end_symbol:
            self._emit()
            self._emit_label("__END__")

        # End directive
        self._emit()
        self._emit("end")

        return "\n".join(str(line) for line in self.output)

    def _build_call_graph_multi(self, modules: list[Module]) -> None:
        """Build call graph by analyzing all procedures across multiple modules."""
        self.call_graph = {}
        self.proc_storage: dict[str, list[tuple[str, int, DataType]]] = {}

        # First pass: collect all procedure names from all modules
        all_procs: set[str] = set()
        for module in modules:
            self._collect_proc_names(module.decls, None, all_procs)

        # Initialize call graph
        for proc in all_procs:
            self.call_graph[proc] = set()

        # Second pass: analyze calls in each procedure across all modules
        for module in modules:
            for decl in module.decls:
                if isinstance(decl, ProcDecl) and not decl.is_external:
                    self._analyze_proc_calls(decl, None)

    def _escape_string(self, s: str) -> str:
        """Escape a string for assembly output."""
        parts: list[str] = []
        in_string = False
        for ch in s:
            if 32 <= ord(ch) < 127 and ch != "'":
                if not in_string:
                    if parts:
                        parts.append(",")
                    parts.append("'")
                    in_string = True
                parts.append(ch)
            else:
                if in_string:
                    parts.append("'")
                    in_string = False
                if parts:
                    parts.append(",")
                parts.append(f"{ord(ch):02X}H")
        if in_string:
            parts.append("'")
        return "".join(parts) if parts else "''"

    # ========================================================================
    # Declaration Code Generation
    # ========================================================================

    def _gen_declaration(self, decl: Declaration) -> None:
        """Generate code/storage for a declaration."""
        if isinstance(decl, VarDecl):
            self._gen_var_decl(decl)
        elif isinstance(decl, ProcDecl):
            self._gen_proc_decl(decl)
        elif isinstance(decl, LiterallyDecl):
            # Record in symbol table and literal_macros
            self.symbols.define(
                Symbol(
                    name=decl.name,
                    kind=SymbolKind.LITERAL,
                    literal_value=decl.value,
                )
            )
            self.literal_macros[decl.name] = decl.value
            # Emit EQU for numeric literals (not for built-in names or text macros)
            try:
                val = self._parse_plm_number(decl.value)
                # Generate EQU in data segment
                asm_name = self._mangle_name(decl.name)
                self.data_segment.append(
                    AsmLine(label=asm_name, opcode="EQU", operands=self._format_number(val))
                )
            except ValueError:
                pass  # Non-numeric literal, no EQU needed
        elif isinstance(decl, LabelDecl):
            self.symbols.define(
                Symbol(
                    name=decl.name,
                    kind=SymbolKind.LABEL,
                    is_public=decl.is_public,
                    is_external=decl.is_external,
                )
            )
            if decl.is_external:
                self._emit("extrn", decl.name)

    def _gen_var_decl(self, decl: VarDecl) -> None:
        """Generate storage for a variable declaration."""
        # Mangle name if it conflicts with register names
        base_name = self._mangle_name(decl.name)
        asm_name: str | None = base_name  # Default, may be overridden below

        # Check if we're in a reentrant procedure - locals go on stack
        in_reentrant = (self.current_proc_decl is not None and
                        self.current_proc_decl.is_reentrant and
                        not decl.is_public and not decl.is_external and
                        not decl.based_on and not decl.at_location and
                        not decl.data_values and not decl.initial_values)

        # Check if this is a procedure local that can use shared storage
        use_shared = False
        if (not in_reentrant and self.current_proc and not decl.is_public and not decl.is_external
            and not decl.based_on and not decl.at_location and not decl.data_values
            and not decl.initial_values):
            # Check if we have shared storage for this proc and var
            if (hasattr(self, 'storage_labels')
                and self.current_proc in self.storage_labels
                and decl.name in self.storage_labels[self.current_proc]):
                asm_name = self.storage_labels[self.current_proc][decl.name]
                use_shared = True

        if not use_shared and not in_reentrant:
            # For non-public local variables in procedures, prefix with scope name to avoid conflicts
            if self.current_proc and not decl.is_public and not decl.is_external:
                asm_name = f"@{self.current_proc}${base_name}"
            else:
                asm_name = base_name
        elif in_reentrant:
            asm_name = None  # Will use stack_offset instead

        # Calculate size
        if decl.struct_members:
            # Size of one structure element
            struct_size = sum(
                (m.dimension or 1) * (1 if m.data_type == DataType.BYTE else 2)
                for m in decl.struct_members
            )
            # Multiply by array dimension if this is an array of structures
            size = struct_size * (decl.dimension or 1)
            elem_size = 2  # Structures are ADDRESS-sized elements
        else:
            elem_size = 1 if decl.data_type == DataType.BYTE else 2
            count = decl.dimension or 1
            size = elem_size * count

        # For reentrant procedures, allocate stack space for locals
        stack_offset = None
        if in_reentrant:
            # Locals are at negative offsets from IX
            # Decrement offset first, then use it (so first local is at IX-size)
            self._reentrant_local_offset -= size
            stack_offset = self._reentrant_local_offset

        # Record in symbol table (with mangled name for asm output)
        sym = Symbol(
            name=decl.name,
            kind=SymbolKind.VARIABLE,
            data_type=decl.data_type,
            dimension=decl.dimension,
            struct_members=decl.struct_members,
            based_on=decl.based_on,  # Keep original name for symbol lookup
            is_public=decl.is_public,
            is_external=decl.is_external,
            size=size,
            asm_name=asm_name,  # Store mangled name (None for reentrant locals)
            stack_offset=stack_offset,  # Stack offset for reentrant locals
        )
        self.symbols.define(sym)

        # External variables don't get storage here
        if decl.is_external:
            self._emit("extrn", asm_name)
            return

        # Public declaration
        if decl.is_public:
            self._emit("public", asm_name)

        # Based variables don't allocate storage - they're pointers to other storage
        if decl.based_on:
            return

        # AT variables use specified address
        if decl.at_location:
            if isinstance(decl.at_location, NumberLiteral):
                addr = decl.at_location.value
                self.data_segment.append(
                    AsmLine(label=asm_name, opcode="EQU", operands=self._format_number(addr))
                )
            elif isinstance(decl.at_location, LocationExpr):
                # AT location is an address expression
                loc_operand = decl.at_location.operand
                if isinstance(loc_operand, Identifier):
                    # Check for built-in MEMORY - address is __END__ (end of program)
                    if loc_operand.name.upper() == "MEMORY":
                        self.needs_end_symbol = True
                        # Use SET instead of EQU to allow forward reference to __END__
                        # __END__ is defined at the end of the file, so this is a forward ref
                        self.data_segment.append(
                            AsmLine(label=asm_name, opcode="SET", operands="__END__")
                        )
                    else:
                        # Reference to another variable - check if external
                        ref_sym = self.symbols.lookup(loc_operand.name)
                        if ref_sym and ref_sym.is_external:
                            # For AT pointing to external, just use external name as alias
                            # Store asm_name so lookups use the external's address
                            sym.asm_name = ref_sym.asm_name if ref_sym.asm_name else self._mangle_name(loc_operand.name)
                            # No EQU needed - we'll reference the external directly
                        else:
                            ref_name = ref_sym.asm_name if ref_sym and ref_sym.asm_name else self._mangle_name(loc_operand.name)
                            # Use SET instead of EQU to allow forward references
                            # The referenced symbol may be declared later in the file
                            self.data_segment.append(
                                AsmLine(label=asm_name, opcode="SET", operands=ref_name)
                            )
                elif isinstance(loc_operand, (SubscriptExpr, CallExpr)):
                    # AT (.array(index)) - generate EQU to array element address
                    # Note: PL/M uses arr(i) syntax which parses as CallExpr
                    if isinstance(loc_operand, SubscriptExpr):
                        base_expr = loc_operand.base
                        index_expr = loc_operand.index
                    else:  # CallExpr
                        base_expr = loc_operand.callee
                        index_expr = loc_operand.args[0] if loc_operand.args else NumberLiteral(0)

                    if isinstance(base_expr, Identifier):
                        base_sym = self.symbols.lookup(base_expr.name)
                        base_name = base_sym.asm_name if base_sym and base_sym.asm_name else self._mangle_name(base_expr.name)
                        # Calculate element size (1 for BYTE, 2 for ADDRESS)
                        elem_size = 1
                        if base_sym and base_sym.data_type == DataType.ADDRESS:
                            elem_size = 2
                        # Check if the resolved base_name is an external symbol
                        # This handles cases like AT(.system$data(0)) where system$data is AT(.fcb)
                        is_base_external = base_sym and base_sym.is_external
                        if not is_base_external and base_sym and base_sym.asm_name:
                            # Check if asm_name refers to an external (indirect reference)
                            # Extract the base symbol name before any offset (e.g., "FCB+5" -> "FCB")
                            asm_base = base_sym.asm_name.split('+')[0].strip()
                            ref_sym = self.symbols.lookup(asm_base)
                            if ref_sym and ref_sym.is_external:
                                is_base_external = True
                        # Get the index - must be a constant for AT declarations
                        if isinstance(index_expr, NumberLiteral):
                            offset = index_expr.value * elem_size
                            if is_base_external:
                                # External base - can't use SET/EQU with external symbols
                                # Store expression as asm_name for direct use
                                if offset == 0:
                                    sym.asm_name = base_name
                                else:
                                    sym.asm_name = f"{base_name}+{offset}"
                                # No SET/EQU directive needed
                            elif offset == 0:
                                self.data_segment.append(
                                    AsmLine(label=asm_name, opcode="SET", operands=base_name)
                                )
                            else:
                                self.data_segment.append(
                                    AsmLine(label=asm_name, opcode="SET", operands=f"{base_name}+{offset}")
                                )
                        else:
                            # Non-constant index - can't handle at compile time
                            self.data_segment.append(
                                AsmLine(label=asm_name, opcode="EQU", operands="$")
                            )
                    else:
                        # Complex base expression
                        self.data_segment.append(
                            AsmLine(label=asm_name, opcode="EQU", operands="$")
                        )
                else:
                    # Complex AT expression - evaluate at assembly time (fallback)
                    self.data_segment.append(
                        AsmLine(label=asm_name, opcode="EQU", operands="$")
                    )
            else:
                # Other AT expression - evaluate at assembly time
                self.data_segment.append(
                    AsmLine(label=asm_name, opcode="EQU", operands="$")
                )
            return

        # Generate storage
        # DATA values can go inline in code (for module-level bootstrap) or data segment
        target_segment = self.code_data_segment if self.emit_data_inline else self.data_segment

        if decl.data_values:
            # DATA initialization
            target_segment.append(AsmLine(label=asm_name))
            self._emit_data_values(decl.data_values, decl.data_type or DataType.BYTE, inline=self.emit_data_inline)
        elif decl.initial_values:
            # INITIAL values
            self.data_segment.append(AsmLine(label=asm_name))
            self._emit_initial_values(decl.initial_values, decl.data_type or DataType.BYTE)
        elif use_shared:
            # Using shared automatic storage - no individual allocation needed
            pass
        elif in_reentrant:
            # Reentrant locals are on the stack - no static allocation needed
            pass
        else:
            # Uninitialized storage
            self.data_segment.append(
                AsmLine(label=asm_name, opcode="ds", operands=str(size))
            )

    def _emit_data_values(self, values: list[Expr], dtype: DataType, inline: bool = False) -> None:
        """Emit DATA values to data segment or inline code segment."""
        target = self.code_data_segment if inline else self.data_segment
        for val in values:
            if isinstance(val, NumberLiteral):
                directive = "db" if dtype == DataType.BYTE else "dw"
                target.append(
                    AsmLine(opcode=directive, operands=self._format_number(val.value))
                )
            elif isinstance(val, StringLiteral):
                target.append(
                    AsmLine(opcode="db", operands=self._escape_string(val.value))
                )
            elif isinstance(val, Identifier):
                # Could be a LITERALLY macro - expand it
                name = val.name
                if name in self.literal_macros:
                    # Try to parse the macro value as a number
                    try:
                        num_val = self._parse_plm_number(self.literal_macros[name])
                        directive = "db" if dtype == DataType.BYTE else "dw"
                        target.append(
                            AsmLine(opcode=directive, operands=self._format_number(num_val))
                        )
                    except ValueError:
                        # Not a number, use as-is
                        target.append(
                            AsmLine(opcode="db", operands=self.literal_macros[name])
                        )
                else:
                    # Unknown identifier - use as label reference
                    target.append(
                        AsmLine(opcode="dw", operands=name)
                    )
            elif isinstance(val, LocationExpr):
                # Address-of expression: .variable or .procedure
                operand = val.operand
                if isinstance(operand, Identifier):
                    # .name means address of name
                    # Look up the symbol to get its scoped asm_name
                    name = operand.name
                    sym = None
                    # Search in current scope hierarchy
                    if self.current_proc:
                        parts = self.current_proc.split('$')
                        for i in range(len(parts), 0, -1):
                            scoped_name = '$'.join(parts[:i]) + '$' + name
                            sym = self.symbols.lookup(scoped_name)
                            if sym:
                                break
                    if sym is None:
                        sym = self.symbols.lookup(name)
                    # Use scoped asm_name if available
                    asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(name)
                    target.append(
                        AsmLine(opcode="dw", operands=asm_name)
                    )
                else:
                    raise CodeGenError(f"Unsupported operand in DATA location expression: {operand}")
            elif isinstance(val, BinaryExpr):
                # Binary expression like .name-3 or name+offset
                # Generate assembly expression string
                expr_str = self._data_expr_to_string(val)
                target.append(
                    AsmLine(opcode="dw", operands=expr_str)
                )
            elif isinstance(val, ConstListExpr):
                # Nested constant list
                for v in val.values:
                    self._emit_data_values([v], dtype, inline=inline)

    def _data_expr_to_string(self, expr: Expr) -> str:
        """Convert a DATA expression to assembly string (for DW/DB operands)."""
        if isinstance(expr, NumberLiteral):
            return self._format_number(expr.value)
        elif isinstance(expr, Identifier):
            name = expr.name
            if name in self.literal_macros:
                return self.literal_macros[name]
            # Look up the symbol to get its scoped asm_name
            sym = None
            if self.current_proc:
                parts = self.current_proc.split('$')
                for i in range(len(parts), 0, -1):
                    scoped_name = '$'.join(parts[:i]) + '$' + name
                    sym = self.symbols.lookup(scoped_name)
                    if sym:
                        break
            if sym is None:
                sym = self.symbols.lookup(name)
            return sym.asm_name if sym and sym.asm_name else self._mangle_name(name)
        elif isinstance(expr, LocationExpr):
            return self._data_expr_to_string(expr.operand)
        elif isinstance(expr, BinaryExpr):
            left = self._data_expr_to_string(expr.left)
            right = self._data_expr_to_string(expr.right)
            op_map = {
                BinaryOp.ADD: '+',
                BinaryOp.SUB: '-',
                BinaryOp.MUL: '*',
                BinaryOp.DIV: '/',
                BinaryOp.AND: ' AND ',
                BinaryOp.OR: ' OR ',
                BinaryOp.XOR: ' XOR ',
            }
            op = op_map.get(expr.op, '+')
            return f"({left}{op}{right})"
        else:
            raise CodeGenError(f"Unsupported expression in DATA: {type(expr)}")

    def _emit_initial_values(self, values: list[Expr], dtype: DataType) -> None:
        """Emit INITIAL values to data segment."""
        for val in values:
            if isinstance(val, NumberLiteral):
                directive = "db" if dtype == DataType.BYTE else "dw"
                self.data_segment.append(
                    AsmLine(opcode=directive, operands=self._format_number(val.value))
                )
            elif isinstance(val, StringLiteral):
                self.data_segment.append(
                    AsmLine(opcode="db", operands=self._escape_string(val.value))
                )

    def _gen_proc_decl(self, decl: ProcDecl) -> None:
        """Generate code for a procedure."""
        old_proc = self.current_proc
        old_proc_decl = self.current_proc_decl

        # For nested procedures, create a unique scoped name
        if old_proc and not decl.is_public and not decl.is_external:
            # Nested procedure - use scoped name
            proc_asm_name = f"@{old_proc}${decl.name}"
            full_proc_name = f"{old_proc}${decl.name}"
            self.current_proc = full_proc_name  # Compound name for further nesting
        else:
            proc_asm_name = decl.name
            full_proc_name = decl.name
            self.current_proc = decl.name

        self.current_proc_decl = decl

        # Look up the procedure (already registered in pass 1)
        # Use full_proc_name to find the correct symbol for nested procs
        sym = self.symbols.lookup(full_proc_name)
        if sym is None:
            sym = Symbol(
                name=full_proc_name,
                kind=SymbolKind.PROCEDURE,
                return_type=decl.return_type,
                params=decl.params,
                is_public=decl.is_public,
                is_external=decl.is_external,
                is_reentrant=decl.is_reentrant,
                interrupt_num=decl.interrupt_num,
                asm_name=proc_asm_name,
            )
            self.symbols.define(sym)
        else:
            # Use the asm_name from pass 1
            proc_asm_name = sym.asm_name or proc_asm_name

        if decl.is_external:
            self._emit("extrn", proc_asm_name)
            self.current_proc = old_proc
            self.current_proc_decl = old_proc_decl
            return

        self._emit()
        if decl.is_public:
            self._emit("public", decl.name)

        self._emit(comment=f"Procedure {decl.name}")
        self._emit_label(proc_asm_name)

        # Enter new scope
        self.symbols.enter_scope(decl.name)

        # Procedure prologue
        if decl.interrupt_num is not None:
            # Interrupt handler - save all registers
            self._emit("push", "af")
            self._emit("push", "bc")
            self._emit("push", "de")
            self._emit("push", "hl")

        # Define parameters as local variables
        # For non-reentrant: use shared automatic storage via storage_labels
        # For reentrant: use IX-relative stack frame
        param_infos: list[tuple[str, str, DataType, int]] = []  # (name, asm_name, type, size)
        use_shared_storage = not decl.is_reentrant and full_proc_name in self.storage_labels

        # For reentrant procedures, set up IX frame pointer first
        # Stack at entry: [params...][ret_addr] <- SP
        # After PUSH IX: [params...][ret_addr][saved_IX] <- SP, IX
        if decl.is_reentrant:
            self._emit("push", "ix")
            self._emit("ld", "ix,0")
            self._emit("add", "ix,sp")

        # Calculate parameter offsets for reentrant procedures
        # Stack after PUSH IX: [params...][ret_addr(2)][saved_IX(2)] <- IX
        # First param is at IX+4, subsequent params at higher offsets
        # Parameters are pushed in order: first arg pushed first, ends up deepest
        # So params[0] is at the highest offset, params[-1] is at IX+4
        reentrant_param_offset = 4  # Start after saved IX (2) and ret addr (2)
        if decl.is_reentrant:
            # Calculate total params size to compute offsets
            # Params are pushed first-to-last, so on stack: [param0][param1]...[paramN][ret][IX]
            # paramN is at IX+4, param(N-1) is at IX+4+size(paramN), etc.
            param_sizes = []
            for param in decl.params:
                param_type = DataType.ADDRESS  # Default
                for d in decl.decls:
                    if isinstance(d, VarDecl) and d.name == param:
                        param_type = d.data_type or DataType.ADDRESS
                        break
                param_sizes.append(2)  # All stack slots are 2 bytes (pushed as 16-bit)
            # Compute offset for each param (last param is at IX+4)
            total_params_size = sum(param_sizes)
            reentrant_param_offset = 4 + total_params_size - param_sizes[-1] if param_sizes else 4

        for i, param in enumerate(decl.params):
            # Find parameter declaration in decl.decls
            param_type = DataType.ADDRESS  # Default
            for d in decl.decls:
                if isinstance(d, VarDecl) and d.name == param:
                    param_type = d.data_type or DataType.ADDRESS
                    break

            param_size = 1 if param_type == DataType.BYTE else 2

            if decl.is_reentrant:
                # Use stack frame - params accessed via IX+offset
                # First param (params[0]) is at highest offset
                # Each subsequent param is 2 bytes lower (all pushed as 16-bit)
                stack_offset = reentrant_param_offset
                reentrant_param_offset -= 2  # Move to next param (all slots are 2 bytes)

                self.symbols.define(
                    Symbol(
                        name=param,
                        kind=SymbolKind.PARAMETER,
                        data_type=param_type,
                        size=param_size,
                        stack_offset=stack_offset,
                    )
                )
                param_infos.append((param, None, param_type, param_size))
            else:
                # Get asm_name from shared storage or create individual
                if use_shared_storage and param in self.storage_labels.get(full_proc_name, {}):
                    asm_name = self.storage_labels[full_proc_name][param]
                else:
                    # Fallback: individual storage
                    asm_name = f"@{decl.name}${self._mangle_name(param)}"
                    # Allocate individual storage in data segment
                    self.data_segment.append(
                        AsmLine(label=asm_name, opcode="ds", operands=str(param_size))
                    )

                self.symbols.define(
                    Symbol(
                        name=param,
                        kind=SymbolKind.PARAMETER,
                        data_type=param_type,
                        size=param_size,
                        asm_name=asm_name,
                    )
                )
                param_infos.append((param, asm_name, param_type, param_size))

        # Generate prologue code for register parameter (last param in A or HL)
        # For non-reentrant procedures, the last param is passed in register and needs to be stored
        if param_infos and not decl.is_reentrant:
            _, last_asm_name, last_param_type, _ = param_infos[-1]
            if last_param_type == DataType.BYTE:
                # Last param came in A - store it
                self._emit("ld", f"({last_asm_name}),a")
            else:
                # Last param came in HL - store it
                self._emit("ld", f"({last_asm_name}),hl")

        # Track locals offset for reentrant procedures (negative from IX)
        self._reentrant_local_offset = 0  # Will be decremented as locals are allocated

        # Generate code for local declarations (skip parameters and nested procedures)
        nested_procs: list[ProcDecl] = []
        for local_decl in decl.decls:
            if isinstance(local_decl, ProcDecl):
                # Defer nested procedures
                nested_procs.append(local_decl)
            elif isinstance(local_decl, VarDecl):
                # Skip if it's a parameter (already defined)
                if local_decl.name not in decl.params:
                    self._gen_declaration(local_decl)
            else:
                self._gen_declaration(local_decl)

        # Process statements, extracting nested procedure declarations
        statements_to_gen: list[Stmt] = []
        for stmt in decl.stmts:
            if isinstance(stmt, DeclareStmt):
                for inner_decl in stmt.declarations:
                    if isinstance(inner_decl, ProcDecl):
                        nested_procs.append(inner_decl)
                    elif isinstance(inner_decl, VarDecl):
                        self._gen_declaration(inner_decl)
                    else:
                        self._gen_declaration(inner_decl)
            else:
                statements_to_gen.append(stmt)

        # For reentrant procedures, allocate stack space for locals
        if decl.is_reentrant and self._reentrant_local_offset < 0:
            # Allocate stack space: SP = SP + offset (offset is negative)
            # ld hl,offset; add hl,sp; ld sp,hl
            self._emit("ld", f"hl,{self._reentrant_local_offset}")
            self._emit("add", "hl,sp")
            self._emit("ld", "sp,hl")

        # Generate code for statements with liveness tracking
        ends_with_return = False
        for i, stmt in enumerate(statements_to_gen):
            # Track remaining statements for liveness analysis
            self.pending_stmts = statements_to_gen[i + 1:]
            self._gen_stmt(stmt)
            ends_with_return = isinstance(stmt, ReturnStmt)
        self.pending_stmts = []  # Clear after procedure

        # Procedure epilogue (implicit return if no explicit RETURN at end)
        if not ends_with_return:
            self._gen_proc_epilogue(decl)

        # Now generate nested procedures (after outer procedure)
        for nested_proc in nested_procs:
            self._gen_proc_decl(nested_proc)

        self.symbols.leave_scope()
        self.current_proc = old_proc
        self.current_proc_decl = old_proc_decl

    def _gen_proc_epilogue(self, decl: ProcDecl) -> None:
        """Generate procedure epilogue."""
        if decl.interrupt_num is not None:
            self._emit("pop", "hl")
            self._emit("pop", "de")
            self._emit("pop", "bc")
            self._emit("pop", "af")
            self._emit("ei")
            self._emit("ret")
        elif decl.is_reentrant:
            # Restore stack pointer and frame pointer for reentrant procedures
            # ld sp,ix restores SP to point to saved IX
            # pop IX restores the old frame pointer
            self._emit("ld", "sp,ix")
            self._emit("pop", "ix")
            self._emit("ret")
        else:
            self._emit("ret")

    # ========================================================================
    # Statement Code Generation
    # ========================================================================

    def _gen_stmt(self, stmt: Stmt) -> None:
        """Generate code for a statement."""
        if isinstance(stmt, AssignStmt):
            self._gen_assign(stmt)
        elif isinstance(stmt, CallStmt):
            self._gen_call_stmt(stmt)
        elif isinstance(stmt, ReturnStmt):
            self._gen_return(stmt)
        elif isinstance(stmt, GotoStmt):
            # Check if target is a LITERALLY macro
            target = stmt.target
            if target in self.literal_macros:
                target = self.literal_macros[target]
            # Check if this is a module-level label or procedure-local label
            # Module-level labels are defined without procedure prefix
            module_label = self.symbols.lookup(target)
            if module_label and module_label.kind == SymbolKind.LABEL:
                # Module-level label - use as-is
                pass
            elif self.current_proc:
                # Procedure-local label - prefix with current procedure
                target = f"@{self.current_proc}${target}"
            self._emit("jp", target)
        elif isinstance(stmt, HaltStmt):
            self._emit("halt")
        elif isinstance(stmt, EnableStmt):
            self._emit("ei")
        elif isinstance(stmt, DisableStmt):
            self._emit("di")
        elif isinstance(stmt, NullStmt):
            pass  # No code
        elif isinstance(stmt, LabeledStmt):
            label = stmt.label
            if self.current_proc:
                # Procedure-local label - prefix with current procedure
                label = f"@{self.current_proc}${label}"
            else:
                # Module-level label - register in symbol table for GOTO lookups
                self.symbols.define(
                    Symbol(
                        name=stmt.label,
                        kind=SymbolKind.LABEL,
                    )
                )
            self._emit_label(label)
            self._gen_stmt(stmt.stmt)
        elif isinstance(stmt, IfStmt):
            self._gen_if(stmt)
        elif isinstance(stmt, DoBlock):
            self._gen_do_block(stmt)
        elif isinstance(stmt, DoWhileBlock):
            self._gen_do_while(stmt)
        elif isinstance(stmt, DoIterBlock):
            self._gen_do_iter(stmt)
        elif isinstance(stmt, DoCaseBlock):
            self._gen_do_case(stmt)
        elif isinstance(stmt, DeclareStmt):
            for decl in stmt.declarations:
                self._gen_declaration(decl)

    def _gen_assign(self, stmt: AssignStmt) -> None:
        """Generate code for assignment."""
        # Special case: storing small constant to BYTE variable
        # Use Xor a (for 0) or ld a,n (for other bytes) instead of ld hl,n
        if isinstance(stmt.value, NumberLiteral) and stmt.value.value <= 255:
            # Check if all targets are BYTE variables or BYTE array elements
            all_byte_targets = True
            for target in stmt.targets:
                if isinstance(target, Identifier):
                    sym = self.symbols.lookup(target.name)
                    if not sym or sym.data_type != DataType.BYTE:
                        all_byte_targets = False
                        break
                elif isinstance(target, SubscriptExpr):
                    # Check if array element type is BYTE
                    if isinstance(target.base, Identifier):
                        sym = self.symbols.lookup(target.base.name)
                        if not sym or sym.data_type != DataType.BYTE:
                            all_byte_targets = False
                            break
                    else:
                        all_byte_targets = False
                        break
                elif isinstance(target, CallExpr):
                    # Parser may create CallExpr for array subscript
                    if isinstance(target.callee, Identifier) and len(target.args) == 1:
                        sym = self.symbols.lookup(target.callee.name)
                        if sym and sym.kind != SymbolKind.PROCEDURE and sym.data_type == DataType.BYTE:
                            pass  # It's a BYTE array element
                        else:
                            all_byte_targets = False
                            break
                    else:
                        all_byte_targets = False
                        break
                else:
                    all_byte_targets = False
                    break

            if all_byte_targets:
                # Generate efficient byte constant
                if stmt.value.value == 0:
                    self._emit("xor", "a")
                else:
                    self._emit("ld", f"a,{self._format_number(stmt.value.value)}")

                for i, target in enumerate(stmt.targets):
                    if i < len(stmt.targets) - 1:
                        self._emit("push", "af")
                    self._gen_store(target, DataType.BYTE)
                    if i < len(stmt.targets) - 1:
                        self._emit("pop", "af")
                return

        # Evaluate the value expression (result in A for BYTE, HL for ADDRESS)
        value_type = self._gen_expr(stmt.value)

        # Store to each target (multiple assignment support)
        for i, target in enumerate(stmt.targets):
            if i < len(stmt.targets) - 1:
                # Need to preserve value for next target
                if value_type == DataType.BYTE:
                    self._emit("push", "af")
                else:
                    self._emit("push", "hl")

            self._gen_store(target, value_type)

            if i < len(stmt.targets) - 1:
                if value_type == DataType.BYTE:
                    self._emit("pop", "af")
                else:
                    self._emit("pop", "hl")

    def _gen_call_stmt(self, stmt: CallStmt) -> None:
        """Generate code for a CALL statement."""
        # Look up procedure symbol to check if it's user-defined
        sym = None
        call_name = None
        if isinstance(stmt.callee, Identifier):
            name = stmt.callee.name
            # Check if user defined a procedure with this name
            if self.current_proc:
                parts = self.current_proc.split('$')
                for i in range(len(parts), 0, -1):
                    scoped_name = '$'.join(parts[:i]) + '$' + name
                    sym = self.symbols.lookup(scoped_name)
                    if sym:
                        break
            if sym is None:
                sym = self.symbols.lookup(name)
            # Set call_name early if we found the symbol
            if sym:
                call_name = sym.asm_name if sym.asm_name else name

        # Treat as builtin if it's a BUILTIN symbol (not user-defined)
        # Builtins are registered in symbol table with SymbolKind.BUILTIN
        if isinstance(stmt.callee, Identifier):
            is_builtin = (sym is None or sym.kind == SymbolKind.BUILTIN)
            if is_builtin:
                upper_name = stmt.callee.name.upper()
                # Handle built-in procedures that don't return values
                if upper_name in self.BUILTIN_FUNCS:
                    result = self._gen_builtin(upper_name, stmt.args)
                    if result is not None or upper_name in ('TIME', 'MOVE'):
                        # Built-in was handled
                        return

        # If sym/call_name weren't set yet, look up again (for member access etc.)
        if isinstance(stmt.callee, Identifier) and call_name is None:
            name = stmt.callee.name
            if self.current_proc:
                parts = self.current_proc.split('$')
                for i in range(len(parts), 0, -1):
                    scoped_name = '$'.join(parts[:i]) + '$' + name
                    sym = self.symbols.lookup(scoped_name)
                    if sym:
                        break
            if sym is None:
                sym = self.symbols.lookup(name)
            call_name = sym.asm_name if sym and sym.asm_name else name

        # Optimize CP/M BDOS calls: MON1(func, arg) and MON2(func, arg)
        # This must be checked AFTER symbol resolution but regardless of call_name status
        if isinstance(stmt.callee, Identifier):
            upper_name = stmt.callee.name.upper()
            if upper_name in ('MON1', 'MON2') and len(stmt.args) == 2:
                func_arg, addr_arg = stmt.args
                # Check if function number is a constant
                func_num = self._get_const_byte_value(func_arg)

                if func_num is not None:
                    # Generate direct BDOS call: ld c,func; ld de,addr; CALL 5
                    self._emit("ld", f"c,{self._format_number(func_num)}")
                    addr_type = self._gen_expr(addr_arg)
                    if addr_type == DataType.BYTE:
                        # BYTE arg goes in E; BDOS ignores D for byte-only functions
                        self._emit("ld", "e,a")
                    else:
                        self._emit("ex", "de,hl")  # DE = addr
                    self._emit("call", "5")  # BDOS entry point
                    return  # Done - no stack cleanup needed

        # For non-reentrant LOCAL procedures, store args directly to parameter memory
        # For reentrant procedures, external procedures, or indirect calls, use stack
        use_stack = True
        full_callee_name = None
        if sym and sym.kind == SymbolKind.PROCEDURE and not sym.is_reentrant and not sym.is_external:
            use_stack = False
            # Get the full procedure name (needed for storage_labels lookup)
            full_callee_name = sym.name

        if use_stack:
            # Stack-based parameter passing (reentrant or indirect calls)
            for arg in stmt.args:
                arg_type = self._gen_expr(arg)
                if arg_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("push", "hl")
        else:
            # Direct memory parameter passing (non-reentrant)
            # Last param is passed in register (A for BYTE, HL for ADDRESS)
            # Other params are stored to memory
            last_param_idx = len(stmt.args) - 1
            uses_reg = sym.uses_reg_param and len(stmt.args) > 0

            for i, arg in enumerate(stmt.args):
                if i < len(sym.params):
                    param_name = sym.params[i]
                    param_type = sym.param_types[i] if i < len(sym.param_types) else DataType.ADDRESS
                    is_last = (i == last_param_idx)

                    # Last param passed in register - just evaluate it
                    if is_last and uses_reg:
                        # Optimize constants for BYTE
                        if param_type == DataType.BYTE:
                            if isinstance(arg, NumberLiteral) and arg.value <= 255:
                                self._emit("ld", f"a,{self._format_number(arg.value)}")
                                continue
                            elif isinstance(arg, StringLiteral) and len(arg.value) == 1:
                                self._emit("ld", f"a,{self._format_number(ord(arg.value[0]))}")
                                continue
                            elif isinstance(arg, Identifier) and arg.name in self.literal_macros:
                                try:
                                    val = self._parse_plm_number(self.literal_macros[arg.name])
                                    if val <= 255:
                                        self._emit("ld", f"a,{self._format_number(val)}")
                                        continue
                                except (ValueError, TypeError):
                                    pass
                        # Evaluate arg - result in A (BYTE) or HL (ADDRESS)
                        arg_type = self._gen_expr(arg)
                        if param_type == DataType.BYTE and arg_type == DataType.ADDRESS:
                            self._emit("ld", "a,l")
                        elif param_type == DataType.ADDRESS and arg_type == DataType.BYTE:
                            self._emit("ld", "l,a")
                            self._emit("ld", "h,0")
                        continue

                    # Non-last params: store to memory
                    # Try to get param asm name from shared storage
                    param_asm = None
                    if (hasattr(self, 'storage_labels')
                        and full_callee_name in self.storage_labels
                        and param_name in self.storage_labels[full_callee_name]):
                        param_asm = self.storage_labels[full_callee_name][param_name]
                    else:
                        # Fallback: build param asm name: @procname$param
                        proc_base = sym.asm_name if sym.asm_name else name
                        if proc_base.startswith('@'):
                            proc_base = proc_base[1:]
                        param_asm = f"@{proc_base}${self._mangle_name(param_name)}"

                    # Optimize: for BYTE parameter with constant, use ld a,n directly
                    if param_type == DataType.BYTE:
                        if isinstance(arg, NumberLiteral) and arg.value <= 255:
                            self._emit("ld", f"a,{self._format_number(arg.value)}")
                            self._emit("ld", f"({param_asm}),a")
                            continue
                        elif isinstance(arg, StringLiteral) and len(arg.value) == 1:
                            self._emit("ld", f"a,{self._format_number(ord(arg.value[0]))}")
                            self._emit("ld", f"({param_asm}),a")
                            continue
                        # Check for LITERALLY macro
                        elif isinstance(arg, Identifier) and arg.name in self.literal_macros:
                            try:
                                val = self._parse_plm_number(self.literal_macros[arg.name])
                                if val <= 255:
                                    self._emit("ld", f"a,{self._format_number(val)}")
                                    self._emit("ld", f"({param_asm}),a")
                                    continue
                            except (ValueError, TypeError):
                                pass

                    arg_type = self._gen_expr(arg)
                    if param_type == DataType.BYTE or arg_type == DataType.BYTE:
                        # BYTE param - ensure value is in A, use LD (addr),A
                        if arg_type == DataType.ADDRESS:
                            self._emit("ld", "a,l")
                        self._emit("ld", f"({param_asm}),a")
                    else:
                        # ADDRESS param - use LD (addr),HL
                        self._emit("ld", f"({param_asm}),hl")

        # Call the procedure
        if isinstance(stmt.callee, Identifier):
            self._emit("call", call_name)
        else:
            # Indirect call through address
            self._gen_expr(stmt.callee)
            self._emit("jp", "(hl)")

        # Clean up stack (caller cleanup) - only for stack-based calls
        if use_stack and stmt.args:
            stack_bytes = len(stmt.args) * 2
            if stack_bytes == 2:
                self._emit("pop", "de")  # Dummy pop
            elif stack_bytes == 4:
                self._emit("pop", "de")
                self._emit("pop", "de")
            elif stack_bytes <= 8:
                for _ in range(len(stmt.args)):
                    self._emit("pop", "de")
            else:
                # Adjust stack pointer directly
                self._emit("ld", f"de,{stack_bytes}")
                self._emit("add", "hl,sp")
                self._emit("ld", "sp,hl")

    def _gen_return(self, stmt: ReturnStmt) -> None:
        """Generate code for RETURN statement."""
        if stmt.value:
            # Check if A already has the value from embedded assignment optimization
            skip_load = False
            if (self.embedded_assign_target and
                isinstance(stmt.value, Identifier) and
                stmt.value.name == self.embedded_assign_target):
                # A already has this value - skip the load
                skip_load = True
                self.embedded_assign_target = None  # Clear after use

            if skip_load:
                # A already contains the return value - just return
                pass
            # Optimize: if returning BYTE and value is a small constant, use ld a,n directly
            elif (self.current_proc_decl and
                self.current_proc_decl.return_type == DataType.BYTE and
                isinstance(stmt.value, NumberLiteral) and stmt.value.value <= 255):
                self._emit("ld", f"a,{self._format_number(stmt.value.value)}")
            else:
                result_type = self._gen_expr(stmt.value)
                # Return value is in A (BYTE) or HL (ADDRESS)
                # If procedure returns BYTE but we have ADDRESS, convert
                if (self.current_proc_decl and
                    self.current_proc_decl.return_type == DataType.BYTE and
                    result_type == DataType.ADDRESS):
                    # Convert HL to A: non-zero HL -> 0FFH (TRUE), zero HL -> 0 (FALSE)
                    self._emit("ld", "a,l")
                    self._emit("or", "h")
                    # Now A is non-zero if true, zero if false
                    # For proper PL/M TRUE (0FFH), normalize:
                    end_label = self._new_label("RETE")
                    self._emit("jp", f"z,{end_label}")
                    self._emit("ld", "a,0ffh")
                    self._emit_label(end_label)
                # If procedure returns ADDRESS but we have BYTE, zero-extend A to HL
                elif (self.current_proc_decl and
                      self.current_proc_decl.return_type == DataType.ADDRESS and
                      result_type == DataType.BYTE):
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")

        if self.current_proc_decl and self.current_proc_decl.interrupt_num is not None:
            # Interrupt handler return
            self._emit("pop", "hl")
            self._emit("pop", "de")
            self._emit("pop", "bc")
            self._emit("pop", "af")
            self._emit("ei")
            self._emit("ret")
        elif self.current_proc_decl and self.current_proc_decl.is_reentrant:
            # Reentrant procedure return - restore frame pointer
            self._emit("ld", "sp,ix")
            self._emit("pop", "ix")
            self._emit("ret")
        else:
            self._emit("ret")

    def _gen_if(self, stmt: IfStmt) -> None:
        """Generate code for IF statement."""
        # Warn about trivial constant conditions (IF 0, IF 1)
        self._warn_trivial_if(stmt.condition)

        else_label = self._new_label("ELSE")
        end_label = self._new_label("ENDIF")
        false_target = else_label if stmt.else_stmt else end_label

        # Track current IF statement for embedded assignment optimization
        old_if_stmt = self.current_if_stmt
        self.current_if_stmt = stmt

        # Try to generate optimized conditional jump for comparisons
        if self._gen_condition_jump_false(stmt.condition, false_target):
            # Condition jump was generated directly
            pass
        else:
            # Fallback: evaluate condition and test result
            result_type = self._gen_expr(stmt.condition)
            # Test result - BYTE in A, ADDRESS in HL
            if result_type == DataType.BYTE:
                # Value is in A - just or a to set flags
                self._emit("or", "a")
            else:
                # Value is in HL - test if zero
                self._emit("ld", "a,l")
                self._emit("or", "h")  # A = L | H
            self._emit("jp", f"z,{false_target}")

        self.current_if_stmt = old_if_stmt  # Restore before generating body

        # Then branch
        self._gen_stmt(stmt.then_stmt)

        if stmt.else_stmt:
            self._emit("jp", end_label)
            self._emit_label(else_label)
            self._gen_stmt(stmt.else_stmt)

        self._emit_label(end_label)

    def _gen_condition_jump_false(self, condition: Expr, false_label: str) -> bool:
        """Generate conditional jump to false_label if condition is false.

        Returns True if optimized jump was generated, False if caller should use fallback.
        """
        # Handle constant conditions - no code needed for always-true, unconditional jump for always-false
        if isinstance(condition, NumberLiteral):
            if condition.value == 0:
                # Always false - unconditional jump
                self._emit("jp", false_label)
            # If non-zero (always true), no code needed - just fall through
            return True

        # Handle simple identifier - load and test directly
        if isinstance(condition, Identifier):
            cond_type = self._get_expr_type(condition)
            if cond_type == DataType.BYTE:
                self._gen_expr(condition)  # Loads into A
                self._emit("or", "a")     # Set Z flag
                self._emit("jp", f"z,{false_label}")
                return True
            else:
                self._gen_expr(condition)  # Loads into HL
                self._emit("ld", "a,l")
                self._emit("or", "h")
                self._emit("jp", f"z,{false_label}")
                return True

        # Handle function call - evaluate and test result
        if isinstance(condition, CallExpr):
            cond_type = self._gen_call_expr(condition)
            if cond_type == DataType.BYTE:
                self._emit("or", "a")     # Set Z flag (result in A)
                self._emit("jp", f"z,{false_label}")
            else:
                self._emit("ld", "a,l")
                self._emit("or", "h")
                self._emit("jp", f"z,{false_label}")
            return True

        # Handle NOT - invert the condition
        if isinstance(condition, UnaryExpr) and condition.op == UnaryOp.NOT:
            # NOT x is false when x is true, so jump to false_label when x is true
            return self._gen_condition_jump_true(condition.operand, false_label)

        if not isinstance(condition, BinaryExpr):
            return False

        op = condition.op

        # NOTE: PL/M-80 AND and OR are BITWISE operators, not short-circuit logical operators.
        # IF X AND Y tests if (X bitwise-and Y) is non-zero, NOT if both X and Y are non-zero.
        # So we do NOT handle AND/OR specially here - they fall through to expression evaluation.

        if op not in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.GT, BinaryOp.LE, BinaryOp.GE):
            return False

        # Check for impossible comparisons (e.g., BYTE compared to -1)
        self._check_impossible_comparison(condition.left, condition.right, op)

        # Check if both operands are bytes for optimized comparison
        left_type = self._get_expr_type(condition.left)
        right_type = self._get_expr_type(condition.right)
        both_bytes = (left_type == DataType.BYTE and right_type == DataType.BYTE)

        # Byte comparison with constant - use cp n
        # Handle both regular bytes (0-255) and "negative bytes" (0xFF00-0xFFFF like -1)
        if left_type == DataType.BYTE:
            const_val = None
            if isinstance(condition.right, NumberLiteral):
                val = condition.right.value
                # Allow direct byte values (0-255) or negative byte values (0xFF00-0xFFFF)
                if val <= 255 or (val & 0xFF00) == 0xFF00:
                    const_val = val & 0xFF
            elif isinstance(condition.right, StringLiteral) and len(condition.right.value) == 1:
                const_val = ord(condition.right.value[0])

            if const_val is not None:
                self._gen_expr(condition.left)  # Result in A
                self._emit("cp", self._format_number(const_val))
                self._emit_jump_on_false(op, false_label)
                return True
            elif both_bytes:
                # Byte-to-byte comparison - load right first for efficient SUB
                self._gen_expr(condition.right)  # Result in A
                self._emit("ld", "b,a")  # Save right
                self._gen_expr(condition.left)  # Result in A (left)
                self._emit("sub", "b")    # A = left - right, flags set
                self._emit_jump_on_false(op, false_label)
                return True

        if both_bytes:
            # Both bytes but not constant - already handled above
            pass
        else:
            # Optimize ADDRESS comparison with 0: use ld a,l / or h instead of subtraction
            if op in (BinaryOp.EQ, BinaryOp.NE) and isinstance(condition.right, NumberLiteral) and condition.right.value == 0:
                self._gen_expr(condition.left)  # Result in HL
                if left_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("ld", "a,l")
                self._emit("or", "h")  # Z flag set if HL == 0
                if op == BinaryOp.EQ:
                    self._emit("jp", f"nz,{false_label}")  # If HL != 0, condition is false
                else:  # NE
                    self._emit("jp", f"z,{false_label}")  # If HL == 0, condition is false
                return True

            # 16-bit comparison - optimize evaluation order when possible
            # Only optimize if left is simple AND right is complex
            # (if right is simple, loading it to DE directly is more efficient)
            left_simple = self._expr_preserves_de(condition.left)
            right_simple = self._expr_preserves_de(condition.right)

            if left_simple and not right_simple:
                # Evaluate complex right first, save to DE, then simple left
                self._gen_expr(condition.right)
                if right_type == DataType.BYTE:
                    self._emit("ld", "e,a")
                    self._emit("ld", "d,0")
                else:
                    self._emit("ex", "de,hl")  # DE = right
                # Evaluate left - DE is preserved
                self._gen_expr(condition.left)
                if left_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                # Now: HL = left, DE = right (no PUSH/POP needed!)
            else:
                # Either left is complex, or right is simple - use standard approach
                actual_left_type = self._gen_expr(condition.left)
                if actual_left_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("push", "hl")

                actual_right_type = self._gen_expr(condition.right)
                if actual_right_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")

                self._emit("ex", "de,hl")  # DE = right
                self._emit("pop", "hl")  # HL = left

            # 16-bit subtract: HL = HL - DE
            self._emit_sub16()

            # For EQ/NE, check if result is zero
            if op in (BinaryOp.EQ, BinaryOp.NE):
                self._emit("ld", "a,l")
                self._emit("or", "h")
                if op == BinaryOp.EQ:
                    self._emit("jp", f"nz,{false_label}")  # If not zero, condition is false
                else:
                    self._emit("jp", f"z,{false_label}")   # If zero, condition is false
                return True
            else:
                # For LT/GT/LE/GE with 16-bit, use sign + zero flags
                # After HL = left - right:
                # LT: left < right -> result is negative (sign bit set)
                # GE: left >= right -> result is non-negative
                # GT: left > right -> result is positive and non-zero
                # LE: left <= right -> result is negative or zero
                self._emit_jump_on_false_16bit(op, false_label)
                return True

        return False

    def _gen_condition_jump_true(self, condition: Expr, true_label: str) -> bool:
        """Generate conditional jump to true_label if condition is true.

        Returns True if optimized jump was generated, False if caller should use fallback.
        """
        # Handle constant conditions
        if isinstance(condition, NumberLiteral):
            if condition.value != 0:
                # Always true - unconditional jump
                self._emit("jp", true_label)
            # If zero (always false), no code needed - just fall through
            return True

        # Handle simple identifier
        if isinstance(condition, Identifier):
            cond_type = self._get_expr_type(condition)
            if cond_type == DataType.BYTE:
                self._gen_expr(condition)  # Loads into A
                self._emit("or", "a")     # Set Z flag
                self._emit("jp", f"nz,{true_label}")
                return True
            else:
                self._gen_expr(condition)  # Loads into HL
                self._emit("ld", "a,l")
                self._emit("or", "h")
                self._emit("jp", f"nz,{true_label}")
                return True

        # Handle function call - evaluate and test result
        if isinstance(condition, CallExpr):
            cond_type = self._gen_call_expr(condition)
            if cond_type == DataType.BYTE:
                self._emit("or", "a")     # Set Z flag (result in A)
                self._emit("jp", f"nz,{true_label}")
            else:
                self._emit("ld", "a,l")
                self._emit("or", "h")
                self._emit("jp", f"nz,{true_label}")
            return True

        # Handle NOT - invert the condition
        if isinstance(condition, UnaryExpr) and condition.op == UnaryOp.NOT:
            # NOT x is true when x is false, so jump to true_label when x is false
            return self._gen_condition_jump_false(condition.operand, true_label)

        if not isinstance(condition, BinaryExpr):
            return False

        op = condition.op

        # NOTE: PL/M-80 AND and OR are BITWISE operators, not short-circuit logical operators.
        # IF X OR Y tests if (X bitwise-or Y) is non-zero, NOT if either X or Y is non-zero.
        # So we do NOT handle AND/OR specially here - they fall through to expression evaluation.

        if op not in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.GT, BinaryOp.LE, BinaryOp.GE):
            return False

        # Check for impossible comparisons (e.g., BYTE compared to -1)
        self._check_impossible_comparison(condition.left, condition.right, op)

        # Check if both operands are bytes for optimized comparison
        left_type = self._get_expr_type(condition.left)
        right_type = self._get_expr_type(condition.right)
        both_bytes = (left_type == DataType.BYTE and right_type == DataType.BYTE)

        # Byte comparison with constant - use cp n
        # Handle both regular bytes (0-255) and "negative bytes" (0xFF00-0xFFFF like -1)
        if left_type == DataType.BYTE:
            const_val = None
            if isinstance(condition.right, NumberLiteral):
                val = condition.right.value
                # Allow direct byte values (0-255) or negative byte values (0xFF00-0xFFFF)
                if val <= 255 or (val & 0xFF00) == 0xFF00:
                    const_val = val & 0xFF
            elif isinstance(condition.right, StringLiteral) and len(condition.right.value) == 1:
                const_val = ord(condition.right.value[0])

            if const_val is not None:
                self._gen_expr(condition.left)
                self._emit("cp", self._format_number(const_val))
                self._emit_jump_on_true(op, true_label)
                return True
            elif both_bytes:
                # Byte-to-byte comparison - load right first for efficient SUB
                self._gen_expr(condition.right)
                self._emit("ld", "b,a")  # Save right
                self._gen_expr(condition.left)
                self._emit("sub", "b")    # A = left - right
                self._emit_jump_on_true(op, true_label)
                return True

        if not both_bytes:
            # Optimize ADDRESS comparison with 0: use ld a,l / or h instead of subtraction
            if op in (BinaryOp.EQ, BinaryOp.NE) and isinstance(condition.right, NumberLiteral) and condition.right.value == 0:
                self._gen_expr(condition.left)  # Result in HL
                if left_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("ld", "a,l")
                self._emit("or", "h")  # Z flag set if HL == 0
                if op == BinaryOp.EQ:
                    self._emit("jp", f"z,{true_label}")  # If HL == 0, condition is true
                else:  # NE
                    self._emit("jp", f"nz,{true_label}")  # If HL != 0, condition is true
                return True

            # 16-bit comparison
            self._gen_expr(condition.left)
            if left_type == DataType.BYTE:
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")
            self._emit("push", "hl")

            self._gen_expr(condition.right)
            if right_type == DataType.BYTE:
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")

            self._emit("ex", "de,hl")
            self._emit("pop", "hl")

            self._emit_sub16()

            if op in (BinaryOp.EQ, BinaryOp.NE):
                self._emit("ld", "a,l")
                self._emit("or", "h")
                if op == BinaryOp.EQ:
                    self._emit("jp", f"z,{true_label}")
                else:
                    self._emit("jp", f"nz,{true_label}")
                return True
            else:
                self._emit_jump_on_true_16bit(op, true_label)
                return True

        return False

    def _emit_jump_on_true(self, op: BinaryOp, true_label: str) -> None:
        """Emit jump to true_label if comparison result is true (8-bit compare)."""
        if op == BinaryOp.EQ:
            self._emit("jp", f"z,{true_label}")
        elif op == BinaryOp.NE:
            self._emit("jp", f"nz,{true_label}")
        elif op == BinaryOp.LT:
            self._emit("jp", f"c,{true_label}")
        elif op == BinaryOp.GE:
            self._emit("jp", f"nc,{true_label}")
        elif op == BinaryOp.GT:
            skip = self._new_label("SKIP")
            self._emit("jp", f"c,{skip}")
            self._emit("jp", f"z,{skip}")
            self._emit("jp", true_label)
            self._emit_label(skip)
        elif op == BinaryOp.LE:
            self._emit("jp", f"c,{true_label}")
            self._emit("jp", f"z,{true_label}")

    def _emit_jump_on_true_16bit(self, op: BinaryOp, true_label: str) -> None:
        """Emit jump to true_label for 16-bit unsigned comparison.

        After CALL ??SUBDE (SBC HL,DE), carry flag is set if HL < DE (borrow).
        """
        if op == BinaryOp.LT:
            # left < right: true if carry set
            self._emit("jp", f"c,{true_label}")
        elif op == BinaryOp.GE:
            # left >= right: true if no carry
            self._emit("jp", f"nc,{true_label}")
        elif op == BinaryOp.GT:
            # left > right: true if no carry AND result != 0
            skip = self._new_label("SKIP")
            self._emit("jp", f"c,{skip}")  # left < right -> not greater, skip
            self._emit("ld", "a,l")
            self._emit("or", "h")
            self._emit("jp", f"nz,{true_label}")  # not equal -> greater
            self._emit_label(skip)
        elif op == BinaryOp.LE:
            # left <= right: true if carry OR result == 0
            self._emit("jp", f"c,{true_label}")  # left < right -> true
            self._emit("ld", "a,l")
            self._emit("or", "h")
            self._emit("jp", f"z,{true_label}")  # left == right -> true

    def _emit_jump_on_false(self, op: BinaryOp, false_label: str) -> None:
        """Emit jump to false_label if comparison result is false (8-bit compare)."""
        # After cp n or SUB, flags reflect left - right
        if op == BinaryOp.EQ:
            self._emit("jp", f"nz,{false_label}")  # Jump if not equal (Z=0)
        elif op == BinaryOp.NE:
            self._emit("jp", f"z,{false_label}")   # Jump if equal (Z=1)
        elif op == BinaryOp.LT:
            self._emit("jp", f"nc,{false_label}")  # Jump if not less (C=0)
        elif op == BinaryOp.GE:
            self._emit("jp", f"c,{false_label}")   # Jump if less (C=1)
        elif op == BinaryOp.GT:
            # Greater: not less AND not equal -> C=0 AND Z=0
            self._emit("jp", f"c,{false_label}")   # Jump if less
            self._emit("jp", f"z,{false_label}")   # Jump if equal
        elif op == BinaryOp.LE:
            # Less or equal: C=1 OR Z=1
            # Jump if greater (C=0 AND Z=0)
            skip = self._new_label("SKIP")
            self._emit("jp", f"c,{skip}")   # Less -> condition true, skip jump
            self._emit("jp", f"z,{skip}")   # Equal -> condition true, skip jump
            self._emit("jp", false_label)  # Greater -> condition false
            self._emit_label(skip)

    def _emit_jump_on_false_16bit(self, op: BinaryOp, false_label: str) -> None:
        """Emit jump to false_label for 16-bit unsigned comparison.

        After CALL ??SUBDE (SBC HL,DE), carry flag is set if HL < DE (borrow).
        PL/M ADDRESS is unsigned, so we use carry-based comparisons.
        """
        if op == BinaryOp.LT:
            # left < right: true if carry set (borrow occurred)
            # Jump to false if NO carry (left >= right)
            self._emit("jp", f"nc,{false_label}")
        elif op == BinaryOp.GE:
            # left >= right: true if no carry
            # Jump to false if carry set (left < right)
            self._emit("jp", f"c,{false_label}")
        elif op == BinaryOp.GT:
            # left > right: true if no carry AND result != 0
            # Jump to false if carry OR result == 0
            self._emit("jp", f"c,{false_label}")  # left < right -> false
            self._emit("ld", "a,l")
            self._emit("or", "h")
            self._emit("jp", f"z,{false_label}")  # left == right -> false
        elif op == BinaryOp.LE:
            # left <= right: true if carry OR result == 0
            # Jump to false if no carry AND result != 0
            skip = self._new_label("SKIP")
            self._emit("jp", f"c,{skip}")  # left < right -> true, skip to end
            self._emit("ld", "a,l")
            self._emit("or", "h")
            self._emit("jp", f"z,{skip}")  # left == right -> true
            self._emit("jp", false_label)  # left > right -> false
            self._emit_label(skip)

    def _gen_do_block(self, stmt: DoBlock) -> None:
        """Generate code for simple DO block."""
        # Enter scope with unique identifier for DO block local variables
        self.block_scope_counter += 1
        block_id = self.block_scope_counter
        self.symbols.enter_scope(f"B{block_id}")

        # Save and extend current_proc to include block scope for unique asm names
        old_proc = self.current_proc
        if stmt.decls:  # Only modify if there are declarations
            if self.current_proc:
                self.current_proc = f"{self.current_proc}$B{block_id}"
            else:
                self.current_proc = f"B{block_id}"

        # Local declarations
        for decl in stmt.decls:
            self._gen_declaration(decl)

        # Restore current_proc for statements
        self.current_proc = old_proc

        # Statements
        for s in stmt.stmts:
            self._gen_stmt(s)

        self.symbols.leave_scope()

    def _is_byte_counter_loop(self, condition: Expr) -> tuple[str, int] | None:
        """
        Check if condition matches the pattern (var := var - 1) <> 255.
        Returns (var_asm_name, compare_value) if matched, None otherwise.

        This pattern is a countdown loop: decrement and check for wrap-around.
        """
        if not isinstance(condition, BinaryExpr):
            return None
        if condition.op != BinaryOp.NE:
            return None
        if not isinstance(condition.right, NumberLiteral):
            return None
        if condition.right.value != 255:
            return None

        # Left should be (var := var - 1)
        if not isinstance(condition.left, EmbeddedAssignExpr):
            return None
        embed = condition.left
        if not isinstance(embed.target, Identifier):
            return None

        # Value should be var - 1
        if not isinstance(embed.value, BinaryExpr):
            return None
        if embed.value.op != BinaryOp.SUB:
            return None
        if not isinstance(embed.value.left, Identifier):
            return None
        if embed.value.left.name != embed.target.name:
            return None
        if not isinstance(embed.value.right, NumberLiteral):
            return None
        if embed.value.right.value != 1:
            return None

        # Check that it's a BYTE variable
        var_name = embed.target.name

        # Look up with scoping like _gen_load does
        sym = None
        if self.current_proc:
            parts = self.current_proc.split('$')
            for i in range(len(parts), 0, -1):
                scoped_name = '$'.join(parts[:i]) + '$' + var_name
                sym = self.symbols.lookup(scoped_name)
                if sym:
                    break
        if sym is None:
            sym = self.symbols.lookup(var_name)

        if not sym or sym.data_type != DataType.BYTE:
            return None

        asm_name = sym.asm_name if sym.asm_name else self._mangle_name(var_name)
        return (asm_name, 255)

    def _gen_do_while(self, stmt: DoWhileBlock) -> None:
        """Generate code for DO WHILE block."""
        # Note: DO WHILE 1 is a valid pattern (loop exits in middle via RETURN/GOTO)
        # We only error on impossible comparisons like BYTE <> -1

        loop_label = self._new_label("WHILE")
        end_label = self._new_label("WEND")

        self.loop_stack.append((loop_label, end_label))

        # Check for optimized byte counter loop: DO WHILE (n := n - 1) <> 255
        # NOTE: This optimization is disabled because it doesn't save code -
        # the existing _gen_condition_jump_false already handles this efficiently.
        # For the optimization to help, we'd need to keep the counter in a register
        # and avoid the LD (addr),A inside the loop, which requires data flow analysis to
        # confirm the counter isn't used in the loop body.
        counter_info = None  # self._is_byte_counter_loop(stmt.condition)
        if counter_info:
            var_asm, _ = counter_info
            # Optimized loop: keep counter in C register (C is less commonly used than B)
            # Load counter into C at start
            self._emit("ld", f"a,({var_asm})")
            self._emit("ld", "c,a")

            self._emit_label(loop_label)
            # Decrement C and check for 0xFF (wrap from 0 to 255)
            self._emit("dec", "c")
            self._emit("ld", "a,c")
            self._emit("cp", "0FFH")
            self._emit("jp", f"z,{end_label}")

            # Mark that C is being used as loop counter
            old_loop_reg = getattr(self, 'loop_counter_reg', None)
            self.loop_counter_reg = 'C'

            # Loop body
            for s in stmt.stmts:
                self._gen_stmt(s)

            # Restore loop register tracking
            self.loop_counter_reg = old_loop_reg

            self._emit("jp", loop_label)
            self._emit_label(end_label)

            # Store C back to memory (in case it's used after loop)
            self._emit("ld", "a,c")
            self._emit("ld", f"({var_asm}),a")
        else:
            self._emit_label(loop_label)

            # Try optimized condition jump, fallback to generic
            if not self._gen_condition_jump_false(stmt.condition, end_label):
                result_type = self._gen_expr(stmt.condition)
                # Test result - BYTE in A, ADDRESS in HL
                if result_type == DataType.BYTE:
                    self._emit("or", "a")
                else:
                    self._emit("ld", "a,l")
                    self._emit("or", "h")
                self._emit("jp", f"z,{end_label}")

            # Loop body
            for s in stmt.stmts:
                self._gen_stmt(s)

            self._emit("jp", loop_label)
            self._emit_label(end_label)

        self.loop_stack.pop()

    def _gen_do_iter(self, stmt: DoIterBlock) -> None:
        """Generate code for iterative DO block."""
        loop_label = self._new_label("FOR")
        test_label = self._new_label("TEST")
        incr_label = self._new_label("INCR")
        end_label = self._new_label("NEXT")

        self.loop_stack.append((incr_label, end_label))

        # Determine if index variable is BYTE
        index_type = DataType.ADDRESS
        if isinstance(stmt.index_var, Identifier):
            sym = self._lookup_symbol(stmt.index_var.name)
            if sym and sym.data_type == DataType.BYTE:
                index_type = DataType.BYTE

        # Also check bound type
        bound_type = self._get_expr_type(stmt.bound)
        both_bytes = (index_type == DataType.BYTE and bound_type == DataType.BYTE)

        # Get step value
        step_val = 1
        if stmt.step and isinstance(stmt.step, NumberLiteral):
            step_val = stmt.step.value

        # Check if loop index is used in body - if not, we can use DJNZ on Z80
        index_used = self._index_used_in_body(stmt.index_var, stmt.stmts)

        # Z80 DJNZ optimization: DO I = 0 TO N where I is not used
        # Convert to: B = N+1; do { body } while (--B != 0)
        if (self.target == Target.Z80 and both_bytes and
            step_val == 1 and not index_used and
            isinstance(stmt.start, NumberLiteral) and stmt.start.value == 0):
            # Calculate iteration count = bound + 1
            # If bound is constant, emit LD B,bound+1
            # If bound is variable, emit: load bound; INC A; LD B,A
            if isinstance(stmt.bound, NumberLiteral):
                iter_count = stmt.bound.value + 1
                if iter_count <= 255:
                    self._emit("ld", f"b,{self._format_number(iter_count)}")
                else:
                    # Too many iterations for DJNZ
                    pass  # Fall through to regular loop
            else:
                # Variable bound: A = bound; A++; B = A
                bound_type = self._gen_expr(stmt.bound)
                if bound_type == DataType.ADDRESS:
                    self._emit("ld", "a,l")
                self._emit("inc", "a")  # A = bound + 1 = iteration count
                self._emit("ld", "b,a")  # B = iteration count

            # Only proceed with B-counter loop if we set up B
            if isinstance(stmt.bound, NumberLiteral) and stmt.bound.value + 1 <= 255:
                # Loop body - save B since body may clobber it
                self._emit_label(loop_label)
                self._emit("push", "bc")
                for s in stmt.stmts:
                    self._gen_stmt(s)
                self._emit("pop", "bc")

                # Decrement B and jump if not zero
                # Use dec b; jp nz instead of DJNZ - peephole will convert to DJNZ if in range
                self._emit_label(incr_label)
                self._emit("dec", "b")
                self._emit("jp", f"nz,{loop_label}")

                self._emit_label(end_label)
                self.loop_stack.pop()
                return
            elif not isinstance(stmt.bound, NumberLiteral):
                # Variable bound case - we set up B above
                # But need to handle the case where bound might be 255 (iter count = 256 = 0 in byte)
                # Skip loop if B is 0 (this handles bound = 255 case)
                self._emit("ld", "a,b")
                self._emit("or", "a")
                self._emit("jp", f"z,{end_label}")  # Skip if iteration count is 0

                # Loop body - save B since body may clobber it
                self._emit_label(loop_label)
                self._emit("push", "bc")
                for s in stmt.stmts:
                    self._gen_stmt(s)
                self._emit("pop", "bc")

                # Decrement B and jump if not zero
                # Use dec b; jp nz instead of DJNZ - peephole will convert to DJNZ if in range
                self._emit_label(incr_label)
                self._emit("dec", "b")
                self._emit("jp", f"nz,{loop_label}")

                self._emit_label(end_label)
                self.loop_stack.pop()
                return

        # Check for optimized down-counting loop: DO I = N TO 0
        # When start is variable, bound is 0, and step is -1 (or default counting down)
        is_downcount_to_zero = (
            both_bytes and
            isinstance(stmt.bound, NumberLiteral) and stmt.bound.value == 0 and
            (step_val == -1 or step_val == 0xFF)
        )

        if is_downcount_to_zero:
            # Optimized down-counting byte loop
            # Initialize: load start into A, store to index
            start_type = self._gen_expr(stmt.start)
            if start_type == DataType.ADDRESS:
                self._emit("ld", "a,l")
            self._gen_store(stmt.index_var, DataType.BYTE)

            # Jump to test
            self._emit("jp", test_label)

            # Loop body
            self._emit_label(loop_label)
            for s in stmt.stmts:
                self._gen_stmt(s)

            # Decrement
            self._emit_label(incr_label)
            self._gen_load(stmt.index_var)  # A = index
            self._emit("dec", "a")
            self._gen_store(stmt.index_var, DataType.BYTE)

            # Test: if A >= 0 (not wrapped), continue
            # After DEC, if result is not negative (i.e., >= 0), continue
            self._emit_label(test_label)
            self._gen_load(stmt.index_var)  # A = index
            self._emit("or", "a")  # Set flags
            self._emit("jp", f"p,{loop_label}")  # Jump if positive (bit 7 clear)

            self._emit_label(end_label)
            self.loop_stack.pop()
            return

        # Check for optimized byte loop with constant bound
        if both_bytes and isinstance(stmt.bound, NumberLiteral):
            bound_val = stmt.bound.value

            # Initialize index variable
            start_type = self._gen_expr(stmt.start)
            if start_type == DataType.ADDRESS:
                self._emit("ld", "a,l")
            self._gen_store(stmt.index_var, DataType.BYTE)

            # Jump to test
            self._emit("jp", test_label)

            # Loop body
            self._emit_label(loop_label)
            for s in stmt.stmts:
                self._gen_stmt(s)

            # Increment/Decrement
            self._emit_label(incr_label)
            self._gen_load(stmt.index_var)  # A = index
            if step_val == 1:
                self._emit("inc", "a")
            elif step_val == -1 or step_val == 0xFF:
                self._emit("dec", "a")
            else:
                self._emit("add", f"a,{self._format_number(step_val & 0xFF)}")
            self._gen_store(stmt.index_var, DataType.BYTE)

            # Test condition: compare index with bound
            self._emit_label(test_label)
            self._gen_load(stmt.index_var)  # A = index
            if bound_val == 255:
                # Special case: loop to 0xFF can't use cp 0x100 (truncates to 0)
                # Instead, check if index wrapped to 0 (meaning we exceeded 0xFF)
                self._emit("or", "a")  # Sets Z flag if A == 0
                self._emit("jp", f"nz,{loop_label}")  # Continue if index != 0 (not wrapped)
            else:
                self._emit("cp", self._format_number(bound_val + 1))  # Compare with bound+1
                self._emit("jp", f"C,{loop_label}")  # Continue if index < bound+1 (i.e., index <= bound)

            self._emit_label(end_label)
            self.loop_stack.pop()
            return

        # Check for byte loop with variable bound
        if both_bytes:
            # Initialize index variable as BYTE
            start_type = self._gen_expr(stmt.start)
            if start_type == DataType.ADDRESS:
                self._emit("ld", "a,l")
            self._gen_store(stmt.index_var, DataType.BYTE)

            # Jump to test
            self._emit("jp", test_label)

            # Loop body
            self._emit_label(loop_label)
            for s in stmt.stmts:
                self._gen_stmt(s)

            # Increment/Decrement
            self._emit_label(incr_label)
            self._gen_load(stmt.index_var)  # A = index
            if step_val == 1:
                self._emit("inc", "a")
            elif step_val == -1 or step_val == 0xFF:
                self._emit("dec", "a")
            else:
                self._emit("add", f"a,{self._format_number(step_val & 0xFF)}")
            self._gen_store(stmt.index_var, DataType.BYTE)

            # Test condition: compare index with bound variable
            # Evaluate bound first, then compare with index
            self._emit_label(test_label)
            bound_result = self._gen_expr(stmt.bound)  # A = bound (or HL if ADDRESS)
            if bound_result == DataType.ADDRESS:
                self._emit("ld", "a,l")  # Get low byte if ADDRESS
            self._emit("inc", "a")  # A = bound + 1
            self._emit("ld", "b,a")  # B = bound + 1
            self._gen_load(stmt.index_var)  # A = index
            # cp b computes a - b (index - (bound+1)), sets C if index < bound+1
            self._emit("cp", "B")  # Compare index with bound+1
            self._emit("jp", f"C,{loop_label}")  # Continue if index < bound+1 (i.e., index <= bound)

            self._emit_label(end_label)
            self.loop_stack.pop()
            return

        # General case: 16-bit loop (original code)
        # Initialize index variable
        self._gen_expr(stmt.start)
        self._gen_store(stmt.index_var, DataType.ADDRESS)

        # Jump to test
        self._emit("jp", test_label)

        # Loop body
        self._emit_label(loop_label)
        for s in stmt.stmts:
            self._gen_stmt(s)

        # Increment
        self._emit_label(incr_label)
        self._gen_load(stmt.index_var)
        if step_val == 1:
            self._emit("inc", "hl")
        elif step_val == -1 or step_val == 0xFFFF:
            self._emit("dec", "hl")
        else:
            self._emit("ld", f"de,{self._format_number(step_val)}")
            self._emit("add", "hl,de")
        self._gen_store(stmt.index_var, DataType.ADDRESS)

        # Test condition
        self._emit_label(test_label)
        self._gen_load(stmt.index_var)
        self._emit("ex", "de,hl")  # DE = index
        self._gen_expr(stmt.bound)  # HL = bound

        # Compare: if index > bound, exit (for positive step)
        # HL - DE: if negative (carry), index > bound
        self._emit_sub16()

        # If no borrow (NC), bound >= index, continue
        self._emit("jp", f"nc,{loop_label}")

        self._emit_label(end_label)
        self.loop_stack.pop()

    def _gen_do_case(self, stmt: DoCaseBlock) -> None:
        """Generate code for DO CASE block."""
        end_label = self._new_label("CASEND")

        # Create labels for each case
        case_labels = [self._new_label(f"CASE{i}") for i in range(len(stmt.cases))]

        # Evaluate selector
        selector_type = self._gen_expr(stmt.selector)

        # Generate jump table
        # For small number of cases, use sequential comparisons
        # For larger, use computed jump

        if len(stmt.cases) <= 8:
            # Sequential comparisons - selector can stay in A for BYTE
            if selector_type == DataType.ADDRESS:
                # ADDRESS selector is in HL, move L to A for comparisons
                self._emit("ld", "a,l")
            # else: BYTE selector already in A
            for i, label in enumerate(case_labels):
                self._emit("cp", str(i))
                self._emit("jp", f"z,{label}")
            self._emit("jp", end_label)  # Default: skip all
        else:
            # Jump table approach - needs selector in HL
            if selector_type == DataType.BYTE:
                # Extend BYTE in A to HL
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")
            table_label = self._new_label("JMPTBL")
            self._emit("add", "hl,hl")  # HL = HL * 2 (addresses are 2 bytes)
            self._emit("ld", f"de,{table_label}")
            self._emit("add", "hl,de")  # HL = table + index*2
            self._emit("ld", "e,(hl)")
            self._emit("inc", "hl")
            self._emit("ld", "d,(hl)")
            self._emit("ex", "de,hl")
            self._emit("jp", "(hl)")

            # Jump table (in code segment, right after the jp (hl))
            self._emit_label(table_label)
            for label in case_labels:
                self.output.append(AsmLine(opcode="dw", operands=label))

        # Generate each case
        for i, (case_stmts, label) in enumerate(zip(stmt.cases, case_labels)):
            self._emit_label(label)
            for s in case_stmts:
                self._gen_stmt(s)
            # Only emit JP end_label if last statement doesn't transfer control
            if not self._stmt_transfers_control(case_stmts[-1] if case_stmts else None):
                self._emit("jp", end_label)

        self._emit_label(end_label)

    def _stmt_transfers_control(self, stmt: Stmt | None) -> bool:
        """Check if a statement unconditionally transfers control (no fallthrough)."""
        if stmt is None:
            return False
        if isinstance(stmt, GotoStmt):
            return True
        if isinstance(stmt, ReturnStmt):
            return True
        if isinstance(stmt, HaltStmt):
            return True
        # A labeled statement transfers if its inner statement does
        if isinstance(stmt, LabeledStmt):
            return self._stmt_transfers_control(stmt.stmt)
        # A DO block transfers if its last statement does
        if isinstance(stmt, DoBlock):
            if stmt.stmts:
                return self._stmt_transfers_control(stmt.stmts[-1])
        return False

    # ========================================================================
    # Expression Code Generation
    # ========================================================================

    def _get_expr_type(self, expr: Expr) -> DataType:
        """Determine the type of an expression."""
        if isinstance(expr, NumberLiteral):
            return DataType.BYTE if expr.value <= 255 else DataType.ADDRESS
        elif isinstance(expr, StringLiteral):
            # Single-character strings are treated as byte values in PL/M-80
            if len(expr.value) == 1:
                return DataType.BYTE
            return DataType.ADDRESS  # Address of string
        elif isinstance(expr, Identifier):
            sym = self.symbols.lookup(expr.name)
            if sym:
                # For procedures, return the return type (a bare identifier is a call)
                if sym.kind == SymbolKind.PROCEDURE:
                    return sym.return_type or DataType.ADDRESS
                return sym.data_type or DataType.ADDRESS
            return DataType.ADDRESS
        elif isinstance(expr, EmbeddedAssignExpr):
            # Type is determined by the target variable
            return self._get_expr_type(expr.target)
        elif isinstance(expr, BinaryExpr):
            # Comparisons return BYTE (0 or 1 in A)
            if expr.op in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.GT,
                          BinaryOp.LE, BinaryOp.GE):
                return DataType.BYTE
            # For arithmetic ops, check if both operands are bytes
            left_type = self._get_expr_type(expr.left)
            right_type = self._get_expr_type(expr.right)
            if left_type == DataType.BYTE and right_type == DataType.BYTE:
                if expr.op in (BinaryOp.ADD, BinaryOp.SUB, BinaryOp.AND, BinaryOp.OR, BinaryOp.XOR):
                    return DataType.BYTE
            return DataType.ADDRESS
        elif isinstance(expr, LocationExpr):
            return DataType.ADDRESS
        elif isinstance(expr, CallExpr):
            # Check for built-in functions first
            if isinstance(expr.callee, Identifier):
                name = expr.callee.name.upper()
                # Built-ins that return BYTE
                if name in ('LOW', 'HIGH', 'INPUT', 'ROL', 'ROR'):
                    return DataType.BYTE
                # MEMORY(addr) returns BYTE - it's a byte array
                if name == 'MEMORY':
                    return DataType.BYTE
                # Built-ins that return ADDRESS
                if name in ('SHL', 'SHR', 'DOUBLE', 'LENGTH', 'LAST', 'SIZE',
                           'STACKPTR', 'TIME', 'CPUTIME'):
                    return DataType.ADDRESS
                # Look up symbol to determine type
                sym = self.symbols.lookup(expr.callee.name)
                if sym:
                    if sym.kind == SymbolKind.PROCEDURE:
                        return sym.return_type or DataType.ADDRESS
                    # It's a variable - if it has dimension, this is an array subscript
                    if sym.dimension is not None:
                        # Array subscript returns element type
                        return sym.data_type or DataType.BYTE
                    # Non-array variable being "called" - return its type
                    return sym.data_type or DataType.ADDRESS
            return DataType.ADDRESS
        elif isinstance(expr, UnaryExpr):
            # Unary operations - check which ones return BYTE
            if expr.op in (UnaryOp.NOT, UnaryOp.LOW, UnaryOp.HIGH):
                return DataType.BYTE
            # MINUS and others preserve operand type
            return self._get_expr_type(expr.operand)
        elif isinstance(expr, SubscriptExpr):
            # Array subscript - check the element type of the array
            if isinstance(expr.base, Identifier):
                # Check for MEMORY built-in
                if expr.base.name.upper() == "MEMORY":
                    return DataType.BYTE  # MEMORY is a BYTE array
                sym = self.symbols.lookup(expr.base.name)
                if sym:
                    return sym.data_type or DataType.BYTE
            return DataType.BYTE  # Default to BYTE for array elements
        return DataType.ADDRESS

    def _is_simple_address_expr(self, expr: Expr) -> bool:
        """
        Check if expression is simple enough to load directly into DE.
        Simple expressions are: constants, identifiers (variables), location-of.
        """
        if isinstance(expr, NumberLiteral):
            return True
        if isinstance(expr, Identifier):
            name = expr.name
            # Check for LITERALLY macro
            if name in self.literal_macros:
                return True
            # Look up symbol - simple variables are fine
            sym = self.symbols.lookup(name)
            if sym and sym.kind != SymbolKind.PROCEDURE:
                return True
            # Procedures are not simple - they need to be called
            return False
        if isinstance(expr, LocationExpr):
            # .VAR is simple - just loads address, unless it's a stack-based variable
            if isinstance(expr.operand, Identifier):
                sym = self.symbols.lookup(expr.operand.name)
                if sym and sym.stack_offset is not None:
                    return False  # Stack-based variables need IX+offset calculation
            return True
        return False

    def _gen_simple_to_de(self, expr: Expr) -> None:
        """Load a simple address expression directly into DE."""
        if isinstance(expr, NumberLiteral):
            self._emit("ld", f"de,{self._format_number(expr.value)}")
        elif isinstance(expr, Identifier):
            name = expr.name
            # Handle built-in MEMORY array
            if name.upper() == "MEMORY":
                self.needs_end_symbol = True
                self._emit("ld", "de,__END__")
                return
            # Check for LITERALLY macro
            if name in self.literal_macros:
                macro_val = self.literal_macros[name]
                try:
                    val = self._parse_plm_number(macro_val)
                    self._emit("ld", f"de,{self._format_number(val)}")
                    return
                except ValueError:
                    name = macro_val  # Use expanded name
            # Look up symbol
            sym = self.symbols.lookup(name)
            asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(name)
            if sym:
                # Arrays: load address of array (ld de,label)
                if sym.dimension:
                    self._emit("ld", f"de,{asm_name}")
                elif sym.data_type == DataType.BYTE:
                    # Byte variable - load and extend
                    self._emit("ld", f"a,({asm_name})")
                    self._emit("ld", "e,a")
                    self._emit("ld", "d,0")
                else:
                    # Address variable - load contents into DE
                    self._emit("ld", f"de,({asm_name})")  # Z80: ld de,(addr)
            else:
                # Unknown - assume it's a label/address constant
                self._emit("ld", f"de,{asm_name}")
        elif isinstance(expr, LocationExpr):
            # .VAR - load address of variable
            if isinstance(expr.operand, Identifier):
                name = expr.operand.name
                # Handle built-in MEMORY array
                if name.upper() == "MEMORY":
                    self.needs_end_symbol = True
                    self._emit("ld", "de,__END__")
                    return
                sym = self.symbols.lookup(name)
                # Check for stack-based variable (reentrant procedure parameter/local)
                if sym and sym.stack_offset is not None:
                    # Fall back to gen_expr which handles IX+offset
                    self._gen_expr(expr)
                    self._emit("ex", "de,hl")
                    return
                asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(name)
                self._emit("ld", f"de,{asm_name}")
            else:
                # Complex location - fall back to gen_expr
                self._gen_expr(expr)
                self._emit("ex", "de,hl")

    def _expr_preserves_de(self, expr: Expr) -> bool:
        """
        Check if evaluating this expression preserves the DE register.
        Used to optimize binary expression evaluation order.

        TODO: This ad-hoc approach should be replaced with automatic register
        tracking. See docs/REGISTER_TRACKING_DESIGN.md for the proposed refactor.
        """
        if isinstance(expr, NumberLiteral):
            return True  # ld hl,n doesn't touch DE
        if isinstance(expr, StringLiteral):
            return True  # ld a,n or ld hl,n doesn't touch DE
        if isinstance(expr, Identifier):
            name = expr.name
            # Check for LITERALLY macro
            if name in self.literal_macros:
                return True  # Expands to constant or simple identifier
            # Look up symbol
            sym = None
            if self.current_proc:
                parts = self.current_proc.split('$')
                for i in range(len(parts), 0, -1):
                    scoped_name = '$'.join(parts[:i]) + '$' + name
                    sym = self.symbols.lookup(scoped_name)
                    if sym:
                        break
            if sym is None:
                sym = self.symbols.lookup(name)
            if sym:
                # Procedure calls can touch any register
                if sym.kind == SymbolKind.PROCEDURE:
                    return False
                # BASED variables: ld hl,(addr) then LD - no DE touch
                if sym.based_on:
                    return True
                # Simple variable: ld a,(addr)/ld hl,(addr) - no DE touch
                return True
            # Unknown symbol - assume ld hl,(addr)
            return True
        if isinstance(expr, UnaryExpr):
            # Unary ops on simple expressions don't touch DE
            return self._expr_preserves_de(expr.operand)
        # Complex expressions (BinaryExpr, SubscriptExpr, CallExpr, etc.)
        # may touch DE
        return False

    def _label_reg_need(self, expr: Expr) -> int:
        """
        Label expression with minimum registers needed (Sethi-Ullman algorithm).

        For Z80 with HL as primary and DE as secondary register:
        - Leaf expressions (literals, identifiers): need 1 register
        - Unary expressions: same as operand
        - Binary expressions: if both subtrees need same count, need one more
          to hold intermediate; otherwise need max(left, right)

        Returns the minimum number of 16-bit registers needed to evaluate this
        expression without spilling.
        """
        if isinstance(expr, (NumberLiteral, StringLiteral)):
            return 1  # Load constant into register

        if isinstance(expr, Identifier):
            # Simple variable load needs 1 register
            # But procedure calls may need more (conservatively 2)
            sym = self._lookup_symbol(expr.name)
            if sym and sym.kind == SymbolKind.PROCEDURE:
                return 2  # Function calls may use both registers
            return 1

        if isinstance(expr, UnaryExpr):
            # Unary ops don't need extra registers
            return self._label_reg_need(expr.operand)

        if isinstance(expr, BinaryExpr):
            left_need = self._label_reg_need(expr.left)
            right_need = self._label_reg_need(expr.right)

            if left_need == right_need:
                # Both need same - need one more to hold intermediate
                return left_need + 1
            else:
                # Evaluate harder side first, reuse registers
                return max(left_need, right_need)

        if isinstance(expr, SubscriptExpr):
            # Array subscript: base + index calculation
            # Needs 2 registers (base in DE, index in HL)
            base_need = 1  # Base address is typically 1
            if isinstance(expr.index, (NumberLiteral, Identifier)):
                return max(base_need, 1)  # Simple index
            else:
                index_need = self._label_reg_need(expr.index)
                if base_need == index_need:
                    return base_need + 1
                return max(base_need, index_need)

        if isinstance(expr, CallExpr):
            # Function calls may clobber registers - conservatively need 2
            return 2

        if isinstance(expr, MemberExpr):
            # Structure member access - similar to subscript
            return self._label_reg_need(expr.base)

        # Default: assume 2 registers for safety
        return 2

    def _lookup_symbol(self, name: str) -> 'Symbol | None':
        """Helper to look up a symbol by name, checking scopes."""
        sym = None
        if self.current_proc:
            parts = self.current_proc.split('$')
            for i in range(len(parts), 0, -1):
                scoped_name = '$'.join(parts[:i]) + '$' + name
                sym = self.symbols.lookup(scoped_name)
                if sym:
                    break
        if sym is None:
            sym = self.symbols.lookup(name)
        return sym

    def _gen_expr(self, expr: Expr) -> DataType:
        """
        Generate code for an expression.
        Result is left in A (for BYTE) or HL (for ADDRESS).
        Returns the type of the expression.
        """
        # Clear a_has_l for most expression types (embedded assign sets it)
        if not isinstance(expr, (EmbeddedAssignExpr, CallExpr)):
            self.a_has_l = False

        if isinstance(expr, NumberLiteral):
            # Use ld hl,n for all constants - more efficient (3 bytes vs 5 bytes)
            # Always return ADDRESS since value is in HL, not A
            self._emit("ld", f"hl,{self._format_number(expr.value)}")
            return DataType.ADDRESS

        elif isinstance(expr, StringLiteral):
            # Single-character strings are byte values in PL/M-80
            if len(expr.value) == 1:
                char_val = ord(expr.value[0])
                self._emit("ld", f"a,{self._format_number(char_val)}")
                return DataType.BYTE
            # Load address of string
            label = self._new_string_label()
            self.string_literals.append((label, expr.value))
            self._emit("ld", f"hl,{label}")
            return DataType.ADDRESS

        elif isinstance(expr, Identifier):
            return self._gen_load(expr)

        elif isinstance(expr, BinaryExpr):
            return self._gen_binary(expr)

        elif isinstance(expr, UnaryExpr):
            return self._gen_unary(expr)

        elif isinstance(expr, SubscriptExpr):
            return self._gen_subscript(expr)

        elif isinstance(expr, MemberExpr):
            return self._gen_member(expr)

        elif isinstance(expr, CallExpr):
            return self._gen_call_expr(expr)

        elif isinstance(expr, LocationExpr):
            return self._gen_location(expr)

        elif isinstance(expr, ConstListExpr):
            # .('string') or .(const, const...) - generate inline data and return address
            # This handles both string literals and constant lists
            label = self._new_label("DATA")
            self.data_segment.append(AsmLine(label=label))
            for val in expr.values:
                if isinstance(val, NumberLiteral):
                    self.data_segment.append(
                        AsmLine(opcode="db", operands=self._format_number(val.value))
                    )
                elif isinstance(val, StringLiteral):
                    self.data_segment.append(
                        AsmLine(opcode="db", operands=self._escape_string(val.value))
                    )
            self._emit("ld", f"hl,{label}")
            return DataType.ADDRESS

        elif isinstance(expr, EmbeddedAssignExpr):
            # Evaluate value
            val_type = self._gen_expr(expr.value)

            # Track embedded assignment target for liveness optimization
            target_name = None
            if isinstance(expr.target, Identifier):
                target_name = expr.target.name

            # Check if we can skip the store because A survives to return
            # Conditions:
            # 1. Value is BYTE (in A)
            # 2. Target is simple identifier
            # 3. A survives through IF body (if in IF condition) and remaining stmts
            # 4. Final statement is RETURN of same variable
            skip_store = False
            if val_type == DataType.BYTE and target_name:
                # Gather all statements that A must survive through
                stmts_to_check: list[Stmt] = []

                # If we're in an IF condition, include the IF body
                if self.current_if_stmt:
                    stmts_to_check.append(self.current_if_stmt.then_stmt)
                    if self.current_if_stmt.else_stmt:
                        stmts_to_check.append(self.current_if_stmt.else_stmt)

                # Add remaining statements (after current IF if any)
                stmts_to_check.extend(self.pending_stmts)

                # Check if all statements except last preserve A, and last is RETURN of target
                if stmts_to_check:
                    # All but last must not clobber A
                    last_stmt = stmts_to_check[-1]
                    preceding = stmts_to_check[:-1]

                    if self._a_survives_stmts(preceding):
                        if isinstance(last_stmt, ReturnStmt):
                            if isinstance(last_stmt.value, Identifier):
                                if last_stmt.value.name == target_name:
                                    # A survives to return of same variable - skip store
                                    skip_store = True
                                    self.embedded_assign_target = target_name

            if skip_store:
                # Value is already in A, will be used by return - skip store entirely
                pass
            elif val_type == DataType.BYTE:
                # Value is in A - check if store will clobber A
                # For simple identifier stores to BYTE targets, _gen_store only does
                # ld (asm_name),a which does NOT clobber A
                store_clobbers_a = True
                if isinstance(expr.target, Identifier):
                    sym = self._lookup_symbol(expr.target.name)
                    if sym and sym.data_type == DataType.BYTE:
                        # Simple BYTE store - doesn't clobber A
                        if not sym.based_on and sym.stack_offset is None:
                            store_clobbers_a = False

                if store_clobbers_a:
                    # Need to save A across the store
                    self._emit("ld", "b,a")
                    self._gen_store(expr.target, val_type)
                    self._emit("ld", "a,b")
                else:
                    # Store doesn't clobber A - no need to save/restore
                    self._gen_store(expr.target, val_type)
            else:
                # Value is in HL
                # Check if target is BYTE - _gen_store only touches A, not HL
                target_sym = None
                if isinstance(expr.target, Identifier):
                    target_sym = self.symbols.lookup(expr.target.name)

                if target_sym and target_sym.data_type == DataType.BYTE:
                    # BYTE target - _gen_store does ld a,L; LD (addr),A - HL preserved
                    # After this, A contains L
                    self._gen_store(expr.target, val_type)
                    self.a_has_l = True  # Signal that A already has L
                else:
                    # ADDRESS target - need to preserve HL
                    self._emit("push", "hl")
                    self._gen_store(expr.target, val_type)
                    self._emit("pop", "hl")
            return val_type

        return DataType.ADDRESS

    def _gen_load(self, expr: Expr) -> DataType:
        """Load a variable value into A/HL. Returns the type."""
        if isinstance(expr, Identifier):
            name = expr.name
            upper_name = name.upper()

            # Handle built-in STACKPTR variable
            if upper_name == "STACKPTR":
                # Read stack pointer into HL
                self._emit("ld", "hl,0")
                self._emit("add", "hl,sp")  # HL = HL + SP = SP
                return DataType.ADDRESS

            # Handle flag-testing builtins (can be used without parentheses)
            if upper_name == "CARRY":
                # Return carry flag value
                self._emit("ld", "a,0")
                self._emit("rla")  # Rotate carry into A
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")
                return DataType.BYTE

            if upper_name == "ZERO":
                # Return zero flag value
                true_label = self._new_label("ZF")
                end_label = self._new_label("ZFE")
                self._emit("jp", f"z,{true_label}")
                self._emit("ld", "hl,0")
                self._emit("jp", end_label)
                self._emit_label(true_label)
                self._emit("ld", "hl,0ffh")
                self._emit_label(end_label)
                return DataType.BYTE

            if upper_name == "SIGN":
                # Return sign flag value
                true_label = self._new_label("SF")
                end_label = self._new_label("SFE")
                self._emit("jp", f"m,{true_label}")
                self._emit("ld", "hl,0")
                self._emit("jp", end_label)
                self._emit_label(true_label)
                self._emit("ld", "hl,0ffh")
                self._emit_label(end_label)
                return DataType.BYTE

            if upper_name == "PARITY":
                # Return parity flag value
                true_label = self._new_label("PF")
                end_label = self._new_label("PFE")
                self._emit("jp", f"pe,{true_label}")
                self._emit("ld", "hl,0")
                self._emit("jp", end_label)
                self._emit_label(true_label)
                self._emit("ld", "hl,0ffh")
                self._emit_label(end_label)
                return DataType.BYTE

            # Check for LITERALLY macro - expand recursively
            if name in self.literal_macros:
                macro_val = self.literal_macros[name]
                try:
                    val = self._parse_plm_number(macro_val)
                    # Use ld hl,n for all constants - more efficient (3 bytes vs 5 bytes)
                    # Always return ADDRESS since value is in HL, not A
                    self._emit("ld", f"hl,{self._format_number(val)}")
                    return DataType.ADDRESS
                except ValueError:
                    # Non-numeric literal - recursively process as identifier
                    return self._gen_load(Identifier(name=macro_val))

            # Look up symbol in scope hierarchy
            sym = None
            if self.current_proc:
                parts = self.current_proc.split('$')
                for i in range(len(parts), 0, -1):
                    scoped_name = '$'.join(parts[:i]) + '$' + name
                    sym = self.symbols.lookup(scoped_name)
                    if sym:
                        break
            if sym is None:
                sym = self.symbols.lookup(name)

            # Use mangled asm_name if available, otherwise mangle the name
            asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(name)

            if sym:
                # If it's a procedure with no args, generate a call
                if sym.kind == SymbolKind.PROCEDURE:
                    call_name = sym.asm_name if sym.asm_name else name
                    self._emit("call", call_name)
                    # Result is in A (for BYTE) or HL (for ADDRESS/untyped)
                    if sym.return_type == DataType.BYTE:
                        return DataType.BYTE
                    return sym.return_type or DataType.ADDRESS

                if sym.kind == SymbolKind.LITERAL:
                    try:
                        val = int(sym.literal_value or "0", 0)
                        # Use ld hl,n for all constants - more efficient (3 bytes vs 5 bytes)
                        # Always return ADDRESS since value is in HL, not A
                        self._emit("ld", f"hl,{self._format_number(val)}")
                        return DataType.ADDRESS
                    except ValueError:
                        self._emit("ld", f"hl,{sym.literal_value}")
                        return DataType.ADDRESS

                # Check for BASED variable
                if sym.based_on:
                    # Load the base pointer first - look up the actual asm_name
                    base_sym = self.symbols.lookup(sym.based_on)
                    base_asm_name = base_sym.asm_name if base_sym and base_sym.asm_name else sym.based_on
                    self._emit("ld", f"hl,({base_asm_name})")
                    # Then load from the pointed-to address
                    if sym.data_type == DataType.BYTE:
                        self._emit("ld", "a,(hl)")
                        # Keep BYTE value in A register
                        return DataType.BYTE
                    else:
                        self._emit("ld", "e,(hl)")
                        self._emit("inc", "hl")
                        self._emit("ld", "d,(hl)")
                        self._emit("ex", "de,hl")
                        return DataType.ADDRESS

                # Check for stack-based variable (reentrant procedure local)
                if sym.stack_offset is not None:
                    offset = sym.stack_offset
                    if sym.data_type == DataType.BYTE:
                        self._emit("ld", f"a,(ix+{offset})")
                        return DataType.BYTE
                    else:
                        self._emit("ld", f"l,(ix+{offset})")
                        self._emit("ld", f"h,(ix+{offset + 1})")
                        return DataType.ADDRESS

                if sym.data_type == DataType.BYTE:
                    self._emit("ld", f"a,({asm_name})")
                    # Keep BYTE value in A register for efficient byte operations
                    return DataType.BYTE
                else:
                    self._emit("ld", f"hl,({asm_name})")
                    return DataType.ADDRESS

            # Unknown symbol - assume ADDRESS
            self._emit("ld", f"hl,({asm_name})")
            return DataType.ADDRESS

        else:
            # Complex lvalue - generate address then load
            self._gen_location(LocationExpr(operand=expr))
            self._emit("ld", "a,(hl)")
            # Keep BYTE value in A register
            return DataType.BYTE

    def _gen_store(self, expr: Expr, val_type: DataType) -> None:
        """Store A/HL to a variable."""
        if isinstance(expr, Identifier):
            name = expr.name

            # Handle built-in STACKPTR variable
            if name == "STACKPTR":
                # Set stack pointer from HL
                self._emit("ld", "sp,hl")
                return

            # Check for LITERALLY macro - expand recursively
            if name in self.literal_macros:
                macro_val = self.literal_macros[name]
                try:
                    self._parse_plm_number(macro_val)
                    # Numeric literal can't be stored to
                except ValueError:
                    # Non-numeric literal - recursively process as identifier
                    self._gen_store(Identifier(name=macro_val), val_type)
                    return

            sym = self.symbols.lookup(name)
            # Use mangled asm_name if available, otherwise mangle the name
            asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(name)

            # Check for BASED variable
            if sym and sym.based_on:
                # Load base pointer - look up the actual asm_name
                base_sym = self.symbols.lookup(sym.based_on)
                base_asm_name = base_sym.asm_name if base_sym and base_sym.asm_name else sym.based_on
                if sym.data_type == DataType.BYTE:
                    # Value is in A (if val_type==BYTE) or L (if val_type==ADDRESS)
                    if val_type != DataType.BYTE:
                        self._emit("ld", "a,l")  # Get byte value into A
                    self._emit("ld", "b,a")  # Save value in B
                    self._emit("ld", f"hl,({base_asm_name})")
                    self._emit("ld", "a,b")  # Restore value
                    self._emit("ld", "(hl),a")  # Store via HL
                else:
                    # Save value in HL
                    self._emit("push", "hl")
                    self._emit("ld", f"hl,({base_asm_name})")
                    self._emit("ex", "de,hl")  # DE = address
                    self._emit("pop", "hl")  # HL = value
                    self._emit("ex", "de,hl")  # HL = address, DE = value
                    self._emit("ld", "(hl),e")
                    self._emit("inc", "hl")
                    self._emit("ld", "(hl),d")
                return

            # Check for stack-based variable (reentrant procedure local)
            if sym and sym.stack_offset is not None:
                offset = sym.stack_offset
                if sym.data_type == DataType.BYTE:
                    # Value may be in A (if val_type==BYTE) or L (if val_type==ADDRESS)
                    if val_type != DataType.BYTE:
                        self._emit("ld", "a,l")
                    self._emit("ld", f"(ix+{offset}),a")
                else:
                    # Target is ADDRESS
                    if val_type == DataType.BYTE:
                        # Value is in A, need to zero-extend to HL
                        self._emit("ld", "l,a")
                        self._emit("ld", "h,0")
                    self._emit("ld", f"(ix+{offset}),l")
                    self._emit("ld", f"(ix+{offset + 1}),h")
                return

            if sym and sym.data_type == DataType.BYTE:
                # Value may be in A (if val_type==BYTE) or L (if val_type==ADDRESS)
                if val_type != DataType.BYTE:
                    self._emit("ld", "a,l")
                self._emit("ld", f"({asm_name}),a")
            else:
                # Target is ADDRESS
                if val_type == DataType.BYTE:
                    # Value is in A, need to zero-extend to HL
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("ld", f"({asm_name}),hl")

        elif isinstance(expr, SubscriptExpr):
            # Check for MEMORY(addr) = value special case
            if isinstance(expr.base, Identifier) and expr.base.name.upper() == "MEMORY":
                # MEMORY(addr) = value - store byte to __END__ + addr
                # MEMORY is the predeclared byte array starting at end of program
                self.needs_end_symbol = True
                self._emit("push", "hl")  # Save value
                if isinstance(expr.index, NumberLiteral) and expr.index.value == 0:
                    # MEMORY(0) - just use __END__ directly
                    self._emit("ld", "hl,__END__")
                else:
                    # MEMORY(n) - compute __END__ + n
                    self._gen_expr(expr.index)  # HL = offset
                    self._emit("ld", "de,__END__")
                    self._emit("add", "hl,de")  # HL = __END__ + offset
                self._emit("ex", "de,hl")  # DE = address
                self._emit("pop", "hl")  # HL = value
                self._emit("ld", "a,l")
                self._emit("ld", "(de),a")
                return

            # Check for OUTPUT(port) = value special case
            if isinstance(expr.base, Identifier) and expr.base.name.upper() == "OUTPUT":
                # OUTPUT(port) = value - output byte to I/O port
                # Value is in HL (low byte)
                # Check if port is a constant
                port_arg = expr.index
                port_num = None
                if isinstance(port_arg, NumberLiteral):
                    port_num = port_arg.value
                elif isinstance(port_arg, Identifier):
                    if port_arg.name in self.literal_macros:
                        try:
                            port_num = self._parse_plm_number(self.literal_macros[port_arg.name])
                        except ValueError:
                            pass

                if port_num is not None:
                    # Constant port - use OUT instruction directly
                    self._emit("ld", "a,l")  # Value in A
                    self._emit("out", f"({self._format_number(port_num)}),a")
                else:
                    # Variable port - need runtime support
                    self._emit("push", "hl")  # Save value
                    self._gen_expr(port_arg)  # Evaluate port number
                    self._emit("ld", "c,l")  # Port in C
                    self._emit("pop", "hl")  # Restore value
                    self._emit("ld", "a,l")  # Value in A
                    self._emit("call", "??outp")
                    self.needs_runtime.add("outp")
                return

            # Array element store
            # Check element type
            elem_type = DataType.BYTE
            if isinstance(expr.base, Identifier):
                sym = self.symbols.lookup(expr.base.name)
                if sym and sym.data_type == DataType.ADDRESS:
                    elem_type = DataType.ADDRESS

            self._emit("push", "hl")  # Save value
            self._gen_subscript_addr(expr)  # HL = address
            self._emit("ex", "de,hl")  # DE = address
            self._emit("pop", "hl")  # HL = value

            if elem_type == DataType.ADDRESS:
                # Store 16-bit value
                self._emit("ex", "de,hl")  # HL = address, DE = value
                self._emit("ld", "(hl),e")
                self._emit("inc", "hl")
                self._emit("ld", "(hl),d")
            else:
                # Store BYTE value
                self._emit("ld", "a,l")
                self._emit("ld", "(de),a")

        elif isinstance(expr, MemberExpr):
            # Structure member store
            _, member_type = self._get_member_info(expr)
            self._emit("push", "hl")
            self._gen_member_addr(expr)
            self._emit("ex", "de,hl")  # DE = member address
            self._emit("pop", "hl")  # HL = value
            if member_type == DataType.ADDRESS:
                # Store 16-bit value
                self._emit("ex", "de,hl")  # HL = address, DE = value
                self._emit("ld", "(hl),e")
                self._emit("inc", "hl")
                self._emit("ld", "(hl),d")
            else:
                # Store 8-bit value
                self._emit("ld", "a,l")
                self._emit("ld", "(de),a")

        elif isinstance(expr, CallExpr):
            # Special built-in assignment targets: OUTPUT(port) = value
            if isinstance(expr.callee, Identifier) and expr.callee.name.upper() == "OUTPUT":
                # OUTPUT(port) = value - output byte to I/O port
                # Value is in HL (low byte)
                # Check if port is a constant
                port_arg = expr.args[0]
                port_num = None
                if isinstance(port_arg, NumberLiteral):
                    port_num = port_arg.value
                elif isinstance(port_arg, Identifier):
                    if port_arg.name in self.literal_macros:
                        try:
                            port_num = self._parse_plm_number(self.literal_macros[port_arg.name])
                        except ValueError:
                            pass

                if port_num is not None:
                    # Constant port - use OUT instruction directly
                    self._emit("ld", "a,l")  # Value in A
                    self._emit("out", f"({self._format_number(port_num)}),a")
                else:
                    # Variable port - need runtime support
                    self._emit("push", "hl")  # Save value
                    self._gen_expr(port_arg)  # Evaluate port number
                    self._emit("ld", "c,l")  # Port in C
                    self._emit("pop", "hl")  # Restore value
                    self._emit("ld", "a,l")  # Value in A
                    self._emit("call", "??outp")
                    self.needs_runtime.add("outp")
                return

            # Check for MEMORY(addr) = value special case (built-in byte array at __END__)
            if isinstance(expr.callee, Identifier) and expr.callee.name.upper() == "MEMORY" and len(expr.args) == 1:
                # MEMORY(addr) = value - store byte to __END__ + addr
                self.needs_end_symbol = True
                addr_arg = expr.args[0]
                # Check for constant address
                addr_val = None
                if isinstance(addr_arg, NumberLiteral):
                    addr_val = addr_arg.value
                elif isinstance(addr_arg, Identifier) and addr_arg.name in self.literal_macros:
                    try:
                        addr_val = self._parse_plm_number(self.literal_macros[addr_arg.name])
                    except (ValueError, TypeError):
                        pass

                if addr_val is not None:
                    # Constant offset - use LD (__END__+offset),A
                    if val_type != DataType.BYTE:
                        self._emit("ld", "a,l")
                    if addr_val == 0:
                        self._emit("ld", "(__END__),a")
                    else:
                        self._emit("ld", f"(__END__+{self._format_number(addr_val)}),a")
                else:
                    # Variable offset - compute __END__ + offset, then store
                    if val_type == DataType.BYTE:
                        # Save value on stack - cannot use B as _gen_expr may clobber it
                        self._emit("push", "af")  # Save value on stack
                        self._gen_expr(addr_arg)  # HL = offset
                        self._emit("ld", "de,__END__")
                        self._emit("add", "hl,de")  # HL = __END__ + offset
                        self._emit("pop", "af")  # Restore value to A
                        self._emit("ld", "(hl),a")  # Store value at (HL)
                    else:
                        self._emit("push", "hl")  # Save value
                        self._gen_expr(addr_arg)  # HL = offset
                        self._emit("ld", "de,__END__")
                        self._emit("add", "hl,de")  # HL = __END__ + offset
                        self._emit("ex", "de,hl")  # DE = address
                        self._emit("pop", "hl")  # HL = value
                        self._emit("ld", "a,l")
                        self._emit("ld", "(de),a")
                return

            # Check if this is actually an array subscript (parser creates CallExpr for arr(idx))
            if isinstance(expr.callee, Identifier) and len(expr.args) == 1:
                sym = self.symbols.lookup(expr.callee.name)
                if sym and sym.kind != SymbolKind.PROCEDURE:
                    # It's an array access - delegate to SubscriptExpr handling
                    subscript = SubscriptExpr(expr.callee, expr.args[0])
                    # Check for constant index optimization (but NOT for BASED variables)
                    if isinstance(expr.args[0], NumberLiteral) and not sym.based_on:
                        # Constant index - can compute address directly
                        asm_name = sym.asm_name if sym.asm_name else self._mangle_name(expr.callee.name)
                        # Check element type
                        elem_type = sym.data_type if sym else DataType.BYTE
                        elem_size = 2 if elem_type == DataType.ADDRESS else 1
                        offset = expr.args[0].value * elem_size

                        if elem_type == DataType.ADDRESS:
                            # Store 16-bit value (value in HL)
                            if val_type == DataType.BYTE:
                                # Expand BYTE to ADDRESS
                                self._emit("ld", "l,a")
                                self._emit("ld", "h,0")
                            if offset == 0:
                                self._emit("ld", f"({asm_name}),hl")
                            else:
                                # Need to store at offset - use ld de,addr; then store via DE
                                self._emit("ld", f"de,{asm_name}+{offset}")
                                self._emit("ex", "de,hl")  # HL = address, DE = value
                                self._emit("ld", "(hl),e")
                                self._emit("inc", "hl")
                                self._emit("ld", "(hl),d")
                        else:
                            # Store BYTE value (value in A)
                            if val_type != DataType.BYTE:
                                self._emit("ld", "a,l")  # Get low byte
                            if offset == 0:
                                self._emit("ld", f"({asm_name}),a")
                            else:
                                self._emit("ld", f"({asm_name}+{offset}),a")
                    else:
                        # Variable index - need to compute address
                        elem_type = sym.data_type if sym else DataType.BYTE
                        if elem_type == DataType.ADDRESS:
                            # ADDRESS array - need to store 16-bit value
                            if val_type == DataType.BYTE:
                                # Expand BYTE (in A) to ADDRESS (in HL)
                                self._emit("ld", "l,a")
                                self._emit("ld", "h,0")
                            # Value in HL - save it, compute address, store
                            self._emit("push", "hl")  # Save value
                            self._gen_subscript_addr(subscript)  # HL = address
                            self._emit("pop", "de")  # DE = value
                            self._emit("ld", "(hl),e")  # Store low byte at (HL)
                            self._emit("inc", "hl")
                            self._emit("ld", "(hl),d")  # Store high byte at (HL+1)
                        else:
                            # BYTE array - store single byte
                            if val_type != DataType.BYTE:
                                self._emit("ld", "a,l")  # Get low byte from ADDRESS
                            # Value in A - save it on stack, compute address, store
                            # NOTE: Cannot use B register here because _gen_subscript_addr
                            # may call _gen_byte_binary which uses B for its calculations
                            self._emit("push", "af")  # Save value on stack
                            self._gen_subscript_addr(subscript)  # HL = address
                            self._emit("pop", "af")  # Restore value to A
                            self._emit("ld", "(hl),a")  # Store value
                    return

            # Check if this is a member array subscript: struct.member(idx)
            if isinstance(expr.callee, MemberExpr) and len(expr.args) == 1:
                member_expr = expr.callee
                idx_expr = expr.args[0]
                _, member_type = self._get_member_info(member_expr)
                elem_size = 2 if member_type == DataType.ADDRESS else 1

                # Save value, compute address, then store based on ELEMENT type (not value type)
                if member_type == DataType.ADDRESS:
                    # ADDRESS array member - store 16-bit
                    if val_type == DataType.BYTE:
                        self._emit("ld", "l,a")
                        self._emit("ld", "h,0")
                    self._emit("push", "hl")  # Save value
                    self._gen_member_addr(member_expr)
                    # Add index offset
                    if isinstance(idx_expr, NumberLiteral):
                        self._emit_add_hl_const(idx_expr.value * elem_size)
                    else:
                        self._emit("push", "hl")
                        idx_type = self._gen_expr(idx_expr)
                        if idx_type == DataType.BYTE:
                            self._emit("ld", "l,a")
                            self._emit("ld", "h,0")
                        self._emit("add", "hl,hl")  # *2 for ADDRESS elements
                        self._emit("pop", "de")
                        self._emit("add", "hl,de")
                    self._emit("pop", "de")  # DE = value
                    self._emit("ld", "(hl),e")
                    self._emit("inc", "hl")
                    self._emit("ld", "(hl),d")
                else:
                    # BYTE array member - store 8-bit only
                    if val_type != DataType.BYTE:
                        self._emit("ld", "a,l")  # Get low byte from ADDRESS value
                    self._emit("push", "af")  # Save value on stack
                    self._gen_member_addr(member_expr)
                    # Add index offset
                    if isinstance(idx_expr, NumberLiteral):
                        self._emit_add_hl_const(idx_expr.value)
                    else:
                        self._emit("push", "hl")
                        idx_type = self._gen_expr(idx_expr)
                        if idx_type == DataType.BYTE:
                            self._emit("ld", "l,a")
                            self._emit("ld", "h,0")
                        self._emit("pop", "de")
                        self._emit("add", "hl,de")
                    self._emit("pop", "af")  # Restore value to A
                    self._emit("ld", "(hl),a")  # Store single byte
                return

            # Unknown call target - fall through to complex store
            self._emit("push", "hl")
            self._gen_location(LocationExpr(operand=expr))
            self._emit("ex", "de,hl")
            self._emit("pop", "hl")
            if val_type == DataType.BYTE:
                self._emit("ld", "a,l")
                self._emit("ld", "(de),a")
            else:
                self._emit("ex", "de,hl")
                self._emit("ld", "(hl),e")
                self._emit("inc", "hl")
                self._emit("ld", "(hl),d")
            return

        else:
            # Complex store
            self._emit("push", "hl")  # Save value
            self._gen_location(LocationExpr(operand=expr))  # HL = address
            self._emit("ex", "de,hl")  # DE = address
            self._emit("pop", "hl")  # HL = value
            # Store based on type
            if val_type == DataType.BYTE:
                self._emit("ld", "a,l")
                self._emit("ld", "(de),a")
            else:
                self._emit("ex", "de,hl")  # HL = address, DE = value
                self._emit("ld", "(hl),e")
                self._emit("inc", "hl")
                self._emit("ld", "(hl),d")

    def _match_shl_double_8(self, expr: Expr) -> Expr | None:
        """
        Match the pattern SHL(DOUBLE(x), 8) and return x.

        This pattern represents: x * 256 (shift byte to high position)
        Returns None if pattern doesn't match.
        """
        # Must be a call to SHL
        if not isinstance(expr, CallExpr):
            return None
        if not isinstance(expr.callee, Identifier):
            return None
        if expr.callee.name.upper() != 'SHL':
            return None
        if len(expr.args) != 2:
            return None

        # Second arg must be 8 (shift count)
        shift_arg = expr.args[1]
        shift_count = None
        if isinstance(shift_arg, NumberLiteral):
            shift_count = shift_arg.value
        elif isinstance(shift_arg, Identifier) and shift_arg.name in self.literal_macros:
            try:
                shift_count = self._parse_plm_number(self.literal_macros[shift_arg.name])
            except ValueError:
                pass
        if shift_count != 8:
            return None

        # First arg must be DOUBLE(x)
        double_expr = expr.args[0]
        if not isinstance(double_expr, CallExpr):
            return None
        if not isinstance(double_expr.callee, Identifier):
            return None
        if double_expr.callee.name.upper() != 'DOUBLE':
            return None
        if len(double_expr.args) != 1:
            return None

        # Return the inner expression (the byte value)
        inner = double_expr.args[0]
        # Verify it's a byte type
        if self._get_expr_type(inner) != DataType.BYTE:
            return None

        return inner

    def _gen_binary(self, expr: BinaryExpr) -> DataType:
        """Generate code for binary expression."""
        op = expr.op

        # Special case: SHL(DOUBLE(hi), 8) OR lo -> combine two bytes into address
        # Pattern: (hi * 256) OR lo where hi and lo are bytes
        if op == BinaryOp.OR:
            hi_expr = self._match_shl_double_8(expr.left)
            if hi_expr is not None:
                lo_type = self._get_expr_type(expr.right)
                if lo_type == DataType.BYTE:
                    # Generate optimized: hi -> H, lo -> L
                    self._gen_expr(hi_expr)  # Result in A
                    self._emit("ld", "h,a")  # H = high byte
                    self._gen_expr(expr.right)  # Result in A
                    self._emit("ld", "l,a")  # L = low byte
                    # HL now contains combined address
                    return DataType.ADDRESS

        # Determine operand types for optimization
        left_type = self._get_expr_type(expr.left)
        right_type = self._get_expr_type(expr.right)
        both_bytes = (left_type == DataType.BYTE and right_type == DataType.BYTE)

        # Special case: ADDRESS comparison with 0 - use OR L,H for zero test
        if op in (BinaryOp.EQ, BinaryOp.NE) and left_type == DataType.ADDRESS:
            if isinstance(expr.right, NumberLiteral) and expr.right.value == 0:
                return self._gen_addr_zero_comparison(expr.left, op)

        # Check for impossible comparisons (e.g., BYTE compared to -1)
        if op in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.GT, BinaryOp.LE, BinaryOp.GE):
            self._check_impossible_comparison(expr.left, expr.right, op)

        # Special case: byte comparison with constant - use cp n
        # When comparing BYTE to a constant, truncate constant to byte if valid
        # (values 0-255 or 0xFF00-0xFFFF for "negative bytes" like -1)
        if op in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.GT,
                  BinaryOp.LE, BinaryOp.GE):
            if left_type == DataType.BYTE:
                const_val = None
                if isinstance(expr.right, NumberLiteral):
                    val = expr.right.value
                    # Allow direct byte values (0-255) or negative byte values (0xFF00-0xFFFF)
                    if val <= 255 or (val & 0xFF00) == 0xFF00:
                        const_val = val & 0xFF
                elif isinstance(expr.right, StringLiteral) and len(expr.right.value) == 1:
                    const_val = ord(expr.right.value[0])

                if const_val is not None:
                    return self._gen_byte_comparison_const(expr.left, op, const_val)
                elif both_bytes:
                    return self._gen_byte_comparison(expr.left, expr.right, op)

        # For byte operations, use efficient byte path
        if both_bytes and op in (BinaryOp.ADD, BinaryOp.SUB, BinaryOp.AND,
                                  BinaryOp.OR, BinaryOp.XOR):
            return self._gen_byte_binary(expr.left, expr.right, op)

        # Optimize BYTE PLUS/MINUS 0: just ADC A,0 or SBC A,0 (preserves carry chain)
        if op == BinaryOp.PLUS and left_type == DataType.BYTE and isinstance(expr.right, NumberLiteral) and expr.right.value == 0:
            self._gen_expr(expr.left)  # Result in A
            self._emit("adc", "a,0")  # add carry
            return DataType.BYTE

        if op == BinaryOp.MINUS and left_type == DataType.BYTE and isinstance(expr.right, NumberLiteral) and expr.right.value == 0:
            self._gen_expr(expr.left)  # Result in A
            self._emit("sbc", "a,0")  # subtract carry
            return DataType.BYTE

        # Optimize ADDRESS +/- constant: use inc hl/dec hl for small, LD DE + add hl,DE for larger
        # Only apply if left operand actually ends up in HL (ADDRESS type)
        if op == BinaryOp.ADD and isinstance(expr.right, NumberLiteral) and left_type == DataType.ADDRESS:
            const_val = expr.right.value
            if 1 <= const_val <= 4:  # Small constants: use repeated inc hl
                self._gen_expr(expr.left)
                for _ in range(const_val):
                    self._emit("inc", "hl")
                return DataType.ADDRESS
            else:
                # Larger constants: use ld de,const; add hl,DE (no PUSH/POP needed)
                self._gen_expr(expr.left)  # HL = left
                self._emit("ld", f"de,{self._format_number(const_val)}")
                self._emit("add", "hl,de")  # HL = HL + DE
                return DataType.ADDRESS
        elif op == BinaryOp.SUB and isinstance(expr.right, NumberLiteral) and left_type == DataType.ADDRESS:
            const_val = expr.right.value
            if 1 <= const_val <= 4:  # Small constants: use repeated dec hl
                self._gen_expr(expr.left)
                for _ in range(const_val):
                    self._emit("dec", "hl")
                return DataType.ADDRESS
            else:
                # Larger constants: use subtraction without PUSH/POP
                self._gen_expr(expr.left)  # HL = left
                self._emit("ld", f"de,{self._format_number(const_val)}")
                # HL = HL - DE
                self._emit_sub16()
                return DataType.ADDRESS

        # Optimize MUL by constant power of 2: use shifts instead of runtime call
        if op == BinaryOp.MUL:
            const_val = None
            other_expr = None
            if isinstance(expr.right, NumberLiteral):
                const_val = expr.right.value
                other_expr = expr.left
            elif isinstance(expr.left, NumberLiteral):
                const_val = expr.left.value
                other_expr = expr.right

            if const_val is not None and const_val > 0:
                # Check if power of 2: x & (x-1) == 0 for powers of 2
                if (const_val & (const_val - 1)) == 0:
                    # Count shifts needed (log2)
                    shift_count = 0
                    temp = const_val
                    while temp > 1:
                        temp >>= 1
                        shift_count += 1

                    # Generate the non-constant operand
                    other_type = self._gen_expr(other_expr)
                    if other_type == DataType.BYTE:
                        self._emit("ld", "l,a")
                        self._emit("ld", "h,0")

                    # Apply shifts
                    for _ in range(shift_count):
                        self._emit("add", "hl,hl")  # HL *= 2

                    return DataType.ADDRESS

        # Fall through to 16-bit operations
        # Use register allocator to manage DE for holding one operand
        # This automatically handles nested expressions via spill/restore

        # Sethi-Ullman optimization: evaluate the subtree needing more registers first
        # This minimizes spills by freeing registers before the simpler subtree needs them
        left_need = self._label_reg_need(expr.left)
        right_need = self._label_reg_need(expr.right)

        # Path 1: Simple left optimization - if left is simple AND DE is free,
        # evaluate right first to avoid the extra ex de,hl
        if self._expr_preserves_de(expr.left) and self.regs.is_free('de'):
            # Evaluate right first -> HL, then move to DE
            right_result = self._gen_expr(expr.right)
            if right_result == DataType.BYTE:
                self._emit("ld", "e,a")
                self._emit("ld", "d,0")
            else:
                self._emit("ex", "de,hl")  # DE = right
            # Mark DE as busy (it holds right operand)
            self.regs.mark_busy('de', 'binary_right_simple')
            # Now evaluate left - DE is preserved since left is simple
            left_result = self._gen_expr(expr.left)
            if left_result == DataType.BYTE:
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")
            # Now: HL = left, DE = right
            self.regs.mark_free('de')
            used_general_path = False

        # Path 2: Sethi-Ullman - right needs more registers, evaluate it first
        # This minimizes spills by computing the harder subtree before claiming DE
        elif right_need > left_need:
            # Evaluate right first (harder subtree) -> HL
            right_result = self._gen_expr(expr.right)
            if right_result == DataType.BYTE:
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")

            # Claim DE to hold right operand while we evaluate left
            self.regs.need_reg('de', 'binary_right_sethi', self._emit)
            self._emit("ex", "de,hl")  # DE = right

            # Evaluate left operand (simpler subtree) -> HL
            # This may recursively need DE, causing spill of our right value
            left_result = self._gen_expr(expr.left)
            if left_result == DataType.BYTE:
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")

            # Now: HL = left, DE = right (restored from spill if inner expr used DE)
            # This is already in the correct order for HL op DE
            used_general_path = True

        else:
            # Path 3: General case - left needs >= registers, evaluate it first
            # Evaluate left operand -> HL
            left_result = self._gen_expr(expr.left)
            if left_result == DataType.BYTE:
                # Extend A to HL
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")

            # Claim DE to hold left operand while we evaluate right
            # If DE is already busy (nested binary expr), it will be spilled
            self.regs.need_reg('de', 'binary_left', self._emit)
            self._emit("ex", "de,hl")  # DE = left

            # Evaluate right operand -> HL
            # This may recursively need DE, causing spill of our left value.
            # When inner expr releases DE, our left value is restored.
            right_result = self._gen_expr(expr.right)
            if right_result == DataType.BYTE:
                # Extend A to HL
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")

            # Now: HL = right, DE = left (restored from spill if inner expr used DE)
            # Swap so HL = left, DE = right (standard order for operations)
            self._emit("ex", "de,hl")

            # NOTE: Don't release DE yet - we still need it for the operation!
            # Will release after the operation below.
            used_general_path = True

        if op == BinaryOp.ADD:
            self._emit("add", "hl,de")  # HL = HL + DE

        elif op == BinaryOp.SUB:
            # HL = HL - DE
            self._emit_sub16()

        elif op == BinaryOp.MUL:
            # Power of 2 cases handled above with early return
            # This handles non-power-of-2 multiplies
            self.needs_runtime.add("mul16")
            self._emit("call", "??mul16")

        elif op == BinaryOp.DIV:
            self.needs_runtime.add("div16")
            self._emit("call", "??div16")

        elif op == BinaryOp.MOD:
            self.needs_runtime.add("mod16")
            self._emit("call", "??mod16")

        elif op == BinaryOp.AND:
            self._emit("ld", "a,l")
            self._emit("and", "e")
            self._emit("ld", "l,a")
            self._emit("ld", "a,h")
            self._emit("and", "d")
            self._emit("ld", "h,a")

        elif op == BinaryOp.OR:
            self._emit("ld", "a,l")
            self._emit("or", "e")
            self._emit("ld", "l,a")
            self._emit("ld", "a,h")
            self._emit("or", "d")
            self._emit("ld", "h,a")

        elif op == BinaryOp.XOR:
            self._emit("ld", "a,l")
            self._emit("xor", "e")
            self._emit("ld", "l,a")
            self._emit("ld", "a,h")
            self._emit("xor", "d")
            self._emit("ld", "h,a")

        elif op in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.GT,
                   BinaryOp.LE, BinaryOp.GE):
            # Comparison returns result in A (BYTE), not HL
            # Release DE before returning if we used the general path
            if used_general_path:
                self.regs.release_reg('de', self._emit)
            return self._gen_comparison(op)

        elif op == BinaryOp.PLUS:
            # PLUS: add with carry from previous operation
            self._emit("ld", "a,l")
            self._emit("adc", "a,e")
            self._emit("ld", "l,a")
            self._emit("ld", "a,h")
            self._emit("adc", "a,d")
            self._emit("ld", "h,a")

        elif op == BinaryOp.MINUS:
            # MINUS: subtract with borrow from previous operation
            self._emit("ld", "a,l")
            self._emit("sbc", "a,e")
            self._emit("ld", "l,a")
            self._emit("ld", "a,h")
            self._emit("sbc", "a,d")
            self._emit("ld", "h,a")

        # Release DE after operation if we used the general path
        if used_general_path:
            self.regs.release_reg('de', self._emit)

        return DataType.ADDRESS

    def _gen_comparison(self, op: BinaryOp) -> DataType:
        """Generate code for comparison. HL=left, DE=right. Result in A (0 or 0FFH)."""
        true_label = self._new_label("TRUE")
        false_label = self._new_label("FALSE")
        end_label = self._new_label("CMP")

        # Subtract: HL = HL - DE, flags set from high byte subtraction
        self._emit_sub16()
        # Now: HL = left - right, flags set from SBB D (carry = borrow)

        if op == BinaryOp.EQ:
            self._emit("ld", "a,l")
            self._emit("or", "h")  # or high and low
            self._emit("jp", f"z,{true_label}")
        elif op == BinaryOp.NE:
            self._emit("ld", "a,l")
            self._emit("or", "h")
            self._emit("jp", f"nz,{true_label}")
        elif op == BinaryOp.LT:
            # left < right if borrow occurred
            self._emit("jp", f"c,{true_label}")
        elif op == BinaryOp.GE:
            self._emit("jp", f"nc,{true_label}")
        elif op == BinaryOp.GT:
            # left > right: no borrow AND not equal
            self._emit("jp", f"c,{false_label}")  # If left < right, false
            self._emit("ld", "a,l")
            self._emit("or", "h")
            self._emit("jp", f"nz,{true_label}")  # If not equal, left > right
        elif op == BinaryOp.LE:
            self._emit("jp", f"c,{true_label}")  # left < right
            self._emit("ld", "a,l")
            self._emit("or", "h")
            self._emit("jp", f"z,{true_label}")  # left == right

        # False case - return 0 in A
        self._emit_label(false_label)
        self._emit("xor", "a")
        self._emit("jp", end_label)

        # True case - return 0FFH in A (PL/M TRUE = 0FFH)
        self._emit_label(true_label)
        self._emit("ld", "a,0ffh")

        self._emit_label(end_label)
        return DataType.BYTE

    def _gen_addr_zero_comparison(self, left: Expr, op: BinaryOp) -> DataType:
        """Generate optimized ADDRESS comparison with 0 using OR.

        For N = 0 or N <> 0 where N is ADDRESS, use:
            ld a,L
            or h
            jp z/jp nz label
        instead of full 16-bit subtraction.
        """
        # Load left operand into HL
        self._gen_expr(left)

        # Test if HL is zero: A = L | H
        self._emit("ld", "a,l")
        self._emit("or", "h")

        # Generate result based on comparison type
        true_label = self._new_label("TRUE")
        end_label = self._new_label("CMP")

        if op == BinaryOp.EQ:
            self._emit("jp", f"z,{true_label}")  # If zero, equal
        elif op == BinaryOp.NE:
            self._emit("jp", f"nz,{true_label}")  # If not zero, not equal

        # False case - return 0 in A
        self._emit("xor", "a")
        self._emit("jp", end_label)

        # True case - return 0FFH in A (PL/M TRUE = 0FFH)
        self._emit_label(true_label)
        self._emit("ld", "a,0ffh")

        self._emit_label(end_label)
        return DataType.BYTE

    def _gen_byte_comparison_const(self, left: Expr, op: BinaryOp, const_val: int) -> DataType:
        """Generate optimized byte comparison with constant using cp n."""
        # Load left operand into A
        left_type = self._gen_expr(left)
        if left_type != DataType.BYTE:
            # If not already a byte, take low byte
            self._emit("ld", "a,l")

        # Compare with constant
        self._emit("cp", self._format_number(const_val))

        # Generate result based on comparison type
        true_label = self._new_label("TRUE")
        end_label = self._new_label("CMP")

        if op == BinaryOp.EQ:
            self._emit("jp", f"z,{true_label}")
        elif op == BinaryOp.NE:
            self._emit("jp", f"nz,{true_label}")
        elif op == BinaryOp.LT:
            self._emit("jp", f"c,{true_label}")
        elif op == BinaryOp.GE:
            self._emit("jp", f"nc,{true_label}")
        elif op == BinaryOp.GT:
            # A > const: not equal AND not less (jp nc and jp nz)
            self._emit("jp", f"c,{end_label}")  # If less, false
            self._emit("jp", f"z,{end_label}")  # If equal, false
            self._emit("jp", true_label)  # Otherwise true
        elif op == BinaryOp.LE:
            self._emit("jp", f"c,{true_label}")  # Less than -> true
            self._emit("jp", f"z,{true_label}")  # Equal -> true

        # False case - return 0 in A
        self._emit("xor", "a")
        self._emit("jp", end_label)

        # True case - return 0FFH in A (PL/M TRUE = 0FFH)
        self._emit_label(true_label)
        self._emit("ld", "a,0ffh")

        self._emit_label(end_label)
        return DataType.BYTE  # Comparisons return BYTE (0 or 0FFH)

    def _gen_byte_comparison(self, left: Expr, right: Expr, op: BinaryOp) -> DataType:
        """Generate optimized byte comparison between two byte values."""
        # Load right first, then left, so we can SUB B directly
        self._gen_expr(right)  # Result in A
        self._emit("ld", "b,a")  # Save right in B

        self._gen_expr(left)  # Result in A (left)
        self._emit("sub", "b")    # A = left - right, flags set

        # Generate result
        true_label = self._new_label("TRUE")
        end_label = self._new_label("CMP")

        if op == BinaryOp.EQ:
            self._emit("jp", f"z,{true_label}")
        elif op == BinaryOp.NE:
            self._emit("jp", f"nz,{true_label}")
        elif op == BinaryOp.LT:
            self._emit("jp", f"c,{true_label}")
        elif op == BinaryOp.GE:
            self._emit("jp", f"nc,{true_label}")
        elif op == BinaryOp.GT:
            self._emit("jp", f"c,{end_label}")
            self._emit("jp", f"z,{end_label}")
            self._emit("jp", true_label)
        elif op == BinaryOp.LE:
            self._emit("jp", f"c,{true_label}")
            self._emit("jp", f"z,{true_label}")

        # False case - return 0 in A
        self._emit("xor", "a")
        self._emit("jp", end_label)

        # True case - return 0FFH in A (PL/M TRUE = 0FFH)
        self._emit_label(true_label)
        self._emit("ld", "a,0ffh")

        self._emit_label(end_label)
        return DataType.BYTE  # Comparisons return BYTE (0 or 0FFH)

    def _gen_byte_binary(self, left: Expr, right: Expr, op: BinaryOp) -> DataType:
        """Generate optimized byte arithmetic/logical operation."""
        # Special case: right is constant - use immediate instructions
        right_const = self._get_const_byte_value(right)
        if right_const is not None:
            self._gen_expr_to_a(left)  # Load left into A
            const = self._format_number(right_const)
            if op == BinaryOp.ADD:
                self._emit("add", f"a,{const}")  # A = A + const
            elif op == BinaryOp.SUB:
                self._emit("sub", const)  # A = A - const
            elif op == BinaryOp.AND:
                self._emit("and", const)  # A = A AND const
            elif op == BinaryOp.OR:
                self._emit("or", const)  # A = A OR const
            elif op == BinaryOp.XOR:
                self._emit("xor", const)  # A = A XOR const
            return DataType.BYTE

        # Special case: const - var (left is constant, subtraction)
        left_const = self._get_const_byte_value(left)
        if op == BinaryOp.SUB and left_const is not None:
            if left_const == 1:
                # 1 - x is a boolean toggle: use XOR 1
                self._gen_expr_to_a(right)
                self._emit("xor", "1")
            else:
                # const - x: negate x then add const
                # -x = NOT(x) + 1, so const - x = NOT(x) + 1 + const = NOT(x) + (const+1)
                # But we need to handle overflow: use CPL; ADD A,const; INC A for (const-x)
                # Actually: A = right; CPL; INC A gives -right; then ADD A,const
                self._gen_expr_to_a(right)
                self._emit("cpl")  # A = NOT(right)
                self._emit("inc", "a")  # A = -right (two's complement)
                self._emit("add", f"a,{self._format_number(left_const)}")  # A = const - right
            return DataType.BYTE

        # For subtraction, load right first so we can do SUB B directly
        if op == BinaryOp.SUB:
            self._gen_expr_to_a(right)
            self._emit("ld", "b,a")  # Save right in B
            self._gen_expr_to_a(left)
            self._emit("sub", "b")    # A = left - right
            return DataType.BYTE

        # General case: load left into A, save to B
        self._gen_expr_to_a(left)
        self._emit("ld", "b,a")  # Save left in B

        # Load right into A
        self._gen_expr_to_a(right)
        # Now B = left, A = right

        # Perform operation: result = left op right
        if op == BinaryOp.ADD:
            self._emit("add", "a,b")  # A = A + B = right + left
        elif op == BinaryOp.AND:
            self._emit("and", "b")  # A = A AND B
        elif op == BinaryOp.OR:
            self._emit("or", "b")  # A = A OR B
        elif op == BinaryOp.XOR:
            self._emit("xor", "b")  # A = A XOR B

        # Result is in A, return BYTE
        return DataType.BYTE

    def _gen_expr_to_a(self, expr: Expr) -> None:
        """Generate code to load an expression into A (for byte operations)."""
        # Check for constant value (NumberLiteral, StringLiteral, or LITERALLY macro)
        const_val = self._get_const_byte_value(expr)
        if const_val is not None:
            self._emit("ld", f"a,{self._format_number(const_val)}")
        elif isinstance(expr, NumberLiteral):
            # Large constant - load low byte
            self._emit("ld", f"a,{self._format_number(expr.value & 0xFF)}")
        else:
            result_type = self._gen_expr(expr)
            if result_type == DataType.ADDRESS:
                # Value is in HL, get low byte into A
                self._emit("ld", "a,l")

    def _gen_unary(self, expr: UnaryExpr) -> DataType:
        """Generate code for unary expression."""
        operand_type = self._gen_expr(expr.operand)

        if expr.op == UnaryOp.NEG:
            if operand_type == DataType.BYTE:
                # Negate A: A = 0 - A
                self._emit("cpl")
                self._emit("inc", "a")
                return DataType.BYTE
            else:
                # Negate HL: HL = 0 - HL
                self._emit("ld", "a,l")
                self._emit("cpl")
                self._emit("ld", "l,a")
                self._emit("ld", "a,h")
                self._emit("cpl")
                self._emit("ld", "h,a")
                self._emit("inc", "hl")
                return DataType.ADDRESS

        elif expr.op == UnaryOp.NOT:
            if operand_type == DataType.BYTE:
                # Bitwise NOT: complement all bits
                # A contains the byte value
                self._emit("cpl")  # A = ~A (bitwise complement)
                return DataType.BYTE
            else:
                # Bitwise NOT for ADDRESS: complement both bytes
                self._emit("ld", "a,l")
                self._emit("cpl")
                self._emit("ld", "l,a")
                self._emit("ld", "a,h")
                self._emit("cpl")
                self._emit("ld", "h,a")
                return DataType.ADDRESS

        elif expr.op == UnaryOp.LOW:
            # Value is in HL (ADDRESS) or A (BYTE from operand)
            if operand_type == DataType.ADDRESS:
                self._emit("ld", "a,l")  # Get low byte into A
            # else: already in A from BYTE operand
            return DataType.BYTE

        elif expr.op == UnaryOp.HIGH:
            # Value is in HL (ADDRESS) or A (BYTE from operand)
            if operand_type == DataType.ADDRESS:
                self._emit("ld", "a,h")  # Get high byte into A
            else:
                self._emit("xor", "a")  # BYTE has no high byte, return 0
            return DataType.BYTE

        return DataType.ADDRESS

    # Built-in functions that might be parsed as subscripts
    BUILTIN_FUNCS = {'LENGTH', 'LAST', 'SIZE', 'HIGH', 'LOW', 'DOUBLE', 'ROL', 'ROR',
                     'SHL', 'SHR', 'SCL', 'SCR', 'INPUT', 'OUTPUT', 'TIME', 'MOVE',
                     'CPUTIME', 'MEMORY', 'STACKPTR', 'DEC'}

    def _gen_subscript(self, expr: SubscriptExpr) -> DataType:
        """Generate code for array subscript - load value."""
        # Check if this is actually a built-in function call
        if isinstance(expr.base, Identifier) and expr.base.name.upper() in self.BUILTIN_FUNCS:
            # Treat as function call
            call = CallExpr(callee=expr.base, args=[expr.index])
            return self._gen_call_expr(call)

        # Check element type
        elem_type = DataType.BYTE
        if isinstance(expr.base, Identifier):
            sym = self.symbols.lookup(expr.base.name)
            if sym and sym.data_type == DataType.ADDRESS:
                elem_type = DataType.ADDRESS

        self._gen_subscript_addr(expr)

        if elem_type == DataType.ADDRESS:
            # Load 16-bit value: low byte first, then high byte
            self._emit("ld", "e,(hl)")
            self._emit("inc", "hl")
            self._emit("ld", "d,(hl)")
            self._emit("ex", "de,hl")  # HL = value
            return DataType.ADDRESS
        else:
            # Load BYTE value into A
            self._emit("ld", "a,(hl)")
            return DataType.BYTE

    def _gen_subscript_addr(self, expr: SubscriptExpr) -> None:
        """Generate code to compute address of array element."""
        # Check if this is actually a built-in function call (in a .func(arg) context)
        if isinstance(expr.base, Identifier) and expr.base.name.upper() in self.BUILTIN_FUNCS:
            # Generate the function call - result in HL
            call = CallExpr(callee=expr.base, args=[expr.index])
            self._gen_call_expr(call)
            return

        # Check element size
        elem_size = 1  # Default BYTE
        if isinstance(expr.base, Identifier):
            sym = self.symbols.lookup(expr.base.name)
            if sym:
                if sym.struct_members:
                    # Structure array - element size is sum of all member sizes
                    elem_size = 0
                    for member in sym.struct_members:
                        member_size = 2 if member.data_type == DataType.ADDRESS else 1
                        if member.dimension:
                            member_size *= member.dimension
                        elem_size += member_size
                elif sym.data_type == DataType.ADDRESS:
                    elem_size = 2

        # OPTIMIZATION: Constant folding for label+constant
        # If base is a simple identifier (label) and index is constant, fold them
        if isinstance(expr.base, Identifier) and isinstance(expr.index, NumberLiteral):
            sym = self.symbols.lookup(expr.base.name)
            if sym and not sym.based_on:
                # Regular array with constant index - can fold: ld hl,label+offset
                asm_name = sym.asm_name if sym.asm_name else self._mangle_name(expr.base.name)
                offset = expr.index.value * elem_size
                if offset == 0:
                    self._emit("ld", f"hl,{asm_name}")
                else:
                    self._emit("ld", f"hl,{asm_name}+{offset}")
                return

        # Check for optimized BYTE index path first (avoids loading base into HL)
        if not isinstance(expr.index, NumberLiteral):
            idx_type = self._get_expr_type(expr.index)
            if idx_type == DataType.BYTE and elem_size == 1 and isinstance(expr.base, Identifier):
                # Optimized byte index with identifier base
                # Evaluate index first (before loading base), then load base into DE
                self._gen_expr(expr.index)  # A = index (byte)
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")  # HL = index (zero-extended)
                # Load base directly into DE
                sym = self.symbols.lookup(expr.base.name)
                if sym and sym.based_on:
                    base_sym = self.symbols.lookup(sym.based_on)
                    base_asm_name = base_sym.asm_name if base_sym and base_sym.asm_name else self._mangle_name(sym.based_on)
                    self._emit("ld", f"de,({base_asm_name})")  # Z80: ld de,(addr)
                else:
                    asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(expr.base.name)
                    self._emit("ld", f"de,{asm_name}")
                self._emit("add", "hl,de")  # HL = index + base
                return

        # Get base address (non-constant or BASED variable case)
        if isinstance(expr.base, Identifier):
            sym = self.symbols.lookup(expr.base.name)
            if sym and sym.based_on:
                # BASED variable - load the base pointer from the based_on variable
                base_sym = self.symbols.lookup(sym.based_on)
                base_asm_name = base_sym.asm_name if base_sym and base_sym.asm_name else self._mangle_name(sym.based_on)
                self._emit("ld", f"hl,({base_asm_name})")
            else:
                # Regular array - use address of array
                asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(expr.base.name)
                self._emit("ld", f"hl,{asm_name}")
        else:
            self._gen_expr(expr.base)

        # Optimize for constant index (only reached for BASED or computed base)
        if isinstance(expr.index, NumberLiteral):
            offset = expr.index.value * elem_size
            self._emit_add_hl_const(offset)
        else:
            # Variable index with complex base or ADDRESS index
            # Use register allocator for DE to hold base while evaluating index
            # This integrates with binary expression spill/restore mechanism
            idx_type = self._get_expr_type(expr.index)

            # Claim DE to hold base address while evaluating index
            # If DE is already busy (e.g., from outer binary expression), it will be spilled
            self.regs.need_reg('de', 'subscript_base', self._emit)
            self._emit("ex", "de,hl")  # DE = base

            # Get index - may recursively use DE (triggers spill/restore)
            result_type = self._gen_expr(expr.index)

            # If index was BYTE (in A), extend to HL
            if result_type == DataType.BYTE:
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")

            if elem_size > 1:
                # Multiply index by element size
                # Check if elem_size is a power of 2: x & (x-1) == 0
                if (elem_size & (elem_size - 1)) == 0:
                    # Power of 2: use repeated add hl,hl
                    temp = elem_size
                    while temp > 1:
                        self._emit("add", "hl,hl")  # HL *= 2
                        temp >>= 1
                else:
                    # General case: HL = HL * elem_size using multiply routine
                    # Need to save DE (base) across multiply call
                    self._emit("push", "de")
                    self._emit("ld", f"de,{elem_size}")
                    self._emit("call", "??mul16")
                    self._emit("pop", "de")
                    self.needs_runtime.add("mul16")

            # Add index to base - DE holds base (possibly restored from spill)
            self._emit("add", "hl,de")

            # Release DE after operation
            self.regs.release_reg('de', self._emit)

    def _get_member_info(self, expr: MemberExpr) -> tuple[int, DataType]:
        """Get offset and type for a structure member."""
        offset = 0
        member_type = DataType.BYTE

        # Get the base variable's symbol to find struct_members
        sym = None
        if isinstance(expr.base, Identifier):
            sym = self.symbols.lookup(expr.base.name)
        elif isinstance(expr.base, CallExpr) and isinstance(expr.base.callee, Identifier):
            # Array of structures: POINTS(0).X - look up array's struct_members
            sym = self.symbols.lookup(expr.base.callee.name)
        elif isinstance(expr.base, SubscriptExpr) and isinstance(expr.base.base, Identifier):
            # Subscript form: look up array's struct_members
            sym = self.symbols.lookup(expr.base.base.name)

        if sym and sym.struct_members:
            for member in sym.struct_members:
                if member.name == expr.member:
                    member_type = member.data_type
                    break
                # Add size of this member
                member_size = 2 if member.data_type == DataType.ADDRESS else 1
                if member.dimension:
                    member_size *= member.dimension
                offset += member_size

        return offset, member_type

    def _gen_member(self, expr: MemberExpr) -> DataType:
        """Generate code for structure member access - load value."""
        _, member_type = self._get_member_info(expr)
        self._gen_member_addr(expr)

        if member_type == DataType.ADDRESS:
            # Load 16-bit value
            self._emit("ld", "e,(hl)")
            self._emit("inc", "hl")
            self._emit("ld", "d,(hl)")
            self._emit("ex", "de,hl")  # HL = value
            return DataType.ADDRESS
        else:
            # Load 8-bit value
            self._emit("ld", "a,(hl)")
            self._emit("ld", "l,a")
            self._emit("ld", "h,0")
            return DataType.BYTE

    def _gen_member_addr(self, expr: MemberExpr) -> None:
        """Generate code to compute address of structure member."""
        # Handle base expression - need ADDRESS, not value
        base = expr.base
        if isinstance(base, Identifier):
            # Simple identifier - look it up to determine if it's a structure
            name = base.name
            sym = None
            if self.current_proc:
                parts = self.current_proc.split('$')
                for i in range(len(parts), 0, -1):
                    scoped_name = '$'.join(parts[:i]) + '$' + name
                    sym = self.symbols.lookup(scoped_name)
                    if sym:
                        break
            if sym is None:
                sym = self.symbols.lookup(name)

            if sym and sym.struct_members:
                # This is a structure variable - we need its ADDRESS, not its value
                if sym.based_on:
                    # BASED structure - load the pointer from the base variable
                    base_sym = self.symbols.lookup(sym.based_on)
                    base_asm_name = base_sym.asm_name if base_sym and base_sym.asm_name else self._mangle_name(sym.based_on)
                    self._emit("ld", f"hl,({base_asm_name})")
                else:
                    # Regular structure - generate the address directly
                    asm_name = sym.asm_name or name
                    self._emit("ld", f"hl,{asm_name}")
            elif sym and sym.based_on:
                # BASED variable without struct_members - load the pointer
                base_sym = self.symbols.lookup(sym.based_on)
                base_asm_name = base_sym.asm_name if base_sym and base_sym.asm_name else self._mangle_name(sym.based_on)
                self._emit("ld", f"hl,({base_asm_name})")
            else:
                # Not a structure, load as expression (pointer to structure)
                self._gen_expr(base)
        elif isinstance(base, CallExpr) and isinstance(base.callee, Identifier):
            # Check if this is actually an array subscript (variable, not procedure)
            name = base.callee.name
            sym = None
            if self.current_proc:
                parts = self.current_proc.split('$')
                for i in range(len(parts), 0, -1):
                    scoped_name = '$'.join(parts[:i]) + '$' + name
                    sym = self.symbols.lookup(scoped_name)
                    if sym:
                        break
            if sym is None:
                sym = self.symbols.lookup(name)

            if sym and sym.kind in (SymbolKind.VARIABLE, SymbolKind.PARAMETER) and len(base.args) == 1:
                # This is an array subscript - we need its ADDRESS, not value
                subscript = SubscriptExpr(base.callee, base.args[0])
                self._gen_subscript_addr(subscript)
            else:
                # Regular expression
                self._gen_expr(base)
        else:
            self._gen_expr(base)

        offset, _ = self._get_member_info(expr)

        # Add offset to base address (in HL)
        self._emit_add_hl_const(offset)

    def _gen_call_expr(self, expr: CallExpr) -> DataType:
        """Generate code for function call expression or array subscript.

        Since the parser can't distinguish array(index) from func(arg), this is
        determined here by looking up the symbol type.
        """
        # Handle built-in functions
        if isinstance(expr.callee, Identifier):
            name = expr.callee.name
            result = self._gen_builtin(name, expr.args)
            if result is not None:
                return result

            # Check if this is actually an array subscript (variable, not procedure)
            # Try each level of the scope hierarchy (innermost to outermost)
            sym = None
            if self.current_proc:
                parts = self.current_proc.split('$')
                for i in range(len(parts), 0, -1):
                    scoped_name = '$'.join(parts[:i]) + '$' + name
                    sym = self.symbols.lookup(scoped_name)
                    if sym:
                        break
            if sym is None:
                sym = self.symbols.lookup(name)

            # If it's DEFINITELY a variable (not procedure, not unknown) with single arg,
            # treat as subscript. If unknown, assume it's a procedure call.
            if sym and sym.kind in (SymbolKind.VARIABLE, SymbolKind.PARAMETER) and len(expr.args) == 1:
                # This is an array subscript expression
                subscript = SubscriptExpr(expr.callee, expr.args[0])
                return self._gen_subscript(subscript)

        # Handle member array subscript: struct.member(idx) or struct(idx).member(idx2)
        if isinstance(expr.callee, MemberExpr) and len(expr.args) == 1:
            # This is subscripting a structure member array, not a function call
            # First get the address of the member, then subscript it
            member_expr = expr.callee
            idx_expr = expr.args[0]

            # Generate the member address
            self._gen_member_addr(member_expr)

            # Get the member element type
            _, member_type = self._get_member_info(member_expr)
            elem_size = 2 if member_type == DataType.ADDRESS else 1

            # Add the subscript offset
            if isinstance(idx_expr, NumberLiteral):
                offset = idx_expr.value * elem_size
                self._emit_add_hl_const(offset)
            else:
                # Variable index - use allocator for DE
                self.regs.need_reg('de', 'member_subscript_base', self._emit)
                self._emit("ex", "de,hl")  # DE = member addr
                idx_type = self._gen_expr(idx_expr)
                if idx_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                if elem_size == 2:
                    self._emit("add", "hl,hl")  # HL *= 2
                self._emit("add", "hl,de")  # HL = addr + offset
                self.regs.release_reg('de', self._emit)

            # Load the value from the computed address
            if member_type == DataType.ADDRESS:
                self._emit("ld", "e,(hl)")
                self._emit("inc", "hl")
                self._emit("ld", "d,(hl)")
                self._emit("ex", "de,hl")
                return DataType.ADDRESS
            else:
                self._emit("ld", "a,(hl)")
                return DataType.BYTE

        # Regular function call
        # Look up procedure symbol first to determine calling convention
        sym = None
        call_name = None
        full_callee_name = None
        if isinstance(expr.callee, Identifier):
            name = expr.callee.name
            if self.current_proc:
                parts = self.current_proc.split('$')
                for i in range(len(parts), 0, -1):
                    scoped_name = '$'.join(parts[:i]) + '$' + name
                    sym = self.symbols.lookup(scoped_name)
                    if sym:
                        break
            if sym is None:
                sym = self.symbols.lookup(name)
            call_name = sym.asm_name if sym and sym.asm_name else name
            if sym:
                full_callee_name = sym.name

            # Optimize CP/M BDOS calls: MON1(func, arg) and MON2(func, arg)
            # These are the standard PL/M wrappers for BDOS calls
            if name.upper() in ('MON1', 'MON2') and len(expr.args) == 2:
                func_arg, addr_arg = expr.args
                # Check if function number is a constant
                func_num = None
                if isinstance(func_arg, NumberLiteral):
                    func_num = func_arg.value
                elif isinstance(func_arg, Identifier) and func_arg.name in self.literal_macros:
                    try:
                        func_num = self._parse_plm_number(self.literal_macros[func_arg.name])
                    except (ValueError, TypeError):
                        pass

                if func_num is not None and func_num <= 255:
                    # Generate direct BDOS call: ld c,func; ld de,addr; CALL 5
                    self._emit("ld", f"c,{self._format_number(func_num)}")
                    addr_type = self._gen_expr(addr_arg)
                    if addr_type == DataType.BYTE:
                        # BYTE arg goes in E; BDOS ignores D for byte-only functions
                        self._emit("ld", "e,a")
                    else:
                        self._emit("ex", "de,hl")  # DE = addr
                    self._emit("call", "5")  # BDOS entry point
                    # Result in A for MON2 (BYTE), HL for MON3 (ADDRESS)
                    # MON1 is void but returns whatever was in registers
                    return DataType.BYTE if name.upper() == 'MON2' else DataType.ADDRESS

        # For non-reentrant LOCAL procedures, store args directly to parameter memory
        use_stack = True
        if sym and sym.kind == SymbolKind.PROCEDURE and not sym.is_reentrant and not sym.is_external:
            use_stack = False

        if use_stack:
            # Stack-based parameter passing
            for arg in expr.args:
                arg_type = self._gen_expr(arg)
                if arg_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("push", "hl")
        else:
            # Direct memory parameter passing (non-reentrant)
            # Last param is passed in register (A for BYTE, HL for ADDRESS)
            last_param_idx = len(expr.args) - 1
            uses_reg = sym.uses_reg_param and len(expr.args) > 0

            for i, arg in enumerate(expr.args):
                if sym and i < len(sym.params):
                    param_name = sym.params[i]
                    param_type = sym.param_types[i] if i < len(sym.param_types) else DataType.ADDRESS
                    is_last = (i == last_param_idx)

                    # Last param passed in register - just evaluate it
                    if is_last and uses_reg:
                        # Optimize constants for BYTE
                        if param_type == DataType.BYTE:
                            if isinstance(arg, NumberLiteral) and arg.value <= 255:
                                self._emit("ld", f"a,{self._format_number(arg.value)}")
                                continue
                            elif isinstance(arg, StringLiteral) and len(arg.value) == 1:
                                self._emit("ld", f"a,{self._format_number(ord(arg.value[0]))}")
                                continue
                        # Evaluate arg - result in A (BYTE) or HL (ADDRESS)
                        arg_type = self._gen_expr(arg)
                        if param_type == DataType.BYTE and arg_type == DataType.ADDRESS:
                            self._emit("ld", "a,l")
                        elif param_type == DataType.ADDRESS and arg_type == DataType.BYTE:
                            self._emit("ld", "l,a")
                            self._emit("ld", "h,0")
                        continue

                    # Non-last params: store to memory
                    param_asm = None
                    if (hasattr(self, 'storage_labels')
                        and full_callee_name in self.storage_labels
                        and param_name in self.storage_labels[full_callee_name]):
                        param_asm = self.storage_labels[full_callee_name][param_name]
                    else:
                        proc_base = sym.asm_name if sym.asm_name else name
                        if proc_base.startswith('@'):
                            proc_base = proc_base[1:]
                        param_asm = f"@{proc_base}${self._mangle_name(param_name)}"

                    # Optimize: for BYTE parameter with constant, use ld a,n directly
                    if param_type == DataType.BYTE:
                        if isinstance(arg, NumberLiteral) and arg.value <= 255:
                            self._emit("ld", f"a,{self._format_number(arg.value)}")
                            self._emit("ld", f"({param_asm}),a")
                            continue
                        elif isinstance(arg, StringLiteral) and len(arg.value) == 1:
                            self._emit("ld", f"a,{self._format_number(ord(arg.value[0]))}")
                            self._emit("ld", f"({param_asm}),a")
                            continue

                    arg_type = self._gen_expr(arg)
                    if param_type == DataType.BYTE or arg_type == DataType.BYTE:
                        # For BYTE param, ensure we have result in A
                        if arg_type == DataType.ADDRESS:
                            self._emit("ld", "a,l")
                        self._emit("ld", f"({param_asm}),a")
                    else:
                        self._emit("ld", f"({param_asm}),hl")

        if isinstance(expr.callee, Identifier):
            self._emit("call", call_name)
        else:
            self._gen_expr(expr.callee)
            self._emit("jp", "(hl)")

        # Clean up stack - only for stack-based calls
        if use_stack and expr.args:
            for _ in expr.args:
                self._emit("pop", "de")  # Dummy pop

        # Result is in HL (or A for BYTE)
        return sym.return_type if sym and sym.return_type else DataType.ADDRESS

    def _gen_builtin(self, name: str, args: list[Expr]) -> DataType | None:
        """Generate code for built-in function. Returns type if handled, None otherwise."""

        if name == "INPUT":
            if args:
                # For 8080, IN instruction requires immediate port number
                # Check if we can resolve to a constant (number or LITERALLY macro)
                arg = args[0]
                port_num = None
                if isinstance(arg, NumberLiteral):
                    port_num = arg.value
                elif isinstance(arg, Identifier):
                    # Check if it's a LITERALLY macro
                    if arg.name in self.literal_macros:
                        try:
                            port_num = self._parse_plm_number(self.literal_macros[arg.name])
                        except ValueError:
                            pass

                if port_num is not None:
                    self._emit("in", f"a,({self._format_number(port_num)})")
                else:
                    # Variable port - need runtime support (rare in practice)
                    self._gen_expr(arg)
                    self._emit("call", "??inp")
                    self.needs_runtime.add("inp")
            else:
                self._emit("in", "a,(0)")
            self._emit("ld", "l,a")
            self._emit("ld", "h,0")
            return DataType.BYTE

        if name == "LOW":
            arg_type = self._gen_expr(args[0])
            if arg_type == DataType.ADDRESS:
                # Check if A already has L (from embedded assign to BYTE)
                if self.a_has_l:
                    self.a_has_l = False  # Consume the flag
                else:
                    self._emit("ld", "a,l")  # Get low byte into A
            # else: already in A from BYTE operand
            return DataType.BYTE

        if name == "HIGH":
            arg_type = self._gen_expr(args[0])
            if arg_type == DataType.ADDRESS:
                self._emit("ld", "a,h")  # Get high byte into A
            else:
                self._emit("xor", "a")  # BYTE has no high byte, return 0
            return DataType.BYTE

        if name == "DOUBLE":
            # DOUBLE(x) zero-extends a BYTE to ADDRESS (e.g., DOUBLE(0xFF) = 0x00FF)
            arg_type = self._gen_expr(args[0])
            if arg_type == DataType.BYTE:
                # BYTE value is in A, zero-extend to HL
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")
            # else: ADDRESS value is already in HL, no conversion needed
            return DataType.ADDRESS

        if name == "SHL":
            # Check for constant shift amount
            shift_arg = args[1]
            shift_count = None
            if isinstance(shift_arg, NumberLiteral):
                shift_count = shift_arg.value
            elif isinstance(shift_arg, Identifier) and shift_arg.name in self.literal_macros:
                try:
                    shift_count = self._parse_plm_number(self.literal_macros[shift_arg.name])
                except ValueError:
                    pass

            if shift_count is not None and 0 <= shift_count <= 15:
                arg_type = self._gen_expr(args[0])  # Value in HL (or A if BYTE)
                if arg_type == DataType.BYTE:
                    # BYTE value is in A, move to HL
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")

                if shift_count == 0:
                    pass  # No shift needed
                elif shift_count >= 8:
                    # Shift by 8+: L goes to H, L becomes 0, then shift H left
                    self._emit("ld", "h,l")  # H = L (shift by 8)
                    self._emit("ld", "l,0")
                    remaining = shift_count - 8
                    for _ in range(remaining):
                        self._emit("add", "hl,hl")  # HL *= 2
                else:
                    # Inline add hl,HL for shifts 1-7 (1 byte each, no loop overhead)
                    for _ in range(shift_count):
                        self._emit("add", "hl,hl")  # HL *= 2
                # TODO: Investigate root cause. MUL16 zeroes DE as side effect,
                # and some code path relies on this. Without this ld de,0,
                # strength-reduced multiplications fail. See tests/bug_80un.plm.
                self._emit("ld", "de,0")
                return DataType.ADDRESS

            # Variable shift - use loop
            arg_type = self._gen_expr(args[0])
            if arg_type == DataType.BYTE:
                # BYTE value is in A, move to HL
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")
            self._emit("push", "hl")
            count_type = self._gen_expr(args[1])
            if count_type == DataType.BYTE:
                self._emit("ld", "c,a")  # Count in C (from A for byte)
            else:
                self._emit("ld", "c,l")  # Count in C (from L for address)
            self._emit("pop", "hl")   # Value in HL
            shift_loop = self._new_label("SHL")
            end_label = self._new_label("SHLE")
            self._emit_label(shift_loop)
            self._emit("dec", "c")
            self._emit("jp", f"m,{end_label}")
            self._emit("add", "hl,hl")  # HL = HL * 2
            self._emit("jp", shift_loop)
            self._emit_label(end_label)
            return DataType.ADDRESS

        if name == "SHR":
            # Check for constant shift amount
            shift_arg = args[1]
            shift_count = None
            if isinstance(shift_arg, NumberLiteral):
                shift_count = shift_arg.value
            elif isinstance(shift_arg, Identifier) and shift_arg.name in self.literal_macros:
                try:
                    shift_count = self._parse_plm_number(self.literal_macros[shift_arg.name])
                except ValueError:
                    pass

            if shift_count is not None and 0 <= shift_count <= 15:
                arg_type = self._gen_expr(args[0])  # Value in HL (or A if BYTE)
                if arg_type == DataType.BYTE:
                    # BYTE value is in A, move to HL
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")

                if shift_count == 0:
                    pass  # No shift needed
                elif shift_count >= 8:
                    # Shift by 8+ : result is H >> (count-8)
                    remaining = shift_count - 8
                    if remaining == 0:
                        # Exact shift by 8
                        self._emit("ld", "l,h")  # L = H
                        self._emit("ld", "h,0")
                    elif self.target == Target.Z80 and remaining <= 4:
                        # Z80: use SRL which doesn't need carry clearing
                        # Note: SRL is Z80-only instruction
                        self._emit("ld", "a,h")
                        for _ in range(remaining):
                            self._emit("srl", "a")  # Z80-only instruction
                        self._emit("ld", "l,a")
                        self._emit("ld", "h,0")
                    else:
                        # Larger shifts (>4): load H into A, shift, store
                        self._emit("ld", "a,h")
                        for _ in range(remaining):
                            self._emit("or", "a")  # Clear carry
                            self._emit("rra")
                        self._emit("ld", "l,a")
                        self._emit("ld", "h,0")
                elif shift_count == 7:
                    # Special case for shift by 7: result = (H << 1) | (L >> 7)
                    # This is faster than 7 iterations
                    # RLC sets carry from bit 7, so no need to clear carry first
                    self._emit("ld", "a,l")
                    self._emit("rlca")        # Carry = bit 7 of L (A also rotated but we discard it)
                    self._emit("ld", "a,h")
                    self._emit("rla")        # A = (H << 1) | carry
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                elif shift_count <= 3:
                    # Small shifts: inline the loop
                    for _ in range(shift_count):
                        if self.target == Target.Z80:
                            # Z80: use SRL/RR - 2 instructions per shift
                            self._emit("srl", "h")  # H >>= 1, bit 0 -> carry
                            self._emit("rr", "l")   # L = (carry << 7) | (L >> 1)
                        else:
                            # 8080: need to go through accumulator - 7 instructions per shift
                            self._emit("or", "a")  # Clear carry
                            self._emit("ld", "a,h")
                            self._emit("rra")
                            self._emit("ld", "h,a")
                            self._emit("ld", "a,l")
                            self._emit("rra")
                            self._emit("ld", "l,a")
                else:
                    # For 4-6 shifts, use a counted loop
                    if self.target == Target.Z80:
                        # Z80: use DJNZ for tight loop
                        self._emit("ld", f"b,{shift_count}")
                        shift_loop = self._new_label("SHR")
                        self._emit_label(shift_loop)
                        self._emit("srl", "h")  # H >>= 1, bit 0 -> carry
                        self._emit("rr", "l")   # L = (carry << 7) | (L >> 1)
                        self._emit("djnz", shift_loop)
                    else:
                        # 8080: use C counter and JP m
                        self._emit("ld", f"c,{shift_count}")
                        shift_loop = self._new_label("SHR")
                        end_label = self._new_label("SHRE")
                        self._emit_label(shift_loop)
                        self._emit("dec", "c")
                        self._emit("jp", f"m,{end_label}")
                        self._emit("or", "a")
                        self._emit("ld", "a,h")
                        self._emit("rra")
                        self._emit("ld", "h,a")
                        self._emit("ld", "a,l")
                        self._emit("rra")
                        self._emit("ld", "l,a")
                        self._emit("jp", shift_loop)
                        self._emit_label(end_label)
                return DataType.ADDRESS

            # Variable shift - use loop
            arg_type = self._gen_expr(args[0])
            if arg_type == DataType.BYTE:
                # BYTE value is in A, move to HL
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")
            self._emit("push", "hl")
            count_type = self._gen_expr(args[1])
            if self.target == Target.Z80:
                # Z80: use B register with DJNZ, check for zero first
                if count_type == DataType.BYTE:
                    self._emit("ld", "b,a")
                else:
                    self._emit("ld", "b,l")
                self._emit("pop", "hl")
                end_label = self._new_label("SHRE")
                self._emit("inc", "b")  # Pre-increment so DJNZ works with count=0
                self._emit("dec", "b")  # Test for zero
                self._emit("jp", f"z,{end_label}")
                shift_loop = self._new_label("SHR")
                self._emit_label(shift_loop)
                self._emit("srl", "h")
                self._emit("rr", "l")
                self._emit("djnz", shift_loop)
                self._emit_label(end_label)
            else:
                # 8080: use C counter
                if count_type == DataType.BYTE:
                    self._emit("ld", "c,a")  # Count in C (from A for byte)
                else:
                    self._emit("ld", "c,l")  # Count in C (from L for address)
                self._emit("pop", "hl")
                shift_loop = self._new_label("SHR")
                end_label = self._new_label("SHRE")
                self._emit_label(shift_loop)
                self._emit("dec", "c")
                self._emit("jp", f"m,{end_label}")
                self._emit("or", "a")  # Clear carry
                self._emit("ld", "a,h")
                self._emit("rra")
                self._emit("ld", "h,a")
                self._emit("ld", "a,l")
                self._emit("rra")
                self._emit("ld", "l,a")
                self._emit("jp", shift_loop)
                self._emit_label(end_label)
            return DataType.ADDRESS

        if name == "ROL":
            arg_type = self._gen_expr(args[0])
            if arg_type == DataType.BYTE:
                # BYTE value is in A, move to HL
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")
            self._emit("push", "hl")
            count_type = self._gen_expr(args[1])
            if count_type == DataType.BYTE:
                self._emit("ld", "c,a")  # Count in C (from A for byte)
            else:
                self._emit("ld", "c,l")  # Count in C (from L for address)
            self._emit("pop", "hl")
            self._emit("ld", "a,l")
            shift_loop = self._new_label("ROL")
            end_label = self._new_label("ROLE")
            self._emit_label(shift_loop)
            self._emit("dec", "c")
            self._emit("jp", f"m,{end_label}")
            self._emit("rlca")
            self._emit("jp", shift_loop)
            self._emit_label(end_label)
            self._emit("ld", "l,a")
            self._emit("ld", "h,0")
            return DataType.BYTE

        if name == "ROR":
            arg_type = self._gen_expr(args[0])
            if arg_type == DataType.BYTE:
                # BYTE value is in A, move to HL
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")
            self._emit("push", "hl")
            count_type = self._gen_expr(args[1])
            if count_type == DataType.BYTE:
                self._emit("ld", "c,a")  # Count in C (from A for byte)
            else:
                self._emit("ld", "c,l")  # Count in C (from L for address)
            self._emit("pop", "hl")
            self._emit("ld", "a,l")
            shift_loop = self._new_label("ROR")
            end_label = self._new_label("RORE")
            self._emit_label(shift_loop)
            self._emit("dec", "c")
            self._emit("jp", f"m,{end_label}")
            self._emit("rrca")
            self._emit("jp", shift_loop)
            self._emit_label(end_label)
            self._emit("ld", "l,a")
            self._emit("ld", "h,0")
            return DataType.BYTE

        if name == "LENGTH":
            # Returns array dimension
            if args and isinstance(args[0], Identifier):
                sym = self.symbols.lookup(args[0].name)
                if sym and sym.dimension:
                    self._emit("ld", f"hl,{sym.dimension}")
                    return DataType.ADDRESS
            self._emit("ld", "hl,0")
            return DataType.ADDRESS

        if name == "LAST":
            # Returns array dimension - 1
            if args and isinstance(args[0], Identifier):
                sym = self.symbols.lookup(args[0].name)
                if sym and sym.dimension:
                    self._emit("ld", f"hl,{sym.dimension - 1}")
                    return DataType.ADDRESS
            self._emit("ld", "hl,0")
            return DataType.ADDRESS

        if name == "SIZE":
            # Returns size in bytes
            if args and isinstance(args[0], Identifier):
                sym = self.symbols.lookup(args[0].name)
                if sym:
                    self._emit("ld", f"hl,{sym.size}")
                    return DataType.ADDRESS
            self._emit("ld", "hl,0")
            return DataType.ADDRESS

        if name == "MEMORY":
            # MEMORY(addr) - direct memory access as byte array starting at __END__
            # Generate __END__ + offset into HL
            self.needs_end_symbol = True
            if isinstance(args[0], NumberLiteral) and args[0].value == 0:
                # MEMORY(0) - just use __END__ directly
                self._emit("ld", "hl,__END__")
            else:
                # MEMORY(n) - compute __END__ + n
                self._gen_expr(args[0])  # HL = offset
                self._emit("ld", "de,__END__")
                self._emit("add", "hl,de")  # HL = __END__ + offset
            # Load byte from (HL)
            self._emit("ld", "a,(hl)")
            return DataType.BYTE

        if name == "MOVE":
            # MOVE(count, source, dest) - use Z80 LDIR for efficiency
            # Generate inline code to avoid calling convention issues
            # LDIR: (DE) <- (HL), HL++, DE++, BC-- until BC=0
            # PL/M MOVE order: count, source, dest

            # Check if count is a constant
            count_const = None
            if isinstance(args[0], NumberLiteral):
                count_const = args[0].value

            if count_const is not None:
                # Optimized path for constant count
                if count_const == 0:
                    # Zero count - no-op
                    return None
                # Generate: dest -> DE, source -> HL, bc=count, ldir
                # Must check if source expression clobbers DE
                source_preserves_de = self._expr_preserves_de(args[1])
                if source_preserves_de:
                    # Source is simple - can load dest to DE first
                    self._gen_expr(args[2])  # dest -> HL
                    self._emit("ex", "de,hl")  # dest -> DE
                    self._gen_expr(args[1])  # source -> HL (preserves DE)
                else:
                    # Source is complex and may clobber DE - must save dest
                    self._gen_expr(args[2])  # dest -> HL
                    self._emit("push", "hl")  # save dest
                    self._gen_expr(args[1])  # source -> HL (may clobber DE)
                    self._emit("pop", "de")  # dest -> DE
                self._emit("ld", f"bc,{self._format_number(count_const)}")
                self._emit("ldir")
            else:
                # Variable count - need to evaluate and check for zero
                # count -> BC, source -> HL, dest -> DE
                self._gen_expr(args[2])  # dest -> HL
                self._emit("push", "hl")
                self._gen_expr(args[1])  # source -> HL
                self._emit("push", "hl")
                self._gen_expr(args[0])  # count -> HL
                # Move count from HL to BC
                self._emit("ld", "b,h")
                self._emit("ld", "c,l")
                self._emit("pop", "hl")  # source -> HL
                self._emit("pop", "de")  # dest -> DE
                # Check if count is 0
                self._emit("ld", "a,b")
                self._emit("or", "c")
                skip_label = self._new_label("MOVEX")
                self._emit("jr", f"z,{skip_label}")
                self._emit("ldir")
                self._emit_label(skip_label)
            return None

        if name == "TIME":
            # Delay loop
            self._gen_expr(args[0])
            loop_label = self._new_label("TIME")
            self._emit_label(loop_label)
            self._emit("dec", "hl")
            self._emit("ld", "a,h")
            self._emit("or", "l")
            self._emit("jp", f"nz,{loop_label}")
            return None

        if name == "CARRY":
            # Return carry flag value
            self._emit("ld", "a,0")
            self._emit("rla")  # Rotate carry into A
            self._emit("ld", "l,a")
            self._emit("ld", "h,0")
            return DataType.BYTE

        if name == "ZERO":
            # Return zero flag value
            true_label = self._new_label("ZF")
            end_label = self._new_label("ZFE")
            self._emit("jp", f"z,{true_label}")
            self._emit("ld", "hl,0")
            self._emit("jp", end_label)
            self._emit_label(true_label)
            self._emit("ld", "hl,0ffh")
            self._emit_label(end_label)
            return DataType.BYTE

        if name == "SIGN":
            # Return sign flag value
            true_label = self._new_label("SF")
            end_label = self._new_label("SFE")
            self._emit("jp", f"m,{true_label}")
            self._emit("ld", "hl,0")
            self._emit("jp", end_label)
            self._emit_label(true_label)
            self._emit("ld", "hl,0ffh")
            self._emit_label(end_label)
            return DataType.BYTE

        if name == "PARITY":
            # Return parity flag value
            true_label = self._new_label("PF")
            end_label = self._new_label("PFE")
            self._emit("jp", f"pe,{true_label}")
            self._emit("ld", "hl,0")
            self._emit("jp", end_label)
            self._emit_label(true_label)
            self._emit("ld", "hl,0ffh")
            self._emit_label(end_label)
            return DataType.BYTE

        if name == "DEC":
            # DEC is the Decimal Adjust procedure for BCD arithmetic.
            # It performs DAA (Decimal Adjust Accumulator) on the result
            # of an addition to convert the binary result to BCD.
            # Usage: R = DEC(A + B) where A and B are BCD values.
            arg_type = self._gen_expr(args[0])
            if arg_type == DataType.ADDRESS:
                self._emit("ld", "a,l")  # Get low byte from L
            # Apply DAA to convert binary addition result to BCD
            self._emit("daa")
            return DataType.BYTE

        if name == "SCL":
            # Shift through carry left
            arg_type = self._gen_expr(args[0])
            if arg_type == DataType.BYTE:
                # BYTE value is in A, move to HL
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")
            self._emit("push", "hl")
            count_type = self._gen_expr(args[1])
            if count_type == DataType.BYTE:
                self._emit("ld", "c,a")  # Count in C (from A for byte)
            else:
                self._emit("ld", "c,l")  # Count in C (from L for address)
            self._emit("pop", "hl")
            self._emit("ld", "a,l")
            shift_loop = self._new_label("SCL")
            end_label = self._new_label("SCLE")
            self._emit_label(shift_loop)
            self._emit("dec", "c")
            self._emit("jp", f"m,{end_label}")
            self._emit("rla")  # Rotate through carry
            self._emit("jp", shift_loop)
            self._emit_label(end_label)
            self._emit("ld", "l,a")
            self._emit("ld", "h,0")
            return DataType.BYTE

        if name == "SCR":
            # Shift through carry right
            arg_type = self._gen_expr(args[0])
            if arg_type == DataType.BYTE:
                # BYTE value is in A, move to HL
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")
            self._emit("push", "hl")
            count_type = self._gen_expr(args[1])
            if count_type == DataType.BYTE:
                self._emit("ld", "c,a")  # Count in C (from A for byte)
            else:
                self._emit("ld", "c,l")  # Count in C (from L for address)
            self._emit("pop", "hl")
            self._emit("ld", "a,l")
            shift_loop = self._new_label("SCR")
            end_label = self._new_label("SCRE")
            self._emit_label(shift_loop)
            self._emit("dec", "c")
            self._emit("jp", f"m,{end_label}")
            self._emit("rra")  # Rotate through carry
            self._emit("jp", shift_loop)
            self._emit_label(end_label)
            self._emit("ld", "l,a")
            self._emit("ld", "h,0")
            return DataType.BYTE

        # Not a built-in we handle inline
        return None

    def _gen_location(self, expr: LocationExpr) -> DataType:
        """Generate code to load address of expression."""
        operand = expr.operand
        if isinstance(operand, Identifier):
            name = operand.name

            # Check for built-in MEMORY - its address is the end of program data
            # In PL/M-80, .MEMORY gives the first free byte after all variables
            if name.upper() == "MEMORY":
                self.needs_end_symbol = True
                self._emit("ld", "hl,__END__")
                return DataType.ADDRESS

            # Check for LITERALLY macro - expand recursively
            if name in self.literal_macros:
                macro_val = self.literal_macros[name]
                try:
                    # Numeric literal - load as immediate address
                    val = self._parse_plm_number(macro_val)
                    self._emit("ld", f"hl,{self._format_number(val)}")
                    return DataType.ADDRESS
                except ValueError:
                    # Non-numeric literal - recursively process
                    return self._gen_location(LocationExpr(operand=Identifier(name=macro_val)))
            # Mangle name if needed
            sym = self.symbols.lookup(name)

            # Handle reentrant procedure parameters/locals (IX-relative)
            if sym and sym.stack_offset is not None:
                # Compute address: HL = IX + offset
                # push IX; POP HL; ld de,offset; add hl,DE
                self._emit("push", "ix")
                self._emit("pop", "hl")
                if sym.stack_offset != 0:
                    self._emit("ld", f"de,{sym.stack_offset}")
                    self._emit("add", "hl,de")
            elif sym and sym.based_on:
                # BASED variable - its "address" is the value of the base pointer
                # e.g., if fcbv BASED fcbp, then .fcbv returns the value of fcbp
                base_sym = self.symbols.lookup(sym.based_on)
                base_asm_name = base_sym.asm_name if base_sym and base_sym.asm_name else self._mangle_name(sym.based_on)
                self._emit("ld", f"hl,({base_asm_name})")
            else:
                asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(name)
                self._emit("ld", f"hl,{asm_name}")
        elif isinstance(operand, SubscriptExpr):
            self._gen_subscript_addr(operand)
        elif isinstance(operand, MemberExpr):
            self._gen_member_addr(operand)
        elif isinstance(operand, StringLiteral):
            # .('string') - address of inline string
            label = self._new_string_label()
            self.string_literals.append((label, operand.value))
            self._emit("ld", f"hl,{label}")
        elif isinstance(operand, ConstListExpr):
            # .(const, const, ...) - address of inline data
            label = self._new_label("DATA")
            self.data_segment.append(AsmLine(label=label))
            for val in operand.values:
                if isinstance(val, NumberLiteral):
                    self.data_segment.append(
                        AsmLine(opcode="db", operands=self._format_number(val.value))
                    )
                elif isinstance(val, StringLiteral):
                    self.data_segment.append(
                        AsmLine(opcode="db", operands=self._escape_string(val.value))
                    )
            self._emit("ld", f"hl,{label}")
        elif isinstance(operand, CallExpr):
            # Check if this is actually an array subscript (parser creates CallExpr for arr(idx))
            if isinstance(operand.callee, Identifier) and len(operand.args) == 1:
                sym = self.symbols.lookup(operand.callee.name)
                if sym and sym.kind != SymbolKind.PROCEDURE:
                    # It's an array access, not a function call - treat as subscript
                    subscript = SubscriptExpr(operand.callee, operand.args[0])
                    self._gen_subscript_addr(subscript)
                    return DataType.ADDRESS
            elif isinstance(operand.callee, MemberExpr) and len(operand.args) == 1:
                # Subscripting a structure member array: struct.member(idx)
                # This is NOT a function call - it's array access on a member
                # Generate: member_addr + idx * elem_size
                member_expr = operand.callee
                idx_expr = operand.args[0]

                # Generate the member address
                self._gen_member_addr(member_expr)

                # Get the member element type (BYTE arrays have size 1, ADDRESS arrays size 2)
                _, member_type = self._get_member_info(member_expr)
                elem_size = 2 if member_type == DataType.ADDRESS else 1

                # Add the subscript offset
                if isinstance(idx_expr, NumberLiteral):
                    offset = idx_expr.value * elem_size
                    self._emit_add_hl_const(offset)
                else:
                    # Variable index - use allocator for DE
                    self.regs.need_reg('de', 'member_subscript_addr', self._emit)
                    self._emit("ex", "de,hl")  # DE = member addr
                    idx_type = self._gen_expr(idx_expr)
                    if idx_type == DataType.BYTE:
                        self._emit("ld", "l,a")
                        self._emit("ld", "h,0")
                    if elem_size == 2:
                        self._emit("add", "hl,hl")  # HL *= 2
                    self._emit("add", "hl,de")  # HL = addr + offset
                    self.regs.release_reg('de', self._emit)
                return DataType.ADDRESS
            # Otherwise evaluate as expression
            self._gen_expr(operand)
        else:
            # Just evaluate the expression
            self._gen_expr(operand)
        return DataType.ADDRESS


def generate(module: Module, target: Target = Target.Z80) -> str:
    """Convenience function to generate code from a module."""
    gen = CodeGenerator(target)
    return gen.generate(module)
