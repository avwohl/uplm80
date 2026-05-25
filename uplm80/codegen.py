"""
Code Generator for PL/M-80.

Generates Z80 assembly code from the optimized AST.
Outputs MACRO-80 compatible .MAC files.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Iterator

from .ast_nodes import DataType
from . import _plm_parser as P
from ._plm_parser import K
from .ast_view import (
    module_shape,
    proc_attrs,
    proc_name,
    proc_param_names,
    proc_return_type,
    proc_body_items,
    proc_local_decls_stmts,
    block_items_split,
    proc_end_label,
    iter_declare_items,
    decl_item_names,
    array_size_value,
    decl_attrs,
    decl_item_type as _view_decl_item_type,
    decl_item_struct_members,
    decl_item_based,
    struct_member_names,
    struct_member_type,
    struct_member_dim,
    literally_value,
    binop_kind,
    unop_kind,
    ident_text,
    parse_plm_number,
    number_value,
    string_value,
    string_bytes,
    unwrap_paren,
    DataType as ViewDataType,
    BinaryOpKind,
    UnaryOpKind,
)
from . import ast_nodes as _ast_nodes
from .symbols import SymbolTable, Symbol, SymbolKind
from .errors import CodeGenError
from .runtime import get_runtime_library


# Map ast_view's DataType (used by typed AST helpers) to the legacy
# ast_nodes.DataType still used by the symbol table and unmigrated
# codegen paths. Same enum names, different identity.
_VIEW_DT_TO_LEGACY = {
    ViewDataType.BYTE: DataType.BYTE,
    ViewDataType.ADDRESS: DataType.ADDRESS,
    ViewDataType.LABEL: DataType.LABEL,
    ViewDataType.PROCEDURE: DataType.PROCEDURE,
}

# Equality-operator TokenKinds — the comparison ops that allow
# negative-byte truncation (`BYTE = -1`, `BYTE <> -1`) in the
# BYTE-range diagnostic below.
_BYTE_EQ_KINDS = frozenset({K.EQ, K.NE})


def _legacy_dt(dt):
    """Convert an ast_view DataType (or None) to the legacy enum."""
    if dt is None:
        return None
    return _VIEW_DT_TO_LEGACY[dt]


class _SynthToken:
    """Minimal stand-in for a :class:`uplox.Token` carrying ``.text`` / ``.name`` / ``.kind``.

    The typed AST nodes constructed by codegen during macro expansion
    don't ever flow back to the parser, so the source-location and
    file-id fields on a real :class:`Token` aren't needed; we only need
    the bits :func:`ast_view.ident_text` and friends read off the token
    (its ``.text`` plus a ``.kind`` for any K-based dispatch downstream).
    Define this with ``__slots__`` so the synthetic nodes have no
    per-instance dict overhead — codegen creates a fresh one for every
    LITERALLY substitution.
    """

    __slots__ = ("text", "name", "kind")

    def __init__(self, text: str, name: str = "IDENT", kind: int = K.IDENT) -> None:
        self.text = text
        self.name = name
        self.kind = kind


def _make_ident(name: str) -> "P.Identifier":
    """Build a typed :class:`P.Identifier` carrying the given name.

    Used by codegen paths that recurse on the expansion of a LITERALLY
    macro — the macro's replacement text is a string, not a token, so
    we wrap it in a synthetic token before re-entering the dispatch.
    """
    return P.Identifier(name=_SynthToken(name))


def _make_location(operand) -> "P.LocationOf":
    """Build a typed :class:`P.LocationOf` wrapping the given operand.

    Mirrors :func:`_make_ident` for the ``.expr`` case so codegen can
    funnel "store via complex lvalue" / "load address of complex
    expression" paths through the typed dispatch.
    """
    return P.LocationOf(operand=operand)


def _decl_item_has_data(item):
    """True if a typed ``DeclItem`` carries a DATA initializer.

    DATA initializers live on the ``DeclTailData`` / ``DeclTailTypeData``
    / ``DeclTailStructureData`` variants of ``DeclItem.tail``.
    """
    tail = getattr(item, "tail", None)
    if tail is None:
        return False
    return isinstance(
        tail,
        (P.DeclTailData, P.DeclTailTypeData, P.DeclTailStructureData),
    )


def _decl_item_type(item):
    """Extract (legacy DataType, dimension) from a typed ``DeclItem``.

    Thin shim over :func:`ast_view.decl_item_type` that converts the
    view-side enum to the legacy :class:`ast_nodes.DataType` still
    consumed by the symbol table. ``dimension`` is ``None`` for
    scalars, an int for fixed-size arrays, or ``-1`` for ``(*)``.
    """
    dt, dim = _view_decl_item_type(item)
    return _legacy_dt(dt), dim


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

    # Reserved assembler names that conflict with Z80 registers
    RESERVED_NAMES = {'A', 'B', 'C', 'D', 'E', 'H', 'L', 'M', 'SP', 'PSW',
                      'AF', 'BC', 'DE', 'HL', 'IX', 'IY', 'I', 'R'}

    def __init__(self, mode: Mode = Mode.CPM, warn_trivial_if: bool = True, reg_debug: bool = False) -> None:
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
        # ``current_proc_decl`` now holds a typed :class:`P.ProcDecl`; its
        # flattened attribute view (and the legacy-shaped return type) is
        # cached on the side so unmigrated paths can read them without
        # walking the typed signature each time.
        self.current_proc_decl: "P.ProcDecl | None" = None
        self.current_proc_attrs = None  # type: ignore[var-annotated]
        self.current_proc_return_type: DataType | None = None
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
        self.pending_stmts: list = []
        # For tracking embedded assignment target for return optimization
        self.embedded_assign_target: str | None = None  # Variable name of last embedded assignment
        # Current IF statement being processed (for embedded assign optimization)
        self.current_if_stmt = None  # P.IfStmt | P.IfStmtElse | None
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

    def _get_const_byte_value(self, expr) -> int | None:
        """Extract a constant byte value from an expression if possible.

        Returns the constant value (0-255) or None if not a constant.
        Handles NumberLiteral, StringLiteral (single char), and LITERALLY macros.
        """
        expr = unwrap_paren(expr)
        if isinstance(expr, P.NumberLiteral):
            val = number_value(expr)
            if val <= 255:
                return val
        elif isinstance(expr, P.StringLiteral):
            s = string_value(expr)
            if len(s) == 1:
                return ord(s[0])
        elif isinstance(expr, P.Identifier):
            name = ident_text(expr.name)
            if name in self.literal_macros:
                try:
                    val = self._parse_plm_number(self.literal_macros[name])
                    if val <= 255:
                        return val
                except ValueError:
                    pass
        return None

    def _try_eval_const(self, expr) -> int | None:
        """Try to evaluate an expression as a compile-time constant.

        Returns the integer value or None if not a constant.
        Handles NumberLiteral, StringLiteral, LITERALLY macros, and UnaryOp(NEG).
        Values are returned as-is (may be negative or > 255).
        """
        expr = unwrap_paren(expr)
        if isinstance(expr, P.NumberLiteral):
            return number_value(expr)
        elif isinstance(expr, P.StringLiteral):
            s = string_value(expr)
            if len(s) == 1:
                return ord(s[0])
            return None
        elif isinstance(expr, P.Identifier):
            name = ident_text(expr.name)
            if name in self.literal_macros:
                try:
                    return self._parse_plm_number(self.literal_macros[name])
                except ValueError:
                    pass
        elif isinstance(expr, P.UnaryOp):
            kind = unop_kind(expr)
            if kind == UnaryOpKind.NEG:
                operand_val = self._try_eval_const(expr.operand)
                if operand_val is not None:
                    return -operand_val
            elif kind == UnaryOpKind.NOT:
                operand_val = self._try_eval_const(expr.operand)
                if operand_val is not None:
                    return (~operand_val) & 0xFFFF
        elif isinstance(expr, P.BinaryOp):
            left_val = self._try_eval_const(expr.left)
            right_val = self._try_eval_const(expr.right)
            if left_val is not None and right_val is not None:
                op = binop_kind(expr)
                if op == BinaryOpKind.ADD:
                    return (left_val + right_val) & 0xFFFF
                elif op == BinaryOpKind.SUB:
                    return (left_val - right_val) & 0xFFFF
                elif op == BinaryOpKind.AND:
                    return left_val & right_val
                elif op == BinaryOpKind.OR:
                    return left_val | right_val
                elif op == BinaryOpKind.XOR:
                    return left_val ^ right_val
        return None

    def _check_impossible_comparison(self, left, right, op) -> None:
        """Check for comparisons that can never or always be true and raise an error.

        ``op`` is either the legacy :class:`BinaryOp` enum or the
        typed-AST :class:`ast_view.BinaryOpKind`; both flavours share
        member names (``EQ``/``NE``/``LT``/``LE``/``GT``/``GE``) so
        we dispatch on ``op.name`` to stay agnostic during the AST
        migration. For BYTE compared to constant outside 0-255:

        - For ``=`` and ``<>``, allow truncation only for "negative byte"
          values (``0xFF00`` - ``0xFFFF``, i.e. -256 to -1).
        - For ``<``, ``>``, ``<=``, ``>=``, the comparison is always
          true / false so we error.
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

                op_kind = op.kind

                # For = and <>, allow truncation only for "negative byte" values (high byte is 0xFF)
                # This handles BYTE <> -1 (0xFFFF -> 0xFF) but catches BYTE <> 0x123
                if op_kind in _BYTE_EQ_KINDS:
                    if (unsigned_val & 0xFF00) == 0xFF00:
                        return  # Valid: -256 to -1 range, will truncate to byte
                    # Otherwise, error - constant like 256 or 0x123 shouldn't be compared to BYTE
                    byte_val = unsigned_val & 0xFF
                    if op_kind is K.EQ:
                        msg = f"comparison BYTE = {unsigned_val} is always false (BYTE can only hold 0-255; truncating to {byte_val} would change semantics)"
                    else:
                        msg = f"comparison BYTE <> {unsigned_val} is always true (BYTE can only hold 0-255; truncating to {byte_val} would change semantics)"
                    raise CodeGenError(msg, loc)

                # For ordering comparisons, values outside 0-255 give always true/false
                if op_kind is K.LT:
                    msg = f"comparison BYTE < {right_val} is always true (BYTE can only hold 0-255)"
                elif op_kind is K.LE:
                    msg = f"comparison BYTE <= {right_val} is always true (BYTE can only hold 0-255)"
                elif op_kind is K.GT:
                    msg = f"comparison BYTE > {right_val} is always false (BYTE can only hold 0-255)"
                elif op_kind is K.GE:
                    msg = f"comparison BYTE >= {right_val} is always false (BYTE can only hold 0-255)"
                else:
                    return  # Unknown comparison operator

                raise CodeGenError(msg, loc)

    def _check_trivial_condition(self, condition, context: str = "condition") -> None:
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

    def _warn_trivial_if(self, condition) -> None:
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

    def _var_used_in_expr(self, var_name: str, expr) -> bool:
        """Check if variable is referenced in expression."""
        expr = unwrap_paren(expr)
        if isinstance(expr, P.Identifier):
            return ident_text(expr.name) == var_name
        elif isinstance(expr, (P.NumberLiteral, P.StringLiteral)):
            return False
        elif isinstance(expr, P.BinaryOp):
            return (
                self._var_used_in_expr(var_name, expr.left)
                or self._var_used_in_expr(var_name, expr.right)
            )
        elif isinstance(expr, P.UnaryOp):
            return self._var_used_in_expr(var_name, expr.operand)
        elif isinstance(expr, P.Call):
            for arg in expr.args:
                if self._var_used_in_expr(var_name, arg):
                    return True
            return self._var_used_in_expr(var_name, expr.callee)
        elif isinstance(expr, P.CallNoArgs):
            return self._var_used_in_expr(var_name, expr.callee)
        elif isinstance(expr, P.MemberAccess):
            return self._var_used_in_expr(var_name, expr.base)
        elif isinstance(expr, P.LocationOf):
            return self._var_used_in_expr(var_name, expr.operand)
        elif isinstance(expr, P.LocationOfList):
            for v in expr.values or []:
                if self._var_used_in_expr(var_name, v):
                    return True
            return False
        elif isinstance(expr, P.LocationOfString):
            return False
        elif isinstance(expr, P.EmbeddedAssign):
            return (
                self._var_used_in_expr(var_name, expr.target)
                or self._var_used_in_expr(var_name, expr.value)
            )
        return False

    def _var_used_in_stmt(self, var_name: str, stmt) -> bool:
        """Check if variable is referenced in a typed statement node."""
        if isinstance(stmt, P.AssignStmt):
            if self._var_used_in_expr(var_name, stmt.value):
                return True
            for target in stmt.targets:
                t = unwrap_paren(target)
                # Subscript-as-Call: var in index counts as use.
                if isinstance(t, P.Call):
                    for arg in t.args:
                        if self._var_used_in_expr(var_name, arg):
                            return True
            return False
        elif isinstance(stmt, P.CallStmt):
            inner = stmt.callee
            if isinstance(inner, P.Call):
                if self._var_used_in_expr(var_name, inner.callee):
                    return True
                for arg in inner.args:
                    if self._var_used_in_expr(var_name, arg):
                        return True
                return False
            if isinstance(inner, P.CallNoArgs):
                return self._var_used_in_expr(var_name, inner.callee)
            return self._var_used_in_expr(var_name, inner)
        elif isinstance(stmt, P.ReturnStmtValue):
            return self._var_used_in_expr(var_name, stmt.value)
        elif isinstance(stmt, P.ReturnStmt):
            return False
        elif isinstance(stmt, (P.IfStmt, P.IfStmtElse)):
            if self._var_used_in_expr(var_name, stmt.condition):
                return True
            if self._var_used_in_stmt(var_name, stmt.then_stmt):
                return True
            if isinstance(stmt, P.IfStmtElse) and self._var_used_in_stmt(
                var_name, stmt.else_stmt
            ):
                return True
            return False
        elif isinstance(stmt, P.DoBlock):
            _, body_stmts = block_items_split(stmt.items)
            for s in body_stmts:
                if self._var_used_in_stmt(var_name, s):
                    return True
            return False
        elif isinstance(stmt, P.DoWhileBlock):
            if self._var_used_in_expr(var_name, stmt.condition):
                return True
            _, body_stmts = block_items_split(stmt.items)
            for s in body_stmts:
                if self._var_used_in_stmt(var_name, s):
                    return True
            return False
        elif isinstance(stmt, (P.DoIterBlock, P.DoIterByBlock)):
            # Don't recurse into nested DO-ITER as inner loop var shadows outer
            if self._var_used_in_expr(var_name, stmt.start):
                return True
            if self._var_used_in_expr(var_name, stmt.bound):
                return True
            if isinstance(stmt, P.DoIterByBlock) and self._var_used_in_expr(
                var_name, stmt.step
            ):
                return True
            _, body_stmts = block_items_split(stmt.items)
            for s in body_stmts:
                if self._var_used_in_stmt(var_name, s):
                    return True
            return False
        elif isinstance(stmt, P.DoCaseBlock):
            if self._var_used_in_expr(var_name, stmt.selector):
                return True
            for s in stmt.items or []:
                if self._var_used_in_stmt(var_name, s):
                    return True
            return False
        elif isinstance(stmt, P.LabeledStmt):
            return self._var_used_in_stmt(var_name, stmt.stmt)
        return False

    def _index_used_in_body(self, index_var, stmts) -> bool:
        """Check if loop index variable is used in loop body."""
        if isinstance(index_var, P.Identifier):
            var_name = ident_text(index_var.name)
            for stmt in stmts:
                if self._var_used_in_stmt(var_name, stmt):
                    return True
        return False

    def _stmts_contain_goto(self, stmts) -> bool:
        """Recursively check whether any statement in the tree is a GotoStmt.

        Used to disable loop optimizations (DJNZ) that push state onto the
        stack across iterations. A GOTO escaping such a loop body would
        leave that pushed state stranded — see test_goto_loops.
        """
        for stmt in stmts:
            if self._stmt_contains_goto(stmt):
                return True
        return False

    def _stmt_contains_goto(self, stmt) -> bool:
        if isinstance(stmt, P.GotoStmt):
            return True
        if isinstance(stmt, P.LabeledStmt):
            return self._stmt_contains_goto(stmt.stmt)
        if isinstance(stmt, (P.IfStmt, P.IfStmtElse)):
            if self._stmt_contains_goto(stmt.then_stmt):
                return True
            if isinstance(stmt, P.IfStmtElse) and self._stmt_contains_goto(
                stmt.else_stmt
            ):
                return True
            return False
        if isinstance(stmt, (P.DoBlock, P.DoWhileBlock, P.DoIterBlock, P.DoIterByBlock)):
            _, body_stmts = block_items_split(stmt.items)
            return self._stmts_contain_goto(body_stmts)
        if isinstance(stmt, P.DoCaseBlock):
            return self._stmts_contain_goto(stmt.items or [])
        return False

    # ========================================================================
    # Register Liveness Analysis
    # ========================================================================

    def _expr_clobbers_a(self, expr) -> bool:
        """Check if evaluating expression will clobber A register.

        Most expressions clobber A because they compute into A (for BYTE) or use A
        as a scratch register. Only certain simple operations preserve A.
        """
        expr = unwrap_paren(expr)
        if isinstance(expr, P.NumberLiteral):
            return False  # ld hl,const doesn't touch A

        if isinstance(expr, P.Identifier):
            sym = self._lookup_symbol(ident_text(expr.name))
            if sym and sym.data_type == DataType.BYTE:
                return True  # ld a,(addr) clobbers A
            return False  # ld hl,(addr) doesn't clobber A

        if isinstance(expr, P.BinaryOp):
            expr_type = self._get_expr_type(expr)
            if expr_type == DataType.ADDRESS:
                if binop_kind(expr) == BinaryOpKind.ADD:
                    return (
                        self._expr_clobbers_a(expr.left)
                        or self._expr_clobbers_a(expr.right)
                    )
            return True

        # Most other expressions clobber A
        return True

    def _stmt_clobbers_a(self, stmt) -> bool:
        """Check if a statement will clobber the A register."""
        if isinstance(stmt, P.NullStmt):
            return False

        if isinstance(stmt, P.LabeledStmt):
            return self._stmt_clobbers_a(stmt.stmt)

        if isinstance(stmt, P.AssignStmt):
            for target in stmt.targets:
                t = unwrap_paren(target)
                if isinstance(t, P.Identifier):
                    sym = self._lookup_symbol(ident_text(t.name))
                    if not sym or sym.data_type == DataType.BYTE:
                        return True
                else:
                    return True
            return self._expr_clobbers_a(stmt.value)

        if isinstance(stmt, P.CallStmt):
            return True

        if isinstance(stmt, P.ReturnStmtValue):
            return True
        if isinstance(stmt, P.ReturnStmt):
            return False

        if isinstance(stmt, P.GotoStmt):
            return False

        if isinstance(stmt, P.HaltStmt):
            return False

        if isinstance(stmt, (P.EnableStmt, P.DisableStmt)):
            return False

        if isinstance(stmt, (P.IfStmt, P.IfStmtElse)):
            condition = unwrap_paren(stmt.condition)
            if isinstance(condition, P.Identifier):
                return True
            if isinstance(condition, P.BinaryOp):
                op = binop_kind(condition)
                if op in (
                    BinaryOpKind.EQ, BinaryOpKind.NE,
                    BinaryOpKind.LT, BinaryOpKind.GT,
                    BinaryOpKind.LE, BinaryOpKind.GE,
                ):
                    left_type = self._get_expr_type(condition.left)
                    if left_type == DataType.BYTE:
                        if isinstance(unwrap_paren(condition.right), P.NumberLiteral):
                            then_clobbers = self._stmt_clobbers_a(stmt.then_stmt)
                            else_clobbers = (
                                isinstance(stmt, P.IfStmtElse)
                                and self._stmt_clobbers_a(stmt.else_stmt)
                            )
                            return then_clobbers or else_clobbers
            return True

        if isinstance(stmt, (P.DoBlock, P.DoWhileBlock, P.DoIterBlock,
                             P.DoIterByBlock, P.DoCaseBlock)):
            return True

        if isinstance(stmt, P.DeclareStmt):
            return False

        return True

    def _a_survives_stmts(self, stmts) -> bool:
        """Check if A register survives through a list of statements."""
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

    def _build_call_graph(self, module) -> None:
        """Build call graph by analyzing all procedure bodies."""
        self.call_graph = {}
        self.proc_storage: dict[str, list[tuple[str, int, DataType]]] = {}  # proc -> [(var_name, size, type)]

        shape = module_shape(module)

        # First pass: collect all procedure names
        all_procs: set[str] = set()
        self._collect_proc_names(shape.decls, None, all_procs)

        # Initialize call graph
        for proc in all_procs:
            self.call_graph[proc] = set()

        # Second pass: analyze calls in each procedure
        for decl in shape.decls:
            if isinstance(decl, P.ProcDecl):
                attrs = proc_attrs(decl)
                if not attrs.is_external:
                    self._analyze_proc_calls(decl, None)

    def _collect_proc_names(self, decls: list, parent_proc: str | None, all_procs: set[str]) -> None:
        """Recursively collect all procedure names."""
        for decl in decls:
            if isinstance(decl, P.ProcDecl):
                attrs = proc_attrs(decl)
                name = proc_name(decl)
                if parent_proc and not attrs.is_public and not attrs.is_external:
                    full_name = f"{parent_proc}${name}"
                else:
                    full_name = name
                all_procs.add(full_name)
                # Recurse into nested procedures: the body items are a
                # flat list mixing decls (incl. nested ProcDecls or
                # DeclareStmts wrapping them) and statements.
                body_items = proc_body_items(decl)
                nested: list = []
                for item in body_items:
                    if isinstance(item, P.ProcDecl):
                        nested.append(item)
                    elif isinstance(item, P.DeclareStmt):
                        for inner in iter_declare_items(item):
                            if isinstance(inner, P.ProcDecl):
                                nested.append(inner)
                if nested:
                    self._collect_proc_names(nested, full_name, all_procs)

    def _analyze_proc_calls(self, decl, parent_proc: str | None) -> None:
        """Analyze a procedure to find all calls it makes."""
        attrs = proc_attrs(decl)
        name = proc_name(decl)
        params = proc_param_names(decl)

        if parent_proc and not attrs.is_public and not attrs.is_external:
            full_name = f"{parent_proc}${name}"
        else:
            full_name = name

        if attrs.is_external:
            return

        # Split the procedure body into nested decls vs statements.
        body_items = proc_body_items(decl)
        nested_procs: list = []
        decl_items: list = []  # typed DeclItem / DeclItemBasedGroup / LiterallyDecl
        stmt_items: list = []
        for item in body_items:
            if isinstance(item, P.ProcDecl):
                nested_procs.append(item)
            elif isinstance(item, P.DeclareStmt):
                for inner in iter_declare_items(item):
                    if isinstance(inner, P.ProcDecl):
                        nested_procs.append(inner)
                    else:
                        decl_items.append(inner)
            else:
                stmt_items.append(item)

        # Find all calls in this procedure's body. The downstream
        # _find_calls_in_stmts walker is not yet migrated, so it will
        # fail at runtime on the typed nodes; that's expected for this
        # migration chunk.
        calls: set[str] = set()
        self._find_calls_in_stmts(stmt_items, full_name, calls)
        self.call_graph[full_name] = calls

        # Index DeclItems by declared name for parameter type lookup.
        decl_by_name: dict[str, tuple[DataType | None, int | None]] = {}
        for d in decl_items:
            if isinstance(d, P.DeclItem):
                d_type, d_dim = _decl_item_type(d)
                for n in decl_item_names(d):
                    decl_by_name[n] = (d_type, d_dim)

        # Collect storage requirements (params + locals)
        storage: list[tuple[str, int, DataType]] = []

        # Parameters
        for param in params:
            param_type = DataType.ADDRESS
            info = decl_by_name.get(param)
            if info is not None and info[0] is not None:
                param_type = info[0]
            size = 1 if param_type == DataType.BYTE else 2
            storage.append((param, size, param_type))

        # Local variables (non-parameter DeclItems)
        for d in decl_items:
            if not isinstance(d, P.DeclItem):
                continue
            d_type, d_dim = _decl_item_type(d)
            var_type = d_type or DataType.ADDRESS
            for n in decl_item_names(d):
                if n in params:
                    continue
                if d_dim and d_dim > 0:
                    elem_size = 1 if var_type == DataType.BYTE else 2
                    size = d_dim * elem_size
                else:
                    size = 1 if var_type == DataType.BYTE else 2
                storage.append((n, size, var_type))

        self.proc_storage[full_name] = storage

        # Recurse into nested procedures
        for nested in nested_procs:
            self._analyze_proc_calls(nested, full_name)

    def _find_calls_in_stmts(self, stmts, current_proc: str, calls: set[str]) -> None:
        """Find all procedure calls in a list of typed statements."""
        for stmt in stmts:
            self._find_calls_in_stmt(stmt, current_proc, calls)

    def _find_calls_in_stmt(self, stmt, current_proc: str, calls: set[str]) -> None:
        """Find procedure calls in a typed statement."""
        if isinstance(stmt, P.CallStmt):
            inner = stmt.callee
            if isinstance(inner, P.Call):
                callee_expr = inner.callee
                args = inner.args
            elif isinstance(inner, P.CallNoArgs):
                callee_expr = inner.callee
                args = []
            else:
                callee_expr = inner
                args = []
            if isinstance(callee_expr, P.Identifier):
                callee = self._resolve_proc_name(ident_text(callee_expr.name), current_proc)
                if callee:
                    calls.add(callee)
            else:
                self._find_calls_in_expr(callee_expr, current_proc, calls)
            for arg in args:
                self._find_calls_in_expr(arg, current_proc, calls)
        elif isinstance(stmt, P.AssignStmt):
            for target in stmt.targets:
                self._find_calls_in_expr(target, current_proc, calls)
            self._find_calls_in_expr(stmt.value, current_proc, calls)
        elif isinstance(stmt, P.ReturnStmtValue):
            self._find_calls_in_expr(stmt.value, current_proc, calls)
        elif isinstance(stmt, P.ReturnStmt):
            pass
        elif isinstance(stmt, (P.IfStmt, P.IfStmtElse)):
            self._find_calls_in_expr(stmt.condition, current_proc, calls)
            self._find_calls_in_stmt(stmt.then_stmt, current_proc, calls)
            if isinstance(stmt, P.IfStmtElse):
                self._find_calls_in_stmt(stmt.else_stmt, current_proc, calls)
        elif isinstance(stmt, P.DoBlock):
            _, body_stmts = block_items_split(stmt.items)
            self._find_calls_in_stmts(body_stmts, current_proc, calls)
        elif isinstance(stmt, P.DoWhileBlock):
            self._find_calls_in_expr(stmt.condition, current_proc, calls)
            _, body_stmts = block_items_split(stmt.items)
            self._find_calls_in_stmts(body_stmts, current_proc, calls)
        elif isinstance(stmt, (P.DoIterBlock, P.DoIterByBlock)):
            self._find_calls_in_expr(stmt.start, current_proc, calls)
            self._find_calls_in_expr(stmt.bound, current_proc, calls)
            if isinstance(stmt, P.DoIterByBlock):
                self._find_calls_in_expr(stmt.step, current_proc, calls)
            _, body_stmts = block_items_split(stmt.items)
            self._find_calls_in_stmts(body_stmts, current_proc, calls)
        elif isinstance(stmt, P.DoCaseBlock):
            self._find_calls_in_expr(stmt.selector, current_proc, calls)
            for s in stmt.items or []:
                self._find_calls_in_stmt(s, current_proc, calls)
        elif isinstance(stmt, P.LabeledStmt):
            self._find_calls_in_stmt(stmt.stmt, current_proc, calls)

    def _find_calls_in_expr(self, expr, current_proc: str, calls: set[str]) -> None:
        """Find procedure calls in a typed expression."""
        expr = unwrap_paren(expr)
        if isinstance(expr, P.Call):
            if isinstance(expr.callee, P.Identifier):
                callee = self._resolve_proc_name(ident_text(expr.callee.name), current_proc)
                if callee:
                    calls.add(callee)
            else:
                self._find_calls_in_expr(expr.callee, current_proc, calls)
            for arg in expr.args:
                self._find_calls_in_expr(arg, current_proc, calls)
        elif isinstance(expr, P.CallNoArgs):
            if isinstance(expr.callee, P.Identifier):
                callee = self._resolve_proc_name(ident_text(expr.callee.name), current_proc)
                if callee:
                    calls.add(callee)
            else:
                self._find_calls_in_expr(expr.callee, current_proc, calls)
        elif isinstance(expr, P.Identifier):
            # Bare identifier referring to a typed procedure is an implicit call.
            callee = self._resolve_proc_name(ident_text(expr.name), current_proc)
            if callee:
                calls.add(callee)
        elif isinstance(expr, P.BinaryOp):
            self._find_calls_in_expr(expr.left, current_proc, calls)
            self._find_calls_in_expr(expr.right, current_proc, calls)
        elif isinstance(expr, P.UnaryOp):
            self._find_calls_in_expr(expr.operand, current_proc, calls)
        elif isinstance(expr, P.MemberAccess):
            self._find_calls_in_expr(expr.base, current_proc, calls)
        elif isinstance(expr, P.LocationOf):
            self._find_calls_in_expr(expr.operand, current_proc, calls)
        elif isinstance(expr, P.LocationOfList):
            for v in expr.values or []:
                self._find_calls_in_expr(v, current_proc, calls)
        elif isinstance(expr, P.EmbeddedAssign):
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
            if isinstance(decl, P.ProcDecl):
                self._register_procedure(decl, parent_proc)

        # Also check statements for DeclareStmt containing procedures
        if stmts:
            for stmt in stmts:
                if isinstance(stmt, P.DeclareStmt):
                    for inner_decl in iter_declare_items(stmt):
                        if isinstance(inner_decl, P.ProcDecl):
                            self._register_procedure(inner_decl, parent_proc)

    def _register_procedure(self, decl, parent_proc: str | None) -> None:
        """Register a single procedure in the symbol table at module level."""
        attrs = proc_attrs(decl)
        name = proc_name(decl)
        params = proc_param_names(decl)
        return_type = _legacy_dt(proc_return_type(decl))

        # Compute the asm_name for this procedure
        if parent_proc and not attrs.is_public and not attrs.is_external:
            # Nested procedure - use scoped name
            proc_asm_name = f"@{parent_proc}${name}"
            full_proc_name = f"{parent_proc}${name}"
        else:
            proc_asm_name = name
            full_proc_name = name

        # Extract parameter types from the procedure body's DeclItems
        # (parameters get a DECLARE inside the body to set their type).
        body_items = proc_body_items(decl)
        decl_by_name: dict[str, DataType | None] = {}
        for item in body_items:
            if isinstance(item, P.DeclareStmt):
                for inner in iter_declare_items(item):
                    if isinstance(inner, P.DeclItem):
                        d_type, _ = _decl_item_type(inner)
                        for n in decl_item_names(inner):
                            decl_by_name[n] = d_type

        param_types = []
        for param in params:
            param_type = decl_by_name.get(param) or DataType.ADDRESS
            param_types.append(param_type)

        # For non-reentrant procedures with params, pass the LAST param in register
        # Byte params in A, ADDRESS params in HL - saves a store/load pair
        uses_reg_param = (len(params) >= 1 and
                         not attrs.is_reentrant and
                         not attrs.is_external)

        # Register in symbol table at the GLOBAL level so it's always accessible
        # This allows forward references from anywhere in the module
        # Use full_proc_name as the symbol name to avoid collisions between
        # nested procedures with the same local name (e.g., multiple ZN procs)
        sym = Symbol(
            name=full_proc_name,
            kind=SymbolKind.PROCEDURE,
            return_type=return_type,
            params=params,
            param_types=param_types,
            is_public=attrs.is_public,
            is_external=attrs.is_external,
            is_reentrant=attrs.is_reentrant,
            uses_reg_param=uses_reg_param,
            interrupt_num=attrs.interrupt_num,
            asm_name=proc_asm_name,
        )
        # Define at module (root) level - walk up to root scope
        root_scope = self.symbols.current_scope
        while root_scope.parent is not None:
            root_scope = root_scope.parent
        root_scope.define(sym)

        # Recursively collect nested procedures from the body items.
        # The new typed AST has a single flat body list mixing
        # declarations and statements, so split it for the legacy
        # _collect_procedures (decls, stmts) signature.
        nested_decls: list = []
        nested_stmts: list = []
        for item in body_items:
            if isinstance(item, P.ProcDecl):
                nested_decls.append(item)
            elif isinstance(item, P.DeclareStmt):
                # Hand DeclareStmts through as-is via the stmts slot so
                # the recursive call can rescan them for nested ProcDecls.
                nested_stmts.append(item)
            else:
                nested_stmts.append(item)
        if nested_decls or nested_stmts:
            self._collect_procedures(nested_decls, full_proc_name, nested_stmts)

    # ========================================================================
    # Main Entry Point
    # ========================================================================

    def generate(self, module) -> str:
        """Generate assembly code for a module."""
        self.output = []
        self.data_segment = []
        self.code_data_segment = []
        self.string_literals = []
        self.needs_runtime = set()
        self.needs_end_symbol = False
        self.literal_macros = {}

        shape = module_shape(module)

        # Header
        self._emit(comment=f"PL/M-80 Compiler Output - {shape.name}")
        self._emit(comment="Target: Z80")
        self._emit(comment="Generated by uplm80")
        self._emit()

        # Emit .z80 directive for assembler
        self._emit(".z80")
        self._emit()

        # Origin if specified
        if shape.origin is not None:
            self._emit("org", self._format_number(shape.origin))
            self._emit()

        # First pass: collect LITERALLY macros
        for decl in shape.decls:
            if isinstance(decl, P.LiterallyDecl):
                # LiterallyDecl.value is a Token whose .text retains the
                # surrounding quotes; strip them for the macro body.
                lit_name = ident_text(decl.name)
                lit_text = decl.value.text
                if lit_text.startswith("'") and lit_text.endswith("'"):
                    lit_text = lit_text[1:-1]
                self.literal_macros[lit_name] = lit_text

        # Separate procedures from other declarations
        procedures: list = []
        data_decls: list = []  # Module-level DATA declarations (typed DeclItems)
        other_decls: list = []
        entry_proc = None
        entry_proc_name: str | None = None

        for decl in shape.decls:
            if isinstance(decl, P.ProcDecl):
                attrs = proc_attrs(decl)
                pname = proc_name(decl)
                procedures.append(decl)
                # First non-external procedure with same name as module, or first procedure
                if not attrs.is_external and entry_proc is None:
                    if pname == shape.name or len(procedures) == 1:
                        entry_proc = decl
                        entry_proc_name = pname
            elif isinstance(decl, P.DeclItem) and _decl_item_has_data(decl):
                # Module-level DATA declaration - goes at start of code
                data_decls.append(decl)
            else:
                other_decls.append(decl)

        # Pass 1: Pre-register all procedures in symbol table for forward references
        # This allows procedures to call each other regardless of declaration order
        self._collect_procedures(shape.decls, parent_proc=None)

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
        if entry_proc and not shape.stmts:
            self._emit()
            self._emit(comment="Entry point")
            if self.mode == Mode.CPM:
                # CP/M: Set stack from BDOS, call main, return to OS
                self._emit("ld", "hl,(6)")
                self._emit("ld", "sp,hl")
                self._emit("call", entry_proc_name)
                self._emit("jp", "0")  # Warm boot to return to CP/M
            else:
                # BARE: Use locally-defined stack, jump to entry
                self._emit("ld", "sp,??STACK")
                self._emit("jp", entry_proc_name)

        # Generate code for module-level statements
        if shape.stmts:
            self._emit()
            self._emit(comment="Module initialization code")
            if self.mode == Mode.CPM:
                # CP/M: Set stack from BDOS address at 0006H
                self._emit("ld", "hl,(6)")
                self._emit("ld", "sp,hl")
            else:
                # BARE: Use locally-defined stack
                self._emit("ld", "sp,??STACK")
            for stmt in shape.stmts:
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
            print(f"[REG DEBUG] Statistics for {shape.name}:", file=sys.stderr)
            for key, val in sorted(self.regs.stats.items()):
                print(f"  {key}: {val}", file=sys.stderr)

        # Convert to string
        return "\n".join(str(line) for line in self.output)

    def generate_multi(self, modules: list) -> str:
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

        # Compute the shape view for each module once.
        shapes = [module_shape(m) for m in modules]

        # Header
        module_names = ', '.join(s.name for s in shapes)
        self._emit(comment=f"PL/M-80 Compiler Output - {module_names}")
        self._emit(comment="Target: Z80")
        self._emit(comment="Generated by uplm80")
        self._emit()

        # Emit .z80 directive for assembler
        self._emit(".z80")
        self._emit()

        # Use origin from first module if specified
        if shapes[0].origin is not None:
            self._emit("org", self._format_number(shapes[0].origin))
            self._emit()

        # Collect LITERALLY macros from all modules
        for shape in shapes:
            for decl in shape.decls:
                if isinstance(decl, P.LiterallyDecl):
                    lit_name = ident_text(decl.name)
                    lit_text = decl.value.text
                    if lit_text.startswith("'") and lit_text.endswith("'"):
                        lit_text = lit_text[1:-1]
                    self.literal_macros[lit_name] = lit_text

        # Pre-register all procedures from all modules for forward references
        for shape in shapes:
            self._collect_procedures(shape.decls, parent_proc=None)

        # Build unified call graph across all modules
        self._build_call_graph_multi(modules)
        self._compute_active_together()
        self._allocate_shared_storage()

        # First pass: collect all module info
        all_procedures: list = []   # list of (module, proc, proc_name)
        all_data_decls: list = []   # list of (module, DeclItem)
        all_other_decls: list = []  # list of (module, decl)
        entry_proc = None
        entry_proc_name: str | None = None
        first_module_with_stmts = None
        first_module_stmts: list = []

        for module, shape in zip(modules, shapes):
            if shape.stmts and first_module_with_stmts is None:
                first_module_with_stmts = module
                first_module_stmts = shape.stmts

            for decl in shape.decls:
                if isinstance(decl, P.ProcDecl):
                    attrs = proc_attrs(decl)
                    pname = proc_name(decl)
                    all_procedures.append((module, decl, pname, attrs))
                    if not attrs.is_external and entry_proc is None:
                        entry_proc = decl
                        entry_proc_name = pname
                elif isinstance(decl, P.DeclItem) and _decl_item_has_data(decl):
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
            for stmt in first_module_stmts:
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
                self._emit("call", entry_proc_name)
                self._emit("jp", "0")
            else:
                self._emit("ld", "sp,??STACK")
                self._emit("call", entry_proc_name)

        # Generate code for all procedures
        for module, proc, pname, attrs in all_procedures:
            if not attrs.is_external:
                self._emit()
                # Find the matching shape's name for the per-module comment.
                shape_name = next(
                    (s.name for m, s in zip(modules, shapes) if m is module),
                    "<input>",
                )
                self._emit(comment=f"Module: {shape_name}")
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

    def _build_call_graph_multi(self, modules: list) -> None:
        """Build call graph by analyzing all procedures across multiple modules."""
        self.call_graph = {}
        self.proc_storage: dict[str, list[tuple[str, int, DataType]]] = {}

        shapes = [module_shape(m) for m in modules]

        # First pass: collect all procedure names from all modules
        all_procs: set[str] = set()
        for shape in shapes:
            self._collect_proc_names(shape.decls, None, all_procs)

        # Initialize call graph
        for proc in all_procs:
            self.call_graph[proc] = set()

        # Second pass: analyze calls in each procedure across all modules
        for shape in shapes:
            for decl in shape.decls:
                if isinstance(decl, P.ProcDecl):
                    attrs = proc_attrs(decl)
                    if not attrs.is_external:
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

    def _gen_declaration(self, decl) -> None:
        """Generate code/storage for a typed declaration node.

        Dispatches over the uplox-generated typed AST kinds:
        :class:`P.DeclItem` (scalar/array/structure/based variable),
        :class:`P.DeclItemBasedGroup` (parenthesised BASED group),
        :class:`P.LiterallyDecl` (LITERALLY macro), or
        :class:`P.ProcDecl` (procedure).
        """
        if isinstance(decl, P.ProcDecl):
            self._gen_proc_decl(decl)
        elif isinstance(decl, P.LiterallyDecl):
            self._gen_literally_decl(decl)
        elif isinstance(decl, (P.DeclItem, P.DeclItemBasedGroup)):
            self._gen_var_decl(decl)

    def _gen_literally_decl(self, decl) -> None:
        """Register a LITERALLY macro in the symbol table and emit an
        EQU directive if the replacement text parses as a number."""
        name = ident_text(decl.name)
        value = literally_value(decl)
        self.symbols.define(
            Symbol(
                name=name,
                kind=SymbolKind.LITERAL,
                literal_value=value,
            )
        )
        self.literal_macros[name] = value
        # Emit EQU for numeric literals only.
        try:
            val = self._parse_plm_number(value)
            asm_name = self._mangle_name(name)
            self.data_segment.append(
                AsmLine(label=asm_name, opcode="EQU", operands=self._format_number(val))
            )
        except ValueError:
            pass  # Non-numeric replacement text, no EQU needed

    def _gen_var_decl(self, decl) -> None:
        """Generate storage for a typed variable declaration.

        ``decl`` may be a :class:`P.DeclItem` (one or many names sharing
        a tail) or a :class:`P.DeclItemBasedGroup` (a parenthesised list
        of based names, each becoming one symbol). A single ``DeclItem``
        with multiple names emits one storage row per name with each
        getting its own symbol entry.
        """
        if isinstance(decl, P.DeclItemBasedGroup):
            for bd in decl.based_decls or []:
                base_name = (
                    ident_text(bd.base.name)
                    if isinstance(bd.base, P.DottedIdent) else None
                )
                self._gen_one_var(
                    name=ident_text(bd.name),
                    based_on=base_name,
                    based_member=None,
                    item=decl,
                )
            return

        # P.DeclItem: one or more names sharing the same tail/clauses.
        based_on, based_member = decl_item_based(decl)
        for name in decl_item_names(decl):
            self._gen_one_var(
                name=name,
                based_on=based_on,
                based_member=based_member,
                item=decl,
            )

    def _gen_one_var(self, *, name: str, based_on, based_member, item) -> None:
        """Generate storage for a single name from a typed DeclItem.

        Split out so a ``(A, B, C) BYTE`` decl can emit one row per
        identifier while sharing tail/attribute extraction. The legacy
        ``VarDecl`` carried only one name per node, so this used to live
        inline in :meth:`_gen_var_decl`.
        """
        attrs = decl_attrs(item)
        data_type, dimension = _decl_item_type(item)
        members_nodes = decl_item_struct_members(item)
        is_public = attrs.is_public
        is_external = attrs.is_external
        at_location = attrs.at_location  # typed expression node | None
        data_values_nodes = attrs.data_values
        initial_values_nodes = attrs.initial_values

        # Build the legacy StructMember list the symbol table expects.
        struct_members = None
        if members_nodes is not None:
            struct_members = []
            for m in members_nodes:
                m_type = struct_member_type(m)
                m_dim = struct_member_dim(m)
                for sn in struct_member_names(m):
                    struct_members.append(
                        _ast_nodes.StructMember(
                            name=sn,
                            data_type=_legacy_dt(m_type),
                            dimension=m_dim,
                        )
                    )

        # Mangle name if it conflicts with register names
        base_name = self._mangle_name(name)
        asm_name: str | None = base_name  # Default, may be overridden below

        # Check if we're in a reentrant procedure - locals go on stack
        in_reentrant = (self.current_proc_attrs is not None and
                        self.current_proc_attrs.is_reentrant and
                        not is_public and not is_external and
                        not based_on and not at_location and
                        not data_values_nodes and not initial_values_nodes)

        # Check if this is a procedure local that can use shared storage
        use_shared = False
        if (not in_reentrant and self.current_proc and not is_public and not is_external
            and not based_on and not at_location and not data_values_nodes
            and not initial_values_nodes):
            if (hasattr(self, 'storage_labels')
                and self.current_proc in self.storage_labels
                and name in self.storage_labels[self.current_proc]):
                asm_name = self.storage_labels[self.current_proc][name]
                use_shared = True

        if not use_shared and not in_reentrant:
            # For non-public local variables in procedures, prefix with scope name to avoid conflicts
            if self.current_proc and not is_public and not is_external:
                asm_name = f"@{self.current_proc}${base_name}"
            else:
                asm_name = base_name
        elif in_reentrant:
            asm_name = None  # Will use stack_offset instead

        # Calculate size
        if struct_members:
            # Size of one structure element
            struct_size = sum(
                (m.dimension or 1) * (1 if m.data_type == DataType.BYTE else 2)
                for m in struct_members
            )
            # Multiply by array dimension if this is an array of structures
            size = struct_size * (dimension or 1)
            elem_size = 2  # Structures are ADDRESS-sized elements
        else:
            elem_size = 1 if data_type == DataType.BYTE else 2
            count = dimension or 1
            size = elem_size * count

        # For reentrant procedures, allocate stack space for locals
        stack_offset = None
        if in_reentrant:
            # Locals are at negative offsets from IX
            # Decrement offset first, then use it (so first local is at IX-size)
            self._reentrant_local_offset -= size
            stack_offset = self._reentrant_local_offset

        # LABEL declarations: register the label and emit any extrn/public
        # directive — labels never get storage.
        if data_type == DataType.LABEL and not struct_members:
            self.symbols.define(
                Symbol(
                    name=name,
                    kind=SymbolKind.LABEL,
                    is_public=is_public,
                    is_external=is_external,
                )
            )
            if is_external:
                self._emit("extrn", base_name)
            elif is_public:
                self._emit("public", base_name)
            return

        # Record in symbol table (with mangled name for asm output)
        sym = Symbol(
            name=name,
            kind=SymbolKind.VARIABLE,
            data_type=data_type,
            dimension=dimension,
            struct_members=struct_members,
            based_on=based_on,  # Keep original name for symbol lookup
            is_public=is_public,
            is_external=is_external,
            size=size,
            asm_name=asm_name,  # Store mangled name (None for reentrant locals)
            stack_offset=stack_offset,  # Stack offset for reentrant locals
        )
        self.symbols.define(sym)

        # External variables don't get storage here
        if is_external:
            self._emit("extrn", asm_name)
            return

        # Public declaration
        if is_public:
            self._emit("public", asm_name)

        # Based variables don't allocate storage - they're pointers to other storage
        if based_on:
            return

        # AT variables use specified address
        if at_location is not None:
            self._emit_at_decl(asm_name, at_location, sym)
            return

        # Generate storage
        # DATA values can go inline in code (for module-level bootstrap) or data segment
        target_segment = self.code_data_segment if self.emit_data_inline else self.data_segment

        if data_values_nodes:
            target_segment.append(AsmLine(label=asm_name))
            self._emit_data_values(
                data_values_nodes,
                data_type or DataType.BYTE,
                inline=self.emit_data_inline,
            )
        elif initial_values_nodes:
            self.data_segment.append(AsmLine(label=asm_name))
            self._emit_initial_values(initial_values_nodes, data_type or DataType.BYTE)
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

    def _emit_at_decl(self, asm_name: str | None, at_expr, sym: Symbol) -> None:
        """Emit the EQU/SET line(s) for a ``DECLARE ... AT(addr)`` clause.

        ``at_expr`` is a typed expression node. A bare ``NUMBER`` is
        emitted as a direct EQU; a ``.NAME`` (LocationOf) becomes a SET
        to the referenced symbol; ``.ARR(i)`` resolves to a SET with the
        appropriate element offset when the index is a constant.
        """
        # AT(<number>): direct address EQU.
        if isinstance(at_expr, P.NumberLiteral):
            addr = parse_plm_number(at_expr.value.text)
            self.data_segment.append(
                AsmLine(label=asm_name, opcode="EQU", operands=self._format_number(addr))
            )
            return

        if isinstance(at_expr, P.LocationOf):
            loc_operand = at_expr.operand
            # AT(.NAME)
            if isinstance(loc_operand, P.Identifier):
                ref_name_text = ident_text(loc_operand.name)
                if ref_name_text.upper() == "MEMORY":
                    self.needs_end_symbol = True
                    # SET (not EQU) — forward reference to __END__ at file end.
                    self.data_segment.append(
                        AsmLine(label=asm_name, opcode="SET", operands="__END__")
                    )
                else:
                    ref_sym = self.symbols.lookup(ref_name_text)
                    if ref_sym and ref_sym.is_external:
                        # AT(.external) — alias the external's name; no directive.
                        sym.asm_name = (
                            ref_sym.asm_name if ref_sym.asm_name
                            else self._mangle_name(ref_name_text)
                        )
                    else:
                        ref_asm = (
                            ref_sym.asm_name if ref_sym and ref_sym.asm_name
                            else self._mangle_name(ref_name_text)
                        )
                        # SET allows forward references.
                        self.data_segment.append(
                            AsmLine(label=asm_name, opcode="SET", operands=ref_asm)
                        )
                return

            # AT(.ARR(i)) — the subscript parses as a Call in the typed AST.
            if isinstance(loc_operand, P.Call):
                base_expr = loc_operand.callee
                index_expr = loc_operand.args[0] if loc_operand.args else None
                if isinstance(base_expr, P.Identifier):
                    base_name_text = ident_text(base_expr.name)
                    base_sym = self.symbols.lookup(base_name_text)
                    base_asm = (
                        base_sym.asm_name if base_sym and base_sym.asm_name
                        else self._mangle_name(base_name_text)
                    )
                    elem_size = 1
                    if base_sym and base_sym.data_type == DataType.ADDRESS:
                        elem_size = 2
                    # External-base detection (direct or through an AT alias).
                    is_base_external = bool(base_sym and base_sym.is_external)
                    if not is_base_external and base_sym and base_sym.asm_name:
                        asm_base = base_sym.asm_name.split('+')[0].strip()
                        ref_sym = self.symbols.lookup(asm_base)
                        if ref_sym and ref_sym.is_external:
                            is_base_external = True
                    if isinstance(index_expr, P.NumberLiteral):
                        offset = parse_plm_number(index_expr.value.text) * elem_size
                        if is_base_external:
                            # External base — store the expression as asm_name.
                            sym.asm_name = (
                                base_asm if offset == 0 else f"{base_asm}+{offset}"
                            )
                        elif offset == 0:
                            self.data_segment.append(
                                AsmLine(label=asm_name, opcode="SET", operands=base_asm)
                            )
                        else:
                            self.data_segment.append(
                                AsmLine(
                                    label=asm_name, opcode="SET",
                                    operands=f"{base_asm}+{offset}",
                                )
                            )
                    else:
                        # Non-constant index - can't resolve at compile time.
                        self.data_segment.append(
                            AsmLine(label=asm_name, opcode="EQU", operands="$")
                        )
                else:
                    # Complex base expression
                    self.data_segment.append(
                        AsmLine(label=asm_name, opcode="EQU", operands="$")
                    )
                return

            # Other LocationOf forms (string-of, etc.) — fall back.
            self.data_segment.append(
                AsmLine(label=asm_name, opcode="EQU", operands="$")
            )
            return

        # Catch-all: evaluate at assembly time.
        self.data_segment.append(
            AsmLine(label=asm_name, opcode="EQU", operands="$")
        )

    def _emit_data_values(self, values, dtype: DataType, inline: bool = False) -> None:
        """Emit typed DATA values to the data segment or inline code segment.

        Accepts the raw typed expression nodes from the AST so callers
        don't need a separate conversion pass.
        """
        target = self.code_data_segment if inline else self.data_segment
        for val in values:
            if isinstance(val, P.NumberLiteral):
                directive = "db" if dtype == DataType.BYTE else "dw"
                target.append(
                    AsmLine(opcode=directive, operands=self._format_number(number_value(val)))
                )
            elif isinstance(val, P.StringLiteral):
                target.append(
                    AsmLine(opcode="db", operands=self._escape_string(string_value(val)))
                )
            elif isinstance(val, P.Identifier):
                # Could be a LITERALLY macro - expand it
                name = ident_text(val.name)
                if name in self.literal_macros:
                    try:
                        num_val = self._parse_plm_number(self.literal_macros[name])
                        directive = "db" if dtype == DataType.BYTE else "dw"
                        target.append(
                            AsmLine(opcode=directive, operands=self._format_number(num_val))
                        )
                    except ValueError:
                        # Not a number, use the macro body as-is.
                        target.append(
                            AsmLine(opcode="db", operands=self.literal_macros[name])
                        )
                else:
                    # Unknown identifier - use as label reference
                    target.append(
                        AsmLine(opcode="dw", operands=name)
                    )
            elif isinstance(val, P.LocationOf):
                # Address-of expression: .variable or .procedure
                operand = val.operand
                if isinstance(operand, P.Identifier):
                    name = ident_text(operand.name)
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
                    asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(name)
                    target.append(
                        AsmLine(opcode="dw", operands=asm_name)
                    )
                else:
                    raise CodeGenError(f"Unsupported operand in DATA location expression: {operand}")
            elif isinstance(val, P.BinaryOp):
                # Binary expression like .name-3 or name+offset
                expr_str = self._data_expr_to_string(val)
                target.append(
                    AsmLine(opcode="dw", operands=expr_str)
                )
            elif isinstance(val, P.LocationOfList):
                # Nested address-of list: .(a, b, c)
                for v in val.values or []:
                    self._emit_data_values([v], dtype, inline=inline)
            elif isinstance(val, P.ParenExpr):
                # Parenthesised single value — unwrap and re-emit.
                self._emit_data_values([val.inner], dtype, inline=inline)

    def _data_expr_to_string(self, expr) -> str:
        """Convert a typed DATA expression to an assembly operand string."""
        if isinstance(expr, P.NumberLiteral):
            return self._format_number(number_value(expr))
        elif isinstance(expr, P.Identifier):
            name = ident_text(expr.name)
            if name in self.literal_macros:
                return self.literal_macros[name]
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
        elif isinstance(expr, P.LocationOf):
            return self._data_expr_to_string(expr.operand)
        elif isinstance(expr, P.ParenExpr):
            return self._data_expr_to_string(expr.inner)
        elif isinstance(expr, P.BinaryOp):
            left = self._data_expr_to_string(expr.left)
            right = self._data_expr_to_string(expr.right)
            op_map = {
                BinaryOpKind.ADD: '+',
                BinaryOpKind.SUB: '-',
                BinaryOpKind.MUL: '*',
                BinaryOpKind.DIV: '/',
                BinaryOpKind.AND: ' AND ',
                BinaryOpKind.OR: ' OR ',
                BinaryOpKind.XOR: ' XOR ',
            }
            op = op_map.get(binop_kind(expr), '+')
            return f"({left}{op}{right})"
        else:
            raise CodeGenError(f"Unsupported expression in DATA: {type(expr)}")

    def _emit_initial_values(self, values, dtype: DataType) -> None:
        """Emit typed INITIAL values to the data segment."""
        for val in values:
            if isinstance(val, P.NumberLiteral):
                directive = "db" if dtype == DataType.BYTE else "dw"
                self.data_segment.append(
                    AsmLine(opcode=directive, operands=self._format_number(number_value(val)))
                )
            elif isinstance(val, P.StringLiteral):
                self.data_segment.append(
                    AsmLine(opcode="db", operands=self._escape_string(string_value(val)))
                )

    def _gen_proc_decl(self, decl) -> None:
        """Generate code for a procedure declaration.

        ``decl`` is a typed :class:`P.ProcDecl` from the uplox-generated
        parser. The signature's attribute clauses (EXTERNAL / PUBLIC /
        REENTRANT / INTERRUPT) are walked into a :class:`ProcAttrs` view
        via :func:`ast_view.proc_attrs`; the body's flat item list is
        split into local declarations and statements via
        :func:`ast_view.proc_local_decls_stmts`.
        """
        old_proc = self.current_proc
        old_proc_decl = self.current_proc_decl
        old_proc_attrs = self.current_proc_attrs
        old_proc_return_type = self.current_proc_return_type

        attrs = proc_attrs(decl)
        name = proc_name(decl)
        params = proc_param_names(decl)
        return_type = _legacy_dt(proc_return_type(decl))
        local_decls, body_stmts = proc_local_decls_stmts(decl)

        # For nested procedures, create a unique scoped name
        if old_proc and not attrs.is_public and not attrs.is_external:
            # Nested procedure - use scoped name
            proc_asm_name = f"@{old_proc}${name}"
            full_proc_name = f"{old_proc}${name}"
            self.current_proc = full_proc_name  # Compound name for further nesting
        else:
            proc_asm_name = name
            full_proc_name = name
            self.current_proc = name

        self.current_proc_decl = decl
        self.current_proc_attrs = attrs
        self.current_proc_return_type = return_type

        # Look up the procedure (already registered in pass 1)
        # Use full_proc_name to find the correct symbol for nested procs
        sym = self.symbols.lookup(full_proc_name)
        if sym is None:
            sym = Symbol(
                name=full_proc_name,
                kind=SymbolKind.PROCEDURE,
                return_type=return_type,
                params=params,
                is_public=attrs.is_public,
                is_external=attrs.is_external,
                is_reentrant=attrs.is_reentrant,
                interrupt_num=attrs.interrupt_num,
                asm_name=proc_asm_name,
            )
            self.symbols.define(sym)
        else:
            # Use the asm_name from pass 1
            proc_asm_name = sym.asm_name or proc_asm_name

        if attrs.is_external:
            self._emit("extrn", proc_asm_name)
            self.current_proc = old_proc
            self.current_proc_decl = old_proc_decl
            self.current_proc_attrs = old_proc_attrs
            self.current_proc_return_type = old_proc_return_type
            return

        self._emit()
        if attrs.is_public:
            self._emit("public", name)

        self._emit(comment=f"Procedure {name}")
        self._emit_label(proc_asm_name)

        # Enter new scope
        self.symbols.enter_scope(name)

        # Procedure prologue
        if attrs.interrupt_num is not None:
            # Interrupt handler - save all registers
            self._emit("push", "af")
            self._emit("push", "bc")
            self._emit("push", "de")
            self._emit("push", "hl")

        # Build a name -> (legacy) DataType map from local DeclItem
        # nodes so we can resolve parameter types declared inside the
        # body's DECLARE statements.
        param_type_by_name: dict[str, DataType] = {}
        for d in local_decls:
            if isinstance(d, P.DeclItem):
                d_view_type, _ = _view_decl_item_type(d)
                d_legacy_type = _legacy_dt(d_view_type)
                if d_legacy_type is None:
                    continue
                for n in decl_item_names(d):
                    param_type_by_name[n] = d_legacy_type

        # Define parameters as local variables
        # For non-reentrant: use shared automatic storage via storage_labels
        # For reentrant: use IX-relative stack frame
        param_infos: list[tuple[str, str, DataType, int]] = []  # (name, asm_name, type, size)
        use_shared_storage = not attrs.is_reentrant and full_proc_name in self.storage_labels

        # For reentrant procedures, set up IX frame pointer first
        # Stack at entry: [params...][ret_addr] <- SP
        # After PUSH IX: [params...][ret_addr][saved_IX] <- SP, IX
        if attrs.is_reentrant:
            self._emit("push", "ix")
            self._emit("ld", "ix,0")
            self._emit("add", "ix,sp")

        # Calculate parameter offsets for reentrant procedures
        # Stack after PUSH IX: [params...][ret_addr(2)][saved_IX(2)] <- IX
        # First param is at IX+4, subsequent params at higher offsets
        # Parameters are pushed in order: first arg pushed first, ends up deepest
        # So params[0] is at the highest offset, params[-1] is at IX+4
        reentrant_param_offset = 4  # Start after saved IX (2) and ret addr (2)
        if attrs.is_reentrant:
            # All stack slots are 2 bytes (pushed as 16-bit) regardless of
            # the declared parameter type — last param is at IX+4.
            param_sizes = [2 for _ in params]
            total_params_size = sum(param_sizes)
            reentrant_param_offset = (
                4 + total_params_size - param_sizes[-1] if param_sizes else 4
            )

        for i, param in enumerate(params):
            param_type = param_type_by_name.get(param) or DataType.ADDRESS
            param_size = 1 if param_type == DataType.BYTE else 2

            if attrs.is_reentrant:
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
                    asm_name = f"@{name}${self._mangle_name(param)}"
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
        if param_infos and not attrs.is_reentrant:
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
        nested_procs: list = []
        for local_decl in local_decls:
            if isinstance(local_decl, P.ProcDecl):
                # Defer nested procedures
                nested_procs.append(local_decl)
            elif isinstance(local_decl, P.DeclItem):
                # Skip if every declared name is a parameter (already defined)
                local_names = decl_item_names(local_decl)
                non_param_names = [n for n in local_names if n not in params]
                if not non_param_names:
                    continue
                # If the DeclItem declares a mix of params and non-params,
                # still hand the whole item to _gen_declaration — the
                # symbol-table side handles already-defined names.
                self._gen_declaration(local_decl)
            else:
                self._gen_declaration(local_decl)

        # For reentrant procedures, allocate stack space for locals
        if attrs.is_reentrant and self._reentrant_local_offset < 0:
            # Allocate stack space: SP = SP + offset (offset is negative)
            # ld hl,offset; add hl,sp; ld sp,hl
            self._emit("ld", f"hl,{self._reentrant_local_offset}")
            self._emit("add", "hl,sp")
            self._emit("ld", "sp,hl")

        # Generate code for statements with liveness tracking
        ends_with_return = False
        for i, stmt in enumerate(body_stmts):
            # Track remaining statements for liveness analysis
            self.pending_stmts = body_stmts[i + 1:]
            self._gen_stmt(stmt)
            ends_with_return = isinstance(stmt, (P.ReturnStmt, P.ReturnStmtValue))
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
        self.current_proc_attrs = old_proc_attrs
        self.current_proc_return_type = old_proc_return_type

    def _gen_proc_epilogue(self, decl) -> None:
        """Generate procedure epilogue for a typed :class:`P.ProcDecl`."""
        attrs = proc_attrs(decl)
        if attrs.interrupt_num is not None:
            self._emit("pop", "hl")
            self._emit("pop", "de")
            self._emit("pop", "bc")
            self._emit("pop", "af")
            self._emit("ei")
            self._emit("ret")
        elif attrs.is_reentrant:
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

    def _gen_stmt(self, stmt) -> None:
        """Generate code for a single typed statement node.

        Dispatches over the uplox-generated :mod:`uplm80._plm_parser`
        statement classes. The legacy ``IfStmt``/``ReturnStmt``/
        ``DoIterBlock`` shapes are each split into two typed kinds (with
        / without an else / value / step) — both variants funnel into
        the same handler with the optional field defaulted.
        """
        if isinstance(stmt, P.AssignStmt):
            self._gen_assign(stmt)
        elif isinstance(stmt, P.CallStmt):
            self._gen_call_stmt(stmt)
        elif isinstance(stmt, (P.ReturnStmt, P.ReturnStmtValue)):
            self._gen_return(stmt)
        elif isinstance(stmt, P.GotoStmt):
            # Check if target is a LITERALLY macro
            target = ident_text(stmt.label)
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
        elif isinstance(stmt, P.HaltStmt):
            self._emit("halt")
        elif isinstance(stmt, P.EnableStmt):
            self._emit("ei")
        elif isinstance(stmt, P.DisableStmt):
            self._emit("di")
        elif isinstance(stmt, P.NullStmt):
            pass  # No code
        elif isinstance(stmt, P.LabeledStmt):
            raw_label = ident_text(stmt.label)
            if self.current_proc:
                # Procedure-local label - prefix with current procedure
                label = f"@{self.current_proc}${raw_label}"
            else:
                # Module-level label - register in symbol table for GOTO lookups
                self.symbols.define(
                    Symbol(
                        name=raw_label,
                        kind=SymbolKind.LABEL,
                    )
                )
                label = raw_label
            self._emit_label(label)
            self._gen_stmt(stmt.stmt)
        elif isinstance(stmt, (P.IfStmt, P.IfStmtElse)):
            self._gen_if(stmt)
        elif isinstance(stmt, P.DoBlock):
            self._gen_do_block(stmt)
        elif isinstance(stmt, P.DoWhileBlock):
            self._gen_do_while(stmt)
        elif isinstance(stmt, (P.DoIterBlock, P.DoIterByBlock)):
            self._gen_do_iter(stmt)
        elif isinstance(stmt, P.DoCaseBlock):
            self._gen_do_case(stmt)
        elif isinstance(stmt, P.DeclareStmt):
            for decl in stmt.declarations:
                self._gen_declaration(decl)

    def _gen_assign(self, stmt) -> None:
        """Generate code for assignment.

        ``stmt`` is a typed :class:`P.AssignStmt`; ``stmt.targets`` is
        the list of LHS expressions (one or many) and ``stmt.value`` is
        the RHS expression — both kept as raw typed nodes so the
        downstream :meth:`_gen_expr` / :meth:`_gen_store` (still on the
        legacy AST) walks them via ``isinstance``. The byte-constant
        optimisation peeks at the typed shape via :func:`number_value`.
        """
        targets = list(stmt.targets)

        # Special case: storing small constant to BYTE variable.
        # Use ``xor a`` (for 0) or ``ld a,n`` (for other bytes)
        # instead of ``ld hl,n``.
        if isinstance(stmt.value, P.NumberLiteral):
            const_val = number_value(stmt.value)
            if const_val <= 255 and all(
                self._is_byte_target(t) for t in targets
            ):
                # Generate efficient byte constant
                if const_val == 0:
                    self._emit("xor", "a")
                else:
                    self._emit("ld", f"a,{self._format_number(const_val)}")

                for i, target in enumerate(targets):
                    if i < len(targets) - 1:
                        self._emit("push", "af")
                    self._gen_store(target, DataType.BYTE)
                    if i < len(targets) - 1:
                        self._emit("pop", "af")
                return

        # Evaluate the value expression (result in A for BYTE, HL for ADDRESS)
        value_type = self._gen_expr(stmt.value)

        # Store to each target (multiple assignment support)
        for i, target in enumerate(targets):
            if i < len(targets) - 1:
                # Need to preserve value for next target
                if value_type == DataType.BYTE:
                    self._emit("push", "af")
                else:
                    self._emit("push", "hl")

            self._gen_store(target, value_type)

            if i < len(targets) - 1:
                if value_type == DataType.BYTE:
                    self._emit("pop", "af")
                else:
                    self._emit("pop", "hl")

    def _is_byte_target(self, target) -> bool:
        """Return True when a typed assignment target is a BYTE variable.

        Recognises bare identifiers, member access, and parser ``Call``
        forms (which PL/M-80 uses for both calls and array subscripts —
        the typed AST doesn't distinguish them syntactically). Anything
        else conservatively returns False so the byte-constant
        optimisation falls back to the general path.
        """
        if isinstance(target, P.Identifier):
            sym = self.symbols.lookup(ident_text(target.name))
            return bool(sym and sym.data_type == DataType.BYTE)
        if isinstance(target, P.MemberAccess):
            # Member access — let the slow path resolve struct members.
            return False
        if isinstance(target, P.Call):
            # PL/M call syntax doubles as array subscript. Treat a
            # single-arg call on an identifier as a subscript and look
            # up the array element type.
            callee = target.callee
            if isinstance(callee, P.Identifier) and len(target.args) == 1:
                sym = self.symbols.lookup(ident_text(callee.name))
                if (
                    sym
                    and sym.kind != SymbolKind.PROCEDURE
                    and sym.data_type == DataType.BYTE
                ):
                    return True
            return False
        return False

    def _gen_call_stmt(self, stmt) -> None:
        """Generate code for a CALL statement.

        ``stmt`` is a typed :class:`P.CallStmt` whose ``.callee`` field
        carries the call expression itself: a :class:`P.Call`
        (callee + args), a :class:`P.CallNoArgs` (just the callee), or
        a bare :class:`P.Identifier` for parameterless invocations. The
        legacy AST had a separate ``CallStmt(callee, args)`` shape with
        the arg list hoisted to the statement; unpack into the same
        ``(callee_expr, args)`` pair here so the rest of the body keeps
        the legacy structure.
        """
        # Unpack the call form into (callee_expr, args).
        inner = stmt.callee
        if isinstance(inner, P.Call):
            callee_expr = inner.callee
            args = list(inner.args)
        elif isinstance(inner, P.CallNoArgs):
            callee_expr = inner.callee
            args = []
        else:
            # Bare identifier or other expression form — treat as a
            # parameterless call on the expression itself.
            callee_expr = inner
            args = []

        # Look up procedure symbol to check if it's user-defined
        sym = None
        call_name = None
        callee_name_str: str | None = None
        if isinstance(callee_expr, P.Identifier):
            callee_name_str = ident_text(callee_expr.name)
            name = callee_name_str
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
        if callee_name_str is not None:
            is_builtin = (sym is None or sym.kind == SymbolKind.BUILTIN)
            if is_builtin:
                upper_name = callee_name_str.upper()
                # Handle built-in procedures that don't return values
                if upper_name in self.BUILTIN_FUNCS:
                    result = self._gen_builtin(upper_name, args)
                    if result is not None or upper_name in ('TIME', 'MOVE'):
                        # Built-in was handled
                        return

        # If sym/call_name weren't set yet, look up again (for member access etc.)
        if callee_name_str is not None and call_name is None:
            name = callee_name_str
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
        if callee_name_str is not None:
            upper_name = callee_name_str.upper()
            if upper_name in ('MON1', 'MON2') and len(args) == 2:
                func_arg, addr_arg = args
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
            for arg in args:
                arg_type = self._gen_expr(arg)
                if arg_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("push", "hl")
        else:
            # Direct memory parameter passing (non-reentrant)
            # Last param is passed in register (A for BYTE, HL for ADDRESS)
            # Other params are stored to memory
            last_param_idx = len(args) - 1
            uses_reg = sym.uses_reg_param and len(args) > 0

            for i, arg in enumerate(args):
                if i < len(sym.params):
                    param_name = sym.params[i]
                    param_type = sym.param_types[i] if i < len(sym.param_types) else DataType.ADDRESS
                    is_last = (i == last_param_idx)

                    # Last param passed in register - just evaluate it
                    if is_last and uses_reg:
                        # Optimize constants for BYTE
                        if param_type == DataType.BYTE:
                            const = self._get_const_byte_value(arg)
                            if const is not None:
                                self._emit("ld", f"a,{self._format_number(const)}")
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
                    # Try to get param asm name from shared storage
                    param_asm = None
                    if (hasattr(self, 'storage_labels')
                        and full_callee_name in self.storage_labels
                        and param_name in self.storage_labels[full_callee_name]):
                        param_asm = self.storage_labels[full_callee_name][param_name]
                    else:
                        # Fallback: build param asm name: @procname$param
                        proc_base = sym.asm_name if sym.asm_name else callee_name_str or ""
                        if proc_base.startswith('@'):
                            proc_base = proc_base[1:]
                        param_asm = f"@{proc_base}${self._mangle_name(param_name)}"

                    # Optimize: for BYTE parameter with constant, use ld a,n directly
                    if param_type == DataType.BYTE:
                        const = self._get_const_byte_value(arg)
                        if const is not None:
                            self._emit("ld", f"a,{self._format_number(const)}")
                            self._emit("ld", f"({param_asm}),a")
                            continue

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
        if callee_name_str is not None:
            self._emit("call", call_name)
        else:
            # Indirect call through address
            self._gen_expr(callee_expr)
            self._emit("jp", "(hl)")

        # Clean up stack (caller cleanup) - only for stack-based calls
        if use_stack and args:
            stack_bytes = len(args) * 2
            if stack_bytes == 2:
                self._emit("pop", "de")  # Dummy pop
            elif stack_bytes == 4:
                self._emit("pop", "de")
                self._emit("pop", "de")
            elif stack_bytes <= 8:
                for _ in range(len(args)):
                    self._emit("pop", "de")
            else:
                # Adjust stack pointer directly
                self._emit("ld", f"de,{stack_bytes}")
                self._emit("add", "hl,sp")
                self._emit("ld", "sp,hl")

    def _gen_return(self, stmt) -> None:
        """Generate code for a RETURN statement.

        ``stmt`` is either :class:`P.ReturnStmt` (no value) or
        :class:`P.ReturnStmtValue` (with a typed expression in
        ``.value``). The legacy single class with ``value=None`` was
        split into two kinds; treat them uniformly by reading the
        optional value off the variant. Procedure-context attributes
        (return type, interrupt-handler flag, reentrant flag) come from
        the side-cached :attr:`current_proc_return_type` /
        :attr:`current_proc_attrs` rather than the typed
        :class:`P.ProcDecl` directly.
        """
        value = stmt.value if isinstance(stmt, P.ReturnStmtValue) else None
        return_type = self.current_proc_return_type
        proc_attrs_view = self.current_proc_attrs

        if value is not None:
            # Check if A already has the value from embedded assignment optimization
            skip_load = False
            if (
                self.embedded_assign_target
                and isinstance(value, P.Identifier)
                and ident_text(value.name) == self.embedded_assign_target
            ):
                # A already has this value - skip the load
                skip_load = True
                self.embedded_assign_target = None  # Clear after use

            if skip_load:
                # A already contains the return value - just return
                pass
            # Optimize: if returning BYTE and value is a small constant, use ld a,n directly
            elif (
                return_type == DataType.BYTE
                and isinstance(value, P.NumberLiteral)
                and number_value(value) <= 255
            ):
                self._emit(
                    "ld",
                    f"a,{self._format_number(number_value(value))}",
                )
            else:
                result_type = self._gen_expr(value)
                # Return value is in A (BYTE) or HL (ADDRESS)
                # If procedure returns BYTE but we have ADDRESS, convert
                if return_type == DataType.BYTE and result_type == DataType.ADDRESS:
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
                elif return_type == DataType.ADDRESS and result_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")

        if proc_attrs_view is not None and proc_attrs_view.interrupt_num is not None:
            # Interrupt handler return
            self._emit("pop", "hl")
            self._emit("pop", "de")
            self._emit("pop", "bc")
            self._emit("pop", "af")
            self._emit("ei")
            self._emit("ret")
        elif proc_attrs_view is not None and proc_attrs_view.is_reentrant:
            # Reentrant procedure return - restore frame pointer
            self._emit("ld", "sp,ix")
            self._emit("pop", "ix")
            self._emit("ret")
        else:
            self._emit("ret")

    def _gen_if(self, stmt) -> None:
        """Generate code for an IF statement.

        ``stmt`` is either :class:`P.IfStmt` (no ELSE) or
        :class:`P.IfStmtElse` (with ELSE) — the typed grammar splits
        the two; treat them uniformly by reading the optional
        else-branch off the variant.
        """
        else_stmt = stmt.else_stmt if isinstance(stmt, P.IfStmtElse) else None

        # Warn about trivial constant conditions (IF 0, IF 1)
        self._warn_trivial_if(stmt.condition)

        else_label = self._new_label("ELSE")
        end_label = self._new_label("ENDIF")
        false_target = else_label if else_stmt is not None else end_label

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

        if else_stmt is not None:
            self._emit("jp", end_label)
            self._emit_label(else_label)
            self._gen_stmt(else_stmt)

        self._emit_label(end_label)

    # Comparison kinds the optimised branch generators know how to emit.
    _COMPARISON_KINDS = frozenset(
        {
            BinaryOpKind.EQ,
            BinaryOpKind.NE,
            BinaryOpKind.LT,
            BinaryOpKind.GT,
            BinaryOpKind.LE,
            BinaryOpKind.GE,
        }
    )

    def _gen_condition_jump_false(self, condition, false_label: str) -> bool:
        """Generate conditional jump to ``false_label`` when ``condition``
        evaluates to false.

        Accepts a typed expression node; comparison operators are
        decoded via :func:`binop_kind` so the branch-emit helpers
        ((:meth:`_emit_jump_on_false`, :meth:`_emit_jump_on_false_16bit`)
        consume :class:`ast_view.BinaryOpKind`. Returns True if an
        optimised jump was generated (caller skips the fallback),
        False otherwise.
        """
        # Handle constant conditions - no code needed for always-true, unconditional jump for always-false
        if isinstance(condition, P.NumberLiteral):
            if number_value(condition) == 0:
                # Always false - unconditional jump
                self._emit("jp", false_label)
            # If non-zero (always true), no code needed - just fall through
            return True

        # Handle simple identifier - load and test directly
        if isinstance(condition, P.Identifier):
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
        if isinstance(condition, (P.Call, P.CallNoArgs)):
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
        if isinstance(condition, P.UnaryOp) and unop_kind(condition) == UnaryOpKind.NOT:
            # NOT x is false when x is true, so jump to false_label when x is true
            return self._gen_condition_jump_true(condition.operand, false_label)

        if not isinstance(condition, P.BinaryOp):
            return False

        op = binop_kind(condition)

        # NOTE: PL/M-80 AND and OR are BITWISE operators, not short-circuit logical operators.
        # IF X AND Y tests if (X bitwise-and Y) is non-zero, NOT if both X and Y are non-zero.
        # So we do NOT handle AND/OR specially here - they fall through to expression evaluation.

        if op not in self._COMPARISON_KINDS:
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
            if isinstance(condition.right, P.NumberLiteral):
                val = number_value(condition.right)
                # Allow direct byte values (0-255) or negative byte values (0xFF00-0xFFFF)
                if val <= 255 or (val & 0xFF00) == 0xFF00:
                    const_val = val & 0xFF
            elif isinstance(condition.right, P.StringLiteral):
                s = string_value(condition.right)
                if len(s) == 1:
                    const_val = ord(s[0])

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
            if (
                op in (BinaryOpKind.EQ, BinaryOpKind.NE)
                and isinstance(condition.right, P.NumberLiteral)
                and number_value(condition.right) == 0
            ):
                self._gen_expr(condition.left)  # Result in HL
                if left_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("ld", "a,l")
                self._emit("or", "h")  # Z flag set if HL == 0
                if op == BinaryOpKind.EQ:
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
            if op in (BinaryOpKind.EQ, BinaryOpKind.NE):
                self._emit("ld", "a,l")
                self._emit("or", "h")
                if op == BinaryOpKind.EQ:
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

    def _gen_condition_jump_true(self, condition, true_label: str) -> bool:
        """Generate conditional jump to ``true_label`` when ``condition``
        evaluates to true. Mirror of :meth:`_gen_condition_jump_false`.

        Accepts a typed expression node; comparison operators are
        decoded via :func:`binop_kind`. Returns True if an optimised
        jump was generated, False if the caller should fall back to
        the generic ``_gen_expr`` + test-flags sequence.
        """
        # Handle constant conditions
        if isinstance(condition, P.NumberLiteral):
            if number_value(condition) != 0:
                # Always true - unconditional jump
                self._emit("jp", true_label)
            # If zero (always false), no code needed - just fall through
            return True

        # Handle simple identifier
        if isinstance(condition, P.Identifier):
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
        if isinstance(condition, (P.Call, P.CallNoArgs)):
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
        if isinstance(condition, P.UnaryOp) and unop_kind(condition) == UnaryOpKind.NOT:
            # NOT x is true when x is false, so jump to true_label when x is false
            return self._gen_condition_jump_false(condition.operand, true_label)

        if not isinstance(condition, P.BinaryOp):
            return False

        op = binop_kind(condition)

        # NOTE: PL/M-80 AND and OR are BITWISE operators, not short-circuit logical operators.
        # IF X OR Y tests if (X bitwise-or Y) is non-zero, NOT if either X or Y is non-zero.
        # So we do NOT handle AND/OR specially here - they fall through to expression evaluation.

        if op not in self._COMPARISON_KINDS:
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
            if isinstance(condition.right, P.NumberLiteral):
                val = number_value(condition.right)
                # Allow direct byte values (0-255) or negative byte values (0xFF00-0xFFFF)
                if val <= 255 or (val & 0xFF00) == 0xFF00:
                    const_val = val & 0xFF
            elif isinstance(condition.right, P.StringLiteral):
                s = string_value(condition.right)
                if len(s) == 1:
                    const_val = ord(s[0])

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
            if (
                op in (BinaryOpKind.EQ, BinaryOpKind.NE)
                and isinstance(condition.right, P.NumberLiteral)
                and number_value(condition.right) == 0
            ):
                self._gen_expr(condition.left)  # Result in HL
                if left_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("ld", "a,l")
                self._emit("or", "h")  # Z flag set if HL == 0
                if op == BinaryOpKind.EQ:
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

            if op in (BinaryOpKind.EQ, BinaryOpKind.NE):
                self._emit("ld", "a,l")
                self._emit("or", "h")
                if op == BinaryOpKind.EQ:
                    self._emit("jp", f"z,{true_label}")
                else:
                    self._emit("jp", f"nz,{true_label}")
                return True
            else:
                self._emit_jump_on_true_16bit(op, true_label)
                return True

        return False

    def _emit_jump_on_true(self, op: BinaryOpKind, true_label: str) -> None:
        """Emit jump to ``true_label`` if comparison result is true (8-bit compare)."""
        if op == BinaryOpKind.EQ:
            self._emit("jp", f"z,{true_label}")
        elif op == BinaryOpKind.NE:
            self._emit("jp", f"nz,{true_label}")
        elif op == BinaryOpKind.LT:
            self._emit("jp", f"c,{true_label}")
        elif op == BinaryOpKind.GE:
            self._emit("jp", f"nc,{true_label}")
        elif op == BinaryOpKind.GT:
            skip = self._new_label("SKIP")
            self._emit("jp", f"c,{skip}")
            self._emit("jp", f"z,{skip}")
            self._emit("jp", true_label)
            self._emit_label(skip)
        elif op == BinaryOpKind.LE:
            self._emit("jp", f"c,{true_label}")
            self._emit("jp", f"z,{true_label}")

    def _emit_jump_on_true_16bit(self, op: BinaryOpKind, true_label: str) -> None:
        """Emit jump to ``true_label`` for 16-bit unsigned comparison.

        After CALL ??SUBDE (SBC HL,DE), carry flag is set if HL < DE (borrow).
        """
        if op == BinaryOpKind.LT:
            # left < right: true if carry set
            self._emit("jp", f"c,{true_label}")
        elif op == BinaryOpKind.GE:
            # left >= right: true if no carry
            self._emit("jp", f"nc,{true_label}")
        elif op == BinaryOpKind.GT:
            # left > right: true if no carry AND result != 0
            skip = self._new_label("SKIP")
            self._emit("jp", f"c,{skip}")  # left < right -> not greater, skip
            self._emit("ld", "a,l")
            self._emit("or", "h")
            self._emit("jp", f"nz,{true_label}")  # not equal -> greater
            self._emit_label(skip)
        elif op == BinaryOpKind.LE:
            # left <= right: true if carry OR result == 0
            self._emit("jp", f"c,{true_label}")  # left < right -> true
            self._emit("ld", "a,l")
            self._emit("or", "h")
            self._emit("jp", f"z,{true_label}")  # left == right -> true

    def _emit_jump_on_false(self, op: BinaryOpKind, false_label: str) -> None:
        """Emit jump to ``false_label`` if comparison result is false (8-bit compare)."""
        # After cp n or SUB, flags reflect left - right
        if op == BinaryOpKind.EQ:
            self._emit("jp", f"nz,{false_label}")  # Jump if not equal (Z=0)
        elif op == BinaryOpKind.NE:
            self._emit("jp", f"z,{false_label}")   # Jump if equal (Z=1)
        elif op == BinaryOpKind.LT:
            self._emit("jp", f"nc,{false_label}")  # Jump if not less (C=0)
        elif op == BinaryOpKind.GE:
            self._emit("jp", f"c,{false_label}")   # Jump if less (C=1)
        elif op == BinaryOpKind.GT:
            # Greater: not less AND not equal -> C=0 AND Z=0
            self._emit("jp", f"c,{false_label}")   # Jump if less
            self._emit("jp", f"z,{false_label}")   # Jump if equal
        elif op == BinaryOpKind.LE:
            # Less or equal: C=1 OR Z=1
            # Jump if greater (C=0 AND Z=0)
            skip = self._new_label("SKIP")
            self._emit("jp", f"c,{skip}")   # Less -> condition true, skip jump
            self._emit("jp", f"z,{skip}")   # Equal -> condition true, skip jump
            self._emit("jp", false_label)  # Greater -> condition false
            self._emit_label(skip)

    def _emit_jump_on_false_16bit(self, op: BinaryOpKind, false_label: str) -> None:
        """Emit jump to ``false_label`` for 16-bit unsigned comparison.

        After CALL ??SUBDE (SBC HL,DE), carry flag is set if HL < DE (borrow).
        PL/M ADDRESS is unsigned, so we use carry-based comparisons.
        """
        if op == BinaryOpKind.LT:
            # left < right: true if carry set (borrow occurred)
            # Jump to false if NO carry (left >= right)
            self._emit("jp", f"nc,{false_label}")
        elif op == BinaryOpKind.GE:
            # left >= right: true if no carry
            # Jump to false if carry set (left < right)
            self._emit("jp", f"c,{false_label}")
        elif op == BinaryOpKind.GT:
            # left > right: true if no carry AND result != 0
            # Jump to false if carry OR result == 0
            self._emit("jp", f"c,{false_label}")  # left < right -> false
            self._emit("ld", "a,l")
            self._emit("or", "h")
            self._emit("jp", f"z,{false_label}")  # left == right -> false
        elif op == BinaryOpKind.LE:
            # left <= right: true if carry OR result == 0
            # Jump to false if no carry AND result != 0
            skip = self._new_label("SKIP")
            self._emit("jp", f"c,{skip}")  # left < right -> true, skip to end
            self._emit("ld", "a,l")
            self._emit("or", "h")
            self._emit("jp", f"z,{skip}")  # left == right -> true
            self._emit("jp", false_label)  # left > right -> false
            self._emit_label(skip)

    def _gen_do_block(self, stmt) -> None:
        """Generate code for a simple ``DO ... END`` block.

        ``stmt`` is a typed :class:`P.DoBlock`; its mixed ``items``
        list is split into local declarations and statements via
        :func:`block_items_split` (declarations include nested
        :class:`P.ProcDecl` and the contents of inner ``DECLARE``
        statements).
        """
        decls, stmts = block_items_split(stmt.items)

        # Enter scope with unique identifier for DO block local variables
        self.block_scope_counter += 1
        block_id = self.block_scope_counter
        self.symbols.enter_scope(f"B{block_id}")

        # Save and extend current_proc to include block scope for unique asm names
        old_proc = self.current_proc
        if decls:  # Only modify if there are declarations
            if self.current_proc:
                self.current_proc = f"{self.current_proc}$B{block_id}"
            else:
                self.current_proc = f"B{block_id}"

        # Local declarations
        for decl in decls:
            self._gen_declaration(decl)

        # Restore current_proc for statements
        self.current_proc = old_proc

        # Statements
        for s in stmts:
            self._gen_stmt(s)

        self.symbols.leave_scope()

    def _is_byte_counter_loop(self, condition) -> tuple[str, int] | None:
        """
        Check if condition matches the pattern (var := var - 1) <> 255.
        Returns (var_asm_name, compare_value) if matched, None otherwise.

        This pattern is a countdown loop: decrement and check for wrap-around.
        """
        condition = unwrap_paren(condition)
        if not isinstance(condition, P.BinaryOp):
            return None
        if binop_kind(condition) != BinaryOpKind.NE:
            return None
        right = unwrap_paren(condition.right)
        if not isinstance(right, P.NumberLiteral) or number_value(right) != 255:
            return None

        # Left should be (var := var - 1)
        embed = unwrap_paren(condition.left)
        if not isinstance(embed, P.EmbeddedAssign):
            return None
        target = unwrap_paren(embed.target)
        if not isinstance(target, P.Identifier):
            return None

        value = unwrap_paren(embed.value)
        if not isinstance(value, P.BinaryOp):
            return None
        if binop_kind(value) != BinaryOpKind.SUB:
            return None
        vl = unwrap_paren(value.left)
        if not isinstance(vl, P.Identifier):
            return None
        if ident_text(vl.name) != ident_text(target.name):
            return None
        vr = unwrap_paren(value.right)
        if not isinstance(vr, P.NumberLiteral) or number_value(vr) != 1:
            return None

        var_name = ident_text(target.name)
        sym = self._lookup_symbol(var_name)
        if not sym or sym.data_type != DataType.BYTE:
            return None

        asm_name = sym.asm_name if sym.asm_name else self._mangle_name(var_name)
        return (asm_name, 255)

    def _gen_do_while(self, stmt) -> None:
        """Generate code for a ``DO WHILE ... END`` block.

        ``stmt`` is a typed :class:`P.DoWhileBlock`; its mixed
        ``items`` list is split into local declarations and statements
        via :func:`block_items_split` — declarations inside a
        ``DO WHILE`` are rare but legal, and are emitted before the
        loop entry like in a plain ``DO`` block.
        """
        # Note: DO WHILE 1 is a valid pattern (loop exits in middle via RETURN/GOTO)
        # We only error on impossible comparisons like BYTE <> -1
        decls, stmts = block_items_split(stmt.items)

        loop_label = self._new_label("WHILE")
        end_label = self._new_label("WEND")

        self.loop_stack.append((loop_label, end_label))

        # Local declarations (emit storage before loop body).
        for decl in decls:
            self._gen_declaration(decl)

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
            for s in stmts:
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
            for s in stmts:
                self._gen_stmt(s)

            self._emit("jp", loop_label)
            self._emit_label(end_label)

        self.loop_stack.pop()

    def _gen_do_iter(self, stmt) -> None:
        """Generate code for an iterative ``DO I = start TO bound [BY step]`` block.

        ``stmt`` is either :class:`P.DoIterBlock` (no BY clause) or
        :class:`P.DoIterByBlock` (with explicit step). ``stmt.index``
        is a :class:`Token` rather than an expression; wrap it as a
        :class:`P.Identifier` for downstream load/store/usage analysis.
        Local declarations inside the loop body come out of
        :func:`block_items_split` just like a plain ``DO`` block.
        """
        # Build an Identifier-shaped wrapper around the index Token so
        # downstream code (which expects an expression node) sees a
        # uniform shape regardless of which DoIter variant we got.
        index_var = P.Identifier(name=stmt.index)
        index_name = ident_text(stmt.index)
        step_expr = stmt.step if isinstance(stmt, P.DoIterByBlock) else None
        body_stmts = block_items_split(stmt.items)[1]

        loop_label = self._new_label("FOR")
        test_label = self._new_label("TEST")
        incr_label = self._new_label("INCR")
        end_label = self._new_label("NEXT")

        self.loop_stack.append((incr_label, end_label))

        # Determine if index variable is BYTE
        index_type = DataType.ADDRESS
        sym = self._lookup_symbol(index_name)
        if sym and sym.data_type == DataType.BYTE:
            index_type = DataType.BYTE

        # Also check bound type
        bound_type = self._get_expr_type(stmt.bound)
        both_bytes = (index_type == DataType.BYTE and bound_type == DataType.BYTE)

        # Get step value (default +1 when no BY clause; only constant
        # NumberLiteral steps drive the byte-loop optimisations).
        step_val = 1
        if step_expr is not None and isinstance(step_expr, P.NumberLiteral):
            step_val = number_value(step_expr)

        # Check if loop index is used in body - if not, we can use DJNZ on Z80.
        # _index_used_in_body / _stmts_contain_goto still walk the
        # legacy AST shape; they recurse via isinstance and return
        # False for unrecognised typed nodes, which is conservative
        # (forces the safe fallback path).
        index_used = self._index_used_in_body(index_var, body_stmts)

        # Skip DJNZ optimization when the body has a GOTO — the pattern
        # pushes BC at the top of each iteration and pops at the bottom,
        # so a GOTO escaping the body strands the pushed BC on the stack.
        body_has_goto = self._stmts_contain_goto(body_stmts)

        # Z80 DJNZ optimization: DO I = 0 TO N where I is not used
        # Convert to: B = N+1; do { body } while (--B != 0)
        if (
            both_bytes
            and step_val == 1
            and not index_used
            and not body_has_goto
            and isinstance(stmt.start, P.NumberLiteral)
            and number_value(stmt.start) == 0
        ):
            # Calculate iteration count = bound + 1
            # If bound is constant, emit LD B,bound+1
            # If bound is variable, emit: load bound; INC A; LD B,A
            if isinstance(stmt.bound, P.NumberLiteral):
                bound_const = number_value(stmt.bound)
                iter_count = bound_const + 1
                if iter_count <= 255:
                    self._emit("ld", f"b,{self._format_number(iter_count)}")
                else:
                    # Too many iterations for DJNZ
                    pass  # Fall through to regular loop
            else:
                # Variable bound: A = bound; A++; B = A
                bt = self._gen_expr(stmt.bound)
                if bt == DataType.ADDRESS:
                    self._emit("ld", "a,l")
                self._emit("inc", "a")  # A = bound + 1 = iteration count
                self._emit("ld", "b,a")  # B = iteration count

            # Only proceed with B-counter loop if we set up B
            if (
                isinstance(stmt.bound, P.NumberLiteral)
                and number_value(stmt.bound) + 1 <= 255
            ):
                # Loop body - save B since body may clobber it
                self._emit_label(loop_label)
                self._emit("push", "bc")
                for s in body_stmts:
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
            elif not isinstance(stmt.bound, P.NumberLiteral):
                # Variable bound case - we set up B above
                # But need to handle the case where bound might be 255 (iter count = 256 = 0 in byte)
                # Skip loop if B is 0 (this handles bound = 255 case)
                self._emit("ld", "a,b")
                self._emit("or", "a")
                self._emit("jp", f"z,{end_label}")  # Skip if iteration count is 0

                # Loop body - save B since body may clobber it
                self._emit_label(loop_label)
                self._emit("push", "bc")
                for s in body_stmts:
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
            both_bytes
            and isinstance(stmt.bound, P.NumberLiteral)
            and number_value(stmt.bound) == 0
            and (step_val == -1 or step_val == 0xFF)
        )

        if is_downcount_to_zero:
            # Optimized down-counting byte loop
            # Initialize: load start into A, store to index
            start_type = self._gen_expr(stmt.start)
            if start_type == DataType.ADDRESS:
                self._emit("ld", "a,l")
            self._gen_store(index_var, DataType.BYTE)

            # Jump to test
            self._emit("jp", test_label)

            # Loop body
            self._emit_label(loop_label)
            for s in body_stmts:
                self._gen_stmt(s)

            # Decrement
            self._emit_label(incr_label)
            self._gen_load(index_var)  # A = index
            self._emit("dec", "a")
            self._gen_store(index_var, DataType.BYTE)

            # Test: if A >= 0 (not wrapped), continue
            # After DEC, if result is not negative (i.e., >= 0), continue
            self._emit_label(test_label)
            self._gen_load(index_var)  # A = index
            self._emit("or", "a")  # Set flags
            self._emit("jp", f"p,{loop_label}")  # Jump if positive (bit 7 clear)

            self._emit_label(end_label)
            self.loop_stack.pop()
            return

        # Check for optimized byte loop with constant bound
        if both_bytes and isinstance(stmt.bound, P.NumberLiteral):
            bound_val = number_value(stmt.bound)

            # Initialize index variable
            start_type = self._gen_expr(stmt.start)
            if start_type == DataType.ADDRESS:
                self._emit("ld", "a,l")
            self._gen_store(index_var, DataType.BYTE)

            # Jump to test
            self._emit("jp", test_label)

            # Loop body
            self._emit_label(loop_label)
            for s in body_stmts:
                self._gen_stmt(s)

            # Increment/Decrement
            self._emit_label(incr_label)
            self._gen_load(index_var)  # A = index
            if step_val == 1:
                self._emit("inc", "a")
            elif step_val == -1 or step_val == 0xFF:
                self._emit("dec", "a")
            else:
                self._emit("add", f"a,{self._format_number(step_val & 0xFF)}")
            self._gen_store(index_var, DataType.BYTE)

            # Test condition: compare index with bound
            self._emit_label(test_label)
            self._gen_load(index_var)  # A = index
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
            self._gen_store(index_var, DataType.BYTE)

            # Jump to test
            self._emit("jp", test_label)

            # Loop body
            self._emit_label(loop_label)
            for s in body_stmts:
                self._gen_stmt(s)

            # Increment/Decrement
            self._emit_label(incr_label)
            self._gen_load(index_var)  # A = index
            if step_val == 1:
                self._emit("inc", "a")
            elif step_val == -1 or step_val == 0xFF:
                self._emit("dec", "a")
            else:
                self._emit("add", f"a,{self._format_number(step_val & 0xFF)}")
            self._gen_store(index_var, DataType.BYTE)

            # Test condition: compare index with bound variable
            # Evaluate bound first, then compare with index
            self._emit_label(test_label)
            bound_result = self._gen_expr(stmt.bound)  # A = bound (or HL if ADDRESS)
            if bound_result == DataType.ADDRESS:
                self._emit("ld", "a,l")  # Get low byte if ADDRESS
            self._emit("inc", "a")  # A = bound + 1
            self._emit("ld", "b,a")  # B = bound + 1
            self._gen_load(index_var)  # A = index
            # cp b computes a - b (index - (bound+1)), sets C if index < bound+1
            self._emit("cp", "B")  # Compare index with bound+1
            self._emit("jp", f"C,{loop_label}")  # Continue if index < bound+1 (i.e., index <= bound)

            self._emit_label(end_label)
            self.loop_stack.pop()
            return

        # General case: 16-bit loop (original code)
        # Initialize index variable
        self._gen_expr(stmt.start)
        self._gen_store(index_var, DataType.ADDRESS)

        # Jump to test
        self._emit("jp", test_label)

        # Loop body
        self._emit_label(loop_label)
        for s in body_stmts:
            self._gen_stmt(s)

        # Increment
        self._emit_label(incr_label)
        self._gen_load(index_var)
        if step_val == 1:
            self._emit("inc", "hl")
        elif step_val == -1 or step_val == 0xFFFF:
            self._emit("dec", "hl")
        else:
            self._emit("ld", f"de,{self._format_number(step_val)}")
            self._emit("add", "hl,de")
        self._gen_store(index_var, DataType.ADDRESS)

        # Test condition
        self._emit_label(test_label)
        self._gen_load(index_var)
        self._emit("ex", "de,hl")  # DE = index
        self._gen_expr(stmt.bound)  # HL = bound

        # Compare: if index > bound, exit (for positive step)
        # HL - DE: if negative (carry), index > bound
        self._emit_sub16()

        # If no borrow (NC), bound >= index, continue
        self._emit("jp", f"nc,{loop_label}")

        self._emit_label(end_label)
        self.loop_stack.pop()

    def _gen_do_case(self, stmt) -> None:
        """Generate code for a ``DO CASE selector ... END`` block.

        ``stmt`` is a typed :class:`P.DoCaseBlock` whose ``items`` is
        a flat list of case bodies — one statement per case (the
        legacy frontend pre-grouped multi-statement cases into a
        ``cases: list[list[Stmt]]``, but the typed parser keeps them
        flat). Treat each entry as a single-statement case here; if
        the input has multiple statements per case the PL/M grammar
        nests them inside a ``DO ... END`` block, which appears as
        one :class:`P.DoBlock` item.
        """
        end_label = self._new_label("CASEND")

        cases = list(stmt.items)
        # Create labels for each case
        case_labels = [self._new_label(f"CASE{i}") for i in range(len(cases))]

        # Evaluate selector
        selector_type = self._gen_expr(stmt.selector)

        # Generate jump table
        # For small number of cases, use sequential comparisons
        # For larger, use computed jump

        if len(cases) <= 8:
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
        for i, (case_item, label) in enumerate(zip(cases, case_labels)):
            self._emit_label(label)
            self._gen_stmt(case_item)
            # Only emit JP end_label if last statement doesn't transfer control
            if not self._stmt_transfers_control(case_item):
                self._emit("jp", end_label)

        self._emit_label(end_label)

    def _stmt_transfers_control(self, stmt) -> bool:
        """Check if a typed statement unconditionally transfers control."""
        if stmt is None:
            return False
        if isinstance(stmt, P.GotoStmt):
            return True
        if isinstance(stmt, (P.ReturnStmt, P.ReturnStmtValue)):
            return True
        if isinstance(stmt, P.HaltStmt):
            return True
        if isinstance(stmt, P.LabeledStmt):
            return self._stmt_transfers_control(stmt.stmt)
        if isinstance(stmt, P.DoBlock):
            _, body_stmts = block_items_split(stmt.items)
            if body_stmts:
                return self._stmt_transfers_control(body_stmts[-1])
        return False

    # ========================================================================
    # Expression Code Generation
    # ========================================================================

    def _get_expr_type(self, expr) -> DataType:
        """Determine the type of a typed expression."""
        expr = unwrap_paren(expr)
        if isinstance(expr, P.NumberLiteral):
            return DataType.BYTE if number_value(expr) <= 255 else DataType.ADDRESS
        elif isinstance(expr, P.StringLiteral):
            s = string_value(expr)
            return DataType.BYTE if len(s) == 1 else DataType.ADDRESS
        elif isinstance(expr, P.Identifier):
            name = ident_text(expr.name)
            sym = self.symbols.lookup(name)
            if sym:
                if sym.kind == SymbolKind.PROCEDURE:
                    return sym.return_type or DataType.ADDRESS
                return sym.data_type or DataType.ADDRESS
            return DataType.ADDRESS
        elif isinstance(expr, P.EmbeddedAssign):
            return self._get_expr_type(expr.target)
        elif isinstance(expr, P.BinaryOp):
            op = binop_kind(expr)
            if op in (
                BinaryOpKind.EQ, BinaryOpKind.NE,
                BinaryOpKind.LT, BinaryOpKind.GT,
                BinaryOpKind.LE, BinaryOpKind.GE,
            ):
                return DataType.BYTE
            left_type = self._get_expr_type(expr.left)
            right_type = self._get_expr_type(expr.right)
            if left_type == DataType.BYTE and right_type == DataType.BYTE:
                if op in (
                    BinaryOpKind.ADD, BinaryOpKind.SUB,
                    BinaryOpKind.AND, BinaryOpKind.OR, BinaryOpKind.XOR,
                ):
                    return DataType.BYTE
            return DataType.ADDRESS
        elif isinstance(expr, (P.LocationOf, P.LocationOfList, P.LocationOfString)):
            return DataType.ADDRESS
        elif isinstance(expr, (P.Call, P.CallNoArgs)):
            callee = unwrap_paren(expr.callee)
            if isinstance(callee, P.Identifier):
                name = ident_text(callee.name).upper()
                if name in ('LOW', 'HIGH', 'INPUT', 'ROL', 'ROR'):
                    return DataType.BYTE
                if name == 'MEMORY':
                    return DataType.BYTE
                if name in ('SHL', 'SHR', 'DOUBLE', 'LENGTH', 'LAST', 'SIZE',
                            'STACKPTR', 'TIME', 'CPUTIME'):
                    return DataType.ADDRESS
                sym = self.symbols.lookup(ident_text(callee.name))
                if sym:
                    if sym.kind == SymbolKind.PROCEDURE:
                        return sym.return_type or DataType.ADDRESS
                    if sym.dimension is not None:
                        return sym.data_type or DataType.BYTE
                    return sym.data_type or DataType.ADDRESS
            return DataType.ADDRESS
        elif isinstance(expr, P.UnaryOp):
            # NEG / NOT preserve operand type (LOW/HIGH are now Calls).
            return self._get_expr_type(expr.operand)
        elif isinstance(expr, P.MemberAccess):
            return DataType.BYTE
        return DataType.ADDRESS

    def _is_simple_address_expr(self, expr) -> bool:
        """Check if expression is simple enough to load directly into DE."""
        expr = unwrap_paren(expr)
        if isinstance(expr, P.NumberLiteral):
            return True
        if isinstance(expr, P.Identifier):
            name = ident_text(expr.name)
            if name in self.literal_macros:
                return True
            sym = self.symbols.lookup(name)
            if sym and sym.kind != SymbolKind.PROCEDURE:
                return True
            return False
        if isinstance(expr, P.LocationOf):
            inner = unwrap_paren(expr.operand)
            if isinstance(inner, P.Identifier):
                sym = self.symbols.lookup(ident_text(inner.name))
                if sym and sym.stack_offset is not None:
                    return False
            return True
        return False

    def _gen_simple_to_de(self, expr) -> None:
        """Load a simple address expression directly into DE."""
        expr = unwrap_paren(expr)
        if isinstance(expr, P.NumberLiteral):
            self._emit("ld", f"de,{self._format_number(number_value(expr))}")
        elif isinstance(expr, P.Identifier):
            name = ident_text(expr.name)
            if name.upper() == "MEMORY":
                self.needs_end_symbol = True
                self._emit("ld", "de,__END__")
                return
            if name in self.literal_macros:
                macro_val = self.literal_macros[name]
                try:
                    val = self._parse_plm_number(macro_val)
                    self._emit("ld", f"de,{self._format_number(val)}")
                    return
                except ValueError:
                    name = macro_val
            sym = self.symbols.lookup(name)
            asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(name)
            if sym:
                if sym.dimension:
                    self._emit("ld", f"de,{asm_name}")
                elif sym.data_type == DataType.BYTE:
                    self._emit("ld", f"a,({asm_name})")
                    self._emit("ld", "e,a")
                    self._emit("ld", "d,0")
                else:
                    self._emit("ld", f"de,({asm_name})")
            else:
                self._emit("ld", f"de,{asm_name}")
        elif isinstance(expr, P.LocationOf):
            inner = unwrap_paren(expr.operand)
            if isinstance(inner, P.Identifier):
                name = ident_text(inner.name)
                if name.upper() == "MEMORY":
                    self.needs_end_symbol = True
                    self._emit("ld", "de,__END__")
                    return
                sym = self.symbols.lookup(name)
                if sym and sym.stack_offset is not None:
                    self._gen_expr(expr)
                    self._emit("ex", "de,hl")
                    return
                asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(name)
                self._emit("ld", f"de,{asm_name}")
            else:
                self._gen_expr(expr)
                self._emit("ex", "de,hl")

    def _expr_preserves_de(self, expr) -> bool:
        """Check if evaluating this expression preserves the DE register."""
        expr = unwrap_paren(expr)
        if isinstance(expr, P.NumberLiteral):
            return True
        if isinstance(expr, P.StringLiteral):
            return True
        if isinstance(expr, P.Identifier):
            name = ident_text(expr.name)
            if name in self.literal_macros:
                return True
            sym = self._lookup_symbol(name)
            if sym:
                if sym.kind == SymbolKind.PROCEDURE:
                    return False
                return True
            return True
        if isinstance(expr, P.UnaryOp):
            return self._expr_preserves_de(expr.operand)
        # Binary / Call / subscript / member etc. may touch DE.
        return False

    def _label_reg_need(self, expr) -> int:
        """Label expression with minimum registers needed (Sethi-Ullman)."""
        expr = unwrap_paren(expr)
        if isinstance(expr, (P.NumberLiteral, P.StringLiteral)):
            return 1

        if isinstance(expr, P.Identifier):
            sym = self._lookup_symbol(ident_text(expr.name))
            if sym and sym.kind == SymbolKind.PROCEDURE:
                return 2
            return 1

        if isinstance(expr, P.UnaryOp):
            return self._label_reg_need(expr.operand)

        if isinstance(expr, P.BinaryOp):
            left_need = self._label_reg_need(expr.left)
            right_need = self._label_reg_need(expr.right)
            if left_need == right_need:
                return left_need + 1
            return max(left_need, right_need)

        if isinstance(expr, P.Call):
            # If this is actually a subscript (variable callee), behave like one.
            callee = unwrap_paren(expr.callee)
            if isinstance(callee, P.Identifier):
                sym = self._lookup_symbol(ident_text(callee.name))
                if sym and sym.kind != SymbolKind.PROCEDURE and len(expr.args) == 1:
                    idx = unwrap_paren(expr.args[0])
                    if isinstance(idx, (P.NumberLiteral, P.Identifier)):
                        return 1
                    index_need = self._label_reg_need(expr.args[0])
                    if index_need == 1:
                        return 2
                    return max(1, index_need)
            return 2

        if isinstance(expr, P.CallNoArgs):
            return 2

        if isinstance(expr, P.MemberAccess):
            return self._label_reg_need(expr.base)

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

    def _gen_expr(self, expr) -> DataType:
        """Generate code for a typed expression.

        Result is left in A (for BYTE) or HL (for ADDRESS).
        Returns the type of the expression.
        """
        expr = unwrap_paren(expr)
        # Clear a_has_l for most expression types (embedded assign sets it)
        if not isinstance(expr, (P.EmbeddedAssign, P.Call, P.CallNoArgs)):
            self.a_has_l = False

        if isinstance(expr, P.NumberLiteral):
            self._emit("ld", f"hl,{self._format_number(number_value(expr))}")
            return DataType.ADDRESS

        elif isinstance(expr, P.StringLiteral):
            s = string_value(expr)
            if len(s) == 1:
                self._emit("ld", f"a,{self._format_number(ord(s[0]))}")
                return DataType.BYTE
            label = self._new_string_label()
            self.string_literals.append((label, s))
            self._emit("ld", f"hl,{label}")
            return DataType.ADDRESS

        elif isinstance(expr, P.Identifier):
            return self._gen_load(expr)

        elif isinstance(expr, P.BinaryOp):
            return self._gen_binary(expr)

        elif isinstance(expr, P.UnaryOp):
            return self._gen_unary(expr)

        elif isinstance(expr, P.MemberAccess):
            return self._gen_member(expr)

        elif isinstance(expr, (P.Call, P.CallNoArgs)):
            return self._gen_call_expr(expr)

        elif isinstance(expr, (P.LocationOf, P.LocationOfString, P.LocationOfList)):
            return self._gen_location(expr)

        elif isinstance(expr, P.EmbeddedAssign):
            val_type = self._gen_expr(expr.value)

            target = unwrap_paren(expr.target)
            target_name = ident_text(target.name) if isinstance(target, P.Identifier) else None

            skip_store = False
            if val_type == DataType.BYTE and target_name:
                stmts_to_check: list = []
                if self.current_if_stmt:
                    stmts_to_check.append(self.current_if_stmt.then_stmt)
                    if isinstance(self.current_if_stmt, P.IfStmtElse):
                        stmts_to_check.append(self.current_if_stmt.else_stmt)

                stmts_to_check.extend(self.pending_stmts)

                if stmts_to_check:
                    last_stmt = stmts_to_check[-1]
                    preceding = stmts_to_check[:-1]

                    if self._a_survives_stmts(preceding):
                        if isinstance(last_stmt, P.ReturnStmtValue):
                            val = unwrap_paren(last_stmt.value)
                            if isinstance(val, P.Identifier) and ident_text(val.name) == target_name:
                                skip_store = True
                                self.embedded_assign_target = target_name

            if skip_store:
                pass
            elif val_type == DataType.BYTE:
                store_clobbers_a = True
                if isinstance(target, P.Identifier):
                    sym = self._lookup_symbol(target_name)
                    if sym and sym.data_type == DataType.BYTE:
                        if not sym.based_on and sym.stack_offset is None:
                            store_clobbers_a = False

                if store_clobbers_a:
                    self._emit("ld", "b,a")
                    self._gen_store(target, val_type)
                    self._emit("ld", "a,b")
                else:
                    self._gen_store(target, val_type)
            else:
                target_sym = None
                if isinstance(target, P.Identifier):
                    target_sym = self.symbols.lookup(target_name)

                if target_sym and target_sym.data_type == DataType.BYTE:
                    self._gen_store(target, val_type)
                    self.a_has_l = True
                else:
                    self._emit("push", "hl")
                    self._gen_store(target, val_type)
                    self._emit("pop", "hl")
            return val_type

        return DataType.ADDRESS

    def _gen_load(self, expr) -> DataType:
        """Load a variable value into A/HL. Returns the type."""
        expr = unwrap_paren(expr)
        if isinstance(expr, P.Identifier):
            name = ident_text(expr.name)
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
                    self._emit("ld", f"hl,{self._format_number(val)}")
                    return DataType.ADDRESS
                except ValueError:
                    return self._gen_load(_make_ident(macro_val))

            # Look up symbol in scope hierarchy
            sym = self._lookup_symbol(name)

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
            self._gen_location(_make_location(expr))
            self._emit("ld", "a,(hl)")
            return DataType.BYTE

    def _gen_store(self, expr, val_type: DataType) -> None:
        """Store A/HL to a variable."""
        expr = unwrap_paren(expr)
        if isinstance(expr, P.Identifier):
            name = ident_text(expr.name)

            if name == "STACKPTR":
                self._emit("ld", "sp,hl")
                return

            if name in self.literal_macros:
                macro_val = self.literal_macros[name]
                try:
                    self._parse_plm_number(macro_val)
                except ValueError:
                    self._gen_store(_make_ident(macro_val), val_type)
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

        elif isinstance(expr, P.MemberAccess):
            # Structure member store
            _, member_type = self._get_member_info(expr)
            self._emit("push", "hl")
            self._gen_member_addr(expr)
            self._emit("ex", "de,hl")
            self._emit("pop", "hl")
            if member_type == DataType.ADDRESS:
                self._emit("ex", "de,hl")
                self._emit("ld", "(hl),e")
                self._emit("inc", "hl")
                self._emit("ld", "(hl),d")
            else:
                self._emit("ld", "a,l")
                self._emit("ld", "(de),a")

        elif isinstance(expr, P.Call):
            callee = unwrap_paren(expr.callee)
            # Special built-in assignment targets: OUTPUT(port) = value
            if isinstance(callee, P.Identifier) and ident_text(callee.name).upper() == "OUTPUT":
                port_arg = expr.args[0]
                port_num = self._try_eval_const(port_arg)
                if port_num is not None and 0 <= port_num <= 255:
                    self._emit("ld", "a,l")
                    self._emit("out", f"({self._format_number(port_num)}),a")
                else:
                    self._emit("push", "hl")
                    self._gen_expr(port_arg)
                    self._emit("ld", "c,l")
                    self._emit("pop", "hl")
                    self._emit("ld", "a,l")
                    self._emit("call", "??outp")
                    self.needs_runtime.add("outp")
                return

            # Special built-in: MEMORY(addr) = value
            if (
                isinstance(callee, P.Identifier)
                and ident_text(callee.name).upper() == "MEMORY"
                and len(expr.args) == 1
            ):
                self.needs_end_symbol = True
                addr_arg = expr.args[0]
                addr_val = self._try_eval_const(addr_arg)
                if addr_val is not None:
                    if val_type != DataType.BYTE:
                        self._emit("ld", "a,l")
                    if addr_val == 0:
                        self._emit("ld", "(__END__),a")
                    else:
                        self._emit("ld", f"(__END__+{self._format_number(addr_val)}),a")
                else:
                    if val_type == DataType.BYTE:
                        self._emit("push", "af")
                        self._gen_expr(addr_arg)
                        self._emit("ld", "de,__END__")
                        self._emit("add", "hl,de")
                        self._emit("pop", "af")
                        self._emit("ld", "(hl),a")
                    else:
                        self._emit("push", "hl")
                        self._gen_expr(addr_arg)
                        self._emit("ld", "de,__END__")
                        self._emit("add", "hl,de")
                        self._emit("ex", "de,hl")
                        self._emit("pop", "hl")
                        self._emit("ld", "a,l")
                        self._emit("ld", "(de),a")
                return

            # Array element store via subscript-as-Call: arr(idx) = value
            if isinstance(callee, P.Identifier) and len(expr.args) == 1:
                sym = self.symbols.lookup(ident_text(callee.name))
                if sym and sym.kind != SymbolKind.PROCEDURE:
                    idx_arg = unwrap_paren(expr.args[0])
                    if isinstance(idx_arg, P.NumberLiteral) and not sym.based_on:
                        asm_name = sym.asm_name if sym.asm_name else self._mangle_name(ident_text(callee.name))
                        elem_type = sym.data_type if sym else DataType.BYTE
                        elem_size = 2 if elem_type == DataType.ADDRESS else 1
                        offset = number_value(idx_arg) * elem_size

                        if elem_type == DataType.ADDRESS:
                            if val_type == DataType.BYTE:
                                self._emit("ld", "l,a")
                                self._emit("ld", "h,0")
                            if offset == 0:
                                self._emit("ld", f"({asm_name}),hl")
                            else:
                                self._emit("ld", f"de,{asm_name}+{offset}")
                                self._emit("ex", "de,hl")
                                self._emit("ld", "(hl),e")
                                self._emit("inc", "hl")
                                self._emit("ld", "(hl),d")
                        else:
                            if val_type != DataType.BYTE:
                                self._emit("ld", "a,l")
                            if offset == 0:
                                self._emit("ld", f"({asm_name}),a")
                            else:
                                self._emit("ld", f"({asm_name}+{offset}),a")
                    else:
                        elem_type = sym.data_type if sym else DataType.BYTE
                        if elem_type == DataType.ADDRESS:
                            if val_type == DataType.BYTE:
                                self._emit("ld", "l,a")
                                self._emit("ld", "h,0")
                            self._emit("push", "hl")
                            self._gen_subscript_addr(expr)
                            self._emit("pop", "de")
                            self._emit("ld", "(hl),e")
                            self._emit("inc", "hl")
                            self._emit("ld", "(hl),d")
                        else:
                            if val_type != DataType.BYTE:
                                self._emit("ld", "a,l")
                            self._emit("push", "af")
                            self._gen_subscript_addr(expr)
                            self._emit("pop", "af")
                            self._emit("ld", "(hl),a")
                    return

            # Member array subscript: struct.member(idx)
            if isinstance(callee, P.MemberAccess) and len(expr.args) == 1:
                member_expr = callee
                idx_expr = unwrap_paren(expr.args[0])
                _, member_type = self._get_member_info(member_expr)
                elem_size = 2 if member_type == DataType.ADDRESS else 1

                if member_type == DataType.ADDRESS:
                    if val_type == DataType.BYTE:
                        self._emit("ld", "l,a")
                        self._emit("ld", "h,0")
                    self._emit("push", "hl")
                    self._gen_member_addr(member_expr)
                    if isinstance(idx_expr, P.NumberLiteral):
                        self._emit_add_hl_const(number_value(idx_expr) * elem_size)
                    else:
                        self._emit("push", "hl")
                        idx_type = self._gen_expr(idx_expr)
                        if idx_type == DataType.BYTE:
                            self._emit("ld", "l,a")
                            self._emit("ld", "h,0")
                        self._emit("add", "hl,hl")
                        self._emit("pop", "de")
                        self._emit("add", "hl,de")
                    self._emit("pop", "de")
                    self._emit("ld", "(hl),e")
                    self._emit("inc", "hl")
                    self._emit("ld", "(hl),d")
                else:
                    if val_type != DataType.BYTE:
                        self._emit("ld", "a,l")
                    self._emit("push", "af")
                    self._gen_member_addr(member_expr)
                    if isinstance(idx_expr, P.NumberLiteral):
                        self._emit_add_hl_const(number_value(idx_expr))
                    else:
                        self._emit("push", "hl")
                        idx_type = self._gen_expr(idx_expr)
                        if idx_type == DataType.BYTE:
                            self._emit("ld", "l,a")
                            self._emit("ld", "h,0")
                        self._emit("pop", "de")
                        self._emit("add", "hl,de")
                    self._emit("pop", "af")
                    self._emit("ld", "(hl),a")
                return

            # Unknown call target - fall through to complex store
            self._emit("push", "hl")
            self._gen_location(_make_location(expr))
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
            # Complex store via location-of fallback
            self._emit("push", "hl")
            self._gen_location(_make_location(expr))
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

    def _match_shl_double_8(self, expr):
        """Match the pattern ``SHL(DOUBLE(x), 8)`` and return ``x``.

        This pattern represents ``x * 256`` (shift byte to high position).
        Returns None if pattern doesn't match.
        """
        expr = unwrap_paren(expr)
        if not isinstance(expr, P.Call):
            return None
        callee = unwrap_paren(expr.callee)
        if not isinstance(callee, P.Identifier):
            return None
        if ident_text(callee.name).upper() != 'SHL':
            return None
        if len(expr.args) != 2:
            return None

        shift_count = self._try_eval_const(expr.args[1])
        if shift_count != 8:
            return None

        double_expr = unwrap_paren(expr.args[0])
        if not isinstance(double_expr, P.Call):
            return None
        d_callee = unwrap_paren(double_expr.callee)
        if not isinstance(d_callee, P.Identifier):
            return None
        if ident_text(d_callee.name).upper() != 'DOUBLE':
            return None
        if len(double_expr.args) != 1:
            return None

        inner = double_expr.args[0]
        if self._get_expr_type(inner) != DataType.BYTE:
            return None

        return inner

    def _gen_binary(self, expr) -> DataType:
        """Generate code for a typed binary expression."""
        op = binop_kind(expr)
        left = unwrap_paren(expr.left)
        right = unwrap_paren(expr.right)

        # Special case: SHL(DOUBLE(hi), 8) OR lo -> combine two bytes into address
        if op == BinaryOpKind.OR:
            hi_expr = self._match_shl_double_8(left)
            if hi_expr is not None:
                lo_type = self._get_expr_type(right)
                if lo_type == DataType.BYTE:
                    self._gen_expr(hi_expr)
                    self._emit("ld", "h,a")
                    self._gen_expr(right)
                    self._emit("ld", "l,a")
                    return DataType.ADDRESS

        left_type = self._get_expr_type(left)
        right_type = self._get_expr_type(right)
        both_bytes = (left_type == DataType.BYTE and right_type == DataType.BYTE)

        if op in (BinaryOpKind.EQ, BinaryOpKind.NE) and left_type == DataType.ADDRESS:
            if isinstance(right, P.NumberLiteral) and number_value(right) == 0:
                return self._gen_addr_zero_comparison(left, op)

        if op in (
            BinaryOpKind.EQ, BinaryOpKind.NE, BinaryOpKind.LT,
            BinaryOpKind.GT, BinaryOpKind.LE, BinaryOpKind.GE,
        ):
            self._check_impossible_comparison(left, right, op)

        # Byte comparison with constant: use cp n
        if op in (
            BinaryOpKind.EQ, BinaryOpKind.NE, BinaryOpKind.LT,
            BinaryOpKind.GT, BinaryOpKind.LE, BinaryOpKind.GE,
        ):
            if left_type == DataType.BYTE:
                const_val = None
                if isinstance(right, P.NumberLiteral):
                    val = number_value(right)
                    if val <= 255 or (val & 0xFF00) == 0xFF00:
                        const_val = val & 0xFF
                elif isinstance(right, P.StringLiteral):
                    s = string_value(right)
                    if len(s) == 1:
                        const_val = ord(s[0])

                if const_val is not None:
                    return self._gen_byte_comparison_const(left, op, const_val)
                elif both_bytes:
                    return self._gen_byte_comparison(left, right, op)

        if both_bytes and op in (
            BinaryOpKind.ADD, BinaryOpKind.SUB,
            BinaryOpKind.AND, BinaryOpKind.OR, BinaryOpKind.XOR,
        ):
            return self._gen_byte_binary(left, right, op)

        if (
            op == BinaryOpKind.PLUS
            and left_type == DataType.BYTE
            and isinstance(right, P.NumberLiteral)
            and number_value(right) == 0
        ):
            self._gen_expr(left)
            self._emit("adc", "a,0")
            return DataType.BYTE

        if (
            op == BinaryOpKind.MINUS
            and left_type == DataType.BYTE
            and isinstance(right, P.NumberLiteral)
            and number_value(right) == 0
        ):
            self._gen_expr(left)
            self._emit("sbc", "a,0")
            return DataType.BYTE

        if (
            op == BinaryOpKind.ADD
            and isinstance(right, P.NumberLiteral)
            and left_type == DataType.ADDRESS
        ):
            const_val = number_value(right)
            if 1 <= const_val <= 4:
                self._gen_expr(left)
                for _ in range(const_val):
                    self._emit("inc", "hl")
                return DataType.ADDRESS
            else:
                self._gen_expr(left)
                self._emit("ld", f"de,{self._format_number(const_val)}")
                self._emit("add", "hl,de")
                return DataType.ADDRESS
        elif (
            op == BinaryOpKind.SUB
            and isinstance(right, P.NumberLiteral)
            and left_type == DataType.ADDRESS
        ):
            const_val = number_value(right)
            if 1 <= const_val <= 4:
                self._gen_expr(left)
                for _ in range(const_val):
                    self._emit("dec", "hl")
                return DataType.ADDRESS
            else:
                self._gen_expr(left)
                self._emit("ld", f"de,{self._format_number(const_val)}")
                self._emit_sub16()
                return DataType.ADDRESS

        # Optimize MUL by constant power of 2: shifts instead of runtime call.
        if op == BinaryOpKind.MUL:
            const_val = None
            other_expr = None
            if isinstance(right, P.NumberLiteral):
                const_val = number_value(right)
                other_expr = left
            elif isinstance(left, P.NumberLiteral):
                const_val = number_value(left)
                other_expr = right

            if const_val is not None and const_val > 0:
                if (const_val & (const_val - 1)) == 0:
                    shift_count = 0
                    temp = const_val
                    while temp > 1:
                        temp >>= 1
                        shift_count += 1

                    other_type = self._gen_expr(other_expr)
                    if other_type == DataType.BYTE:
                        self._emit("ld", "l,a")
                        self._emit("ld", "h,0")

                    for _ in range(shift_count):
                        self._emit("add", "hl,hl")

                    return DataType.ADDRESS

        # Fall through to 16-bit operations
        left_need = self._label_reg_need(left)
        right_need = self._label_reg_need(right)

        # Path 1: left is simple AND DE is free
        if self._expr_preserves_de(left) and self.regs.is_free('de'):
            right_result = self._gen_expr(right)
            if right_result == DataType.BYTE:
                self._emit("ld", "e,a")
                self._emit("ld", "d,0")
            else:
                self._emit("ex", "de,hl")
            self.regs.mark_busy('de', 'binary_right_simple')
            left_result = self._gen_expr(left)
            if left_result == DataType.BYTE:
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")
            self.regs.mark_free('de')
            used_general_path = False

        # Path 2: Sethi-Ullman - right needs more registers.
        elif right_need > left_need:
            right_result = self._gen_expr(right)
            if right_result == DataType.BYTE:
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")

            if not self._expr_preserves_de(left):
                self._emit("push", "hl")
                left_result = self._gen_expr(left)
                if left_result == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("pop", "de")
            else:
                self.regs.need_reg('de', 'binary_right_sethi', self._emit)
                self._emit("ex", "de,hl")
                left_result = self._gen_expr(left)
                if left_result == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
            used_general_path = True

        else:
            # Path 3: General - left first.
            left_result = self._gen_expr(left)
            if left_result == DataType.BYTE:
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")

            if not self._expr_preserves_de(right):
                self._emit("push", "hl")
                right_result = self._gen_expr(right)
                if right_result == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("ex", "de,hl")
                self._emit("pop", "hl")
            else:
                self.regs.need_reg('de', 'binary_left', self._emit)
                self._emit("ex", "de,hl")
                right_result = self._gen_expr(right)
                if right_result == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("ex", "de,hl")

            used_general_path = True

        if op == BinaryOpKind.ADD:
            self._emit("add", "hl,de")

        elif op == BinaryOpKind.SUB:
            self._emit_sub16()

        elif op == BinaryOpKind.MUL:
            self.needs_runtime.add("mul16")
            self._emit("call", "??mul16")

        elif op == BinaryOpKind.DIV:
            self.needs_runtime.add("div16")
            self._emit("call", "??div16")

        elif op == BinaryOpKind.MOD:
            self.needs_runtime.add("mod16")
            self._emit("call", "??mod16")

        elif op == BinaryOpKind.AND:
            self._emit("ld", "a,l")
            self._emit("and", "e")
            self._emit("ld", "l,a")
            self._emit("ld", "a,h")
            self._emit("and", "d")
            self._emit("ld", "h,a")

        elif op == BinaryOpKind.OR:
            self._emit("ld", "a,l")
            self._emit("or", "e")
            self._emit("ld", "l,a")
            self._emit("ld", "a,h")
            self._emit("or", "d")
            self._emit("ld", "h,a")

        elif op == BinaryOpKind.XOR:
            self._emit("ld", "a,l")
            self._emit("xor", "e")
            self._emit("ld", "l,a")
            self._emit("ld", "a,h")
            self._emit("xor", "d")
            self._emit("ld", "h,a")

        elif op in (
            BinaryOpKind.EQ, BinaryOpKind.NE, BinaryOpKind.LT,
            BinaryOpKind.GT, BinaryOpKind.LE, BinaryOpKind.GE,
        ):
            if used_general_path:
                self.regs.release_reg('de', self._emit)
            return self._gen_comparison(op)

        elif op == BinaryOpKind.PLUS:
            self._emit("ld", "a,l")
            self._emit("adc", "a,e")
            self._emit("ld", "l,a")
            self._emit("ld", "a,h")
            self._emit("adc", "a,d")
            self._emit("ld", "h,a")

        elif op == BinaryOpKind.MINUS:
            self._emit("ld", "a,l")
            self._emit("sbc", "a,e")
            self._emit("ld", "l,a")
            self._emit("ld", "a,h")
            self._emit("sbc", "a,d")
            self._emit("ld", "h,a")

        if used_general_path:
            self.regs.release_reg('de', self._emit)

        return DataType.ADDRESS

    def _gen_comparison(self, op: BinaryOpKind) -> DataType:
        """Generate code for comparison. HL=left, DE=right. Result in A (0 or 0FFH)."""
        true_label = self._new_label("TRUE")
        false_label = self._new_label("FALSE")
        end_label = self._new_label("CMP")

        self._emit_sub16()

        if op == BinaryOpKind.EQ:
            self._emit("ld", "a,l")
            self._emit("or", "h")
            self._emit("jp", f"z,{true_label}")
        elif op == BinaryOpKind.NE:
            self._emit("ld", "a,l")
            self._emit("or", "h")
            self._emit("jp", f"nz,{true_label}")
        elif op == BinaryOpKind.LT:
            self._emit("jp", f"c,{true_label}")
        elif op == BinaryOpKind.GE:
            self._emit("jp", f"nc,{true_label}")
        elif op == BinaryOpKind.GT:
            self._emit("jp", f"c,{false_label}")
            self._emit("ld", "a,l")
            self._emit("or", "h")
            self._emit("jp", f"nz,{true_label}")
        elif op == BinaryOpKind.LE:
            self._emit("jp", f"c,{true_label}")
            self._emit("ld", "a,l")
            self._emit("or", "h")
            self._emit("jp", f"z,{true_label}")

        self._emit_label(false_label)
        self._emit("xor", "a")
        self._emit("jp", end_label)

        self._emit_label(true_label)
        self._emit("ld", "a,0ffh")

        self._emit_label(end_label)
        return DataType.BYTE

    def _gen_addr_zero_comparison(self, left, op: BinaryOpKind) -> DataType:
        """Generate optimized ADDRESS comparison with 0 using OR."""
        self._gen_expr(left)
        self._emit("ld", "a,l")
        self._emit("or", "h")

        true_label = self._new_label("TRUE")
        end_label = self._new_label("CMP")

        if op == BinaryOpKind.EQ:
            self._emit("jp", f"z,{true_label}")
        elif op == BinaryOpKind.NE:
            self._emit("jp", f"nz,{true_label}")

        self._emit("xor", "a")
        self._emit("jp", end_label)

        self._emit_label(true_label)
        self._emit("ld", "a,0ffh")

        self._emit_label(end_label)
        return DataType.BYTE

    def _gen_byte_comparison_const(self, left, op: BinaryOpKind, const_val: int) -> DataType:
        """Generate optimized byte comparison with constant using cp n."""
        left_type = self._gen_expr(left)
        if left_type != DataType.BYTE:
            self._emit("ld", "a,l")

        self._emit("cp", self._format_number(const_val))

        true_label = self._new_label("TRUE")
        end_label = self._new_label("CMP")

        if op == BinaryOpKind.EQ:
            self._emit("jp", f"z,{true_label}")
        elif op == BinaryOpKind.NE:
            self._emit("jp", f"nz,{true_label}")
        elif op == BinaryOpKind.LT:
            self._emit("jp", f"c,{true_label}")
        elif op == BinaryOpKind.GE:
            self._emit("jp", f"nc,{true_label}")
        elif op == BinaryOpKind.GT:
            self._emit("jp", f"c,{end_label}")
            self._emit("jp", f"z,{end_label}")
            self._emit("jp", true_label)
        elif op == BinaryOpKind.LE:
            self._emit("jp", f"c,{true_label}")
            self._emit("jp", f"z,{true_label}")

        self._emit("xor", "a")
        self._emit("jp", end_label)

        self._emit_label(true_label)
        self._emit("ld", "a,0ffh")

        self._emit_label(end_label)
        return DataType.BYTE

    def _gen_byte_comparison(self, left, right, op: BinaryOpKind) -> DataType:
        """Generate optimized byte comparison between two byte values."""
        self._gen_expr(right)
        self._emit("ld", "b,a")

        self._gen_expr(left)
        self._emit("sub", "b")

        true_label = self._new_label("TRUE")
        end_label = self._new_label("CMP")

        if op == BinaryOpKind.EQ:
            self._emit("jp", f"z,{true_label}")
        elif op == BinaryOpKind.NE:
            self._emit("jp", f"nz,{true_label}")
        elif op == BinaryOpKind.LT:
            self._emit("jp", f"c,{true_label}")
        elif op == BinaryOpKind.GE:
            self._emit("jp", f"nc,{true_label}")
        elif op == BinaryOpKind.GT:
            self._emit("jp", f"c,{end_label}")
            self._emit("jp", f"z,{end_label}")
            self._emit("jp", true_label)
        elif op == BinaryOpKind.LE:
            self._emit("jp", f"c,{true_label}")
            self._emit("jp", f"z,{true_label}")

        self._emit("xor", "a")
        self._emit("jp", end_label)

        self._emit_label(true_label)
        self._emit("ld", "a,0ffh")

        self._emit_label(end_label)
        return DataType.BYTE

    def _gen_byte_binary(self, left, right, op: BinaryOpKind) -> DataType:
        """Generate optimized byte arithmetic/logical operation."""
        right_const = self._get_const_byte_value(right)
        if right_const is not None:
            self._gen_expr_to_a(left)
            const = self._format_number(right_const)
            if op == BinaryOpKind.ADD:
                self._emit("add", f"a,{const}")
            elif op == BinaryOpKind.SUB:
                self._emit("sub", const)
            elif op == BinaryOpKind.AND:
                self._emit("and", const)
            elif op == BinaryOpKind.OR:
                self._emit("or", const)
            elif op == BinaryOpKind.XOR:
                self._emit("xor", const)
            return DataType.BYTE

        left_const = self._get_const_byte_value(left)
        if op == BinaryOpKind.SUB and left_const is not None:
            if left_const == 1:
                self._gen_expr_to_a(right)
                self._emit("xor", "1")
            else:
                self._gen_expr_to_a(right)
                self._emit("cpl")
                self._emit("inc", "a")
                self._emit("add", f"a,{self._format_number(left_const)}")
            return DataType.BYTE

        if op == BinaryOpKind.SUB:
            self._gen_expr_to_a(right)
            self._emit("ld", "b,a")
            self._gen_expr_to_a(left)
            self._emit("sub", "b")
            return DataType.BYTE

        self._gen_expr_to_a(left)
        self._emit("ld", "b,a")

        self._gen_expr_to_a(right)

        if op == BinaryOpKind.ADD:
            self._emit("add", "a,b")
        elif op == BinaryOpKind.AND:
            self._emit("and", "b")
        elif op == BinaryOpKind.OR:
            self._emit("or", "b")
        elif op == BinaryOpKind.XOR:
            self._emit("xor", "b")

        return DataType.BYTE

    def _gen_expr_to_a(self, expr) -> None:
        """Generate code to load an expression into A (for byte operations)."""
        expr = unwrap_paren(expr)
        const_val = self._get_const_byte_value(expr)
        if const_val is not None:
            self._emit("ld", f"a,{self._format_number(const_val)}")
        elif isinstance(expr, P.NumberLiteral):
            self._emit("ld", f"a,{self._format_number(number_value(expr) & 0xFF)}")
        else:
            result_type = self._gen_expr(expr)
            if result_type == DataType.ADDRESS:
                self._emit("ld", "a,l")

    def _gen_unary(self, expr) -> DataType:
        """Generate code for a typed unary expression."""
        kind = unop_kind(expr)
        operand_type = self._gen_expr(expr.operand)

        if kind == UnaryOpKind.NEG:
            if operand_type == DataType.BYTE:
                self._emit("cpl")
                self._emit("inc", "a")
                return DataType.BYTE
            else:
                self._emit("ld", "a,l")
                self._emit("cpl")
                self._emit("ld", "l,a")
                self._emit("ld", "a,h")
                self._emit("cpl")
                self._emit("ld", "h,a")
                self._emit("inc", "hl")
                return DataType.ADDRESS

        elif kind == UnaryOpKind.NOT:
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

        return DataType.ADDRESS

    # Built-in functions that might be parsed as subscripts
    BUILTIN_FUNCS = {'LENGTH', 'LAST', 'SIZE', 'HIGH', 'LOW', 'DOUBLE', 'ROL', 'ROR',
                     'SHL', 'SHR', 'SCL', 'SCR', 'INPUT', 'OUTPUT', 'TIME', 'MOVE',
                     'CPUTIME', 'MEMORY', 'STACKPTR', 'DEC'}

    def _gen_subscript(self, expr) -> DataType:
        """Generate code for array subscript (typed ``P.Call`` form) — load value."""
        base = unwrap_paren(expr.callee)
        index = expr.args[0]

        if isinstance(base, P.Identifier) and ident_text(base.name).upper() in self.BUILTIN_FUNCS:
            return self._gen_call_expr(expr)

        elem_type = DataType.BYTE
        if isinstance(base, P.Identifier):
            sym = self.symbols.lookup(ident_text(base.name))
            if sym and sym.data_type == DataType.ADDRESS:
                elem_type = DataType.ADDRESS

        self._gen_subscript_addr(expr)

        if elem_type == DataType.ADDRESS:
            self._emit("ld", "e,(hl)")
            self._emit("inc", "hl")
            self._emit("ld", "d,(hl)")
            self._emit("ex", "de,hl")
            return DataType.ADDRESS
        else:
            self._emit("ld", "a,(hl)")
            return DataType.BYTE

    def _gen_subscript_addr(self, expr) -> None:
        """Generate code to compute address of an array element.

        Accepts a typed :class:`P.Call` (which is how the grammar
        models ``arr(idx)``). The callee is the array reference and
        ``args[0]`` is the index expression.
        """
        base = unwrap_paren(expr.callee)
        index = unwrap_paren(expr.args[0])

        if isinstance(base, P.Identifier) and ident_text(base.name).upper() in self.BUILTIN_FUNCS:
            self._gen_call_expr(expr)
            return

        elem_size = 1
        if isinstance(base, P.Identifier):
            sym = self.symbols.lookup(ident_text(base.name))
            if sym:
                if sym.struct_members:
                    elem_size = 0
                    for member in sym.struct_members:
                        member_size = 2 if member.data_type == DataType.ADDRESS else 1
                        if member.dimension:
                            member_size *= member.dimension
                        elem_size += member_size
                elif sym.data_type == DataType.ADDRESS:
                    elem_size = 2

        # Constant folding: label + constant.
        if isinstance(base, P.Identifier) and isinstance(index, P.NumberLiteral):
            sym = self.symbols.lookup(ident_text(base.name))
            if sym and not sym.based_on:
                asm_name = sym.asm_name if sym.asm_name else self._mangle_name(ident_text(base.name))
                offset = number_value(index) * elem_size
                if offset == 0:
                    self._emit("ld", f"hl,{asm_name}")
                else:
                    self._emit("ld", f"hl,{asm_name}+{offset}")
                return

        # Optimised BYTE-index path with identifier base.
        if not isinstance(index, P.NumberLiteral):
            idx_type = self._get_expr_type(index)
            if idx_type == DataType.BYTE and elem_size == 1 and isinstance(base, P.Identifier):
                self._gen_expr(index)
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")
                sym = self.symbols.lookup(ident_text(base.name))
                if sym and sym.based_on:
                    base_sym = self.symbols.lookup(sym.based_on)
                    base_asm_name = base_sym.asm_name if base_sym and base_sym.asm_name else self._mangle_name(sym.based_on)
                    self._emit("ld", f"de,({base_asm_name})")
                else:
                    asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(ident_text(base.name))
                    self._emit("ld", f"de,{asm_name}")
                self._emit("add", "hl,de")
                return

        # Get base address.
        if isinstance(base, P.Identifier):
            sym = self.symbols.lookup(ident_text(base.name))
            if sym and sym.based_on:
                base_sym = self.symbols.lookup(sym.based_on)
                base_asm_name = base_sym.asm_name if base_sym and base_sym.asm_name else self._mangle_name(sym.based_on)
                self._emit("ld", f"hl,({base_asm_name})")
            else:
                asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(ident_text(base.name))
                self._emit("ld", f"hl,{asm_name}")
        else:
            self._gen_expr(base)

        if isinstance(index, P.NumberLiteral):
            offset = number_value(index) * elem_size
            self._emit_add_hl_const(offset)
        else:
            self.regs.need_reg('de', 'subscript_base', self._emit)
            self._emit("ex", "de,hl")

            result_type = self._gen_expr(index)

            if result_type == DataType.BYTE:
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")

            if elem_size > 1:
                if (elem_size & (elem_size - 1)) == 0:
                    temp = elem_size
                    while temp > 1:
                        self._emit("add", "hl,hl")
                        temp >>= 1
                else:
                    self._emit("push", "de")
                    self._emit("ld", f"de,{elem_size}")
                    self._emit("call", "??mul16")
                    self._emit("pop", "de")
                    self.needs_runtime.add("mul16")

            self._emit("add", "hl,de")
            self.regs.release_reg('de', self._emit)

    def _get_member_info(self, expr) -> tuple[int, DataType]:
        """Get offset and type for a typed ``MemberAccess`` node."""
        offset = 0
        member_type = DataType.BYTE

        base = unwrap_paren(expr.base)
        member_name = ident_text(expr.member)

        sym = None
        if isinstance(base, P.Identifier):
            sym = self.symbols.lookup(ident_text(base.name))
        elif isinstance(base, P.Call):
            callee = unwrap_paren(base.callee)
            if isinstance(callee, P.Identifier):
                sym = self.symbols.lookup(ident_text(callee.name))

        if sym and sym.struct_members:
            for member in sym.struct_members:
                if member.name == member_name:
                    member_type = member.data_type
                    break
                member_size = 2 if member.data_type == DataType.ADDRESS else 1
                if member.dimension:
                    member_size *= member.dimension
                offset += member_size

        return offset, member_type

    def _gen_member(self, expr) -> DataType:
        """Generate code for structure member access — load value."""
        _, member_type = self._get_member_info(expr)
        self._gen_member_addr(expr)

        if member_type == DataType.ADDRESS:
            self._emit("ld", "e,(hl)")
            self._emit("inc", "hl")
            self._emit("ld", "d,(hl)")
            self._emit("ex", "de,hl")
            return DataType.ADDRESS
        else:
            self._emit("ld", "a,(hl)")
            self._emit("ld", "l,a")
            self._emit("ld", "h,0")
            return DataType.BYTE

    def _gen_member_addr(self, expr) -> None:
        """Generate code to compute address of structure member."""
        base = unwrap_paren(expr.base)
        if isinstance(base, P.Identifier):
            name = ident_text(base.name)
            sym = self._lookup_symbol(name)

            if sym and sym.struct_members:
                if sym.based_on:
                    base_sym = self.symbols.lookup(sym.based_on)
                    base_asm_name = base_sym.asm_name if base_sym and base_sym.asm_name else self._mangle_name(sym.based_on)
                    self._emit("ld", f"hl,({base_asm_name})")
                else:
                    asm_name = sym.asm_name or name
                    self._emit("ld", f"hl,{asm_name}")
            elif sym and sym.based_on:
                base_sym = self.symbols.lookup(sym.based_on)
                base_asm_name = base_sym.asm_name if base_sym and base_sym.asm_name else self._mangle_name(sym.based_on)
                self._emit("ld", f"hl,({base_asm_name})")
            else:
                self._gen_expr(base)
        elif isinstance(base, P.Call):
            callee = unwrap_paren(base.callee)
            if isinstance(callee, P.Identifier):
                name = ident_text(callee.name)
                sym = self._lookup_symbol(name)
                if sym and sym.kind in (SymbolKind.VARIABLE, SymbolKind.PARAMETER) and len(base.args) == 1:
                    self._gen_subscript_addr(base)
                else:
                    self._gen_expr(base)
            else:
                self._gen_expr(base)
        else:
            self._gen_expr(base)

        offset, _ = self._get_member_info(expr)
        self._emit_add_hl_const(offset)

    def _gen_call_expr(self, expr) -> DataType:
        """Generate code for a call expression or array subscript.

        ``expr`` is a typed :class:`P.Call` (callee + args) or
        :class:`P.CallNoArgs` (parameterless). Since the grammar can't
        distinguish ``arr(idx)`` from ``func(arg)`` syntactically, we
        decide here by looking up the callee's symbol kind:

        * unresolved or :data:`SymbolKind.PROCEDURE` -> a procedure call;
        * :data:`SymbolKind.VARIABLE` / :data:`SymbolKind.PARAMETER`
          with a single arg -> array subscript (delegate to
          :meth:`_gen_subscript`).
        """
        callee = unwrap_paren(expr.callee)
        args = list(expr.args) if isinstance(expr, P.Call) else []

        # Handle built-in functions
        if isinstance(callee, P.Identifier):
            name = ident_text(callee.name)
            result = self._gen_builtin(name, args)
            if result is not None:
                return result

            sym = self._lookup_symbol(name)

            # Variable callee with a single arg -> array subscript.
            if (
                sym
                and sym.kind in (SymbolKind.VARIABLE, SymbolKind.PARAMETER)
                and len(args) == 1
            ):
                return self._gen_subscript(expr)

        # Member array subscript: struct.member(idx)
        if isinstance(callee, P.MemberAccess) and len(args) == 1:
            member_expr = callee
            idx_expr = unwrap_paren(args[0])

            self._gen_member_addr(member_expr)
            _, member_type = self._get_member_info(member_expr)
            elem_size = 2 if member_type == DataType.ADDRESS else 1

            if isinstance(idx_expr, P.NumberLiteral):
                offset = number_value(idx_expr) * elem_size
                self._emit_add_hl_const(offset)
            else:
                self.regs.need_reg('de', 'member_subscript_base', self._emit)
                self._emit("ex", "de,hl")
                idx_type = self._gen_expr(idx_expr)
                if idx_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                if elem_size == 2:
                    self._emit("add", "hl,hl")
                self._emit("add", "hl,de")
                self.regs.release_reg('de', self._emit)

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
        sym = None
        call_name = None
        full_callee_name = None
        name = None
        if isinstance(callee, P.Identifier):
            name = ident_text(callee.name)
            sym = self._lookup_symbol(name)
            call_name = sym.asm_name if sym and sym.asm_name else name
            if sym:
                full_callee_name = sym.name

            # CP/M BDOS optimisation: MON1/MON2(func, arg).
            if name.upper() in ('MON1', 'MON2') and len(args) == 2:
                func_arg, addr_arg = args
                func_num = self._get_const_byte_value(func_arg)
                if func_num is not None and func_num <= 255:
                    self._emit("ld", f"c,{self._format_number(func_num)}")
                    addr_type = self._gen_expr(addr_arg)
                    if addr_type == DataType.BYTE:
                        self._emit("ld", "e,a")
                    else:
                        self._emit("ex", "de,hl")
                    self._emit("call", "5")
                    return DataType.BYTE if name.upper() == 'MON2' else DataType.ADDRESS

        use_stack = True
        if sym and sym.kind == SymbolKind.PROCEDURE and not sym.is_reentrant and not sym.is_external:
            use_stack = False

        if use_stack:
            for arg in args:
                arg_type = self._gen_expr(arg)
                if arg_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("push", "hl")
        else:
            last_param_idx = len(args) - 1
            uses_reg = sym.uses_reg_param and len(args) > 0

            for i, arg in enumerate(args):
                if sym and i < len(sym.params):
                    param_name = sym.params[i]
                    param_type = sym.param_types[i] if i < len(sym.param_types) else DataType.ADDRESS
                    is_last = (i == last_param_idx)

                    if is_last and uses_reg:
                        if param_type == DataType.BYTE:
                            const = self._get_const_byte_value(arg)
                            if const is not None:
                                self._emit("ld", f"a,{self._format_number(const)}")
                                continue
                        arg_type = self._gen_expr(arg)
                        if param_type == DataType.BYTE and arg_type == DataType.ADDRESS:
                            self._emit("ld", "a,l")
                        elif param_type == DataType.ADDRESS and arg_type == DataType.BYTE:
                            self._emit("ld", "l,a")
                            self._emit("ld", "h,0")
                        continue

                    param_asm = None
                    if (hasattr(self, 'storage_labels')
                        and full_callee_name in self.storage_labels
                        and param_name in self.storage_labels[full_callee_name]):
                        param_asm = self.storage_labels[full_callee_name][param_name]
                    else:
                        proc_base = sym.asm_name if sym.asm_name else name or ""
                        if proc_base.startswith('@'):
                            proc_base = proc_base[1:]
                        param_asm = f"@{proc_base}${self._mangle_name(param_name)}"

                    if param_type == DataType.BYTE:
                        const = self._get_const_byte_value(arg)
                        if const is not None:
                            self._emit("ld", f"a,{self._format_number(const)}")
                            self._emit("ld", f"({param_asm}),a")
                            continue

                    arg_type = self._gen_expr(arg)
                    if param_type == DataType.BYTE or arg_type == DataType.BYTE:
                        if arg_type == DataType.ADDRESS:
                            self._emit("ld", "a,l")
                        self._emit("ld", f"({param_asm}),a")
                    else:
                        self._emit("ld", f"({param_asm}),hl")

        if isinstance(callee, P.Identifier):
            self._emit("call", call_name)
        else:
            self._gen_expr(callee)
            self._emit("jp", "(hl)")

        if use_stack and args:
            for _ in args:
                self._emit("pop", "de")

        return sym.return_type if sym and sym.return_type else DataType.ADDRESS

    def _gen_builtin(self, name: str, args) -> DataType | None:
        """Generate code for built-in function. Returns type if handled, None otherwise.

        ``name`` is the un-mangled identifier text; built-in names are
        matched case-insensitively. ``args`` is the raw typed argument
        list from the call expression.
        """
        name = name.upper()

        if name == "INPUT":
            if args:
                arg = args[0]
                port_num = self._try_eval_const(arg)
                if port_num is not None:
                    self._emit("in", f"a,({self._format_number(port_num)})")
                else:
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
            shift_count = self._try_eval_const(args[1])

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
            shift_count = self._try_eval_const(args[1])

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
                    elif remaining <= 4:
                        # Short shift: SRL doesn't need carry clearing.
                        self._emit("ld", "a,h")
                        for _ in range(remaining):
                            self._emit("srl", "a")
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
                    # Small shifts: inline the loop (SRL/RR — 2 insns per shift).
                    for _ in range(shift_count):
                        self._emit("srl", "h")  # H >>= 1, bit 0 -> carry
                        self._emit("rr", "l")   # L = (carry << 7) | (L >> 1)
                else:
                    # For 4-6 shifts, use a counted loop with DJNZ.
                    self._emit("ld", f"b,{shift_count}")
                    shift_loop = self._new_label("SHR")
                    self._emit_label(shift_loop)
                    self._emit("srl", "h")  # H >>= 1, bit 0 -> carry
                    self._emit("rr", "l")   # L = (carry << 7) | (L >> 1)
                    self._emit("djnz", shift_loop)
                return DataType.ADDRESS

            # Variable shift - use loop
            arg_type = self._gen_expr(args[0])
            if arg_type == DataType.BYTE:
                # BYTE value is in A, move to HL
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")
            self._emit("push", "hl")
            count_type = self._gen_expr(args[1])
            # Variable shift: B holds the count, DJNZ drives the loop.
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
            if args:
                arg0 = unwrap_paren(args[0])
                if isinstance(arg0, P.Identifier):
                    sym = self.symbols.lookup(ident_text(arg0.name))
                    if sym and sym.dimension:
                        self._emit("ld", f"hl,{sym.dimension}")
                        return DataType.ADDRESS
            self._emit("ld", "hl,0")
            return DataType.ADDRESS

        if name == "LAST":
            if args:
                arg0 = unwrap_paren(args[0])
                if isinstance(arg0, P.Identifier):
                    sym = self.symbols.lookup(ident_text(arg0.name))
                    if sym and sym.dimension:
                        self._emit("ld", f"hl,{sym.dimension - 1}")
                        return DataType.ADDRESS
            self._emit("ld", "hl,0")
            return DataType.ADDRESS

        if name == "SIZE":
            if args:
                arg0 = unwrap_paren(args[0])
                if isinstance(arg0, P.Identifier):
                    sym = self.symbols.lookup(ident_text(arg0.name))
                    if sym:
                        self._emit("ld", f"hl,{sym.size}")
                        return DataType.ADDRESS
            self._emit("ld", "hl,0")
            return DataType.ADDRESS

        if name == "MEMORY":
            self.needs_end_symbol = True
            arg0 = unwrap_paren(args[0])
            if isinstance(arg0, P.NumberLiteral) and number_value(arg0) == 0:
                self._emit("ld", "hl,__END__")
            else:
                self._gen_expr(args[0])
                self._emit("ld", "de,__END__")
                self._emit("add", "hl,de")
            self._emit("ld", "a,(hl)")
            return DataType.BYTE

        if name == "MOVE":
            arg0 = unwrap_paren(args[0])
            count_const = number_value(arg0) if isinstance(arg0, P.NumberLiteral) else None

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

    def _gen_location(self, expr) -> DataType:
        """Generate code to load address of a typed location expression.

        ``expr`` is one of:

        * :class:`P.LocationOf` — ``.expr``,
        * :class:`P.LocationOfString` — ``.'string literal'``,
        * :class:`P.LocationOfList` — ``.(a, b, c)`` const list.

        The :class:`P.LocationOf` operand may itself be a typed
        :class:`P.Identifier`, :class:`P.MemberAccess`, or a
        :class:`P.Call` (which in PL/M's grammar covers both array
        subscripts and procedure calls; we disambiguate via the symbol
        table).
        """
        if isinstance(expr, P.LocationOfString):
            # ``expr.value`` is a ``STRING`` Token whose text retains the
            # surrounding quotes; strip them and decode ``''`` escapes.
            raw = expr.value.text
            if raw.startswith("'") and raw.endswith("'"):
                raw = raw[1:-1]
            s = raw.replace("''", "'")
            label = self._new_string_label()
            self.string_literals.append((label, s))
            self._emit("ld", f"hl,{label}")
            return DataType.ADDRESS

        if isinstance(expr, P.LocationOfList):
            label = self._new_label("DATA")
            self.data_segment.append(AsmLine(label=label))
            for val in expr.values or []:
                v = unwrap_paren(val)
                if isinstance(v, P.NumberLiteral):
                    self.data_segment.append(
                        AsmLine(opcode="db", operands=self._format_number(number_value(v)))
                    )
                elif isinstance(v, P.StringLiteral):
                    self.data_segment.append(
                        AsmLine(opcode="db", operands=self._escape_string(string_value(v)))
                    )
            self._emit("ld", f"hl,{label}")
            return DataType.ADDRESS

        # P.LocationOf
        operand = unwrap_paren(expr.operand)
        if isinstance(operand, P.Identifier):
            name = ident_text(operand.name)

            if name.upper() == "MEMORY":
                self.needs_end_symbol = True
                self._emit("ld", "hl,__END__")
                return DataType.ADDRESS

            if name in self.literal_macros:
                macro_val = self.literal_macros[name]
                try:
                    val = self._parse_plm_number(macro_val)
                    self._emit("ld", f"hl,{self._format_number(val)}")
                    return DataType.ADDRESS
                except ValueError:
                    return self._gen_location(_make_location(_make_ident(macro_val)))

            sym = self.symbols.lookup(name)

            if sym and sym.stack_offset is not None:
                self._emit("push", "ix")
                self._emit("pop", "hl")
                if sym.stack_offset != 0:
                    self._emit("ld", f"de,{sym.stack_offset}")
                    self._emit("add", "hl,de")
            elif sym and sym.based_on:
                base_sym = self.symbols.lookup(sym.based_on)
                base_asm_name = base_sym.asm_name if base_sym and base_sym.asm_name else self._mangle_name(sym.based_on)
                self._emit("ld", f"hl,({base_asm_name})")
            else:
                asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(name)
                self._emit("ld", f"hl,{asm_name}")
        elif isinstance(operand, P.MemberAccess):
            self._gen_member_addr(operand)
        elif isinstance(operand, P.Call):
            callee = unwrap_paren(operand.callee)
            if isinstance(callee, P.Identifier) and len(operand.args) == 1:
                sym = self.symbols.lookup(ident_text(callee.name))
                if sym and sym.kind != SymbolKind.PROCEDURE:
                    self._gen_subscript_addr(operand)
                    return DataType.ADDRESS
            if isinstance(callee, P.MemberAccess) and len(operand.args) == 1:
                member_expr = callee
                idx_expr = unwrap_paren(operand.args[0])

                self._gen_member_addr(member_expr)
                _, member_type = self._get_member_info(member_expr)
                elem_size = 2 if member_type == DataType.ADDRESS else 1

                if isinstance(idx_expr, P.NumberLiteral):
                    offset = number_value(idx_expr) * elem_size
                    self._emit_add_hl_const(offset)
                else:
                    self.regs.need_reg('de', 'member_subscript_addr', self._emit)
                    self._emit("ex", "de,hl")
                    idx_type = self._gen_expr(idx_expr)
                    if idx_type == DataType.BYTE:
                        self._emit("ld", "l,a")
                        self._emit("ld", "h,0")
                    if elem_size == 2:
                        self._emit("add", "hl,hl")
                    self._emit("add", "hl,de")
                    self.regs.release_reg('de', self._emit)
                return DataType.ADDRESS
            self._gen_expr(operand)
        else:
            self._gen_expr(operand)
        return DataType.ADDRESS


def generate(module) -> str:
    """Convenience function to generate code from a typed :class:`P.Module`."""
    gen = CodeGenerator()
    return gen.generate(module)
