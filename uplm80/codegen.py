"""
Code Generator for PL/M-80.

Generates Z80 assembly code from the optimized AST.
Outputs MACRO-80 compatible .MAC files.
"""

import re
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
    iter_block_proc_decls,
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
    make_binary,
    make_number_literal,
    make_unary,
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
from .runtime import get_runtime_library, plm_div, plm_mod
from .plm_types import (
    BYTE_BUILTINS,
    PATTERN_TYPED_BUILTINS,
    eval_typed,
    is_derived,
    is_double_call,
    typed_const,
)


# Map ast_view's DataType (used by typed AST helpers) to the legacy
# ast_nodes.DataType still used by the symbol table and unmigrated
# codegen paths. Same enum names, different identity.
_VIEW_DT_TO_LEGACY = {
    ViewDataType.BYTE: DataType.BYTE,
    ViewDataType.ADDRESS: DataType.ADDRESS,
    ViewDataType.LABEL: DataType.LABEL,
    ViewDataType.PROCEDURE: DataType.PROCEDURE,
}

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
    MPM = auto()   # MP/M relocatable program (.PRL/.SPR/.RSP)


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

    # Per register, the claims not yet released and whether each spilled
    # the one before it. Claims nest (a subscript inside a subscript), and
    # a release restores only what its own claim spilled.
    claims: dict[str, list[tuple[bool, str]]] = field(default_factory=dict)

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

        spilled = desc.state == RegState.BUSY
        self.claims.setdefault(reg, []).append((spilled, desc.owner))
        if spilled:
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
        claims = self.claims.get(reg)
        spilled, outer_owner = claims.pop() if claims else (False, "")

        # Restore what this claim spilled, and only that: an inner claim's
        # release used to pop whatever an OUTER claim had spilled, from
        # under the values pushed since -- `aw(aw(aw(i) AND 7) AND 7)'
        # added a stale DE in place of the array's base.
        if spilled and self.spill_stack and self.spill_stack[-1] == reg:
            pop_reg = 'af' if reg == 'a' else reg
            emit_fn("pop", pop_reg)
            self.spill_stack.pop()
            self.stats['restores'] = self.stats.get('restores', 0) + 1
            desc.state = RegState.BUSY
            desc.owner = outer_owner
            desc.spill_depth = 0
            return

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
        self.claims.clear()
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

    # Page-zero addresses a hosted program reaches directly.  Under CP/M they
    # are fixed locations and the literal is emitted.  Under MP/M page zero
    # belongs to the memory segment the process was loaded into, so the same
    # address has to reach the linker as a symbol: only a resolved symbol
    # reference ends up in the .PRL relocation bitmap, and an unrelocated
    # "call 5" would call into whatever happens to live at absolute 0005H.
    # The names match DRI's X0100.ASM, which is how PL/M-80 itself got these
    # relocated.
    # The names carry the compiler's own ``??`` prefix.  A PL/M-80 identifier
    # cannot contain a question mark, so nothing the programmer writes can
    # collide with them - and something does: UTIL7/DM.PLM declares a variable
    # called ``bdos``, and a plain ``BDOS`` extern resolved to that variable
    # instead of the BDOS entry, turning every ``call 5`` into a call into the
    # data segment.
    _PAGE_ZERO = {
        0x0000: "??BOOT",
        0x0005: "??BDOS",
        0x0006: "??MAXB",
    }

    # Bytes of stack a mode carves out of the program image.  CP/M mode is
    # absent because it takes the stack from the BDOS pointer instead.
    _STACK_BYTES = {Mode.BARE: 64, Mode.MPM: 512}

    def _emit_entry_stack(self) -> None:
        """Emit the stack setup that runs before the module's first statement.

        Under CP/M the stack comes from the BDOS pointer at 0006H, which hands
        the program everything between its end and the BDOS.  That takes two
        instructions and four bytes.  DRI's PL/M-80 emitted one instruction and
        three bytes, ``LXI SP,stack``, and its sources depend on that width:
        they reach their own entry point through

            declare jump byte data (0c3h),
                    jadr address data (.start-3);

        so the program is entered by a jump to three bytes in front of the first
        statement.  With a four-byte prologue that jump lands one byte inside
        the ``LHLD 0006H`` operand: the program runs on whatever SP it was given
        and eventually walks the stack out of its own memory segment.  MP/M mode
        therefore emits the three-byte form, over a stack inside the image, the
        way DRI did - ``ld sp,nn`` also cannot be shortened by the peephole, the
        way a jump to nearby code would be.
        """
        if self.mode == Mode.MPM:
            self._emit("ld", "sp,??STACK")
            self._needs_stack = True
        else:
            self._emit("ld", f"hl,({self._pz(0x0006)})")
            self._emit("ld", "sp,hl")

    def _emit_stack_storage(self) -> None:
        """Emit the stack buffer, for the modes that carry their own.

        Only the module that sets SP needs one.  A program linked from several
        PL/M modules compiles each separately, and giving every one its own
        buffer would waste the space in all but the module holding the entry
        code.
        """
        size = self._STACK_BYTES.get(self.mode)
        if not size or not self._needs_stack:
            return
        self._emit()
        self._emit(comment=f"Stack storage ({size} bytes)")
        self._emit("ds", str(size))
        self._emit_label("??STACK")   # label above the buffer: SP starts here

    def _pz(self, addr: int) -> str:
        """Render a page-zero address: a literal, or an extern under MP/M."""
        if self.mode != Mode.MPM:
            return self._format_number(addr)
        name = self._PAGE_ZERO[addr]
        self._page_zero_refs.add(name)
        return name

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
        # Page-zero symbols referenced under MP/M; emitted as extrn.
        self._page_zero_refs: set[str] = set()
        # Whether this module sets SP and so needs the ??STACK buffer.
        self._needs_stack = False
        self.literal_macros: dict[str, str] = {}  # LITERALLY macro expansions
        self.block_scope_counter = 0  # Counter for unique DO block scopes
        # Procedures declared at the head of a DO block.  They are
        # hoisted out of the block and emitted after the body of the
        # enclosing procedure, the way a nested procedure is: emitted
        # in place, control would fall straight into them.
        self.deferred_block_procs: list = []
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
        elif isinstance(expr, P.Call) and len(expr.args) == 1:
            # LENGTH / LAST of an array is a BYTE constant when it fits.
            callee = unwrap_paren(expr.callee)
            if isinstance(callee, P.Identifier):
                name = ident_text(callee.name).upper()
                if name in ('LENGTH', 'LAST'):
                    extent = self._array_extent(expr.args[0])
                    if extent is not None:
                        val = extent if name == 'LENGTH' else extent - 1
                        if val <= 255:
                            return val
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
        elif is_double_call(expr):
            # DOUBLE(n) is how the optimizer writes an ADDRESS constant
            # below 256; its value is n.
            return self._try_eval_const(expr.args[0])
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
                elif op == BinaryOpKind.MUL:
                    return (left_val * right_val) & 0xFFFF
                # PL/M-80's divide: x / 0 is 0FFFFH, x MOD 0 is x.
                elif op == BinaryOpKind.DIV:
                    return plm_div(left_val, right_val)
                elif op == BinaryOpKind.MOD:
                    return plm_mod(left_val, right_val)
        return None

    def _literal_macro_value(self, name: str) -> int | None:
        """The numeric value of a LITERALLY macro, or None."""
        text = self.literal_macros.get(name)
        if text is None:
            return None
        try:
            return self._parse_plm_number(text)
        except ValueError:
            return None

    def _check_impossible_comparison(self, left, right, op) -> None:
        """Reject a BYTE compared with a constant it can never equal.

        ``op`` is the typed-AST :class:`ast_view.BinaryOpKind` decoded
        by :func:`binop_kind` at every call site (``EQ``/``NE``/``LT``/
        ``LE``/``GT``/``GE``). A BYTE is zero-extended into a comparison
        with an ADDRESS, so against a constant above 255 ``=`` is always
        false, ``<>`` and ``<`` always true, and so on -- almost certainly
        not what was meant, so it is an error.

        The constant is typed and evaluated the way the program would
        compute it: ``-1`` and ``NOT 0`` are the BYTE 0FFH (4.2.2, 4.3), and
        ``7 MOD 0`` an ADDRESS 7. A constant the optimizer derived -- from
        a variable whose value it knew, or an operand it dropped -- is not
        something the programmer wrote, and is not checked: the comparison
        is compiled as it stands, as it is at -O 0.
        """
        left_type = self._get_expr_type(left)
        if left_type != DataType.BYTE:
            return
        # Two constants make a constant, which the optimizer folds; the
        # check is about a BYTE the program computes.
        if eval_typed(left, self._literal_macro_value) is not None:
            return
        typed = eval_typed(right, self._literal_macro_value)
        if typed is None:
            return
        value, _, derived = typed
        if derived or value <= 255:
            return
        from .errors import CodeGenError, SourceLocation
        loc = None
        pos = getattr(right, 'pos', None)
        if pos is not None and getattr(pos, 'start_line', 0):
            loc = SourceLocation(pos.start_line, pos.start_column)
        verdict = {
            BinaryOpKind.EQ: ("=", "false"), BinaryOpKind.NE: ("<>", "true"),
            BinaryOpKind.LT: ("<", "true"), BinaryOpKind.LE: ("<=", "true"),
            BinaryOpKind.GT: (">", "false"), BinaryOpKind.GE: (">=", "false"),
        }.get(op)
        if verdict is None:
            return
        raise CodeGenError(
            f"comparison BYTE {verdict[0]} {value} is always {verdict[1]} "
            "(BYTE can only hold 0-255)", loc)

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

            # Truth is bit 0, not non-zero (see _emit_truth_test), so the
            # diagnostic has to agree with the code the generator emits:
            # `IF 2` is always FALSE.
            if const_val & 1:
                msg = f"IF condition is always true (constant {const_val})"
            else:
                msg = (f"IF condition is always false (constant {const_val}: "
                       "PL/M-80 tests bit 0)" if const_val
                       else "IF condition is always false (constant 0)")

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
        # callee -> procedures called while evaluating that callee's arguments
        self.arg_overlaps: dict[str, set[str]] = {}
        self.proc_storage: dict[str, list[tuple[str, int, DataType]]] = {}  # proc -> [(var_name, size, type)]

        shape = module_shape(module)

        # First pass: collect all procedure names.  Module-level
        # statements can hold a DO block that declares procedures, so
        # they are scanned alongside the declarations.
        all_procs: set[str] = set()
        module_procs = list(iter_block_proc_decls(list(shape.decls) + list(shape.stmts)))
        self._collect_proc_names(module_procs, None, all_procs)

        # Initialize call graph
        for proc in all_procs:
            self.call_graph[proc] = set()

        # Second pass: analyze calls in each procedure
        for decl in module_procs:
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
                nested = list(iter_block_proc_decls(body_items))
                if nested:
                    self._collect_proc_names(nested, full_name, all_procs)

    def _struct_item_size(self, item) -> int | None:
        """Bytes one STRUCTURE element of ``item`` occupies, or None if it is
        not a structure."""
        members = decl_item_struct_members(item)
        if not members:
            return None
        total = 0
        for m in members:
            width = 1 if _legacy_dt(struct_member_type(m)) == DataType.BYTE else 2
            m_dim = struct_member_dim(m)
            total += width * (m_dim or 1) * len(list(struct_member_names(m)))
        return total

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
        # Procedures declared anywhere in this body -- including at the
        # head of a nested DO block -- belong to this procedure's scope.
        nested_procs: list = list(iter_block_proc_decls(body_items))
        decl_items: list = []  # typed DeclItem / DeclItemBasedGroup / LiterallyDecl
        stmt_items: list = []
        for item in body_items:
            if isinstance(item, P.ProcDecl):
                continue
            if isinstance(item, P.DeclareStmt):
                for inner in iter_declare_items(item):
                    if not isinstance(inner, P.ProcDecl):
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
            # A STRUCTURE has no data type of its own - ast_view returns None
            # and expects the caller to consult its members.  Falling back to
            # ADDRESS gave every structure two bytes however many members it
            # had, so a procedure's frame was short and the next procedure's
            # frame, overlaid on the strength of that figure, landed inside it.
            # Member arithmetic elsewhere uses the true size, so a store to a
            # late member could also run off the end of ??AUTO altogether.
            struct_size = self._struct_item_size(d)
            for n in decl_item_names(d):
                if n in params:
                    continue
                if struct_size is not None:
                    size = struct_size * (d_dim or 1)
                elif d_dim and d_dim > 0:
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

    def _note_arg_overlap(self, callee: str | None, arg_calls: set[str]) -> None:
        """Record that ``callee``'s frame is live while ``arg_calls`` run.

        A non-reentrant local procedure takes its arguments in its own shared
        slots, and the caller writes them there one at a time, so the callee's
        frame holds live data from the first store - before the call, and
        therefore while every later argument is still being evaluated.  If one
        of those arguments calls a procedure whose frame is overlaid on this
        one's, it lands on an argument already stored.
        """
        if not callee or not arg_calls:
            return
        self.arg_overlaps.setdefault(callee, set()).update(arg_calls)

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
                callee = None
                self._find_calls_in_expr(callee_expr, current_proc, calls)
            arg_calls: set[str] = set()
            for arg in args:
                self._find_calls_in_expr(arg, current_proc, arg_calls)
            self._note_arg_overlap(callee, arg_calls)
            calls.update(arg_calls)
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
                callee = None
                self._find_calls_in_expr(expr.callee, current_proc, calls)
            arg_calls: set[str] = set()
            for arg in expr.args:
                self._find_calls_in_expr(arg, current_proc, arg_calls)
            self._note_arg_overlap(callee, arg_calls)
            calls.update(arg_calls)
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

        # A callee's frame is live from the moment the caller stores its first
        # argument, which is before the call and so before the later arguments
        # have even been evaluated.  Anything reached while evaluating them is
        # therefore active at the same time as the callee.
        for callee, during in self.arg_overlaps.items():
            if callee not in self.can_be_active_together:
                continue
            live = set(during)
            for g in list(during):
                live.update(reachable.get(g, set()))
            for g in live:
                if g in self.can_be_active_together:
                    self.can_be_active_together[callee].add(g)
                    self.can_be_active_together[g].add(callee)

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
        for decl in iter_block_proc_decls(decls):
            self._register_procedure(decl, parent_proc)

        # Statements can hold procedures too: PL/M-80 allows a PROCEDURE
        # at the head of any DO block, and those blocks arrive here as
        # statements rather than declarations.
        if stmts:
            for decl in iter_block_proc_decls(stmts):
                self._register_procedure(decl, parent_proc)

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
        nested_decls = list(iter_block_proc_decls(body_items))
        if nested_decls:
            self._collect_procedures(nested_decls, full_proc_name)

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
        self._page_zero_refs = set()
        self._needs_stack = False
        self.literal_macros = {}
        self._survey_stores([module])

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
        self._collect_procedures(shape.decls, parent_proc=None, stmts=shape.stmts)

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
            if self.mode in (Mode.CPM, Mode.MPM):
                # CP/M: Set stack from BDOS, call main, return to OS
                self._emit_entry_stack()
                self._emit("call", entry_proc_name)
                self._emit("jp", self._pz(0x0000))  # Warm boot to return to CP/M
            else:
                # BARE: Use locally-defined stack, jump to entry
                self._emit("ld", "sp,??STACK")
                self._needs_stack = True
                self._emit("jp", entry_proc_name)

        # Generate code for module-level statements
        if shape.stmts:
            self._emit()
            self._emit(comment="Module initialization code")
            if self.mode in (Mode.CPM, Mode.MPM):
                # CP/M: Set stack from BDOS address at 0006H
                self._emit_entry_stack()
            else:
                # BARE: Use locally-defined stack
                self._emit("ld", "sp,??STACK")
                self._needs_stack = True
            for stmt in shape.stmts:
                self._gen_stmt(stmt)
            # For CPM mode, add warm boot after module statements
            if self.mode in (Mode.CPM, Mode.MPM):
                self._emit("jp", self._pz(0x0000))  # Warm boot to return to CP/M

        # Procedures hoisted out of DO blocks in the module body, then
        # the module's own procedures.
        self._drain_block_procs([])

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

        # Emit stack storage for the modes that carry their own
        self._emit_stack_storage()

        # Note: For CPM mode, stack is provided by CP/M (set from BDOS address at 0006H).
        # For BARE mode, stack storage (??STACK) is emitted above.

        # Define __END__ label if program uses .MEMORY built-in
        # __END__ marks the first free byte after all code/data
        if self.needs_end_symbol and not any(
                l.opcode == "extrn" and l.operands == "__END__"
                for l in self.data_segment):
            self._emit()
            self._emit("extrn", "__END__")

        # Page-zero references have to be declared so the linker resolves them
        # and, for MP/M relocatable output, records them in the bitmap.
        if self._page_zero_refs:
            self._emit()
            for name in sorted(self._page_zero_refs):
                self._emit("extrn", name)

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
        self._page_zero_refs = set()
        self._needs_stack = False
        self.literal_macros = {}
        self._survey_stores(modules)

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
            self._collect_procedures(shape.decls, parent_proc=None, stmts=shape.stmts)

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
            if self.mode in (Mode.CPM, Mode.MPM):
                self._emit_entry_stack()
            else:
                self._emit("ld", "sp,??STACK")
                self._needs_stack = True
            for stmt in first_module_stmts:
                self._gen_stmt(stmt)
            if self.mode in (Mode.CPM, Mode.MPM):
                self._emit("jp", self._pz(0x0000))
        elif entry_proc:
            # No statements - call entry procedure
            self._emit()
            self._emit(comment="Entry point")
            if self.mode in (Mode.CPM, Mode.MPM):
                self._emit_entry_stack()
                self._emit("call", entry_proc_name)
                self._emit("jp", self._pz(0x0000))
            else:
                self._emit("ld", "sp,??STACK")
                self._needs_stack = True
                self._emit("call", entry_proc_name)

        # Procedures hoisted out of DO blocks in the module body.
        self._drain_block_procs([])

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

        # Emit stack storage for the modes that carry their own
        self._emit_stack_storage()

        # Define __END__ label if program uses .MEMORY built-in
        # __END__ marks the first free byte after all code/data
        if self.needs_end_symbol and not any(
                l.opcode == "extrn" and l.operands == "__END__"
                for l in self.data_segment):
            self._emit()
            self._emit("extrn", "__END__")

        # Page-zero references have to be declared so the linker resolves them
        # and, for MP/M relocatable output, records them in the bitmap.
        if self._page_zero_refs:
            self._emit()
            for name in sorted(self._page_zero_refs):
                self._emit("extrn", name)

        # End directive
        self._emit()
        self._emit("end")

        return "\n".join(str(line) for line in self.output)

    def _build_call_graph_multi(self, modules: list) -> None:
        """Build call graph by analyzing all procedures across multiple modules."""
        self.call_graph = {}
        self.arg_overlaps = {}
        self.proc_storage: dict[str, list[tuple[str, int, DataType]]] = {}

        shapes = [module_shape(m) for m in modules]

        # First pass: collect all procedure names from all modules
        all_procs: set[str] = set()
        module_procs: list = []
        for shape in shapes:
            procs = list(iter_block_proc_decls(list(shape.decls) + list(shape.stmts)))
            module_procs.extend(procs)
            self._collect_proc_names(procs, None, all_procs)

        # Initialize call graph
        for proc in all_procs:
            self.call_graph[proc] = set()

        # Second pass: analyze calls in each procedure across all modules
        for decl in module_procs:
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

        # `DECLARE x (*) BYTE DATA (...)' takes its extent from the data.  The
        # parser marks (*) as -1 and that was left on the symbol, so LAST(x)
        # came out as -2.  UTIL6/PIP.PLM declares its delimiter table that way,
        #     DECLARE DEL(*) BYTE DATA (' =.:;,<>',CR,LA,LB,RB);
        #     DO I = 0 TO LAST(DEL);
        # so PIP recognised no delimiter at all and answered "INVALID FORMAT"
        # to every command it was given.
        if dimension == -1:
            values = data_values_nodes or initial_values_nodes
            if values:
                dimension = self._data_element_count(
                    values, data_type or DataType.BYTE)

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
            based_member=based_member,
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
            self._emit_initial_values(
                initial_values_nodes,
                data_type or DataType.BYTE,
                struct_members=struct_members,
                dimension=dimension,
                size=size,
            )
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

    def _at_designator(self, expr):
        """Resolve a constant `.designator' to (base symbol, name, offset, elem).

        Handles NAME, NAME(const), STRUCT.MEMBER and any chain of those, which
        is what DRI's sources put in an AT clause: UTIL6/PIP.PLM declares
        ``DESTR ADDRESS AT(.DEST.FCB(33))`` and UTIL5/PRLCM.PLM
        ``code$size ADDRESS AT (.buffer(0).sector(1))``.  ``elem`` is the width
        of one element of whatever a further subscript would index.

        Returns None when any part is not a compile-time constant.
        """
        expr = unwrap_paren(expr)
        if isinstance(expr, P.Identifier):
            name = ident_text(expr.name)
            base_sym = self._lookup_scoped(name)
            asm = (base_sym.asm_name if base_sym and base_sym.asm_name
                   else self._mangle_name(name))
            if base_sym is not None and base_sym.struct_members:
                elem = sum((m.dimension or 1)
                           * (1 if m.data_type == DataType.BYTE else 2)
                           for m in base_sym.struct_members)
            elif base_sym is not None and base_sym.data_type != DataType.BYTE:
                elem = 2
            else:
                elem = 1
            return base_sym, asm, 0, elem
        if isinstance(expr, P.MemberAccess):
            base = self._at_designator(expr.base)
            if base is None:
                return None
            base_sym, asm, off, _ = base
            m_off, m_type = self._get_member_info(expr)
            return base_sym, asm, off + m_off, (1 if m_type == DataType.BYTE else 2)
        if isinstance(expr, P.Call):
            args = list(expr.args or [])
            if len(args) != 1:
                return None
            index = self._try_eval_const(args[0])
            if index is None:
                return None
            base = self._at_designator(expr.callee)
            if base is None:
                return None
            base_sym, asm, off, elem = base
            return base_sym, asm, off + index * elem, elem
        return None

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
                    # .MEMORY is the first free byte after the whole PROGRAM,
                    # and only the linker knows where that is.  A label at the
                    # end of this module marks the end of the MODULE, which in
                    # a program linked from several of them is somewhere in the
                    # middle: SDIR is eight modules, and its 128-entry hash
                    # table, declared AT (.MEMORY) in UTIL7/DSE.PLM, landed on
                    # top of another module's strings and cleared them.
                    # __END__ is the linker's own symbol, so name it as one.
                    # The EXTRN has to come first: an EQU is evaluated where
                    # it stands, and if __END__ is not known to be external by
                    # then the symbol silently takes the value zero.
                    if not self.needs_end_symbol:
                        self.data_segment.append(
                            AsmLine(opcode="extrn", operands="__END__")
                        )
                    self.needs_end_symbol = True
                    self.data_segment.append(
                        AsmLine(label=asm_name, opcode="EQU", operands="__END__")
                    )
                else:
                    ref_sym = self.symbols.lookup(ref_name_text)
                    if ref_sym and ref_sym.is_external:
                        # AT(.external) — alias the external's name, so later
                        # references name the external directly.
                        ref_asm = (
                            ref_sym.asm_name if ref_sym.asm_name
                            else self._mangle_name(ref_name_text)
                        )
                        sym.asm_name = ref_asm
                        # An EQU as well, because a reference can come BEFORE
                        # the declaration: UTIL5/SUB.PLM initialises a
                        # structure with `.a$buff' in the same DECLARE that
                        # goes on to declare `a$buff ... AT(.tbuff)'.  Without
                        # it that forward reference has no definition at all.
                        if asm_name and asm_name != ref_asm:
                            self.data_segment.append(
                                AsmLine(label=asm_name, opcode="EQU",
                                        operands=ref_asm)
                            )
                    else:
                        ref_asm = (
                            ref_sym.asm_name if ref_sym and ref_sym.asm_name
                            else self._mangle_name(ref_name_text)
                        )
                        # EQU: one value everywhere, forward reference or
                        # not.  A SET symbol would read as zero above its
                        # definition.
                        self.data_segment.append(
                            AsmLine(label=asm_name, opcode="EQU", operands=ref_asm)
                        )
                return

            # Anything else: a constant designator - NAME(i), STRUCT.MEMBER,
            # or a chain of them.  Previously only a bare NAME(<literal>) was
            # understood and everything else silently became `EQU $', the
            # assembler's location counter, which pointed the variable at a
            # arbitrary spot: UTIL4/STAT.PLM's
            #     dolla literally '.fcb(6dh-5ch)',  doll byte at (dolla),
            # read a stray byte as its `$' parameter, so `stat <file>' was
            # taken for a request to change the file's attributes.
            resolved = self._at_designator(loc_operand)
            if resolved is None:
                raise CodeGenError(
                    f"AT(...) needs a constant address expression; got "
                    f"{type(loc_operand).__name__}")
            base_sym, base_asm, offset, _ = resolved
            operand = base_asm if offset == 0 else f"{base_asm}+{offset}"
            # An external base is aliased rather than defined, so that
            # references name the external and the linker resolves them.
            is_base_external = bool(base_sym and base_sym.is_external)
            if not is_base_external and base_sym and base_sym.asm_name:
                root = base_sym.asm_name.split('+')[0].strip()
                root_sym = self.symbols.lookup(root)
                if root_sym and root_sym.is_external:
                    is_base_external = True
            if is_base_external:
                sym.asm_name = operand
            else:
                self.data_segment.append(
                    AsmLine(label=asm_name, opcode="EQU", operands=operand)
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
                target.append(
                    AsmLine(opcode="dw", operands=self._location_operand(operand))
                )
            elif isinstance(val, P.BinaryOp):
                const_val = self._try_eval_const(val)
                if const_val is not None:
                    # A constant expression is evaluated here, the way PL/M-80
                    # evaluates it (the assembler would reject `7/0', and
                    # knows nothing of `7 MOD 0' = 7), and takes the width of
                    # the slot like any other value.
                    directive = "db" if dtype == DataType.BYTE else "dw"
                    target.append(AsmLine(
                        opcode=directive,
                        operands=self._format_number(const_val & 0xFFFF)))
                    continue
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
            else:
                raise CodeGenError(
                    f"Unsupported value in DATA/INITIAL: {type(val).__name__}")

    def _location_operand(self, operand) -> str:
        """Assembly operand for the target of a `.' address-of in DATA/INITIAL.

        ``.name`` is the symbol; ``.name(n)`` is the n-th element of it, which
        DRI's sources use to point into an array - MP/M II's UTIL7/DM.PLM
        initialises a structure with ``.buff(0)`` and ``.fcb(0)``.  The offset
        is in elements, so it is scaled by the element width.
        """
        if isinstance(operand, P.ParenExpr):
            return self._location_operand(operand.inner)
        if isinstance(operand, P.Identifier):
            return self._data_expr_to_string(operand)
        if isinstance(operand, P.Call):
            base = self._data_expr_to_string(operand.callee)
            args = list(operand.args or [])
            if len(args) != 1 or not isinstance(args[0], P.NumberLiteral):
                raise CodeGenError(
                    f"Unsupported subscript in DATA location expression: {operand}")
            index = number_value(args[0])
            if index == 0:
                return base
            width = 1
            if isinstance(operand.callee, P.Identifier):
                sym = self._lookup_scoped(ident_text(operand.callee.name))
                if sym is not None and sym.data_type != DataType.BYTE:
                    width = 2
            return f"{base}+{self._format_number(index * width)}"
        raise CodeGenError(f"Unsupported operand in DATA location expression: {operand}")

    def _lookup_scoped(self, name: str):
        """Look a name up in the enclosing procedure scopes, then at module level."""
        if self.current_proc:
            parts = self.current_proc.split('$')
            for i in range(len(parts), 0, -1):
                sym = self.symbols.lookup('$'.join(parts[:i]) + '$' + name)
                if sym:
                    return sym
        return self.symbols.lookup(name)

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
            return self._location_operand(expr.operand)
        elif isinstance(expr, P.Call):
            return self._location_operand(expr)
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
                BinaryOpKind.MOD: ' MOD ',
                BinaryOpKind.AND: ' AND ',
                BinaryOpKind.OR: ' OR ',
                BinaryOpKind.XOR: ' XOR ',
            }
            op = op_map.get(binop_kind(expr))
            if op is None:
                raise CodeGenError(
                    f"Unsupported operator in DATA expression: {binop_kind(expr).name}")
            return f"({left}{op}{right})"
        else:
            raise CodeGenError(f"Unsupported expression in DATA: {type(expr)}")

    def _data_element_count(self, values, dtype: DataType) -> int:
        """How many elements a DATA/INITIAL list supplies."""
        total = sum(self._initial_value_width(v, dtype) for v in values)
        width = 1 if dtype == DataType.BYTE else 2
        return max(1, total // width)

    def _initial_member_widths(self, struct_members, dimension):
        """Byte width of each slot a STRUCTURE initialiser fills, in order."""
        if not struct_members:
            return None
        one = []
        for m in struct_members:
            width = 1 if m.data_type == DataType.BYTE else 2
            one.extend([width] * (m.dimension or 1))
        return one * (dimension or 1)

    def _initial_value_width(self, val, dtype: DataType) -> int:
        """Bytes a single INITIAL value occupies once emitted."""
        if isinstance(val, P.StringLiteral):
            return len(string_value(val))
        if isinstance(val, P.ParenExpr):
            return self._initial_value_width(val.inner, dtype)
        if isinstance(val, P.LocationOfList):
            return 2 * len(val.values or [])
        return 1 if dtype == DataType.BYTE else 2

    def _emit_initial_values(self, values, dtype: DataType,
                             struct_members=None, dimension=None,
                             size: int | None = None) -> None:
        """Emit typed INITIAL values to the data segment.

        A STRUCTURE initialiser supplies one value per member and the members
        have their own widths, so the list cannot be emitted at a single width
        the way an array's can.  Emitting every value as a byte both wrote the
        wrong values and left the structure short, which moved everything
        declared after it: MP/M II's UTIL7/DM.PLM declares a seven-member,
        ten-byte parser control block and got five bytes of zero, so SDIR
        scanned its command line through a null pointer and blanked the BDOS
        entry in page zero.

        Whatever the list does not fill is reserved, so the next declaration
        still lands where it should.
        """
        widths = self._initial_member_widths(struct_members, dimension)
        emitted = 0
        for i, val in enumerate(values):
            if widths is not None and i < len(widths):
                slot = DataType.BYTE if widths[i] == 1 else DataType.ADDRESS
            else:
                slot = dtype
            self._emit_data_values([val], slot)
            emitted += self._initial_value_width(val, slot)
        if size is not None and emitted < size:
            self.data_segment.append(
                AsmLine(opcode="ds", operands=str(size - emitted))
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

        # Procedures hoisted out of DO blocks in THIS body are emitted
        # after it; anything the caller had pending stays pending.
        saved_block_procs = self.deferred_block_procs
        self.deferred_block_procs = []

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
            self.deferred_block_procs = saved_block_procs
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

        # Generate prologue code for parameters.
        #
        # A procedure private to this module is called with its earlier
        # arguments already written into its own storage by the caller, and
        # only the last one arrives in a register.  A PUBLIC procedure cannot
        # be: a caller in another module has no way to name those slots, so it
        # pushes every argument and pops them again.  The two conventions have
        # to agree, so a public procedure takes all of its arguments off the
        # stack here.  (MP/M II's SDIR is eight modules and calls a public
        # `pdecimal(v, prec, zerosup)' across them: the callee was reading two
        # of the three from slots nobody had written, so every number it
        # printed was wrong or missing.)
        if param_infos and not attrs.is_reentrant:
            if attrs.is_public:
                # Pushed left to right, so the last argument is nearest the
                # return address: parameter i sits at SP+2+2*(n-1-i).
                n_params = len(param_infos)
                for idx, (_, p_asm, p_type, _) in enumerate(param_infos):
                    off = 2 + 2 * (n_params - 1 - idx)
                    self._emit("ld", f"hl,{off}")
                    self._emit("add", "hl,sp")
                    if p_type == DataType.BYTE:
                        # A BYTE argument is widened to a word when pushed.
                        self._emit("ld", "a,(hl)")
                        self._emit("ld", f"({p_asm}),a")
                    else:
                        self._emit("ld", "e,(hl)")
                        self._emit("inc", "hl")
                        self._emit("ld", "d,(hl)")
                        self._emit("ex", "de,hl")
                        self._emit("ld", f"({p_asm}),hl")
            else:
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

        # ...and the ones hoisted out of DO blocks inside this body.
        self._drain_block_procs(saved_block_procs)

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
            if all(self._is_byte_target(t) for t in targets):
                # PL/M-80 narrows to the target's width, so truncate here
                # rather than emitting `ld hl,nn` and leaving the peephole
                # to collapse `ld hl,nn / ld a,l` into an `ld a,nn` that
                # keeps all sixteen bits.
                const_val &= 0xFF
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
        value = stmt.value
        if all(self._is_byte_target(t) for t in targets):
            value = self._low_byte_form(value)
        value_type = self._gen_expr(value)

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
                    # Direct BDOS call: ld de,addr; ld c,func; CALL 5.
                    # The function number is loaded LAST: C is not
                    # callee-saved and the argument expression is free to
                    # contain a call (including another MON1/MON2), which
                    # would otherwise leave a different function number in C.
                    addr_type = self._gen_expr(addr_arg)
                    if addr_type == DataType.BYTE:
                        # BYTE arg goes in E; BDOS ignores D for byte-only functions
                        self._emit("ld", "e,a")
                    else:
                        self._emit("ex", "de,hl")  # DE = addr
                    self._emit("ld", f"c,{self._format_number(func_num)}")
                    self._emit("call", self._pz(0x0005))  # BDOS entry point
                    return  # Done - no stack cleanup needed

        # For non-reentrant LOCAL procedures, store args directly to parameter memory
        # For reentrant procedures, external procedures, or indirect calls, use stack
        use_stack = not (sym and sym.kind == SymbolKind.PROCEDURE and not sym.is_reentrant
                         and not sym.is_external and not sym.is_public)

        if use_stack:
            # Stack-based parameter passing (reentrant or indirect calls)
            for arg in args:
                arg_type = self._gen_expr(arg)
                if arg_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("push", "hl")
        else:
            self._gen_slot_args(sym, args, callee_name_str)

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
                # Adjust the stack pointer directly. (This loaded DE and
                # added SP to whatever the procedure left in HL.)
                self._emit("ld", f"hl,{stack_bytes}")
                self._emit("add", "hl,sp")
                self._emit("ld", "sp,hl")

    def _param_slot(self, sym, param_name: str, callee_name: str | None) -> str:
        """The label of parameter ``param_name`` of procedure ``sym``."""
        labels = getattr(self, 'storage_labels', {}).get(sym.name, {})
        if param_name in labels:
            return labels[param_name]
        # Fallback: @procname$param
        proc_base = sym.asm_name if sym.asm_name else callee_name or ""
        if proc_base.startswith('@'):
            proc_base = proc_base[1:]
        return f"@{proc_base}${self._mangle_name(param_name)}"

    def _may_reenter(self, expr, callee: str) -> bool:
        """Whether evaluating ``expr`` may call procedure ``callee`` (its full
        name): a call of it, of a procedure that reaches it, or through an
        address."""
        stack = [expr]
        while stack:
            e = unwrap_paren(stack.pop())
            target = None
            if isinstance(e, (P.Call, P.CallNoArgs)):
                c = unwrap_paren(e.callee)
                if isinstance(c, P.Identifier):
                    target = ident_text(c.name)
                elif not isinstance(c, P.MemberAccess):   # `s.m(i)' is an element
                    return True             # an indirect call
            elif isinstance(e, P.Identifier):
                target = ident_text(e.name)
            elif isinstance(e, P.LocationOf) and isinstance(unwrap_paren(e.operand), P.Identifier):
                continue                    # `.f' names f, it does not call it
            if target is not None:
                g = self._resolve_proc_name(target, self.current_proc or "")
                if g is not None and (g == callee or callee in self._get_reachable(g, set())):
                    return True
            if isinstance(e, (list, tuple)):
                stack.extend(e)
                continue
            fields = getattr(e, "__dataclass_fields__", None)
            if fields:
                stack.extend(getattr(e, f, None) for f in fields if f != "pos")
        return False

    def _gen_slot_args(self, sym, args, callee_name: str | None) -> None:
        """Pass ``args`` to a non-reentrant local procedure ``sym``: each is
        stored in the procedure's own slot for its parameter, converted to
        the parameter's type (8.2), except a last one passed in A or HL.

        An argument whose evaluation calls the procedure again -- ``f(1,
        f(2, 3))'' -- would store over the slots the arguments before it
        already filled, so those are kept on the stack until it has run.
        DRI's PL/M-80 passes them on the stack and lets the callee store
        them, which comes to the same thing.
        """
        last_param_idx = len(args) - 1
        uses_reg = sym.uses_reg_param and len(args) > 0
        reenter = max((j for j in range(1, len(args))
                       if self._may_reenter(args[j], sym.name)), default=0)
        stashed: list[tuple[str, DataType]] = []

        for i, arg in enumerate(args):
            if i >= len(sym.params):
                continue
            param_name = sym.params[i]
            param_type = sym.param_types[i] if i < len(sym.param_types) else DataType.ADDRESS

            # Evaluate into A for a BYTE parameter, HL for an ADDRESS one.
            if param_type == DataType.BYTE:
                self._gen_expr_to_a(arg)
            else:
                self._gen_expr_to_hl(arg)

            if i == last_param_idx and uses_reg:
                continue                    # passed in the register
            slot = self._param_slot(sym, param_name, callee_name)
            if i < reenter:
                self._emit("push", "af" if param_type == DataType.BYTE else "hl")
                stashed.append((slot, param_type))
            elif param_type == DataType.BYTE:
                self._emit("ld", f"({slot}),a")
            else:
                # A BYTE argument is widened: storing A alone left the high
                # byte of an ADDRESS parameter as the last call had left it.
                self._emit("ld", f"({slot}),hl")

        if stashed:
            reg_type = sym.param_types[last_param_idx] if (
                uses_reg and last_param_idx < len(sym.param_types)) else None
            if reg_type == DataType.BYTE:
                self._emit("ld", "e,a")
            elif reg_type is not None:
                self._emit("ex", "de,hl")
            for slot, t in reversed(stashed):
                if t == DataType.BYTE:
                    self._emit("pop", "af")
                    self._emit("ld", f"({slot}),a")
                else:
                    self._emit("pop", "hl")
                    self._emit("ld", f"({slot}),hl")
            if reg_type == DataType.BYTE:
                self._emit("ld", "a,e")
            elif reg_type is not None:
                self._emit("ex", "de,hl")

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
                if return_type == DataType.BYTE:
                    value = self._low_byte_form(value)
                result_type = self._gen_expr(value)
                # Return value is in A (BYTE) or HL (ADDRESS)
                # If procedure returns BYTE but we have ADDRESS, convert
                if return_type == DataType.BYTE and result_type == DataType.ADDRESS:
                    # PL/M-80 narrows ADDRESS to BYTE by truncation, exactly
                    # like LOW(), not by normalising to a 0FFH/00H boolean.
                    # `P: PROCEDURE BYTE; RETURN N + 1; END P;` with N = 64
                    # returns 65, not 0FFH.
                    self._emit("ld", "a,l")
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
            result_type = self._gen_expr(self._low_byte_form(stmt.condition))
            self._emit_truth_test(result_type, false_target, jump_when_true=False)

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

    def _emit_truth_test(self, value_type, label: str, jump_when_true: bool) -> None:
        """Branch on the truth of a condition value already in ``A`` / ``HL``.

        PL/M-80 does not ask whether a condition value is non-zero: it
        tests the value's LEAST-SIGNIFICANT BIT. DRI's own code leans on
        that rule — the ROL/ROR idiom rotates the bit of interest into
        bit 0 and lets ``IF`` read it (``PIP.PLM`` writes every FCB
        attribute test as ``IF ROL(fcb(n),1)``), and DRI's hand
        translation of ``bdos.plm``'s ``IF NOT ROR(ROL(DLOG,1),...)`` is
        literally ``mov a,l! rar! rc``.

        A relational yields 0FFH or 00H, so a non-zero test and a bit-0
        test agree on those. They part company on ``NOT`` — ``NOT 1`` is
        0FEH, true under a non-zero test and false under PL/M-80's —
        and on any masked or rotated value.

        ``bit 0,l`` reads bit 0 of a 16-bit value in place, so unlike the
        old ``ld a,l`` / ``or h`` sequence this leaves ``A`` alone.
        """
        if value_type == DataType.BYTE:
            self._emit("bit", "0,a")
        else:
            self._emit("bit", "0,l")
        self._emit("jp", f"{'nz' if jump_when_true else 'z'},{label}")

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
        # A condition is its bit 0, and a comparison of two bytes is a byte
        # compare, however wide the optimizer left them.
        condition = self._without_redundant_double(
            self._narrowed_comparison(self._low_byte_form(condition)))

        # Handle constant conditions. Truth is bit 0, not non-zero (see
        # _emit_truth_test), so `DO WHILE 2` never runs and `IF NOT TRUE`
        # with `TRUE LITERALLY '1'` -- NOT 1 is 0FEH -- is false.
        if isinstance(condition, P.NumberLiteral):
            if number_value(condition) & 1 == 0:
                # Always false - unconditional jump
                self._emit("jp", false_label)
            # If bit 0 is set (always true), no code needed - fall through
            return True

        # Handle simple identifier - load and test directly
        if isinstance(condition, P.Identifier):
            cond_type = self._get_expr_type(condition)
            self._gen_expr(condition)  # BYTE -> A, ADDRESS -> HL
            self._emit_truth_test(cond_type, false_label, jump_when_true=False)
            return True

        # Handle function call - evaluate and test result
        if isinstance(condition, (P.Call, P.CallNoArgs)):
            cond_type = self._gen_call_expr(condition)
            self._emit_truth_test(cond_type, false_label, jump_when_true=False)
            return True

        # Handle NOT - invert the condition
        if isinstance(condition, P.UnaryOp) and unop_kind(condition) == UnaryOpKind.NOT:
            # NOT x is false when x is true, so jump to false_label when x is true
            return self._gen_condition_jump_true(unwrap_paren(condition.operand), false_label)

        if not isinstance(condition, P.BinaryOp):
            return False

        op = binop_kind(condition)

        # NOTE: PL/M-80 AND and OR are BITWISE operators, not short-circuit logical operators.
        # IF X AND Y computes X bitwise-and Y and then tests BIT 0 of the
        # result (see _emit_truth_test), not whether it is non-zero.
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
        # A constant above 255 is an ADDRESS, and the BYTE is zero-extended
        # to meet it: that is a 16-bit comparison, below.
        if left_type == DataType.BYTE:
            const_val = None
            if isinstance(condition.right, P.NumberLiteral):
                val = number_value(condition.right)
                if val <= 255:
                    const_val = val
            elif isinstance(condition.right, P.StringLiteral):
                s = string_value(condition.right)
                if len(s) == 1:
                    const_val = ord(s[0])

            if const_val is not None:
                # _gen_expr_to_a, not _gen_expr: a NumberLiteral left operand
                # loads as `ld hl,n` and would leave A undefined under the `cp`.
                self._gen_expr_to_a(condition.left)
                self._emit("cp", self._format_number(const_val))
                self._emit_jump_on_false(op, false_label)
                return True
            elif both_bytes:
                # Byte-to-byte comparison. `sub b` wants left in A and right
                # in B, so whichever operand is generated after the park has
                # to leave B alone. Generating right first is one instruction
                # shorter, but it is only safe when the left operand cannot
                # clobber B; otherwise spill the left operand through the
                # stack, the way _gen_byte_binary does.
                if self._expr_preserves_b(condition.left):
                    self._gen_expr_to_a(condition.right)  # A = right
                    self._emit("ld", "b,a")
                    self._gen_expr_to_a(condition.left)   # A = left, B untouched
                else:
                    self._gen_expr_to_a(condition.left)   # A = left
                    self._emit("push", "af")
                    self._gen_expr_to_a(condition.right)  # A = right
                    self._emit("ld", "b,a")
                    self._emit("pop", "af")               # A = left
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
                self._gen_expr_to_hl(condition.left)
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
                # Evaluate complex right first, save to DE, then simple left.
                # Key off the type _gen_expr actually returned, not the
                # statically inferred one: an embedded assignment into an
                # ADDRESS element is typed BYTE by _get_expr_type but lands
                # in HL, and widening that with `ld e,a` spliced in a stale A.
                actual_right_type = self._gen_expr(condition.right)
                if actual_right_type == DataType.BYTE:
                    self._emit("ld", "e,a")
                    self._emit("ld", "d,0")
                else:
                    self._emit("ex", "de,hl")  # DE = right
                # Evaluate left - DE is preserved
                self._gen_expr_to_hl(condition.left)
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
        condition = self._without_redundant_double(
            self._narrowed_comparison(self._low_byte_form(condition)))

        # Handle constant conditions - truth is bit 0, not non-zero.
        if isinstance(condition, P.NumberLiteral):
            if number_value(condition) & 1:
                # Always true - unconditional jump
                self._emit("jp", true_label)
            # If bit 0 is clear (always false), no code needed - fall through
            return True

        # Handle simple identifier
        if isinstance(condition, P.Identifier):
            cond_type = self._get_expr_type(condition)
            self._gen_expr(condition)  # BYTE -> A, ADDRESS -> HL
            self._emit_truth_test(cond_type, true_label, jump_when_true=True)
            return True

        # Handle function call - evaluate and test result
        if isinstance(condition, (P.Call, P.CallNoArgs)):
            cond_type = self._gen_call_expr(condition)
            self._emit_truth_test(cond_type, true_label, jump_when_true=True)
            return True

        # Handle NOT - invert the condition
        if isinstance(condition, P.UnaryOp) and unop_kind(condition) == UnaryOpKind.NOT:
            # NOT x is true when x is false, so jump to true_label when x is false
            return self._gen_condition_jump_false(unwrap_paren(condition.operand), true_label)

        if not isinstance(condition, P.BinaryOp):
            return False

        op = binop_kind(condition)

        # NOTE: PL/M-80 AND and OR are BITWISE operators, not short-circuit logical operators.
        # IF X OR Y computes X bitwise-or Y and then tests BIT 0 of the
        # result (see _emit_truth_test), not whether it is non-zero.
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
        # A constant above 255 is an ADDRESS, and the BYTE is zero-extended
        # to meet it: that is a 16-bit comparison, below.
        if left_type == DataType.BYTE:
            const_val = None
            if isinstance(condition.right, P.NumberLiteral):
                val = number_value(condition.right)
                if val <= 255:
                    const_val = val
            elif isinstance(condition.right, P.StringLiteral):
                s = string_value(condition.right)
                if len(s) == 1:
                    const_val = ord(s[0])

            if const_val is not None:
                # _gen_expr_to_a, not _gen_expr: a NumberLiteral left
                # operand loads as `ld hl,n` and leaves A undefined.
                self._gen_expr_to_a(condition.left)
                self._emit("cp", self._format_number(const_val))
                self._emit_jump_on_true(op, true_label)
                return True
            elif both_bytes:
                # Byte-to-byte comparison. `sub b` wants left in A and right
                # in B, so whichever operand is generated after the park has
                # to leave B alone. Generating right first is one instruction
                # shorter, but it is only safe when the left operand cannot
                # clobber B; otherwise spill the left operand through the
                # stack, the way _gen_byte_binary does.
                if self._expr_preserves_b(condition.left):
                    self._gen_expr_to_a(condition.right)  # A = right
                    self._emit("ld", "b,a")
                    self._gen_expr_to_a(condition.left)   # A = left, B untouched
                else:
                    self._gen_expr_to_a(condition.left)   # A = left
                    self._emit("push", "af")
                    self._gen_expr_to_a(condition.right)  # A = right
                    self._emit("ld", "b,a")
                    self._emit("pop", "af")               # A = left
                self._emit("sub", "b")    # A = left - right, flags set
                self._emit_jump_on_true(op, true_label)
                return True

        if not both_bytes:
            # Optimize ADDRESS comparison with 0: use ld a,l / or h instead of subtraction
            if (
                op in (BinaryOpKind.EQ, BinaryOpKind.NE)
                and isinstance(condition.right, P.NumberLiteral)
                and number_value(condition.right) == 0
            ):
                self._gen_expr_to_hl(condition.left)
                self._emit("ld", "a,l")
                self._emit("or", "h")  # Z flag set if HL == 0
                if op == BinaryOpKind.EQ:
                    self._emit("jp", f"z,{true_label}")  # If HL == 0, condition is true
                else:  # NE
                    self._emit("jp", f"nz,{true_label}")  # If HL != 0, condition is true
                return True

            # 16-bit comparison
            self._gen_expr_to_hl(condition.left)
            self._emit("push", "hl")

            self._gen_expr_to_hl(condition.right)

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

    def _hoist_block_procs(self, decls: list) -> list:
        """Split procedure declarations out of a block's declaration list.

        A ``PROCEDURE`` written at the head of a ``DO ... END`` block is
        legal PL/M-80, but its code must not be emitted where the block
        sits: the enclosing code would run straight into the procedure
        body and take its ``RET``.  The procedures are queued here and
        emitted out of line by :meth:`_drain_block_procs`, and they are
        named in the enclosing procedure's scope, which is the scope
        pass 1 registered them in.

        Two sibling blocks in one procedure that both declare a
        procedure of the same name therefore collide on one asm label.
        The assembler rejects that outright (``multiply defined``), so
        it cannot go unnoticed; no PL/M-80 source in the CP/M or MP/M II
        corpora writes it.
        """
        queued = [d for d in decls if isinstance(d, P.ProcDecl)]
        if not queued:
            return decls
        # The block's own scope goes with them: a procedure declared in
        # a block still reads the block's locals, and by the time it is
        # emitted that scope has been left.
        scope = self.symbols.current_scope
        self.deferred_block_procs.extend((d, scope) for d in queued)
        return [d for d in decls if not isinstance(d, P.ProcDecl)]

    def _drain_block_procs(self, saved: list) -> None:
        """Emit the procedures hoisted out of the blocks just generated.

        Called once the enclosing procedure (or the module body) has
        been emitted in full, so the hoisted code lands out of line.
        Generating one can queue more, hence the loop. ``saved`` is the
        caller's own pending list, restored on the way out.
        """
        while self.deferred_block_procs:
            pending = self.deferred_block_procs
            self.deferred_block_procs = []
            for proc, scope in pending:
                outer_scope = self.symbols.current_scope
                self.symbols.current_scope = scope
                try:
                    self._gen_proc_decl(proc)
                finally:
                    self.symbols.current_scope = outer_scope
        self.deferred_block_procs = saved

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

        # Procedures declared here are emitted out of line, but keep
        # this scope so they still see the block's locals.
        decls = self._hoist_block_procs(decls)

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
        decls = self._hoist_block_procs(decls)

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
                result_type = self._gen_expr(self._low_byte_form(stmt.condition))
                self._emit_truth_test(result_type, end_label, jump_when_true=False)

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
        iter_decls, body_stmts = block_items_split(stmt.items)
        # An iterative DO is a block like any other: it may declare, and a
        # PROCEDURE declared here has to be hoisted out and emitted, or the
        # call sites name a label nothing defines.
        iter_decls = self._hoist_block_procs(iter_decls)
        for decl in iter_decls:
            self._gen_declaration(decl)

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

        # The limit (like the start and the step) is converted to the
        # index's type (5.1.4), so a BYTE index makes a byte loop whatever
        # the limit is: `DO b = 0 TO 300' runs to 44.
        both_bytes = index_type == DataType.BYTE

        # Get step value (default +1 when no BY clause; only constant
        # NumberLiteral steps drive the byte-loop optimisations).
        # A BY clause whose step is not a constant has to be evaluated each
        # time round.  Defaulting step_val to 1 for it silently turned
        # `DO J = A TO B BY I' into `BY 1': MP/M II's UTIL7/DSE.PLM walks an
        # FCB disk map `BY i', where i is 1 or 2 according to whether the disk
        # uses byte or word block numbers, and counted every allocated block
        # twice on a disk with word numbers.
        step_val = 1
        step_is_const = True
        if step_expr is not None:
            if isinstance(step_expr, P.NumberLiteral):
                step_val = number_value(step_expr)
            else:
                step_is_const = False

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
        # The count is taken once, where PL/M-80 evaluates the limit on
        # every pass, so a variable limit has to be one the body cannot
        # change; and the index is not kept up to date, so the body must
        # not be able to change it either.
        bound_fixed = self._limit_is_fixed(stmt.bound, body_stmts)
        if (
            both_bytes
            and step_is_const
            and step_val == 1
            and not index_used
            and not body_has_goto
            and bound_fixed
            and not self._stmts_may_change(index_name, body_stmts)
            and isinstance(stmt.start, P.NumberLiteral)
            and number_value(stmt.start) == 0
        ):
            # Calculate iteration count = bound + 1
            # If bound is constant, emit LD B,bound+1
            # If bound is variable, emit: load bound; INC A; LD B,A
            # The index is not stored while the loop runs; after it, it is
            # given the value the loop leaves in it, limit + 1 -- unless
            # nothing reads it later.
            index_live = self._read_outside(index_name, stmt)
            if isinstance(stmt.bound, P.NumberLiteral):
                bound_const = number_value(stmt.bound) & 0xFF
                iter_count = bound_const + 1
                if iter_count <= 255:
                    self._emit("ld", f"b,{self._format_number(iter_count)}")
                else:
                    # Too many iterations for DJNZ
                    pass  # Fall through to regular loop
            else:
                # Variable bound: A = bound; A++; B = A. A limit of 255
                # makes the count 0, which DJNZ takes as 256 -- the index
                # runs 0..255 and wraps to 0, where the loop stops.
                bt = self._gen_expr(stmt.bound)
                if bt == DataType.ADDRESS:
                    self._emit("ld", "a,l")
                self._emit("inc", "a")  # A = bound + 1 = iteration count
                self._emit("ld", "b,a")  # B = iteration count

            # Only proceed with B-counter loop if we set up B
            if (
                isinstance(stmt.bound, P.NumberLiteral)
                and (number_value(stmt.bound) & 0xFF) + 1 <= 255
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
                if index_live:
                    self._emit("ld", f"a,{self._format_number(iter_count)}")
                    self._gen_store(index_var, DataType.BYTE)
                self.loop_stack.pop()
                return
            elif not isinstance(stmt.bound, P.NumberLiteral):
                # Variable bound case - we set up B above.
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
                if index_live:
                    # The limit is one the body cannot change.
                    self._gen_expr_to_a(stmt.bound)
                    self._emit("inc", "a")
                    self._gen_store(index_var, DataType.BYTE)
                self.loop_stack.pop()
                return

        # PL/M-80's iterative DO, as DRI's compiler codes it (UTIL4/ERAQ.PRL,
        # UTIL3/LOAD.COM): the index is compared with the limit before each
        # pass, and the loop ends when the increment carries out of the
        # index's width -- `INR A / JNZ top' for a BYTE step of 1,
        # `DAD D / JNC top' for a BY step. So `DO b = 0 TO 255' runs 256
        # times and leaves b = 0, and a step that would wrap past the limit
        # stops the loop instead. A step of 0FFH (or -1) is 255, not a
        # count downwards (5.1.4).
        if both_bytes:
            bound_val = (number_value(stmt.bound) & 0xFF
                         if isinstance(stmt.bound, P.NumberLiteral) else None)
            step_byte = step_val & 0xFF if step_is_const else None

            # Initialize index variable
            start_type = self._gen_expr(stmt.start)
            if start_type == DataType.ADDRESS:
                self._emit("ld", "a,l")
            self._gen_store(index_var, DataType.BYTE)

            # A limit of 255 passes every index, so there is nothing to test.
            if bound_val != 0xFF:
                self._emit("jp", test_label)

            # Loop body
            self._emit_label(loop_label)
            for s in body_stmts:
                self._gen_stmt(s)

            # Increment, and stop when it carries out.
            self._emit_label(incr_label)
            self._gen_load(index_var)  # A = index
            if not step_is_const:
                # The step is an expression: keep the index while it runs.
                self._emit("push", "af")
                if self._gen_expr(step_expr) == DataType.ADDRESS:
                    self._emit("ld", "a,l")
                self._emit("ld", "b,a")
                self._emit("pop", "af")
                self._emit("add", "a,b")
                wrap = "c"
            elif step_byte == 1:
                self._emit("inc", "a")
                wrap = "z"
            else:
                self._emit("add", f"a,{self._format_number(step_byte)}")
                wrap = "c"
            self._gen_store(index_var, DataType.BYTE)
            # With a constant limit the index is at most the limit before
            # the step, so if limit + step fits in a byte it cannot wrap.
            can_wrap = (bound_val is None or not step_is_const
                        or bound_val + step_byte > 0xFF)
            if bound_val == 0xFF:
                self._emit("jp", f"n{wrap},{loop_label}")
                self._emit_label(test_label)
            else:
                if can_wrap:
                    self._emit("jp", f"{wrap},{end_label}")
                self._emit_label(test_label)
                if bound_val is not None:
                    self._gen_load(index_var)  # A = index
                    self._emit("cp", self._format_number(bound_val + 1))
                    self._emit("jp", f"C,{loop_label}")  # index <= limit
                else:
                    # The limit is evaluated on every pass, converted to a
                    # BYTE. Continue while index <= limit.
                    bound_result = self._gen_expr(stmt.bound)
                    if bound_result == DataType.ADDRESS:
                        self._emit("ld", "a,l")
                    self._emit("ld", "b,a")  # B = limit
                    self._gen_load(index_var)  # A = index
                    self._emit("cp", "b")
                    self._emit("jp", f"c,{loop_label}")
                    self._emit("jp", f"z,{loop_label}")

            self._emit_label(end_label)
            self.loop_stack.pop()
            return

        # General case: 16-bit loop.  The index still has its own declared
        # width: _gen_load brings a BYTE back in A, not HL, and the arithmetic
        # below is all 16-bit, so it has to be widened.  Without that the
        # increment did `inc hl' on whatever the body had left in HL and the
        # test compared against it - UTIL4/SHOW.PLM's `do i = 0 to last(user)'
        # never terminated, and `show users:' printed until the session died.
        def _index_to_hl() -> None:
            self._gen_load(index_var)
            if index_type == DataType.BYTE:
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")

        bound_val = (number_value(stmt.bound) & 0xFFFF
                     if isinstance(stmt.bound, P.NumberLiteral) else None)
        step_word = step_val & 0xFFFF if step_is_const else None

        # Initialize index variable
        self._gen_expr_to_hl(stmt.start)
        self._gen_store(index_var, DataType.ADDRESS)

        # Jump to test (a limit of 0FFFFH passes every index)
        if bound_val != 0xFFFF:
            self._emit("jp", test_label)

        # Loop body
        self._emit_label(loop_label)
        for s in body_stmts:
            self._gen_stmt(s)

        # Increment, and stop when it carries out.
        self._emit_label(incr_label)
        _index_to_hl()
        if not step_is_const:
            self._emit("push", "hl")
            self._gen_expr_to_hl(step_expr)
            self._emit("ex", "de,hl")
            self._emit("pop", "hl")
            self._emit("add", "hl,de")
            wrap = "c"
        elif step_word == 1:
            self._emit("inc", "hl")
            wrap = "z"
        else:
            # BC, not DE: the peephole turns `ld de,1..3 / add hl,de' into
            # `inc hl', which sets no carry.
            can_carry = bound_val is None or bound_val + step_word > 0xFFFF
            pair = "bc" if can_carry else "de"
            self._emit("ld", f"{pair},{self._format_number(step_word)}")
            self._emit("add", f"hl,{pair}")
            wrap = "c"
        self._gen_store(index_var, DataType.ADDRESS)
        can_wrap = (bound_val is None or not step_is_const
                    or bound_val + step_word > 0xFFFF)
        if can_wrap:
            if wrap == "z":
                # `inc hl' sets no flags: the index wrapped if it is now 0.
                self._emit("ld", "a,h")
                self._emit("or", "l")
            if bound_val == 0xFFFF:
                self._emit("jp", f"n{wrap},{loop_label}")
                self._emit_label(test_label)
                self._emit_label(end_label)
                self.loop_stack.pop()
                return
            self._emit("jp", f"{wrap},{end_label}")

        # Test condition
        self._emit_label(test_label)
        _index_to_hl()
        if self._expr_preserves_de(stmt.bound):
            self._emit("ex", "de,hl")        # DE = index
            self._gen_expr_to_hl(stmt.bound)  # HL = bound, DE untouched
        else:
            # The bound is free to use DE (a call, a 16-bit subexpression,
            # `ld de,nn`), so the index has to survive on the stack.
            self._emit("push", "hl")
            self._gen_expr_to_hl(stmt.bound)  # HL = bound
            self._emit("pop", "de")           # DE = index

        # Compare: if index > bound, exit (for positive step)
        # HL - DE: if negative (carry), index > bound
        self._emit_sub16()

        # If no borrow (NC), bound >= index, continue
        self._emit("jp", f"nc,{loop_label}")

        self._emit_label(end_label)
        self.loop_stack.pop()

    def _read_outside(self, name: str, loop) -> bool:
        """Whether ``name`` may be read anywhere but inside ``loop``: it is
        not a variable of the procedure being compiled, or the procedure
        (nested procedures included) names it somewhere else."""
        if not self._is_private_var(name) or self.current_proc_decl is None:
            return True
        stack = [self.current_proc_decl.body]
        while stack:
            n = stack.pop()
            if n is loop:
                continue
            if isinstance(n, P.Identifier) and ident_text(n.name) == name:
                return True
            if isinstance(n, (P.DoIterBlock, P.DoIterByBlock)) and ident_text(n.index) == name:
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

    _FIXED_BUILTINS = frozenset({"LOW", "HIGH", "DOUBLE", "SHL", "SHR", "ROL", "ROR"})

    def _limit_is_fixed(self, bound, body_stmts) -> bool:
        """Whether a loop limit has the same value on every pass: constants
        and variables the body cannot change, and no call in it."""
        stack = [bound]
        while stack:
            e = unwrap_paren(stack.pop())
            if isinstance(e, (P.NumberLiteral, P.StringLiteral)):
                continue
            if isinstance(e, P.Identifier):
                name = ident_text(e.name)
                if name in self.literal_macros:
                    continue
                sym = self._lookup_symbol(name)
                if (sym is None or sym.kind not in (SymbolKind.VARIABLE, SymbolKind.PARAMETER)
                        or sym.dimension is not None
                        or self._stmts_may_change(name, body_stmts)):
                    return False
                continue
            if isinstance(e, P.BinaryOp):
                stack.extend([e.left, e.right])
                continue
            if isinstance(e, P.UnaryOp):
                stack.append(e.operand)
                continue
            if isinstance(e, P.Call):
                callee = unwrap_paren(e.callee)
                if (isinstance(callee, P.Identifier)
                        and ident_text(callee.name).upper() in self._FIXED_BUILTINS):
                    stack.extend(e.args)
                    continue
            return False
        return True

    def _survey_stores(self, modules) -> None:
        """Record, for every procedure, the names its own body assigns, and
        the names whose address is taken (``.x'') or which are placed AT
        something anywhere. A call can change a variable only if some
        procedure assigns it, or stores through its address."""
        self._proc_assigns: dict[int, set[str]] = {}
        self._assigned_by_procs: dict[str, int] = {}
        self._aliased: set[str] = set()

        def assigned_in(body) -> set[str]:
            names: set[str] = set()
            stack = [body]
            while stack:
                n = stack.pop()
                if isinstance(n, P.ProcDecl):
                    continue            # a procedure of its own
                targets = []
                if isinstance(n, P.AssignStmt):
                    targets = list(n.targets)
                elif isinstance(n, P.EmbeddedAssign):
                    targets = [n.target]
                elif isinstance(n, (P.DoIterBlock, P.DoIterByBlock)):
                    names.add(ident_text(n.index))
                for t in targets:
                    t = unwrap_paren(t)
                    if isinstance(t, P.Identifier):
                        names.add(ident_text(t.name))
                if isinstance(n, (list, tuple)):
                    stack.extend(n)
                    continue
                fields = getattr(n, "__dataclass_fields__", None)
                if fields:
                    stack.extend(getattr(n, f, None) for f in fields if f != "pos")
            return names

        stack: list = list(modules)
        while stack:
            n = stack.pop()
            if isinstance(n, P.ProcDecl):
                own = assigned_in(n.body.items)
                self._proc_assigns[id(n)] = own
                for name in own:
                    self._assigned_by_procs[name] = self._assigned_by_procs.get(name, 0) + 1
            elif isinstance(n, P.LocationOf):
                base = unwrap_paren(n.operand)
                while isinstance(base, (P.Call, P.MemberAccess)):
                    base = unwrap_paren(base.callee if isinstance(base, P.Call) else base.base)
                if isinstance(base, P.Identifier):
                    self._aliased.add(ident_text(base.name))
            elif isinstance(n, P.DeclItem) and decl_attrs(n).at_location is not None:
                self._aliased.update(decl_item_names(n))
            if isinstance(n, (list, tuple)):
                stack.extend(n)
                continue
            fields = getattr(n, "__dataclass_fields__", None)
            if fields:
                stack.extend(getattr(n, f, None) for f in fields if f != "pos")

    def _assigned_by_another_proc(self, name: str) -> bool:
        """Whether a procedure other than the one being compiled assigns
        ``name`` (its bare name: an over-approximation across scopes)."""
        count = getattr(self, "_assigned_by_procs", {}).get(name, 0)
        if self.current_proc_decl is not None:
            if name in self._proc_assigns.get(id(self.current_proc_decl), ()):
                count -= 1
        return count > 0

    def _call_may_change(self, name: str) -> bool:
        """Whether a call of a procedure outside the one being compiled can
        change the variable ``name``."""
        sym = self._lookup_symbol(name)
        if (sym is None or sym.based_on or sym.at_address is not None
                or sym.is_public or sym.is_external
                or name in getattr(self, "_aliased", ())):
            return True
        return self._assigned_by_another_proc(name)

    def _is_private_var(self, name: str) -> bool:
        """``name`` is a plain variable of the procedure being compiled,
        which only it and the procedures nested in it can reach."""
        if not self.current_proc:
            return False
        # Up through the DO-block scopes (B<n>) to the procedure's own.
        sym = None
        scope = self.symbols.current_scope
        while scope is not None and scope.parent is not None:
            sym = scope.symbols.get(name)
            if sym is not None or not re.fullmatch(r"B\d+", scope.name):
                break
            scope = scope.parent
        if sym is None:
            sym = self.symbols.lookup(f"{self.current_proc}${name}")
        # A variable whose address is taken can be stored into from
        # anywhere the address went.
        return (sym is not None
                and sym.kind in (SymbolKind.VARIABLE, SymbolKind.PARAMETER)
                and not sym.based_on and sym.at_address is None
                and not sym.is_public and not sym.is_external
                and name not in getattr(self, "_aliased", ()))

    def _stmts_may_change(self, name: str, stmts) -> bool:
        """Whether ``stmts`` may assign ``name``: an assignment or embedded
        assignment to it, a DO over it, or a call of a procedure that can
        reach it -- any procedure for a global, a nested one for a variable
        of the procedure being compiled -- or of MOVE."""
        private = self._is_private_var(name)
        nested_prefix = f"{self.current_proc}$"
        stack = list(stmts)
        while stack:
            n = stack.pop()
            if isinstance(n, P.CallStmt):
                callee = n.callee
                if isinstance(callee, (P.Call, P.CallNoArgs)):
                    callee = callee.callee
                callee = unwrap_paren(callee)
                cname = ident_text(callee.name) if isinstance(callee, P.Identifier) else ""
                sym = self._lookup_symbol(cname) if cname else None
                if cname.upper() == "MOVE":
                    if name in getattr(self, "_aliased", ()) or not private:
                        return True
                elif (sym is not None and sym.name.startswith(nested_prefix)
                        or (not private and self._call_may_change(name))):
                    return True
            if isinstance(n, P.AssignStmt):
                for t in n.targets:
                    t = unwrap_paren(t)
                    if isinstance(t, P.Identifier) and ident_text(t.name) == name:
                        return True
            elif isinstance(n, P.EmbeddedAssign):
                t = unwrap_paren(n.target)
                if isinstance(t, P.Identifier) and ident_text(t.name) == name:
                    return True
            elif isinstance(n, (P.DoIterBlock, P.DoIterByBlock)):
                if ident_text(n.index) == name:
                    return True
            elif isinstance(n, (P.Identifier, P.Call, P.CallNoArgs)):
                callee = n if isinstance(n, P.Identifier) else unwrap_paren(n.callee)
                if isinstance(callee, P.Identifier):
                    sym = self._lookup_symbol(ident_text(callee.name))
                    if (sym is not None and sym.kind == SymbolKind.PROCEDURE
                            and (sym.name.startswith(nested_prefix)
                                 or (not private and self._call_may_change(name)))):
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
            # Search the enclosing scopes: a nested procedure's symbol is filed
            # under its scoped name, so a plain lookup misses it and everything
            # it returns was typed ADDRESS.  A bare name in an expression is a
            # CALL in PL/M-80, so a BYTE result was then read out of L instead
            # of A - UTIL4/STAT.PLM's getfile calls its own nested
            # `setfilestatus' that way.
            sym = self._lookup_symbol(name)
            if sym and sym.kind != SymbolKind.BUILTIN:
                if sym.kind == SymbolKind.PROCEDURE:
                    return sym.return_type or DataType.ADDRESS
                return sym.data_type or DataType.ADDRESS
            # The condition flags, read without parentheses, are BYTE
            # procedures (12.5) -- unless a variable of the name hides them.
            if name.upper() in ('CARRY', 'ZERO', 'SIGN', 'PARITY'):
                return DataType.BYTE
            if sym:
                return sym.data_type or DataType.ADDRESS
            return DataType.ADDRESS
        elif isinstance(expr, P.EmbeddedAssign):
            # "The value of the embedded assignment is the same as that of
            # its right half" (PL/M-80 manual 4.6.3): `(b := w)' is w, all
            # sixteen bits of it, whatever b keeps.
            return self._get_expr_type(expr.value)
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
                # PLUS and MINUS "perform similarly to + and -" (12.2).
                if op in (
                    BinaryOpKind.ADD, BinaryOpKind.SUB,
                    BinaryOpKind.AND, BinaryOpKind.OR, BinaryOpKind.XOR,
                    BinaryOpKind.PLUS, BinaryOpKind.MINUS,
                ):
                    return DataType.BYTE
            return DataType.ADDRESS
        elif isinstance(expr, (P.LocationOf, P.LocationOfList, P.LocationOfString)):
            return DataType.ADDRESS
        elif isinstance(expr, (P.Call, P.CallNoArgs)):
            callee = unwrap_paren(expr.callee)
            # An element of an array member, `s.m(i)': the member's type, as
            # _gen_call_expr loads it. It was typed ADDRESS, so a BYTE
            # element was compared and added in sixteen bits.
            if (isinstance(callee, P.MemberAccess) and isinstance(expr, P.Call)
                    and len(expr.args) == 1):
                _, member_type = self._get_member_info(callee)
                return member_type
            if isinstance(callee, P.Identifier):
                name = ident_text(callee.name).upper()
                # Built-ins first, as _gen_call_expr dispatches them.
                builtin_type = self._builtin_type(name, expr)
                if builtin_type is not None:
                    return builtin_type
                sym = self._lookup_symbol(ident_text(callee.name))
                if sym:
                    if sym.kind == SymbolKind.PROCEDURE:
                        return sym.return_type or DataType.ADDRESS
                    # A subscripted variable is an element, which
                    # _gen_subscript loads as a BYTE unless the variable is
                    # an ADDRESS -- an untyped `DECLARE hex DATA ('0123')'
                    # too, which was typed ADDRESS here, so `hex(i) + 0FFH'
                    # added the BYTE in A to whatever HL held.
                    if (sym.kind in (SymbolKind.VARIABLE, SymbolKind.PARAMETER)
                            and isinstance(expr, P.Call) and len(expr.args) == 1):
                        return (DataType.ADDRESS if sym.data_type == DataType.ADDRESS
                                else DataType.BYTE)
                    if sym.dimension is not None:
                        return sym.data_type or DataType.BYTE
                    return sym.data_type or DataType.ADDRESS
            return DataType.ADDRESS
        elif isinstance(expr, P.UnaryOp):
            # NEG / NOT preserve operand type (LOW/HIGH are now Calls).
            return self._get_expr_type(expr.operand)
        elif isinstance(expr, P.MemberAccess):
            # Resolve the member's declared type from the structure
            # layout rather than assuming BYTE — a STRUCTURE member can
            # be ADDRESS (e.g. ``rec.len`` where ``len ADDRESS``), and
            # mis-typing it as BYTE makes range checks like
            # ``rec.len <= 1025`` look impossible.
            _, member_type = self._get_member_info(expr)
            return member_type
        return DataType.ADDRESS

    def _builtin_type(self, name: str, expr) -> DataType | None:
        """The type of a call to built-in ``name`` (upper case), or None.

        SCL and SCR have the type of their pattern (12.3), and LENGTH and
        LAST are BYTE when the value fits in one (11.1.2). SHL and SHR are
        always ADDRESS here: the manual gives them their pattern's type
        (11.1.4), but uplm80 has always shifted a BYTE pattern as a
        zero-extended ADDRESS, and programs written for it -- 80un's
        `lo + shl(b, 8)' -- depend on that (see plm_types).
        """
        if name in BYTE_BUILTINS:
            return DataType.BYTE
        if name in ('DOUBLE', 'SIZE', 'STACKPTR', 'TIME', 'CPUTIME', 'SHL', 'SHR'):
            return DataType.ADDRESS
        args = expr.args if isinstance(expr, P.Call) else []
        if name in PATTERN_TYPED_BUILTINS and args:
            return self._get_expr_type(args[0])
        if name in ('LENGTH', 'LAST'):
            extent = self._array_extent(args[0]) if args else None
            if extent is None:
                return DataType.ADDRESS
            value = extent if name == 'LENGTH' else extent - 1
            return DataType.BYTE if value <= 0xFF else DataType.ADDRESS
        return None

    def _array_extent(self, arg) -> int | None:
        """The declared extent of the array ``arg`` names, if known."""
        arg = unwrap_paren(arg)
        if isinstance(arg, P.Identifier):
            sym = self._lookup_scoped(ident_text(arg.name))
            if sym and sym.dimension:
                return sym.dimension
        return None

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

    def _expr_preserves_hl(self, expr) -> bool:
        """Check whether generating this BYTE expression leaves ``HL`` intact.

        Only a byte load out of plain memory qualifies: ``ld a,(name)``
        or ``ld a,n``. A based variable loads its base through ``HL``, a
        subscript computes an address in ``HL``, and a call may return
        in ``HL``.
        """
        expr = unwrap_paren(expr)
        if isinstance(expr, P.NumberLiteral):
            # `ld a,n` via _gen_expr_to_a leaves HL alone, but plain
            # _gen_expr would emit `ld hl,n`. Safe only for the former.
            return True
        if isinstance(expr, P.Identifier):
            name = ident_text(expr.name)
            if name in self.literal_macros:
                return True
            sym = self._lookup_symbol(name)
            if sym is None or sym.kind == SymbolKind.PROCEDURE:
                return False
            if sym.based_on or sym.data_type != DataType.BYTE:
                return False
            return True
        if isinstance(expr, P.UnaryOp):
            return self._expr_preserves_hl(expr.operand)
        return False

    def _expr_preserves_b(self, expr) -> bool:
        """Check whether generating this expression leaves ``B`` intact.

        ``B`` is not callee-saved, and it is also the scratch register
        the byte binary ops and byte comparisons park an operand in, so
        anything that can emit a call, a shift (``B`` is the counter),
        a nested comparison or a byte binary op destroys it. Only a
        literal, a plain variable, or a unary op over one is safe.
        """
        expr = unwrap_paren(expr)
        if isinstance(expr, (P.NumberLiteral, P.StringLiteral)):
            return True
        if isinstance(expr, P.Identifier):
            name = ident_text(expr.name)
            if name in self.literal_macros:
                return True
            sym = self._lookup_symbol(name)
            # A bare typed-procedure reference is a call.
            return not (sym and sym.kind == SymbolKind.PROCEDURE)
        if isinstance(expr, P.UnaryOp):
            return self._expr_preserves_b(expr.operand)
        return False

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
                if sym.based_on and sym.data_type != DataType.BYTE:
                    # A BASED ADDRESS load is `ld hl,(base) / ld e,(hl) /
                    # inc hl / ld d,(hl) / ex de,hl`: it writes DE and
                    # leaves base+1 there. A BASED BYTE is safe - it only
                    # does `ld hl,(base) / ld a,(hl)`.
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

    def _based_ptr_operand(self, sym) -> str:
        """Where the pointer behind a BASED variable lives.

        `x BASED p' keeps it in p.  `x BASED s.m' keeps it in a MEMBER of s, so
        the address is s + that member's offset - MP/M II's UTIL7/DM.PLM
        declares `token BASED pcb.token$adr (12) byte', and reading the pointer
        from the start of `pcb' gave it `pcb.state' instead, which is zero: SDIR
        matched its command-line file specification against whatever was at
        address 0 and answered "File Not Found." to every argument.
        """
        base_sym = self.symbols.lookup(sym.based_on)
        base_asm = (base_sym.asm_name if base_sym and base_sym.asm_name
                    else self._mangle_name(sym.based_on))
        member = getattr(sym, "based_member", None)
        if not member or not (base_sym and base_sym.struct_members):
            return base_asm
        offset = 0
        for m in base_sym.struct_members:
            if m.name == member:
                break
            width = 1 if m.data_type == DataType.BYTE else 2
            offset += width * (m.dimension or 1)
        else:
            return base_asm
        return base_asm if offset == 0 else f"{base_asm}+{offset}"

    def _gen_expr(self, expr) -> DataType:
        """Generate code for a typed expression.

        Result is left in A (for BYTE) or HL (for ADDRESS).
        Returns the type of the expression.
        """
        expr = unwrap_paren(expr)
        # a_has_l says that A holds L of the value generated last: an
        # embedded assignment of an ADDRESS to a BYTE sets it when it
        # returns, and LOW reads it straight after generating its operand.
        # Anything generated since makes it stale -- `(b := w * 5) XOR
        # LOW(LAST(a))' took A for LAST's L.
        self.a_has_l = False

        if isinstance(expr, P.NumberLiteral):
            self._emit("ld", f"hl,{self._format_number(number_value(expr))}")
            return DataType.ADDRESS

        elif isinstance(expr, P.StringLiteral):
            s = string_value(expr)
            if len(s) == 1:
                self._emit("ld", f"a,{self._format_number(ord(s[0]))}")
                return DataType.BYTE
            if len(s) == 2:
                # A two-character string is an ADDRESS constant, the first
                # character in the high byte (4.1.1) -- not a pointer: that
                # is `.('AB')'.
                value = (ord(s[0]) << 8) | ord(s[1])
                self._emit("ld", f"hl,{self._format_number(value)}")
                return DataType.ADDRESS
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
            if self._get_const_byte_value(expr.value) is not None:
                # A BYTE constant is a BYTE here too, in A: generated as a
                # literal it came back in HL, and code that goes by the
                # static type -- a BYTE subscript -- looked for it in A.
                self._gen_expr_to_a(expr.value)
                val_type = DataType.BYTE
            else:
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
                    # Spill through the stack, not through B: storing to a
                    # subscripted or based target generates the index
                    # expression, which is free to call a procedure and
                    # clobber B. `push af`/`pop af` is the same two bytes
                    # as `ld b,a`/`ld a,b`, and the ADDRESS path below
                    # already spills the same way.
                    self._emit("push", "af")
                    self._gen_store(target, val_type)
                    self._emit("pop", "af")
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

            # A variable of the same name shadows a CONDITION-FLAG built-in.
            # CARRY, ZERO, SIGN and PARITY are ordinary words that a program
            # may well use for something of its own: MP/M II's UTIL4/STAT.PLM
            # has `declare (d,zero) byte' for its zero-suppression flag, and
            # reading the Z flag in its place made every number print with
            # leading zeros - `(00001 file, 00001-1k blocks)'.
            #
            # STACKPTR is deliberately not in this list.  A program assigns to
            # it to SET the stack pointer - UTIL3/LOAD.PLM has `STACKPTR = SP'
            # - which registers a variable of that name as a side effect, so
            # shadowing on it would break every later read.
            _shadow_sym = self._lookup_symbol(name)
            shadowed = (_shadow_sym is not None
                        and _shadow_sym.kind != SymbolKind.BUILTIN)

            # Handle built-in STACKPTR variable
            if upper_name == "STACKPTR":
                # Read stack pointer into HL
                self._emit("ld", "hl,0")
                self._emit("add", "hl,sp")  # HL = HL + SP = SP
                return DataType.ADDRESS

            # Handle flag-testing builtins (can be used without parentheses)
            if upper_name == "CARRY" and not shadowed:
                # Return carry flag value. `sbc a,a` reads carry in one
                # instruction (A := -carry, so 0FFH or 00H) and does not
                # depend on A's previous contents. The obvious
                # `ld a,0 / rla` cannot be used: the peephole rewrites
                # `ld a,0` into the one-byte `xor a`, which CLEARS the
                # very flag being read, and CARRY then always reads 0.
                # CARRY is 0FFH when the flag is set (12.5), and `sbc a,a'
                # leaves the flag as it found it for a later PLUS or CARRY.
                self._emit("sbc", "a,a")
                self._emit("ld", "l,a")
                self._emit("ld", "h,0")
                return DataType.BYTE

            if upper_name == "ZERO" and not shadowed:
                # Return zero flag value
                end_label = self._new_label("ZFE")

                # `ld a,0ffh` sets no flags, so the condition survives to the

                # branch; the false path turns it into 0 with `inc a`, which

                # leaves CARRY alone. Loading zero directly would be rewritten

                # by the peephole into `xor a`, and that clears carry for any

                # later CARRY read.

                self._emit("ld", "a,0ffh")

                self._emit("jp", f"z,{end_label}")

                self._emit("inc", "a")

                self._emit_label(end_label)
                return DataType.BYTE

            if upper_name == "SIGN" and not shadowed:
                # Return sign flag value
                end_label = self._new_label("SFE")

                # `ld a,0ffh` sets no flags, so the condition survives to the

                # branch; the false path turns it into 0 with `inc a`, which

                # leaves CARRY alone. Loading zero directly would be rewritten

                # by the peephole into `xor a`, and that clears carry for any

                # later CARRY read.

                self._emit("ld", "a,0ffh")

                self._emit("jp", f"m,{end_label}")

                self._emit("inc", "a")

                self._emit_label(end_label)
                return DataType.BYTE

            if upper_name == "PARITY" and not shadowed:
                # Return parity flag value
                end_label = self._new_label("PFE")

                # `ld a,0ffh` sets no flags, so the condition survives to the

                # branch; the false path turns it into 0 with `inc a`, which

                # leaves CARRY alone. Loading zero directly would be rewritten

                # by the peephole into `xor a`, and that clears carry for any

                # later CARRY read.

                self._emit("ld", "a,0ffh")

                self._emit("jp", f"pe,{end_label}")

                self._emit("inc", "a")

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
                    self._emit("ld", f"hl,({self._based_ptr_operand(sym)})")
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
                    self._emit("ld", f"hl,({self._based_ptr_operand(sym)})")
                    self._emit("ld", "a,b")  # Restore value
                    self._emit("ld", "(hl),a")  # Store via HL
                else:
                    # A BYTE value is in A: widen it into HL, which the
                    # store below takes the value from.
                    if val_type == DataType.BYTE:
                        self._emit("ld", "l,a")
                        self._emit("ld", "h,0")
                    # Save value in HL
                    self._emit("push", "hl")
                    self._emit("ld", f"hl,({self._based_ptr_operand(sym)})")
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
            # Structure member store. The value is in A for a BYTE and in
            # HL for an ADDRESS; the old code always saved and reloaded HL,
            # so `rec.f = ch` stored whatever happened to be in L.
            _, member_type = self._get_member_info(expr)
            if val_type == DataType.BYTE:
                self._emit("push", "af")
                self._gen_member_addr(expr)  # HL = member address
                self._emit("pop", "af")      # A = value
                self._emit("ld", "(hl),a")
                if member_type == DataType.ADDRESS:
                    # PL/M zero-extends a BYTE into an ADDRESS member.
                    self._emit("inc", "hl")
                    self._emit("ld", "(hl),0")
            else:
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
                if port_num is not None:
                    port_num &= 0xFF        # the port is a BYTE
                # A BYTE value is in A already; `ld a,l' replaced it with
                # whatever L held.
                if val_type != DataType.BYTE:
                    self._emit("ld", "a,l")
                if port_num is not None:
                    self._emit("out", f"({self._format_number(port_num)}),a")
                else:
                    self._emit("push", "af")
                    self._gen_expr_to_a(port_arg)
                    self._emit("ld", "c,a")
                    self._emit("pop", "af")
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
                # The subscript is an expression like any other: `MEMORY(-1)'
                # is MEMORY(0FFH), -1 being a BYTE (4.2.2).
                typed = eval_typed(addr_arg, self._literal_macro_value)
                addr_val = None if typed is None else typed[0]
                if addr_val is not None:
                    if val_type != DataType.BYTE:
                        self._emit("ld", "a,l")
                    if addr_val == 0:
                        self._emit("ld", "(__END__),a")
                    else:
                        self._emit("ld", f"(__END__+{self._format_number(addr_val)}),a")
                else:
                    # A BYTE subscript is generated into A: widen it into
                    # HL, where it is added to __END__.
                    if val_type == DataType.BYTE:
                        self._emit("push", "af")
                        self._gen_expr_to_hl(addr_arg)
                        self._emit("ld", "de,__END__")
                        self._emit("add", "hl,de")
                        self._emit("pop", "af")
                        self._emit("ld", "(hl),a")
                    else:
                        self._emit("push", "hl")
                        self._gen_expr_to_hl(addr_arg)
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
        expr = self._without_redundant_double(self._narrowed_comparison(expr))
        op = binop_kind(expr)
        left = unwrap_paren(expr.left)
        right = unwrap_paren(expr.right)

        # Special case: SHL(DOUBLE(hi), 8) OR lo -> combine two bytes into address
        if op == BinaryOpKind.OR:
            hi_expr = self._match_shl_double_8(left)
            if hi_expr is not None:
                lo_type = self._get_expr_type(right)
                if lo_type == DataType.BYTE:
                    self._gen_expr_to_a(hi_expr)
                    if self._expr_preserves_hl(right):
                        self._emit("ld", "h,a")
                        self._gen_expr_to_a(right)
                        self._emit("ld", "l,a")
                    else:
                        # The low operand may compute in HL, so the high
                        # byte cannot sit in H across it.
                        self._emit("push", "af")
                        self._gen_expr_to_a(right)
                        self._emit("ld", "l,a")
                        self._emit("pop", "af")
                        self._emit("ld", "h,a")
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
                    if val <= 255:
                        const_val = val
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

        if both_bytes and op in (BinaryOpKind.PLUS, BinaryOpKind.MINUS):
            return self._gen_byte_carry_op(left, right, op)

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

        # Paths 2 and 3 claim DE only where they keep a value in it; a
        # release without its claim would restore someone else's spill.
        claimed_de = False

        # Path 1: left is simple AND DE is free
        if self._expr_preserves_de(left) and self.regs.is_free('de'):
            right_const = self._get_const_byte_value(right)
            if right_const is not None:
                # LENGTH or LAST, say, which is generated into A.
                self._emit("ld", f"de,{self._format_number(right_const)}")
            elif self._gen_expr(right) == DataType.BYTE:
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
                claimed_de = True
                self._emit("ex", "de,hl")
                left_result = self._gen_expr(left)
                if left_result == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")

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
                claimed_de = True
                self._emit("ex", "de,hl")
                right_result = self._gen_expr(right)
                if right_result == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("ex", "de,hl")


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
            if claimed_de:
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

        if claimed_de:
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
        false_label = self._new_label("FALSE")
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
            # Both false paths have to reach the "xor a" below.  Jumping to
            # end_label instead skips it, so the comparison yields whatever the
            # cp/sub left in A and a false ">" reads as true.
            self._emit("jp", f"c,{false_label}")
            self._emit("jp", f"z,{false_label}")
            self._emit("jp", true_label)
        elif op == BinaryOpKind.LE:
            self._emit("jp", f"c,{true_label}")
            self._emit("jp", f"z,{true_label}")

        self._emit_label(false_label)
        self._emit("xor", "a")
        self._emit("jp", end_label)

        self._emit_label(true_label)
        self._emit("ld", "a,0ffh")

        self._emit_label(end_label)
        return DataType.BYTE

    def _gen_byte_comparison(self, left, right, op: BinaryOpKind) -> DataType:
        """Generate optimized byte comparison between two byte values."""
        # _gen_expr_to_a, not _gen_expr: a NumberLiteral operand loads as
        # `ld hl,n` and would leave A undefined under the `sub b`, so
        # `r = 5 > x` compared garbage.
        self._gen_expr_to_a(left)
        # B is not safe across the other operand: a nested byte comparison uses
        # "ld b,a" as its own scratch move, and any procedure call clobbers B.
        # So spill A through the stack and load B only after the other operand
        # has been generated.  "pop bc" would be shorter than "ld b,a" + "pop
        # af", but "pop bc" also overwrites C, and the CP/M call convention
        # keeps a live argument there.
        self._emit("push", "af")

        self._gen_expr_to_a(right)
        self._emit("ld", "b,a")
        self._emit("pop", "af")
        self._emit("sub", "b")

        true_label = self._new_label("TRUE")
        false_label = self._new_label("FALSE")
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
            # Both false paths have to reach the "xor a" below.  Jumping to
            # end_label instead skips it, so the comparison yields whatever the
            # cp/sub left in A and a false ">" reads as true.
            self._emit("jp", f"c,{false_label}")
            self._emit("jp", f"z,{false_label}")
            self._emit("jp", true_label)
        elif op == BinaryOpKind.LE:
            self._emit("jp", f"c,{true_label}")
            self._emit("jp", f"z,{true_label}")

        self._emit_label(false_label)
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
            # c - x, with the borrow a subtraction leaves. (`1 - x' was
            # `x XOR 1', which is 1 - x only for x = 0 or 1: 1 - 0FFH is 2,
            # not 0FEH; and `-x + c' left the carry inverted.)
            self._gen_expr_to_a(right)
            self._emit("ld", "b,a")
            self._emit("ld", f"a,{self._format_number(left_const)}")
            self._emit("sub", "b")
            return DataType.BYTE

        if op == BinaryOpKind.SUB:
            self._gen_expr_to_a(left)
            self._emit("push", "af")          # B is not safe; see below
            self._gen_expr_to_a(right)
            self._emit("ld", "b,a")
            self._emit("pop", "af")
            self._emit("sub", "b")
            return DataType.BYTE

        self._gen_expr_to_a(left)
        # B is not safe across the other operand: a nested byte comparison uses
        # "ld b,a" as its own scratch move, and any procedure call clobbers B.
        # So spill A through the stack and load B only after the other operand
        # has been generated.  "pop bc" would be shorter than "ld b,a" + "pop
        # af", but "pop bc" also overwrites C, and the CP/M call convention
        # keeps a live argument there.
        self._emit("push", "af")

        self._gen_expr_to_a(right)
        self._emit("ld", "b,a")
        self._emit("pop", "af")

        if op == BinaryOpKind.ADD:
            self._emit("add", "a,b")
        elif op == BinaryOpKind.AND:
            self._emit("and", "b")
        elif op == BinaryOpKind.OR:
            self._emit("or", "b")
        elif op == BinaryOpKind.XOR:
            self._emit("xor", "b")

        return DataType.BYTE

    def _gen_byte_carry_op(self, left, right, op: BinaryOpKind) -> DataType:
        """BYTE PLUS / MINUS BYTE: an 8-bit `adc' / `sbc' with a BYTE result.

        PLUS and MINUS "perform similarly to + and -" (12.2), so two BYTEs
        give a BYTE, taking in whatever carry the left operand's evaluation
        left; `push af' / `pop af' carry the flags across the right one.
        """
        mnemonic = "adc" if op == BinaryOpKind.PLUS else "sbc"
        right_const = self._get_const_byte_value(right)
        self._gen_expr_to_a(left)
        if right_const is not None:
            self._emit(mnemonic, f"a,{self._format_number(right_const)}")
            return DataType.BYTE
        self._emit("push", "af")
        self._gen_expr_to_a(right)
        self._emit("ld", "b,a")
        self._emit("pop", "af")
        self._emit(mnemonic, "a,b")
        return DataType.BYTE

    def _gen_expr_to_hl(self, expr) -> None:
        """Generate an expression into ``HL``, widening a byte result.

        Widening keys off the type ``_gen_expr`` actually returned, not the
        statically inferred one: a NumberLiteral that ``_get_expr_type``
        calls BYTE still loads as ``ld hl,n``, and widening that with
        ``ld l,a`` would splice in an undefined ``A``.
        """
        const_val = self._get_const_byte_value(expr)
        if const_val is not None:
            self._emit("ld", f"hl,{self._format_number(const_val)}")
            return
        if self._gen_expr(expr) == DataType.BYTE:
            self._emit("ld", "l,a")
            self._emit("ld", "h,0")

    def _unwidened(self, expr):
        """``x`` when ``expr`` is ``DOUBLE(x)`` of a BYTE ``x``, else None.

        The optimizer keeps a strength-reduced quotient or remainder ADDRESS
        by wrapping it in DOUBLE (``x MOD 8`` is ``DOUBLE(x AND 7)``), since
        in PL/M-80 one is an ADDRESS even of BYTE operands. That width only
        shows where the value meets 16-bit arithmetic; a use that reads its
        low byte, or compares it with another BYTE, gets the same answer
        from ``x`` itself.
        """
        expr = unwrap_paren(expr)
        if isinstance(expr, P.Call) and len(expr.args) == 1:
            callee = unwrap_paren(expr.callee)
            if (isinstance(callee, P.Identifier)
                    and ident_text(callee.name).upper() == 'DOUBLE'
                    and self._get_expr_type(expr.args[0]) == DataType.BYTE):
                return expr.args[0]
        return None

    def _low_byte_form(self, expr):
        """``expr`` for a use that reads only its low byte (or bit 0).

        The low byte of a sum, difference, bitwise result or complement
        depends only on the low bytes of the operands, so a DOUBLE anywhere
        in that tree can go.
        """
        inner = self._unwidened(expr)
        if inner is not None:
            return self._low_byte_form(inner)
        e = unwrap_paren(expr)
        if isinstance(e, P.BinaryOp) and binop_kind(e) in (
                BinaryOpKind.ADD, BinaryOpKind.SUB, BinaryOpKind.AND,
                BinaryOpKind.OR, BinaryOpKind.XOR):
            left = self._low_byte_form(e.left)
            right = self._low_byte_form(e.right)
            if left is not e.left or right is not e.right:
                return make_binary(binop_kind(e), left, right, pos=e.pos)
        elif isinstance(e, P.UnaryOp):
            operand = self._low_byte_form(e.operand)
            if operand is not e.operand:
                return make_unary(unop_kind(e), operand, pos=e.pos)
        return expr

    def _narrowed_comparison(self, expr):
        """A comparison of ``DOUBLE(x)`` with a BYTE, as one of two BYTEs.

        Both sides are then bytes zero-extended, which a byte compare orders
        exactly as the 16-bit one does.
        """
        e = unwrap_paren(expr)
        if not (isinstance(e, P.BinaryOp) and binop_kind(e) in self._COMPARISON_KINDS):
            return expr
        left = self._unwidened(e.left)
        right = self._unwidened(e.right)
        if left is None and right is None:
            return expr
        left = e.left if left is None else left
        right = e.right if right is None else right
        if (self._get_expr_type(left) != DataType.BYTE
                or self._get_expr_type(right) != DataType.BYTE):
            return expr
        return make_binary(binop_kind(e), left, right, pos=e.pos)

    def _without_redundant_double(self, expr):
        """``expr`` with a constant operand ``DOUBLE(n)`` written ``n`` where
        the operation is 16-bit anyway.

        The optimizer writes an ADDRESS constant below 256 as ``DOUBLE(n)``
        so that it widens what it meets. Against an ADDRESS, or in a
        product, quotient or remainder, the operation is 16-bit either way,
        and the plain literal is what the short forms (`inc hl', `ld de,n')
        look for.
        """
        e = unwrap_paren(expr)
        if not isinstance(e, P.BinaryOp):
            return expr
        op = binop_kind(e)
        always_wide = op in (BinaryOpKind.MUL, BinaryOpKind.DIV, BinaryOpKind.MOD)

        def strip(x, other):
            if (is_double_call(x) and typed_const(x) is not None
                    and (always_wide or self._get_expr_type(other) == DataType.ADDRESS)):
                return unwrap_paren(unwrap_paren(x).args[0])
            return x

        left = strip(e.left, e.right)
        right = strip(e.right, left)
        if left is e.left and right is e.right:
            return expr
        return make_binary(op, left, right, pos=e.pos)

    def _gen_expr_to_a(self, expr) -> None:
        """Generate code to load an expression into A (for byte operations)."""
        expr = unwrap_paren(self._low_byte_form(expr))
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
        """Generate code for a typed unary expression.

        `-x' and `NOT x' have x's type (4.2.2, 4.3): NOT 7 is the BYTE
        0F8H and -1 the BYTE 0FFH. The width is the operand's static type,
        not the register it happens to be generated into -- a literal
        loads into HL, and complementing all of HL made `(NOT 7) MOD w'
        divide 0FFF8H.
        """
        kind = unop_kind(expr)
        if self._get_expr_type(expr.operand) == DataType.BYTE:
            self._gen_expr_to_a(expr.operand)
            operand_type = DataType.BYTE
        else:
            self._gen_expr_to_hl(expr.operand)
            operand_type = DataType.ADDRESS

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

        # An index is zero-extended anyway.
        index = unwrap_paren(self._unwidened(index) or index)
        # LENGTH and LAST are constants like any other (a one-character
        # string too): `ab(LAST(sa))' is `ab(7)', an address the assembler
        # works out.
        if not isinstance(index, P.NumberLiteral):
            const_index = self._get_const_byte_value(index)
            if const_index is not None:
                index = make_number_literal(const_index, pos=getattr(index, "pos", None))

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
                # By what _gen_expr returns, not by idx_type: an embedded
                # assignment of a constant comes back in HL.
                if self._gen_expr(index) == DataType.BYTE:
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
                self._emit("ld", f"hl,({self._based_ptr_operand(sym)})")
            else:
                asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(ident_text(base.name))
                self._emit("ld", f"hl,{asm_name}")
        else:
            self._gen_expr(base)

        if isinstance(index, P.NumberLiteral):
            offset = number_value(index) * elem_size
            self._emit_add_hl_const(offset)
        else:
            # The base address must survive generation of the index, and DE is
            # not safe for that: the index expression can use DE itself - a
            # 16-bit add against a constant emits "ld de,nn" - which silently
            # overwrote the base and made every element address wrong.  PRNT(I +
            # LZH$T) = I wrote to (I + 629) * 2 + 629 instead of PRNT + (I +
            # 629) * 2.  Keep the base on the stack instead.
            self.regs.need_reg('de', 'subscript_base', self._emit)
            self._emit("push", "hl")

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
                    self._emit("ld", f"de,{elem_size}")
                    self._emit("call", "??mul16")
                    self.needs_runtime.add("mul16")

            self._emit("pop", "de")
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
                    self._emit("ld", f"hl,({self._based_ptr_operand(sym)})")
                else:
                    asm_name = sym.asm_name or name
                    self._emit("ld", f"hl,{asm_name}")
            elif sym and sym.based_on:
                base_sym = self.symbols.lookup(sym.based_on)
                base_asm_name = base_sym.asm_name if base_sym and base_sym.asm_name else self._mangle_name(sym.based_on)
                self._emit("ld", f"hl,({self._based_ptr_operand(sym)})")
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
                # Same reason as in _gen_subscript_addr: the index may use DE.
                self.regs.need_reg('de', 'member_subscript_base', self._emit)
                self._emit("push", "hl")
                idx_type = self._gen_expr(idx_expr)
                if idx_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                if elem_size == 2:
                    self._emit("add", "hl,hl")
                self._emit("pop", "de")
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
        name = None
        if isinstance(callee, P.Identifier):
            name = ident_text(callee.name)
            sym = self._lookup_symbol(name)
            call_name = sym.asm_name if sym and sym.asm_name else name

            # CP/M BDOS optimisation: MON1/MON2(func, arg).
            if name.upper() in ('MON1', 'MON2') and len(args) == 2:
                func_arg, addr_arg = args
                func_num = self._get_const_byte_value(func_arg)
                if func_num is not None and func_num <= 255:
                    # Function number loaded last; see the statement-call
                    # site for why C cannot be parked across the argument.
                    addr_type = self._gen_expr(addr_arg)
                    if addr_type == DataType.BYTE:
                        self._emit("ld", "e,a")
                    else:
                        self._emit("ex", "de,hl")
                    self._emit("ld", f"c,{self._format_number(func_num)}")
                    self._emit("call", self._pz(0x0005))
                    return DataType.BYTE if name.upper() == 'MON2' else DataType.ADDRESS

        use_stack = True
        if (sym and sym.kind == SymbolKind.PROCEDURE and not sym.is_reentrant
                and not sym.is_external and not sym.is_public):
            use_stack = False

        if use_stack:
            for arg in args:
                arg_type = self._gen_expr(arg)
                if arg_type == DataType.BYTE:
                    self._emit("ld", "l,a")
                    self._emit("ld", "h,0")
                self._emit("push", "hl")
        else:
            self._gen_slot_args(sym, args, name)

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
                    # The port is a BYTE: `INPUT(-1)' reads port 0FFH.
                    self._emit("in", f"a,({self._format_number(port_num & 0xFF)})")
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
            self._gen_expr_to_hl(args[0])
            self._emit("push", "hl")
            self._gen_count_to_b(args[1])
            self._emit("pop", "hl")   # Value in HL
            self._emit_counted_loop("SHL", [("add", "hl,hl")])
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
                    # Shift by 7 is a shift left by 1 and a byte move. The
                    # result has nine bits: bit 15 comes down to bit 8, which
                    # `(H << 1) | (L >> 7)' alone dropped, so 8000H / 128
                    # (strength-reduced to this) came out 0 instead of 100H.
                    self._emit("add", "hl,hl")  # carry = bit 15
                    self._emit("ld", "l,h")     # L = bits 14..7
                    self._emit("ld", "h,0")
                    self._emit("rl", "h")       # H = bit 15
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

        if name in ("ROL", "ROR", "SCL", "SCR"):
            return self._gen_rotate(name, args)

        if name in ("LENGTH", "LAST"):
            extent = self._array_extent(args[0]) if args else None
            if not extent:
                raise CodeGenError(
                    f"{name}() needs an array whose extent is known")
            value = extent if name == "LENGTH" else extent - 1
            # A BYTE when it fits (11.1.2), and generated as one: code that
            # goes by the static type -- a BYTE subscript, above all -- takes
            # the value from A, and `ld hl,7' left A as it was, so
            # `ab(LAST(sa))' read whatever element A happened to name.
            if value <= 0xFF:
                self._emit("ld", f"a,{self._format_number(value)}")
                return DataType.BYTE
            self._emit("ld", f"hl,{self._format_number(value)}")
            return DataType.ADDRESS

        if name == "SIZE":
            if args:
                arg0 = unwrap_paren(args[0])
                if isinstance(arg0, P.Identifier):
                    sym = self._lookup_scoped(ident_text(arg0.name))
                    if sym:
                        self._emit("ld", f"hl,{sym.size}")
                        return DataType.ADDRESS
            raise CodeGenError(
                "SIZE() needs a declared variable")

        if name == "MEMORY":
            self.needs_end_symbol = True
            arg0 = unwrap_paren(args[0])
            if isinstance(arg0, P.NumberLiteral) and number_value(arg0) == 0:
                self._emit("ld", "hl,__END__")
            else:
                # A BYTE subscript is generated into A; `add hl,de' took
                # whatever HL held instead.
                self._gen_expr_to_hl(args[0])
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
                # Each address is converted to an ADDRESS: a BYTE one is
                # generated into A, and has to be widened into HL.
                if source_preserves_de:
                    # Source is simple - can load dest to DE first
                    self._gen_expr_to_hl(args[2])  # dest -> HL
                    self._emit("ex", "de,hl")  # dest -> DE
                    self._gen_expr_to_hl(args[1])  # source -> HL (preserves DE)
                else:
                    # Source is complex and may clobber DE - must save dest
                    self._gen_expr_to_hl(args[2])  # dest -> HL
                    self._emit("push", "hl")  # save dest
                    self._gen_expr_to_hl(args[1])  # source -> HL (may clobber DE)
                    self._emit("pop", "de")  # dest -> DE
                self._emit("ld", f"bc,{self._format_number(count_const)}")
                self._emit("ldir")
            else:
                # Variable count - need to evaluate and check for zero
                # count -> BC, source -> HL, dest -> DE
                # _gen_expr_to_hl, not _gen_expr: a BYTE count lands in A,
                # and `ld b,h / ld c,l` would then take BC from the source
                # address that was still sitting in HL.
                self._gen_expr_to_hl(args[2])  # dest -> HL
                self._emit("push", "hl")
                self._gen_expr_to_hl(args[1])  # source -> HL
                self._emit("push", "hl")
                self._gen_expr_to_hl(args[0])  # count -> HL
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
            # Delay loop; a BYTE count is generated into A.
            self._gen_expr_to_hl(args[0])
            loop_label = self._new_label("TIME")
            self._emit_label(loop_label)
            self._emit("dec", "hl")
            self._emit("ld", "a,h")
            self._emit("or", "l")
            self._emit("jp", f"nz,{loop_label}")
            return None

        if name == "CARRY":
            # Return carry flag value; see the other CARRY site for why
            # `ld a,0 / rla` cannot be used.
            self._emit("sbc", "a,a")
            self._emit("ld", "l,a")
            self._emit("ld", "h,0")
            return DataType.BYTE

        if name == "ZERO":
            # Return zero flag value
            end_label = self._new_label("ZFE")

            # `ld a,0ffh` sets no flags, so the condition survives to the

            # branch; the false path turns it into 0 with `inc a`, which

            # leaves CARRY alone. Loading zero directly would be rewritten

            # by the peephole into `xor a`, and that clears carry for any

            # later CARRY read.

            self._emit("ld", "a,0ffh")

            self._emit("jp", f"z,{end_label}")

            self._emit("inc", "a")

            self._emit_label(end_label)
            return DataType.BYTE

        if name == "SIGN":
            # Return sign flag value
            end_label = self._new_label("SFE")

            # `ld a,0ffh` sets no flags, so the condition survives to the

            # branch; the false path turns it into 0 with `inc a`, which

            # leaves CARRY alone. Loading zero directly would be rewritten

            # by the peephole into `xor a`, and that clears carry for any

            # later CARRY read.

            self._emit("ld", "a,0ffh")

            self._emit("jp", f"m,{end_label}")

            self._emit("inc", "a")

            self._emit_label(end_label)
            return DataType.BYTE

        if name == "PARITY":
            # Return parity flag value
            end_label = self._new_label("PFE")

            # `ld a,0ffh` sets no flags, so the condition survives to the

            # branch; the false path turns it into 0 with `inc a`, which

            # leaves CARRY alone. Loading zero directly would be rewritten

            # by the peephole into `xor a`, and that clears carry for any

            # later CARRY read.

            self._emit("ld", "a,0ffh")

            self._emit("jp", f"pe,{end_label}")

            self._emit("inc", "a")

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

        # Not a built-in we handle inline
        return None

    def _gen_rotate(self, name: str, args) -> DataType:
        """ROL / ROR, and SCL / SCR through the carry.

        ROL and ROR rotate their pattern converted to a BYTE (11.1.4); SCL
        and SCR rotate a BYTE pattern nine bits round the carry and an
        ADDRESS one seventeen, and return the pattern's type (12.3). The
        count is converted to a BYTE; a constant one is unrolled. INC, DEC
        and DJNZ leave the carry alone.
        """
        wide = (name in ("SCL", "SCR")
                and self._get_expr_type(args[0]) == DataType.ADDRESS)
        ops = {
            "ROL": [("rlca", "")], "ROR": [("rrca", "")],
            "SCL": [("rl", "l"), ("rl", "h")] if wide else [("rla", "")],
            "SCR": [("rr", "h"), ("rr", "l")] if wide else [("rra", "")],
        }[name]
        count = self._try_eval_const(args[1])
        if wide:
            self._gen_expr_to_hl(args[0])
        else:
            self._gen_expr_to_a(args[0])
        if count is not None:
            count &= 0xFF
            if name in ("ROL", "ROR"):
                count &= 7          # eight rotations give the byte back
            if count <= 8:
                for _ in range(count):
                    for opcode, operands in ops:
                        if operands:
                            self._emit(opcode, operands)
                        else:
                            self._emit(opcode)
            else:
                self._emit("ld", f"b,{self._format_number(count)}")
                self._emit_counted_loop(name, ops)
        else:
            self._emit("push", "hl" if wide else "af")
            self._gen_count_to_b(args[1])
            self._emit("pop", "hl" if wide else "af")
            self._emit_counted_loop(name, ops)
        if wide:
            return DataType.ADDRESS
        self._emit("ld", "l,a")
        self._emit("ld", "h,0")
        return DataType.BYTE

    def _gen_count_to_b(self, count) -> None:
        """Load a shift or rotate count, converted to a BYTE, into B."""
        self._gen_expr_to_a(count)
        self._emit("ld", "b,a")

    def _emit_counted_loop(self, prefix: str, ops) -> None:
        """Repeat ``ops`` B times, not at all when B is 0.

        The count is unsigned: the `dec c / jp m' loops this replaces
        stopped at once for a count of 129 or more.
        """
        loop_label = self._new_label(prefix)
        end_label = self._new_label(prefix + "E")
        self._emit("inc", "b")
        self._emit("dec", "b")
        self._emit("jp", f"z,{end_label}")
        self._emit_label(loop_label)
        for opcode, operands in ops:
            if operands:
                self._emit(opcode, operands)
            else:
                self._emit(opcode)
        self._emit("djnz", loop_label)
        self._emit_label(end_label)

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
                self._emit("ld", f"hl,({self._based_ptr_operand(sym)})")
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
                    # Same reason as in _gen_subscript_addr: the index may use DE.
                    self.regs.need_reg('de', 'member_subscript_addr', self._emit)
                    self._emit("push", "hl")
                    idx_type = self._gen_expr(idx_expr)
                    if idx_type == DataType.BYTE:
                        self._emit("ld", "l,a")
                        self._emit("ld", "h,0")
                    if elem_size == 2:
                        self._emit("add", "hl,hl")
                    self._emit("pop", "de")
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
