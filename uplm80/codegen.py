"""
Code Generator for PL/M-80.

Generates 8080 or Z80 assembly code from the optimized AST.
Outputs MACRO-80 compatible .MAC files.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TextIO
from io import StringIO

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
from .errors import CodeGenError, SourceLocation
from .runtime import get_runtime_library


class Target(Enum):
    """Target processor."""

    I8080 = auto()
    Z80 = auto()


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

    def __init__(self, target: Target = Target.I8080) -> None:
        self.target = target
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
        self.literal_macros: dict[str, str] = {}  # LITERALLY macro expansions
        self.block_scope_counter = 0  # Counter for unique DO block scopes
        self.emit_data_inline = False  # If True, DATA goes to code segment
        # Call graph for parameter sharing optimization
        self.call_graph: dict[str, set[str]] = {}  # proc -> set of procs it calls
        self.can_be_active_together: dict[str, set[str]] = {}  # proc -> procs that can be on stack with it
        self.param_slots: dict[str, int] = {}  # param_key -> slot number
        self.slot_storage: list[tuple[str, int]] = []  # (label, size) for each slot
        self.proc_params: dict[str, list[tuple[str, str, DataType, int]]] = {}  # proc -> [(name, asm_name, type, size)]

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
        # OR if they share a common caller (both reachable from same proc)
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

    def _emit_label(self, label: str) -> None:
        """Emit a label."""
        self.output.append(AsmLine(label=label))

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

        # Register in symbol table at the GLOBAL level so it's always accessible
        # This allows forward references from anywhere in the module
        # Use full_proc_name as the symbol name to avoid collisions between
        # nested procedures with the same local name (e.g., multiple ZN procs)
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
        self.literal_macros = {}

        # Header
        self._emit(comment=f"PL/M-80 Compiler Output - {module.name}")
        self._emit(comment=f"Target: {'8080' if self.target == Target.I8080 else 'Z80'}")
        self._emit(comment="Generated by uplm80")
        self._emit()

        # Origin if specified
        if module.origin is not None:
            self._emit("ORG", self._format_number(module.origin))
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
            self._emit("JMP", entry_proc.name)

        # Generate code for module-level statements
        if module.stmts:
            self._emit()
            self._emit(comment="Module initialization code")
            for stmt in module.stmts:
                self._gen_stmt(stmt)

        # Generate procedures
        for proc in procedures:
            self._gen_declaration(proc)

        # Emit runtime library if needed
        if self.needs_runtime:
            self._emit()
            self._emit(comment="Runtime library")
            runtime = get_runtime_library()
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

        # Emit string literals
        if self.string_literals:
            self._emit()
            self._emit(comment="String literals")
            for label, value in self.string_literals:
                self._emit_label(label)
                escaped = self._escape_string(value)
                self._emit("DB", escaped)

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
            self._emit("DS", str(self.total_auto_storage))

        # End directive
        self._emit()
        self._emit("END")

        # Convert to string
        return "\n".join(str(line) for line in self.output)

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
                self._emit("EXTRN", decl.name)

    def _gen_var_decl(self, decl: VarDecl) -> None:
        """Generate storage for a variable declaration."""
        # Mangle name if it conflicts with register names
        base_name = self._mangle_name(decl.name)

        # Check if this is a procedure local that can use shared storage
        use_shared = False
        if (self.current_proc and not decl.is_public and not decl.is_external
            and not decl.based_on and not decl.at_location and not decl.data_values
            and not decl.initial_values):
            # Check if we have shared storage for this proc and var
            if (hasattr(self, 'storage_labels')
                and self.current_proc in self.storage_labels
                and decl.name in self.storage_labels[self.current_proc]):
                asm_name = self.storage_labels[self.current_proc][decl.name]
                use_shared = True

        if not use_shared:
            # For non-public local variables in procedures, prefix with scope name to avoid conflicts
            if self.current_proc and not decl.is_public and not decl.is_external:
                asm_name = f"@{self.current_proc}${base_name}"
            else:
                asm_name = base_name

        # Calculate size
        if decl.struct_members:
            size = sum(
                (m.dimension or 1) * (1 if m.data_type == DataType.BYTE else 2)
                for m in decl.struct_members
            )
            elem_size = 2  # Structures are ADDRESS-sized elements
        else:
            elem_size = 1 if decl.data_type == DataType.BYTE else 2
            count = decl.dimension or 1
            size = elem_size * count

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
            asm_name=asm_name,  # Store mangled name
        )
        self.symbols.define(sym)

        # External variables don't get storage here
        if decl.is_external:
            self._emit("EXTRN", asm_name)
            return

        # Public declaration
        if decl.is_public:
            self._emit("PUBLIC", asm_name)

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
                    # Check for built-in MEMORY - address is 0
                    if loc_operand.name.upper() == "MEMORY":
                        self.data_segment.append(
                            AsmLine(label=asm_name, opcode="EQU", operands="0")
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
                            self.data_segment.append(
                                AsmLine(label=asm_name, opcode="EQU", operands=ref_name)
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
        else:
            # Uninitialized storage
            self.data_segment.append(
                AsmLine(label=asm_name, opcode="DS", operands=str(size))
            )

    def _emit_data_values(self, values: list[Expr], dtype: DataType, inline: bool = False) -> None:
        """Emit DATA values to data segment or inline code segment."""
        target = self.code_data_segment if inline else self.data_segment
        for val in values:
            if isinstance(val, NumberLiteral):
                directive = "DB" if dtype == DataType.BYTE else "DW"
                target.append(
                    AsmLine(opcode=directive, operands=self._format_number(val.value))
                )
            elif isinstance(val, StringLiteral):
                target.append(
                    AsmLine(opcode="DB", operands=self._escape_string(val.value))
                )
            elif isinstance(val, Identifier):
                # Could be a LITERALLY macro - expand it
                name = val.name
                if name in self.literal_macros:
                    # Try to parse the macro value as a number
                    try:
                        num_val = self._parse_plm_number(self.literal_macros[name])
                        directive = "DB" if dtype == DataType.BYTE else "DW"
                        target.append(
                            AsmLine(opcode=directive, operands=self._format_number(num_val))
                        )
                    except ValueError:
                        # Not a number, use as-is
                        target.append(
                            AsmLine(opcode="DB", operands=self.literal_macros[name])
                        )
                else:
                    # Unknown identifier - use as label reference
                    target.append(
                        AsmLine(opcode="DW", operands=name)
                    )
            elif isinstance(val, LocationExpr):
                # Address-of expression: .variable or .procedure
                operand = val.operand
                if isinstance(operand, Identifier):
                    # .name means address of name
                    target.append(
                        AsmLine(opcode="DW", operands=operand.name)
                    )
                else:
                    raise CodeGenError(f"Unsupported operand in DATA location expression: {operand}")
            elif isinstance(val, BinaryExpr):
                # Binary expression like .name-3 or name+offset
                # Generate assembly expression string
                expr_str = self._data_expr_to_string(val)
                target.append(
                    AsmLine(opcode="DW", operands=expr_str)
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
            if expr.name in self.literal_macros:
                return self.literal_macros[expr.name]
            return expr.name
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
                directive = "DB" if dtype == DataType.BYTE else "DW"
                self.data_segment.append(
                    AsmLine(opcode=directive, operands=self._format_number(val.value))
                )
            elif isinstance(val, StringLiteral):
                self.data_segment.append(
                    AsmLine(opcode="DB", operands=self._escape_string(val.value))
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
            self._emit("EXTRN", proc_asm_name)
            self.current_proc = old_proc
            self.current_proc_decl = old_proc_decl
            return

        self._emit()
        if decl.is_public:
            self._emit("PUBLIC", decl.name)

        self._emit(comment=f"Procedure {decl.name}")
        self._emit_label(proc_asm_name)

        # Enter new scope
        self.symbols.enter_scope(decl.name)

        # Procedure prologue
        if decl.interrupt_num is not None:
            # Interrupt handler - save all registers
            self._emit("PUSH", "PSW")
            self._emit("PUSH", "B")
            self._emit("PUSH", "D")
            self._emit("PUSH", "H")

        # Define parameters as local variables
        # For non-reentrant: use shared automatic storage via storage_labels
        # For reentrant: use individual storage (stack-based)
        param_infos: list[tuple[str, str, DataType, int]] = []  # (name, asm_name, type, size)
        use_shared_storage = not decl.is_reentrant and full_proc_name in self.storage_labels

        for i, param in enumerate(decl.params):
            # Find parameter declaration in decl.decls
            param_type = DataType.ADDRESS  # Default
            for d in decl.decls:
                if isinstance(d, VarDecl) and d.name == param:
                    param_type = d.data_type or DataType.ADDRESS
                    break

            param_size = 1 if param_type == DataType.BYTE else 2

            # Get asm_name from shared storage or create individual
            if use_shared_storage and param in self.storage_labels.get(full_proc_name, {}):
                asm_name = self.storage_labels[full_proc_name][param]
            else:
                # Fallback: individual storage (for reentrant or if not in storage_labels)
                asm_name = f"@{decl.name}${self._mangle_name(param)}"
                # Allocate individual storage in data segment
                self.data_segment.append(
                    AsmLine(label=asm_name, opcode="DS", operands=str(param_size))
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

        # Generate prologue code to copy parameters from stack to local storage
        # Only needed for reentrant procedures - non-reentrant have params stored directly by caller
        if param_infos and decl.is_reentrant:
            # Get stack pointer into HL
            # Stack: [ret addr (2)] [params...]
            # We need to access params without destroying return address
            self._emit("POP", "H")  # HL = return address
            for param_name, asm_name, param_type, param_size in param_infos:
                if param_size == 1:
                    self._emit("POP", "D")  # Get param value (only low byte valid)
                    self._emit("MOV", "A,E")
                    self._emit("STA", asm_name)
                else:
                    self._emit("POP", "D")  # Get param value
                    self._emit("XCHG")  # HL <-> DE
                    self._emit("SHLD", asm_name)
                    self._emit("XCHG")  # Restore return address to HL
            self._emit("PUSH", "H")  # Push return address back
            # Note: This changes the stack - caller must not expect params on stack after return

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

        # Generate code for statements
        ends_with_return = False
        for stmt in statements_to_gen:
            self._gen_stmt(stmt)
            ends_with_return = isinstance(stmt, ReturnStmt)

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
            self._emit("POP", "H")
            self._emit("POP", "D")
            self._emit("POP", "B")
            self._emit("POP", "PSW")
            self._emit("EI")
            self._emit("RET")
        else:
            self._emit("RET")

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
            self._emit("JMP", target)
        elif isinstance(stmt, HaltStmt):
            self._emit("HLT")
        elif isinstance(stmt, EnableStmt):
            self._emit("EI")
        elif isinstance(stmt, DisableStmt):
            self._emit("DI")
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
        # Evaluate the value expression (result in A for BYTE, HL for ADDRESS)
        value_type = self._gen_expr(stmt.value)

        # Store to each target (multiple assignment support)
        for i, target in enumerate(stmt.targets):
            if i < len(stmt.targets) - 1:
                # Need to preserve value for next target
                if value_type == DataType.BYTE:
                    self._emit("PUSH", "PSW")
                else:
                    self._emit("PUSH", "H")

            self._gen_store(target, value_type)

            if i < len(stmt.targets) - 1:
                if value_type == DataType.BYTE:
                    self._emit("POP", "PSW")
                else:
                    self._emit("POP", "H")

    def _gen_call_stmt(self, stmt: CallStmt) -> None:
        """Generate code for a CALL statement."""
        # Check for built-in procedures first
        if isinstance(stmt.callee, Identifier):
            name = stmt.callee.name.upper()
            # Handle built-in procedures that don't return values
            if name in self.BUILTIN_FUNCS:
                result = self._gen_builtin(name, stmt.args)
                if result is not None or name in ('TIME', 'MOVE'):
                    # Built-in was handled
                    return

        # Look up procedure symbol to check if reentrant
        sym = None
        call_name = None
        if isinstance(stmt.callee, Identifier):
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
            if name.upper() in ('MON1', 'MON2') and len(stmt.args) == 2:
                func_arg, addr_arg = stmt.args
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
                    # Generate direct BDOS call: MVI C,func; LXI D,addr; CALL 5
                    self._emit("MVI", f"C,{self._format_number(func_num)}")
                    addr_type = self._gen_expr(addr_arg)
                    if addr_type == DataType.BYTE:
                        self._emit("MOV", "E,A")
                        self._emit("MVI", "D,0")
                    else:
                        self._emit("XCHG")  # DE = addr
                    self._emit("CALL", "5")  # BDOS entry point
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
                    self._emit("MOV", "L,A")
                    self._emit("MVI", "H,0")
                self._emit("PUSH", "H")
        else:
            # Direct memory parameter passing (non-reentrant)
            # Store each argument to its parameter's memory location
            for i, arg in enumerate(stmt.args):
                if i < len(sym.params):
                    param_name = sym.params[i]

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

                    arg_type = self._gen_expr(arg)
                    if arg_type == DataType.BYTE:
                        # BYTE param - use STA
                        self._emit("STA", param_asm)
                    else:
                        # ADDRESS param - use SHLD
                        self._emit("SHLD", param_asm)

        # Call the procedure
        if isinstance(stmt.callee, Identifier):
            self._emit("CALL", call_name)
        else:
            # Indirect call through address
            self._gen_expr(stmt.callee)
            self._emit("PCHL")

        # Clean up stack (caller cleanup) - only for stack-based calls
        if use_stack and stmt.args:
            stack_bytes = len(stmt.args) * 2
            if stack_bytes == 2:
                self._emit("POP", "D")  # Dummy pop
            elif stack_bytes == 4:
                self._emit("POP", "D")
                self._emit("POP", "D")
            elif stack_bytes <= 8:
                for _ in range(len(stmt.args)):
                    self._emit("POP", "D")
            else:
                # Adjust stack pointer directly
                self._emit("LXI", f"D,{stack_bytes}")
                self._emit("DAD", "SP")
                self._emit("SPHL")

    def _gen_return(self, stmt: ReturnStmt) -> None:
        """Generate code for RETURN statement."""
        if stmt.value:
            self._gen_expr(stmt.value)
            # Return value is in A (BYTE) or HL (ADDRESS)

        if self.current_proc_decl and self.current_proc_decl.interrupt_num is not None:
            # Interrupt handler return
            self._emit("POP", "H")
            self._emit("POP", "D")
            self._emit("POP", "B")
            self._emit("POP", "PSW")
            self._emit("EI")

        self._emit("RET")

    def _gen_if(self, stmt: IfStmt) -> None:
        """Generate code for IF statement."""
        else_label = self._new_label("ELSE")
        end_label = self._new_label("ENDIF")
        false_target = else_label if stmt.else_stmt else end_label

        # Try to generate optimized conditional jump for comparisons
        if self._gen_condition_jump_false(stmt.condition, false_target):
            # Condition jump was generated directly
            pass
        else:
            # Fallback: evaluate condition and test result
            self._gen_expr(stmt.condition)
            # Test result - for ADDRESS, check if HL is zero
            self._emit("MOV", "A,L")
            self._emit("ORA", "H")  # A = L | H
            self._emit("JZ", false_target)

        # Then branch
        self._gen_stmt(stmt.then_stmt)

        if stmt.else_stmt:
            self._emit("JMP", end_label)
            self._emit_label(else_label)
            self._gen_stmt(stmt.else_stmt)

        self._emit_label(end_label)

    def _gen_condition_jump_false(self, condition: Expr, false_label: str) -> bool:
        """Generate conditional jump to false_label if condition is false.

        Returns True if optimized jump was generated, False if caller should use fallback.
        """
        if not isinstance(condition, BinaryExpr):
            return False

        op = condition.op

        # Handle short-circuit AND: (a AND b) is false if a is false OR b is false
        if op == BinaryOp.AND:
            # If left is false, whole AND is false -> jump to false_label
            if not self._gen_condition_jump_false(condition.left, false_label):
                # Fallback: evaluate left, test for zero
                self._gen_expr(condition.left)
                self._emit("MOV", "A,L")
                self._emit("ORA", "H")
                self._emit("JZ", false_label)
            # If right is false, whole AND is false -> jump to false_label
            if not self._gen_condition_jump_false(condition.right, false_label):
                self._gen_expr(condition.right)
                self._emit("MOV", "A,L")
                self._emit("ORA", "H")
                self._emit("JZ", false_label)
            return True

        # Handle short-circuit OR: (a OR b) is false only if BOTH a and b are false
        if op == BinaryOp.OR:
            true_label = self._new_label("ORTRUE")
            # If left is true, whole OR is true -> skip to after false check
            if not self._gen_condition_jump_true(condition.left, true_label):
                # Fallback: evaluate left, test for non-zero
                self._gen_expr(condition.left)
                self._emit("MOV", "A,L")
                self._emit("ORA", "H")
                self._emit("JNZ", true_label)
            # If right is false, whole OR is false -> jump to false_label
            if not self._gen_condition_jump_false(condition.right, false_label):
                self._gen_expr(condition.right)
                self._emit("MOV", "A,L")
                self._emit("ORA", "H")
                self._emit("JZ", false_label)
            self._emit_label(true_label)
            return True

        if op not in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.GT, BinaryOp.LE, BinaryOp.GE):
            return False

        # Check if both operands are bytes for optimized comparison
        left_type = self._get_expr_type(condition.left)
        right_type = self._get_expr_type(condition.right)
        both_bytes = (left_type == DataType.BYTE and right_type == DataType.BYTE)

        if both_bytes:
            # Byte comparison with constant using CPI
            if isinstance(condition.right, NumberLiteral) and condition.right.value <= 255:
                self._gen_expr(condition.left)  # Result in A
                self._emit("CPI", self._format_number(condition.right.value))
                self._emit_jump_on_false(op, false_label)
                return True
            else:
                # Byte-to-byte comparison
                self._gen_expr(condition.left)  # Result in A
                self._emit("MOV", "B,A")  # Save left
                self._gen_expr(condition.right)  # Result in A (right)
                self._emit("MOV", "C,A")  # Save right
                self._emit("MOV", "A,B")  # A = left
                self._emit("SUB", "C")    # A = left - right, flags set
                self._emit_jump_on_false(op, false_label)
                return True
        else:
            # 16-bit comparison - still optimize by not materializing boolean
            # Evaluate left, push, evaluate right, subtract
            self._gen_expr(condition.left)
            if left_type == DataType.BYTE:
                self._emit("MOV", "L,A")
                self._emit("MVI", "H,0")
            self._emit("PUSH", "H")

            self._gen_expr(condition.right)
            if right_type == DataType.BYTE:
                self._emit("MOV", "L,A")
                self._emit("MVI", "H,0")

            self._emit("XCHG")  # DE = right
            self._emit("POP", "H")  # HL = left

            # 16-bit subtract: HL - DE
            self._emit("MOV", "A,L")
            self._emit("SUB", "E")
            self._emit("MOV", "L,A")
            self._emit("MOV", "A,H")
            self._emit("SBB", "D")
            self._emit("MOV", "H,A")

            # For EQ/NE, check if result is zero
            if op in (BinaryOp.EQ, BinaryOp.NE):
                self._emit("MOV", "A,L")
                self._emit("ORA", "H")
                if op == BinaryOp.EQ:
                    self._emit("JNZ", false_label)  # If not zero, condition is false
                else:
                    self._emit("JZ", false_label)   # If zero, condition is false
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
        if not isinstance(condition, BinaryExpr):
            return False

        op = condition.op

        # Handle short-circuit OR: (a OR b) is true if a is true OR b is true
        if op == BinaryOp.OR:
            # If left is true, whole OR is true -> jump to true_label
            if not self._gen_condition_jump_true(condition.left, true_label):
                self._gen_expr(condition.left)
                self._emit("MOV", "A,L")
                self._emit("ORA", "H")
                self._emit("JNZ", true_label)
            # If right is true, whole OR is true -> jump to true_label
            if not self._gen_condition_jump_true(condition.right, true_label):
                self._gen_expr(condition.right)
                self._emit("MOV", "A,L")
                self._emit("ORA", "H")
                self._emit("JNZ", true_label)
            return True

        # Handle short-circuit AND: (a AND b) is true only if BOTH are true
        if op == BinaryOp.AND:
            false_label = self._new_label("ANDFALSE")
            # If left is false, skip right evaluation
            if not self._gen_condition_jump_false(condition.left, false_label):
                self._gen_expr(condition.left)
                self._emit("MOV", "A,L")
                self._emit("ORA", "H")
                self._emit("JZ", false_label)
            # If right is true, AND is true
            if not self._gen_condition_jump_true(condition.right, true_label):
                self._gen_expr(condition.right)
                self._emit("MOV", "A,L")
                self._emit("ORA", "H")
                self._emit("JNZ", true_label)
            self._emit_label(false_label)
            return True

        if op not in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.GT, BinaryOp.LE, BinaryOp.GE):
            return False

        # Check if both operands are bytes for optimized comparison
        left_type = self._get_expr_type(condition.left)
        right_type = self._get_expr_type(condition.right)
        both_bytes = (left_type == DataType.BYTE and right_type == DataType.BYTE)

        if both_bytes:
            if isinstance(condition.right, NumberLiteral) and condition.right.value <= 255:
                self._gen_expr(condition.left)
                self._emit("CPI", self._format_number(condition.right.value))
                self._emit_jump_on_true(op, true_label)
                return True
            else:
                self._gen_expr(condition.left)
                self._emit("MOV", "B,A")
                self._gen_expr(condition.right)
                self._emit("MOV", "C,A")
                self._emit("MOV", "A,B")
                self._emit("SUB", "C")
                self._emit_jump_on_true(op, true_label)
                return True
        else:
            # 16-bit comparison
            self._gen_expr(condition.left)
            if left_type == DataType.BYTE:
                self._emit("MOV", "L,A")
                self._emit("MVI", "H,0")
            self._emit("PUSH", "H")

            self._gen_expr(condition.right)
            if right_type == DataType.BYTE:
                self._emit("MOV", "L,A")
                self._emit("MVI", "H,0")

            self._emit("XCHG")
            self._emit("POP", "H")

            self._emit("MOV", "A,L")
            self._emit("SUB", "E")
            self._emit("MOV", "L,A")
            self._emit("MOV", "A,H")
            self._emit("SBB", "D")
            self._emit("MOV", "H,A")

            if op in (BinaryOp.EQ, BinaryOp.NE):
                self._emit("MOV", "A,L")
                self._emit("ORA", "H")
                if op == BinaryOp.EQ:
                    self._emit("JZ", true_label)
                else:
                    self._emit("JNZ", true_label)
                return True
            else:
                self._emit_jump_on_true_16bit(op, true_label)
                return True

        return False

    def _emit_jump_on_true(self, op: BinaryOp, true_label: str) -> None:
        """Emit jump to true_label if comparison result is true (8-bit compare)."""
        if op == BinaryOp.EQ:
            self._emit("JZ", true_label)
        elif op == BinaryOp.NE:
            self._emit("JNZ", true_label)
        elif op == BinaryOp.LT:
            self._emit("JC", true_label)
        elif op == BinaryOp.GE:
            self._emit("JNC", true_label)
        elif op == BinaryOp.GT:
            skip = self._new_label("SKIP")
            self._emit("JC", skip)
            self._emit("JZ", skip)
            self._emit("JMP", true_label)
            self._emit_label(skip)
        elif op == BinaryOp.LE:
            self._emit("JC", true_label)
            self._emit("JZ", true_label)

    def _emit_jump_on_true_16bit(self, op: BinaryOp, true_label: str) -> None:
        """Emit jump to true_label for 16-bit comparison."""
        if op == BinaryOp.LT:
            self._emit("MOV", "A,H")
            self._emit("ORA", "A")
            self._emit("JM", true_label)
        elif op == BinaryOp.GE:
            self._emit("MOV", "A,H")
            self._emit("ORA", "A")
            self._emit("JP", true_label)
        elif op == BinaryOp.GT:
            skip = self._new_label("SKIP")
            self._emit("MOV", "A,H")
            self._emit("ORA", "A")
            self._emit("JM", skip)
            self._emit("MOV", "A,L")
            self._emit("ORA", "H")
            self._emit("JNZ", true_label)
            self._emit_label(skip)
        elif op == BinaryOp.LE:
            self._emit("MOV", "A,H")
            self._emit("ORA", "A")
            self._emit("JM", true_label)
            self._emit("MOV", "A,L")
            self._emit("ORA", "H")
            self._emit("JZ", true_label)

    def _emit_jump_on_false(self, op: BinaryOp, false_label: str) -> None:
        """Emit jump to false_label if comparison result is false (8-bit compare)."""
        # After CPI or SUB, flags reflect left - right
        if op == BinaryOp.EQ:
            self._emit("JNZ", false_label)  # Jump if not equal (Z=0)
        elif op == BinaryOp.NE:
            self._emit("JZ", false_label)   # Jump if equal (Z=1)
        elif op == BinaryOp.LT:
            self._emit("JNC", false_label)  # Jump if not less (C=0)
        elif op == BinaryOp.GE:
            self._emit("JC", false_label)   # Jump if less (C=1)
        elif op == BinaryOp.GT:
            # Greater: not less AND not equal -> C=0 AND Z=0
            self._emit("JC", false_label)   # Jump if less
            self._emit("JZ", false_label)   # Jump if equal
        elif op == BinaryOp.LE:
            # Less or equal: C=1 OR Z=1
            # Jump if greater (C=0 AND Z=0)
            skip = self._new_label("SKIP")
            self._emit("JC", skip)   # Less -> condition true, skip jump
            self._emit("JZ", skip)   # Equal -> condition true, skip jump
            self._emit("JMP", false_label)  # Greater -> condition false
            self._emit_label(skip)

    def _emit_jump_on_false_16bit(self, op: BinaryOp, false_label: str) -> None:
        """Emit jump to false_label for 16-bit comparison (signed)."""
        # After 16-bit subtract HL = left - right, H contains high byte
        # For unsigned comparison, check carry from subtraction
        # But we already did the subtract inline, so check sign
        if op == BinaryOp.LT:
            # left < right: result negative (high bit set)
            # Check if high byte is negative
            self._emit("MOV", "A,H")
            self._emit("ORA", "A")
            self._emit("JP", false_label)  # Jump if positive (sign=0)
        elif op == BinaryOp.GE:
            # left >= right: result non-negative
            self._emit("MOV", "A,H")
            self._emit("ORA", "A")
            self._emit("JM", false_label)  # Jump if negative (sign=1)
        elif op == BinaryOp.GT:
            # left > right: result positive and non-zero
            self._emit("MOV", "A,H")
            self._emit("ORA", "A")
            self._emit("JM", false_label)  # Jump if negative
            self._emit("MOV", "A,L")
            self._emit("ORA", "H")
            self._emit("JZ", false_label)  # Jump if zero
        elif op == BinaryOp.LE:
            # left <= right: result negative or zero
            skip = self._new_label("SKIP")
            self._emit("MOV", "A,H")
            self._emit("ORA", "A")
            self._emit("JM", skip)  # Negative -> true
            self._emit("MOV", "A,L")
            self._emit("ORA", "H")
            self._emit("JZ", skip)  # Zero -> true
            self._emit("JMP", false_label)  # Positive non-zero -> false
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
        loop_label = self._new_label("WHILE")
        end_label = self._new_label("WEND")

        self.loop_stack.append((loop_label, end_label))

        # Check for optimized byte counter loop: DO WHILE (n := n - 1) <> 255
        # NOTE: This optimization is disabled because it doesn't save code -
        # the existing _gen_condition_jump_false already handles this efficiently.
        # For the optimization to help, we'd need to keep the counter in a register
        # and avoid the STA inside the loop, which requires data flow analysis to
        # confirm the counter isn't used in the loop body.
        counter_info = None  # self._is_byte_counter_loop(stmt.condition)
        if counter_info:
            var_asm, _ = counter_info
            # Optimized loop: keep counter in C register (C is less commonly used than B)
            # Load counter into C at start
            self._emit("LDA", var_asm)
            self._emit("MOV", "C,A")

            self._emit_label(loop_label)
            # Decrement C and check for 0xFF (wrap from 0 to 255)
            self._emit("DCR", "C")
            self._emit("MOV", "A,C")
            self._emit("CPI", "0FFH")
            self._emit("JZ", end_label)

            # Mark that C is being used as loop counter
            old_loop_reg = getattr(self, 'loop_counter_reg', None)
            self.loop_counter_reg = 'C'

            # Loop body
            for s in stmt.stmts:
                self._gen_stmt(s)

            # Restore loop register tracking
            self.loop_counter_reg = old_loop_reg

            self._emit("JMP", loop_label)
            self._emit_label(end_label)

            # Store C back to memory (in case it's used after loop)
            self._emit("MOV", "A,C")
            self._emit("STA", var_asm)
        else:
            self._emit_label(loop_label)

            # Try optimized condition jump, fallback to generic
            if not self._gen_condition_jump_false(stmt.condition, end_label):
                self._gen_expr(stmt.condition)
                self._emit("MOV", "A,L")
                self._emit("ORA", "H")
                self._emit("JZ", end_label)

            # Loop body
            for s in stmt.stmts:
                self._gen_stmt(s)

            self._emit("JMP", loop_label)
            self._emit_label(end_label)

        self.loop_stack.pop()

    def _gen_do_iter(self, stmt: DoIterBlock) -> None:
        """Generate code for iterative DO block."""
        loop_label = self._new_label("FOR")
        test_label = self._new_label("TEST")
        incr_label = self._new_label("INCR")
        end_label = self._new_label("NEXT")

        self.loop_stack.append((incr_label, end_label))

        # Initialize index variable
        self._gen_expr(stmt.start)
        self._gen_store(stmt.index_var, DataType.ADDRESS)

        # Jump to test
        self._emit("JMP", test_label)

        # Loop body
        self._emit_label(loop_label)
        for s in stmt.stmts:
            self._gen_stmt(s)

        # Increment
        self._emit_label(incr_label)
        step_val = 1
        if stmt.step and isinstance(stmt.step, NumberLiteral):
            step_val = stmt.step.value

        self._gen_load(stmt.index_var)
        if step_val == 1:
            self._emit("INX", "H")
        elif step_val == -1 or step_val == 0xFFFF:
            self._emit("DCX", "H")
        else:
            self._emit("LXI", f"D,{self._format_number(step_val)}")
            self._emit("DAD", "D")
        self._gen_store(stmt.index_var, DataType.ADDRESS)

        # Test condition
        self._emit_label(test_label)
        self._gen_load(stmt.index_var)
        self._emit("XCHG")  # DE = index
        self._gen_expr(stmt.bound)  # HL = bound

        # Compare: if index > bound, exit (for positive step)
        # HL - DE: if negative (carry), index > bound
        self._emit("MOV", "A,L")
        self._emit("SUB", "E")
        self._emit("MOV", "A,H")
        self._emit("SBB", "D")

        # If no borrow (NC), bound >= index, continue
        self._emit("JNC", loop_label)

        self._emit_label(end_label)
        self.loop_stack.pop()

    def _gen_do_case(self, stmt: DoCaseBlock) -> None:
        """Generate code for DO CASE block."""
        end_label = self._new_label("CASEND")

        # Create labels for each case
        case_labels = [self._new_label(f"CASE{i}") for i in range(len(stmt.cases))]

        # Evaluate selector into HL
        self._gen_expr(stmt.selector)

        # Generate jump table
        # For small number of cases, use sequential comparisons
        # For larger, could use computed jump

        if len(stmt.cases) <= 8:
            # Sequential comparisons
            for i, label in enumerate(case_labels):
                self._emit("MOV", "A,L")
                self._emit("CPI", str(i))
                self._emit("JZ", label)
            self._emit("JMP", end_label)  # Default: skip all
        else:
            # Jump table approach
            table_label = self._new_label("JMPTBL")
            self._emit("DAD", "H")  # HL = HL * 2 (addresses are 2 bytes)
            self._emit("LXI", f"D,{table_label}")
            self._emit("DAD", "D")  # HL = table + index*2
            self._emit("MOV", "E,M")
            self._emit("INX", "H")
            self._emit("MOV", "D,M")
            self._emit("XCHG")
            self._emit("PCHL")

            # Jump table
            self._emit_label(table_label)
            for label in case_labels:
                self.data_segment.append(AsmLine(opcode="DW", operands=label))

        # Generate each case
        for i, (case_stmts, label) in enumerate(zip(stmt.cases, case_labels)):
            self._emit_label(label)
            for s in case_stmts:
                self._gen_stmt(s)
            self._emit("JMP", end_label)

        self._emit_label(end_label)

    # ========================================================================
    # Expression Code Generation
    # ========================================================================

    def _get_expr_type(self, expr: Expr) -> DataType:
        """Determine the type of an expression."""
        if isinstance(expr, NumberLiteral):
            return DataType.BYTE if expr.value <= 255 else DataType.ADDRESS
        elif isinstance(expr, StringLiteral):
            return DataType.ADDRESS  # Address of string
        elif isinstance(expr, Identifier):
            sym = self.symbols.lookup(expr.name)
            if sym:
                return sym.data_type or DataType.ADDRESS
            return DataType.ADDRESS
        elif isinstance(expr, EmbeddedAssignExpr):
            # Type is determined by the target variable
            return self._get_expr_type(expr.target)
        elif isinstance(expr, BinaryExpr):
            # Comparisons return BYTE (0 or FF)
            if expr.op in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.GT,
                          BinaryOp.LE, BinaryOp.GE):
                return DataType.ADDRESS  # Actually returns FFFF or 0000
            # For arithmetic ops, check if both operands are bytes
            left_type = self._get_expr_type(expr.left)
            right_type = self._get_expr_type(expr.right)
            if left_type == DataType.BYTE and right_type == DataType.BYTE:
                if expr.op in (BinaryOp.ADD, BinaryOp.SUB, BinaryOp.AND, BinaryOp.OR, BinaryOp.XOR):
                    return DataType.BYTE
            return DataType.ADDRESS
        elif isinstance(expr, LocationExpr):
            return DataType.ADDRESS
        return DataType.ADDRESS

    def _gen_expr(self, expr: Expr) -> DataType:
        """
        Generate code for an expression.
        Result is left in A (for BYTE) or HL (for ADDRESS).
        Returns the type of the expression.
        """
        if isinstance(expr, NumberLiteral):
            # Use LXI H for all constants - more efficient (3 bytes vs 5 bytes)
            # Always return ADDRESS since value is in HL, not A
            self._emit("LXI", f"H,{self._format_number(expr.value)}")
            return DataType.ADDRESS

        elif isinstance(expr, StringLiteral):
            # Load address of string
            label = self._new_string_label()
            self.string_literals.append((label, expr.value))
            self._emit("LXI", f"H,{label}")
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
            # Return address of first constant
            if expr.values and isinstance(expr.values[0], NumberLiteral):
                self._emit("LXI", f"H,{self._format_number(expr.values[0].value)}")
            return DataType.ADDRESS

        elif isinstance(expr, EmbeddedAssignExpr):
            # Evaluate value
            val_type = self._gen_expr(expr.value)
            # Save value and store to target
            if val_type == DataType.BYTE:
                # Value is in A - save it in B, store, restore to A
                self._emit("MOV", "B,A")
                self._gen_store(expr.target, val_type)
                self._emit("MOV", "A,B")
            else:
                # Value is in HL - push, store, pop
                self._emit("PUSH", "H")
                self._gen_store(expr.target, val_type)
                self._emit("POP", "H")
            return val_type

        return DataType.ADDRESS

    def _gen_load(self, expr: Expr) -> DataType:
        """Load a variable value into A/HL. Returns the type."""
        if isinstance(expr, Identifier):
            name = expr.name

            # Handle built-in STACKPTR variable
            if name == "STACKPTR":
                # Read stack pointer into HL
                self._emit("LXI", "H,0")
                self._emit("DAD", "SP")  # HL = HL + SP = SP
                return DataType.ADDRESS

            # Check for LITERALLY macro - expand recursively
            if name in self.literal_macros:
                macro_val = self.literal_macros[name]
                try:
                    val = self._parse_plm_number(macro_val)
                    # Use LXI H for all constants - more efficient (3 bytes vs 5 bytes)
                    # Always return ADDRESS since value is in HL, not A
                    self._emit("LXI", f"H,{self._format_number(val)}")
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
                    self._emit("CALL", call_name)
                    # Result is in HL (for typed procedures) or undefined (for untyped)
                    if sym.return_type == DataType.BYTE:
                        # Move result from L to A for consistency
                        self._emit("MOV", "A,L")
                        self._emit("MOV", "L,A")
                        self._emit("MVI", "H,0")
                        return DataType.BYTE
                    return sym.return_type or DataType.ADDRESS

                if sym.kind == SymbolKind.LITERAL:
                    try:
                        val = int(sym.literal_value or "0", 0)
                        # Use LXI H for all constants - more efficient (3 bytes vs 5 bytes)
                        # Always return ADDRESS since value is in HL, not A
                        self._emit("LXI", f"H,{self._format_number(val)}")
                        return DataType.ADDRESS
                    except ValueError:
                        self._emit("LXI", f"H,{sym.literal_value}")
                        return DataType.ADDRESS

                # Check for BASED variable
                if sym.based_on:
                    # Load the base pointer first - look up the actual asm_name
                    base_sym = self.symbols.lookup(sym.based_on)
                    base_asm_name = base_sym.asm_name if base_sym and base_sym.asm_name else sym.based_on
                    self._emit("LHLD", base_asm_name)
                    # Then load from the pointed-to address
                    if sym.data_type == DataType.BYTE:
                        self._emit("MOV", "A,M")
                        # Keep BYTE value in A register
                        return DataType.BYTE
                    else:
                        self._emit("MOV", "E,M")
                        self._emit("INX", "H")
                        self._emit("MOV", "D,M")
                        self._emit("XCHG")
                        return DataType.ADDRESS

                if sym.data_type == DataType.BYTE:
                    self._emit("LDA", asm_name)
                    # Keep BYTE value in A register for efficient byte operations
                    return DataType.BYTE
                else:
                    self._emit("LHLD", asm_name)
                    return DataType.ADDRESS

            # Unknown symbol - assume ADDRESS
            self._emit("LHLD", asm_name)
            return DataType.ADDRESS

        else:
            # Complex lvalue - generate address then load
            self._gen_location(LocationExpr(operand=expr))
            self._emit("MOV", "A,M")
            # Keep BYTE value in A register
            return DataType.BYTE

    def _gen_store(self, expr: Expr, val_type: DataType) -> None:
        """Store A/HL to a variable."""
        if isinstance(expr, Identifier):
            name = expr.name

            # Handle built-in STACKPTR variable
            if name == "STACKPTR":
                # Set stack pointer from HL
                self._emit("SPHL")  # SP = HL
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
                        self._emit("MOV", "A,L")  # Get byte value into A
                    self._emit("MOV", "B,A")  # Save value in B
                    self._emit("LHLD", base_asm_name)
                    self._emit("MOV", "A,B")  # Restore value
                    self._emit("MOV", "M,A")  # Store via HL
                else:
                    # Save value in HL
                    self._emit("PUSH", "H")
                    self._emit("LHLD", base_asm_name)
                    self._emit("XCHG")  # DE = address
                    self._emit("POP", "H")  # HL = value
                    self._emit("XCHG")  # HL = address, DE = value
                    self._emit("MOV", "M,E")
                    self._emit("INX", "H")
                    self._emit("MOV", "M,D")
                return

            if sym and sym.data_type == DataType.BYTE:
                # Value may be in A (if val_type==BYTE) or L (if val_type==ADDRESS)
                if val_type != DataType.BYTE:
                    self._emit("MOV", "A,L")
                self._emit("STA", asm_name)
            else:
                self._emit("SHLD", asm_name)

        elif isinstance(expr, SubscriptExpr):
            # Check for MEMORY(addr) = value special case
            if isinstance(expr.base, Identifier) and expr.base.name.upper() == "MEMORY":
                # MEMORY(addr) = value - store byte to memory address
                self._emit("PUSH", "H")  # Save value
                self._gen_expr(expr.index)  # HL = address
                self._emit("XCHG")  # DE = address
                self._emit("POP", "H")  # HL = value
                self._emit("MOV", "A,L")
                self._emit("STAX", "D")
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
                    self._emit("MOV", "A,L")  # Value in A
                    self._emit("OUT", self._format_number(port_num))
                else:
                    # Variable port - need runtime support
                    self._emit("PUSH", "H")  # Save value
                    self._gen_expr(port_arg)  # Evaluate port number
                    self._emit("MOV", "C,L")  # Port in C
                    self._emit("POP", "H")  # Restore value
                    self._emit("MOV", "A,L")  # Value in A
                    self._emit("CALL", "??OUTP")
                    self.needs_runtime.add("??OUTP")
                return

            # Array element store
            self._emit("PUSH", "H")  # Save value
            self._gen_subscript_addr(expr)  # HL = address
            self._emit("XCHG")  # DE = address
            self._emit("POP", "H")  # HL = value
            self._emit("MOV", "A,L")
            self._emit("STAX", "D")

        elif isinstance(expr, MemberExpr):
            # Structure member store
            _, member_type = self._get_member_info(expr)
            self._emit("PUSH", "H")
            self._gen_member_addr(expr)
            self._emit("XCHG")  # DE = member address
            self._emit("POP", "H")  # HL = value
            if member_type == DataType.ADDRESS:
                # Store 16-bit value
                self._emit("XCHG")  # HL = address, DE = value
                self._emit("MOV", "M,E")
                self._emit("INX", "H")
                self._emit("MOV", "M,D")
            else:
                # Store 8-bit value
                self._emit("MOV", "A,L")
                self._emit("STAX", "D")

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
                    self._emit("MOV", "A,L")  # Value in A
                    self._emit("OUT", self._format_number(port_num))
                else:
                    # Variable port - need runtime support
                    self._emit("PUSH", "H")  # Save value
                    self._gen_expr(port_arg)  # Evaluate port number
                    self._emit("MOV", "C,L")  # Port in C
                    self._emit("POP", "H")  # Restore value
                    self._emit("MOV", "A,L")  # Value in A
                    self._emit("CALL", "??OUTP")
                    self.needs_runtime.add("??OUTP")
                return
            # Unknown call target - fall through to complex store
            self._emit("PUSH", "H")
            self._gen_location(LocationExpr(operand=expr))
            self._emit("XCHG")
            self._emit("POP", "H")
            if val_type == DataType.BYTE:
                self._emit("MOV", "A,L")
                self._emit("STAX", "D")
            else:
                self._emit("XCHG")
                self._emit("MOV", "M,E")
                self._emit("INX", "H")
                self._emit("MOV", "M,D")
            return

        else:
            # Complex store
            self._emit("PUSH", "H")  # Save value
            self._gen_location(LocationExpr(operand=expr))  # HL = address
            self._emit("XCHG")  # DE = address
            self._emit("POP", "H")  # HL = value
            # Store based on type
            if val_type == DataType.BYTE:
                self._emit("MOV", "A,L")
                self._emit("STAX", "D")
            else:
                self._emit("XCHG")  # HL = address, DE = value
                self._emit("MOV", "M,E")
                self._emit("INX", "H")
                self._emit("MOV", "M,D")

    def _gen_binary(self, expr: BinaryExpr) -> DataType:
        """Generate code for binary expression."""
        op = expr.op

        # Determine operand types for optimization
        left_type = self._get_expr_type(expr.left)
        right_type = self._get_expr_type(expr.right)
        both_bytes = (left_type == DataType.BYTE and right_type == DataType.BYTE)

        # Special case: byte comparison with constant - use CPI
        if op in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.GT,
                  BinaryOp.LE, BinaryOp.GE):
            if both_bytes and isinstance(expr.right, NumberLiteral) and expr.right.value <= 255:
                return self._gen_byte_comparison_const(expr.left, op, expr.right.value)
            elif both_bytes:
                return self._gen_byte_comparison(expr.left, expr.right, op)

        # For byte operations, use efficient byte path
        if both_bytes and op in (BinaryOp.ADD, BinaryOp.SUB, BinaryOp.AND,
                                  BinaryOp.OR, BinaryOp.XOR):
            return self._gen_byte_binary(expr.left, expr.right, op)

        # Optimize ADDRESS +/- small constant: use INX/DCX
        if op == BinaryOp.ADD and isinstance(expr.right, NumberLiteral):
            const_val = expr.right.value
            if 1 <= const_val <= 4:  # Small constants: use repeated INX
                self._gen_expr(expr.left)
                for _ in range(const_val):
                    self._emit("INX", "H")
                return DataType.ADDRESS
        elif op == BinaryOp.SUB and isinstance(expr.right, NumberLiteral):
            const_val = expr.right.value
            if 1 <= const_val <= 4:  # Small constants: use repeated DCX
                self._gen_expr(expr.left)
                for _ in range(const_val):
                    self._emit("DCX", "H")
                return DataType.ADDRESS

        # Fall through to 16-bit operations
        # Evaluate left operand
        left_result = self._gen_expr(expr.left)
        if left_result == DataType.BYTE:
            # Extend A to HL
            self._emit("MOV", "L,A")
            self._emit("MVI", "H,0")
        self._emit("PUSH", "H")  # Save left on stack

        # Evaluate right operand
        right_result = self._gen_expr(expr.right)
        if right_result == DataType.BYTE:
            # Extend A to HL
            self._emit("MOV", "L,A")
            self._emit("MVI", "H,0")

        # Pop left into DE
        self._emit("XCHG")  # DE = right
        self._emit("POP", "H")  # HL = left
        # Now: HL = left, DE = right

        if op == BinaryOp.ADD:
            self._emit("DAD", "D")  # HL = HL + DE

        elif op == BinaryOp.SUB:
            # HL = HL - DE
            self._emit("MOV", "A,L")
            self._emit("SUB", "E")
            self._emit("MOV", "L,A")
            self._emit("MOV", "A,H")
            self._emit("SBB", "D")
            self._emit("MOV", "H,A")

        elif op == BinaryOp.MUL:
            self.needs_runtime.add("MUL16")
            self._emit("CALL", "??MUL16")

        elif op == BinaryOp.DIV:
            self.needs_runtime.add("DIV16")
            self._emit("CALL", "??DIV16")

        elif op == BinaryOp.MOD:
            self.needs_runtime.add("MOD16")
            self._emit("CALL", "??MOD16")

        elif op == BinaryOp.AND:
            self._emit("MOV", "A,L")
            self._emit("ANA", "E")
            self._emit("MOV", "L,A")
            self._emit("MOV", "A,H")
            self._emit("ANA", "D")
            self._emit("MOV", "H,A")

        elif op == BinaryOp.OR:
            self._emit("MOV", "A,L")
            self._emit("ORA", "E")
            self._emit("MOV", "L,A")
            self._emit("MOV", "A,H")
            self._emit("ORA", "D")
            self._emit("MOV", "H,A")

        elif op == BinaryOp.XOR:
            self._emit("MOV", "A,L")
            self._emit("XRA", "E")
            self._emit("MOV", "L,A")
            self._emit("MOV", "A,H")
            self._emit("XRA", "D")
            self._emit("MOV", "H,A")

        elif op in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.GT,
                   BinaryOp.LE, BinaryOp.GE):
            self._gen_comparison(op)

        elif op == BinaryOp.PLUS:
            # PLUS: add with carry from previous operation
            self._emit("MOV", "A,L")
            self._emit("ADC", "E")
            self._emit("MOV", "L,A")
            self._emit("MOV", "A,H")
            self._emit("ADC", "D")
            self._emit("MOV", "H,A")

        elif op == BinaryOp.MINUS:
            # MINUS: subtract with borrow from previous operation
            self._emit("MOV", "A,L")
            self._emit("SBB", "E")
            self._emit("MOV", "L,A")
            self._emit("MOV", "A,H")
            self._emit("SBB", "D")
            self._emit("MOV", "H,A")

        return DataType.ADDRESS

    def _gen_comparison(self, op: BinaryOp) -> None:
        """Generate code for comparison. HL=left, DE=right. Result in HL (0 or FFFF)."""
        true_label = self._new_label("TRUE")
        end_label = self._new_label("CMP")

        # Subtract: HL - DE
        self._emit("MOV", "A,L")
        self._emit("SUB", "E")
        self._emit("MOV", "B,A")  # Save low result
        self._emit("MOV", "A,H")
        self._emit("SBB", "D")
        # Now: A = high byte of (left-right), B = low byte, flags set

        if op == BinaryOp.EQ:
            self._emit("ORA", "B")  # OR high and low
            self._emit("JZ", true_label)
        elif op == BinaryOp.NE:
            self._emit("ORA", "B")
            self._emit("JNZ", true_label)
        elif op == BinaryOp.LT:
            # left < right if borrow occurred
            self._emit("JC", true_label)
        elif op == BinaryOp.GE:
            self._emit("JNC", true_label)
        elif op == BinaryOp.GT:
            # left > right: no borrow AND not equal
            self._emit("JC", end_label)  # If left < right, false
            self._emit("ORA", "B")
            self._emit("JNZ", true_label)  # If not equal, left > right
        elif op == BinaryOp.LE:
            self._emit("JC", true_label)  # left < right
            self._emit("ORA", "B")
            self._emit("JZ", true_label)  # left == right

        # False case
        self._emit("LXI", "H,0")
        self._emit("JMP", end_label)

        # True case
        self._emit_label(true_label)
        self._emit("LXI", "H,0FFFFH")

        self._emit_label(end_label)

    def _gen_byte_comparison_const(self, left: Expr, op: BinaryOp, const_val: int) -> DataType:
        """Generate optimized byte comparison with constant using CPI."""
        # Load left operand into A
        left_type = self._gen_expr(left)
        if left_type != DataType.BYTE:
            # If not already a byte, take low byte
            self._emit("MOV", "A,L")

        # Compare with constant
        self._emit("CPI", self._format_number(const_val))

        # Generate result based on comparison type
        true_label = self._new_label("TRUE")
        end_label = self._new_label("CMP")

        if op == BinaryOp.EQ:
            self._emit("JZ", true_label)
        elif op == BinaryOp.NE:
            self._emit("JNZ", true_label)
        elif op == BinaryOp.LT:
            self._emit("JC", true_label)
        elif op == BinaryOp.GE:
            self._emit("JNC", true_label)
        elif op == BinaryOp.GT:
            # A > const: not equal AND not less (JNC and JNZ)
            self._emit("JC", end_label)  # If less, false
            self._emit("JZ", end_label)  # If equal, false
            self._emit("JMP", true_label)  # Otherwise true
        elif op == BinaryOp.LE:
            self._emit("JC", true_label)  # Less than -> true
            self._emit("JZ", true_label)  # Equal -> true

        # False case
        self._emit("LXI", "H,0")
        self._emit("JMP", end_label)

        # True case
        self._emit_label(true_label)
        self._emit("LXI", "H,0FFFFH")

        self._emit_label(end_label)
        return DataType.ADDRESS  # Comparisons return ADDRESS (0 or FFFF)

    def _gen_byte_comparison(self, left: Expr, right: Expr, op: BinaryOp) -> DataType:
        """Generate optimized byte comparison between two byte values."""
        # Load left into A, save to B
        self._gen_expr(left)  # Result in A
        self._emit("MOV", "B,A")  # Save left in B

        # Load right into A
        self._gen_expr(right)  # Result in A
        # Now B = left, A = right

        # Compare: B - A (left - right)
        self._emit("MOV", "C,A")  # Save right in C
        self._emit("MOV", "A,B")  # A = left
        self._emit("SUB", "C")    # A = left - right, flags set

        # Generate result
        true_label = self._new_label("TRUE")
        end_label = self._new_label("CMP")

        if op == BinaryOp.EQ:
            self._emit("JZ", true_label)
        elif op == BinaryOp.NE:
            self._emit("JNZ", true_label)
        elif op == BinaryOp.LT:
            self._emit("JC", true_label)
        elif op == BinaryOp.GE:
            self._emit("JNC", true_label)
        elif op == BinaryOp.GT:
            self._emit("JC", end_label)
            self._emit("JZ", end_label)
            self._emit("JMP", true_label)
        elif op == BinaryOp.LE:
            self._emit("JC", true_label)
            self._emit("JZ", true_label)

        # False case
        self._emit("LXI", "H,0")
        self._emit("JMP", end_label)

        # True case
        self._emit_label(true_label)
        self._emit("LXI", "H,0FFFFH")

        self._emit_label(end_label)
        return DataType.ADDRESS

    def _gen_byte_binary(self, left: Expr, right: Expr, op: BinaryOp) -> DataType:
        """Generate optimized byte arithmetic/logical operation."""
        # Special case: right is constant - use immediate instructions
        if isinstance(right, NumberLiteral) and right.value <= 255:
            self._gen_expr_to_a(left)  # Load left into A
            const = self._format_number(right.value)
            if op == BinaryOp.ADD:
                self._emit("ADI", const)  # A = A + const
            elif op == BinaryOp.SUB:
                self._emit("SUI", const)  # A = A - const
            elif op == BinaryOp.AND:
                self._emit("ANI", const)  # A = A AND const
            elif op == BinaryOp.OR:
                self._emit("ORI", const)  # A = A OR const
            elif op == BinaryOp.XOR:
                self._emit("XRI", const)  # A = A XOR const
            return DataType.BYTE

        # General case: load left into A, save to B
        self._gen_expr_to_a(left)
        self._emit("MOV", "B,A")  # Save left in B

        # Load right into A
        self._gen_expr_to_a(right)
        # Now B = left, A = right

        # Perform operation: result = left op right
        if op == BinaryOp.ADD:
            self._emit("ADD", "B")  # A = A + B = right + left
        elif op == BinaryOp.SUB:
            self._emit("MOV", "C,A")  # C = right
            self._emit("MOV", "A,B")  # A = left
            self._emit("SUB", "C")    # A = left - right
        elif op == BinaryOp.AND:
            self._emit("ANA", "B")  # A = A AND B
        elif op == BinaryOp.OR:
            self._emit("ORA", "B")  # A = A OR B
        elif op == BinaryOp.XOR:
            self._emit("XRA", "B")  # A = A XOR B

        # Result is in A, return BYTE
        return DataType.BYTE

    def _gen_expr_to_a(self, expr: Expr) -> None:
        """Generate code to load an expression into A (for byte operations)."""
        if isinstance(expr, NumberLiteral):
            if expr.value <= 255:
                self._emit("MVI", f"A,{self._format_number(expr.value)}")
            else:
                # Large constant - load low byte
                self._emit("MVI", f"A,{self._format_number(expr.value & 0xFF)}")
        else:
            result_type = self._gen_expr(expr)
            if result_type == DataType.ADDRESS:
                # Value is in HL, get low byte into A
                self._emit("MOV", "A,L")

    def _gen_unary(self, expr: UnaryExpr) -> DataType:
        """Generate code for unary expression."""
        self._gen_expr(expr.operand)

        if expr.op == UnaryOp.NEG:
            # Negate HL: HL = 0 - HL
            self._emit("MOV", "A,L")
            self._emit("CMA")
            self._emit("MOV", "L,A")
            self._emit("MOV", "A,H")
            self._emit("CMA")
            self._emit("MOV", "H,A")
            self._emit("INX", "H")

        elif expr.op == UnaryOp.NOT:
            # Complement HL
            self._emit("MOV", "A,L")
            self._emit("CMA")
            self._emit("MOV", "L,A")
            self._emit("MOV", "A,H")
            self._emit("CMA")
            self._emit("MOV", "H,A")

        elif expr.op == UnaryOp.LOW:
            self._emit("MVI", "H,0")
            return DataType.BYTE

        elif expr.op == UnaryOp.HIGH:
            self._emit("MOV", "L,H")
            self._emit("MVI", "H,0")
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

        self._gen_subscript_addr(expr)
        # Load value at address
        self._emit("MOV", "A,M")
        self._emit("MOV", "L,A")
        self._emit("MVI", "H,0")
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
            if sym and sym.data_type == DataType.ADDRESS:
                elem_size = 2

        # OPTIMIZATION: Constant folding for label+constant
        # If base is a simple identifier (label) and index is constant, fold them
        if isinstance(expr.base, Identifier) and isinstance(expr.index, NumberLiteral):
            sym = self.symbols.lookup(expr.base.name)
            if sym and not sym.based_on:
                # Regular array with constant index - can fold: LXI H,label+offset
                asm_name = sym.asm_name if sym.asm_name else self._mangle_name(expr.base.name)
                offset = expr.index.value * elem_size
                if offset == 0:
                    self._emit("LXI", f"H,{asm_name}")
                else:
                    self._emit("LXI", f"H,{asm_name}+{offset}")
                return

        # Get base address (non-constant or BASED variable case)
        if isinstance(expr.base, Identifier):
            sym = self.symbols.lookup(expr.base.name)
            if sym and sym.based_on:
                # BASED variable - load the base pointer from the based_on variable
                base_sym = self.symbols.lookup(sym.based_on)
                base_asm_name = base_sym.asm_name if base_sym and base_sym.asm_name else self._mangle_name(sym.based_on)
                self._emit("LHLD", base_asm_name)
            else:
                # Regular array - use address of array
                asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(expr.base.name)
                self._emit("LXI", f"H,{asm_name}")
        else:
            self._gen_expr(expr.base)

        # Optimize for constant index (only reached for BASED or computed base)
        if isinstance(expr.index, NumberLiteral):
            offset = expr.index.value * elem_size
            if offset == 0:
                # Index 0 - base address is already correct
                pass
            elif offset <= 255:
                # Small offset - can use LXI D,offset; DAD D
                self._emit("LXI", f"D,{offset}")
                self._emit("DAD", "D")
            else:
                # Large offset
                self._emit("LXI", f"D,{offset}")
                self._emit("DAD", "D")
        else:
            # Variable index - need to compute
            idx_type = self._get_expr_type(expr.index)

            if idx_type == DataType.BYTE and elem_size == 1:
                # Optimized byte index: avoid PUSH/POP
                # HL = base, put into DE, compute index into HL, add
                self._emit("XCHG")  # DE = base
                self._gen_expr(expr.index)  # A = index (byte)
                self._emit("MOV", "L,A")
                self._emit("MVI", "H,0")  # HL = index (zero-extended)
                self._emit("DAD", "D")  # HL = index + base
            else:
                # General case with PUSH/POP
                self._emit("PUSH", "H")  # Save base

                # Get index
                result_type = self._gen_expr(expr.index)

                # If index was BYTE (in A), extend to HL
                if result_type == DataType.BYTE:
                    self._emit("MOV", "L,A")
                    self._emit("MVI", "H,0")

                if elem_size == 2:
                    # Multiply index by 2
                    self._emit("DAD", "H")

                # Add index to base
                self._emit("POP", "D")
                self._emit("DAD", "D")

    def _get_member_info(self, expr: MemberExpr) -> tuple[int, DataType]:
        """Get offset and type for a structure member."""
        offset = 0
        member_type = DataType.BYTE

        # Get the base variable's symbol to find struct_members
        if isinstance(expr.base, Identifier):
            sym = self.symbols.lookup(expr.base.name)
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
        offset, member_type = self._get_member_info(expr)
        self._gen_member_addr(expr)

        if member_type == DataType.ADDRESS:
            # Load 16-bit value
            self._emit("MOV", "E,M")
            self._emit("INX", "H")
            self._emit("MOV", "D,M")
            self._emit("XCHG")  # HL = value
            return DataType.ADDRESS
        else:
            # Load 8-bit value
            self._emit("MOV", "A,M")
            self._emit("MOV", "L,A")
            self._emit("MVI", "H,0")
            return DataType.BYTE

    def _gen_member_addr(self, expr: MemberExpr) -> None:
        """Generate code to compute address of structure member."""
        self._gen_expr(expr.base)

        offset, _ = self._get_member_info(expr)

        # Add offset to base address (in HL)
        if offset > 0:
            self._emit("LXI", f"D,{offset}")
            self._emit("DAD", "D")

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
                    # Generate direct BDOS call: MVI C,func; LXI D,addr; CALL 5
                    self._emit("MVI", f"C,{self._format_number(func_num)}")
                    addr_type = self._gen_expr(addr_arg)
                    if addr_type == DataType.BYTE:
                        self._emit("MOV", "E,A")
                        self._emit("MVI", "D,0")
                    else:
                        self._emit("XCHG")  # DE = addr
                    self._emit("CALL", "5")  # BDOS entry point
                    # Result is in A (for MON2) - also copy to HL for consistency
                    if name.upper() == 'MON2':
                        self._emit("MOV", "L,A")
                        self._emit("MVI", "H,0")
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
                    self._emit("MOV", "L,A")
                    self._emit("MVI", "H,0")
                self._emit("PUSH", "H")
        else:
            # Direct memory parameter passing (non-reentrant)
            for i, arg in enumerate(expr.args):
                if sym and i < len(sym.params):
                    param_name = sym.params[i]
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

                    arg_type = self._gen_expr(arg)
                    if arg_type == DataType.BYTE:
                        self._emit("STA", param_asm)
                    else:
                        self._emit("SHLD", param_asm)

        if isinstance(expr.callee, Identifier):
            self._emit("CALL", call_name)
        else:
            self._gen_expr(expr.callee)
            self._emit("PCHL")

        # Clean up stack - only for stack-based calls
        if use_stack and expr.args:
            for _ in expr.args:
                self._emit("POP", "D")  # Dummy pop

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
                    self._emit("IN", self._format_number(port_num))
                else:
                    # Variable port - need runtime support (rare in practice)
                    self._gen_expr(arg)
                    self._emit("CALL", "??INP")
                    self.needs_runtime.add("??INP")
            else:
                self._emit("IN", "0")
            self._emit("MOV", "L,A")
            self._emit("MVI", "H,0")
            return DataType.BYTE

        if name == "LOW":
            self._gen_expr(args[0])
            self._emit("MVI", "H,0")
            return DataType.BYTE

        if name == "HIGH":
            self._gen_expr(args[0])
            self._emit("MOV", "L,H")
            self._emit("MVI", "H,0")
            return DataType.BYTE

        if name == "DOUBLE":
            self._gen_expr(args[0])
            # Value is already in HL, just ensure H is set
            return DataType.ADDRESS

        if name == "SHL":
            self._gen_expr(args[0])
            self._emit("PUSH", "H")
            self._gen_expr(args[1])
            self._emit("MOV", "C,L")  # Count in C
            self._emit("POP", "H")   # Value in HL
            shift_loop = self._new_label("SHL")
            end_label = self._new_label("SHLE")
            self._emit_label(shift_loop)
            self._emit("DCR", "C")
            self._emit("JM", end_label)
            self._emit("DAD", "H")  # HL = HL * 2
            self._emit("JMP", shift_loop)
            self._emit_label(end_label)
            return DataType.ADDRESS

        if name == "SHR":
            self._gen_expr(args[0])
            self._emit("PUSH", "H")
            self._gen_expr(args[1])
            self._emit("MOV", "C,L")
            self._emit("POP", "H")
            shift_loop = self._new_label("SHR")
            end_label = self._new_label("SHRE")
            self._emit_label(shift_loop)
            self._emit("DCR", "C")
            self._emit("JM", end_label)
            self._emit("ORA", "A")  # Clear carry
            self._emit("MOV", "A,H")
            self._emit("RAR")
            self._emit("MOV", "H,A")
            self._emit("MOV", "A,L")
            self._emit("RAR")
            self._emit("MOV", "L,A")
            self._emit("JMP", shift_loop)
            self._emit_label(end_label)
            return DataType.ADDRESS

        if name == "ROL":
            self._gen_expr(args[0])
            self._emit("PUSH", "H")
            self._gen_expr(args[1])
            self._emit("MOV", "C,L")
            self._emit("POP", "H")
            self._emit("MOV", "A,L")
            shift_loop = self._new_label("ROL")
            end_label = self._new_label("ROLE")
            self._emit_label(shift_loop)
            self._emit("DCR", "C")
            self._emit("JM", end_label)
            self._emit("RLC")
            self._emit("JMP", shift_loop)
            self._emit_label(end_label)
            self._emit("MOV", "L,A")
            self._emit("MVI", "H,0")
            return DataType.BYTE

        if name == "ROR":
            self._gen_expr(args[0])
            self._emit("PUSH", "H")
            self._gen_expr(args[1])
            self._emit("MOV", "C,L")
            self._emit("POP", "H")
            self._emit("MOV", "A,L")
            shift_loop = self._new_label("ROR")
            end_label = self._new_label("RORE")
            self._emit_label(shift_loop)
            self._emit("DCR", "C")
            self._emit("JM", end_label)
            self._emit("RRC")
            self._emit("JMP", shift_loop)
            self._emit_label(end_label)
            self._emit("MOV", "L,A")
            self._emit("MVI", "H,0")
            return DataType.BYTE

        if name == "LENGTH":
            # Returns array dimension
            if args and isinstance(args[0], Identifier):
                sym = self.symbols.lookup(args[0].name)
                if sym and sym.dimension:
                    self._emit("LXI", f"H,{sym.dimension}")
                    return DataType.ADDRESS
            self._emit("LXI", "H,0")
            return DataType.ADDRESS

        if name == "LAST":
            # Returns array dimension - 1
            if args and isinstance(args[0], Identifier):
                sym = self.symbols.lookup(args[0].name)
                if sym and sym.dimension:
                    self._emit("LXI", f"H,{sym.dimension - 1}")
                    return DataType.ADDRESS
            self._emit("LXI", "H,0")
            return DataType.ADDRESS

        if name == "SIZE":
            # Returns size in bytes
            if args and isinstance(args[0], Identifier):
                sym = self.symbols.lookup(args[0].name)
                if sym:
                    self._emit("LXI", f"H,{sym.size}")
                    return DataType.ADDRESS
            self._emit("LXI", "H,0")
            return DataType.ADDRESS

        if name == "MEMORY":
            # MEMORY(addr) - direct memory access as a byte array at address
            # Generate address into HL
            self._gen_expr(args[0])
            # Load byte from (HL)
            self._emit("MOV", "A,M")
            self._emit("MOV", "L,A")
            self._emit("MVI", "H,0")
            return DataType.BYTE

        if name == "MOVE":
            # MOVE(count, source, dest)
            self.needs_runtime.add("MOVE")
            for arg in args:
                self._gen_expr(arg)
                self._emit("PUSH", "H")
            self._emit("CALL", "??MOVE")
            # Clean up - MOVE does its own stack cleanup
            return None

        if name == "TIME":
            # Delay loop
            self._gen_expr(args[0])
            loop_label = self._new_label("TIME")
            self._emit_label(loop_label)
            self._emit("DCX", "H")
            self._emit("MOV", "A,H")
            self._emit("ORA", "L")
            self._emit("JNZ", loop_label)
            return None

        if name == "CARRY":
            # Return carry flag value
            self._emit("MVI", "A,0")
            self._emit("RAL")  # Rotate carry into A
            self._emit("MOV", "L,A")
            self._emit("MVI", "H,0")
            return DataType.BYTE

        if name == "ZERO":
            # Return zero flag value
            true_label = self._new_label("ZF")
            end_label = self._new_label("ZFE")
            self._emit("JZ", true_label)
            self._emit("LXI", "H,0")
            self._emit("JMP", end_label)
            self._emit_label(true_label)
            self._emit("LXI", "H,0FFH")
            self._emit_label(end_label)
            return DataType.BYTE

        if name == "SIGN":
            # Return sign flag value
            true_label = self._new_label("SF")
            end_label = self._new_label("SFE")
            self._emit("JM", true_label)
            self._emit("LXI", "H,0")
            self._emit("JMP", end_label)
            self._emit_label(true_label)
            self._emit("LXI", "H,0FFH")
            self._emit_label(end_label)
            return DataType.BYTE

        if name == "PARITY":
            # Return parity flag value
            true_label = self._new_label("PF")
            end_label = self._new_label("PFE")
            self._emit("JPE", true_label)
            self._emit("LXI", "H,0")
            self._emit("JMP", end_label)
            self._emit_label(true_label)
            self._emit("LXI", "H,0FFH")
            self._emit_label(end_label)
            return DataType.BYTE

        if name == "DEC":
            # Convert binary value (0-15) to ASCII decimal digit ('0'-'9')
            # Values 10-15 wrap to produce '0'-'5'
            self._gen_expr(args[0])
            self._emit("MOV", "A,L")
            self._emit("ANI", "0FH")  # Mask to 0-15
            self._emit("ADI", "30H")  # Add '0' ASCII code
            self._emit("MOV", "L,A")
            self._emit("MVI", "H,0")
            return DataType.BYTE

        if name == "SCL":
            # Shift through carry left
            self._gen_expr(args[0])
            self._emit("PUSH", "H")
            self._gen_expr(args[1])
            self._emit("MOV", "C,L")
            self._emit("POP", "H")
            self._emit("MOV", "A,L")
            shift_loop = self._new_label("SCL")
            end_label = self._new_label("SCLE")
            self._emit_label(shift_loop)
            self._emit("DCR", "C")
            self._emit("JM", end_label)
            self._emit("RAL")  # Rotate through carry
            self._emit("JMP", shift_loop)
            self._emit_label(end_label)
            self._emit("MOV", "L,A")
            self._emit("MVI", "H,0")
            return DataType.BYTE

        if name == "SCR":
            # Shift through carry right
            self._gen_expr(args[0])
            self._emit("PUSH", "H")
            self._gen_expr(args[1])
            self._emit("MOV", "C,L")
            self._emit("POP", "H")
            self._emit("MOV", "A,L")
            shift_loop = self._new_label("SCR")
            end_label = self._new_label("SCRE")
            self._emit_label(shift_loop)
            self._emit("DCR", "C")
            self._emit("JM", end_label)
            self._emit("RAR")  # Rotate through carry
            self._emit("JMP", shift_loop)
            self._emit_label(end_label)
            self._emit("MOV", "L,A")
            self._emit("MVI", "H,0")
            return DataType.BYTE

        # Not a built-in we handle inline
        return None

    def _gen_location(self, expr: LocationExpr) -> DataType:
        """Generate code to load address of expression."""
        operand = expr.operand
        if isinstance(operand, Identifier):
            name = operand.name

            # Check for built-in MEMORY - its address is 0
            if name.upper() == "MEMORY":
                self._emit("LXI", "H,0")
                return DataType.ADDRESS

            # Check for LITERALLY macro - expand recursively
            if name in self.literal_macros:
                macro_val = self.literal_macros[name]
                try:
                    # Numeric literal - load as immediate address
                    val = self._parse_plm_number(macro_val)
                    self._emit("LXI", f"H,{self._format_number(val)}")
                    return DataType.ADDRESS
                except ValueError:
                    # Non-numeric literal - recursively process
                    return self._gen_location(LocationExpr(operand=Identifier(name=macro_val)))
            # Mangle name if needed
            sym = self.symbols.lookup(name)
            asm_name = sym.asm_name if sym and sym.asm_name else self._mangle_name(name)
            self._emit("LXI", f"H,{asm_name}")
        elif isinstance(operand, SubscriptExpr):
            self._gen_subscript_addr(operand)
        elif isinstance(operand, MemberExpr):
            self._gen_member_addr(operand)
        elif isinstance(operand, StringLiteral):
            # .('string') - address of inline string
            label = self._new_string_label()
            self.string_literals.append((label, operand.value))
            self._emit("LXI", f"H,{label}")
        elif isinstance(operand, ConstListExpr):
            # .(const, const, ...) - address of inline data
            label = self._new_label("DATA")
            self.data_segment.append(AsmLine(label=label))
            for val in operand.values:
                if isinstance(val, NumberLiteral):
                    self.data_segment.append(
                        AsmLine(opcode="DB", operands=self._format_number(val.value))
                    )
                elif isinstance(val, StringLiteral):
                    self.data_segment.append(
                        AsmLine(opcode="DB", operands=self._escape_string(val.value))
                    )
            self._emit("LXI", f"H,{label}")
        else:
            # Just evaluate the expression
            self._gen_expr(operand)
        return DataType.ADDRESS


def generate(module: Module, target: Target = Target.I8080) -> str:
    """Convenience function to generate code from a module."""
    gen = CodeGenerator(target)
    return gen.generate(module)
