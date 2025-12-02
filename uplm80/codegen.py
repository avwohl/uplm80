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

    def __init__(self, target: Target = Target.I8080) -> None:
        self.target = target
        self.symbols = SymbolTable()
        self.output: list[AsmLine] = []
        self.label_counter = 0
        self.string_counter = 0
        self.data_segment: list[AsmLine] = []
        self.string_literals: list[tuple[str, str]] = []  # (label, value)
        self.current_proc: str | None = None
        self.loop_stack: list[tuple[str, str]] = []  # (continue_label, break_label)

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

    # ========================================================================
    # Main Entry Point
    # ========================================================================

    def generate(self, module: Module) -> str:
        """Generate assembly code for a module."""
        self.output = []
        self.data_segment = []
        self.string_literals = []

        # Header
        self._emit(comment=f"PL/M-80 Compiler Output - {module.name}")
        self._emit(comment=f"Target: {'8080' if self.target == Target.I8080 else 'Z80'}")
        self._emit()

        # Origin if specified
        if module.origin is not None:
            self._emit("ORG", f"{module.origin:04X}H")
            self._emit()

        # Process declarations (allocate storage)
        for decl in module.decls:
            self._gen_declaration(decl)

        # Generate code for module-level statements
        if module.stmts:
            self._emit()
            self._emit(comment="Module initialization code")
            for stmt in module.stmts:
                self._gen_stmt(stmt)

        # Emit string literals
        if self.string_literals:
            self._emit()
            self._emit(comment="String literals")
            for label, value in self.string_literals:
                self._emit_label(label)
                # Emit as DB with escaped bytes
                escaped = self._escape_string(value)
                self._emit("DB", escaped)

        # Emit data segment
        if self.data_segment:
            self._emit()
            self._emit(comment="Data segment")
            self.output.extend(self.data_segment)

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
        return "".join(parts)

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
            # Just record in symbol table, no code generated
            self.symbols.define(
                Symbol(
                    name=decl.name,
                    kind=SymbolKind.LITERAL,
                    literal_value=decl.value,
                )
            )
        elif isinstance(decl, LabelDecl):
            self.symbols.define(
                Symbol(
                    name=decl.name,
                    kind=SymbolKind.LABEL,
                    is_public=decl.is_public,
                    is_external=decl.is_external,
                )
            )

    def _gen_var_decl(self, decl: VarDecl) -> None:
        """Generate storage for a variable declaration."""
        # Calculate size
        if decl.struct_members:
            size = sum(
                (m.dimension or 1) * (1 if m.data_type == DataType.BYTE else 2)
                for m in decl.struct_members
            )
        else:
            elem_size = 1 if decl.data_type == DataType.BYTE else 2
            count = decl.dimension or 1
            size = elem_size * count

        # Record in symbol table
        sym = Symbol(
            name=decl.name,
            kind=SymbolKind.VARIABLE,
            data_type=decl.data_type,
            dimension=decl.dimension,
            struct_members=decl.struct_members,
            based_on=decl.based_on,
            is_public=decl.is_public,
            is_external=decl.is_external,
            size=size,
        )
        self.symbols.define(sym)

        # External variables don't get storage here
        if decl.is_external:
            self._emit("EXTRN", decl.name)
            return

        # Public declaration
        if decl.is_public:
            self._emit("PUBLIC", decl.name)

        # Based variables don't allocate storage
        if decl.based_on:
            return

        # AT variables use specified address
        if decl.at_location:
            # AT location is resolved at link time or as EQU
            self._emit(f"{decl.name}", "EQU", "$")  # Placeholder
            return

        # Generate storage in data segment
        if decl.initial_values:
            # Initialized data
            self.data_segment.append(AsmLine(label=decl.name))
            for val in decl.initial_values:
                if isinstance(val, NumberLiteral):
                    directive = "DB" if decl.data_type == DataType.BYTE else "DW"
                    self.data_segment.append(AsmLine(opcode=directive, operands=f"{val.value}"))
                elif isinstance(val, StringLiteral):
                    self.data_segment.append(
                        AsmLine(opcode="DB", operands=self._escape_string(val.value))
                    )
        elif decl.data_values:
            # DATA initialization
            self.data_segment.append(AsmLine(label=decl.name))
            for val in decl.data_values:
                if isinstance(val, NumberLiteral):
                    directive = "DB" if decl.data_type == DataType.BYTE else "DW"
                    self.data_segment.append(AsmLine(opcode=directive, operands=f"{val.value}"))
                elif isinstance(val, StringLiteral):
                    self.data_segment.append(
                        AsmLine(opcode="DB", operands=self._escape_string(val.value))
                    )
        else:
            # Uninitialized storage
            directive = "DS"
            self.data_segment.append(AsmLine(label=decl.name, opcode=directive, operands=str(size)))

    def _gen_proc_decl(self, decl: ProcDecl) -> None:
        """Generate code for a procedure."""
        old_proc = self.current_proc
        self.current_proc = decl.name

        # Record in symbol table
        sym = Symbol(
            name=decl.name,
            kind=SymbolKind.PROCEDURE,
            return_type=decl.return_type,
            params=decl.params,
            is_public=decl.is_public,
            is_external=decl.is_external,
            is_reentrant=decl.is_reentrant,
            interrupt_num=decl.interrupt_num,
        )
        self.symbols.define(sym)

        if decl.is_external:
            self._emit("EXTRN", decl.name)
            self.current_proc = old_proc
            return

        self._emit()
        if decl.is_public:
            self._emit("PUBLIC", decl.name)

        self._emit_label(decl.name)

        # Procedure prologue
        if decl.interrupt_num is not None:
            # Interrupt handler - save all registers
            self._emit("PUSH", "PSW")
            self._emit("PUSH", "B")
            self._emit("PUSH", "D")
            self._emit("PUSH", "H")

        # Enter new scope
        self.symbols.enter_scope(decl.name)

        # Define parameters as local variables
        # Parameters are passed on stack in reverse order
        # TODO: Implement proper parameter passing

        # Generate code for local declarations
        for local_decl in decl.decls:
            self._gen_declaration(local_decl)

        # Generate code for statements
        for stmt in decl.stmts:
            self._gen_stmt(stmt)

        # Procedure epilogue (implicit return)
        if decl.interrupt_num is not None:
            self._emit("POP", "H")
            self._emit("POP", "D")
            self._emit("POP", "B")
            self._emit("POP", "PSW")
            self._emit("EI")
            self._emit("RET")
        else:
            self._emit("RET")

        self.symbols.leave_scope()
        self.current_proc = old_proc

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
            self._emit("JMP", stmt.target)
        elif isinstance(stmt, HaltStmt):
            self._emit("HLT")
        elif isinstance(stmt, EnableStmt):
            self._emit("EI")
        elif isinstance(stmt, DisableStmt):
            self._emit("DI")
        elif isinstance(stmt, NullStmt):
            pass  # No code
        elif isinstance(stmt, LabeledStmt):
            self._emit_label(stmt.label)
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
        self._gen_expr(stmt.value)

        # Store to each target
        for target in stmt.targets:
            self._gen_store(target)

    def _gen_call_stmt(self, stmt: CallStmt) -> None:
        """Generate code for a CALL statement."""
        # Push arguments in reverse order
        for arg in reversed(stmt.args):
            self._gen_expr(arg)
            self._emit("PUSH", "H")  # Assuming ADDRESS size for now

        # Call the procedure
        if isinstance(stmt.callee, Identifier):
            self._emit("CALL", stmt.callee.name)
        else:
            # Indirect call through HL
            self._gen_expr(stmt.callee)
            # PCHL for indirect call
            self._emit("PCHL")

        # Clean up stack (caller cleanup)
        if stmt.args:
            stack_adjust = len(stmt.args) * 2
            if stack_adjust <= 6:
                for _ in range(len(stmt.args)):
                    self._emit("POP", "B")  # Dummy pop
            else:
                self._emit("LXI", f"D,{stack_adjust}")
                self._emit("DAD", "D")
                self._emit("SPHL")

    def _gen_return(self, stmt: ReturnStmt) -> None:
        """Generate code for RETURN statement."""
        if stmt.value:
            self._gen_expr(stmt.value)
            # Return value in A (BYTE) or HL (ADDRESS)
        self._emit("RET")

    def _gen_if(self, stmt: IfStmt) -> None:
        """Generate code for IF statement."""
        else_label = self._new_label("ELSE")
        end_label = self._new_label("ENDIF")

        # Evaluate condition
        self._gen_expr(stmt.condition)

        # Test result (in A for BYTE, or L for ADDRESS)
        self._emit("ORA", "A")  # Set flags based on A
        self._emit("JZ", else_label if stmt.else_stmt else end_label)

        # Then branch
        self._gen_stmt(stmt.then_stmt)

        if stmt.else_stmt:
            self._emit("JMP", end_label)
            self._emit_label(else_label)
            self._gen_stmt(stmt.else_stmt)

        self._emit_label(end_label)

    def _gen_do_block(self, stmt: DoBlock) -> None:
        """Generate code for simple DO block."""
        # Enter scope
        self.symbols.enter_scope("")

        # Local declarations
        for decl in stmt.decls:
            self._gen_declaration(decl)

        # Statements
        for s in stmt.stmts:
            self._gen_stmt(s)

        self.symbols.leave_scope()

    def _gen_do_while(self, stmt: DoWhileBlock) -> None:
        """Generate code for DO WHILE block."""
        loop_label = self._new_label("WHILE")
        end_label = self._new_label("WEND")

        self.loop_stack.append((loop_label, end_label))

        self._emit_label(loop_label)

        # Evaluate condition
        self._gen_expr(stmt.condition)
        self._emit("ORA", "A")
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
        end_label = self._new_label("NEXT")

        self.loop_stack.append((test_label, end_label))

        # Initialize index variable
        self._gen_expr(stmt.start)
        self._gen_store(stmt.index_var)

        # Jump to test
        self._emit("JMP", test_label)

        # Loop body
        self._emit_label(loop_label)
        for s in stmt.stmts:
            self._gen_stmt(s)

        # Increment
        step_val = 1
        if stmt.step and isinstance(stmt.step, NumberLiteral):
            step_val = stmt.step.value

        self._gen_load(stmt.index_var)
        if step_val == 1:
            self._emit("INX", "H")
        else:
            self._emit("LXI", f"D,{step_val}")
            self._emit("DAD", "D")
        self._gen_store(stmt.index_var)

        # Test condition
        self._emit_label(test_label)
        self._gen_load(stmt.index_var)
        self._emit("XCHG")  # DE = index
        self._gen_expr(stmt.bound)  # HL = bound

        # Compare: if index > bound, exit
        # Subtract: HL = bound - index
        self._emit("MOV", "A,L")
        self._emit("SUB", "E")
        self._emit("MOV", "L,A")
        self._emit("MOV", "A,H")
        self._emit("SBB", "D")
        self._emit("MOV", "H,A")

        # If result negative (carry set), exit
        self._emit("JC", end_label)
        self._emit("JMP", loop_label)

        self._emit_label(end_label)
        self.loop_stack.pop()

    def _gen_do_case(self, stmt: DoCaseBlock) -> None:
        """Generate code for DO CASE block."""
        end_label = self._new_label("CASEND")
        case_labels = [self._new_label(f"CASE{i}") for i in range(len(stmt.cases))]

        # Evaluate selector
        self._gen_expr(stmt.selector)

        # Jump table approach for small number of cases
        # For simplicity, use sequential comparisons
        for i, (case_stmts, label) in enumerate(zip(stmt.cases, case_labels)):
            self._emit("CPI", str(i))
            self._emit("JZ", label)

        self._emit("JMP", end_label)  # Default: skip all

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

    def _gen_expr(self, expr: Expr) -> None:
        """
        Generate code for an expression.
        Result is left in A (for BYTE) or HL (for ADDRESS).
        """
        if isinstance(expr, NumberLiteral):
            if expr.value <= 255:
                self._emit("MVI", f"A,{expr.value}")
            else:
                self._emit("LXI", f"H,{expr.value}")

        elif isinstance(expr, StringLiteral):
            # Load address of string
            label = self._new_string_label()
            self.string_literals.append((label, expr.value))
            self._emit("LXI", f"H,{label}")

        elif isinstance(expr, Identifier):
            self._gen_load(expr)

        elif isinstance(expr, BinaryExpr):
            self._gen_binary(expr)

        elif isinstance(expr, UnaryExpr):
            self._gen_unary(expr)

        elif isinstance(expr, SubscriptExpr):
            self._gen_subscript(expr)

        elif isinstance(expr, MemberExpr):
            self._gen_member(expr)

        elif isinstance(expr, CallExpr):
            self._gen_call_expr(expr)

        elif isinstance(expr, LocationExpr):
            self._gen_location(expr)

        elif isinstance(expr, EmbeddedAssignExpr):
            self._gen_expr(expr.value)
            self._emit("PUSH", "H")
            self._gen_store(expr.target)
            self._emit("POP", "H")

    def _gen_load(self, expr: Expr) -> None:
        """Load a variable value into A or HL."""
        if isinstance(expr, Identifier):
            sym = self.symbols.lookup(expr.name)
            if sym and sym.kind == SymbolKind.LITERAL:
                # Macro expansion - try to evaluate
                try:
                    val = int(sym.literal_value or "0", 0)
                    if val <= 255:
                        self._emit("MVI", f"A,{val}")
                    else:
                        self._emit("LXI", f"H,{val}")
                except ValueError:
                    # Non-numeric literal
                    self._emit("LXI", f"H,{sym.literal_value}")
                return

            if sym and sym.data_type == DataType.BYTE:
                self._emit("LDA", expr.name)
            else:
                self._emit("LHLD", expr.name)
        else:
            # Complex lvalue - generate address then load
            self._gen_location(LocationExpr(expr))
            self._emit("MOV", "A,M")

    def _gen_store(self, expr: Expr) -> None:
        """Store A or HL to a variable."""
        if isinstance(expr, Identifier):
            sym = self.symbols.lookup(expr.name)
            if sym and sym.data_type == DataType.BYTE:
                self._emit("STA", expr.name)
            else:
                self._emit("SHLD", expr.name)
        else:
            # Complex lvalue - need to save value, compute address, then store
            self._emit("PUSH", "H")  # Save value
            self._gen_location(LocationExpr(expr))
            self._emit("XCHG")  # DE = address
            self._emit("POP", "H")  # HL = value
            self._emit("XCHG")  # HL = address, DE = value
            self._emit("MOV", "M,E")
            self._emit("INX", "H")
            self._emit("MOV", "M,D")

    def _gen_binary(self, expr: BinaryExpr) -> None:
        """Generate code for binary expression."""
        # Evaluate left operand
        self._gen_expr(expr.left)
        self._emit("PUSH", "H")  # Save left on stack

        # Evaluate right operand
        self._gen_expr(expr.right)

        # Pop left into DE
        self._emit("XCHG")  # DE = right
        self._emit("POP", "H")  # HL = left
        # Now: HL = left, DE = right

        op = expr.op

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
            # Call runtime multiply routine
            self._emit("CALL", "??MUL16")

        elif op == BinaryOp.DIV:
            # Call runtime divide routine
            self._emit("CALL", "??DIV16")

        elif op == BinaryOp.MOD:
            # Call runtime modulo routine
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

        elif op in (BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.GT, BinaryOp.LE, BinaryOp.GE):
            self._gen_comparison(op)

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
            # left < right if result is negative (and not equal)
            self._emit("JC", true_label)  # Borrow = less than
        elif op == BinaryOp.GE:
            self._emit("JNC", true_label)  # No borrow = greater or equal
        elif op == BinaryOp.GT:
            # left > right: swap and check <
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

    def _gen_unary(self, expr: UnaryExpr) -> None:
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
            self._emit("MOV", "A,L")
            self._emit("MVI", "H,0")

        elif expr.op == UnaryOp.HIGH:
            self._emit("MOV", "A,H")
            self._emit("MVI", "H,0")
            self._emit("MOV", "L,A")

    def _gen_subscript(self, expr: SubscriptExpr) -> None:
        """Generate code for array subscript."""
        # Get base address
        if isinstance(expr.base, Identifier):
            self._emit("LXI", f"H,{expr.base.name}")
        else:
            self._gen_expr(expr.base)

        self._emit("PUSH", "H")  # Save base

        # Get index
        self._gen_expr(expr.index)

        # TODO: Multiply by element size if ADDRESS array

        # Add index to base
        self._emit("POP", "D")
        self._emit("DAD", "D")

        # Load value at address
        self._emit("MOV", "A,M")

    def _gen_member(self, expr: MemberExpr) -> None:
        """Generate code for structure member access."""
        # This is simplified - full implementation needs struct layout info
        self._gen_expr(expr.base)
        # TODO: Add offset for member
        self._emit("MOV", "A,M")

    def _gen_call_expr(self, expr: CallExpr) -> None:
        """Generate code for function call expression."""
        # Handle built-in functions
        if isinstance(expr.callee, Identifier):
            name = expr.callee.name
            if self._gen_builtin(name, expr.args):
                return

        # Regular function call
        for arg in reversed(expr.args):
            self._gen_expr(arg)
            self._emit("PUSH", "H")

        if isinstance(expr.callee, Identifier):
            self._emit("CALL", expr.callee.name)
        else:
            self._gen_expr(expr.callee)
            self._emit("PCHL")

        # Clean up stack
        if expr.args:
            for _ in expr.args:
                self._emit("POP", "D")  # Dummy pop

        # Result is in HL (or A for BYTE)

    def _gen_builtin(self, name: str, args: list[Expr]) -> bool:
        """Generate code for built-in function. Returns True if handled."""
        if name == "INPUT":
            if args:
                self._gen_expr(args[0])
                self._emit("MOV", "C,L")
            self._emit("IN", "0")  # Port number should come from arg
            # TODO: Handle port number properly
            return True

        if name == "OUTPUT":
            # OUTPUT(port) = value is handled differently
            return False

        if name == "LOW":
            self._gen_expr(args[0])
            self._emit("MOV", "A,L")
            self._emit("MVI", "H,0")
            return True

        if name == "HIGH":
            self._gen_expr(args[0])
            self._emit("MOV", "A,H")
            self._emit("MVI", "H,0")
            self._emit("MOV", "L,A")
            return True

        if name == "SHL":
            self._gen_expr(args[0])
            self._emit("PUSH", "H")
            self._gen_expr(args[1])
            self._emit("MOV", "C,L")  # Count in C
            self._emit("POP", "H")  # Value in HL
            shift_loop = self._new_label("SHL")
            end_label = self._new_label("SHLE")
            self._emit_label(shift_loop)
            self._emit("DCR", "C")
            self._emit("JM", end_label)
            self._emit("DAD", "H")  # HL = HL * 2
            self._emit("JMP", shift_loop)
            self._emit_label(end_label)
            return True

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
            self._emit("MOV", "A,H")
            self._emit("RAR")  # Through carry
            self._emit("MOV", "H,A")
            self._emit("MOV", "A,L")
            self._emit("RAR")
            self._emit("MOV", "L,A")
            self._emit("JMP", shift_loop)
            self._emit_label(end_label)
            return True

        if name == "LENGTH":
            # Returns array dimension - need symbol info
            # For now, return 0
            self._emit("LXI", "H,0")
            return True

        if name == "DOUBLE":
            self._gen_expr(args[0])
            # Value already in HL, just ensure 16-bit
            return True

        # Not a built-in we handle inline
        return False

    def _gen_location(self, expr: LocationExpr) -> None:
        """Generate code to load address of expression."""
        operand = expr.operand
        if isinstance(operand, Identifier):
            self._emit("LXI", f"H,{operand.name}")
        elif isinstance(operand, SubscriptExpr):
            # Base address + index
            if isinstance(operand.base, Identifier):
                self._emit("LXI", f"H,{operand.base.name}")
            else:
                self._gen_expr(operand.base)
            self._emit("PUSH", "H")
            self._gen_expr(operand.index)
            self._emit("POP", "D")
            self._emit("DAD", "D")
        else:
            # Just evaluate the expression
            self._gen_expr(operand)


def generate(module: Module, target: Target = Target.I8080) -> str:
    """Convenience function to generate code from a module."""
    gen = CodeGenerator(target)
    return gen.generate(module)
