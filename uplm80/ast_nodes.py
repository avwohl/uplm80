"""
Abstract Syntax Tree (AST) node definitions for PL/M-80.

These nodes represent the structure of a PL/M-80 program after parsing.
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum, auto


class DataType(Enum):
    """PL/M-80 data types."""

    BYTE = auto()  # 8-bit unsigned
    ADDRESS = auto()  # 16-bit unsigned
    LABEL = auto()  # Label type
    PROCEDURE = auto()  # Procedure type


class BinaryOp(Enum):
    """Binary operators."""

    ADD = auto()  # +
    SUB = auto()  # -
    MUL = auto()  # *
    DIV = auto()  # /
    MOD = auto()  # MOD
    AND = auto()  # AND
    OR = auto()  # OR
    XOR = auto()  # XOR
    EQ = auto()  # =
    NE = auto()  # <>
    LT = auto()  # <
    GT = auto()  # >
    LE = auto()  # <=
    GE = auto()  # >=
    PLUS = auto()  # PLUS (carry-aware)
    MINUS = auto()  # MINUS (carry-aware)


class UnaryOp(Enum):
    """Unary operators."""

    NEG = auto()  # -
    NOT = auto()  # NOT
    LOW = auto()  # LOW
    HIGH = auto()  # HIGH


@dataclass
class SourceSpan:
    """Source location span for AST nodes."""

    start_line: int
    start_col: int
    end_line: int
    end_col: int
    filename: str = "<unknown>"


# ============================================================================
# Base AST Node
# ============================================================================


@dataclass(kw_only=True)
class ASTNode:
    """Base class for all AST nodes."""

    span: SourceSpan | None = field(default=None, repr=False)


# ============================================================================
# Expression Nodes
# ============================================================================


@dataclass
class Expr(ASTNode):
    """Base class for expression nodes."""

    pass


@dataclass
class NumberLiteral(Expr):
    """Numeric constant."""

    value: int


@dataclass
class StringLiteral(Expr):
    """String constant. Value is stored as bytes."""

    value: str  # The string content
    bytes_value: list[int] = field(default_factory=list)  # ASCII values


@dataclass
class Identifier(Expr):
    """Variable or procedure reference."""

    name: str


@dataclass
class SubscriptExpr(Expr):
    """Array subscript: name(index)"""

    base: Expr  # The array/identifier
    index: Expr  # The subscript expression


@dataclass
class MemberExpr(Expr):
    """Structure member access: struct.member"""

    base: Expr  # The structure
    member: str  # Member name


@dataclass
class CallExpr(Expr):
    """Function/procedure call in expression context."""

    callee: Expr  # The procedure being called
    args: list[Expr] = field(default_factory=list)


@dataclass
class BinaryExpr(Expr):
    """Binary operation."""

    op: BinaryOp
    left: Expr
    right: Expr


@dataclass
class UnaryExpr(Expr):
    """Unary operation."""

    op: UnaryOp
    operand: Expr


@dataclass
class LocationExpr(Expr):
    """Location reference using dot operator: .variable"""

    operand: Expr


@dataclass
class ConstListExpr(Expr):
    """Constant list for DATA: .(const, const, ...)"""

    values: list[Expr] = field(default_factory=list)


@dataclass
class EmbeddedAssignExpr(Expr):
    """Embedded assignment: (var := expr)"""

    target: Expr
    value: Expr


# ============================================================================
# Statement Nodes
# ============================================================================


@dataclass
class Stmt(ASTNode):
    """Base class for statement nodes."""

    pass


@dataclass
class AssignStmt(Stmt):
    """Assignment statement: target = value; or multiple: a,b = value;"""

    targets: list[Expr]  # Multiple targets for multiple assignment
    value: Expr


@dataclass
class CallStmt(Stmt):
    """CALL statement."""

    callee: Expr
    args: list[Expr] = field(default_factory=list)


@dataclass
class ReturnStmt(Stmt):
    """RETURN statement."""

    value: Expr | None = None  # None for untyped procedures


@dataclass
class GotoStmt(Stmt):
    """GOTO statement."""

    target: str  # Label name


@dataclass
class HaltStmt(Stmt):
    """HALT statement."""

    pass


@dataclass
class EnableStmt(Stmt):
    """ENABLE statement."""

    pass


@dataclass
class DisableStmt(Stmt):
    """DISABLE statement."""

    pass


@dataclass
class NullStmt(Stmt):
    """Null statement (just a semicolon)."""

    pass


@dataclass
class LabeledStmt(Stmt):
    """A statement with a label."""

    label: str
    stmt: Stmt


# ============================================================================
# Block and Control Flow Nodes
# ============================================================================


@dataclass
class Block(ASTNode):
    """A block of statements."""

    stmts: list[Stmt] = field(default_factory=list)


@dataclass
class IfStmt(Stmt):
    """IF statement."""

    condition: Expr
    then_stmt: Stmt
    else_stmt: Stmt | None = None


@dataclass
class DoBlock(Stmt):
    """Simple DO block."""

    decls: list["Declaration"] = field(default_factory=list)
    stmts: list[Stmt] = field(default_factory=list)
    end_label: str | None = None


@dataclass
class DoWhileBlock(Stmt):
    """DO WHILE block."""

    condition: Expr
    stmts: list[Stmt] = field(default_factory=list)
    end_label: str | None = None


@dataclass
class DoIterBlock(Stmt):
    """Iterative DO block: DO var = start TO bound [BY step]."""

    index_var: Expr
    start: Expr
    bound: Expr
    step: Expr | None = None
    stmts: list[Stmt] = field(default_factory=list)
    end_label: str | None = None


@dataclass
class DoCaseBlock(Stmt):
    """DO CASE block."""

    selector: Expr
    cases: list[list[Stmt]] = field(default_factory=list)  # List of case bodies
    end_label: str | None = None


# ============================================================================
# Declaration Nodes
# ============================================================================


@dataclass
class Declaration(ASTNode):
    """Base class for declarations."""

    pass


@dataclass
class StructMember:
    """Member of a structure."""

    name: str
    data_type: DataType
    dimension: int | None = None  # Array dimension if any


@dataclass
class VarDecl(Declaration):
    """Variable declaration."""

    name: str
    data_type: DataType | None = None  # None for structure type
    dimension: int | None = None  # Array dimension (None for scalar)
    struct_members: list[StructMember] | None = None  # For STRUCTURE type
    based_on: str | None = None  # For BASED variables
    based_member: str | None = None  # Member of based variable
    at_location: Expr | None = None  # For AT attribute
    is_public: bool = False
    is_external: bool = False
    initial_values: list[Expr] | None = None
    data_values: list[Expr] | None = None  # For DATA attribute


@dataclass
class LabelDecl(Declaration):
    """Label declaration."""

    name: str
    is_public: bool = False
    is_external: bool = False


@dataclass
class LiterallyDecl(Declaration):
    """LITERALLY (macro) declaration."""

    name: str
    value: str  # The replacement text


@dataclass
class ProcDecl(Declaration):
    """Procedure declaration."""

    name: str
    params: list[str] = field(default_factory=list)  # Parameter names
    return_type: DataType | None = None  # None for untyped
    is_public: bool = False
    is_external: bool = False
    is_reentrant: bool = False
    interrupt_num: int | None = None
    decls: list[Declaration] = field(default_factory=list)
    stmts: list[Stmt] = field(default_factory=list)


@dataclass
class DeclareStmt(Stmt):
    """DECLARE statement containing one or more declarations."""

    declarations: list[Declaration] = field(default_factory=list)


# ============================================================================
# Module (Top-Level)
# ============================================================================


@dataclass
class Module(ASTNode):
    """A PL/M-80 module (compilation unit)."""

    name: str
    origin: int | None = None  # Origin address if specified
    decls: list[Declaration] = field(default_factory=list)
    stmts: list[Stmt] = field(default_factory=list)
