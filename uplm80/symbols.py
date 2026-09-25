"""
Symbol table for the PL/M-80 compiler.

Tracks variables, procedures, labels, and their attributes across scopes.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from .ast_nodes import DataType, StructMember


class SymbolKind(Enum):
    """Kind of symbol."""

    VARIABLE = auto()
    PROCEDURE = auto()
    LABEL = auto()
    LITERAL = auto()  # LITERALLY macro
    PARAMETER = auto()
    BUILTIN = auto()


@dataclass
class Symbol:
    """A symbol in the symbol table."""

    name: str
    kind: SymbolKind
    data_type: DataType | None = None
    # For variables
    dimension: int | None = None  # Array dimension
    struct_members: list[StructMember] | None = None
    based_on: str | None = None
    # For `x BASED s.m', the member of `based_on' that holds the
    # pointer.  The pointer is then at based_on + that member's
    # offset, not at based_on itself.
    based_member: str | None = None
    at_address: int | None = None
    is_public: bool = False
    is_external: bool = False
    # For procedures
    params: list[str] = field(default_factory=list)
    param_types: list[DataType] = field(default_factory=list)  # Types of parameters
    return_type: DataType | None = None
    is_reentrant: bool = False
    # Takes its single argument in A (BYTE) or HL (ADDRESS), not in C or BC
    # as PL/M-80's convention passes it: a procedure with one parameter that
    # nothing outside the compile can reach (not PUBLIC, EXTERNAL or
    # REENTRANT, its address never taken).  Set by
    # CodeGenerator._mark_register_params once the compile is surveyed.
    uses_reg_param: bool = False
    interrupt_num: int | None = None
    # For literals (macros)
    literal_value: str | None = None
    # Memory location (assigned during code gen)
    address: int | None = None
    size: int = 0  # Size in bytes
    # Scope level where defined
    scope_level: int = 0
    # Mangled assembly name (for register conflicts)
    asm_name: str | None = None
    # Stack offset for reentrant procedure locals (IX+offset)
    stack_offset: int | None = None


@dataclass
class Scope:
    """A scope containing symbols."""

    symbols: dict[str, Symbol] = field(default_factory=dict)
    parent: "Scope | None" = None
    level: int = 0
    name: str = ""  # Procedure or module name

    def define(self, symbol: Symbol) -> None:
        """Define a symbol in this scope."""
        symbol.scope_level = self.level
        self.symbols[symbol.name] = symbol

    def lookup_local(self, name: str) -> Symbol | None:
        """Look up a symbol in this scope only."""
        return self.symbols.get(name)

    def lookup(self, name: str) -> Symbol | None:
        """Look up a symbol, searching parent scopes."""
        sym = self.symbols.get(name)
        if sym is not None:
            return sym
        if self.parent is not None:
            return self.parent.lookup(name)
        return None


class SymbolTable:
    """Symbol table with nested scopes."""

    def __init__(self) -> None:
        self.global_scope = Scope(level=0, name="<global>")
        self.current_scope = self.global_scope
        self._init_builtins()

    def _init_builtins(self) -> None:
        """Initialize built-in symbols."""
        # Built-in procedures
        builtins = [
            # I/O
            ("INPUT", DataType.BYTE, ["port"]),
            ("OUTPUT", DataType.BYTE, ["port"]),
            # Byte manipulation
            ("LOW", DataType.BYTE, ["value"]),
            ("HIGH", DataType.BYTE, ["value"]),
            ("DOUBLE", DataType.ADDRESS, ["value"]),
            # Array info
            ("LENGTH", DataType.ADDRESS, ["array"]),
            ("LAST", DataType.ADDRESS, ["array"]),
            ("SIZE", DataType.ADDRESS, ["array"]),
            # Shifts and rotates
            ("SHL", DataType.ADDRESS, ["value", "count"]),
            ("SHR", DataType.ADDRESS, ["value", "count"]),
            ("ROL", DataType.BYTE, ["value", "count"]),
            ("ROR", DataType.BYTE, ["value", "count"]),
            ("SCL", DataType.BYTE, ["value", "count"]),
            ("SCR", DataType.BYTE, ["value", "count"]),
            # Memory
            ("MOVE", None, ["count", "source", "dest"]),
            # Timing
            ("TIME", None, ["count"]),
            # Flags
            ("CARRY", DataType.BYTE, []),
            ("SIGN", DataType.BYTE, []),
            ("ZERO", DataType.BYTE, []),
            ("PARITY", DataType.BYTE, []),
            # Carry operations
            ("DEC", DataType.BYTE, ["value"]),
        ]

        for name, return_type, params in builtins:
            self.global_scope.define(
                Symbol(
                    name=name,
                    kind=SymbolKind.BUILTIN,
                    data_type=return_type,
                    return_type=return_type,
                    params=params,
                )
            )

        # Built-in variables
        # MEMORY array - maps to all of memory
        self.global_scope.define(
            Symbol(
                name="MEMORY",
                kind=SymbolKind.VARIABLE,
                data_type=DataType.BYTE,
                dimension=65536,
                at_address=0,
            )
        )

        # STACKPTR - stack pointer
        self.global_scope.define(
            Symbol(
                name="STACKPTR",
                kind=SymbolKind.VARIABLE,
                data_type=DataType.ADDRESS,
            )
        )

    def enter_scope(self, name: str = "") -> Scope:
        """Enter a new scope."""
        new_scope = Scope(
            parent=self.current_scope,
            level=self.current_scope.level + 1,
            name=name,
        )
        self.current_scope = new_scope
        return new_scope

    def leave_scope(self) -> Scope:
        """Leave the current scope."""
        old_scope = self.current_scope
        if self.current_scope.parent is not None:
            self.current_scope = self.current_scope.parent
        return old_scope

    def define(self, symbol: Symbol) -> None:
        """Define a symbol in the current scope."""
        self.current_scope.define(symbol)

    def lookup(self, name: str) -> Symbol | None:
        """Look up a symbol by name."""
        return self.current_scope.lookup(name)

    def lookup_local(self, name: str) -> Symbol | None:
        """Look up a symbol in current scope only."""
        return self.current_scope.lookup_local(name)

    def is_defined_local(self, name: str) -> bool:
        """Check if a symbol is defined in the current scope."""
        return self.current_scope.lookup_local(name) is not None
