"""
PL/M-80 token definitions.

Token types and reserved words for the PL/M-80 lexer.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any


class TokenType(Enum):
    """Token types for PL/M-80."""

    # Literals
    NUMBER = auto()  # Numeric constants (binary, octal, decimal, hex)
    STRING = auto()  # String literals 'like this'

    # Identifier
    IDENTIFIER = auto()

    # Keywords
    ADDRESS = auto()
    AND = auto()
    AT = auto()
    BASED = auto()
    BY = auto()
    BYTE = auto()
    CALL = auto()
    CASE = auto()
    DATA = auto()
    DECLARE = auto()
    DISABLE = auto()
    DO = auto()
    ELSE = auto()
    ENABLE = auto()
    END = auto()
    EOF_KW = auto()  # EOF keyword (not end of file token)
    EXTERNAL = auto()
    GO = auto()
    GOTO = auto()
    HALT = auto()
    IF = auto()
    INITIAL = auto()
    INTERRUPT = auto()
    LABEL = auto()
    LITERALLY = auto()
    MINUS = auto()  # MINUS keyword (not operator)
    MOD = auto()
    NOT = auto()
    OR = auto()
    PLUS = auto()  # PLUS keyword (not operator)
    PROCEDURE = auto()
    PUBLIC = auto()
    REENTRANT = auto()
    RETURN = auto()
    STRUCTURE = auto()
    THEN = auto()
    TO = auto()
    WHILE = auto()
    XOR = auto()

    # Operators
    OP_PLUS = auto()  # +
    OP_MINUS = auto()  # -
    OP_STAR = auto()  # *
    OP_SLASH = auto()  # /
    OP_LT = auto()  # <
    OP_GT = auto()  # >
    OP_LE = auto()  # <=
    OP_GE = auto()  # >=
    OP_NE = auto()  # <>
    OP_EQ = auto()  # =

    # Delimiters
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    COMMA = auto()  # ,
    SEMICOLON = auto()  # ;
    COLON = auto()  # :
    DOT = auto()  # .
    DOLLAR = auto()  # $ (used in identifiers)

    # Special
    NEWLINE = auto()  # For line tracking
    EOF = auto()  # End of file


# Reserved words mapping
RESERVED_WORDS: dict[str, TokenType] = {
    "ADDRESS": TokenType.ADDRESS,
    "AND": TokenType.AND,
    "AT": TokenType.AT,
    "BASED": TokenType.BASED,
    "BY": TokenType.BY,
    "BYTE": TokenType.BYTE,
    "CALL": TokenType.CALL,
    "CASE": TokenType.CASE,
    "DATA": TokenType.DATA,
    "DECLARE": TokenType.DECLARE,
    "DISABLE": TokenType.DISABLE,
    "DO": TokenType.DO,
    "ELSE": TokenType.ELSE,
    "ENABLE": TokenType.ENABLE,
    "END": TokenType.END,
    "EOF": TokenType.EOF_KW,
    "EXTERNAL": TokenType.EXTERNAL,
    "GO": TokenType.GO,
    "GOTO": TokenType.GOTO,
    "HALT": TokenType.HALT,
    "IF": TokenType.IF,
    "INITIAL": TokenType.INITIAL,
    "INTERRUPT": TokenType.INTERRUPT,
    "LABEL": TokenType.LABEL,
    "LITERALLY": TokenType.LITERALLY,
    "MINUS": TokenType.MINUS,
    "MOD": TokenType.MOD,
    "NOT": TokenType.NOT,
    "OR": TokenType.OR,
    "PLUS": TokenType.PLUS,
    "PROCEDURE": TokenType.PROCEDURE,
    "PUBLIC": TokenType.PUBLIC,
    "REENTRANT": TokenType.REENTRANT,
    "RETURN": TokenType.RETURN,
    "STRUCTURE": TokenType.STRUCTURE,
    "THEN": TokenType.THEN,
    "TO": TokenType.TO,
    "WHILE": TokenType.WHILE,
    "XOR": TokenType.XOR,
}


@dataclass
class Token:
    """A single token from the lexer."""

    type: TokenType
    value: Any  # The actual value (string for identifiers, int for numbers, etc.)
    line: int  # Source line number (1-based)
    column: int  # Source column (1-based)
    lexeme: str  # Original source text

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, line={self.line}, col={self.column})"
