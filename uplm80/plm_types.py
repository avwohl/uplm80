"""PL/M-80 expression types and constant arithmetic.

What Intel's PL/M-80 Programming Manual (9800268B) says an operator does to
its operands' types, and what it computes, for code that evaluates PL/M
expressions at compile time -- the AST optimizer's constant folding and
propagation, and the code generator's checks on constant operands:

* A numeric constant not greater than 255 is a BYTE, a larger one an ADDRESS
  (4.1.1). A one-character string is a BYTE constant and a two-character one
  an ADDRESS whose high byte is the first character.
* ``+ -`` (4.2.1) and ``AND OR XOR`` (4.3) of two BYTEs are 8-bit operations
  with a BYTE result; if either operand is an ADDRESS the BYTE one is
  zero-extended and the operation and its result are 16-bit. PLUS and MINUS
  "perform similarly to + and -" (12.2), with the carry added in.
* Unary ``-`` is ``0 - x`` "of the same type as the operand" (4.2.2), so
  ``-1`` is the BYTE 0FFH; ``NOT`` complements the bits of its operand, 8 or
  16 of them.
* ``* / MOD`` always give an ADDRESS (4.2.3, 4.2.4). DRI's PL/M-80 divides
  with PLM80.LIB's one 16-bit routine: see :func:`runtime.plm_div`.
* A relation gives the BYTE 0FFH or 00H (4.4).
* LOW, HIGH, ROL and ROR are BYTE procedures, DOUBLE is ADDRESS (11.1.3),
  and SCL and SCR take the type of their pattern (12.3).

One deliberate exception: the manual gives SHL and SHR their pattern's type
too (11.1.4), and DRI's compiler shifts a BYTE in A (UTIL4/ERA.PRL codes
`shl(dcnt,5)' as five `ADD A'), but uplm80 has always zero-extended a BYTE
pattern and shifted sixteen bits, with an ADDRESS result, and programs written
for it rely on that -- 80un builds words with `lo + shl(b, 8)'. SHL and SHR
are ADDRESS here, consistently in the folder and the code generator.

A folded ADDRESS constant below 256 cannot be written as a plain numeric
literal, which would be a BYTE; it is written ``DOUBLE(n)``, which every
consumer already treats as the ADDRESS it is.
"""

from __future__ import annotations

from typing import Callable, Optional

from . import _plm_parser as P
from .ast_view import (
    BinaryOpKind,
    DataType,
    UnaryOpKind,
    binop_kind,
    ident_text,
    make_identifier,
    make_number_literal,
    number_value,
    string_value,
    unop_kind,
    unwrap_paren,
)
from .runtime import plm_div, plm_mod

BYTE = DataType.BYTE
ADDRESS = DataType.ADDRESS

RELATIONS = frozenset({
    BinaryOpKind.EQ, BinaryOpKind.NE, BinaryOpKind.LT,
    BinaryOpKind.GT, BinaryOpKind.LE, BinaryOpKind.GE,
})
# Operators whose width follows their operands'.
WIDTH_FOLLOWS_OPERANDS = frozenset({
    BinaryOpKind.ADD, BinaryOpKind.SUB, BinaryOpKind.AND,
    BinaryOpKind.OR, BinaryOpKind.XOR, BinaryOpKind.PLUS, BinaryOpKind.MINUS,
})
ALWAYS_ADDRESS = frozenset({BinaryOpKind.MUL, BinaryOpKind.DIV, BinaryOpKind.MOD})

# Built-ins whose result type is that of their first argument.
PATTERN_TYPED_BUILTINS = frozenset({"SCL", "SCR"})
BYTE_BUILTINS = frozenset({"LOW", "HIGH", "ROL", "ROR", "INPUT", "DEC",
                           "CARRY", "ZERO", "SIGN", "PARITY", "MEMORY"})


def literal_type(value: int) -> DataType:
    """The type PL/M-80 gives a numeric constant."""
    return BYTE if value <= 0xFF else ADDRESS


def mask(t: DataType) -> int:
    """All the bits of a value of type ``t``."""
    return 0xFF if t is BYTE else 0xFFFF


def convert(value: int, t: DataType) -> int:
    """A value converted to ``t``: truncated, or zero-extended (4.6.1)."""
    return value & mask(t)


def binary_type(kind: BinaryOpKind, left: Optional[DataType],
                right: Optional[DataType]) -> Optional[DataType]:
    """Result type of ``left kind right``; None when it depends on an
    operand whose type is not known."""
    if kind in RELATIONS:
        return BYTE
    if kind in ALWAYS_ADDRESS:
        return ADDRESS
    if left is ADDRESS or right is ADDRESS:
        return ADDRESS
    if left is BYTE and right is BYTE:
        return BYTE
    return None


def fold_binary(kind: BinaryOpKind, lv: int, lt: DataType, rv: int,
                rt: DataType) -> Optional[tuple[int, DataType]]:
    """``lv kind rv`` on typed constants, as the program would compute it.

    None for PLUS and MINUS, whose carry-in is not a constant.
    """
    t = binary_type(kind, lt, rt)
    lv &= 0xFFFF
    rv &= 0xFFFF
    if kind in RELATIONS:
        holds = {
            BinaryOpKind.EQ: lv == rv, BinaryOpKind.NE: lv != rv,
            BinaryOpKind.LT: lv < rv, BinaryOpKind.GT: lv > rv,
            BinaryOpKind.LE: lv <= rv, BinaryOpKind.GE: lv >= rv,
        }[kind]
        return (0xFF if holds else 0), BYTE
    if kind == BinaryOpKind.MUL:
        return (lv * rv) & 0xFFFF, ADDRESS
    if kind == BinaryOpKind.DIV:
        return plm_div(lv, rv), ADDRESS
    if kind == BinaryOpKind.MOD:
        return plm_mod(lv, rv), ADDRESS
    assert t is not None
    if kind == BinaryOpKind.ADD:
        return convert(lv + rv, t), t
    if kind == BinaryOpKind.SUB:
        return convert(lv - rv, t), t
    if kind == BinaryOpKind.AND:
        return lv & rv, t
    if kind == BinaryOpKind.OR:
        return lv | rv, t
    if kind == BinaryOpKind.XOR:
        return lv ^ rv, t
    return None


def fold_unary(kind: UnaryOpKind, value: int, t: DataType) -> tuple[int, DataType]:
    """``-x`` or ``NOT x`` of a typed constant, in the operand's width."""
    if kind == UnaryOpKind.NEG:
        return convert(-value, t), t
    return convert(~value, t), t


def fold_builtin(name: str, args: list[tuple[int, DataType]]
                 ) -> Optional[tuple[int, DataType]]:
    """A built-in procedure of typed constant arguments, or None."""
    name = name.upper()
    if len(args) == 1:
        (v, t), = args
        if name == "LOW":
            return v & 0xFF, BYTE
        if name == "HIGH":
            return ((v >> 8) & 0xFF if t is ADDRESS else 0), BYTE
        if name == "DOUBLE":
            return v, ADDRESS
        return None
    if len(args) == 2:
        (v, _), (count, _) = args
        count &= 0xFF          # the count is converted to a BYTE
        # uplm80's SHL and SHR are 16-bit, of a BYTE zero-extended.
        if name == "SHL":
            return (v << count) & 0xFFFF, ADDRESS
        if name == "SHR":
            return (v & 0xFFFF) >> count, ADDRESS
        if name in ("ROL", "ROR"):
            v &= 0xFF
            n = count & 7
            if name == "ROL":
                return ((v << n) | (v >> (8 - n))) & 0xFF, BYTE
            return ((v >> n) | (v << (8 - n))) & 0xFF, BYTE
    return None


# ---- typed constants in the AST ---------------------------------------------

# The token name a NumberLiteral carries when the optimizer computed it from
# something that was not a constant in the source (a propagated variable, a
# dropped operand). Code generation's check that a BYTE is not compared with
# a constant it can never equal is about what the programmer wrote, and must
# not fire on these.
DERIVED = "DERIVED_NUMBER"


def is_double_call(expr) -> bool:
    """Whether ``expr`` is a call of the built-in DOUBLE."""
    expr = unwrap_paren(expr)
    if not (isinstance(expr, P.Call) and len(expr.args) == 1):
        return False
    callee = unwrap_paren(expr.callee)
    return isinstance(callee, P.Identifier) and ident_text(callee.name).upper() == "DOUBLE"


def is_derived(expr) -> bool:
    """Whether ``expr`` holds a literal the optimizer derived."""
    expr = unwrap_paren(expr)
    if isinstance(expr, P.NumberLiteral):
        return getattr(expr.value, "name", None) == DERIVED
    if is_double_call(expr):
        return is_derived(unwrap_paren(expr).args[0])
    return False


def typed_const(expr) -> Optional[tuple[int, DataType]]:
    """(value, type) of a constant operand, or None.

    A numeric literal, a string of one character (a BYTE) or two (an
    ADDRESS, the first character high), or ``DOUBLE`` of one.
    """
    expr = unwrap_paren(expr)
    if isinstance(expr, P.NumberLiteral):
        v = number_value(expr)
        return v, literal_type(v)
    if isinstance(expr, P.StringLiteral):
        s = string_value(expr)
        if len(s) == 1:
            return ord(s), BYTE
        if len(s) == 2:
            return (ord(s[0]) << 8) | ord(s[1]), ADDRESS
        return None
    if is_double_call(expr):
        inner = typed_const(expr.args[0])
        if inner is not None:
            return inner[0], ADDRESS
    return None


def make_typed_const(value: int, t: DataType, pos=None, derived: bool = False):
    """An expression node for the constant ``value`` of type ``t``."""
    value = convert(value, t)
    lit = make_number_literal(value, pos=pos)
    if derived:
        lit.value.name = DERIVED
    if t is ADDRESS and value <= 0xFF:
        return P.Call(callee=make_identifier("DOUBLE", pos=pos), args=[lit], pos=pos)
    return lit


def untyped_root(expr):
    """``expr`` with a ``DOUBLE`` of a constant at its root removed.

    Where a value is converted to a known type before anything else sees it
    -- stored, passed, returned, tested, used as a subscript -- a constant's
    own type makes no difference, and the plain literal is what code
    generation has its short forms for.
    """
    e = unwrap_paren(expr)
    if is_double_call(e) and typed_const(e.args[0]) is not None:
        return unwrap_paren(e.args[0])
    return expr


def eval_typed(expr, ident_value: Callable[[str], Optional[int]] | None = None
               ) -> Optional[tuple[int, DataType, bool]]:
    """Evaluate a constant expression by PL/M-80's rules.

    Returns (value, type, derived) or None when ``expr`` is not constant.
    ``ident_value`` resolves an identifier naming a constant (a LITERALLY
    macro); its value is typed like a literal.
    """
    expr = unwrap_paren(expr)
    if isinstance(expr, P.NumberLiteral):
        v = number_value(expr) & 0xFFFF
        return v, literal_type(v), is_derived(expr)
    if isinstance(expr, P.StringLiteral):
        s = string_value(expr)
        if len(s) == 1:
            return ord(s), BYTE, False
        if len(s) == 2:
            return (ord(s[0]) << 8) | ord(s[1]), ADDRESS, False
        return None
    if isinstance(expr, P.Identifier):
        if ident_value is None:
            return None
        v = ident_value(ident_text(expr.name))
        if v is None:
            return None
        v &= 0xFFFF
        return v, literal_type(v), False
    if isinstance(expr, P.UnaryOp):
        inner = eval_typed(expr.operand, ident_value)
        if inner is None:
            return None
        v, t = fold_unary(unop_kind(expr), inner[0], inner[1])
        return v, t, inner[2]
    if isinstance(expr, P.BinaryOp):
        left = eval_typed(expr.left, ident_value)
        right = eval_typed(expr.right, ident_value)
        if left is None or right is None:
            return None
        folded = fold_binary(binop_kind(expr), left[0], left[1], right[0], right[1])
        if folded is None:
            return None
        return folded[0], folded[1], left[2] or right[2]
    if isinstance(expr, P.Call):
        callee = unwrap_paren(expr.callee)
        if not isinstance(callee, P.Identifier):
            return None
        args = [eval_typed(a, ident_value) for a in expr.args]
        if not args or any(a is None for a in args):
            return None
        folded = fold_builtin(ident_text(callee.name),
                              [(a[0], a[1]) for a in args])  # type: ignore[index]
        if folded is None:
            return None
        return folded[0], folded[1], any(a[2] for a in args)  # type: ignore[index]
    return None
