"""Expressions keep the value and the type PL/M-80 gives them, at every -O.

Intel's PL/M-80 Programming Manual types every operand and operator: a
constant up to 255 is a BYTE (4.1.1); `+ - AND OR XOR` of two BYTEs, and
unary `-` and `NOT` of one, are 8-bit and wrap (4.2, 4.3); `* / MOD` are
ADDRESS (4.2.3); a relation is the BYTE 0FFH or 0 (4.4). Every rewrite the
compiler makes -- folding, propagation, strength reduction, algebra,
inlining -- has to leave the value and the type the unoptimized program
computes, or the arithmetic around the rewritten operand changes width.

Each case here printed something else at some level before the fix.
"""

import pytest

from tests.plm_difftest import build_and_run, generate, tools_missing
from uplm80.compiler import Compiler

LEVELS = (0, 1, 2, 3)

_PRELUDE = """t: do;
mon1: procedure (f, p) external; declare f byte, p address; end mon1;
hexd: procedure (d);
  declare d byte;
  d = d and 0fh;
  if d < 10 then call mon1(2, d + '0');
  else call mon1(2, d + 37h);
end hexd;
ph: procedure (v);
  declare v address;
  call hexd(shr(high(v), 4)); call hexd(high(v));
  call hexd(shr(low(v), 4)); call hexd(low(v));
  call mon1(2, ' ');
end ph;
"""


def _printed(body: str, opt: int) -> list[int]:
    reason = tools_missing()
    if reason:
        pytest.skip(reason)
    got, err = build_and_run(_PRELUDE + body + "\nend t;\n", opt)
    assert err is None, err
    return got


def _check(body: str, expect: list[int]) -> None:
    for opt in LEVELS:
        got = _printed(body, opt)
        assert got == expect, (
            f"-O{opt}: printed {' '.join(f'{v:04X}' for v in got)}, "
            f"expected {' '.join(f'{v:04X}' for v in expect)}")


def _asm(src: str, opt: int = 2) -> str:
    out = Compiler(opt_level=opt).compile(src, "<test>")
    assert out is not None, "compilation failed"
    return out


# ---- code generation the differential test found wrong ---------------------

def test_multiply_keeps_only_the_low_sixteen_bits():
    """??mul16 rotated the carry of its own add into the multiplicand, so a
    product that overflowed was wrong: 81H * 511 gave 817FH."""
    values = [0, 1, 0x81, 0xFF, 0x1FF, 0x7FFF, 0x8001, 0xFFFF, 0x1234]
    body = "declare (i, j, a, b) address;\nrun: procedure;\n"
    body += "  declare av(*) address data (" + ", ".join(f"0{v:x}h" for v in values) + ");\n"
    body += "  do i = 0 to last(av); do j = 0 to last(av);\n"
    body += "    a = av(i); b = av(j); call ph(a * b);\n  end; end;\nend run;\ncall run;\n"
    _check(body, [(a * b) & 0xFFFF for a in values for b in values])


def test_nested_subscripts():
    """A subscript's release popped the spill of an enclosing subscript's
    claim on DE, so `aw(aw(aw(i) AND 7) AND 7)' added a stale DE in place of
    the array's base."""
    _check("""
declare (b3) byte, (w4) address;
run: procedure;
  declare aw(*) address data (076h, 081h, 03h, 0eah, 0ff00h, 0d24ah, 07h, 02h);
  b3 = 0; w4 = 1182h;
  call ph(aw(aw(aw(b3 and 7) and 7) and 7));
  call ph(aw(aw(aw(b3) and 7) and 7) + w4);
end run;
call run;
""", [2, 0x1184])


def test_one_minus_a_byte():
    """`1 - x' was compiled `x XOR 1', right only for 0 and 1."""
    _check("""
declare b byte, r address;
run: procedure;
  declare bd(*) byte data (0ffh);
  b = bd(0);
  r = 1 - b; call ph(r);
  r = 5 - b; call ph(r);
end run;
call run;
""", [2, 6])


def test_a_byte_stored_through_a_based_address():
    """A BYTE stored through a BASED ADDRESS was taken from HL, not A."""
    _check("""
declare b byte, (w, p) address;
declare bw based p address;
declare wd(*) address data (1234h, 5678h);
run: procedure;
  w = wd(0); b = 5;
  p = .w; bw = (b = 5); call ph(w);
  w = wd(1);
  p = .w; bw = b; call ph(w);
end run;
call run;
""", [0xFF, 5])


def test_a_byte_subscript_of_memory_is_widened():
    """MEMORY(b) added __END__ to whatever HL held: a BYTE subscript is
    generated into A."""
    _check("""
declare (b, x) byte;
run: procedure;
  b = 3;
  memory(3) = 77h;
  x = memory(b); call ph(x);
  memory(b) = 66h; call ph(memory(3));
end run;
call run;
""", [0x77, 0x66])
