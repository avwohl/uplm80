"""A SHL of a BYTE whose lost bits are used is warned of.

SHL and SHR of a BYTE are a BYTE (Programming Manual 9800268B, 11.1.4), as
Intel's PL/M-80 V3.1 makes them, since 0.4.3; uplm80 before it shifted a
BYTE in sixteen bits, with an ADDRESS result, and kept the bits shifted out
of it.  Where a SHL can shift a set bit out, and what it is part of uses
the bits above the low byte - an ADDRESS stores it, it is an ADDRESS
argument or a subscript, a relation compares it, ... - the program means
something else now, and the compiler says so (uplm80/byte_shifts.py), at
every -O level.  Where it cannot, it says nothing.  V3.1 gives no warning.
"""

import pytest

from uplm80.compiler import Compiler

LEVELS = (0, 1, 2, 3)

# R's parameters B and N can be anything; K is a module-level BYTE that G
# sets and H does not; F returns at most 7.
_HEAD = """t: do;
declare k byte, (w, v) address, a(10) byte, s structure (m byte, n address);
p: procedure (x) address; declare x address; return x; end p;
q: procedure (x) byte; declare x byte; return x; end q;
f: procedure byte; if w = 0 then return 7; return 5; end f;
g: procedure; k = 200; end g;
h: procedure; w = 1; end h;
r: procedure (b, n);
  declare (b, n) byte, c byte;
"""
_TAIL = """end r;
k = 1;
call r(1, 2);
end t;
"""


def _warnings(stmts: str, opt: int, capsys) -> list[str]:
    capsys.readouterr()
    src = _HEAD + stmts + "\n" + _TAIL
    assert Compiler(opt_level=opt).compile(src, "T.PLM") is not None
    return [line for line in capsys.readouterr().err.splitlines() if "SHL of a BYTE" in line]


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("stmts, shl", [
    ("w = shl(b, 4);", "SHL(B, 4)"),                          # stored to an ADDRESS
    ("w = shl(b, 4) + 1;", "SHL(B, 4)"),
    ("w = shl(b, 4) + w;", "SHL(B, 4)"),                      # beside an ADDRESS
    ("w = p(shl(b, 2));", "SHL(B, 2)"),                       # an ADDRESS argument
    ("c = a(shl(b, 2));", "SHL(B, 2)"),                       # a subscript
    ("if shl(b, 4) > 10 then c = 1;", "SHL(B, 4)"),           # a relation
    ("c = shl(b, 4) / 2;", "SHL(B, 4)"),
    ("w = double(shl(b, 1));", "SHL(B, 1)"),
    ("c = high(shl(b, 1));", "SHL(B, 1)"),
    ("c = shr(shl(b, 1), 1);", "SHL(B, 1)"),
    ("s.n = shl(b, 3);", "SHL(B, 3)"),
    ("do case shl(b, 1); c = 1; c = 2; end;", "SHL(B, 1)"),
    ("w = (c := shl(b, 2));", "SHL(B, 2)"),
    ("w = shl(b and 1fh, 4);", "SHL(B AND 1fh, 4)"),
    ("w = shl(b, n);", "SHL(B, N)"),                          # a count not known
    ("w = shl(1, n);", "SHL(1, N)"),
    ("w = shl(b, 8) + c;", "SHL(B, 8)"),
    ("w = shl(0f0h, 4);", "SHL(0f0h, 4)"),
    ("k = 7; call g; w = shl(k, 5);", "SHL(K, 5)"),           # G sets K
    ("c = 0; do while c < n; c = c + 1; end; w = shl(c, 5);", "SHL(C, 5)"),
    ("if b > 8 then return; w = shl(b, 5);", "SHL(B, 5)"),
])
def test_a_shl_whose_lost_bits_are_used_is_warned_of(stmts, shl, opt, capsys):
    got = _warnings(stmts, opt, capsys)
    assert len(got) == 1 and f"warning: {shl}: SHL of a BYTE is a BYTE" in got[0], got


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("stmts", [
    "c = shl(b, 4);",                               # stored to a BYTE
    "c = shl(b, 4) + w;",                           # of which the low byte is kept
    "c = q(shl(b, 2));",                            # a BYTE argument
    "c = low(shl(b, 1));",
    "if shl(b, 4) then c = 1;",                     # a condition is its bit 0
    "w = shl(b, 4) and 0f0h;",                      # AND with a BYTE clears the rest
    "s.m = shl(b, 3);",
    "w = shl(b and 0fh, 4);",                       # cannot shift a set bit out
    "w = shl(3, 4);",
    "w = shl(0, n);",
    "w = shl(1, f);",                               # F is at most 7
    "c = f; w = shl(c, 5);",
    "w = shl(double(b), 4);",                       # SHL of an ADDRESS
    "w = b * 16;",
    "w = shr(b, 2);",                               # SHR of a BYTE: the same value
    "if b > 7 then return; w = shl(b, 5);",         # what B has passed
    "if b = 0 or b > 7 then return; w = shl(b, 5);",
    "c = 7; w = shl(c, 5);",                        # what C was assigned
    "c = b and 3; call h; w = shl(c, 6);",          # H sets neither C nor K
    "k = 7; call h; w = shl(k, 5);",
    "do c = 0 to 7; w = shl(c, 5); end;",
    "c = b; do case c; ; ; ; ; end; w = shl(c, 6);",    # C picked one of four
])
def test_a_shl_that_cannot_differ_is_not(stmts, opt, capsys):
    assert _warnings(stmts, opt, capsys) == []


def test_the_warning_says_what_changed(capsys):
    got = _warnings("w = shl(b, 4);", 2, capsys)
    assert got == [
        "T.PLM:10:5: warning: SHL(B, 4): SHL of a BYTE is a BYTE (Programming Manual "
        "9800268B, 11.1.4), and the bits shifted out of it, which this expression uses, "
        "are lost; SHL(DOUBLE(B), 4) keeps them. uplm80 before 0.4.3 shifted a BYTE in "
        "16 bits"]


def test_each_shl_is_warned_of_once(capsys):
    got = _warnings("w = shl(shl(b, 2), 2) + w; v = shl(b, 3) + shl(b, 3);", 0, capsys)
    assert [line.split(": warning: ")[1].split(":")[0] for line in got] == [
        "SHL(B, 2)", "SHL(SHL(B, 2), 2)", "SHL(B, 3)", "SHL(B, 3)"], got
