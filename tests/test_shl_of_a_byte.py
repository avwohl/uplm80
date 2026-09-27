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


# 0.4.3's final release check: the flags of an operation on a SHL of a BYTE
# that can lose bits are those of an 8-bit operation, where 0.4.2's were a
# 16-bit one's, and PLUS, MINUS, SCL, SCR, DEC, CARRY, ZERO, SIGN and PARITY
# read them - `b = (shl(k, 4) + 10h) plus 0' with k = 0FFH was 00 and is
# 01, `b = shl(k, 1); c = carry;' 00 and 0FFH; and a test that assigns the
# variable it bounds, or calls what assigns it, bounded it all the same.
@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("stmts, shl, reader", [
    ("c = (shl(b, 4) + 10h) plus 0;", "SHL(B, 4)", "PLUS"),
    ("c = shl(b, 4) plus 0;", "SHL(B, 4)", "PLUS"),
    ("c = shl(b, 1); c = carry;", "SHL(B, 1)", "CARRY"),
    ("c = shl(b, 4) - 10h; c = 0 minus 0;", "SHL(B, 4)", "MINUS"),
    ("c = shl(b, 4); if zero then c = 1;", "SHL(B, 4)", "ZERO"),
    ("c = shl(b, 1); if sign then c = 1;", "SHL(B, 1)", "SIGN"),
    ("c = shl(b, 1) + 3; if parity then c = 1;", "SHL(B, 1)", "PARITY"),
    ("c = shl(b, 1); c = scl(c, 1);", "SHL(B, 1)", "SCL"),
    ("c = dec(shl(b, 4) + 1);", "SHL(B, 4)", "DEC"),
    ("c = shl(b, 2) and 3; c = carry;", "SHL(B, 2)", "CARRY"),     # an AND of 8 bits
    ("c = ror(shl(b, 2), 1); c = carry;", "SHL(B, 2)", "CARRY"),
    # Across what ends: a branch, a pass of a loop, a procedure, a GOTO.
    ("if n > 3 then c = shl(b, 1); else c = 1; c = carry;", "SHL(B, 1)", "CARRY"),
    ("do while b; c = carry; c = shl(b, 1); end;", "SHL(B, 1)", "CARRY"),
    ("do c = 0 to n; c = shl(b, 1); end; c = carry;", "SHL(B, 1)", "CARRY"),
    ("sh: procedure byte; return shl(b, 1); end sh; c = sh plus 0;", "SHL(B, 1)", "PLUS"),
    ("c = shl(b, 1); goto l1; l1: c = carry;", "SHL(B, 1)", "CARRY"),
    # What sets no flag, or the carry alone, between (found checking
    # 0.4.4): NOT of a BYTE, `cpl', its minus, `cpl / inc a', and a product
    # of BYTEs, an ADDRESS, `add hl,hl'.
    ("c = shl(b, 1); c = not n; c = carry;", "SHL(B, 1)", "CARRY"),
    ("c = shl(b, 1) or n; c = not n; if sign then c = 1;", "SHL(B, 1)", "SIGN"),
    ("c = shl(b, 1); c = (not n) plus 0;", "SHL(B, 1)", "PLUS"),
    ("c = shl(b, 1); c = (-n) plus 0;", "SHL(B, 1)", "PLUS"),
    ("c = shl(b, 2) and 0f0h; w = n * 2; if zero then c = 1;", "SHL(B, 2)", "ZERO"),
    # A procedure that sets no flag returns with its caller's (G, and a
    # typed one), is entered with them, and one called through an address
    # too.
    ("c = shl(b, 1); call g; c = carry;", "SHL(B, 1)", "CARRY"),
    ("r3: procedure byte; return 3; end r3; c = shl(b, 1); c = r3; if sign then c = 1;",
     "SHL(B, 1)", "SIGN"),
    ("en: procedure; c = carry; end en; c = shl(b, 1); call en;", "SHL(B, 1)", "CARRY"),
    ("en: procedure; call g; end en; c = shl(b, 1); call en; c = carry;", "SHL(B, 1)",
     "CARRY"),
    ("v = .h; c = shl(b, 1); call v; c = carry;", "SHL(B, 1)", "CARRY"),
])
def test_what_reads_the_flags_of_a_shl_of_a_byte_is_warned_of(stmts, shl, reader, opt,
                                                              capsys):
    got = _warnings(stmts, opt, capsys)
    assert len(got) == 1 and f"warning: {shl}: SHL of a BYTE is a BYTE" in got[0] \
        and f"{reader} reads the flags" in got[0], got


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("stmts, shl", [
    ("c = 1; if c < 4 and (c := 200) > 0 then w = shl(c, 6);", "SHL(C, 6)"),
    ("c = 1; if (c := 200) > 0 and c < 4 then w = shl(c, 6);", "SHL(C, 6)"),
    ("gk: procedure byte; k = 200; return 1; end gk;\n"
     "k = 1; if k < 4 and gk > 0 then w = shl(k, 6);", "SHL(K, 6)"),
    ("c = 1; do while c < 4 and (c := 200) > 0; w = shl(c, 6); end;", "SHL(C, 6)"),
    # What a procedure calls through an address may assign what any
    # procedure whose address is taken assigns (found checking 0.4.4).
    ("fq: procedure byte; call v; return 1; end fq;\n"
     "v = .g; k = 1; if k < 4 and fq > 0 then w = shl(k, 6);", "SHL(K, 6)"),
    ("pr: procedure; call v; end pr; v = .g; k = 1; call pr; w = shl(k, 6);", "SHL(K, 6)"),
    ("declare sg structure (g address); sg.g = .g; k = 1; call sg.g; w = shl(k, 6);",
     "SHL(K, 6)"),
    # A name is what it names where it is used (found checking 0.4.4): a
    # parameter or a local called through is not the procedure H of the
    # same name, and a variable a DO block declares is not the K after it.
    ("r2: procedure (h); declare h address; call h; end r2;\n"
     "v = .g; k = 1; call r2(v); w = shl(k, 6);", "SHL(K, 6)"),
    ("r2: procedure (h); declare h address; call h; end r2;\n"
     "v = .g; k = 1; if k < 4 then call r2(v); w = shl(k, 6);", "SHL(K, 6)"),
    ("r3: procedure; declare h address; h = v; call h; end r3;\n"
     "v = .g; k = 1; call r3; w = shl(k, 6);", "SHL(K, 6)"),
    ("s2: procedure; do; declare k byte; k = 5; end; k = 200; end s2;\n"
     "k = 1; call s2; w = shl(k, 6);", "SHL(K, 6)"),
    ("v = .g; k = 1;\ndo while n > 0; w = shl(k, 6); n = n - 1;\n"
     "  do; declare h address; h = v; call h; end;\nend;", "SHL(K, 6)"),
])
def test_a_test_that_assigns_what_it_bounds_does_not_bound_it(stmts, shl, opt, capsys):
    got = _warnings(stmts, opt, capsys)
    assert len(got) == 1 and f"warning: {shl}: SHL of a BYTE is a BYTE" in got[0], got


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("stmts", [
    "c = shl(b and 1fh, 3) + 1; c = carry;",        # loses no bit: the limit
    "c = shl(b, 1); c = c + 1; c = carry;",         # the flags of what follows
    "if (b and 0e0h) <> 0 then return; c = shl(b, 3) + shl(b, 1); c = carry;",
    "if (b and 0e0h) = 0 then c = shl(b, 3) plus 0;",
    "c = 1; if c < 4 then w = shl(c, 6);",
    # A procedure's own flags, and those of the call before a procedure
    # that sets none.
    "fl: procedure; k = k + 3; end fl; c = shl(b, 1); call fl; c = carry;",
    "c = shl(b, 1); call g; c = c + 1; call g; c = carry;",
    "pr: procedure; call v; end pr; v = .h; k = 1; call pr; w = shl(k, 6);",
])
def test_what_reads_other_flags_is_not(stmts, opt, capsys):
    assert _warnings(stmts, opt, capsys) == []


def test_the_flags_warning_says_what_reads_them(capsys):
    got = _warnings("c = shl(b, 1); c = carry;", 2, capsys)
    assert got == [
        "T.PLM:10:5: warning: SHL(B, 1): SHL of a BYTE is a BYTE (Programming Manual "
        "9800268B, 11.1.4), and CARRY reads the flags of an operation of eight bits on it; "
        "SHL(DOUBLE(B), 1) is shifted in 16 bits, as uplm80 before 0.4.3 shifted a BYTE"]


def test_a_variable_an_interrupt_procedure_assigns_has_no_bound(capsys):
    """An INTERRUPT procedure may run between any two statements: what it
    assigns, or what it calls does, through an address too (found
    checking 0.4.4), can have any value anywhere."""
    capsys.readouterr()
    src = ("t: do;\ndeclare (k, j, n, m) byte, (w, q) address;\n"
           "set: procedure; j = 3; end set;\n"
           "setn: procedure; n = 3; end setn;\n"
           "i: procedure interrupt 1; k = 3; call set; call q; end i;\n"
           "k = 1; w = shl(k, 6);\nj = 1; w = shl(j, 6);\nq = .setn; n = 1; w = shl(n, 6);\n"
           "m = 1; w = shl(m, 6);\nend t;\n")
    assert Compiler(opt_level=2).compile(src, "T.PLM") is not None
    got = [line for line in capsys.readouterr().err.splitlines() if "SHL of a BYTE" in line]
    assert [line.split(": warning: ")[1].split(":")[0] for line in got] == [
        "SHL(K, 6)", "SHL(J, 6)", "SHL(N, 6)"], got


def test_what_an_interrupt_procedure_declares_in_a_do_block_is_its_own(capsys):
    """An INTERRUPT procedure that declares K and J in a DO block, and
    assigns the module's K after it, may change the module's K between any
    two statements (found checking 0.4.4; 0.4.3 the same): it has no
    bound.  The module's J it does not assign."""
    capsys.readouterr()
    src = ("t: do;\ndeclare (k, j) byte, w address;\n"
           "i: procedure interrupt 1;\n  do; declare (k, j) byte; k = 5; j = 5; end;\n"
           "  k = 3;\nend i;\n"
           "k = 1; w = shl(k, 6);\nj = 1; w = shl(j, 6);\nend t;\n")
    assert Compiler(opt_level=2).compile(src, "T.PLM") is not None
    got = [line for line in capsys.readouterr().err.splitlines() if "SHL of a BYTE" in line]
    assert [line.split(": warning: ")[1].split(":")[0] for line in got] == ["SHL(K, 6)"], got


def test_a_parameter_called_through_does_not_reset_the_system(capsys):
    """R4's last statement calls through its parameter TERMINATE, not the
    procedure TERMINATE that calls MON1 with the function 0: R4 returns
    (found checking 0.4.4), and K, which SETK assigns, can be anything."""
    capsys.readouterr()
    src = ("t: do;\ndeclare (b, k) byte, (w, v) address;\n"
           "mon1: procedure (f, a) external; declare f byte, a address; end mon1;\n"
           "terminate: procedure; call mon1(0, 0); end terminate;\n"
           "setk: procedure; k = 200; end setk;\n"
           "r4: procedure (terminate); declare terminate address; call terminate; end r4;\n"
           "v = .setk; k = 1; b = input(1); if b > 3 then call r4(v); w = shl(k, 6);\n"
           "end t;\n")
    assert Compiler(opt_level=2).compile(src, "T.PLM") is not None
    got = [line for line in capsys.readouterr().err.splitlines() if "SHL of a BYTE" in line]
    assert [line.split(": warning: ")[1].split(":")[0] for line in got] == ["SHL(K, 6)"], got


@pytest.mark.parametrize("function, warned", [
    (0, ["SHL(B, 1)"]), (1, ["SHL(B, 3)", "SHL(B, 1)"])])
def test_a_procedure_that_resets_the_system_does_not_return(function, warned, capsys):
    """MP/M II's SHOW, MSCHD and TOD read a number with `if (b and 0e0h) <>
    0 then call terminate; b = shl(b, 3) + shl(b, 1); if carry then ...':
    TERMINATE calls MON1 with the function 0, system reset, and does not
    return, so b is below 32 at the SHLs, which lose nothing.  With another
    function, it returns, and b can be anything."""
    capsys.readouterr()
    src = ("t: do;\ndeclare (b, c) byte, w address;\n"
           "mon1: procedure (f, a) external; declare f byte, a address; end mon1;\n"
           f"terminate: procedure; call mon1({function}, 0); end terminate;\n"
           "stop: procedure; call terminate; end stop;\n"
           "b = 0; do while b < 100;\n"
           "  if (b and 0e0h) <> 0 then call terminate;\n"
           "  b = shl(b, 3) + shl(b, 1); if carry then call stop;\n"
           "  b = b + input(1); w = shl(b, 1);\nend;\nend t;\n")
    assert Compiler(opt_level=2).compile(src, "T.PLM") is not None
    got = [line for line in capsys.readouterr().err.splitlines() if "SHL of a BYTE" in line]
    assert [line.split(": warning: ")[1].split(":")[0] for line in got] == warned, got


@pytest.mark.parametrize("end", ["out: end term;", "out:end term;"])
def test_a_procedure_whose_end_a_goto_reaches_returns(end, capsys):
    """A label on a procedure's END, which a GOTO reaches past its last
    statement, a call of MON1 with the function 0: it returns (found
    checking 0.4.4), and k, which it does not bound, can be anything."""
    capsys.readouterr()
    src = ("t: do;\ndeclare (b, k) byte, w address;\n"
           "mon1: procedure (f, a) external; declare f byte, a address; end mon1;\n"
           f"term: procedure; if b = 0 then goto out; call mon1(0, 0); {end}\n"
           "b = 0; k = 200; if k > 3 then call term; w = shl(k, 6);\nend t;\n")
    assert Compiler(opt_level=2).compile(src, "T.PLM") is not None
    got = [line for line in capsys.readouterr().err.splitlines() if "SHL of a BYTE" in line]
    assert [line.split(": warning: ")[1].split(":")[0] for line in got] == ["SHL(K, 6)"], got


@pytest.mark.parametrize("opt", LEVELS)
def test_a_test_of_equality_still_bounds(opt, capsys):
    """`if b = 0' holding leaves b 0, as before the mask tests."""
    assert _warnings("if b = 0 then w = shl(b, 5);", opt, capsys) == []
    assert _warnings("if not (b <> 3) then w = shl(b, 6);", opt, capsys) == []
