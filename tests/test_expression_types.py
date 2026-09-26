"""Expressions keep the value and the type PL/M-80 gives them, at every -O.

Intel's PL/M-80 Programming Manual types every operand and operator: a
constant up to 255 is a BYTE (4.1.1); `+ - AND OR XOR` of two BYTEs, and
unary `-` and `NOT` of one, are 8-bit and wrap (4.2, 4.3); `* / MOD` are
ADDRESS (4.2.3); a relation is the BYTE 0FFH or 0 (4.4). Every rewrite the
compiler makes -- folding, propagation, strength reduction, algebra,
inlining -- has to leave the value and the type the unoptimized program
computes, or the arithmetic around the rewritten operand changes width.

Each case here printed something else, or did not compile, at some level
before its fix -- except test_restricted_expressions_are_plain_numbers,
which holds what the fixes must not change. The last test runs random programs against a model of the
rules (tests/plm_difftest.py); scripts/difftest.py runs many more of them.
"""

import pytest

from tests._toolchain import tools_missing
from tests.plm_difftest import build_and_run, generate
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


# ---- the reported defects --------------------------------------------------

def test_a_folded_quotient_or_remainder_is_an_address():
    """`8 MOD 0FFH' is the ADDRESS 8, so adding the BYTE 0FFH carries into
    the high byte; folded to a BYTE literal it wrapped to 7 (defect 1)."""
    _check("""
declare (b0, b1) byte, (w0, w1, r) address;
run: procedure;
  declare bd(*) byte data (0ffh, 0ffh);
  b0 = bd(0); b1 = bd(1);
  r = (8 mod 0ffh) + b0; call ph(r);
  r = (8 mod 0) + b0; call ph(r);
  w0 = 8; w1 = 0; r = (w0 mod w1) + b1; call ph(r);
  r = (200 / 2) + b0; call ph(r);
end run;
call run;
""", [0x107, 0x107, 0x107, 0x163])


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("src", [
    "t: do; declare rb byte; rb = (7 mod 0) > 1000H; end t;",
    "t: do; declare (w, z) address, rb byte; w = 7; z = 0; rb = (w mod z) > 1000H; end t;",
    "t: do; declare (w, z) address, rb byte; w = 7; z = 3; rb = (w mod z) > 1000H; end t;",
    # A constant the optimizer derived is not one the programmer compared.
    "t: do; declare w address, b byte; w = 5000; if b < w then b = 1; end t;",
])
def test_folding_does_not_make_a_comparison_impossible(src, opt, capsys):
    """A remainder is an ADDRESS, so comparing it with 1000H is not the
    `BYTE > 4096' uplm80 warns of - and, before 0.3.7, rejected (defect 2)."""
    _asm(src, opt)
    assert "comparison BYTE" not in capsys.readouterr().err


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("src, text", [
    ("if b < 256 then b = 1;", "comparison BYTE < 256 is always true"),
    ("if b <> 257 then b = 1;", "comparison BYTE <> 257 is always true"),
    ("if b = 300 then b = 1;", "comparison BYTE = 300 is always false"),
    ("if (b + 1) < 256 then b = 1;", "comparison BYTE < 256 is always true"),
    # A constant on the left is looked at too, whether or not the optimizer
    # moves it to the right.
    ("if 300 > b then b = 1;", "comparison BYTE < 300 is always true"),
    ("if 300 = b then b = 1;", "comparison BYTE = 300 is always false"),
])
def test_a_byte_against_a_constant_above_255_is_a_warning(src, text, opt, capsys):
    """The manual (4.4) compares a BYTE with an ADDRESS as unsigned 16-bit
    numbers, and DRI's compiler accepts `IF b < 256'; uplm80 stopped with an
    error.  The comparison is compiled, and what it is worth a warning for
    is still said."""
    _asm("t: do; declare b byte; " + src + " end t;", opt)
    assert text in capsys.readouterr().err


def test_a_byte_against_a_constant_above_255_compares_as_an_address():
    """Compiled, it has to come out as the manual has it: the BYTE is
    zero-extended, so it is below every such constant - and `b + 1' is a
    BYTE, which wraps, so it is too when b is 0FFH."""
    _check("""
declare (b, r) byte;
run: procedure;
  declare bd (*) byte data (0, 0ffh);
  declare i byte;
  do i = 0 to 1;
    b = bd(i);
    r = b < 256; call ph(r);
    r = b <> 257; call ph(r);
    r = b = 300; call ph(r);
    r = (b + 1) < 256; call ph(r);
    r = 300 > b; call ph(r);
    r = b >= 0ffffh; call ph(r);
    if b < 256 then call ph(1); else call ph(0);
    if (b + 1) >= 256 then call ph(1); else call ph(0);
    if not (b = 1000h) then call ph(1); else call ph(0);
  end;
end run;
call run;
""", [0xFF, 0xFF, 0, 0xFF, 0xFF, 0, 1, 0, 1] * 2)


def test_not_and_minus_of_a_byte_are_bytes():
    """NOT 7 is the BYTE 0F8H and -1 the BYTE 0FFH (4.2.2, 4.3); both were
    evaluated in sixteen bits even at -O0 (defect 3)."""
    _check("""
declare (w2, r) address;
run: procedure;
  w2 = 1000;
  r = (not 7) mod w2; call ph(r);
  r = (3 - 5) mod w2; call ph(r);
  r = (-1) mod w2; call ph(r);
  r = -1; call ph(r);
  r = not 0; call ph(r);
  r = -(-(1)) + 0ffh; call ph(r);
end run;
call run;
""", [0xF8, 0xFE, 0xFF, 0xFF, 0xFF, 0])


def test_a_byte_loop_to_255_runs_256_times():
    """DRI's code for `DO j = common$base TO 0FFH' (MPMLDR/GENSYS.PLM:652,
    GENSYS.COM 106BH) tests the limit, then `INR A / JNZ': the index wraps
    to 0 and the loop ends. uplm80 read the index 0 as the wrap and ran the
    loop no times from 0 (defect 4)."""
    _check("""
declare (lo, lim) byte, n address;
run: procedure;
  n = 0; do lo = 0 to 255; n = n + 1; end; call ph(n); call ph(lo);
  n = 0; do lo = 250 to 0ffh; n = n + lo; end; call ph(n); call ph(lo);
  lim = 255;
  n = 0; do lo = 0 to lim; n = n + 1; end; call ph(n);
  n = 0; do lo = 0 to lim; n = n + lo; end; call ph(n);
end run;
call run;
""", [0x100, 0, 0x5EB, 0, 0x100, 0x7F80])


def test_loops_stop_where_the_increment_carries_out():
    """The same rule for an ADDRESS index, and for a BY step: DRI's LOAD.COM
    adds the step with `DAD D' and leaves on the carry."""
    _check("""
declare (b, s) byte, (w, lw, n) address;
run: procedure;
  n = 0; do w = 0fff0h to 0ffffh; n = n + 1; end; call ph(n); call ph(w);
  lw = 0ffffh;
  n = 0; do w = 0fffeh to lw; n = n + 1; end; call ph(n); call ph(w);
  n = 0; do w = 0ffd6h to 0fffeh by 3; n = n + 1; end; call ph(n); call ph(w);
  n = 0; do b = 250 to 254 by 10; n = n + 1; end; call ph(n); call ph(b);
  s = 10;
  n = 0; do b = 240 to 250 by s; n = n + 1; end; call ph(n); call ph(b);
end run;
call run;
""", [16, 0, 2, 0, 14, 0, 1, 4, 2, 4])


def test_a_loop_converts_its_limit_and_step_to_the_index_type():
    """`DO b = 0 TO 300' runs to 44 (5.1.4), and BY -1 is BY 0FFH: PL/M-80
    cannot count down."""
    _check("""
declare b byte, (w, n) address;
run: procedure;
  w = 300;
  n = 0; do b = 0 to w; n = n + 1; end; call ph(n);
  n = 0; do b = 5 to 0 by -1; n = n + 1; end; call ph(n); call ph(b);
  n = 0; do b = 0 to 0 by -1; n = n + 1; end; call ph(n); call ph(b);
end run;
call run;
""", [45, 0, 5, 1, 0xFF])


def test_a_counted_loop_keeps_the_index():
    """A DJNZ loop must leave the index as the loop would, and cannot be
    used when the body assigns it: ED.PLM's FILLSOURCE and PIP.PLM's fill
    loop end early with `I = N', which the DJNZ form ignored."""
    _check("""
declare (i, k, m) byte, n address;
run: procedure;
  do i = 0 to 9; n = 1; end; call ph(i);
  n = 0; m = 9;
  do i = 0 to m; n = n + 1; if n = 3 then i = m; end; call ph(n);
  k = 255; n = 0;
  do i = 0 to k; n = n + 1; end; call ph(n); call ph(i);
  k = 4;
  do i = 0 to k; n = 1; end; call ph(i);
end run;
call run;
""", [10, 3, 0x100, 0, 5])


def test_a_call_ends_what_is_known_about_a_global():
    """`rw = f' calls f, which may change cnt, so cnt is not the 0 assigned
    before it (defect 5); nor after a CALL, inlined or not."""
    _check("""
declare (cnt, rw) address;
f: procedure address; cnt = cnt + 1; return 5; end f;
g: procedure; cnt = cnt + 1; end g;
run: procedure;
  cnt = 0; rw = f; call ph(cnt); call ph(rw);
  cnt = 0; call g; call ph(cnt);
  cnt = 0; call g; call g; call ph(cnt);
end run;
call run;
""", [1, 5, 1, 2])


def test_times_two_is_an_address():
    """BYTE `x * 2' is an ADDRESS product; it was rewritten `x + x', a BYTE
    add that wraps -- as MPMLDR/GENSYS.PLM's `(mem$top-cur$base+1)*2 + 1'
    was -- and that evaluates x twice (defect 6)."""
    _check("""
declare x byte, (r, cnt) address;
f: procedure byte; cnt = cnt + 1; return 200; end f;
run: procedure;
  declare bd(*) byte data (200);
  x = bd(0);
  r = x * 2; call ph(r);
  r = 2 * x; call ph(r);
  cnt = 0; r = f * 2; call ph(r); call ph(cnt);
  r = x * 1 + 0ffh; call ph(r);
  r = x * 4; call ph(r);
end run;
call run;
""", [400, 400, 400, 1, 200 + 0xFF, 800])


# ---- what the audit of the optimizer and the typing found ------------------

def test_identities_keep_the_width():
    """`b AND 0FFFFH' is the ADDRESS b, and `b XOR 0FFFFH' an ADDRESS
    NOT; `b + DOUBLE(0)' is ADDRESS; `w - w' the ADDRESS 0."""
    _check("""
declare (b) byte, (w, r) address;
run: procedure;
  declare bd(*) byte data (0ffh);
  b = bd(0); w = 1234h;
  r = (b and 0ffffh) + b; call ph(r);
  r = (b xor 0ffffh); call ph(r);
  r = (w - w) - 1; call ph(r);
  r = (b - b) - 1; call ph(r);
  r = (w * 0) - 1; call ph(r);
end run;
call run;
""", [0x1FE, 0xFF00, 0xFFFF, 0xFF, 0xFFFF])


def test_reassociation_keeps_the_width():
    """`(x + 200) + 100' of a BYTE wraps twice at eight bits; `(b + 100)
    + 300' wraps once and then adds in sixteen."""
    _check("""
declare (b) byte, (r) address;
run: procedure;
  declare bd(*) byte data (200);
  b = bd(0);
  r = (b + 200) + 100; call ph(r);
  r = (b + 100) + 300; call ph(r);
  r = (b - 'A') + 'B'; call ph(r);
end run;
call run;
""", [(200 + 300) & 0xFF, ((200 + 100) & 0xFF) + 300, 201])


def test_propagated_constants_and_copies_keep_the_width():
    """At -O3, `w = 8' makes w the ADDRESS 8, and `w = b' makes w the
    zero-extended b: neither is replaced by something of the other width."""
    _check("""
declare (b, c) byte, (w, v, r) address;
run: procedure;
  declare bd(*) byte data (0ffh);
  b = bd(0);
  w = 8; r = w + b; call ph(r);
  w = b; r = w + 0ffh; call ph(r);
  c = 300; r = c + 0ffh; call ph(r);
  v = -1; r = v + 1; call ph(r);
end run;
call run;
""", [0x107, 0x1FE, (44 + 0xFF) & 0xFF, 0x100])


def test_embedded_assignment_is_its_right_half():
    """"The value of the embedded assignment is the same as that of its
    right half" (4.6.3): `(b := w)' is all of w."""
    _check("""
declare b byte, (w, r) address;
run: procedure;
  w = 1234h;
  r = (b := w) + 0; call ph(r); call ph(b);
  r = (b := 0ffh) + 1; call ph(r);
end run;
call run;
""", [0x1234, 0x34, 0])


def test_builtins_have_their_types():
    """LENGTH and LAST are BYTE when they fit (11.1.2); CARRY is 0FFH when
    set (12.5); an element of an array member is the member's type."""
    _check("""
declare b byte, r address;
declare s structure (m(4) byte);
declare a(10) byte, big(300) byte;
run: procedure;
  r = last(a) + 0ffh; call ph(r);
  r = last(big) + 0ffh; call ph(r);
  s.m(1) = 0ffh; r = s.m(1) + 1; call ph(r);
  b = 0ffh; b = b + 1; r = carry; call ph(r);
end run;
call run;
""", [8, 299 + 0xFF, 0, 0xFF])


def test_byte_plus_and_minus_are_bytes():
    """PLUS and MINUS "perform similarly to + and -" (12.2): of two BYTEs,
    a BYTE, with the carry of the operation before them."""
    _check("""
declare (b1, b2) byte, r address;
run: procedure;
  declare bd(*) byte data (0ffh, 1);
  b1 = bd(0); b2 = bd(1);
  r = (b1 + b2) plus 0ffh; call ph(r);
  r = (b2 - b1) minus 0; call ph(r);
end run;
call run;
""", [0, 1])


def test_two_character_strings_are_address_constants():
    """'AB' is 4142H (4.1.1), not the address of a string."""
    _check("""
declare r address;
run: procedure;
  r = 'AB'; call ph(r);
  r = 'AB' + 1; call ph(r);
end run;
call run;
""", [0x4142, 0x4143])


def test_shift_and_rotate_counts_are_unsigned_bytes():
    """The count loops stopped at once for a count of 129 or more."""
    _check("""
declare (b, n) byte, (w, r) address;
run: procedure;
  b = 81h; w = 1; n = 201;
  r = rol(b, n); call ph(r);
  n = 200; r = shl(w, n); call ph(r);
  n = 1; r = scl(w, n) and 0fffeh; call ph(r);
end run;
call run;
""", [0x03, 0, 2])


def test_restricted_expressions_are_plain_numbers():
    """A DATA or INITIAL value is a restricted expression (6.2.8), folded as
    a number and then fitted to its slot, not by the BYTE rules."""
    _check("""
run: procedure;
  declare d(*) address data (200 + 100, 0 - 1);
  call ph(d(0)); call ph(d(1));
end run;
call run;
""", [300, 0xFFFF])


def test_a_constant_subscript_is_typed():
    """MEMORY(-1) is MEMORY(0FFH), -1 being a BYTE; it was stored at
    __END__ - 1."""
    _check("""
declare x byte;
run: procedure;
  memory(255) = 66h;
  x = memory(-1); call ph(x);
  memory(-1) = 55h; call ph(memory(255));
  memory(not 0) = 44h; call ph(memory(0ffh));
end run;
call run;
""", [0x66, 0x55, 0x44])


def test_an_inlined_body_means_what_it_meant():
    """-O3 inlines a small procedure only where its names mean the same
    thing, and not one that returns from the middle."""
    _check("""
declare (w, x) address;
bump: procedure; w = w + 3; end bump;
early: procedure; if x then return; w = w + 100h; end early;
run: procedure;
  declare w address;
  w = 1; call bump; call ph(w);
end run;
run2: procedure;
  x = 1; call early; call ph(w);
end run2;
w = 10h; call run; call ph(w); call run2;
""", [1, 0x13, 0x13])


def test_what_a_loop_body_sets_is_not_known_after_it():
    """The body may not have run, the unrolled loop leaves the index as the
    loop does, and an empty loop still assigns its start."""
    _check("""
declare (x, n, i) address, b byte;
run: procedure;
  x = 1; n = 0;
  do i = 1 to n; x = 5; end; call ph(x);
  do i = 1 to 2; x = x + 1; end; call ph(i); call ph(x);
  do b = 5 to 3; x = 9; end; call ph(b);
end run;
call run;
""", [1, 3, 3, 5])


def test_a_dead_store_is_dropped_only_when_nothing_else_goes():
    """`x = f; x = 5' still calls f; `x = 1; x = bx' reads the 1 through bx."""
    _check("""
declare (x, cnt, p) address;
declare bx based p address;
f: procedure address; cnt = cnt + 1; return 7; end f;
run: procedure;
  cnt = 0; x = f; x = 5; call ph(cnt);
  p = .x; x = 1; x = bx + 1; call ph(x);
end run;
call run;
""", [1, 2])


def test_a_location_is_not_a_value():
    """`.x' after `x = 5' named address 5 at -O3."""
    _check("""
declare (x, p) address;
declare y based p address;
run: procedure;
  x = 5; p = .x; y = 7; call ph(x);
end run;
call run;
""", [7])


def test_a_store_through_a_pointer_ends_what_is_known():
    """At -O3, `b = 5' made b the constant 5 until the next assignment to
    b -- but `bb = 9' through a pointer to b is one."""
    _check("""
declare (b, c) byte, p address;
declare bb based p byte;
run: procedure;
  b = 5; c = 7;
  p = .b; bb = 9; call ph(b); call ph(b + c);
end run;
call run;
""", [9, 16])


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



def test_low_reads_its_own_operand():
    """LOW((b := w)) knows A already holds L; the flag saying so outlived the
    embedded assignment, so `(b := w) + LOW(LAST(a))' added L of w, not 7."""
    _check("""
declare b byte, (w, r) address;
declare ab(8) byte;
run: procedure;
  w = 1234h;
  r = (b := w) + low(last(ab)); call ph(r); call ph(b);
  r = low((b := w)) + 1; call ph(r);
end run;
call run;
""", [0x123B, 0x34, 0x35])

# ---- what review of the typing found ----------------------------------------

def test_length_and_last_as_a_byte_subscript():
    """LENGTH and LAST of an array of up to 255 elements are BYTEs, and a
    BYTE subscript is taken from A -- where `ld hl,7' had left nothing: with
    ab(k) = k, `ab(LAST(sa))' read ab(0FFH), the store went elsewhere, and
    `.ab(LENGTH(sa))' was off by 18H."""
    _check("""
declare ab(256) byte, sa(8) byte, (k, p, q) address;
run: procedure;
  do k = 0 to 255; ab(k) = k; end;
  call ph(ab(last(sa))); call ph(ab(length(sa)));
  ab(last(sa)) = 77h; call ph(ab(7));
  p = .ab(length(sa)); q = .ab(8); call ph(p - q);
  sa(last(sa)) = 5; call ph(sa(7));
  k = 1; call ph(ab(last(sa) - k));
end run;
call run;
""", [7, 8, 0x77, 0, 5, 6])


def test_an_element_of_untyped_data_is_a_byte():
    """`DECLARE hex DATA ('0123')' (as 80un's bas.plm has it) has BYTE
    elements, which _gen_subscript loads into A; typed ADDRESS, `hex(i) +
    0FFH' added A to whatever HL held."""
    _check("""
declare i byte, r address;
run: procedure;
  declare hx data ('0123');
  declare hy byte data ('0123');
  i = 1;
  r = hx(i) + 0ffh; call ph(r);
  r = hy(i) + 0ffh; call ph(r);
  r = hx(i) + 100h; call ph(r);
end run;
call run;
""", [0x30, 0x30, 0x131])


def test_a_constant_moved_right_of_a_relation_is_not_checked():
    """-O3 puts a relation's constant on the right. Marked as derived when it
    was a plain number, but not when it was `'AB'' or a sum left unfolded
    because the procedure uses PLUS; the impossible-comparison check then
    rejected, at -O3 alone, what compiles at every other level."""
    _check("""
declare b byte, w address;
run: procedure;
  b = 5;
  w = 'AB' <> b; call ph(w);
  w = 0ff01h + 170 <> b; call ph(w);
  w = 'AB' = b; call ph(w);
  b = b minus 1;
end run;
call run;
""", [0xFF, 0xFF, 0])


def test_a_relation_of_addresses_as_a_subscript():
    """A 16-bit relation in an ADDRESS array's subscript let go of DE --
    popping the subscript's spill back into it -- before comparing with
    it, so `aw(w = 5)' compared w with a stale DE and read aw(0)."""
    _check("""
declare aw(256) address, (w, k) address;
run: procedure;
  do k = 0 to 255; aw(k) = k; end;
  w = 5;
  call ph(aw(w = 5)); call ph(aw(w > 4)); call ph(aw(w < 4));
  aw(w = 5) = 1234h; call ph(aw(255));
end run;
call run;
""", [0xFF, 0xFF, 0, 0x1234])


def test_a_relation_in_a_structure_or_remainder_subscript():
    """The same stale DE in a structure array's subscript, and in a BYTE
    array's whose subscript goes through MOD: the verification's
    f1_rel_in_wide_subscript, wrong at -O0 to -O2 in 0.3.6 and on both
    branches' heads before the fix above."""
    _check("""
declare (x, y, w) address;
declare aw(2) address initial (1111h, 2222h);
declare ib(6) byte initial (10, 11, 12, 13, 14, 15);
declare st(2) structure (a byte, b address) initial (1, 1111h, 2, 2222h);
run: procedure;
  x = 1000; y = 2000;
  w = aw((x < y) and 1); call ph(w);
  w = aw((y < x) and 1); call ph(w);
  w = st((x < y) and 1).b; call ph(w);
  st((x > y) and 1).b = 7; call ph(st(0).b);
  call ph(ib((x > y) mod 6));
end run;
call run;
""", [0x2222, 0x1111, 0x2222, 7, 10])


def test_low_of_arithmetic_on_an_embedded_assignment():
    """The embedded assignment to a BYTE sets a flag saying A holds L, for
    LOW((b := w)); arithmetic on it that never goes back through _gen_expr
    (`ld de,5 / add hl,de') left the flag set, and LOW took L of w."""
    _check("""
declare b byte, (w, v, r) address;
run: procedure;
  w = 1234h; v = 0101h;
  r = low((b := w) + 5); call ph(r);
  r = low((b := w) + v); call ph(r);
  r = low((b := w) - 1); call ph(r);
  r = low(shl((b := w), 1)); call ph(r);
  r = low(not (b := w)); call ph(r);
  r = low(-(b := w)); call ph(r);
  r = low((b := w) * 2); call ph(r);
  r = low((b := w) / 2); call ph(r);
  r = low((b := w)); call ph(r); call ph(b);
end run;
call run;
""", [0x39, 0x35, 0x33, 0x68, 0xCB, 0xCC, 0x68, 0x1A, 0x34, 0x34])


def test_plus_and_minus_of_a_constant_keep_the_carry():
    """BYTE `c PLUS x' loaded c into A first; `ld a,0' is `xor a' after
    the peephole, which clears the carry PLUS and MINUS read."""
    _check("""
declare (b1, b2, b3, z, one) byte, ab(4) byte;
run: procedure;
  b1 = 0ffh; z = 0; one = 1; ab(1) = 7;
  b2 = b1 + 1; b3 = 0 plus z; call ph(b3);
  b2 = z - 1; b3 = 0 minus z; call ph(b3);
  b2 = b1 + 1; b3 = 5 plus z; call ph(b3);
  b2 = z - 1; b3 = 5 minus one; call ph(b3);
  b2 = b1 + 1; b3 = 0 plus 0; call ph(b3);
  b2 = z - 1; b3 = 0 minus 0; call ph(b3);
  b2 = b1 + 1; b3 = 0 plus ab(one); call ph(b3);
  b2 = z - 1; b3 = 0 minus ab(one); call ph(b3);
  b2 = z + 1; b3 = 0 minus one; call ph(b3);
  b2 = b1 + 1; b3 = (0 minus z) minus z; call ph(b3);
end run;
call run;
""", [1, 0xFF, 6, 3, 1, 0xFF, 8, 0xF8, 0xFF, 0xFE])


def test_plus_minus_scl_and_scr_after_four_read_its_carry():
    """PLUS, MINUS, SCL and SCR read the carry of the operation before them
    (12.2, 12.3).  An ADDRESS plus or minus 1 to 4 was `inc hl' or `dec
    hl', which set none, so they read whatever an earlier instruction left:
    `(w + 4) PLUS z' with w = 0FFFEH gave 2.  Intel's PL/M-80 V3.1 steps an
    ADDRESS by 1 to 3 with INX and DCX, which leave the carry alone too, but
    adds or subtracts 4 and more with DAD, or SUB and SBB, which set it; so
    does uplm80 now where the carry is read, and the program compiled by
    V3.1 prints what is expected here.  (After 1 to 3 the carry is stale in
    both compilers' code; the manual says not to rely on it, 12.1.)  A BYTE
    plus or minus a constant is `add' or `sub', which set it."""
    _check("""
declare (w, z, r) address, (b, c) byte;
run: procedure;
  z = 0;
  w = 0fffeh; r = (w + 4) plus z; call ph(r);
  w = 0; r = (w + 4) plus z; call ph(r);
  w = 2; r = (w - 4) minus z; call ph(r);
  w = 6; r = (w - 4) minus z; call ph(r);
  w = 0fffch; r = scl(w + 4, 1); call ph(r);
  w = 0; r = scl(w + 4, 1); call ph(r);
  w = 2; r = scr(w - 4, 1); call ph(r);
  w = 6; r = scr(w - 4, 1); call ph(r);
  w = 0fffeh; r = z plus (w + 4); call ph(r);
  b = 0feh; c = (b + 3) plus 0; call ph(c);
  b = 2; c = (b - 3) minus 0; call ph(c);
end run;
call run;
""", [3, 4, 0xFFFD, 2, 1, 8, 0xFFFF, 1, 3, 2, 0xFE])


@pytest.mark.parametrize("opt", LEVELS)
def test_one_to_three_is_still_inc_hl_where_the_carry_is_read(opt):
    """As Intel's INX: `(w + 3) PLUS z' is three `inc hl', `(w + 4) PLUS z'
    an `add hl,de', and `w + 4' where nothing reads the carry four `inc
    hl' still."""
    def body(expr):
        return _asm(f"t: do;\ndeclare (w, z, r) address;\nr = {expr};\nend t;\n", opt)
    assert body("(w + 3) plus z").count("inc\thl") == 3
    four = body("(w + 4) plus z")
    assert "inc\thl" not in four and "add\thl,de" in four, four
    assert body("(w - 3) minus z").count("dec\thl") == 3
    assert "dec\thl" not in body("(w - 4) minus z")
    assert body("w + 4").count("inc\thl") == 4


def test_an_embedded_assignment_through_a_pointer_keeps_its_value():
    """`(x := w)' with x a BASED BYTE is w, all of it; the store loads the
    pointer into HL, where the value was, and nothing kept it."""
    _check("""
declare (w, r, p) address, x based p byte, buf(4) byte;
rf: procedure (v) address reentrant;
  declare v address, y byte;
  return (y := v) + 1 + y;
end rf;
run: procedure;
  p = .buf; w = 1234h;
  r = (x := w) + 1; call ph(r); call ph(buf(0));
  r = low((x := w)); call ph(r);
  if (x := w) = 1234h then call ph(1); else call ph(2);
  call ph(rf(1234h));
end run;
call run;
""", [0x1235, 0x34, 0x34, 1, 0x1269])


def test_size_of_a_variable_whose_value_is_known():
    """SIZE's operand names a variable; -O3 propagated `b0 = 5' into it,
    and SIZE(5) does not compile."""
    _check("""
declare (b0, r) byte, (w0) address, arr(10) address;
run: procedure;
  b0 = 5; w0 = 300;
  r = size(b0) + size(w0); call ph(r);
  r = size(arr) + length(arr) + last(arr); call ph(r);
end run;
call run;
""", [3, 39])


def test_low_and_high_of_size():
    """LOW(SIZE(aw)) was `ld hl,16 / ld a,l'; stored to an ADDRESS that is
    `... / ld h,0 / ld (w4),hl', and upeepz80 0.2.4, taking HL for dead
    since the store is not among the reads it knows, made it `ld a,16' --
    w4 got whatever L held (seed 80071 of the differential test)."""
    _check("""
declare (w4) address, b4 byte;
run: procedure;
  declare aw(*) address data (0ch, 07h, 025h, 0100h, 056h, 08001h, 08h, 044ddh);
  declare big(300) byte;
  w4 = low(size(aw));
  call ph(b4); call ph(w4);
  w4 = high(size(big)); call ph(w4);
  w4 = low(length(big)) + high(last(big)); call ph(w4);
end run;
call run;
""", [0, 0x10, 1, 0x2D])


# LENGTH, LAST and SIZE of a qualified reference (11.1.2): a structure's
# member, `st.z'; a member of an element of an array of structures,
# `sa(1).z', or of the array, `sa.z' (partially qualified); an element,
# `sa(2)', `ab(2)', `sa(1).z(1)'; in a procedure, a REENTRANT one, and of
# BASED structures. The subscripts are not evaluated. Intel's PL/M-80 V3.1
# compiles the program to print what is expected (tests/test_intel_oracle.py
# checks it again where Intel's tools are).
QUALIFIED_SIZES = """
declare st structure (x byte, y address, z(4) byte);
declare sa(3) structure (x byte, y address, z(2) byte, w(5) address);
declare ab(6) byte, aw(7) address, (i, n) byte, (p, w) address;
declare bs based p structure (k(9) byte, q address);
declare ba based p (4) structure (k(3) address, m byte);
lp: procedure (a) address;
  declare a byte;
  declare ls structure (u(3) byte, v(6) address);
  declare la(2) structure (u(7) byte);
  n = 0;
  do i = 0 to last(ls.v);
    ls.v(i) = i; n = n + 1;
  end;
  call ph(n);
  call ph(length(la.u) + size(la(1).u) + size(ls.v(2)));
  return size(ls) + size(la) + a;
end lp;
rp: procedure (d) address reentrant;
  declare d byte;
  declare rs structure (u(3) byte, v(6) address);
  if d = 0 then return size(rs.v) + last(rs.u);
  return rp(d - 1) + length(rs.v);
end rp;
i = 1;
call ph(length(st.z)); call ph(last(st.z)); call ph(size(st.z));
call ph(size(st.x)); call ph(size(st.y)); call ph(size(st));
call ph(length(sa.z)); call ph(last(sa.z)); call ph(size(sa.z));
call ph(length(sa(1).z)); call ph(last(sa(i).w)); call ph(size(sa(2).w));
call ph(size(sa(1))); call ph(size(sa(i))); call ph(size(sa));
call ph(size(ab(2))); call ph(size(aw(i))); call ph(size(sa(1).y)); call ph(size(sa.y));
call ph(size(sa(1).z(1))); call ph(size(sa.w)); call ph(length(sa.w)); call ph(size(st.z(1)));
do i = 0 to last(ab); ab(i) = i + 30h; end;
n = 0;
do i = 0 to last(st.z);
  st.z(i) = i; n = n + 1;
end;
call ph(n);
call ph(length(st.z) + 0ffh); call ph(last(sa.w) + 0feh);
call ph(ab(last(st.z)));
ab(last(sa.z)) = 7; call ph(ab(1));
call ph(size(bs.k) + size(bs) + length(ba.k) + size(ba(2).k) + size(ba(1)) + size(ba)
  + size(ba.k(1)));
call ph(lp(1));
call ph(rp(3));
w = size(sa(i).w) * 3; call ph(w);
"""
QUALIFIED_SIZES_PRINT = [4, 3, 4, 1, 2, 7, 2, 1, 2, 2, 4, 10, 15, 15, 45, 1, 2, 2, 2,
                         1, 10, 5, 1, 4, 3, 2, 0x33, 7, 0x42, 6, 0x10, 0x1E, 0x20, 0x1E]


def test_length_last_and_size_of_a_qualified_reference():
    """uplm80 took only a variable's name: `LENGTH(st.z)' was "LENGTH()
    needs an array whose extent is known", and `SIZE(sa(2))' "SIZE() needs
    a declared variable" (0.4.1 the same)."""
    _check(QUALIFIED_SIZES, QUALIFIED_SIZES_PRINT)


@pytest.mark.parametrize("expr", [
    "length(sa(1))", "length(st.x)", "length(ab(1))", "length(st)", "last(i)",
    "length(sa(1).z(1))", "size(i(1))", "size(st.nosuch)", "size(sa(1).x(1))",
    "length(sa.q)"])
def test_length_last_and_size_of_what_is_not_an_array_or_a_variable(expr):
    """What V3.1 rejects - LENGTH of an element or a scalar (ERROR #125,
    #157), a subscript on a scalar (#127), a member no structure has
    (#112) - uplm80 rejects too."""
    src = _PRELUDE + (
        "declare st structure (x byte, y address, z(4) byte);\n"
        "declare sa(3) structure (x byte, y address, z(2) byte, w(5) address);\n"
        "declare ab(6) byte, i byte, w address;\n"
        f"w = {expr};\nend t;\n")
    compiler = Compiler(opt_level=2)
    assert compiler.compile(src, "<test>") is None
    errors = [str(e) for e in compiler.errors.errors]
    assert any("needs an array" in e or "SIZE() needs" in e for e in errors), errors


# ---- a name the program declares hides the built-in ------------------------

def test_a_procedure_named_like_a_built_in_is_the_programs():
    """PL/M-80's built-ins are declared outside the program, and a
    declaration of the name hides one (9.2); Intel's PL/M-80 V3.1 calls the
    program's DOUBLE, LOW, SHL and the rest.  The optimizer wrote an ADDRESS
    constant below 256 as DOUBLE(n), and took every DOUBLE(n) for one, so
    `double(30h)' was 30H at -O1 and up however DOUBLE was written; and code
    generation folded `double(1)' in a condition, and `low(3)' in a DO's
    bound, by the built-ins at every level.  The optimizer's DOUBLE is now
    called by a name no program can declare (ast_view.DOUBLE_MARK), and
    what folds a built-in folds only a name the program does not declare;
    `w * 8' is no longer turned into a call of the program's SHL.  The
    program compiled by Intel's PL/M-80 V3.1 prints what is expected
    here."""
    _check("""
declare (w, z) address, (b, c) byte;
p: procedure;
  double: procedure (x) address; declare x byte; return x + 1000h; end double;
  low: procedure (x) byte; declare x address; return 11h; end low;
  high: procedure (x) byte; declare x address; return 22h; end high;
  shl: procedure (x, n) address; declare x address, n byte; return x + n; end shl;
  shr: procedure (x, n) address; declare x address, n byte; return x - n; end shr;
  rol: procedure (x, n) byte; declare (x, n) byte; return x + n + 1; end rol;
  ror: procedure (x, n) byte; declare (x, n) byte; return x + n + 2; end ror;
  scl: procedure (x, n) byte; declare (x, n) byte; return x + n + 3; end scl;
  scr: procedure (x, n) byte; declare (x, n) byte; return x + n + 4; end scr;
  dec: procedure (x) byte; declare x byte; return x + 5; end dec;
  call ph(double(30h));
  w = double(30h) + 1; call ph(w);
  b = 5; w = double(b); call ph(w);
  call ph(low(1234h)); call ph(high(1234h));
  w = 1234h; call ph(low(w)); call ph(high(w));
  call ph(shl(1, 8)); call ph(shr(8000h, 15));
  call ph(rol(1, 1)); call ph(ror(1, 1)); call ph(scl(1, 1)); call ph(scr(1, 1));
  call ph(dec(9));
  if double(1) = 1001h then call ph(0aaaah);
  c = 0; do b = 0 to low(3); c = c + 1; end;
  call ph(c);
  w = 3; call ph(w * 8); call ph(w / 2); z = w * 4; call ph(z);
end p;
call p;
""", [0x1030, 0x1031, 0x1005, 0x11, 0x22, 0x11, 0x22, 9, 0x7FF1, 3, 4, 5, 6, 0xE,
      0xAAAA, 0x12, 0x18, 1, 0xC])


def test_a_variable_named_like_a_built_in_is_the_programs():
    """An array OUTPUT or MEMORY of the program's was stored to as the port
    or as memory past the end of the program, `.memory' was the end of the
    program, and a STACKPTR of its own was SP, at every level (0.3.6 the
    same).  Intel's PL/M-80 V3.1 prints what is expected here."""
    _check("""
declare w address;
r: procedure;
  declare output (2) byte, memory (2) byte, stackptr address, move byte, input (2) byte;
  declare length address, last byte, size (2) address, time byte, zero byte;
  declare carry byte, sign byte, parity byte;
  output(1) = 16h; memory(1) = 17h; stackptr = 18h; move = 19h; input(0) = 1ah;
  call ph(output(1)); call ph(memory(1)); call ph(stackptr); call ph(move);
  call ph(input(0)); w = .memory(1) - .memory; call ph(w);
  w = stackptr + 1; call ph(w);
  length = 7; last = 8; size(1) = 9; time = 10; zero = 12;
  carry = 13; sign = 14; parity = 15;
  call ph(length); call ph(last); call ph(size(1)); call ph(time);
  call ph(zero); call ph(carry); call ph(sign); call ph(parity);
  if zero then call ph(1); if carry then call ph(2);
end r;
call r;
""", [0x16, 0x17, 0x18, 0x19, 0x1A, 1, 0x19, 7, 8, 9, 0xA, 0xC, 0xD, 0xE, 0xF, 2])


# ---- the differential test -------------------------------------------------

@pytest.mark.parametrize("seed", [11, 12, 13])
def test_random_programs_print_what_the_rules_say(seed):
    """Random typed programs against a Python model of the manual's rules
    (tests/plm_difftest.py), at every -O. scripts/difftest.py runs this with
    as many seeds as you like."""
    src, expect = generate(seed, 50)
    for opt in LEVELS:
        reason = tools_missing()
        if reason:
            pytest.skip(reason)
        got, err = build_and_run(src, opt)
        assert err is None, f"-O{opt}: {err}"
        wrong = [f"{label}: got {g:04X}, model {v:04X}"
                 for (label, v), g in zip(expect, got) if g != v]
        assert len(got) == len(expect) and not wrong, (
            f"seed {seed} -O{opt}:\n" + "\n".join(wrong[:10]))
