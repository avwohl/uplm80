"""Expressions keep the value and the type PL/M-80 gives them, at every -O.

Intel's PL/M-80 Programming Manual types every operand and operator: a
constant up to 255 is a BYTE (4.1.1); `+ - AND OR XOR` of two BYTEs, and
unary `-` and `NOT` of one, are 8-bit and wrap (4.2, 4.3); `* / MOD` are
ADDRESS (4.2.3); a relation is the BYTE 0FFH or 0 (4.4). Every rewrite the
compiler makes -- folding, propagation, strength reduction, algebra,
inlining -- has to leave the value and the type the unoptimized program
computes, or the arithmetic around the rewritten operand changes width.

Each case here printed something else, or did not compile, at some level
before its fix -- except test_the_check_still_catches_what_was_written and
test_restricted_expressions_are_plain_numbers, which hold what the fixes
must not change. The last test runs random programs against a model of the
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
    # Nor is one it moved to the right of a relation.
    "t: do; declare b byte; if 300 = b then b = 1; end t;",
])
def test_folding_does_not_make_a_comparison_impossible(src, opt):
    """A remainder is an ADDRESS, so comparing it with 1000H is not the
    `BYTE > 4096' uplm80 rejects (defect 2)."""
    _asm(src, opt)


def test_the_check_still_catches_what_was_written():
    with pytest.raises(AssertionError):
        _asm("t: do; declare b byte; if b = 300 then b = 1; end t;", 0)


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
