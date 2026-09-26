"""Procedure calls and DO loops run the way PL/M-80 defines them, at every -O.

Arguments are evaluated and then assigned to the parameters, each converted
to its parameter's type (PL/M-80 manual, 8.2). An iterative DO compares the
index with the limit before every pass and ends when stepping the index
wraps (5.1.4). Each case here printed something else at some level before
its fix.
"""

import re

import pytest

from tests.test_expression_types import _PRELUDE, _asm, _check


def test_a_byte_argument_fills_an_address_parameter():
    """A BYTE argument to an ADDRESS parameter that is not the last one was
    stored as one byte, so the parameter's high byte was whatever the last
    call had left there."""
    _check("""
declare b byte, w address;
fp: procedure (p, q) address; declare p address, q byte; return p; end fp;
run: procedure;
  b = 12h; w = 0abcdh;
  call ph(fp(w, 1));
  call ph(fp(b, 1));
  call ph(fp(low(w), 1));
  call ph(fp('A', 1));
end run;
call run;
""", [0xABCD, 0x12, 0xCD, 0x41])


def test_a_procedure_called_in_its_own_arguments():
    """The arguments before one that calls the procedure again went into
    the procedure's parameters first, and the inner call stored over them:
    `f2(1, f2(2, 3))' took p = 2. Legal PL/M -- the outer activation has not
    begun -- and DRI's code, which passes arguments on the stack and lets
    the procedure store them, computes it."""
    _check("""
declare res address;
f2: procedure (p, q) address; declare (p, q) byte; return p * 16 + q; end f2;
f3: procedure (p, q, r) address; declare (p, q) byte, r address;
  return p * 256 + q * 16 + r;
end f3;
g: procedure (x) address; declare x byte; return f2(x, x); end g;
fw: procedure (a, b) address; declare (a, b) address; return a - b; end fw;
s2: procedure (p, q); declare (p, q) byte; res = p * 16 + q; end s2;
hb: procedure byte; call s2(9, 9); return 4; end hb;
run: procedure;
  call ph(f2(1, f2(2, 3)));
  call ph(f3(1, 2, f3(4, 5, 6)));
  call ph(f3(1, low(f2(2, 3)), 6));
  call ph(f2(7, g(1)));
  call ph(fw(10, fw(3, 1)));
  call s2(1, hb); call ph(res);
end run;
call run;
""", [0x33, 0x576, 0x336, 0x81, 8, 0x14])


def test_a_call_of_five_stacked_arguments_leaves_the_stack_as_it_was():
    """A CALL statement pops the arguments of a REENTRANT, PUBLIC or
    EXTERNAL procedure off the stack; for five or more it set SP to
    HL + SP, HL being whatever the procedure left there, not 10 + SP."""
    _check("""
declare (res, s0, s1) address;
r5: procedure (a, b, c, d, e) reentrant;
  declare (a, b, c, d, e) address;
  res = a + b + c + d + e;
end r5;
run: procedure;
  s0 = stackptr;
  call r5(1, 2, 3, 4, 5);
  s1 = stackptr;
  call ph(res); call ph(s0 - s1);
end run;
call run;
""", [15, 0])


def test_a_loop_stops_where_the_step_wraps_even_if_the_body_moved_the_index():
    """Step 4 of 5.1.4 ends the loop when stepping the index wraps, whatever
    put it there. With a constant limit and step that cannot wrap each
    other, the wrap exit was left out even when the body assigns the index:
    `DO i = 0 TO 0FEH' that adds 3 to i never stopped."""
    _check("""
declare (i, b) byte, (n, w) address;
run: procedure;
  n = 0; do i = 1 to 240; n = n + 1; i = i + 16; end; call ph(n); call ph(i);
  n = 0; do i = 0 to 0feh; n = n + 1; i = i + 3; end; call ph(n); call ph(i);
  n = 0; do b = 0 to 200 by 10; n = n + 1; if b = 50 then b = 250; end;
  call ph(n); call ph(b);
  n = 0; do w = 1 to 240; n = n + 1; if w = 5 then w = 0ffffh; end;
  call ph(n); call ph(w);
end run;
call run;
""", [15, 0, 64, 0, 6, 4, 5, 0])


def test_a_counted_loop_sees_what_pointers_change():
    """The DJNZ form counts the passes once and leaves the index alone
    until the end, so it cannot be used when a store through a pointer can
    change the index or the limit, or a pointer can read the index."""
    _check("""
declare (b, c, i) byte, n address;
declare p address, x based p byte;
declare buf(10) byte, q address, m based q byte;
run: procedure;
  p = .b;
  n = 0; do b = 0 to 9; n = n + 1; x = 9; end; call ph(n);
  q = .buf(3); buf(3) = 5;
  n = 0; do c = 0 to m; n = n + 1; buf(3) = 1; end; call ph(n);
  p = .i;
  n = 0; do i = 0 to 9; n = n + x; end; call ph(n);
end run;
call run;
""", [1, 2, 45])


def test_an_unrolled_loop_sees_what_calls_and_pointers_change():
    """-O3 unrolls a loop of two passes into two copies of the body, each
    after an assignment of the index -- which a call, or a store through a
    pointer to the index, in the body would have changed."""
    _check("""
declare i byte, n address;
declare p address, x based p byte;
bump: procedure; i = 10; end bump;
run: procedure;
  n = 0; do i = 0 to 1; call bump; n = n + 1; end; call ph(n); call ph(i);
  p = .i;
  n = 0; do i = 0 to 1; x = 20; n = n + 1; end; call ph(n); call ph(i);
end run;
call run;
""", [1, 0x0B, 1, 0x15])


def test_an_unrolled_loop_over_a_based_or_at_index():
    """An index BASED on a pointer moves when the body sets the pointer, and
    an index AT another variable changes when the body sets that variable by
    name -- directly, or in a procedure that -O3 inlines into the body.
    Neither is a store through a pointer, and -O3 unrolled both loops: the
    BASED one stored 2 and 82H through the moved pointer and added 210H,
    the AT one left the index 0FDH. Found by the integration verification's
    fuzzer (seeds 1540 and 20514)."""
    _check("""
declare acc address, buf(4) byte, p address, x based p byte, i byte;
declare g byte, y byte at (.g);
setp: procedure; p = .buf(2); end setp;
setg: procedure; g = 10; end setg;
run: procedure;
  acc = 0; buf(2) = 80h; buf(3) = 0ffh; p = .buf(3);
  do x = 2 to 0feh by 128;
    do i = 0 to 3; acc = acc + x; end;
    p = .buf(2);
  end;
  call ph(acc); call ph(buf(2)); call ph(buf(3));
  acc = 0; buf(2) = 80h; buf(3) = 0ffh; p = .buf(3);
  do x = 2 to 0feh by 128; acc = acc + x; call setp; end;
  call ph(acc); call ph(buf(2)); call ph(buf(3));
  do y = 254 to 0feh by 0ffh; g = 10; end; call ph(y);
  do y = 254 to 0feh by 0ffh; call setg; end; call ph(y);
end run;
call run;
""", [8, 0, 2, 2, 0, 2, 9, 9])


def test_a_return_from_inside_a_counted_loop():
    """The DJNZ form keeps its count on the stack while the body runs; a
    RETURN from the body returned through it."""
    _check("""
declare (k) byte, n address;
f: procedure byte;
  declare j byte;
  do j = 0 to 9; n = n + 1; if n = 3 then return 7; end;
  return 9;
end f;
run: procedure;
  n = 0; k = f; call ph(k); call ph(n);
end run;
call run;
""", [7, 3])


def test_loops_over_every_kind_of_index_and_limit():
    """The limit is compared with the index where the index lives -- a
    plain variable, one BASED on a pointer, a REENTRANT procedure's local
    -- and LAST is a constant limit like any other."""
    _check("""
declare (i, lim) byte, (n, w) address, ab(8) byte, big(1024) byte;
declare pb address, bi based pb byte;
rl: procedure (top) byte reentrant;
  declare top byte, (j, c) byte;
  c = 0; do j = 1 to top; c = c + j; end;
  return c;
end rl;
run: procedure;
  n = 0; do i = 0 to last(ab); n = n + 1; end; call ph(n); call ph(i);
  n = 0; do i = 0 to last(ab); n = n + i; end; call ph(n); call ph(i);
  n = 0; do w = 0 to last(big) by 128; n = n + 1; end; call ph(n); call ph(w);
  pb = .ab(2); lim = 3;
  n = 0; do bi = 0 to lim; n = n + 1; end; call ph(n); call ph(ab(2));
  call ph(rl(4)); call ph(rl(0));
end run;
call run;
""", [8, 8, 28, 8, 8, 1024, 4, 4, 10, 0])


def test_a_based_address_index_wraps():
    """A step of 1 to an ADDRESS index is `inc hl', which sets no flags, so
    the wrap is tested as `ld a,h / or l' -- after the store, which for a
    BASED index leaves the pointer in HL: such a loop to 0FFFFH never
    ended."""
    _check("""
declare (n, pw, lim) address, bw based pw address, wbuf(2) address;
run: procedure;
  pw = .wbuf(1);
  n = 0; do bw = 0fffeh to 0ffffh; n = n + 1; end; call ph(n); call ph(wbuf(1));
  lim = 0ffffh;
  n = 0; do bw = 0fffdh to lim; n = n + 1; end; call ph(n); call ph(wbuf(1));
end run;
call run;
""", [2, 0, 3, 0])


def test_a_counted_loop_inside_a_loop_over_the_same_index():
    """A counted loop gives its index the final value, one past the bound,
    unless nothing else reads the index. Another DO over the same index
    assigns it before reading it -- unless it encloses this one, whose value
    its step then reads: the outer loop below runs once, since the inner
    one leaves g's i at 10. With the store left out, it ran ten times."""
    _check("""
declare n address;
g: procedure;
  declare (i, c) byte;
  c = 0;
  do i = 0 to 9; c = c + 1; do i = 0 to 9; end; end;
  n = c;
end g;
g2: procedure;
  declare (i, c) byte;
  c = 0;
  do i = 0 to 9; end;
  do i = 0 to 3; c = c + 1; end;
  n = n + c;
end g2;
run: procedure;
  call g; call ph(n);
  call g2; call ph(n);
end run;
call run;
""", [1, 5])


def test_a_loop_whose_bound_reads_its_index_is_not_counted():
    """PL/M-80 evaluates the limit before every pass, after the index has
    been set and stepped, so `DO k = 0 TO k + 5' runs until k + 5 wraps: 251
    passes, leaving k at 251. The counted form took its count once, from k
    as it was before the loop: 106 passes, and k = 106. (Found by the
    verification's random programs, seeds 50092, 50114 and 50242.)"""
    _check("""
declare k byte, n address;
run: procedure;
  n = 0; k = 100;
  do k = 0 to k + 5; n = n + 1; end;
  call ph(n); call ph(k);
  n = 0; k = 100;
  do k = 0 to shr(k, 1) + 3; n = n + 1; end;
  call ph(n); call ph(k);
end run;
call run;
""", [251, 251, 7, 7])


def test_a_call_in_the_main_program_whose_argument_calls_another():
    """A call's arguments go into the callee's own slots one at a time, so
    the callee's frame is live while the later ones are evaluated, and a
    procedure they call must not be given storage that overlaps it. That
    was recorded for calls inside procedures but not for the main program:
    `CALL p2(5, g(1, 2))' at module level printed 1 3, g's parameter having
    landed on p2's first argument."""
    _check("""
declare (r1, r2, r3) address;
g: procedure (x, y) byte; declare (x, y) byte; return x + y; end g;
p2: procedure (a, b); declare (a, b) byte; r1 = a; r2 = b; end p2;
p3: procedure (a, b, c); declare (a, b, c) byte; r1 = a; r2 = b; r3 = c; end p3;
f2: procedure (a, b) address; declare (a, b) byte; return a * 16 + b; end f2;
call p2(5, g(1, 2)); call ph(r1); call ph(r2);
call p3(4, 6, g(1, 1)); call ph(r1); call ph(r2); call ph(r3);
call p3(g(3, 4), 1, g(1, 1)); call ph(r1); call ph(r2); call ph(r3);
r1 = f2(5, g(1, 2)); call ph(r1);
""", [5, 3, 4, 6, 2, 7, 1, 2, 0x53])


def test_every_pass_tests_the_carry_as_dri_does():
    """DRI's code tests the step's carry on every pass, even where the limit
    and the step cannot carry each other: UTIL3/LOAD.COM's `DO I = 0 TO 127'
    ends `LXI H,I / INR M / JNZ 0629H', and UTIL6/ED.PRL steps an ADDRESS
    index with `LXI D,1 / DAD D / SHLD I / JNC'. So a body that moves the
    index past the limit ends the loop at the step that carries, however it
    moved it -- here through a pointer to the variable declared next to it,
    which no analysis of the body sees. uplm80 left the carry test out of
    an ADDRESS loop whose constant limit and step fit, unless it saw the
    body change the index: the first loop went round again from 0 and ran
    107 times. (The BYTE loops read their index, or they would be counted
    in B, which trusts that nothing but the loop moves the index.)"""
    src = """
declare (pad, w) address, (pb, b) byte, (n, p, q) address;
declare wp based p address, bp based q byte;
run: procedure;
  p = .pad + 2; q = .pb + 1;
  w = 1234h; if wp <> 1234h then call ph(0eeeeh);
  b = 12h; if bp <> 12h then call ph(0eeeeh);
  n = 0; do w = 0 to 100; n = n + 1; if n = 6 then wp = 0ffffh; end;
  call ph(n); call ph(w);
  n = 0; do w = 0 to 1000 by 16; n = n + 1; if n = 3 then wp = 0fff8h; end;
  call ph(n); call ph(w);
  n = 0; do b = 0 to 100; n = n + 1; if b = 5 then bp = 0ffh; end;
  call ph(n); call ph(b);
  n = 0; do b = 0 to 11 by 2; n = n + 1; if b = 4 then b = 254; end;
  call ph(n); call ph(b);
end run;
call run;
"""
    _check(src, [6, 0, 3, 8, 6, 0, 3, 0])
    # The back edge is conditional in every one of those loops: a jump on
    # the carry (or the INC's zero) to the test at the top, never a plain one.
    asm = _asm(_PRELUDE + src + "\nend t;\n", 2)
    run = asm[asm.index("RUN:"):]
    assert not re.search(r"\n\t(jp|jr)\s+\?\?TEST", run), run


def test_a_loop_whose_index_is_moved_past_the_limit_stops_at_the_carry():
    """The verification's f4_index_past_limit: a body that sets the index
    past the limit, so that the step carries out, ends the loop there."""
    _check("""
declare (i, n) byte, w address;
run: procedure;
  n = 0; do i = 0 to 11 by 2; if i = 4 then i = 254; n = n + 1; end;
  call ph(n); call ph(i);
  n = 0; do i = 0 to 10; if i = 3 then i = 255; n = n + 1; end;
  call ph(n); call ph(i);
  n = 0; do w = 0 to 100; if w = 5 then w = 0ffffh; n = n + 1; end;
  call ph(n); call ph(w);
end run;
call run;
""", [3, 0, 4, 0, 6, 0])


def test_a_reentrant_procedures_arrays_and_block_locals_are_in_its_frame():
    """An array or a structure local to a REENTRANT procedure was given room
    in the frame but addressed by a label that was never defined, so the
    program did not assemble ("Undefined symbol 'V'"); a local declared in a
    DO block inside one was put below SP, since the frame was sized before
    the body was read, and the recursive call's push wrote over it. Each
    activation now keeps its own copy, arrays after the scalars that
    `(ix+d)' reaches (0.3.6 did the same; found by the integration
    verification)."""
    _check("""
declare gb(4) byte;
sumb: procedure (p, n) address;
  declare p address, n byte, b based p (1) byte, (s, i) address;
  s = 0; do i = 0 to n - 1; s = s + b(i); end; return s;
end sumb;
rv: procedure (n) byte reentrant;
  declare n byte, v(2) byte;
  v(0) = n; v(1) = n + 1;
  if n = 0 then return v(1);
  return rv(n - 1) + v(0);
end rv;
r: procedure (n) address reentrant;
  declare n byte;
  declare big(200) byte, w(3) address, (i, j) byte, a address;
  declare st structure (x byte, y address, z(3) byte);
  declare sa(2) structure (m byte, q(2) address);
  do i = 0 to last(big); big(i) = i + n; end;
  w(0) = n; w(1) = 1000h + n; j = 2; w(j) = w(1) + w(0);
  st.x = n; st.y = 1234h; st.z(0) = 1; st.z(j) = n + 7;
  sa(1).m = n + 1; sa(1).q(1) = 0abcdh; sa(0).q(0) = n;
  a = size(big) + length(w);
  if n > 0 then a = a + r(n - 1);
  if n = 1 then do;
    call ph(big(199)); call ph(big(j)); call ph(w(2)); call ph(st.z(j));
    call ph(sa(1).m); call ph(sa(1).q(1)); call ph(sa(0).q(0)); call ph(st.y);
    call ph(sumb(.big(10), 3));
    call move(3, .big(20), .gb); call ph(gb(0) + gb(2));
  end;
  do;
    declare l(2) address;
    l(0) = 7; l(1) = n;
    a = a + l(0) + l(1);
  end;
  return a;
end r;
q: procedure (n) byte reentrant;
  declare n byte;
  do;
    declare c byte;
    c = n + 1;
    if n = 0 then return c;
    return q(n - 1) + c;
  end;
end q;
run: procedure;
  call ph(rv(3)); call ph(r(2)); call ph(q(3));
end run;
call run;
""", [7, 0xC8, 3, 0x1002, 8, 2, 0xABCD, 1, 0x1234, 0x24, 0x2C, 0x279, 10])


def test_a_counted_loop_ends_where_a_pointer_or_an_overrun_sets_its_index():
    """A BYTE `DO i = 0 TO n' whose body does not name i counted its passes
    in B when i was a module-level variable no procedure named and no `.i'
    reached - but a pointer made from the address of the variable declared
    before i reaches i as well (`p = .a + 1'), and so does an array
    declared before it, subscripted past its end (`buf(2)' of a `buf (2)
    byte'), and a store through either did not end the loop: n was 11, not
    3 and 4.  So does a pointer from the name before it in a factored
    declaration, `(a2, i2)'.  A procedure's local was not counted in that
    case already.
    DRI's PL/M-80 never counts a loop, and the program compiled by Intel's
    PL/M-80 V3.1 prints what is expected here."""
    _check("""
declare a byte, i byte, n byte;
declare p address, x based p byte;
declare buf (2) byte, j byte, k byte;
declare (a2, i2) byte;
run: procedure;
  n = 0;
  p = .a + 1;
  do i = 0 to 10;
    n = n + 1;
    if n = 3 then x = 20;
  end;
  call ph(n); call ph(i);
  n = 0; k = 2;
  do j = 0 to 10;
    n = n + 1;
    if n = 4 then buf(k) = 30;
  end;
  call ph(n); call ph(j);
  n = 0; p = .a2 + 1;
  do i2 = 0 to 10;
    n = n + 1;
    if n = 5 then x = 40;
  end;
  call ph(n); call ph(i2);
end run;
call run;
""", [3, 0x15, 4, 0x1F, 5, 0x29])


@pytest.mark.parametrize("decl, store", [
    ("declare s structure (m(2) byte);", "s.m(2) = 20;"),
    ("declare s structure (m(2) byte, q byte);", "s.m(3) = 20;"),
    ("declare s2 (2) structure (m(2) byte);", "s2(1).m(2) = 20;"),
    ("declare s2 (2) structure (m(2) byte);", "s2(1).m(k) = 20;"),
])
def test_a_counted_loop_ends_where_a_members_overrun_sets_its_index(decl, store):
    """A structure's member subscripted past its end reaches the variables
    laid out after the structure, as an array does: `s.m(2)' of a module-
    level `s structure (m(2) byte)' is the i declared after it, and so is
    `s2(1).m(2)' of an `s2 (2) structure (m(2) byte)', and `s2(1).m(k)'
    with k = 2.  A loop over i was still counted in B, so the store did
    not end it: n was 11, not 3, at every level.  Nothing noted a
    member's constant subscript, nor any subscript of a member whose
    structure is itself subscripted.  A procedure's local was right.
    Intel's PL/M-80 V3.1 build of each prints what is expected here."""
    _check(f"""
{decl}
declare i byte, n byte;
declare k byte;
run: procedure;
  n = 0; k = 2;
  do i = 0 to 10;
    n = n + 1;
    if n = 3 then {store}
  end;
  call ph(n); call ph(i);
end run;
call run;
""", [3, 0x15])


@pytest.mark.parametrize("decl, before, store", [
    ("i byte, x byte, n byte, p address, b based p byte", "p = .x - 1;", "b = 20;"),
    ("i byte, x byte, n byte, z byte at (.x - 1)", "", "z = 20;"),
    ("i byte, a (2) byte, n byte", "", "a(0ffffh) = 20;"),
    ("i byte, a (2) byte, n byte, k address", "k = 0ffffh;", "a(k) = 20;"),
])
def test_a_counted_loop_ends_where_a_store_runs_back_to_its_index(decl, before, store):
    """A pointer or a subscript runs backwards as well as on: `.x - 1' is
    the i laid out just before x, and so is `a(0ffffh)' of an `a (2) byte'
    declared after i, the subscript wrapping, and `a(k)' with k = 0FFFFH.
    A module-level index was not counted only when what reaches it is laid
    out before it, so the loop over i was counted in B and the store did
    not end it: n was 11, not 3, at every level (0.4.0 the same).  Intel's
    PL/M-80 V3.1 build of each prints what is expected here."""
    _check(f"""
declare {decl};
run: procedure;
  n = 0; {before}
  do i = 0 to 10;
    n = n + 1;
    if n = 3 then {store}
  end;
  call ph(n); call ph(i);
end run;
call run;
""", [3, 0x15])


def test_a_counted_loop_ends_where_a_store_runs_back_to_a_static_index():
    """A procedure's static local is laid out among the module's variables,
    in the source's order, as in DRI's layout: the i read before it is
    written is static, and so is the x after it whose address is taken, so
    `.x - 1' is i.  The loop over i was counted, since only a pointer from
    a local declared before i was taken to reach it: n was 11, not 3, at
    every level (0.4.0 the same).  Intel's PL/M-80 V3.1 build prints what
    is expected here."""
    _check("""
run: procedure;
  declare i byte, x byte, n byte, p address, b based p byte;
  if i = 99 then n = 1;
  n = 0; p = .x - 1;
  do i = 0 to 10;
    n = n + 1;
    if n = 3 then b = 20;
  end;
  call ph(n); call ph(i);
end run;
call run;
""", [3, 0x15])


# A counted loop over the last variable, which MEMORY follows, and
# a store through MEMORY that reaches it: a pointer made from `.memory', a
# constant subscript that wraps, one that is not a constant.  Each body
# follows _PRELUDE; tests/test_intel_oracle.py builds each with Intel's
# PL/M-80 V3.1 too.
_MEMORY_LOOP = """
{decl}
run: procedure;
  {local}
  n = 0; {before}
  do i = 0 to 10;
    n = n + 1;
    if n = 3 then {store}
  end;
  call ph(n); call ph(i);
end run;
call run;
"""
MEMORY_RUNS_BACK = {
    "pointer": _MEMORY_LOOP.format(
        decl="declare n byte, p address, b based p byte, i byte;", local="",
        before="p = .memory - 1;", store="b = 20;"),
    "wraps": _MEMORY_LOOP.format(
        decl="declare n byte, i byte;", local="", before="", store="memory(0ffffh) = 20;"),
    "variable": _MEMORY_LOOP.format(
        decl="declare n byte, k address, i byte;", local="", before="k = 0ffffh;",
        store="memory(k) = 20;"),
    "static": _MEMORY_LOOP.format(
        decl="", local="declare n byte, i byte;\n  if i = 99 then n = 1;", before="",
        store="memory(0ffffh) = 20;"),
}


@pytest.mark.parametrize("case", sorted(MEMORY_RUNS_BACK))
def test_a_counted_loop_ends_where_a_store_through_memory_sets_its_index(case):
    """MEMORY begins where the last variable ends, in DRI's layout as in
    uplm80's, so `.memory - 1' is the last variable, and so is
    `memory(0ffffh)', the subscript wrapping, and `memory(k)' with k =
    0FFFFH; a procedure's static local is laid out among the module's
    variables.  Nothing but a declared variable was taken to reach the
    others, so a loop over the last one was counted in B and the store did
    not end it: 000B 0014, where Intel's PL/M-80 V3.1 build prints 0003
    0015, at every level (0.4.1 the same)."""
    _check(MEMORY_RUNS_BACK[case], [3, 0x15])


MEMORY_RUNS_ON = """
declare n byte, i byte;
run: procedure;
  n = 0;
  do i = 0 to 10;
    n = n + 1;
    if n = 3 then memory(5) = 20;
  end;
  call ph(n); call ph(i + memory(5));
end run;
call run;
"""


def test_memory_run_on_from_its_start_leaves_a_loop_counted():
    """A constant subscript below 8000H runs on from the end of the
    variables, never back into them: the loop over the last one is still
    counted in B, and prints what Intel's V3.1 build prints."""
    _check(MEMORY_RUNS_ON, [11, 0x1F])
    asm = _asm(_PRELUDE + MEMORY_RUNS_ON + "\nend t;\n", 2)
    assert "djnz" in asm[asm.index("RUN:"):]


# Labels on END statements (9800268B, A.4.4.1), and a GOTO to each: on to
# the next step of an iterative DO and the next test of a DO WHILE, out of
# a DO CASE, a DO and a procedure.  Follows _PRELUDE; Intel's PL/M-80 V3.1
# compiles it to print what is expected (tests/test_intel_oracle.py).
END_LABELS = """
declare (i, n, k, r) byte;
p: procedure (x) byte;
  declare x byte;
  r = 1;
  if x > 3 then goto out;
  n = n + 10;
  r = 2;
  return r;
out: end p;
q: procedure;
  if n > 100 then goto done;
  n = n + 1;
done: finish:end q;
n = 0;
do i = 1 to 5;
  if i = 3 then goto next;
  n = n + 1;
next: end;
call ph(n); call ph(i);
n = 0;
do k = 0 to 2;
  do case k;
    n = n + 1;
    goto cend;
    n = n + 4;
  cend: end;
  n = n + 16;
end;
call ph(n);
n = 0; i = 0;
do while i < 4;
  i = i + 1;
  if i = 2 then goto wend;
  n = n + 1;
wend: /* the end */ end;
call ph(n); call ph(i);
n = 0;
k = p(2); call ph(r); call ph(n); k = p(9); call ph(r); call ph(n);
n = 200; call q; call ph(n); n = 5; call q; call ph(n);
blk: do;
  n = 1;
  goto bend;
  n = 2;
bend: end blk;
call ph(n);
"""


def test_a_label_on_an_end_statement():
    """`out: end p;' was a syntax error: the grammar took a label only on a
    statement a block holds (0.4.1 the same).  A GOTO to it goes on to the
    loop's next step or test, out of a DO CASE (not a case of it: the jump
    table is the three cases'), a DO or a procedure."""
    _check(END_LABELS, [4, 6, 0x35, 3, 4, 2, 10, 1, 10, 0xC8, 6, 1])
    for opt in (0, 2):
        asm = _asm(_PRELUDE + END_LABELS + "\nend t;\n", opt)
        assert len(re.findall(r"^\?\?CASE\d+:", asm, re.M)) == 3, asm


@pytest.mark.parametrize("where", ["module", "local"])
def test_a_member_of_an_unsubscripted_array_of_structures_runs_on(where):
    """`s2.m(4)' of an `s2 (2) structure (m(2) byte)' is s2(0).m(4) to
    uplm80 - Intel's PL/M-80 V3.1 rejects it, ERROR #133 (CHANGELOG, Known
    issues) - which is the i declared after s2.  For a procedure's local
    local_storage took the reference to reach no further than s2, and the
    loop over i was counted: n was 11, not 3."""
    decls = "declare s2 (2) structure (m(2) byte);\ndeclare i byte;\n"
    _check(("" if where == "local" else decls) + """declare n byte;
run: procedure;
""" + (decls if where == "local" else "") + """  n = 0;
  do i = 0 to 10;
    n = n + 1;
    if n = 3 then s2.m(4) = 20;
  end;
  call ph(n); call ph(i);
end run;
call run;
""", [3, 0x15])


def test_a_reentrant_procedures_parameter_factored_with_its_locals():
    """`DECLARE (TOP, C) BYTE' names a REENTRANT procedure's parameter with
    a local. The parameter was declared a second time, as a local in the
    frame that nothing had set, and every use of it read that: rp(3) was 1
    (0.3.6 the same). Declared on its own it was right. Intel's PL/M-80
    V3.1 compiles this program to print what is expected here."""
    _check("""
rp: procedure (top) byte reentrant;
  declare (top, c) byte;
  c = top + 1;
  if top = 0 then return c;
  return rp(top - 1) + c;
end rp;
rq: procedure (n, s) address reentrant;
  declare (k, n) byte, (s, w) address;
  k = n; w = s + k;
  if n = 0 then return w;
  return rq(n - 1, w) + k;
end rq;
rr: procedure (a, b, c) address reentrant;
  declare (a, x, b) address, (y, c) byte;
  x = a + 1; y = c + 2;
  if c = 0 then return x + b + y;
  return rr(x, b, c - 1) + y;
end rr;
run: procedure;
  call ph(rp(3)); call ph(rq(3, 100h)); call ph(rr(10h, 200h, 2));
end run;
call run;
""", [0xA, 0x10C, 0x21C])


def test_a_based_array_on_a_structure_member_indexed_by_a_byte():
    """`token BASED pcb.tok (4) BYTE' keeps its pointer in the member
    `pcb.tok'. An element with a variable BYTE subscript took the pointer
    from the first word of `pcb' instead, so `token(i)' read `junk(2)' for
    `buf(2)' at -O0 to -O2 (at -O3 `i' is known and the subscript constant);
    0.3.6 did the same. A constant subscript and a scalar BASED on a member
    were right."""
    _check("""
declare pcb structure (state address, tok address);
declare token based pcb.tok (4) byte;
declare buf (4) byte, junk (4) byte, i byte;
run: procedure;
  buf(2) = 43h; buf(1) = 42h; junk(2) = 58h;
  pcb.state = .junk; pcb.tok = .buf;
  i = 2;
  call ph(token(i)); call ph(token(1));
end run;
call run;
""", [0x43, 0x42])


def test_a_reentrant_procedures_local_pointer_bases_a_variable():
    """A BASED variable whose pointer is a REENTRANT procedure's local or
    parameter read the pointer by a label nothing defines, and did not
    assemble (0.3.6 the same). The pointer is in the frame, and each
    activation has its own."""
    _check("""
declare g (3) byte;
r: procedure (n) byte reentrant;
  declare n byte, p address, x based p byte;
  p = .g(n);
  if n = 0 then return x;
  return r(n - 1) + x;
end r;
s: procedure (q, n) address reentrant;
  declare q address, n byte, w based q (1) address, t address;
  if n = 0 then return w(0);
  w(1) = w(1) + 1;
  t = s(q, n - 1);
  return t + w(1);
end s;
declare ws (2) address;
run: procedure;
  g(0) = 1; g(1) = 2; g(2) = 3;
  call ph(r(2));
  ws(0) = 100h; ws(1) = 0;
  call ph(s(.ws, 3));
end run;
call run;
""", [6, 0x100 + 3 + 3 + 3])
