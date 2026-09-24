"""Procedure calls and DO loops run the way PL/M-80 defines them, at every -O.

Arguments are evaluated and then assigned to the parameters, each converted
to its parameter's type (PL/M-80 manual, 8.2). An iterative DO compares the
index with the limit before every pass and ends when stepping the index
wraps (5.1.4). Each case here printed something else at some level before
its fix.
"""

import re

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
