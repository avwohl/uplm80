"""Procedure calls and DO loops run the way PL/M-80 defines them, at every -O.

Arguments are evaluated and then assigned to the parameters, each converted
to its parameter's type (PL/M-80 manual, 8.2). An iterative DO compares the
index with the limit before every pass and ends when stepping the index
wraps (5.1.4). Each case here printed something else at some level before
its fix.
"""

from tests.test_expression_types import _check


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


