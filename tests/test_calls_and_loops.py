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
