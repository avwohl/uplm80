"""Cases where an optimisation or a storage decision changed what a program does.

All four were found by an adversarial sweep over the compiler while chasing a
miscompilation of MP/M II's STAT.PLM, and each was reduced to a few lines of
PL/M and confirmed against the emitted Z80.
"""

import pytest

from uplm80.codegen import Mode
from uplm80.compiler import Compiler

from ._toolchain import run_plm


def _asm(src: str, opt: int = 2, mode: Mode = Mode.CPM) -> str:
    out = Compiler(mode=mode, opt_level=opt).compile(src, "<test>")
    assert out is not None, "compilation failed"
    return out


ARG_OVERLAP_SRC = """
t1: do;
  declare r address;
  f: procedure(x,y); declare (x,y) address; r = x*100 + y; end f;
  g: procedure address; declare (p,q,s) address; p=1; q=2; s=3; return p+q+s; end g;
  h: procedure; call f(7, g); end h;
  call h;
end t1;
"""


def test_a_callee_frame_is_live_while_its_own_arguments_are_evaluated():
    """`call f(7, g)' stores 7 into f's slot, then runs g.

    A non-reentrant local procedure takes its arguments in its own shared
    slots, and the caller fills them one at a time, so f's frame holds live
    data before the call is made. The overlay analysis only considered a
    procedure live once it was ON the stack, so it was free to put g's locals
    on top of f's parameters - and g ran between the two stores.
    """
    asm = _asm(ARG_OVERLAP_SRC)
    slots = {}
    for line in asm.splitlines():
        s = line.strip()
        if s.startswith("ld\t(??AUTO+") or s.startswith("ld\thl,(??AUTO+"):
            off = int(s.split("??AUTO+")[1].split(")")[0])
            slots.setdefault(off, 0)
    # f has two ADDRESS parameters and g three ADDRESS locals: five distinct
    # slots, so at least 10 bytes, and none of them shared.
    assert "ds\t10" in [l.strip() for l in asm.splitlines()], \
        [l.strip() for l in asm.splitlines() if l.strip().startswith("ds")]


def test_embedded_assignment_target_is_not_folded_to_a_constant():
    """`q = (k := 7)' after `k = 5' must still store into k.

    Constant propagation rewrote the lvalue into the literal 5, and codegen
    then stored through it - address 0005H, the BDOS entry vector.
    """
    for opt in (2, 3):
        asm = _asm("""
b: do;
pc: procedure(c) external; declare c byte; end pc;
declare k byte, q byte;
    k = 5;
    q = (k := 7);
    call pc(k);
end b;
""", opt=opt)
        lines = [l.strip() for l in asm.splitlines()]
        assert "ld\t(K),a" in lines, f"-O{opt}: k is never stored:\n{asm}"
        # a store through a literal address looks like ld hl,<n> ... ld (hl),e
        assert "ld\thl,5" not in lines, f"-O{opt}: still folding the lvalue:\n{asm}"


def test_an_induction_step_inside_a_subscript_invalidates_the_variable():
    """`arr(i := i + 1) = 9' modifies i, so the loop test is not invariant.

    The walker that decides which variables a loop body changes looked at the
    array name and at the right-hand side, never at the subscript, so at -O3
    the exit test was folded away and the loop never ended.
    """
    for opt in (2, 3):
        asm = _asm("""
c: do;
declare arr (8) byte, i byte, n byte;
i = 0; n = 0;
do while i < 5;
  arr(i := i + 1) = 9;
  n = n + 1;
  end;
end c;
""", opt=opt)
        lines = [l.strip() for l in asm.splitlines()]
        assert "cp\t5" in lines, f"-O{opt}: the loop lost its exit test:\n{asm}"


def test_cpm_mode_is_unaffected_by_the_mpm_page_zero_symbols():
    asm = _asm("""
t: do;
mon2: procedure (f,a) byte external; declare f byte; declare a address; end mon2;
declare c byte, r byte;
r = mon2(2,c) + 1;
end t;
""")
    assert "??BDOS" not in asm, asm


O3_SRC = """
0100H:
t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
declare (i, j, c, k) byte;
putc: procedure (ch); declare ch byte; call mon1(2, ch); end putc;
/* a RETURN inside a loop, and inside an IF */
ru: procedure;
    do i = 0 to 3; c = c + 1; if c = 2 then return; end;
    c = 0;
end ru;
rv: procedure; c = c + 1; if c = 1 then return; c = 9; end rv;
/* a label, which inlining twice defines twice */
rl: procedure; lab: c = c + 1; if c < 3 then goto lab; end rl;
/* the module's c, which q's own c hides */
rc: procedure; c = c + 1; end rc;
q: procedure; declare c byte; c = 5; call rc; call putc('0' + c); end q;
/* three procedures called zn */
p1: procedure; zn: procedure; c = 1; end zn; call zn; end p1;
p2: procedure; zn: procedure; c = 2; end zn; call zn; end p2;
zn: procedure; c = 3; end zn;
g: procedure byte; k = k + 1; return k; end g;
r2: procedure (x); declare x byte; c = x + x; end r2;
setv: procedure (a); declare a address, v based a byte; v = 7; end setv;

c = 0; call rv; call putc('0' + c);
c = 0; call ru; call putc('0' + c);
c = 0; call rl; call putc('0' + c); c = 0; call rl; call putc('0' + c);
c = 0; call q; call putc('0' + c);
call p1; call putc('0' + c); call p2; call putc('0' + c); call zn; call putc('0' + c);
call putc('.');
/* what a call assigns */
k = 0; c = 0; call r2(g); call putc('0' + c); call putc('0' + k);
c = 0; k = 0; j = g; call putc('0' + k);
k = 0; if g then call putc('0' + k);
call putc('.');
/* an unrolled loop leaves its index one past the bound */
do i = 0 to 1; c = c + 1; end; call putc('0' + i);
do i = 0 to 1; lab2: c = c + 1; end; call putc('0' + i);
call putc('.');
/* `.c' is where c is */
c = 1; call setv(.c); call putc('0' + c);
call putc('.');
end t;
"""


@pytest.mark.parametrize("opt", [2, 3])
def test_o3_inlining_and_propagation_keep_what_the_program_does(opt):
    """-O3 inlines small procedures, unrolls short loops and propagates
    constants, and each of those changed what this program does:

      - the inliner dropped a top-level RETURN and kept any other, which
        then returned from the caller: nothing after `call ru' ran;
      - a label in an inlined procedure, or an unrolled loop, was defined
        once per copy, and the program did not assemble;
      - an inlined body named the caller's local where it meant the
        module's variable, and a nested procedure was inlined for another
        of the same name;
      - nothing learned before a CALL was forgotten after it, so `c' and
        `k' still read 0 after procedures that assign them;
      - an unrolled loop left its index at the last value, not one past;
      - `c = 1; CALL setv(.c)' passed setv the address 1.

    -O2 is the reference."""
    assert run_plm(O3_SRC, opt).strip() == "123351123.2111.22.7.", opt


SUBSCRIPTED_SCALAR_SRC = """
0100H:
t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
putc: procedure (ch); declare ch byte; call mon1(2, ch); end putc;
declare b byte, c byte, w address, v address, k byte;
/* after a character constant */
q1: procedure byte;
    declare (x, y) byte;
    x = 'x'; y = 'y';
    return x(1);
end q1;
/* after a copy: x(1) is the byte after x, not the byte after y */
q2: procedure byte;
    declare (x, y, z) byte;
    y = k; z = 'z';
    x = y;
    return x(1);
end q2;
b = 'b'; c = 'c';
call putc(b(1));
w = 1234h; v = 4443h;
call putc(low(w(1))); call putc(high(w(1)));
call putc(q1);
k = 'k';
call putc(q2);
call putc('.');
end t;
"""


@pytest.mark.parametrize("opt", [0, 1, 2, 3])
def test_a_subscripted_scalar_is_a_place_not_its_value(opt):
    """PL/M-80 lets a scalar be subscripted: `x(1)' is the byte after x.

    -O3 propagated into the name what it knew x held.  After `x = 'x'',
    `return x(1)' became a CALL through 78H with the argument 1, and the
    program ran into page zero; after `w = 1234H', `w(1)' was a CALL
    through 1234H.  After `x = y', `x(1)' read the byte after y.  The name
    of a subscripted variable is a place, like the target of an
    assignment.  -O0 to -O2 did not propagate there and are the reference."""
    assert run_plm(SUBSCRIPTED_SCALAR_SRC, opt) == "cCDyk.", opt


def test_a_subscripted_scalar_is_not_called():
    """The same at the level of the code: nothing in the program is an
    indirect call, so ??jpde is never needed."""
    asm = _asm(SUBSCRIPTED_SCALAR_SRC, opt=3)
    assert "??jpde" not in asm.lower(), asm
