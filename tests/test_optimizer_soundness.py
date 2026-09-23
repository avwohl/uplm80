"""Cases where an optimisation or a storage decision changed what a program does.

All four were found by an adversarial sweep over the compiler while chasing a
miscompilation of MP/M II's STAT.PLM, and each was reduced to a few lines of
PL/M and confirmed against the emitted Z80.
"""

from uplm80.codegen import Mode
from uplm80.compiler import Compiler


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
