"""PL/M-80's calling convention, run: uplm80 code with assembly written to
it, with itself across modules, and with Intel's own compiled code.

Every routine in the assembly here takes its arguments the way PL/M-80 V3.1
passes them: one in BC (C for a BYTE), two in BC (C) then DE (E), and more
with the earlier ones pushed left to right and taken off the stack by the
callee.  It returns a BYTE in A and an ADDRESS in HL, and preserves nothing.
And each routine that calls the program's procedures passes arguments that
way, with garbage in the high byte of a BYTE.  Every program checks, as it
goes, that SP is where it started.
"""
import os
import re
import subprocess
import tempfile

import pytest

from tests._toolchain import compile_cmd, compiler_env, run_asm, run_plm, tools_missing
from uplm80.compiler import Compiler

LEVELS = (0, 1, 2, 3)

PRELUDE = """t: do;
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
declare res (5) address external;
declare (sp0, i) address;
dump: procedure (n);
  declare n byte;
  do i = 0 to n - 1; call ph(res(i)); end;
  call mon1(2, '|');
end dump;
"""

# Routines written to PL/M-80's convention.  CAPn store what they get in RES
# (a BYTE widened here: B, D, H of a BYTE argument are not defined).
CALLEES_ASM = """
    public RES, CAP0, CAP1B, CAP1A, CAP2, CAP3, CAP4, CAP5, F2, F3
CAP0:   ld hl,0C0DEh
        ld (RES),hl
        ret
CAP1B:  ld a,c
        ld (RES),a
        xor a
        ld (RES+1),a
        ld bc,0FFFFh            ; nothing is preserved
        ld de,0FFFFh
        ret
CAP1A:  ld (RES),bc
        ret
; cap2(x byte, y address)
CAP2:   ld a,c
        ld (RES),a
        xor a
        ld (RES+1),a
        ld (RES+2),de
        ret
; cap3(a address, b byte, c address): a is stacked
CAP3:   ld (RES+4),de
        ld a,c
        ld (RES+2),a
        xor a
        ld (RES+3),a
        pop hl
        ex (sp),hl
        ld (RES),hl
        ret
; cap4(w byte, x address, y byte, z address)
CAP4:   ld (RES+6),de
        ld a,c
        ld (RES+4),a
        xor a
        ld (RES+5),a
        pop hl                  ; return address
        pop de                  ; x
        ld (RES+2),de
        ex (sp),hl              ; w, and the return address back on top
        ld a,l
        ld (RES),a
        xor a
        ld (RES+1),a
        ret
; cap5(v address, w byte, x address, y byte, z address)
CAP5:   ld (RES+8),de
        ld a,c
        ld (RES+6),a
        xor a
        ld (RES+7),a
        pop hl
        pop de                  ; x
        ld (RES+4),de
        pop de                  ; w
        ld a,e
        ld (RES+2),a
        xor a
        ld (RES+3),a
        ex (sp),hl              ; v
        ld (RES),hl
        ret
; f2(x byte, y address) byte: x - low(y), in A only
F2:     ld a,c
        sub e
        ld hl,0FFFFh
        ret
; f3(a address, b address, c address) address: a - c + b, in HL only
F3:     pop hl
        ex (sp),hl              ; a
        or a
        sbc hl,de
        add hl,bc
        ld a,0EEh
        ret
"""

# Assembly calling the program's PUBLIC procedures the way PL/M-80 does.
CALLERS_ASM = """
    public ACALL1, ACALL2, ACALL5, ACALLR, ARES
    extrn PUB1, PUB2, PUB5, SUM3
ACALL1: ld bc,0FF12h            ; pub1(b byte): B is not defined
        call PUB1
        ld (ARES),hl
        ret
ACALL2: ld bc,1234h             ; pub2(a address, b byte)
        ld de,0FF56h
        call PUB2
        ld (ARES),hl
        ret
ACALL5: ld hl,1111h             ; pub5(v address, w byte, x address, y byte, z address)
        push hl
        ld hl,0AA22h
        push hl
        ld hl,3333h
        push hl
        ld bc,0BB44h
        ld de,5555h
        call PUB5
        ld (ARES),hl
        ret
ACALLR: ld hl,3                 ; sum3(3, 10h, 100h), REENTRANT
        push hl
        ld bc,10h
        ld de,100h
        call SUM3
        ld (ARES),hl
        ret
"""


def _module(*parts: str) -> str:
    """An assembly module of ``parts``, with the storage they share."""
    return ("    .z80\n    cseg\n" + "".join(parts)
            + "    dseg\nRES:    ds 10\nARES:   ds 2\n    end\n")


BODY = """
cap0: procedure external; end cap0;
cap1b: procedure (x) external; declare x byte; end cap1b;
cap1a: procedure (x) external; declare x address; end cap1a;
cap2: procedure (x, y) external; declare x byte, y address; end cap2;
cap3: procedure (a, b, c) external; declare (a, c) address, b byte; end cap3;
cap4: procedure (w, x, y, z) external; declare (w, y) byte, (x, z) address; end cap4;
cap5: procedure (v, w, x, y, z) external; declare (v, x, z) address, (w, y) byte; end cap5;
f2: procedure (x, y) byte external; declare x byte, y address; end f2;
f3: procedure (a, b, c) address external; declare (a, b, c) address; end f3;
acall1: procedure external; end acall1;
acall2: procedure external; end acall2;
acall5: procedure external; end acall5;
acallr: procedure external; end acallr;
declare ares address external;
declare (g1, g2, g3, g4, g5) address;
pub1: procedure (b) address public; declare b byte; g1 = b; return b + 1; end pub1;
pub2: procedure (a, b) address public; declare a address, b byte; g1 = a; g2 = b; return a - b; end pub2;
pub5: procedure (v, w, x, y, z) address public;
  declare (v, x, z) address, (w, y) byte;
  g1 = v; g2 = w; g3 = x; g4 = y; g5 = z; return v + x + z;
end pub5;
sum3: procedure (a, b, c) address reentrant public;
  declare (a, b, c) address;
  if a = 0 then return b + c;
  return sum3(a - 1, b + 1, c) + 1;
end sum3;
tailc: procedure; call cap3(0abch, 0deh, 0f00h); end tailc;
declare (bv, bpad) byte, wv address;
bv = 0f5h; bpad = 0ffh; wv = 1234h;
sp0 = stackptr;
call cap0; call dump(1);
call cap1b(bv); call dump(1);
call cap1b(300); call dump(1);
call cap1a(bv); call dump(1);
call cap1a(wv); call dump(1);
call cap2(bv, wv); call dump(2);
call cap2(wv, bv); call dump(2);
call cap3(wv, bv, wv + 1); call dump(3);
call cap3(pub1(7), f2(9, 2), f3(10h, 1, 5)); call dump(3);
call cap4(1, 2, 3, 4); call dump(4);
call cap4(bv, bv, wv, wv); call dump(4);
call cap5(wv, 2, 3, bv, f3(wv, 1, 1)); call dump(5);
call tailc; call dump(3);
call ph(f2(9, 2)); call ph(f3(100h, 20h, 1)); call ph(stackptr - sp0);
call mon1(2, '|');
call acall1; call ph(ares); call ph(g1);
call acall2; call ph(ares); call ph(g1); call ph(g2);
call acall5; call ph(ares); call ph(g1); call ph(g2); call ph(g3); call ph(g4); call ph(g5);
call acallr; call ph(ares);
call ph(sum3(2, 1, 1));
call ph(stackptr - sp0);
"""

EXPECT = ("C0DE |00F5 |002C |00F5 |1234 |00F5 1234 |0034 00F5 |1234 00F5 1235 |"
          "0008 0007 000C |0001 0002 0003 0004 |00F5 00F5 0034 1234 |1234 0002 0003 00F5 1234 |0ABC 00DE 0F00 |"
          "0007 011F 0000 |0013 0012 11DE 1234 0056 9999 1111 0022 3333 0044 5555 0116 0006 0000 ")


@pytest.mark.parametrize("opt", LEVELS)
def test_calls_to_and_from_assembly_written_to_the_convention(opt):
    """0 to 5 arguments, BYTE and ADDRESS in every place, converted both
    ways, and arguments that are calls themselves; results in A only or HL
    only; `tailc', a procedure that ends in a call with an argument pushed
    (upeepz80 0.2.5 made it a jump, and CAP3 took the return address for
    its first argument); and assembly calling PUBLIC and REENTRANT
    procedures, recursive ones among them."""
    out = run_plm(PRELUDE + BODY + "\nend t;\n", opt, _module(CALLEES_ASM, CALLERS_ASM))
    assert out.replace("\0", "").strip() == EXPECT.strip()


INDIRECT = """
declare (q, r2) address;
declare (g1, g2, g3, g4) address;
p0: procedure; g1 = 0a0h; end p0;
p1: procedure (a); declare a address; g1 = a; end p1;
p2: procedure (a, b); declare a address, b byte; g1 = a; g2 = b; end p2;
p3: procedure (a, b, c); declare (a, b, c) address; g1 = a; g2 = b; g3 = c; end p3;
p4: procedure (a, b, c, d); declare (a, c) address, (b, d) byte; g1 = a; g2 = b; g3 = c; g4 = d; end p4;
u3: procedure (a, b, c) public; declare (a, b, c) address; g1 = a + 1; g2 = b + 1; g3 = c + 1; end u3;
r3: procedure (a, b, c) address reentrant; declare (a, b, c) address;
  if a = 0 then return b - c; return r3(a - 1, b, c); end r3;
show: procedure; call ph(g1); call ph(g2); call ph(g3); call ph(g4); call mon1(2, '|');
  g1, g2, g3, g4 = 0; end show;
sp0 = stackptr;
q = .p0; call q; call show;
q = .p1; call q(1234h); call show;
q = .p2; call q(1234h, 56h); call show;
q = .p3; call q(1, 2, 3); call show;
q = .p4; call q(4321h, 5, 6789h, 7); call show;
q = .u3; call q(1, 2, 3); call show;
q = .r3; call q(2, 50h, 8); call show;
q = .cap3; call q(7, 8, 9); call dump(3);
call ph(stackptr - sp0);
"""

INDIRECT_EXPECT = ("00A0 0000 0000 0000 |1234 0000 0000 0000 |1234 0056 0000 0000 |"
                   "0001 0002 0003 0000 |4321 0005 6789 0007 |0002 0003 0004 0000 |"
                   "0000 0000 0000 0000 |0007 0008 0009 |0000 ")


@pytest.mark.parametrize("opt", LEVELS)
def test_a_call_through_an_address_passes_any_number_of_arguments(opt):
    """To private procedures (whose address is taken, so they take PL/M-80's
    convention too), a PUBLIC one, a REENTRANT one and an assembly routine,
    with 0 to 4 arguments.  0.3.x passed only one, except to a PUBLIC or
    REENTRANT procedure."""
    decls = "\n".join(l for l in BODY.splitlines() if l.startswith("cap3:"))
    out = run_plm(PRELUDE + decls + INDIRECT + "\nend t;\n", opt, _module(CALLEES_ASM))
    assert out.replace("\0", "").strip() == INDIRECT_EXPECT.strip()


MOD_A = """ma: do;
declare acc address public;
m3: procedure (a, b, c) address public; declare (a, c) address, b byte;
  acc = acc + 1; return a - c + b; end m3;
m1: procedure (b) byte public; declare b byte; return b xor 0ffh; end m1;
end ma;
"""

MOD_B = """t: do;
mon1: procedure (f, p) external; declare f byte, p address; end mon1;
m3: procedure (a, b, c) address external; declare (a, c) address, b byte; end m3;
m1: procedure (b) byte external; declare b byte; end m1;
declare acc address external;
declare v address;
hexd: procedure (d); declare d byte; d = d and 0fh;
  if d < 10 then call mon1(2, d + '0'); else call mon1(2, d + 37h); end hexd;
ph: procedure (v); declare v address;
  call hexd(shr(high(v), 4)); call hexd(high(v)); call hexd(shr(low(v), 4)); call hexd(low(v));
  call mon1(2, ' '); end ph;
acc = 0;
v = m3(1000h, m1(0f0h), 1);
call ph(v); call ph(acc); call ph(m3(m3(10h, 1, 1), 2, m1(0ffh)));
end t;
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_public_procedures_across_separately_compiled_modules(opt):
    """Two uplm80 modules: a BYTE function, and calls in the arguments."""
    if tools_missing():
        pytest.skip(tools_missing())
    a = Compiler(opt_level=opt).compile(MOD_A, "<a>")
    b = Compiler(opt_level=opt).compile(MOD_B, "<b>")
    out = run_asm(b, a).stdout.replace("\r", "").replace("\0", "").strip()
    assert out == "100E 0001 0012"


@pytest.mark.parametrize("opt", LEVELS)
def test_public_procedures_in_a_multi_file_compile(opt):
    """The same two modules in one compile, where the EXTERNAL and the
    PUBLIC declaration name one procedure."""
    if tools_missing():
        pytest.skip(tools_missing())
    with tempfile.TemporaryDirectory() as d:
        fa, fb, mac = (os.path.join(d, n) for n in ("B.PLM", "A.PLM", "T.MAC"))
        for path, text in ((fa, MOD_B), (fb, MOD_A)):
            with open(path, "w") as fh:
                fh.write(text)
        r = subprocess.run(compile_cmd("-O", str(opt), "-o", mac, fa, fb), capture_output=True,
                           text=True, env=compiler_env(), timeout=60, check=False)
        assert r.returncode == 0, r.stderr
        with open(mac) as fh:
            asm = fh.read()
    out = run_asm(asm).stdout.replace("\r", "").replace("\0", "").strip()
    assert out == "100E 0001 0012"


DRI_MON1 = """t: do;
declare bdos label external;
/* DRI's own sources (CP/M 1.x CCP, LOAD, ED) define their BDOS interface this
   way; it works because the entry leaves C and DE as the caller set them.
   (They write `GO TO 5', which uplm80 does not parse.) */
mon1x: procedure (f, a); declare f byte, a address; go to bdos; end mon1x;
declare fn byte;
fn = 2;
call mon1x(fn, 'O'); call mon1x(fn, 'K');
end t;
"""

BDOS_EQU = """
    public BDOS
BDOS equ 5
    end
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_a_dri_style_mon1_body_sees_c_and_de(opt):
    """`MON1: PROCEDURE (F, A); ... GO TO BDOS; END', called with a function
    number that is not a constant, so the call is not open-coded."""
    out = run_plm(DRI_MON1, opt, BDOS_EQU)
    assert out.replace("\0", "").strip() == "OK"


# ---- Intel's own code ----------------------------------------------------------
#
# tests/fixtures/plm80_v31 holds T2.PLM, a module of procedures with two to
# five parameters of every mix of types, and T2.LST, Intel's PL/M-80 V3.1
# listing of it with the code, transcribed into t2_procs.asm (the procedures)
# and t2_main.asm (MAIN, which calls each).  ul80 cannot read Intel's object
# format, so this is as close as a test gets to linking with PL/M-80 objects:
# Intel's MAIN calls uplm80's procedures, and uplm80's MAIN calls Intel's.

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "plm80_v31")


def _fixture(name: str) -> str:
    with open(os.path.join(FIXTURES, name)) as fh:
        return fh.read()


def _t2_parts() -> tuple[str, str]:
    """T2.PLM's procedures P2BB ... P5, and its MAIN."""
    text = _fixture("T2.PLM")
    start = text.index("P2BB:")
    main = text.index("MAIN:")
    return text[start:main], text[main:text.index("END MAIN;") + len("END MAIN;")]


def _t2_program(procs: str, main: str) -> str:
    """A program with T2's variables, public for the assembly, that sets
    them, calls MAIN and prints them; SHOW prints them too."""
    return PRELUDE[:PRELUDE.index("declare res")].replace("t: do;", "t2run: do;") + """
declare sp0 address;
declare (b1, b2, b3, b4, b5) byte public;
declare (a1, a2, a3, a4, a5) address public;
show: procedure;
  call ph(b1); call ph(b2); call ph(b3); call ph(b4); call ph(b5);
  call ph(a1); call ph(a2); call ph(a3); call ph(a4); call ph(a5);
  call mon1(2, '|');
end show;
""" + procs + main + """
b1 = 12h; b2 = 34h; b3 = 56h; b4 = 78h; b5 = 9ah;
a1 = 1357h; a2 = 2468h; a3 = 369ch; a4 = 48d0h; a5 = 5a5ah;
sp0 = stackptr;
call main;
call show;
call ph(stackptr - sp0);
end t2run;
"""


# Each of MAIN's calls stores its arguments in T2's variables, which the next
# calls pass on, and the last, P5(7, 1234H, B1+B2, A1+A2, 9), overwrites B1,
# A2, B3, A4 and B5.  Checked only at the end, a wrong store by an earlier
# call showed late, or not at all.  So SHOW prints the ten after every call,
# called from the uplm80 side: at the end of each procedure where uplm80
# compiles them (MAIN is then T2's own, or Intel's), and after each call in
# MAIN where Intel compiled them.  Every call but the last passes the
# variables back to themselves, so leaves them as they started.


def _show_at_each_procedures_end(procs: str) -> str:
    """T2's procedures, each calling SHOW when it has stored its arguments."""
    procs, n = re.subn(r"(END P\w+;)", r"CALL SHOW; \1", procs)
    assert n == 11, n
    return procs


def _show_after_each_call(main: str) -> str:
    """T2's MAIN, calling SHOW after each of its calls."""
    main, n = re.subn(r"(CALL P\w+\(.*?\);)", r"\1 CALL SHOW;", main)
    assert n == 12, n
    return main


_T2_START = "0012 0034 0056 0078 009A 1357 2468 369C 48D0 5A5A |"
_T2_END = "0007 0034 0046 0078 0009 1357 1234 369C 37BF 5A5A |"
# Eleven calls that change nothing, the last call, then the program's own
# SHOW after MAIN, and SP where it started.
T2_EXPECT = _T2_START * 11 + _T2_END + _T2_END + "0000"

_T2_EXTERNALS = """
P2BB: PROCEDURE(X, Y) EXTERNAL; DECLARE (X, Y) BYTE; END P2BB;
P2AA: PROCEDURE(X, Y) EXTERNAL; DECLARE (X, Y) ADDRESS; END P2AA;
P2AB: PROCEDURE(X, Y) EXTERNAL; DECLARE X ADDRESS, Y BYTE; END P2AB;
P2BA: PROCEDURE(X, Y) EXTERNAL; DECLARE X BYTE, Y ADDRESS; END P2BA;
P3BBB: PROCEDURE(X, Y, Z) EXTERNAL; DECLARE (X, Y, Z) BYTE; END P3BBB;
P3AAA: PROCEDURE(X, Y, Z) EXTERNAL; DECLARE (X, Y, Z) ADDRESS; END P3AAA;
P3BAB: PROCEDURE(X, Y, Z) EXTERNAL; DECLARE (X, Z) BYTE, Y ADDRESS; END P3BAB;
P3ABA: PROCEDURE(X, Y, Z) EXTERNAL; DECLARE (X, Z) ADDRESS, Y BYTE; END P3ABA;
P4: PROCEDURE(W, X, Y, Z) EXTERNAL; DECLARE (W, Y) BYTE, (X, Z) ADDRESS; END P4;
P4X: PROCEDURE(W, X, Y, Z) EXTERNAL; DECLARE (W, Y) ADDRESS, (X, Z) BYTE; END P4X;
P5: PROCEDURE(V, W, X, Y, Z) EXTERNAL; DECLARE (V, X, Z) BYTE, (W, Y) ADDRESS; END P5;
"""


def _t2_run(opt: int, program: str, extra: str | None) -> str:
    return run_plm(program, opt, extra).replace("\0", "").strip()


@pytest.mark.parametrize("opt", LEVELS)
def test_t2_as_uplm80_compiles_it(opt):
    procs, main = _t2_parts()
    program = _t2_program(_show_at_each_procedures_end(procs), main)
    assert _t2_run(opt, program, None) == T2_EXPECT


@pytest.mark.parametrize("opt", LEVELS)
def test_intels_main_calls_uplm80s_procedures(opt):
    procs, _ = _t2_parts()
    program = _t2_program(_show_at_each_procedures_end(procs),
                          "MAIN: PROCEDURE EXTERNAL; END MAIN;\n")
    assert _t2_run(opt, program, _fixture("t2_main.asm")) == T2_EXPECT


@pytest.mark.parametrize("opt", LEVELS)
def test_uplm80s_main_calls_intels_procedures(opt):
    _, main = _t2_parts()
    program = _t2_program(_T2_EXTERNALS, _show_after_each_call(main))
    assert _t2_run(opt, program, _fixture("t2_procs.asm")) == T2_EXPECT


# STACKPTR in a call's last argument reads SP with what is pushed while that
# argument is evaluated.  PL/M-80 V3.1 compiled this program, and its code
# printed STACKPTR_EXPECT: it pushes the first arguments, and for these calls
# nothing more.  uplm80 pushes BC round the last argument's code where that
# may write B or C, and `stackptr - sp0' calls ??subde, which does not.
STACKPTR_ARGS = PRELUDE[:PRELUDE.index("declare res")] + """
declare (sp0, r, pp) address, bv byte;
p5: procedure (a, b, c, d, e); declare (a, d) byte, (b, c, e) address; r = e; end p5;
p3: procedure (a, b, c); declare (a, b, c) address; r = c; end p3;
p2: procedure (a, b); declare (a, b) address; r = b; end p2;
pb: procedure (a, b); declare a byte, b address; r = b; end pb;
bv = 7; pp = .p2;
sp0 = stackptr;
call p5(1, 2, 3, 4, stackptr - sp0); call ph(r);
call p5(1, 2, 3, 4, stackptr); call ph(r - sp0);
call p3(1, 2, stackptr - sp0); call ph(r);
call p2(4, stackptr - sp0); call ph(r);
call pb(bv, stackptr - sp0); call ph(r);
call p2(4, stackptr - sp0 - 2); call ph(r);
call pp(4, stackptr - sp0); call ph(r);
end t;
"""

STACKPTR_EXPECT = "FFFA FFFA FFFE 0000 0000 FFFE 0000"


@pytest.mark.parametrize("opt", LEVELS)
def test_stackptr_in_the_last_argument_reads_what_plm80s_code_reads(opt):
    out = run_plm(STACKPTR_ARGS, opt)
    assert out.replace("\0", "").strip() == STACKPTR_EXPECT
