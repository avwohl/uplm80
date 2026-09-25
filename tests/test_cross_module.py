"""Things that only go wrong once a program is more than one module.

MP/M II's SDIR is linked from eight separately-compiled PL/M modules, and each
of these produced a program that built cleanly and then misbehaved at run time.
"""

import os
import subprocess
import tempfile

import pytest

from uplm80.codegen import Mode
from uplm80.compiler import Compiler

from ._toolchain import compile_cmd, compiler_env, run_asm, tools_missing


def _asm(src: str, mode: Mode = Mode.CPM, opt: int = 2) -> str:
    out = Compiler(mode=mode, opt_level=opt).compile(src, "<test>")
    assert out is not None, "compilation failed"
    return out


def test_a_public_procedure_takes_its_arguments_as_plm80_passes_them():
    """A caller in another module cannot name the callee's storage.

    Up to 0.3.x a procedure private to its module was called with its
    earlier arguments already written into its own slots, and a PUBLIC one
    with all of them pushed; SDIR's public `pdecimal(v, prec, zerosup)'
    once read two of its three from slots no one had written.  Now every
    call passes them as PL/M-80 does: the last in DE (E), the one before in
    BC (C), the first pushed, and the callee takes the pushed one off the
    stack itself.
    """
    asm = _asm("""
t: do;
p: procedure (a,b,c) public;
   declare a address, b address, c byte;
   declare r address;
   r = a + b + c;
   end p;
call p(1,2,3);
end t;
""", opt=0)
    lines = [l.strip() for l in asm.splitlines()]
    i = lines.index("P:")
    entry = lines[i + 1:i + 8]
    assert entry[:2] == ["ld\ta,e", "ld\t(??AUTO+4),a"], entry
    assert entry[2] == "ld\t(??AUTO+2),bc", entry
    assert entry[3:6] == ["pop\thl", "ex\t(sp),hl", "ld\t(??AUTO+0),hl"], entry
    assert "add\thl,sp" not in lines, asm
    # The caller pushes the first, and pops nothing after the call.
    j = lines.index("call\tP")
    assert lines[j - 4:j + 2] == ["ld\thl,1", "push\thl", "ld\tbc,2", "ld\te,3", "call\tP",
                                  "jp\t0"], lines[j - 5:j + 2]


def test_a_private_procedure_with_one_parameter_keeps_the_register_convention():
    """Nothing outside the module can call it, so it takes its argument in A
    or HL.  One with two parameters takes them in BC and DE, and no caller
    writes into a callee's storage."""
    asm = _asm("""
t: do;
declare (s, w) address, v byte;
p: procedure (a,b);
   declare a address, b byte;
   declare r address;
   r = a + b;
   s = r;
   end p;
q: procedure (a);
   declare a address;
   w = a + 1;
   end q;
k: procedure (b);
   declare b byte;
   v = b + 1;
   end k;
call p(1,2);
call q(w);
call k(v);
end t;
""", opt=0)
    lines = [l.strip() for l in asm.splitlines()]
    # No stack-reading entry: `ld hl,<n> / add hl,sp' is its signature.
    assert "add\thl,sp" not in lines, asm
    # p: BC and DE, stored at its entry.
    i = lines.index("call\tP")
    assert lines[i - 2:i] == ["ld\tbc,1", "ld\te,2"], lines[i - 4:i + 1]
    i = lines.index("P:")
    assert lines[i + 1:i + 6] == ["ld\ta,e", "ld\t(??AUTO+2),a", "ld\th,b", "ld\tl,c",
                                  "ld\t(??AUTO+0),hl"], lines[i + 1:i + 6]
    # q and k: HL and A.
    i = lines.index("call\tQ")
    assert lines[i - 1] == "ld\thl,(W)", lines[i - 2:i + 1]
    assert lines[lines.index("Q:") + 1].startswith("ld\t(??AUTO+") and \
        lines[lines.index("Q:") + 1].endswith("),hl"), asm
    i = lines.index("call\tK")
    assert lines[i - 1] == "ld\ta,(V)", lines[i - 2:i + 1]
    assert lines[lines.index("K:") + 1].startswith("ld\t(??AUTO+") and \
        lines[lines.index("K:") + 1].endswith("),a"), asm
    # Only a procedure's own entry stores into its parameters.
    main = lines[:lines.index("P:")]
    assert not any(l.startswith("ld\t(??AUTO+") for l in main), main


def test_a_variable_by_step_is_actually_used():
    """`DO J = A TO B BY I' must step by I, not by one.

    Only a constant step was read; anything else fell back to one. UTIL7/DSE.PLM
    walks an FCB disk map `BY i', where i is 1 or 2 according to whether the
    disk uses byte or word block numbers, and so counted every allocated block
    on a large disk twice - SDIR reported 8k for a 4k file.
    """
    asm = _asm("""
t: do;
declare i byte, j byte, n byte;
i = 2; n = 0;
do j = 0 to 7 by i;
  n = n + 1;
  end;
end t;
""")
    lines = [l.strip() for l in asm.splitlines()]
    assert "add\ta,b" in lines, f"the step is not added:\n{asm}"
    # and it must not have become a plain counted loop that ignores the step
    assert "ld\tb,8" not in lines, asm


def test_a_constant_by_step_is_unchanged():
    asm = _asm("t: do; declare j byte, n byte; n=0; do j = 0 to 7 by 2; n=n+1; end; end t;")
    assert "add\ta,2" in [l.strip() for l in asm.splitlines()], asm


def test_a_based_variable_over_a_structure_member_uses_that_member():
    """`x BASED s.m' keeps its pointer in a MEMBER of s, not at s.

    The member was parsed and then dropped, so every access read the pointer
    from the start of the structure. MP/M II's UTIL7/DM.PLM declares
    `token BASED pcb.token$adr (12) byte'; reading `pcb.state' in its place
    gave it zero, so SDIR matched its command-line file specification against
    whatever sat at address 0 and answered "File Not Found." to every
    argument it was given.
    """
    asm = _asm("""
t: do;
declare pcb structure (aa address, bb address);
declare tok based pcb.bb (12) byte;
declare r byte;
r = tok(1);
tok(2) = 5;
end t;
""")
    lines = [l.strip() for l in asm.splitlines()]
    assert "ld\thl,(PCB+2)" in lines, asm
    assert "ld\thl,(PCB)" not in lines, asm


def test_a_plain_based_variable_is_unchanged():
    asm = _asm("t: do; declare p address; declare v based p (4) byte; "
               "declare r byte; r = v(1); v(2) = 3; end t;")
    assert "ld\thl,(P)" in [l.strip() for l in asm.splitlines()], asm


def test_a_nested_procedures_return_type_is_known_at_its_use_site():
    """A bare procedure name is a CALL in PL/M-80, and its result has a type.

    A nested procedure's symbol is filed under its scoped name, and the type
    lookup searched only the top level, so everything a nested procedure
    returned was typed ADDRESS. A BYTE result was then read out of L instead of
    A: `if inner then' tested the wrong register, and `inner + 1' did its
    arithmetic on a stale HL.
    """
    asm = _asm("""
t: do;
declare g byte;
outer: procedure;
  declare r byte;
  inner: procedure byte; return 7; end inner;
  if inner then r = 1; else r = 2;
  g = inner + 1;
  end outer;
call outer;
end t;
""")
    lines = [l.strip() for l in asm.splitlines()]
    i = lines.index("OUTER:")
    body = lines[i:i + 20]
    assert "bit\t0,a" in body, f"the condition reads the wrong register:\n{asm}"
    assert "bit\t0,l" not in body, asm
    assert "add\ta,1" in body, f"the arithmetic is not on the BYTE result:\n{asm}"


def test_a_top_level_procedures_return_type_is_unchanged():
    asm = _asm("t: do; declare h byte; top: procedure byte; return 7; end top; "
               "if top then h = 1; else h = 2; end t;")
    assert "bit\t0,a" in [l.strip() for l in asm.splitlines()], asm


def test_a_multi_file_compile_declares_an_external_no_file_defines():
    """`uplm80 A.PLM B.PLM' compiles both modules into one; an EXTERNAL
    procedure that neither defines belongs to a third, and needs its
    `extrn' as it does when A.PLM is compiled alone. The multi-file compile
    left out every EXTERNAL procedure, so um80 stopped at `call BOOTX' with
    "Undefined symbol" (0.3.6 too). One the other file defines, F here,
    still gets none."""
    reason = tools_missing()
    if reason:
        pytest.skip(reason)
    a = """a: do;
mon1: procedure (f, p) external; declare f byte, p address; end mon1;
bootx: procedure external; end bootx;
f: procedure external; end f;
declare x byte public;
x = 1; call f; call mon1(2, '0' + x); call bootx;
end a;
"""
    b = """b: do;
declare x byte external;
f: procedure public; x = 2; end f;
end b;
"""
    third = "\t.z80\n\tpublic\tBOOTX\n\tcseg\nBOOTX:\tld\tc,2\n\tld\te,'K'\n\tjp\t5\n\tend\n"
    with tempfile.TemporaryDirectory() as d:
        pa, pb, mac = (os.path.join(d, n) for n in ("A.PLM", "B.PLM", "AB.MAC"))
        for path, text in ((pa, a), (pb, b)):
            with open(path, "w") as fh:
                fh.write(text)
        r = subprocess.run(compile_cmd("-o", mac, pa, pb), capture_output=True, text=True,
                           timeout=60, env=compiler_env(), check=False)
        assert r.returncode == 0, r.stderr
        with open(mac) as fh:
            asm = fh.read()
    lines = [l.strip() for l in asm.splitlines()]
    assert "extrn\tBOOTX" in lines, asm
    assert "extrn\tF" not in lines, asm
    assert run_asm(asm, third).stdout.replace("\r", "") == "2K"
