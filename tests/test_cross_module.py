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


def _asm(src: str, mode: Mode = Mode.CPM) -> str:
    out = Compiler(mode=mode).compile(src, "<test>")
    assert out is not None, "compilation failed"
    return out


def test_a_public_procedure_takes_its_arguments_on_the_stack():
    """A caller in another module cannot name the callee's storage.

    A procedure private to its module is called with its earlier arguments
    already written into its own slots, and only the last in a register. A
    PUBLIC one cannot be: the caller pushes everything. The two conventions
    have to agree, and they did not - SDIR's public `pdecimal(v, prec, zerosup)'
    read two of its three arguments from slots no one had written, so every
    number it printed was wrong or missing.
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
""")
    lines = [l.strip() for l in asm.splitlines()]
    i = lines.index("P:")
    prologue = lines[i + 1:i + 6]
    # Reads its arguments off the stack rather than taking one in a register.
    assert prologue[0] == "ld\thl,6" and prologue[1] == "add\thl,sp", prologue
    # And the caller pushes all three and pops them again.
    assert lines.count("push\thl") >= 3, asm
    assert lines.count("pop\tde") >= 3, asm


def test_a_private_procedure_keeps_the_register_convention():
    """Nothing outside the module can call it, so the cheap form still applies."""
    asm = _asm("""
t: do;
p: procedure (a,b);
   declare a address, b byte;
   declare r address;
   r = a + b;
   end p;
call p(1,2);
end t;
""")
    lines = [l.strip() for l in asm.splitlines()]
    # No stack-reading prologue: `ld hl,<n> / add hl,sp' is its signature.
    assert "add\thl,sp" not in lines, asm
    # The caller writes the earlier argument straight into the callee's slot.
    assert any(l.startswith("ld\t(??AUTO+") and l.endswith("),hl") for l in lines), asm


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
