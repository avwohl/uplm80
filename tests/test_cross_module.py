"""Things that only go wrong once a program is more than one module.

MP/M II's SDIR is linked from eight separately-compiled PL/M modules, and each
of these produced a program that built cleanly and then misbehaved at run time.
"""

from uplm80.codegen import Mode
from uplm80.compiler import Compiler


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
