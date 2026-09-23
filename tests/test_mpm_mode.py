"""MP/M mode emits page-zero addresses as relocatable symbols.

A CP/M .COM lives at a fixed place, so the compiler can reach page zero with a
literal: ``call 5``, ``ld hl,(6)``, ``jp 0``.  An MP/M .PRL cannot.  MP/M gives
each process a memory segment, loads the transient at segment_bottom+0100H and
puts the process's page zero at segment_bottom, so those three addresses have
to be relocated when the program is loaded.  Only a *resolved symbol* reference
reaches the .PRL relocation bitmap, so in MP/M mode they are emitted as externs
(BDOS/MAXB/BOOT) which the runtime defines - the same thing DRI's X0100.ASM did
for PL/M-80.

Emitting the literal instead links, loads and runs, then calls into whatever
sits at absolute 0005H, which is not the process's BDOS entry.
"""

from uplm80.codegen import Mode
from uplm80.compiler import Compiler

SRC = """
t: do;
mon1: procedure (f,a) external; declare (f,a) address; end mon1;
p: procedure (a); declare a address; call mon1(9,a); end p;
call p(.('HI$'));
end t;
"""


def _compile(mode):
    asm = Compiler(mode=mode).compile(SRC, "<test>")
    assert asm is not None, "compilation failed"
    return asm


def test_cpm_mode_uses_literal_page_zero():
    asm = _compile(Mode.CPM)
    assert "hl,(6)" in asm
    assert "extrn\tBDOS" not in asm
    # The BDOS call is emitted as a literal 5 (tail-merged to a jp).
    assert any(line.split() == ["jp", "5"] or line.split() == ["call", "5"]
               for line in asm.splitlines())


def test_mpm_mode_uses_relocatable_page_zero():
    asm = _compile(Mode.MPM)
    assert "hl,(MAXB)" in asm, asm
    assert "jp\tBOOT" in asm or "jp BOOT" in asm, asm
    assert "jp BDOS" in asm or "call\tBDOS" in asm, asm
    # Every page-zero symbol used has to be declared, or the link fails.
    for name in ("BDOS", "BOOT", "MAXB"):
        assert f"extrn\t{name}" in asm, f"{name} not declared:\n{asm}"


def test_mpm_mode_declares_only_what_it_uses():
    """A program with no BDOS call must not drag in an undefined BDOS."""
    asm = Compiler(mode=Mode.MPM).compile("t: do; declare x byte; x = 1; end t;",
                                          "<test>")
    assert asm is not None
    assert "extrn\tMAXB" in asm and "extrn\tBOOT" in asm
    assert "extrn\tBDOS" not in asm, asm


def test_bare_mode_has_no_page_zero_externs():
    asm = Compiler(mode=Mode.BARE).compile(SRC, "<test>")
    assert asm is not None
    for name in ("BDOS", "BOOT", "MAXB"):
        assert f"extrn\t{name}" not in asm
