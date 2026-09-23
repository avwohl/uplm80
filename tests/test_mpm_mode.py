"""MP/M mode: relocatable page zero, and a three-byte stack setup.

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
    assert "jp\t??BOOT" in asm or "jp ??BOOT" in asm, asm
    assert "jp ??BDOS" in asm or "call\t??BDOS" in asm, asm
    # Every page-zero symbol used has to be declared, or the link fails.
    for name in ("??BDOS", "??BOOT"):
        assert f"extrn\t{name}" in asm, f"{name} not declared:\n{asm}"


def test_mpm_mode_declares_only_what_it_uses():
    """A program with no BDOS call must not drag in an undefined BDOS."""
    asm = Compiler(mode=Mode.MPM).compile("t: do; declare x byte; x = 1; end t;",
                                          "<test>")
    assert asm is not None
    assert "extrn\t??BOOT" in asm
    assert "extrn\t??BDOS" not in asm, asm


def test_mpm_mode_sets_the_stack_in_one_three_byte_instruction():
    """DRI's sources enter themselves at `.start-3'.

    They declare

        declare jump byte data (0c3h), jadr address data (.start-3);

    so the program is entered by a jump to three bytes in front of its first
    statement, and that only works when the stack setup is a single three-byte
    instruction - which is what PL/M-80 emitted, `LXI SP,stack'.  CP/M mode's
    `LHLD 0006H / SPHL' is four bytes, so the jump would land one byte inside
    the LHLD operand and SP would never be set.
    """
    asm = _compile(Mode.MPM)
    lines = [l for l in asm.splitlines() if l.strip() and not l.strip().startswith(";")]
    i = next(i for i, l in enumerate(lines) if "sp," in l)
    assert lines[i].split() == ["ld", "sp,??STACK"], lines[i]
    # and nothing else between it and the first statement
    assert "??STACK:" in asm and "ds\t512" in asm, asm


def test_cpm_mode_keeps_its_four_byte_maximum_stack_setup():
    """CP/M mode is unchanged: the stack comes from the BDOS pointer."""
    asm = _compile(Mode.CPM)
    assert "hl,(6)" in asm and "sp,hl" in asm, asm
    assert "??STACK" not in asm, asm


def test_bare_mode_has_no_page_zero_externs():
    asm = Compiler(mode=Mode.BARE).compile(SRC, "<test>")
    assert asm is not None
    for name in ("??BDOS", "??BOOT", "??MAXB"):
        assert f"extrn\t{name}" not in asm


DRI_ENTRY_SRC = """
t: do;
declare jump byte data (0c3h),
        jadr address data (.start-3);
mon1: procedure (f,a) external; declare f byte; declare a address; end mon1;
start:
  do;
  call mon1 (9,.('HI$'));
  end;
end t;
"""


def test_dri_start_minus_three_entry_reaches_the_stack_setup():
    """End to end: compile, assemble, link, and follow the entry jump.

    This is the shape every MP/M II utility has. The first three bytes of the
    image are `JMP start-3', and what sits there has to be the instruction that
    sets SP. Getting this wrong is invisible in the assembly and fatal at run
    time: the program runs on whatever SP it inherited and eventually walks its
    stack out of its own memory segment.
    """
    import os
    import shutil
    import subprocess
    import sys
    import tempfile

    import pytest

    for tool in ("um80", "ul80"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} not installed")

    with tempfile.TemporaryDirectory() as d:
        plm = os.path.join(d, "T.PLM")
        with open(plm, "w") as f:
            f.write(DRI_ENTRY_SRC)
        mac, rel, prl = (os.path.join(d, n) for n in ("T.MAC", "T.REL", "T.PRL"))
        run = lambda *a: subprocess.run(a, capture_output=True, text=True)

        r = run(sys.executable, "-m", "uplm80.compiler", "--mode", "mpm", "-o", mac, plm)
        assert r.returncode == 0, r.stderr
        r = run("um80", "-o", rel, mac)
        assert r.returncode == 0, r.stderr
        # The page-zero symbols and MON1 come from the runtime, as they do in
        # a real build; a minimal stand-in is enough to close the link.
        pz, pzrel = os.path.join(d, "PZ.MAC"), os.path.join(d, "PZ.REL")
        with open(pz, "w") as f:
            f.write("??BDOS\tEQU 5\n??BOOT\tEQU 0\n??MAXB\tEQU 6\n"
                    "\tPUBLIC ??BDOS,??BOOT,??MAXB\n"
                    "\tCSEG\nMON1:\tRET\n\tPUBLIC MON1\n\tEND\n")
        r = run("um80", "-o", pzrel, pz)
        assert r.returncode == 0, r.stderr
        r = run("ul80", "--prl", "-o", prl, rel, pzrel)
        assert r.returncode == 0, r.stderr

        with open(prl, "rb") as f:
            image = f.read()
        code = image[256:]
        assert code[0] == 0xC3, f"entry is not a JMP: {code[:3].hex()}"
        target = code[1] | (code[2] << 8)
        # The image is linked at 0100H, so the target indexes it from there.
        at = code[target - 0x100:target - 0x100 + 3]
        assert at[0] == 0x31, (
            f"JMP start-3 lands on {at.hex(' ')}, not on LXI SP,nn; "
            "the stack setup is not three bytes wide"
        )
