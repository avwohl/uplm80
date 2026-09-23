"""Storage and initialisers for declarations that MP/M II's sources rely on.

Every case here was found by running the MP/M II utilities: each one produced a
program that assembled and linked cleanly and then wrote through a null or
short pointer at run time.
"""

import pytest

from uplm80.codegen import Mode
from uplm80.compiler import Compiler


def _asm(src: str, mode: Mode = Mode.CPM, opt: int = 2) -> str:
    out = Compiler(mode=mode, opt_level=opt).compile(src, "<test>")
    assert out is not None, "compilation failed"
    return out


def _data_lines(asm: str) -> list[str]:
    """The data segment, as a list of stripped non-comment lines."""
    tail = asm.split("; Data segment", 1)[1]
    return [l.strip() for l in tail.splitlines() if l.strip() and not l.strip().startswith(";")]


STRUCT_SRC = """
t: do;
declare buf (4) byte;
declare rec structure (
    st address, sa address, ta address,
    ty byte, ln byte, lv byte, nx byte)
    initial (0,.buf(0),.buf(2),0,0,0,0);
declare after byte initial (0AAH);
end t;
"""


def test_structure_initial_uses_each_member_width():
    """A STRUCTURE initialiser has one value per member, at the member's width.

    UTIL7/DM.PLM declares a seven-member, ten-byte parser control block with
    INITIAL. Emitting every value as a byte made it five bytes long and left
    the two pointers at zero, so SDIR scanned its command line through a null
    pointer and blanked the BDOS entry in page zero.
    """
    lines = _data_lines(_asm(STRUCT_SRC))
    i = lines.index("REC:")
    body = lines[i + 1:i + 8]
    assert body == ["dw\t0", "dw\tBUF", "dw\tBUF+2",
                    "db\t0", "db\t0", "db\t0", "db\t0"], body


def test_structure_initial_reserves_the_whole_structure():
    """What the value list does not fill is still reserved."""
    src = STRUCT_SRC.replace("initial (0,.buf(0),.buf(2),0,0,0,0)", "initial (0,.buf(0))")
    lines = _data_lines(_asm(src))
    i = lines.index("REC:")
    # 2 + 2 emitted, 6 of the 10 bytes left to reserve.
    assert lines[i + 1:i + 4] == ["dw\t0", "dw\tBUF", "ds\t6"], lines[i + 1:i + 4]


def test_structure_without_initial_is_unchanged():
    # `r' is mangled to @R because R names a register.
    lines = _data_lines(_asm("t: do; declare r structure (p address, q byte); end t;"))
    assert "@R:\tds\t3" in lines, lines


@pytest.mark.parametrize("expr,want", [(".buf(0)", "dw\tBUF"),
                                       (".buf(2)", "dw\tBUF+2"),
                                       (".buf", "dw\tBUF")])
def test_subscripted_address_of_in_data(expr, want):
    """`.name(n)' points at the n-th element; DRI's sources use it."""
    asm = _asm(f"t: do; declare buf (4) byte; declare p address data ({expr}); end t;")
    assert want in [l.strip() for l in asm.splitlines()], asm


def test_at_memory_names_the_linker_symbol_not_a_local_label():
    """AT(.MEMORY) is the first free byte after the whole PROGRAM.

    A label at the end of this module marks the end of the MODULE, which in a
    program linked from several of them is somewhere in the middle. MP/M II's
    SDIR is eight modules, and its 128-entry hash table — declared
    `AT (.MEMORY)` in UTIL7/DSE.PLM — landed on top of another module's
    strings and cleared them, so the directory header printed as NULs.
    __END__ is the linker's own symbol, so it is named as an external.
    """
    asm = _asm("""
t: do;
declare i byte;
declare tbl (4) address at (.memory);
tbl(0) = 1;
end t;
""")
    lines = [l.strip() for l in asm.splitlines()]
    assert "extrn\t__END__" in lines, asm
    assert "__END__:" not in lines, "a local label makes this the module end, not the program end"
    assert "TBL:\tEQU\t__END__" in lines, asm


def test_at_external_can_be_referenced_before_it_is_declared():
    """UTIL5/SUB.PLM initialises a structure with `.a$buff' and declares
    `a$buff ... AT(.tbuff)' further down the same DECLARE."""
    asm = _asm("""
t: do;
declare tbuff (1) byte external;
declare pfcb structure (a address, b address) initial (.ab, .ab);
declare ab (128) byte at (.tbuff);
end t;
""")
    lines = [l.strip() for l in asm.splitlines()]
    assert any(l.startswith("AB:") and "EQU" in l and "TBUFF" in l for l in lines), asm


def test_an_unplaceable_initial_value_is_reported_not_dropped():
    """A value the emitter cannot place used to vanish silently.

    That is how `.a$buff' disappeared from UTIL5/SUB.PLM's structure: the
    declaration came out shorter than it was declared, so everything after it
    in the data segment moved. Failing the compile is the safe answer.
    """
    out = Compiler().compile(
        "t: do; declare buf (4) byte; declare p address data (buf(0)); end t;",
        "<test>")
    assert out is None, out


def test_structure_local_gets_its_real_size_in_shared_storage():
    """A procedure-local STRUCTURE is sized from its members.

    A STRUCTURE has no data type of its own, and the shared-storage allocator
    fell back to ADDRESS — two bytes, however many members it had. The frame
    was then short, the next procedure's frame was overlaid inside it, and
    member arithmetic (which uses the true size) could store past the end of
    the shared block entirely.
    """
    asm = _asm("""
t6: do;
  declare r address;
  q: procedure; declare (v,w,x,y) address; v=1;w=2;x=3;y=4; r=v+w+x+y; end q;
  p: procedure;
     declare arr (6) structure (a address, b address);
     arr(5).b = 99;
     call q;
  end p;
  call p;
end t6;
""")
    lines = [l.strip() for l in asm.splitlines()]
    # arr is 6 * (2+2) = 24 bytes, so q's frame cannot start below 24.
    assert "ds\t32" in lines, [l for l in lines if l.startswith("ds")]
    assert "ld\t(??AUTO+24),hl" in lines, "q's frame overlaps arr"
    # and the highest member store stays inside the block
    assert "ld\thl,??AUTO+0+20" in lines, asm


def test_mpm_bdos_call_is_relocatable_in_an_expression_too():
    """Both the statement and the expression form of a BDOS call.

    The expression form kept a literal `call 5', which is not relocatable: it
    only works where the process's memory segment happens to start at zero.
    """
    src = """
t: do;
mon2: procedure (f,a) byte external; declare f byte; declare a address; end mon2;
declare c byte, r byte;
r = mon2(2,c) + 1;
call mon2(2,c);
end t;
"""
    mpm = _asm(src, Mode.MPM)
    assert "call\t5" not in mpm and "jp 5" not in mpm, mpm
    assert mpm.count("call\t??BDOS") == 2, mpm
    # CP/M keeps the literal, which is right for a .COM.
    cpm = _asm(src, Mode.CPM)
    assert "??BDOS" not in cpm, cpm


def test_at_accepts_a_constant_expression_subscript():
    """`AT(.arr(6dh-5ch))' is a constant, just not a bare literal.

    Only a NumberLiteral was understood; anything else fell through to an
    `EQU $' - the assembler's location counter - which pointed the variable at
    whatever happened to be there. MP/M II's UTIL4/STAT.PLM declares

        dolla literally '.fcb(6dh-5ch)',
        doll byte at (dolla),

    to reach the second FCB, read a stray byte as its `$' parameter and so took
    every `stat <file>' for a request to change the file's attributes.
    """
    # `fcb' is declared after `doll', exactly as in STAT.PLM.
    asm = _asm("""
t: do;
declare
    dolla literally '.fcb(6dh-5ch)',
    doll byte at(dolla),
    z byte;
declare fcb (1) byte external;
z = doll;
end t;
""", opt=0)
    lines = [l.strip() for l in asm.splitlines()]
    assert "DOLL:\tEQU\tFCB+17" in lines, asm
    assert not any("EQU\t$" in l for l in lines), asm


def test_at_accepts_a_structure_member_designator():
    """`AT(.DEST.FCB(33))' and `AT(.buffer(0).sector(1))'.

    Both are in DRI's sources - UTIL6/PIP.PLM and UTIL5/PRLCM.PLM - and both
    used to become `EQU $'.
    """
    asm = _asm("""
t: do;
declare dest structure (fcb (36) byte, user byte);
declare destr address at (.dest.fcb(33));
declare z address;
z = destr;
end t;
""")
    lines = [l.strip() for l in asm.splitlines()]
    assert "DESTR:\tEQU\tDEST+33" in lines, asm


def test_a_variable_shadows_a_condition_flag_builtin():
    """CARRY, ZERO, SIGN and PARITY are ordinary words.

    UTIL4/STAT.PLM declares `(d,zero) byte' for its zero-suppression flag.
    Reading the Z flag in its place made every number print with leading
    zeros.
    """
    asm = _asm("t: do; declare (d,zero) byte; declare r byte; zero = 0; r = zero; end t;",
               opt=0)
    lines = [l.strip() for l in asm.splitlines()]
    assert "ld\ta,(ZERO)" in lines, asm
    assert "ld\ta,0ffh" not in lines, asm


def test_the_flag_builtin_still_works_when_nothing_declares_it():
    asm = _asm("t: do; declare r byte; r = zero; end t;", opt=0)
    lines = [l.strip() for l in asm.splitlines()]
    assert "ld\ta,0ffh" in lines, asm


def test_an_implicitly_dimensioned_data_array_knows_its_extent():
    """`DECLARE x (*) BYTE DATA (...)' takes its extent from the data.

    The parser marks `(*)' as -1 and that was left on the symbol, so LAST(x)
    came out as -2. UTIL6/PIP.PLM declares its delimiter table that way:

        DECLARE DEL(*) BYTE DATA (' =.:;,<>',CR,LA,LB,RB);
        DO I = 0 TO LAST(DEL);

    so PIP recognised no delimiter at all and answered "INVALID FORMAT" to
    every command.
    """
    asm = _asm("""
t: do;
d: procedure (c) byte;
   declare (i,c) byte;
   declare del(*) byte data (' =.:;,<>',13,10,91,93);
   do i = 0 to last(del);
     if c = del(i) then return 0ffh;
     end;
   return 0;
   end d;
declare r byte;
r = d('=');
end t;
""", opt=0)
    lines = [l.strip() for l in asm.splitlines()]
    # 8 characters plus 4 bytes = 12 elements, so LAST is 11.
    assert "ld\thl,11" in lines, asm
    assert "ld\thl,-2" not in lines, asm


def test_last_of_a_fixed_array_is_unchanged():
    asm = _asm("t: do; declare a (15) byte, i byte, n byte; n=0; "
               "do i = 0 to last(a); n=n+1; end; end t;", opt=0)
    assert "ld\thl,14" in [l.strip() for l in asm.splitlines()], asm
