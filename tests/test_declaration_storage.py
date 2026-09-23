"""Storage and initialisers for declarations that MP/M II's sources rely on.

Every case here was found by running the MP/M II utilities: each one produced a
program that assembled and linked cleanly and then wrote through a null or
short pointer at run time.
"""

import pytest

from uplm80.codegen import Mode
from uplm80.compiler import Compiler


def _asm(src: str, mode: Mode = Mode.CPM) -> str:
    out = Compiler(mode=mode).compile(src, "<test>")
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


def test_at_memory_is_a_label_not_an_equ():
    """AT(.MEMORY) must resolve for references ABOVE the declaration.

    Its address is only known at the end of the file. A forward label
    reference resolves on the assembler's second pass; a forward EQU reads as
    zero, which is what UTIL7/DSE.PLM's hash table did.
    """
    asm = _asm("""
t: do;
declare i byte;
declare tbl (4) address at (.memory);
tbl(0) = 1;
end t;
""")
    lines = [l.strip() for l in asm.splitlines()]
    assert "TBL:" in lines, asm
    assert not any(l.startswith("TBL:") and "EQU" in l for l in lines), asm
    # and it sits at __END__
    assert lines.index("TBL:") == lines.index("__END__:") + 1, asm


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
