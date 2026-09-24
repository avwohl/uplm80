"""Storage and initialisers for declarations that MP/M II's sources rely on.

Every case here was found by running the MP/M II utilities: each one produced a
program that assembled and linked cleanly and then wrote through a null or
short pointer at run time.
"""

import os
import re

import pytest

from uplm80.codegen import Mode
from uplm80.compiler import Compiler

from ._toolchain import run_plm


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
    # 8 characters plus 4 bytes = 12 elements, so LAST is 11: a BYTE, loaded
    # into A, or the loop's limit (compared with 11 + 1, or 12 passes).
    assert {"ld\thl,11", "ld\ta,0BH", "cp\t0CH", "ld\tb,0CH"} & set(lines), asm
    assert "ld\thl,-2" not in lines, asm


def test_last_of_a_fixed_array_is_unchanged():
    asm = _asm("t: do; declare a (15) byte, i byte, n byte; n=0; "
               "do i = 0 to last(a); n=n+1; end; end t;", opt=0)
    lines = {l.strip() for l in asm.splitlines()}
    # 15 elements, so LAST is 14: loaded into A, compared with 14 + 1 as the
    # loop's limit, or the count of 15 passes (into A to be stored as the
    # index's final value, or straight into B).
    assert {"ld\thl,14", "ld\ta,0EH", "cp\t0FH", "ld\ta,0FH", "ld\tb,0FH"} & lines, asm


def test_a_string_in_a_structure_initial_fills_one_member_per_character():
    """`'WXYZ'' supplies four BYTE scalars, not one member.

    The width of each value was looked up by the value's position in the list
    instead of by the scalar it lands in, so everything after a string was
    emitted at the width of the wrong member.  UTIL2/SCRSP.PLM and MSRSP.PLM
    initialise their queue control blocks with the queue's name,

        declare sched$lqcb structure (lqueue, buf (71) byte)
          initial (0,'Sched   ',69,1);

    and msglen and nmbmsgs, both ADDRESS, came out as two bytes.
    """
    lines = _data_lines(_asm("""
t: do;
declare q structure (a address, n (4) byte, c address, d address)
    initial (1, 'WXYZ', 3, 4);
declare after byte initial (0AAH);
end t;
"""))
    i = lines.index("Q:")
    assert lines[i + 1:i + 6] == ["dw\t1", "db\t'WXYZ'", "dw\t3", "dw\t4", "AFTER:"], \
        lines[i + 1:i + 6]


def test_a_string_that_runs_into_an_address_member_fills_it_two_characters_at_a_time():
    """PL/M-80 Programming Manual, 6.2.9: one character to each BYTE scalar
    and two to each ADDRESS scalar.  An odd character left over still takes a
    whole ADDRESS scalar, so the next value lands on the next member."""
    lines = _data_lines(_asm("""
t: do;
declare q structure (n (2) byte, w address, x address, y byte)
    initial ('ABCDE', 7);
end t;
"""))
    i = lines.index("Q:")
    # A,B -> n; C,D -> w; E -> x (and a zero byte); 7 -> y.
    assert lines[i + 1:i + 4] == ["db\t'ABCDE'", "db\t0", "db\t7"], lines[i + 1:i + 4]


def _layout(asm: str) -> dict[str, int]:
    """Offset of every label in the module's leading DATA, from its first byte.

    Counts what each db/dw/ds line puts in the image, up to the first line
    that is neither a label nor one of those.
    """
    offsets: dict[str, int] = {}
    at = 0
    for raw in asm.splitlines():
        line = raw.split(";")[0].strip()
        if not line or line == ".z80" or line.startswith(("public", "extrn")):
            continue
        if line.endswith(":"):
            offsets[line[:-1]] = at
            continue
        op, _, arg = line.partition("\t")
        if op == "ds":
            at += int(arg)
        elif op == "dw":
            at += 2 * len(arg.split(","))
        elif op == "db":
            for piece in re.findall(r"'(?:[^']|'')*'|[^,]+", arg):
                at += len(piece[1:-1].replace("''", "'")) if piece.startswith("'") else 1
        else:
            break
    offsets["$"] = at
    return offsets


def test_structure_data_uses_each_member_width_and_reserves_the_whole_structure():
    """DATA is INITIAL stored with the code (PL/M-80 Programming Manual,
    6.2.9), so a STRUCTURE's DATA list is placed member by member and the rest
    of the structure is reserved.  The 0.3.6 fix covered INITIAL only: DATA
    still came out one byte per value and stopped at the last one."""
    asm = _asm("""
t: do;
declare s structure (a address, b byte, c address) data (1, 2, 3);
declare z structure (a address, b (6) byte) data (5);
declare last byte data (0);
end t;
""")
    lines = [l.strip() for l in asm.splitlines()]
    i = lines.index("S:")
    assert lines[i + 1:i + 9] == ["dw\t1", "db\t2", "dw\t3",
                                  "Z:", "dw\t5", "ds\t6",
                                  "LAST:", "db\t0"], lines[i + 1:i + 9]
    where = _layout(asm)
    assert where["Z"] - where["S"] == 5 and where["LAST"] - where["Z"] == 8, where


def test_a_short_data_array_still_has_its_declared_dimension():
    """PL/M-80 Programming Manual, 6.2.9: a list may have fewer values than
    the declaration has scalars, and the rest are left uninitialised - they are
    still part of the variable.  DATA is the same as INITIAL but for where it is
    stored, and INITIAL already reserved them."""
    asm = _asm("""
t: do;
declare x address;
declare tbl (4) address data (.x, 1234H);
declare after byte data (7);
end t;
""")
    lines = [l.strip() for l in asm.splitlines()]
    i = lines.index("TBL:")
    assert lines[i + 1:i + 5] == ["dw\tX", "dw\t1234H", "ds\t4", "AFTER:"], lines[i + 1:i + 5]


# UTIL2/SPRSP.PLM, with the queue and process literals it includes.
SPRSP_SRC = """
spool: do;
declare queueheader literally 'ql address, name(8) byte, msglen address,
    nmbmsgs address, dqph address, nqph address';
declare cqueue literally 'queueheader, msgin address, msgout address, msgcnt address';
declare circularqueue literally 'structure (cqueue, buf (1) byte)';
declare lqueue literally 'queueheader, mh address, mt address, bh address';
declare process$header literally 'structure (pl address, status byte,
    priority byte, stkptr address';
declare bdos$save literally 'disk$set$dma address, disk$slct byte, dcnt address,
    searchl byte, searcha address, drvact address, registers (20) byte,
    scratch (2) byte)';
declare process$descriptor literally 'process$header, name (8) byte,
    console byte, memseg byte, b address, thread address, bdos$save';
declare os address public data (0);
declare spool$pd process$descriptor public
    data (0,0,20,0, 'Sp',0efh,'ol  ', 0a0h,0,0,0);
declare spool$lqcb structure (lqueue, buf (128) byte)
    data (0,'SPOOLQ  ',62,2);
declare stpspl$cqcb circularqueue data (0,'STOPSPLR',0,1);
declare last byte data (0);
end spool;
"""


def test_spool_rsp_lays_out_its_queues_where_dris_binary_has_them():
    """The resident half of MP/M II's spooler is nothing but DATA.

    GENSYS and the XDOS find the process descriptor and the queues by their
    offsets, and DRI's SPOOL.RSP has the spooler's queue at 36H, the stop
    queue at 0CEH and `last' at 0E7H - an image 0E8H bytes long.  Emitting
    every value as a byte and stopping at the last one put them at 11H, 1CH
    and 27H.
    """
    where = _layout(_asm(SPRSP_SRC))
    assert (where["SPOOLPD"], where["SPOOLLQCB"], where["STPSPLCQCB"], where["LAST"], where["$"]) \
        == (0x02, 0x36, 0xCE, 0xE7, 0xE8), where


def test_a_literally_list_stands_for_the_whole_list_inside_initial():
    """LITERALLY is text substitution, in an INITIAL list as anywhere else.

    A special case kept only the body's first element there, to match the
    output of an earlier uplm80 that parsed a macro body as one expression.
    """
    lines = _data_lines(_asm("""
t: do;
declare fill literally '0C7C7H,0C7C7H,0C7C7H';
declare stk (4) address initial (fill, 1234H);
declare after byte initial (0AAH);
end t;
"""))
    i = lines.index("STK:")
    assert lines[i + 1:i + 6] == ["dw\t0C7C7H"] * 3 + ["dw\t1234H", "AFTER:"], lines[i + 1:i + 6]


def test_a_resident_process_stack_holds_its_restarts_and_its_entry():
    """UTIL2/SCBRS.PLM, MSBRS.PLM and SPBRS.PLM build their process stack as

        declare sched$stk (20) address initial (restarts,.sched);
        declare sched$stack$pointer address data (.sched$stk+38);

    with `restarts' nineteen 0C7C7H words: the initial SP points at the
    twentieth word, which holds the process's entry point.  DRI's SCHED.BRS
    has exactly that.  With only the first 0C7C7H the entry point sat in the
    second word and the one SP pointed at was zero.
    """
    restarts = ",".join(["0C7C7H"] * 19)
    lines = _data_lines(_asm(f"""
sched: do;
declare restarts literally '{restarts}';
declare stkp address data (.stk+38);
declare stk (20) address initial (restarts,.sched);
sched: procedure; end sched;
end sched;
"""))
    i = lines.index("STK:")
    assert lines[i + 1:i + 21] == ["dw\t0C7C7H"] * 19 + ["dw\tSCHED"], lines[i + 1:i + 21]


def _defs(asm: str) -> list[str]:
    return [" ".join(l.split()) for l in asm.splitlines() if "EQU" in l]


def test_at_an_external_minus_a_constant_is_that_address():
    """`AT (.ext - 1)' - a location reference and a constant, the restricted
    expression the PL/M-80 manual (6.2.8) allows.  It fell past every case
    `_emit_at_decl' knew into a catch-all `EQU $', the location counter.

    UTIL5/MSPL.PLM builds the spooler's message one byte below the command
    tail, `spool$msg (1) byte at (.tbuff-1)', and SPOOL.PRL wrote it over the
    queue control block declared after it.
    """
    asm = _asm("""
t: do;
declare ext (4) byte external;
declare m (1) byte at (.ext-1);
declare y byte;
y = m(0);
y = m(3);
end t;
""", opt=0)
    assert "@M: EQU EXT-1" in _defs(asm), _defs(asm)
    assert not any("$" in d.split()[-1] for d in _defs(asm)), _defs(asm)
    lines = [l.strip() for l in asm.splitlines()]
    # The references name the external with one offset, not `EXT-1+3'.
    assert "ld\thl,EXT-1" in lines and "ld\thl,EXT+2" in lines, asm


@pytest.mark.parametrize("expr,want", [
    (".buf+128", "X: EQU BUF+128"),
    (".buf(2)+3-1", "X: EQU BUF+4"),
    ("3+.buf(1)", "X: EQU BUF+4"),
    ("5CH+1", "X: EQU 5DH"),
    (".w(1)", "X: EQU W+2"),
])
def test_at_accepts_every_restricted_expression(expr, want):
    asm = _asm(f"t: do; declare buf (200) byte, w (4) address; declare x byte at ({expr}); "
               "declare y byte; y = x; end t;")
    assert want in _defs(asm), _defs(asm)


@pytest.mark.parametrize("expr", [".buf+y", ".buf*2", "y", ".buf-.w"])
def test_an_at_that_is_not_a_constant_address_is_an_error(expr):
    """Never `EQU $'."""
    out = Compiler().compile(
        f"t: do; declare buf (4) byte, w (4) byte, y byte; declare x byte at ({expr}); "
        "y = x; end t;", "<test>")
    assert out is None, out


def test_at_a_variable_declared_further_down_uses_that_declaration():
    """PL/M-80 wants an AT's variable declared first; DRI's compiler did not
    insist (UTIL4/STAT.PLM's `.fcb(6dh-5ch)' comes before fcb).  A subscript
    or member of such a variable is measured from its own declaration, not
    taken to be a byte."""
    asm = _asm("""
t: do;
declare x address at (.later(2)), y byte at (.rec.b);
declare later (4) address, rec structure (a address, b byte);
declare z address; z = x + y;
end t;
""")
    assert "X: EQU LATER+4" in _defs(asm) and "Y: EQU REC+2" in _defs(asm), _defs(asm)


def test_at_a_negative_subscript_is_a_negative_offset():
    """`AT (.tbuff(-1))' is TBUFF-1.  The index was taken modulo 65536 and
    written `TBUFF+65535', which um80 0.3.48 assembles without the external's
    relocation; one signed offset is what every assembler reads right."""
    asm = _asm("""
t: do;
declare tbuff (4) byte external;
declare m (1) byte at (.tbuff(-1));
declare w (2) address external;
declare n address at (.w(-2));
declare y byte, z address;
y = m(0);
y = m(1);
z = n;
end t;
""")
    assert "65535" not in asm and "65532" not in asm, asm
    lines = [l.strip() for l in asm.splitlines()]
    # m(0), m(1) and n name the externals with one signed offset each.
    assert "ld\thl,TBUFF-1" in lines and "ld\thl,TBUFF" in lines, asm
    assert "ld\thl,(W-4)" in lines, asm


NEGATIVE_SUBSCRIPT_SRC = """
0100H:
t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
declare pad (4) byte;
declare buf (256) byte;
putc: procedure (ch); declare ch byte; call mon1(2, ch); end putc;
chk: procedure (ok); declare ok byte;
    if ok then call putc('Y'); else call putc('N');
end chk;
/* an ADDRESS subscript of 0FFFFH is the byte before buf */
pad(3) = 5;
call chk(.buf(0ffffh) = .pad + 3);
call chk(buf(0ffffh) = 5);
buf(0ffffh) = 9; call chk(pad(3) = 9);
/* -1 is the BYTE 0 - 1, 0FFH, so buf(-1) is buf(255) */
call chk(.buf(-1) = .buf + 255);
buf(255) = 7; call chk(buf(-1) = 7);
buf(-1) = 8; call chk(buf(255) = 8);
end t;
"""


@pytest.mark.parametrize("opt", [0, 2])
def test_a_negative_subscript_is_below_the_array_when_run(opt):
    """A subscript of 0FFFFH is the element before the array, and every level
    agrees on it.  At -O0 the BYTE-index path took A for an index that had
    come out in HL, so what `buf(-1)' addressed depended on the level.

    `-1' itself is the BYTE 0 - 1 (PL/M-80 manual, 4.2.2), 0FFH, as
    tests/test_expression_types.py has it for MEMORY(-1): `buf(-1)' is
    buf(255), at every level."""
    assert run_plm(NEGATIVE_SUBSCRIPT_SRC, opt).strip() == "YYYYYY", opt


def test_at_a_variable_declared_further_down_is_defined_after_it():
    """An EQU is evaluated where it stands, and um80 0.3.48 takes a symbol it
    has not reached yet as zero.  UTIL5/SUB.PLM declares

        declare rbuff(1) byte at (.minimum$buffer), ...
        ...
        declare minimum$buffer (1024) byte;

    so SUBMIT built its command file at address 0.  The AT definitions now
    follow all the storage they can name.
    """
    asm = _asm("""
t: do;
declare rbuff (1) byte at (.minimum$buffer), rbp address;
p: procedure; declare loc (8) byte; declare l3 byte at (.loc(3)); l3 = 1; end p;
declare minimum$buffer (1024) byte;
rbuff(0) = 0ffh;
call p;
end t;
""")
    lines = [l.strip() for l in asm.splitlines()]
    where = {l.split(":")[0]: i for i, l in enumerate(lines) if ":" in l and not l.startswith(";")}
    rb = next(i for i, l in enumerate(lines) if l.startswith("RBUFF:") and "EQU" in l)
    assert where["MINIMUMBUFFER"] < rb, asm
    l3 = next(i for i, l in enumerate(lines) if "L3:" in l and "EQU" in l)
    assert where["??AUTO"] < l3, asm


def test_a_forward_at_writes_where_it_should_when_run():
    """The same shape, assembled and run: `rbuff(0)' has to land in
    minimum$buffer, and `l3' in the procedure's `loc'."""
    out = run_plm("""
0100H:
t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
declare rbuff (1) byte at (.minimum$buffer);
p: procedure;
    declare loc (4) byte;
    declare l3 byte at (.loc(3));
    loc(3) = 'C';
    l3 = 'D';
    call mon1(2, loc(3));
end p;
declare minimum$buffer (4) byte;
minimum$buffer(0) = 'A';
rbuff(0) = 'B';
call mon1(2, minimum$buffer(0));
call p;
end t;
""")
    assert out.strip() == "BD", out


def _equ_forward_refs(asm: str) -> list[str]:
    """EQUs whose operand names a label defined further down the file.

    um80 0.3.48 evaluates an EQU where it stands and takes a symbol it has not
    reached yet as zero, so each of these is a wrong address that links."""
    lines = [l.split(";")[0] for l in asm.splitlines()]
    labels = {}
    for i, l in enumerate(lines):
        m = re.match(r"([A-Za-z@?$_][\w@?$]*):", l)
        if m:
            labels.setdefault(m.group(1).upper(), i)
    bad = []
    for i, l in enumerate(lines):
        m = re.match(r"([A-Za-z@?$_][\w@?$]*):?\s+EQU\s+(.*)", l)
        if m:
            for tok in re.findall(r"[A-Za-z@?$_][\w@?$]*", m.group(2)):
                if labels.get(tok.upper(), -1) > i:
                    bad.append(" ".join(l.split()))
    return bad


FORWARD_AT_SRC = """
t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
declare a1 byte at (.b1 + 1);
declare a2 byte at (.m1(3));
declare (a3, a4) byte at (.b2(1));
declare b1 (4) byte at (.buf(2));
declare m1 (8) byte at (.memory);
declare (b0, b2) (2) address at (.buf(1));
declare buf (16) byte;
putc: procedure (ch); declare ch byte; call mon1(2, ch); end putc;
chk: procedure (ok); declare ok byte;
    if ok then call putc('Y'); else call putc('N');
end chk;
call chk(.a1 = .buf + 3);
call chk(.a2 = .memory + 3);
call chk(.a3 = .buf + 7);
call chk(.a4 = .buf + 8);
a1 = 5; call chk(buf(3) = 5);
a4 = 6; call chk(buf(8) = 6);
end t;
"""


def test_an_at_naming_a_later_at_variable_resolves_to_where_that_one_is():
    """`a1 AT (.b1 + 1)' with `b1 AT (.buf(2))' declared further down.  The
    EQU for a1 named b1, and stood above b1's own EQU, so um80 read b1 as 0
    and a1 was at 0001H.  a1 is now defined by what b1 itself stands for:
    buf, with the offsets added up.  The same for a variable further down at
    .MEMORY, and a factored one."""
    asm = _asm(FORWARD_AT_SRC)
    d = _defs(asm)
    assert "A1: EQU BUF+3" in d, d
    assert "A2: EQU __END__+3" in d, d
    assert "A3: EQU BUF+7" in d and "A4: EQU BUF+8" in d, d
    assert not _equ_forward_refs(asm), _equ_forward_refs(asm)


@pytest.mark.parametrize("opt", [0, 2])
def test_an_at_naming_a_later_at_variable_runs(opt):
    assert run_plm(FORWARD_AT_SRC, opt=opt).strip() == "YYYYYY"


FORWARD_EXT_SRC = """
t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
declare a (4) byte at (.e1(1));
declare b byte at (.a(2));
declare e1 (8) byte external;
declare i byte;
putc: procedure (ch); declare ch byte; call mon1(2, ch); end putc;
chk: procedure (ok); declare ok byte;
    if ok then call putc('Y'); else call putc('N');
end chk;
call chk(.a = .e1 + 1);
call chk(.b = .e1 + 3);
call chk(.a(2) = .e1 + 3);
a(2) = 5; call chk(e1(3) = 5);
e1(4) = 6; call chk(a(3) = 6);
i = 1; a(i) = 7; call chk(e1(2) = 7);
end t;
"""

E1_ASM = "\t.z80\n\tpublic E1\n\tds 20h\nE1:\tds 8\n\tend\n"


def test_an_at_naming_a_later_external_names_the_external():
    """`a AT (.e1(1))' with `e1 EXTERNAL' declared further down.  A variable
    at an external is aliased to the external and one offset, because um80
    0.3.48 assembles `@A+2', with `@A EQU E1+1', as E1+2.  That was done only
    for an external already declared; now it is done for one further down."""
    asm = _asm(FORWARD_EXT_SRC)
    lines = [" ".join(l.split()) for l in asm.splitlines()]
    assert not any(re.search(r"@A[+-]", l) for l in lines if "EQU" not in l), asm
    assert "ld (E1+3),a" in lines or "ld hl,E1+3" in lines, asm


@pytest.mark.parametrize("opt", [0, 2])
def test_an_at_naming_a_later_external_runs(opt):
    assert run_plm(FORWARD_EXT_SRC, opt=opt, extra_asm=E1_ASM).strip() == "YYYYYY"


def test_ats_that_name_each_other_are_an_error():
    src = "t: do; declare a byte at (.b); declare b byte at (.c); declare c byte at (.b); end t;"
    assert Compiler().compile(src, "<test>") is None


def test_a_factored_at_places_each_name_after_the_last():
    """PL/M-80 manual, 6.2.8: `DECLARE (CHAR$A, CHAR$B, CHAR$C) BYTE AT
    (.BUFFER)' puts CHAR$B and CHAR$C in the next two bytes.  Every name was
    put at the one address."""
    asm = _asm("t: do; declare buffer (8) byte; declare (a, b, c) address at (.buffer(1)); "
               "declare y address; y = a + b + c; end t;")
    d = _defs(asm)
    assert "@A: EQU BUFFER+1" in d and "@B: EQU BUFFER+3" in d and "@C: EQU BUFFER+5" in d, d


@pytest.mark.parametrize("kind", ["initial", "data"])
def test_a_factored_initialisation_runs_across_the_names(kind):
    """PL/M-80 manual, 6.2.9: `DECLARE (COUNTER, LIMIT, INCR) ADDRESS
    INITIAL (0, 1024, 2)' sets COUNTER to 0, LIMIT to 1024 and INCR to 2.
    Each name got the whole list, so LIMIT and INCR both read 0."""
    asm = _asm(f"""
t: do;
declare (counter, limit, incr) address {kind} (0, 1024, 2);
declare (pp, qq) byte {kind} ('X');
declare after byte {kind} (0AAH);
declare r address; r = counter + limit + incr + pp + qq + after;
end t;
""")
    lines = [l.strip() for l in asm.splitlines()]
    i = lines.index("COUNTER:")
    assert lines[i + 1:i + 4] == ["dw\t0", "dw\t0400H", "dw\t2"], lines[i + 1:i + 6]
    assert "LIMIT:\tEQU\tCOUNTER+2" in lines and "INCR:\tEQU\tCOUNTER+4" in lines, asm
    j = lines.index("PP:")
    assert lines[j + 1:j + 3] == ["db\t'X'", "ds\t1"], lines[j + 1:j + 4]
    assert "QQ:\tEQU\tPP+1" in lines, asm


def test_intel_link_reaches_the_pointer_after_inrecord_p():
    """Intel's LINK - tests/link1a.plm, from Mark Ogden's reconstruction -
    declares `(s, e) ADDRESS AT(.inRecord$p)': s is inRecord$p and e the word
    after it.  With both at inRecord$p, `e = s + inRecord.len + 2' overwrote
    the record pointer."""
    path = os.path.join(os.path.dirname(__file__), "link1a.plm")
    with open(path, encoding="latin-1") as fh:
        asm = Compiler().compile(fh.read(), path)
    assert asm is not None
    d = _defs(asm)
    assert "@GETRECORD$S: EQU INRECORDP" in d and "@GETRECORD$@E: EQU INRECORDP+2" in d, d


@pytest.mark.parametrize("opt", [0, 2])
def test_an_expression_in_data_fills_one_scalar_at_its_width(opt):
    """A value that is an expression was always emitted as a word.  At -O0,
    where nothing folds it first, `x (4) BYTE DATA (68H+80H, k+1, 6)' came
    out `dw / dw / db / ds 1', six bytes, and moved everything declared
    after it; a unary minus was not accepted at all, and UTIL4/SET.PLM did
    not compile at -O0.  A folded -1 in a BYTE is 0FFH, not `db 0FFFFH'."""
    asm = _asm("""
t: do;
declare k literally '5';
declare x (4) byte data (68h+80h, k+1, 6);
declare n (2) byte data (-1, not 0);
declare s structure (a byte, b address, c byte) initial (2+3, k-1, 0ffh and 7);
declare z byte data (.x-1);
declare after byte data (0eeh);
declare q address; q = .after;
end t;
""", opt=opt)
    lines = [" ".join(l.split()) for l in asm.splitlines() if l.strip() and not l.strip().startswith(";")]

    def body(label, n):
        i = lines.index(label + ":")
        return lines[i + 1:i + 1 + n]

    assert body("X", 4) == ["db 0E8H", "db 6", "db 6", "ds 1"], lines
    assert body("N", 2) == ["db 0FFH", "db 0FFH"], lines
    assert body("S", 3) == ["db 5", "dw 4", "db 7"], lines
    assert body("Z", 1) == ["db (X-1)"], lines


CONSTANT_LIST_SRC = """
0100H:
t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
declare msgs (3) address data (.('one$'), .('two$'), .'three$');
declare tbl structure (p address, b byte) initial (.(1, 2, 'XY'), 0AAH);
declare after byte data (0EEH);
declare pb address, b based pb byte;
putc: procedure (ch); declare ch byte; call mon1(2, ch); end putc;
call mon1(9, msgs(1));
call mon1(9, msgs(2));
pb = tbl.p; call putc('0' + b); pb = pb + 1; call putc('0' + b);
pb = pb + 1; call putc(b); pb = pb + 1; call putc(b);
call putc('0' + (tbl.b = 0AAH) + 1);
call putc('0' + (after = 0EEH) + 1);
end t;
"""


@pytest.mark.parametrize("opt", [0, 2])
def test_a_constant_list_in_data_is_its_address(opt):
    """PL/M-80 manual, 4.1.3: `.(constant, ...)' is the location of the
    constants, stored somewhere.  In DATA and INITIAL the constants were laid
    out in place of it, so `msgs (3) ADDRESS DATA (.('one$'), ...)' held the
    characters, and `CALL mon1(9, msgs(2))' printed whatever they pointed
    at.  `.'string'' was not accepted at all."""
    assert run_plm(CONSTANT_LIST_SRC, opt).strip() == "twothree12XY00", opt


@pytest.mark.parametrize("kind", ["initial", "data"])
def test_at_with_initial_or_data_is_an_error_not_dropped(kind):
    """`DECLARE x (2) BYTE AT (.buf) INITIAL (5, 6)' compiled to `X EQU BUF'
    and nothing else: the values were dropped without a word.  They belong
    in someone else's storage, or at an address with none, which this
    compiler cannot lay out; saying so is better than losing them."""
    src = f"t: do; declare buf (4) byte; declare x (2) byte at (.buf) {kind} (5, 6); end t;"
    assert Compiler().compile(src, "<test>") is None
