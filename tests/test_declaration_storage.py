"""Storage and initialisers for declarations that MP/M II's sources rely on.

Every case here was found by running the MP/M II utilities: each one produced a
program that assembled and linked cleanly and then wrote through a null or
short pointer at run time.
"""

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
    # 8 characters plus 4 bytes = 12 elements, so LAST is 11.
    assert "ld\thl,11" in lines, asm
    assert "ld\thl,-2" not in lines, asm


def test_last_of_a_fixed_array_is_unchanged():
    asm = _asm("t: do; declare a (15) byte, i byte, n byte; n=0; "
               "do i = 0 to last(a); n=n+1; end; end t;", opt=0)
    assert "ld\thl,14" in [l.strip() for l in asm.splitlines()], asm


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
