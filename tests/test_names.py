"""Every name means the declaration PL/M-80 gives it, and the assembler gets
one label for each.

PL/M-80 scopes a name to the block that declares it (Programming Manual
9800268B, chapter 9), and every DO block and procedure is a block.  Code
generation names a module-level name as it is and a procedure's own
names as `@proc$name', so two declarations PL/M-80 keeps apart could meet
in one assembler name, or a GOTO reach the wrong one of two labels
(uplm80/names.py).
"""

import os
import re
import subprocess
import tempfile

import pytest

from uplm80.compiler import Compiler
from uplm80.errors import CodeGenError
from uplm80.frontend import parse_source
from uplm80.names import check_names, resolve_names

from ._toolchain import compile_cmd, compiler_env, run_asm, run_plm, tools_missing
from .test_expression_types import _PRELUDE as _PH_PRELUDE
from .test_expression_types import _check

LEVELS = (0, 1, 2, 3)

PRELUDE = """0100H:
t: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
pc: procedure (c); declare c byte; call mon1(2, c); end pc;
"""


def _compile(src: str, opt: int = 2) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as d:
        plm = os.path.join(d, "T.PLM")
        with open(plm, "w") as fh:
            fh.write(src)
        return subprocess.run(compile_cmd("-O", str(opt), "-o", os.path.join(d, "T.MAC"), plm),
                              capture_output=True, text=True, timeout=60, env=compiler_env(),
                              check=False)


def _compile_error(src: str, opt: int = 2) -> str:
    """The compiler's stderr for a program it has to reject."""
    with tempfile.TemporaryDirectory() as d:
        plm = os.path.join(d, "T.PLM")
        with open(plm, "w") as fh:
            fh.write(src)
        r = subprocess.run(compile_cmd("-O", str(opt), "-o", os.path.join(d, "T.MAC"), plm),
                           capture_output=True, text=True, timeout=60, env=compiler_env(),
                           check=False)
    assert r.returncode != 0, "compiled a program PL/M-80 does not allow"
    return r.stderr


# ---- GOTO -----------------------------------------------------------------

@pytest.mark.parametrize("opt", LEVELS)
def test_a_goto_out_of_a_nested_procedure_to_its_parent_is_an_error(opt):
    """PL/M-80: "the label in the GOTO must be the label of a statement in
    the outermost level of the main program module" (5.3.2, 8.1.3, 9.3).
    uplm80 emitted `jp @M1$BAIL$OUT', which nothing defines, so the output
    did not assemble (0.3.6 assembled it at -O3 only, by inlining BAIL)."""
    err = _compile_error(PRELUDE + """
m1: procedure (f) byte;
    declare f byte;
    declare v byte;
    bail: procedure; goto out; end bail;
    v = 'V';
    if f then call bail;
    v = 'W';
out:
    return v;
end m1;
call pc(m1(1));
end t;
""", opt)
    assert "T.PLM:9:22: error: GOTO OUT leaves procedure BAIL for a label in procedure M1" in err
    assert "outer level of the main program module" in err and "8.1.3 and 9.3" in err


@pytest.mark.parametrize("opt", LEVELS)
def test_a_goto_out_of_a_procedure_to_the_main_program(opt):
    """What PL/M-80 does allow: a label at the main program's outer level."""
    assert run_plm(PRELUDE + """
declare n byte;
bail: procedure; n = n + 1; goto again; end bail;
n = 0;
again:
call pc('0' + n);
if n < 3 then call bail;
end t;
""", opt) == "0123"


@pytest.mark.parametrize("opt", LEVELS)
def test_a_procedures_label_is_not_the_main_programs_of_the_same_name(opt):
    """A GOTO in a procedure went to a main-program label of the same name
    rather than to the procedure's own: P jumped out to the main program's
    DONE and never printed its letter."""
    assert run_plm(PRELUDE + """
declare i byte;
p: procedure;
  declare j byte;
  j = 0;
  goto done;
  j = 5;
done:
  call pc('a' + j);
end p;
i = 0;
call p;
if i = 0 then goto done;
call pc('X');
done:
call pc('Z');
end t;
""", opt) == "aZ"


@pytest.mark.parametrize("opt", LEVELS)
def test_one_label_in_two_blocks_of_the_main_program(opt):
    """Each DO block is a block, so each LP is its own; both were `LP:',
    "multiply defined"."""
    assert run_plm(PRELUDE + """
declare i byte;
i = 0;
do; lp: i = i + 1; if i < 3 then goto lp; end;
call pc('0' + i);
do; lp: i = i + 1; if i < 6 then goto lp; end;
call pc('0' + i);
end t;
""", opt) == "36"


@pytest.mark.parametrize("opt", LEVELS)
def test_one_label_in_two_blocks_of_a_procedure(opt):
    assert run_plm(PRELUDE + """
p: procedure;
  declare j byte;
  j = 0;
  do; again: j = j + 1; if j < 2 then goto again; end;
  do; again: j = j + 1; if j < 4 then goto again; end;
  call pc('0' + j);
end p;
call p;
end t;
""", opt) == "4"


@pytest.mark.parametrize("opt", LEVELS)
def test_gotos_out_of_loops_to_enclosing_blocks(opt):
    assert run_plm(PRELUDE + """
declare (i, j) byte;
p: procedure;
  declare k byte;
  k = 0;
again:
  k = k + 1;
  do while k < 10;
    do j = 0 to 3;
      if k = 2 then goto again;
      if k = 5 then goto out;
      k = k + 1;
    end;
  end;
out:
  call pc('0' + k);
end p;
i = 0;
top:
  i = i + 1;
  do j = 1 to 2;
    if i < 3 then goto top;
  end;
call p;
call pc('0' + i);
end t;
""", opt) == "53"


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("where, stmt, name, at", [
    ("module", "a = .here;", "HERE", "13:6"),
    ("module", "a = .there + 1;", "THERE", "13:6"),
    ("procedure", "q = .there;", "THERE", "9:9"),
    ("procedure", "if .here <> 0 then q = 1;", "HERE", "9:8"),
])
def test_the_address_of_a_label_in_an_expression_is_an_error(opt, where, stmt, name, at):
    """The dot operator takes a variable or a procedure (4.1.3).  Intel's
    PL/M-80 V3.1 rejects `.label' in an expression, ERROR #158, INVALID DOT
    OPERAND, LABEL ILLEGAL, whether or not the label is declared LABEL and
    whether it is defined before or after; uplm80 compiled it to the
    label's address (0.3.6 the same).  In a DATA or INITIAL list it is
    allowed (below)."""
    src = (PRELUDE + "declare a address;\ndeclare there label;\np: procedure;\n"
           "   declare q address;\n" + (f"   {stmt}\n" if where == "procedure" else "")
           + "   goto inner;\ninner:\n   call pc('P');\nend p;\n"
           + (f"{stmt}\n" if where == "module" else "")
           + "call p;\nhere:\nthere:\ncall pc('M');\nend t;\n")
    err = _compile_error(src, opt)
    assert (f"T.PLM:{at}: error: .{name}: {name} is a label, and the dot operator takes a "
            "variable or a procedure (Programming Manual 9800268B, 4.1.3); the address of "
            "a label may be given only in a DATA or an INITIAL list") in err, err


@pytest.mark.parametrize("opt", LEVELS)
def test_the_address_of_a_label_in_data_and_initial(opt):
    """`.there' in a procedure's DATA or INITIAL list named the bare THERE.
    (`.here' may not be compared with `m' in an expression; see above.)"""
    assert run_plm(PRELUDE + """
declare m address data (.here);
declare m2 address initial (.here);
p: procedure;
   declare t address data (.there);
   declare q address initial (.there);
   if t = q then call pc('=');
   goto there;
there:
   call pc('P');
end p;
call p;
if m = m2 and m <> 0 then call pc('m');
here:
call pc('.');
end t;
""", opt) == "=Pm."


def test_a_declared_label():
    assert run_plm(PRELUDE + """
p: procedure;
  declare l label;
  goto l;
  call pc('X');
l: call pc('L');
end p;
call p;
end t;
""") == "L"


def test_a_goto_into_a_nested_block_is_an_error():
    err = _compile_error(PRELUDE + "declare i byte;\ni = 0;\ngoto inner;\ndo; inner: i = 1; end;\nend t;\n")
    assert "T.PLM:7:1: error: GOTO INNER: the label INNER is in a DO block this GOTO is not in" in err


def test_a_goto_from_a_procedure_into_a_main_program_block_is_an_error():
    err = _compile_error(PRELUDE + """declare i byte;
p: procedure; goto lp; end p;
i = 0;
do; lp: i = i + 1; if i < 3 then call p; end;
end t;
""")
    assert "error: GOTO LP: the label LP is in a DO block this GOTO is not in" in err


def _compile_asm(src: str, opt: int, mode: str = "cpm") -> tuple[subprocess.CompletedProcess, str]:
    """The compiler's result for a program it has to compile, and its assembly."""
    with tempfile.TemporaryDirectory() as d:
        plm, mac = os.path.join(d, "T.PLM"), os.path.join(d, "T.MAC")
        with open(plm, "w") as fh:
            fh.write(src)
        r = subprocess.run(compile_cmd("-O", str(opt), "--mode", mode, "-o", mac, plm),
                           capture_output=True, text=True, timeout=60, env=compiler_env(),
                           check=False)
        assert r.returncode == 0, r.stderr
        with open(mac) as fh:
            return r, fh.read()


DO_BLOCK_GOTO = PRELUDE + """declare n byte;
n = 0;
do;
    bail: procedure; n = n + 1; goto l1; end bail;
l1:
    call pc('0' + n);
    if n < 3 then call bail;
end;
call pc('.');
call mon1(0, 0);
end t;
"""


@pytest.mark.parametrize("mode", ("cpm", "bare"))
@pytest.mark.parametrize("opt", LEVELS)
def test_a_goto_from_a_procedure_to_a_do_block_of_the_main_program_warns(opt, mode):
    """PL/M-80 does not allow it either: a GOTO out of a procedure goes to
    the outer level of the main program module (9.3), the module's
    exclusive extent (10.1), and a DO block is not in that.  But Intel's
    PL/M-80 V3.1 compiles it with no error, BAIL's GOTO to a plain `JMP L1'
    and no `LXI SP' at L1, and so did uplm80 0.3.6; this release's check of
    GOTOs made it an error.  It is a warning, once, and the jump Intel's."""
    reason = tools_missing()
    if reason:
        pytest.skip(reason)
    r, asm = _compile_asm(DO_BLOCK_GOTO, opt, mode)
    warnings = [x for x in r.stderr.splitlines() if "warning:" in x]
    assert len(warnings) == 1, r.stderr
    assert ("T.PLM:8:33: warning: GOTO L1 leaves procedure BAIL for a label in a DO block "
            "of the main program; ") in warnings[0], warnings
    assert "9800268B, 9.3 and 10.1" in warnings[0]
    lines = [" ".join(x.split(";")[0].split()) for x in asm.splitlines()]
    assert "jp L1" in lines or "jr L1" in lines
    after = lines[lines.index("L1:") + 1]
    assert not after.startswith("ld sp") and after != "ld hl,(6)", after
    assert run_asm(asm).stdout.replace("\r", "") == "0123."


def test_a_goto_to_a_variable_is_an_error():
    err = _compile_error(PRELUDE + "declare i byte;\ni = 0;\ngoto i;\nend t;\n")
    assert "error: GOTO I: I is not a label" in err


def test_a_label_defined_twice_in_one_block_is_an_error():
    err = _compile_error(PRELUDE + "declare i byte;\nl: i = 0;\nl: i = 1;\nend t;\n")
    assert "error: label L is defined twice in the same block" in err


# ---- the address of a procedure --------------------------------------------

EXTP = "\t.z80\n\tpublic\tEXTP\n\tcseg\nEXTP:\tld\tc,2\n\tld\te,45H\n\tjp\t5\n\tend\n"


@pytest.mark.parametrize("opt", LEVELS)
def test_the_address_of_every_kind_of_procedure_and_a_call_through_it(opt):
    """`.show' of a procedure nested in another named the bare SHOW, which
    nothing defines: it is filed as OUTER$SHOW, and `.' looked it up only
    by its own name.  And a CALL through an address (8.2.1) did not call:
    `CALL q' was `call Q', which ran the bytes of Q itself, and `CALL s.p'
    a `jp (hl)' with no return address.  Here every kind of procedure -
    nested, outer, PUBLIC, REENTRANT, EXTERNAL - has its address taken in
    an expression and a DATA and an INITIAL list, and is called through
    it, with an argument where it takes one.  (An AT, which this had too,
    V3.1 does not take of a procedure, #211, nor uplm80 since 0.4.4:
    V31_REJECTS.)"""
    src = PRELUDE + """
extp: procedure external; end extp;
declare q address;
declare s structure (p address);
declare mt (3) address data (.top, .extp, .pub);
declare mi address initial (.top);
top: procedure; call pc('T'); end top;
pub: procedure (w) public; declare w address; call pc(low(w)); end pub;
re: procedure (c) reentrant; declare c byte; call pc(c); end re;
outer: procedure;
  declare t (2) address data (.show, .inner2);
  declare r address initial (.show);
  show: procedure; call pc('S'); end show;
  inner2: procedure (c); declare c byte; call pc(c); end inner2;
  q = .show; call q;
  q = t(0); call q;
  q = t(1); call q('I');
  q = r; call q;
  s.p = .show; call s.p;
end outer;
call outer;
call pc('/');
q = .top; call q;
q = mt(0); call q;
q = mt(1); call q;
q = mt(2); call q('P');
q = mi; call q;
q = .pub; call q('p');
q = .re; call q('R');
q = .extp; call q;
s.p = .extp; call s.p;
call pc('.');
end t;
"""
    assert run_plm(src, opt, extra_asm=EXTP) == "SSISS/TTEPTpREE."


def test_a_call_through_an_address_with_two_arguments_does_not_warn():
    """A call through an address passes its arguments as a direct call does
    (uplm80 0.4.0), so any procedure can take any number of them that way.
    0.3.x warned, since a procedure private to its module took all but its
    last argument in its own storage.  (tests/test_calling_convention_run.py
    runs such calls.)"""
    r = _compile(PRELUDE + """declare q address;
pub: procedure (a, b) public; declare (a, b) byte; call pc(a); call pc(b); end pub;
priv: procedure (a, b, c); declare (a, b, c) byte; call pc(a); call pc(c); end priv;
q = .pub;
call q(1, 2);
q = .priv;
call q(1, 2, 3);
end t;
""")
    assert r.returncode == 0, r.stderr
    assert "warning" not in r.stderr, r.stderr



# ---- LITERALLY -------------------------------------------------------------

@pytest.mark.parametrize("opt", LEVELS)
def test_a_literally_in_each_of_two_procedures(opt):
    """Each is the EQU of its name, `K EQU 1' and `K EQU 2': "Symbol 'K'
    multiply defined"."""
    assert run_plm(PRELUDE + """
pa: procedure; declare k literally '1'; call pc('0' + k); end pa;
pb: procedure; declare k literally '2'; call pc('0' + k); end pb;
call pa; call pb;
end t;
""", opt) == "12"


@pytest.mark.parametrize("opt", LEVELS)
def test_a_literally_does_not_reach_a_variable_of_its_name_elsewhere(opt):
    """Code generation kept every LITERALLY in one table, so once PA had
    declared K LITERALLY '1' the variable K of PB read as 1 (silently: the
    program printed 11)."""
    assert run_plm(PRELUDE + """
pa: procedure; declare k literally '1'; call pc('0' + k); end pa;
pb: procedure; declare k byte; k = 5; call pc('0' + k); end pb;
call pa; call pb;
end t;
""", opt) == "15"


@pytest.mark.parametrize("opt", LEVELS)
def test_a_literally_named_like_a_module_variable_or_label(opt):
    """`K EQU 1' and the variable's `K: ds 1', the label's `DONE:'."""
    assert run_plm(PRELUDE + """
declare k byte;
pa: procedure; declare k literally '1', done literally '1'; call pc('0' + k + done); end pa;
k = 5;
call pa; call pc('0' + k);
goto done;
call pc('X');
done: call pc('.');
end t;
""", opt) == "25."



# ---- procedures ------------------------------------------------------------

@pytest.mark.parametrize("opt", LEVELS)
def test_a_procedure_in_each_of_two_blocks(opt):
    """Code generation files a procedure under its parent's name and its
    own, P$Q, whichever block of P declares it; two Qs met there, and in
    the assembly (@P$Q "multiply defined")."""
    assert run_plm(PRELUDE + """
p: procedure;
  do;
    q: procedure; call pc('1'); end q;
    call q;
  end;
  do;
    q: procedure (c); declare c byte; call pc(c); end q;
    call q('2');
  end;
end p;
call p;
end t;
""", opt) == "12"


@pytest.mark.parametrize("opt", LEVELS)
def test_a_procedure_in_a_block_of_the_main_program_named_like_a_variable(opt):
    """Both were the label N."""
    assert run_plm(PRELUDE + """
declare n byte;
n = 7;
do;
  n: procedure; call pc('N'); end n;
  call n;
end;
call pc('0' + n);
end t;
""", opt) == "N7"


@pytest.mark.parametrize("opt", LEVELS)
def test_a_variable_hides_a_procedure_of_an_enclosing_procedure(opt):
    """Q's X is its variable; code generation looks a name up as a
    procedure nested in each enclosing procedure first, and took P's
    procedure X for it (the program printed X0X)."""
    assert run_plm(PRELUDE + """
p: procedure;
  x: procedure; call pc('X'); end x;
  q: procedure;
    declare x byte;
    x = 5;
    call pc('0' + x);
  end q;
  call q;
  call x;
end p;
call p;
end t;
""", opt) == "5X"


@pytest.mark.parametrize("opt", LEVELS)
def test_a_procedure_in_a_block_does_not_reach_past_it(opt):
    """V and Y outside the DO block are the variables; both were taken for
    the procedure the block declares (VV, YY)."""
    assert run_plm(PRELUDE + """
declare y byte;
p: procedure;
  declare v byte;
  do;
    v: procedure; call pc('V'); end v;
    y: procedure; call pc('Y'); end y;
    call v; call y;
  end;
  v = 'v'; y = 'y';
  call pc(v); call pc(y);
end p;
call p;
end t;
""", opt) == "VYvy"


@pytest.mark.parametrize("opt", LEVELS)
def test_a_procedure_a_block_declares_behind_a_variable_of_its_name(opt):
    """Code generation keeps every procedure in its outermost scope, so
    through the scopes around the inner block, where Q is the procedure,
    the outer block's variable Q came first: the procedure was generated
    under the variable's label, @B1$Q, "multiply defined"."""
    assert run_plm(PRELUDE + """
do;
  declare q byte;
  q = 'q';
  do;
    declare e byte;
    q: procedure; call pc('P'); end q;
    call q;
  end;
  call pc(q);
end;
end t;
""", opt) == "Pq"


@pytest.mark.parametrize("opt", LEVELS)
def test_a_call_of_the_outer_of_two_procedures_of_one_name(opt):
    """The call outside the DO block is the module's NUL.  Code generation
    found the block's, P$NUL, first (names renames it now), and -O3 inlined
    the block's there: its table of procedures it may inline is by name,
    and it checked that the names in the body mean the same at the call,
    not that the procedure's own does."""
    assert run_plm(PRELUDE + """
nul: procedure; call pc('M'); end nul;
p: procedure;
  do;
    nul: procedure; call pc('I'); end nul;
    call nul;
  end;
  call nul;
end p;
call p;
end t;
""", opt) == "IM"


@pytest.mark.parametrize("opt", LEVELS)
def test_a_procedure_is_not_inlined_where_a_label_hides_its_variable(opt):
    """-O3 inlined P, which reads the variable Q, into a block with a label
    Q: its model of scope had no labels (the program printed `q***')."""
    assert run_plm(PRELUDE + """
declare q byte;
p: procedure; call pc(q); end p;
q = 'q';
do;
  call p;
  goto q;
  call pc('X');
  q: call p;
end;
do case 0;
  do; call p; goto q; call pc('X'); q: call p; end;
end;
end t;
""", opt) == "qqqq"


@pytest.mark.parametrize("opt", LEVELS)
def test_a_blocks_variables_are_not_a_procedure_b1s(opt):
    """A DO block's variables are named after the block's number, @B1$X;
    a procedure B1's X is @B1$X too, and the block's X took its place in
    ??AUTO (PP)."""
    assert run_plm(PRELUDE + """
b1: procedure; declare x byte; x = 'P'; call pc(x); end b1;
do;
  declare x byte;
  x = 'D';
  call b1;
  call pc(x);
end;
end t;
""", opt) == "PD"


@pytest.mark.parametrize("opt", LEVELS)
def test_a_procedure_or_label_in_a_block_named_like_a_static_parameter(opt):
    """A parameter that is static (here X and Y, declared after A, whose
    address is taken) is @P$X, as a local is; a procedure X or a label Y
    in a DO block of P is @P$X and @P$Y too, and they were "multiply
    defined".  Such a parameter has a label now as far as names.py is
    concerned, and the procedure or label is renamed."""
    assert run_plm(PRELUDE + """
declare keep address, kv based keep byte;
p: procedure (a, x, y);
    declare (a, x, y) byte;
    keep = .a;
    do;
        x: procedure; call pc('!'); end x;
        call x;
    end;
    do;
        goto y;
        call pc('?');
    y:  call pc(a);
    end;
end p;
call p('A', 'B', 'C');
keep = keep + 1;
call pc(kv);
keep = keep + 1;
call pc(kv);
end t;
""", opt) == "!ABC"



# ---- names the assembler reads as something else ---------------------------

@pytest.mark.parametrize("opt", LEVELS)
def test_names_that_are_registers_conditions_or_operators_to_the_assembler(opt):
    """um80 reads `A' and `HL' as registers ("Register 'A' used as value",
    for procedures A and HL), and EQ, NE, LT, LE, GT, GE, SHR and NUL as
    operators - `call EQ' calls 0FFFFH, without a word.  Such a procedure,
    label or variable is `@A', `@EQ' ... now, as a variable named like a
    register already was.  It reads `Z' and `P' after `jp' as conditions
    (the peephole makes `call z / ret' `jp Z'): the jump is `jp 0+Z'.  And
    a symbol whose letters end in one of its word operators, followed by +
    or -, is read as that operator: `ld hl,TYPE+2' is TYPE(+2), `X1EQ+2'
    does not parse, and neither does a procedure's own `@Q$NUL+1'.  Such an
    offset is written `2+TYPE'."""
    assert run_plm(PRELUDE + """
declare (eq, ne, lt, le, gt, ge, shr) byte;
declare nul (3) byte, type (4) byte, x1eq (3) byte, w address;
a: procedure; call pc('a'); end a;
nz: procedure; call pc('n'); end nz;
z: procedure (c); declare c byte; call pc(c); end z;
p: procedure; call z('p'); end p;
hl: procedure byte; return 'h'; end hl;
q: procedure;
   declare (type, nul) (3) byte;
   type(2) = 'y'; nul(1) = 'u';
   call pc(type(2)); call pc(nul(1));
end q;
eq = '='; ne = '#'; lt = '<'; le = '['; gt = '>'; ge = ']'; shr = '/';
nul(2) = 'N'; type(3) = 'T'; type(1) = 't'; x1eq(2) = 'x'; w = .type(3);
call a; call nz; call p; call pc(hl); call q;
call pc(eq); call pc(ne); call pc(lt); call pc(le); call pc(gt); call pc(ge); call pc(shr);
call pc(nul(2)); call pc(type(3)); call pc(type(1)); call pc(x1eq(2));
if w = .type + 3 then call pc('w');
goto c;
call pc('X');
c: call pc('.');
end t;
""", opt) == "anphyu=#<[>]/NTtxw."


def test_an_offset_from_such_a_symbol_is_written_the_other_way_round():
    from uplm80.names import fix_symbols
    asm = ("\tld\ta,(TYPE+2)\t; TYPE+2\n\tdb\t'TYPE+1',TYPE-3\nX\tEQU\tlow+1\n"
           "\tld\thl,@P$NUL+2\n\tld\thl,M?EQ+2-1\n\tld\ta,(ix+2)\n\tld\thl,COLOR+2")
    assert fix_symbols(asm) == (
        "\tld\ta,(2+TYPE)\t; TYPE+2\n\tdb\t'TYPE+1',-3+TYPE\nX\tEQU\t1+low\n"
        "\tld\thl,2+@P$NUL\n\tld\thl,2-1+M?EQ\n\tld\ta,(ix+2)\n\tld\thl,COLOR+2")


def test_a_jump_to_a_symbol_named_like_a_condition():
    from uplm80.names import fix_symbols
    assert fix_symbols("\tjp\tP\n\tjr\tZ\t; z\nL1:\tjp\tNZ\n\tjp\tnz,P\n\tjp\tPX\n\tcall\tP") == (
        "\tjp\t0+P\n\tjr\t0+Z\t; z\nL1:\tjp\t0+NZ\n\tjp\tnz,P\n\tjp\tPX\n\tcall\tP")



@pytest.mark.parametrize("opt", (0, 2))
def test_input_and_output_of_a_port_that_is_not_a_constant(opt):
    """They called ??inp and ??outp, which the runtime library did not have:
    "Undefined symbol '??outp'"."""
    src = PRELUDE + """
declare (p, v) byte, w address;
p = 1; v = 41h; w = 102h;
output(p) = v;
output(w) = v;
v = input(p);
v = input(w + 1);
call pc('.');
end t;
"""
    assert run_plm(src, opt) == "."



@pytest.mark.parametrize("opt", LEVELS)
def test_a_declaration_hides_the_built_in_of_its_name(opt):
    """PL/M-80's built-ins are declared outside the program, and a
    declaration of the name hides one (9.2).  Code generation took
    `size(2)' of an array SIZE for SIZE(2) ("SIZE() needs a declared
    variable"), and `high(1)' of an array HIGH for HIGH(1); it already let
    a variable ZERO hide the flag, as MP/M II's STAT needs."""
    assert run_plm(PRELUDE + """
declare size (4) byte, length address, input byte, last (3) byte, high (2) address;
declare w address, b byte;
time: procedure (n) byte; declare n byte; return n + 1; end time;
size(2) = 'S'; length = 'L'; input = 'I'; last(1) = 'T'; high(1) = 'H';
call pc(size(2)); call pc(length); call pc(input); call pc(last(1)); call pc(high(1));
call pc(time('a'));
w = .size(2); b = size(2) + 1; call pc(b);
if high(1) = 'H' then call pc('=');
end t;
""", opt) == "SLITHbT="


# ---- a multi-file compile --------------------------------------------------

def _compile_modules(sources: list[str], opt: int = 2) -> subprocess.CompletedProcess:
    """`uplm80 A.PLM B.PLM ... -o AB.MAC'; stdout is the assembly."""
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, text in enumerate(sources):
            paths.append(os.path.join(d, f"M{i}.PLM"))
            with open(paths[-1], "w") as fh:
                fh.write(text)
        mac = os.path.join(d, "AB.MAC")
        r = subprocess.run(compile_cmd("-O", str(opt), "-o", mac, *paths), capture_output=True,
                           text=True, timeout=60, env=compiler_env(), check=False)
        if r.returncode == 0:
            with open(mac) as fh:
                r.stdout = fh.read()
    return r


def _run_modules(sources: list[str], opt: int) -> str:
    reason = tools_missing()
    if reason:
        pytest.skip(reason)
    r = _compile_modules(sources, opt)
    assert r.returncode == 0, r.stderr
    return run_asm(r.stdout).stdout.replace("\r", "")


MAIN = """0100H:
m: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
t2: procedure byte external; end t2;
declare k literally '1';
declare tag (2) byte data ('m', 'M');
declare seen byte;
helper: procedure byte;
    declare (n, seen) byte;
    if seen <> 77h then do; seen = 77h; n = 'a' - 1; end;
    n = n + k;
    return n;
end helper;
declare i byte;
seen = 0;
do i = 1 to 3;
    call mon1(2, helper);
    call mon1(2, t2);
end;
call mon1(2, tag(0));
end m;
"""

LIB = """lib: do;
declare k literally '2';
declare tag (2) byte data ('l', 'L');
declare seen byte;
helper: procedure byte;
    declare (n, seen) byte;
    if seen <> 77h then do; seen = 77h; n = 'A' - 1; end;
    n = n + k - 1;
    return n;
end helper;
t2: procedure byte public;
    return helper + tag(1) - 'L';
end t2;
end lib;
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_each_module_of_a_multi_file_compile_has_its_own_names(opt):
    """Modules have separate name spaces for everything not PUBLIC or
    EXTERNAL (Programming Manual, 10.4).  Compiled together they made one
    assembly in which two private procedures called HELPER, their static
    locals, and the modules' own SEEN, TAG and K were each defined twice
    ("Symbol 'HELPER' multiply defined", "'@HELPER$N' multiply defined").
    Each is now qualified with its module's name (LIB?HELPER), and T2,
    PUBLIC in one and EXTERNAL in the other, still binds."""
    assert _run_modules([MAIN, LIB], opt) == "aAbBcCm"


def test_the_qualified_names():
    r = _compile_modules([MAIN, LIB])
    assert r.returncode == 0, r.stderr
    labels = {l.split(":")[0] for l in r.stdout.splitlines() if ":" in l.split("\t")[0]}
    for name in ("M?HELPER", "LIB?HELPER", "M?SEEN", "LIB?SEEN", "M?TAG", "LIB?TAG",
                 "T2", "@M?HELPER$N", "@LIB?HELPER$N"):
        assert name in labels, sorted(labels)
    assert "public\tT2" in r.stdout


@pytest.mark.parametrize("opt", (0, 2))
def test_a_public_procedures_parameter_keeps_its_name(opt):
    """An EXTERNAL declaration of a procedure, in a module before the one
    that defines it, gave its parameter a label as if it were static
    there, `@KEEPIT$V', which the definition's static parameter then met:
    that one was renamed `@KEEPIT$V?2'.  It worked, but the name was not
    the one the module compiled alone has.  An EXTERNAL procedure's
    parameters are in the module that defines it."""
    user = """0100H:
a: do;
mon1: procedure (f, x) external; declare f byte, x address; end mon1;
keepit: procedure (v) external; declare v byte; end keepit;
call keepit('K');
call keepit('L');
call mon1(0, 0);
end a;
"""
    lib = """b: do;
mon1: procedure (f, x) external; declare f byte, x address; end mon1;
declare keep address;
declare kept based keep byte;
keepit: procedure (v) public; declare v byte; keep = .v; call mon1(2, kept); end keepit;
end b;
"""
    r = _compile_modules([user, lib], opt)
    assert r.returncode == 0, r.stderr
    assert "@KEEPIT$V:" in r.stdout and "?2" not in r.stdout, r.stdout
    assert _run_modules([user, lib], opt) == "KL"


def test_a_private_name_of_another_module_is_an_error():
    """Compiled alone, a module cannot reach another's private procedure;
    together, it silently could.  Now it is an error that says what to do."""
    r = _compile_modules([MAIN.replace("t2: procedure byte external; end t2;\n", ""),
                          LIB.replace("t2: procedure byte public;", "t2: procedure byte;")])
    assert r.returncode != 0
    assert ("M0.PLM:17:18: error: T2 is not declared in module M; module LIB declares it but "
            "does not make it PUBLIC (declare it PUBLIC there and EXTERNAL here)") in r.stderr


@pytest.mark.parametrize("opt", (0, 2))
def test_a_public_procedure_needs_no_external_declaration(opt):
    """What a multi-file compile has always allowed, and 80un relies on."""
    assert _run_modules([MAIN.replace("t2: procedure byte external; end t2;\n", ""), LIB],
                        opt) == "aAbBcCm"


_BUILTINS_LIB = """lib: do;
shl: procedure (a, b) address public;
  declare a address, b byte;
  return 0abcdh;
end shl;
shr: procedure (a, b) address public;
  declare a address, b byte;
  return 0dcbah;
end shr;
double: procedure (a) address public;
  declare a byte;
  return 0bbbbh;
end double;
declare memory (4) byte public initial ('L', 'L', 'L', 'L');
declare stackptr address public initial (1234h);
peek: procedure byte public;
  return memory(0);
end peek;
own: procedure address public;
  return shl(1, 1) + double(1) + stackptr;
end own;
end lib;
"""

_BUILTINS_USER = """0100H:
m: do;
mon1: procedure (f, a) external; declare f byte, a address; end mon1;
peek: procedure byte external; end peek;
own: procedure address external; end own;
%s
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
declare w address, b byte;
w = 1234h; b = 5;
call ph(shl(w, 4)); call ph(w * 8); call ph(double(b)); call ph(double(30h));
call ph(w / 2);
memory(0) = 'Q'; call ph(peek);
call ph(stackptr = 1234h);
call ph(own);
end m;
"""


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("external, expect", [
    ("", "2340 91A0 0005 0030 091A 004C 0000 79BC "),
    ("shl: procedure (a, b) address external; declare a address, b byte; end shl;\n"
     "double: procedure (a) address external; declare a byte; end double;",
     "ABCD 91A0 BBBB BBBB 091A 004C 0000 79BC "),
])
def test_a_built_in_another_module_makes_public_is_still_the_built_in(opt, external, expect):
    """A module that does not declare SHL, DOUBLE, MEMORY or STACKPTR
    means the built-in, as it does compiled alone: another module's PUBLIC
    procedure or variable of the name is not declared in it (9.2, 10.4).
    Compiled together with a module that makes them PUBLIC, it called that
    module's SHL for `shl(w, 4)' at -O0 to -O2, and at -O2 and -O3 for
    `w * 8' too, which the optimizer makes SHL(w, 3); `double(30h)' was
    the PUBLIC DOUBLE at -O0 and 30H from -O1 up; `memory(0) = 'Q'' stored
    to the other module's MEMORY, and STACKPTR was its variable (0.4.0 the
    same).  Declared EXTERNAL, each is the other module's."""
    user = _BUILTINS_USER % external
    assert _run_modules([user, _BUILTINS_LIB], opt) == expect


def test_a_module_without_a_name_goes_by_its_files():
    r = _compile_modules(["declare x byte public;\nq: procedure; x = 1; end q;\ncall q;\n",
                          "declare x byte external;\nq: procedure; x = 2; end q;\n"])
    assert r.returncode == 0, r.stderr
    assert "M0?Q:" in r.stdout and "M1?Q:" in r.stdout, r.stdout


def test_a_module_without_a_name_goes_by_its_own_file_not_an_included_one():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "COMMON.LIT"), "w") as fh:
            fh.write("declare true literally '0ffh';\n")
        paths = []
        for name, text in (("ONE.PLM", "declare x byte public;\nq: procedure; x = 1; end q;\ncall q;\n"),
                           ("TWO.PLM", "declare x byte external;\nq: procedure; x = true; end q;\n")):
            paths.append(os.path.join(d, name))
            with open(paths[-1], "w") as fh:
                fh.write("$include (common.lit)\n" + text)
        mac = os.path.join(d, "AB.MAC")
        r = subprocess.run(compile_cmd("-o", mac, *paths), capture_output=True, text=True,
                           timeout=60, env=compiler_env(), check=False)
        assert r.returncode == 0, r.stderr
        with open(mac) as fh:
            asm = fh.read()
    assert "ONE?Q:" in asm and "TWO?Q:" in asm, asm


def test_two_main_program_modules_are_an_error():
    """Only the first module's statements were compiled; the second's were
    dropped without a word."""
    r = _compile_modules(["a: do; declare x byte; x = 1; end a;\n",
                          "b: do; declare y byte; y = 2; end b;\n"])
    assert r.returncode != 0
    assert ("M1.PLM:1:24: error: modules A and B both have statements at their outer level; "
            "only the main program module may") in r.stderr, r.stderr


def test_a_name_declared_twice_in_one_block_is_an_error():
    """Both declarations were generated, and the assembler said "multiply
    defined", or, for a procedure and a variable, took one for the other."""
    err = _compile_error(PRELUDE + "declare i byte;\np: procedure; return; end p;\ndeclare p byte;\nend t;\n")
    assert "T.PLM:7:9: error: P is declared twice in the same block" in err


def test_a_literally_declared_again_as_it_was():
    """An $INCLUDE file and the file including it often both declare TRUE."""
    assert run_plm(PRELUDE + """declare true literally '0ffh';
declare true literally '0ffh';
if true then call pc('y');
end t;
""") == "y"


@pytest.mark.parametrize("opt", LEVELS)
def test_a_goto_to_a_public_label_of_another_module(opt):
    """The third GOTO PL/M-80 allows (9.3): to the main program's PUBLIC
    label, from a module that declares it EXTERNAL.  Compiled together the
    EXTERNAL declaration was an EXTRN of a label the same assembly defines,
    and the peephole's `jr AGAIN' was "JR to 'AGAIN': its target is the
    external symbol AGAIN" at -O1 and up.  No EXTRN is emitted now for a
    name one of the modules makes PUBLIC."""
    main = """0100H:
a: do;
mon1: procedure (f, x) external; declare f byte, x address; end mon1;
declare again label public;
declare n byte public;
bump: procedure external; end bump;
n = 0;
again:
call mon1(2, '0' + n);
if n < 3 then call bump;
end a;
"""
    other = """b: do;
declare again label external;
declare n byte external;
bump: procedure public; n = n + 1; goto again; end bump;
end b;
"""
    assert _run_modules([main, other], opt) == "0123"


PUBLIC_IN_A_BLOCK = """0100H:
a: do;
mon1: procedure (f, x) external; declare f byte, x address; end mon1;
declare again label public;
declare n byte public;
bump: procedure external; end bump;
n = 0;
do;
again:
    call mon1(2, '0' + n);
    if n < 3 then call bump;
end;
end a;
"""


def test_a_public_label_that_labels_no_statement_at_the_outer_level_is_an_error():
    """A PUBLIC label is attached to a statement at the outer level of the
    main program module (9.3); `again:' in a DO block is another label, the
    block's.  uplm80 emitted `public AGAIN' with nothing defining it, so
    only the linker, or in a multi-file compile the assembler, found it
    ("Undefined symbol 'AGAIN'"); Intel's PL/M-80 V3.1 rejects the program,
    ERROR #172, INVALID LABEL: UNDEFINED."""
    err = _compile_error(PUBLIC_IN_A_BLOCK)
    assert ("T.PLM:4:9: error: AGAIN is declared a PUBLIC LABEL but labels no statement "
            "at the outer level of the main program module") in err, err
    assert "9800268B, 9.3" in err and "the AGAIN: in a DO block is another label" in err
    err = _compile_error("0100H:\nt: do;\ndeclare x label public;\n"
                         "p: procedure public; return; end p;\nend t;\n")
    assert "T.PLM:3:9: error: X is declared a PUBLIC LABEL but labels no statement" in err


def test_a_public_label_that_labels_no_statement_in_a_multi_file_compile():
    other = """b: do;
declare again label external;
declare n byte external;
bump: procedure public; n = n + 1; goto again; end bump;
end b;
"""
    r = _compile_modules([PUBLIC_IN_A_BLOCK, other])
    assert r.returncode != 0
    assert "M0.PLM:4:9: error: AGAIN is declared a PUBLIC LABEL" in r.stderr, r.stderr


# ---- what Intel's PL/M-80 V3.1 rejects --------------------------------------

@pytest.mark.parametrize("opt", LEVELS)
def test_an_interrupt_procedure_is_at_the_outer_level_of_its_module(opt):
    """"It may only be used in a PROCEDURE statement at the outer level of a
    program module" (8.1.6).  One nested in a procedure was compiled, and a
    local of the procedure around it that the interrupt read could be in
    ??AUTO, in another procedure's frame (0.3.6 the same).  Intel's PL/M-80
    V3.1 rejects one in a procedure and one in a DO block of the module,
    ERROR #39, INVALID ATTRIBUTE OR INITIALIZATION, NOT AT MODULE LEVEL."""
    err = _compile_error(PRELUDE + """declare w address;
outer: procedure;
  declare l byte;
  ih: procedure interrupt 3;
    w = l;
  end ih;
  l = 1;
end outer;
call outer;
end t;
""", opt)
    assert ("T.PLM:8:3: error: IH: an INTERRUPT procedure must be declared at the outer "
            "level of the module, not in procedure OUTER (Programming Manual 9800268B, "
            "8.1.6)") in err, err
    err = _compile_error(PRELUDE + """declare w address;
do;
  ih2: procedure interrupt 4;
    w = 2;
  end ih2;
end;
end t;
""", opt)
    assert "T.PLM:7:3: error: IH2: an INTERRUPT procedure must be declared at the outer " \
           "level of the module, not in a DO block" in err, err
    ok = _compile(PRELUDE + """declare w address;
ih3: procedure interrupt 5;
  w = 3;
end ih3;
w = 0;
end t;
""", opt)
    assert ok.returncode == 0, ok.stderr


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("stmt, name, kind", [
    ("y = x() + 1;", "X", "a variable"),
    ("w = x();", "X", "a variable"),
    ("y = a();", "A", "a variable"),
    ("y = bb();", "BB", "a variable"),
    ("y = s();", "S", "a variable"),
    ("call w();", "W", "a variable"),
    ("y = q(1) + x;", None, None),
])
def test_empty_parentheses_after_a_variable_are_an_error(opt, stmt, name, kind):
    """PL/M-80 has no empty argument list or subscript.  `y = x() + 1'
    with x a BYTE compiled to a CALL through x's value (0.3.6: `call X'),
    and so did an array, a BASED variable, a structure and `CALL w()'.
    Intel's PL/M-80 V3.1 rejects each, ERROR #127, INVALID SUBSCRIPT ON
    NON-ARRAY, and ERROR #102, MISSING PRIMARY OPERAND.  A procedure's
    `f()' is still taken for `f', with a warning (V3.1 rejects that too:
    test_what_v31_rejects_and_programs_rely_on_is_a_warning)."""
    src = PRELUDE + """declare (x, y) byte, (w, p) address, a (3) byte, bb based p byte;
declare s structure (m byte);
f: procedure byte; return 3; end f;
q: procedure (v) byte; declare v byte; return v + f(); end q;
""" + stmt + "\nend t;\n"
    if name is None:
        r = _compile(src, opt)
        assert r.returncode == 0, r.stderr
        return
    err = _compile_error(src, opt)
    assert (f"T.PLM:9:{stmt.index(name.lower()) + 1}: error: {name}(): {name} is {kind}, "
            "and PL/M-80 has neither an empty subscript nor an empty argument list") in err, err


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("stmt, text, kind", [
    ("y = s.m();", "S.M", "a structure member"),
    ("w = s.a();", "S.A", "a structure member"),
    ("y = s.n();", "S.N", "a structure member"),
    ("y = sa(1).m();", "SA(1).M", "a structure member"),
    ("y = sa(k).m() + 1;", "SA(K).M", "a structure member"),
    ("call s.a();", "S.A", "a structure member"),
    ("y = s.n(1)();", "S.N(1)", "a subscripted variable"),
    ("y = a(1)();", "A(1)", "a subscripted variable"),
    ("call a(1)();", "A(1)", "a subscripted variable"),
    ("y = sa(k + 1)();", "SA(...)", "a subscripted variable"),
    ("y = q(1)();", "Q(1)", "a call"),
])
def test_empty_parentheses_after_a_member_or_a_subscript_are_an_error(opt, stmt, text, kind):
    """Empty parentheses after a structure member, or after a subscript,
    are no more a subscript or an argument list than after a variable;
    what they follow is never a procedure's name.  `y = s.m()' compiled to
    `ld a,(S) / ld l,a / ld h,0 / call ??jphl', a call through the
    member's value, and so did the others.  Intel's PL/M-80 V3.1 rejects
    each: ERROR #127, INVALID SUBSCRIPT ON NON-ARRAY, and #32 for `s.m()',
    `s.a()' and `sa(1).m()'; #102, MISSING PRIMARY OPERAND, for `s.n()',
    an array member, and `call s.a()'; #32, INVALID SYNTAX, after a
    subscript, with #118 in a CALL and #135 after a structure element."""
    src = PRELUDE + """declare (x, y, k) byte, (w, p) address, a (3) byte;
declare s structure (m byte, a address, n (2) byte), sa (3) structure (m byte);
f: procedure byte; return 3; end f;
q: procedure (v) byte; declare v byte; return v + f(); end q;
""" + stmt + "\nend t;\n"
    err = _compile_error(src, opt)
    col = re.search(r"\b" + text.split("(")[0].split(".")[0].lower() + r"\b", stmt).start() + 1
    assert (f"T.PLM:9:{col}: error: {text}(): {text} is {kind}, "
            "and PL/M-80 has neither an empty subscript nor an empty argument list") in err, err


def test_empty_parentheses_after_a_parameter_are_an_error():
    err = _compile_error(PRELUDE + "q: procedure (x) byte; declare x byte; return x(); end q;\n"
                         "call pc(q(1));\nend t;\n")
    assert "T.PLM:5:47: error: X(): X is a parameter" in err, err


# What Intel's PL/M-80 V3.1 rejects, and uplm80 compiled (0.4.1's Known
# issues): the statements, after PRELUDE and V31_DECLS; V3.1's errors; and
# what uplm80 says.  tests/test_intel_oracle.py builds each with V3.1 too.
V31_DECLS = """declare (b, c) byte, w address;
f: procedure byte; return 3; end f;
g: procedure; b = 1; end g;
"""
_OUTER = "must be declared at the outer level of the module, not in "
V31_REJECTS = {
    "zero-dimension": ("declare z (0) byte;\nz(0) = 1;\n", (57,),
                       "T.PLM:8:12: error: (0): an array has at least one element"),
    "zero-member": ("declare s structure (m (0) byte, x byte);\ns.x = 1;\n", (57,),
                    "T.PLM:8:25: error: (0): an array has at least one element"),
    "zero-local": ("p: procedure;\n  declare a (0) address;\n  a(0) = 1;\nend p;\ncall p;\n",
                   (57,), "T.PLM:9:14: error: (0): an array has at least one element"),
    "dot-double": ("w = .double;\n", (123,),
                   "T.PLM:8:6: error: .DOUBLE: DOUBLE is a built-in, and of the built-ins only "
                   "MEMORY has an address"),
    "dot-stackptr": ("w = .stackptr;\n", (123,), "error: .STACKPTR: STACKPTR is a built-in"),
    "dot-move": ("w = .move;\n", (123,), "error: .MOVE: MOVE is a built-in"),
    "dot-output": ("w = .output(3);\n", (123,), "error: .OUTPUT: OUTPUT is a built-in"),
    "carry()": ("b = carry();\n", (102, 153),
                "T.PLM:8:5: error: CARRY(): CARRY is a built-in, and PL/M-80 has neither an "
                "empty subscript nor an empty argument list"),
    "zero()": ("if zero() then b = 1;\n", (102, 153), "error: ZERO(): ZERO is a built-in"),
    "dec()": ("b = dec();\n", (102,), "error: DEC(): DEC is a built-in"),
    "public-in-procedure": ("p: procedure;\n  declare x byte public;\n  x = 1;\nend p;\ncall p;\n",
                            (73,), "T.PLM:9:11: error: X: a PUBLIC variable " + _OUTER
                            + "procedure P"),
    "external-in-procedure": ("p: procedure;\n  declare x byte external;\n  b = x;\nend p;\n"
                              "call p;\n", (73,),
                              "error: X: an EXTERNAL variable " + _OUTER + "procedure P"),
    "public-in-do": ("do;\n  declare y byte public;\n  y = 1;\nend;\n", (73,),
                     "error: Y: a PUBLIC variable " + _OUTER + "a DO block"),
    "public-procedure-in-procedure": ("p: procedure;\n  q: procedure public;\n    b = 2;\n"
                                      "  end q;\n  call q;\nend p;\ncall p;\n", (39,),
                                      "T.PLM:9:3: error: Q: a PUBLIC procedure " + _OUTER
                                      + "procedure P"),
    "external-procedure-in-do": ("do;\n  r: procedure external;\n  end r;\n  call r;\nend;\n",
                                 (39, 174), "error: R: an EXTERNAL procedure " + _OUTER
                                 + "a DO block; Intel's PL/M-80 V3.1 rejects it (ERROR #39, "
                                 "INVALID ATTRIBUTE OR INITIALIZATION, NOT AT MODULE LEVEL, and "
                                 "#174, INVALID NULL PROCEDURE)"),
    # 0.4.2's Known issues, 0.4.3.
    "end-names-another-procedure": ("p: procedure;\n  b = 1;\nend q;\ncall p;\n", (20,),
                                    "T.PLM:10:5: error: END Q: the END of procedure P names Q; "
                                    "Intel's PL/M-80 V3.1 rejects it (ERROR #20, MISMATCHED "
                                    "IDENTIFIER AT END OF BLOCK)"),
    "labelled-end-names-another": ("p: procedure;\n  b = 1;\nout: end q;\ncall p;\n", (20,),
                                   "T.PLM:10:10: error: END Q: the END of procedure P names Q"),
    "end-names-another-block": ("l: do;\n  b = 1;\nend m;\n", (20,),
                                "T.PLM:10:5: error: END M: the END of a block labelled L names M"),
    "end-of-a-block-with-no-label": ("do;\n  b = 1;\nend n;\n", (20,),
                                     "error: END N: the END of a block with no label names N"),
    "do-case-with-no-case": ("do case b;\nend;\n", (201,),
                             "T.PLM:8:1: error: DO CASE: a DO CASE block has at least one case; "
                             "Intel's PL/M-80 V3.1 rejects it (ERROR #201, INVALID DO CASE "
                             "BLOCK, AT LEAST ONE CASE REQUIRED)"),
    "do-case-with-a-labelled-end": ("do case b;\nl: end;\n", (201,),
                                    "error: DO CASE: a DO CASE block has at least one case"),
    "dot-call": ("h: procedure (x) byte; declare x byte; return x; end h;\nw = .h(1);\n", (104,),
                 "T.PLM:9:6: error: .H(1): H is a procedure, and the dot operator takes the "
                 "address of a procedure, not of a call of it (Programming Manual 9800268B, "
                 "4.1.3); Intel's PL/M-80 V3.1 rejects it (ERROR #104, ILLEGAL PROCEDURE "
                 "INVOCATION WITH DOT OPERATOR)"),
    "size-of-a-call": ("declare ab(4) byte;\nh: procedure (x) byte; declare x byte; return x; "
                       "end h;\nw = size(ab(h(1)));\n", (32,),
                       "T.PLM:10:5: error: SIZE(AB(H(1))): the subscripts of SIZE's argument are "
                       "not evaluated, and none has anything in parentheses in it, a call, a "
                       "subscript or an expression; Intel's PL/M-80 V3.1 rejects it (ERROR #32, "
                       "INVALID SYNTAX, TEXT IGNORED UNTIL ';')"),
    "length-of-a-parenthesized-subscript": ("declare sa(3) structure (z(2) byte);\n"
                                            "w = length(sa((b)).z);\n", (125, 32),
                                            "error: LENGTH(SA((B)).Z): the subscripts of "
                                            "LENGTH's argument are not evaluated, and none has "
                                            "anything in parentheses in it, a call, a subscript "
                                            "or an expression; Intel's PL/M-80 V3.1 rejects it "
                                            "(ERROR #125, ILLEGAL ARGUMENT FOR BUILT-IN "
                                            "PROCEDURE, and #32, INVALID SYNTAX, TEXT IGNORED "
                                            "UNTIL ';')"),
    "null-procedure": ("p: procedure;\n  declare k byte;\nend p;\ncall p;\n", (174,),
                       "T.PLM:8:1: error: P: a procedure has at least one statement, and P has "
                       "none; Intel's PL/M-80 V3.1 rejects it (ERROR #174, INVALID NULL "
                       "PROCEDURE)"),
    "null-procedure-labelled-end": ("p: procedure;\nl: end p;\ncall p;\n", (174,),
                                    "error: P: a procedure has at least one statement"),
    "two-subscripts-on-a-scalar": ("declare shl address;\nw = shl(w, 3);\n", (127, 114),
                                   "T.PLM:9:5: error: SHL(W, 3): SHL is not an array, and only an "
                                   "array takes a subscript, and only one; Intel's PL/M-80 V3.1 "
                                   "rejects it (ERROR #127, INVALID SUBSCRIPT ON NON-ARRAY, and "
                                   "#114, INVALID SUBSCRIPT, MULTIPLE SUBSCRIPTS ILLEGAL)"),
    "unsubscripted-array": ("declare a(4) byte;\na = 3;\n", (133,),
                            "T.PLM:9:1: error: A: A is an array, and an array is named without a "
                            "subscript only as the operand of a dot or the argument of LENGTH, "
                            "LAST or SIZE (Programming Manual 9800268B, 3.6.2); Intel's PL/M-80 "
                            "V3.1 rejects it (ERROR #133, ILLEGAL REFERENCE TO UNSUBSCRIPTED "
                            "ARRAY)"),
    "unsubscripted-array-read": ("declare a(4) byte;\nb = a + 1;\n", (133,),
                                 "T.PLM:9:5: error: A: A is an array"),
    "unsubscripted-array-subscript": ("declare a(4) byte, sz(3) byte;\nb = sz(a);\n", (133,),
                                      "T.PLM:9:8: error: A: A is an array"),
    "unsubscripted-array-argument": ("declare a(4) byte;\ncall pc(a);\n", (133,),
                                     "T.PLM:9:9: error: A: A is an array"),
    "unsubscripted-member-array": ("declare s structure (m(3) byte, n byte);\ns.m = 4;\n", (134,),
                                   "T.PLM:9:1: error: S.M: M is an array, and a member array is "
                                   "named without a subscript only as the operand of a dot or the "
                                   "argument of LENGTH, LAST or SIZE (Programming Manual 9800268B, "
                                   "3.6.2); Intel's PL/M-80 V3.1 rejects it (ERROR #134, ILLEGAL "
                                   "REFERENCE TO UNSUBSCRIPTED MEMBER ARRAY)"),
    "forward-call": ("p: procedure;\n  call q;\nend p;\nq: procedure;\n  b = 1;\nend q;\n"
                     "call p;\n", (169,),
                     "T.PLM:9:8: error: Q: procedure Q is declared after this call of it, and a "
                     "procedure is called only after its declaration, but by a REENTRANT "
                     "procedure if it is REENTRANT too; Intel's PL/M-80 V3.1 rejects it (ERROR "
                     "#169, ILLEGAL FORWARD CALL)"),
    "forward-typed-call": ("p: procedure;\n  b = h + 1;\nend p;\nh: procedure byte;\n  return 2;"
                           "\nend h;\ncall p;\n", (169,),
                           "T.PLM:9:7: error: H: procedure H is declared after this call of it"),
    "forward-call-of-a-reentrant": ("p: procedure;\n  call q;\nend p;\nq: procedure reentrant;\n"
                                    "  b = 1;\nend q;\ncall p;\n", (169,),
                                    "error: Q: procedure Q is declared after this call of it"),
    "forward-call-from-a-reentrant": ("r: procedure reentrant;\n  call s;\nend r;\n"
                                      "s: procedure;\n  b = 1;\nend s;\ncall r;\n", (169,),
                                      "error: S: procedure S is declared after this call of it"),
    # 0.4.3's Known issues: a built-in in a restricted expression, which
    # -O1 and up folded (SHL and SHR of a BYTE in 16 bits) and -O0 refused
    # with a message of its own, or took `.a + low(3)' and `at (double(12h))'
    # for something else.
    "shl-in-data": ("declare d address data (shl(0f0h, 4));\nw = d;\n", (151, 152),
                    "T.PLM:8:25: error: SHL(0f0h, 4): SHL is a built-in, and a DATA or INITIAL "
                    "value is a restricted expression, of constants and locations only; Intel's "
                    "PL/M-80 V3.1 rejects it (ERROR #151, INVALID OPERAND IN RESTRICTED "
                    "EXPRESSION, and #152, MISSING ')' AFTER CONSTANT LIST)"),
    "shr-in-initial": ("declare d address initial (shr(0f00h, 4));\nw = d;\n", (151, 152),
                       "T.PLM:8:28: error: SHR(0f00h, 4): SHR is a built-in, and a DATA or "
                       "INITIAL value is a restricted expression"),
    "rol-in-data": ("declare d byte data (rol(81h, 1));\nb = d;\n", (151, 152),
                    "T.PLM:8:22: error: ROL(81h, 1): ROL is a built-in"),
    "ror-in-a-sum-in-data": ("declare d byte data (1 + ror(81h, 1));\nb = d;\n", (151, 152),
                             "T.PLM:8:26: error: ROR(81h, 1): ROR is a built-in"),
    "low-in-data": ("declare d(2) byte data (low(1234h), 5);\nb = d(1);\n", (151, 152),
                    "T.PLM:8:25: error: LOW(1234h): LOW is a built-in"),
    "high-in-initial": ("declare d byte initial (high(1234h));\nb = d;\n", (151, 152),
                        "T.PLM:8:25: error: HIGH(1234h): HIGH is a built-in"),
    "double-in-data": ("declare d address data (double(12h));\nw = d;\n", (151, 152),
                       "T.PLM:8:25: error: DOUBLE(12h): DOUBLE is a built-in"),
    "size-in-data": ("declare a(5) byte, d address data (.a + size(a));\nw = d;\n", (151, 152),
                     "T.PLM:8:41: error: SIZE(A): SIZE is a built-in"),
    "length-in-initial": ("declare a(5) byte;\ndeclare d address initial (length(a) - last(a));\n"
                          "w = d;\n", (151, 152),
                          "T.PLM:9:28: error: LENGTH(A): LENGTH is a built-in"),
    "shl-in-at": ("declare d byte at (shl(1, 12));\nb = d;\n", (151, 146),
                  "T.PLM:8:20: error: SHL(1, 12): SHL is a built-in, and an AT address is a "
                  "restricted expression, a constant or a location plus or minus constants; "
                  "Intel's PL/M-80 V3.1 rejects it (ERROR #151, INVALID OPERAND IN RESTRICTED "
                  "EXPRESSION, and #146, MISSING ')' AFTER 'AT' RESTRICTED EXPRESSION)"),
    "double-in-at": ("declare d byte at (double(12h));\nb = d;\n", (151, 146),
                     "T.PLM:8:20: error: DOUBLE(12h): DOUBLE is a built-in, and an AT address"),
    "size-in-at": ("declare a(5) byte, d byte at (.a + size(a));\nb = d;\n", (151, 146),
                   "T.PLM:8:36: error: SIZE(A): SIZE is a built-in, and an AT address"),
    "low-in-a-subscript-in-data": ("declare a(5) byte, d address data (.a(low(1)));\nw = d;\n",
                                   (151, 150),
                                   "T.PLM:8:39: error: LOW(1): LOW is a built-in, and a DATA or "
                                   "INITIAL value is a restricted expression, of constants and "
                                   "locations only; Intel's PL/M-80 V3.1 rejects it (ERROR #151, "
                                   "INVALID OPERAND IN RESTRICTED EXPRESSION, and #150, MISSING "
                                   "')' AT END OF RESTRICTED SUBSCRIPT)"),
    "shl-in-a-subscript-in-at": ("declare a(5) byte, d byte at (.a(shl(1, 1)));\nb = d;\n",
                                 (151, 150), "T.PLM:8:34: error: SHL(1, 1): SHL is a built-in, "
                                 "and an AT address"),
    "memory-in-data": ("declare d address data (memory);\nw = d;\n", (151,),
                       "T.PLM:8:25: error: MEMORY: MEMORY is a built-in, and a DATA or INITIAL "
                       "value is a restricted expression, of constants and locations only; "
                       "Intel's PL/M-80 V3.1 rejects it (ERROR #151, INVALID OPERAND IN "
                       "RESTRICTED EXPRESSION)\n"),
    "stackptr-in-at": ("declare d byte at (stackptr);\nb = d;\n", (151,),
                       "T.PLM:8:20: error: STACKPTR: STACKPTR is a built-in, and an AT address"),
    "dot-stackptr-in-at": ("declare d byte at (.stackptr);\nb = d;\n", (211,),
                           "T.PLM:8:21: error: .STACKPTR: STACKPTR is a built-in, and the "
                           "location in an AT address is a variable's, or MEMORY's; Intel's "
                           "PL/M-80 V3.1 rejects it (ERROR #211, INVALID IDENTIFIER IN 'AT' "
                           "RESTRICTED REFERENCE)"),
    "shl-in-a-constant-list": ("w = .(shl(1, 2), 3);\n", (151, 152, 32, 172),
                               "T.PLM:8:7: error: SHL(1, 2): SHL is a built-in, and a constant "
                               "list holds constants only; Intel's PL/M-80 V3.1 rejects it "
                               "(ERROR #151, INVALID OPERAND IN RESTRICTED EXPRESSION, and #152, "
                               "MISSING ')' AFTER CONSTANT LIST, and #32, INVALID SYNTAX, TEXT "
                               "IGNORED UNTIL ';', and #172, INVALID LABEL: UNDEFINED)"),
    "memory-in-a-constant-list": ("w = .(memory, 7);\n", (151, 209),
                                  "T.PLM:8:7: error: MEMORY: MEMORY is a built-in, and a "
                                  "constant list holds constants only; Intel's PL/M-80 V3.1 "
                                  "rejects it (ERROR #151, INVALID OPERAND IN RESTRICTED "
                                  "EXPRESSION, and #209, ILLEGAL INITIALIZATION OF MORE SPACE "
                                  "THAN DECLARED)"),
    # 0.4.3, again: what else a restricted expression does not take, which
    # uplm80 compiled - a name, as its address in a DATA list and as its
    # value in a constant list at -O3, where -O0 to -O2 refused it; a byte
    # of a larger number; parentheses, a product, NOT, a string in a sum -
    # at every level, or at some.  In a constant list V3.1 gives more
    # errors than the one: #209 for a value taken for a word, #32 at a
    # parenthesis, #172 after a built-in's call.
    "low-in-a-constant-list": ("w = .(low(1234h), 3);\n", (151, 152, 32, 172),
                               "T.PLM:8:7: error: LOW(1234h): LOW is a built-in, and a constant "
                               "list holds constants only"),
    "size-in-a-constant-list": ("declare ar (5) byte;\nw = .(size(ar), 7);\n",
                                (151, 152, 32, 172),
                                "T.PLM:9:7: error: SIZE(AR): SIZE is a built-in, and a constant "
                                "list holds constants only"),
    "stackptr-in-a-constant-list": ("w = .(stackptr, 7);\n", (151, 209),
                                    "T.PLM:8:7: error: STACKPTR: STACKPTR is a built-in, and a "
                                    "constant list holds constants only"),
    "carry-in-a-constant-list": ("w = .(carry, 7);\n", (151, 209),
                                 "T.PLM:8:7: error: CARRY: CARRY is a built-in, and a constant "
                                 "list holds constants only"),
    "variable-in-a-constant-list": ("w = .(b, 7);\n", (151, 209),
                                    "T.PLM:8:7: error: B: B is a variable, and a constant list "
                                    "holds constants only; Intel's PL/M-80 V3.1 rejects it (ERROR "
                                    "#151, INVALID OPERAND IN RESTRICTED EXPRESSION, and #209, "
                                    "ILLEGAL INITIALIZATION OF MORE SPACE THAN DECLARED)"),
    "sum-with-a-variable-in-a-constant-list": ("b = 3; w = .(b + 1, 7);\n", (151, 209),
                                               "T.PLM:8:14: error: B: B is a variable, and a "
                                               "constant list holds constants only"),
    "memory-variable-in-a-constant-list": ("declare memory byte;\nmemory = 3; w = .(memory, 7);\n",
                                           (151, 209),
                                           "T.PLM:9:19: error: MEMORY: MEMORY is a variable, and "
                                           "a constant list holds constants only"),
    "element-in-a-constant-list": ("declare ar (5) byte;\nw = .(7, ar(1));\n", (151, 152, 32, 209),
                                   "T.PLM:9:10: error: AR(1): AR is an array, and a constant list "
                                   "holds constants only"),
    "member-in-a-constant-list": ("declare s structure (m (2) byte, k address);\nw = .(s.k, 7);\n",
                                  (151, 152),
                                  "T.PLM:9:7: error: S.K: S.K is a member of structure S, and a "
                                  "constant list holds constants only"),
    "location-in-a-constant-list": ("w = .(.w, 7);\n", (210, 209),
                                    "T.PLM:8:7: error: .W: a constant list holds constants only, "
                                    "and .W is a location; Intel's PL/M-80 V3.1 rejects it (ERROR "
                                    "#210, ILLEGAL INITIALIZATION OF A BYTE TO A VALUE > 255, and "
                                    "#209, ILLEGAL INITIALIZATION OF MORE SPACE THAN DECLARED)"),
    "parenthesis-in-a-constant-list": ("w = .((1 + 2), 7);\n", (151, 152, 32),
                                       "T.PLM:8:7: error: (1 + 2): a constant list has nothing in "
                                       "parentheses; Intel's PL/M-80 V3.1 rejects it (ERROR #151, "
                                       "INVALID OPERAND IN RESTRICTED EXPRESSION, and #152, "
                                       "MISSING ')' AFTER CONSTANT LIST, and #32, INVALID SYNTAX, "
                                       "TEXT IGNORED UNTIL ';')"),
    "negated-parenthesis-in-a-constant-list": ("w = .(-(1), 7);\n", (151, 152, 32),
                                               "T.PLM:8:8: error: (1): a constant list has "
                                               "nothing in parentheses"),
    "300-in-a-constant-list": ("w = .(300, 7);\n", (210,),
                               "T.PLM:8:7: error: 300: a constant list holds bytes, and 300 is "
                               "12CH, more than 0FFH; Intel's PL/M-80 V3.1 rejects it (ERROR "
                               "#210, ILLEGAL INITIALIZATION OF A BYTE TO A VALUE > 255)"),
    "0ffffh-in-a-constant-list": ("w = .(0ffffh, 7);\n", (210,),
                                  "T.PLM:8:7: error: 0ffffh: a constant list holds bytes, and "
                                  "0ffffh is 0FFFFH, more than 0FFH"),
    "sum-over-255-in-a-constant-list": ("w = .(299 + 1, 7);\n", (210,),
                                        "T.PLM:8:7: error: 299 + 1: a constant list holds bytes, "
                                        "and 299 + 1 is 12CH, more than 0FFH"),
    "product-in-a-constant-list": ("w = .(2 * 3);\n", (152,),
                                   "T.PLM:8:7: error: 2 * 3: of the operators, a constant list "
                                   "takes + and - only; Intel's PL/M-80 V3.1 rejects it (ERROR "
                                   "#152, MISSING ')' AFTER CONSTANT LIST)"),
    "not-in-a-constant-list": ("w = .(not 0f0h);\n", (151, 152),
                               "T.PLM:8:7: error: NOT 0f0h: of the operators, a constant list "
                               "takes + and - only; Intel's PL/M-80 V3.1 rejects it (ERROR #151, "
                               "INVALID OPERAND IN RESTRICTED EXPRESSION, and #152, MISSING ')' "
                               "AFTER CONSTANT LIST)"),
    "constant-list-in-a-constant-list": ("w = .(.(5), 7);\n", (147, 32),
                                         "T.PLM:8:7: error: .(5): a constant list does not take "
                                         "the location of constants"),
    # V3.1 lays a constant list out as an untyped DATA list: a value with a
    # name in it, or a location first, is a word, and a string after the
    # last such value makes the list bytes again; where it does not, the
    # list has room for its first value only, and every value after it is
    # #209 and not checked further.  A value laid out that is more than
    # 255 is #210 - a name there is a BYTE 0 - and a list of one value
    # whose last name is a built-in's but MEMORY's #172.  The messages
    # named #209 for a value that starts with a name or a location and
    # not otherwise, #210 only with no name in the value, and #172 only
    # after a built-in's call.
    "variable-then-string-in-a-constant-list": ("w = .(b, '$');\n", (151,),
                                                "T.PLM:8:7: error: B: B is a variable, and a "
                                                "constant list holds constants only; Intel's "
                                                "PL/M-80 V3.1 rejects it (ERROR #151, INVALID "
                                                "OPERAND IN RESTRICTED EXPRESSION)\n"),
    "memory-then-string-in-a-constant-list": ("w = .(memory, '$');\n", (151,),
                                              "T.PLM:8:7: error: MEMORY: MEMORY is a built-in"),
    "location-then-string-in-a-constant-list": ("w = .(.w, '$');\n", (210,),
                                                "T.PLM:8:7: error: .W: a constant list holds "
                                                "constants only, and .W is a location"),
    "variable-after-a-string-in-a-constant-list": ("w = .(b, '$', b);\n", (151, 209),
                                                   "T.PLM:8:7: error: B: B is a variable"),
    "sum-with-a-variable-second-in-a-constant-list": ("w = .(1 + b, 7);\n", (151, 209),
                                                      "T.PLM:8:11: error: B: B is a variable"),
    "negated-variable-second-in-a-constant-list": ("w = .(7, -b);\n", (151, 209),
                                                   "T.PLM:8:11: error: B: B is a variable"),
    "difference-after-a-string-in-a-constant-list": ("w = .('$', 1 - b);\n", (151, 209),
                                                     "T.PLM:8:16: error: B: B is a variable"),
    "stackptr-alone-in-a-constant-list": ("w = .(stackptr);\n", (151, 172),
                                          "T.PLM:8:7: error: STACKPTR: STACKPTR is a built-in, "
                                          "and a constant list holds constants only; Intel's "
                                          "PL/M-80 V3.1 rejects it (ERROR #151, INVALID OPERAND "
                                          "IN RESTRICTED EXPRESSION, and #172, INVALID LABEL: "
                                          "UNDEFINED)"),
    "sum-ending-in-a-built-in-in-a-constant-list": ("w = .(1 + time);\n", (151, 172),
                                                    "T.PLM:8:11: error: TIME: TIME is a "
                                                    "built-in"),
    "location-of-a-built-in-in-a-constant-list": ("w = .(.shl);\n", (210, 172),
                                                  "T.PLM:8:7: error: .SHL: a constant list holds "
                                                  "constants only, and .SHL is a location"),
    "sum-over-255-with-a-variable-in-a-constant-list": ("w = .(300 + b);\n", (151, 210),
                                                        "T.PLM:8:13: error: B: B is a variable"),
    "number-after-a-variable-in-a-constant-list": ("w = .(7, 300, b);\n", (151, 209),
                                                   "T.PLM:8:15: error: B: B is a variable"),
    "number-after-a-string-in-a-constant-list": ("w = .(b, '$', 300);\n", (151, 210),
                                                 "T.PLM:8:7: error: B: B is a variable"),
    "location-before-a-string-in-a-constant-list": ("w = .(7, .w, 'AB');\n", (210,),
                                                    "T.PLM:8:10: error: .W: a constant list holds "
                                                    "constants only, and .W is a location"),
    "product-after-a-variable-in-a-constant-list": ("w = .(b, 2 * 3);\n", (151, 152, 209),
                                                    "T.PLM:8:7: error: B: B is a variable"),
    "string-sum-after-a-variable-in-a-constant-list": ("w = .(b, 'A' + 1);\n", (151, 152),
                                                       "T.PLM:8:7: error: B: B is a variable"),
    "parenthesis-after-300-in-a-constant-list": ("w = .(300 + (1));\n", (151, 152, 32, 210),
                                                 "T.PLM:8:13: error: (1): a constant list has "
                                                 "nothing in parentheses"),
    # After what it does not take V3.1 looks for the list's `)', and one
    # of a parenthesis further on leaves the rest of the statement #32.
    "parenthesis-after-a-negated-location-in-a-constant-list": ("w = .(-.w, (1));\n",
                                                                (151, 152, 32),
                                                                "T.PLM:8:7: error: -.W: a "
                                                                "constant list holds constants "
                                                                "only, and a location is none"),
    "variable-in-data": ("declare d address data (b);\nw = d;\n", (151,),
                         "T.PLM:8:25: error: B: B is a variable, and a DATA or INITIAL value is a "
                         "restricted expression, of constants and locations only; Intel's "
                         "PL/M-80 V3.1 rejects it (ERROR #151, INVALID OPERAND IN RESTRICTED "
                         "EXPRESSION)"),
    "variable-in-initial": ("declare d address initial (w);\nw = d;\n", (151,),
                            "T.PLM:8:28: error: W: W is a variable, and a DATA or INITIAL value"),
    "procedure-in-data": ("declare d address data (g);\nw = d;\n", (151,),
                          "T.PLM:8:25: error: G: G is a procedure, and a DATA or INITIAL value"),
    "variable-in-at": ("declare d byte at (w);\nb = d;\n", (151,),
                       "T.PLM:8:20: error: W: W is a variable, and an AT address is a restricted "
                       "expression"),
    "string-sum-in-data": ("declare d byte data ('A' + 1);\nb = d;\n", (152,),
                           "T.PLM:8:22: error: 'A': a string in a DATA or INITIAL value is a "
                           "value of its own, not added to or subtracted from; Intel's PL/M-80 "
                           "V3.1 rejects it (ERROR #152, MISSING ')' AFTER CONSTANT LIST)"),
    "product-in-data": ("declare d byte data (2 * 3);\nb = d;\n", (152,),
                        "T.PLM:8:22: error: 2 * 3: of the operators, a DATA or INITIAL value "
                        "takes + and - only"),
    "not-in-initial": ("declare d byte initial (not 0f0h);\nb = d;\n", (151, 152),
                       "T.PLM:8:25: error: NOT 0f0h: of the operators, a DATA or INITIAL value "
                       "takes + and - only"),
    "parenthesis-in-data": ("declare d byte data ((1 + 2));\nb = d;\n", (151, 152),
                            "T.PLM:8:22: error: (1 + 2): a DATA or INITIAL value has nothing in "
                            "parentheses"),
    "negated-negation-in-data": ("declare d byte data (- -1);\nb = d;\n", (151,),
                                 "T.PLM:8:22: error: --1: in a DATA or INITIAL value a minus sign "
                                 "goes before a number only"),
    "location-second-in-data": ("declare d address data (1 + .w);\nw = d;\n", (151, 152),
                                "T.PLM:8:29: error: .W: a DATA or INITIAL value is a location "
                                "plus or minus constants, the location first, or constants "
                                "alone"),
    "constant-list-in-data": ("declare d address data (.(5));\nw = d;\n", (147,),
                              "T.PLM:8:25: error: .(5): a DATA or INITIAL value does not take the "
                              "location of constants; Intel's PL/M-80 V3.1 rejects it (ERROR "
                              "#147, MISSING IDENTIFIER FOLLOWING DOT OPERATOR)"),
    "text-location-in-data": ("declare d address data (.'AB');\nw = d;\n", (147,),
                              "T.PLM:8:25: error: .'AB': a DATA or INITIAL value does not take the "
                              "location of constants"),
    "string-location-in-initial": ("declare d address initial (.('AB'));\nw = d;\n", (147,),
                                   "T.PLM:8:28: error: .('AB'): a DATA or INITIAL value does not "
                                   "take the location of constants"),
    "300-in-byte-data": ("declare d (2) byte data (7, 300);\nb = d(1);\n", (210,),
                         "T.PLM:8:29: error: 300: this value fills a BYTE, and 300 is 12CH, more "
                         "than 0FFH; Intel's PL/M-80 V3.1 rejects it (ERROR #210, ILLEGAL "
                         "INITIALIZATION OF A BYTE TO A VALUE > 255)"),
    "location-in-byte-initial": ("declare d byte initial (.w);\nb = d;\n", (210,),
                                 "T.PLM:8:25: error: .W: a location is an address, and this value "
                                 "fills a BYTE"),
    "location-in-a-structures-byte": ("declare d structure (p byte, q address) data (.w, 7);\n"
                                      "b = d.p;\n", (210,),
                                      "T.PLM:8:47: error: .W: a location is an address, and this "
                                      "value fills a BYTE"),
    # A value that fills a BYTE with a name in it is more than 255 as V3.1
    # computes it, a name a BYTE 0 (#210, besides the name's #151); and a
    # list with more values than its declaration holds is #209 besides
    # the errors of what else it has.  The messages named neither.
    "sum-over-255-with-a-variable-in-byte-data": ("declare d byte data (300 + b);\nc = d;\n",
                                                  (151, 210),
                                                  "T.PLM:8:28: error: B: B is a variable, and a "
                                                  "DATA or INITIAL value is a restricted "
                                                  "expression"),
    "subscript-variable-in-byte-data": ("declare ar (5) byte, d byte data (.ar(b));\nc = d;\n",
                                        (151, 210),
                                        "T.PLM:8:39: error: B: B is a variable, and a DATA or "
                                        "INITIAL value is a restricted expression"),
    "variable-past-an-arrays-end-in-data": ("declare d (2) byte data (1, 2, b);\nc = d(0);\n",
                                            (151, 209),
                                            "T.PLM:8:32: error: B: B is a variable, and a DATA or "
                                            "INITIAL value is a restricted expression, of "
                                            "constants and locations only; Intel's PL/M-80 V3.1 "
                                            "rejects it (ERROR #151, INVALID OPERAND IN "
                                            "RESTRICTED EXPRESSION, and #209, ILLEGAL "
                                            "INITIALIZATION OF MORE SPACE THAN DECLARED)"),
    "variable-before-a-value-past-a-scalar": ("declare d byte data (b, 1);\nc = d;\n", (151, 209),
                                              "T.PLM:8:22: error: B: B is a variable"),
    # V3.1 reads a DATA or INITIAL list to the first value it does not
    # read to the end, and gives the errors of each value it reads; the
    # message named those of the first value wrong only.
    "variable-then-300-in-byte-data": ("declare d (3) byte data (b, 300, 7);\nc = d(0);\n",
                                       (151, 210), "T.PLM:8:26: error: B: B is a variable"),
    "300-then-variable-in-byte-initial": ("declare d (3) byte initial (7, 300, b);\nc = d(0);\n",
                                          (151, 210),
                                          "T.PLM:8:32: error: 300: this value fills a BYTE, and "
                                          "300 is 12CH, more than 0FFH; Intel's PL/M-80 V3.1 "
                                          "rejects it (ERROR #151, INVALID OPERAND IN RESTRICTED "
                                          "EXPRESSION, and #210, ILLEGAL INITIALIZATION OF A BYTE "
                                          "TO A VALUE > 255)"),
    "300-past-a-string-in-byte-data": ("declare d (2) byte data (b, 'AB', 300);\nc = d(0);\n",
                                       (151, 209), "T.PLM:8:26: error: B: B is a variable"),
    "variable-after-a-product-in-byte-data": ("declare d (3) byte data (300, 2 * 3, b);\n"
                                              "c = d(0);\n", (152, 210),
                                              "T.PLM:8:26: error: 300: this value fills a BYTE"),
    "parenthesis-in-a-subscript-in-data": ("declare ar (5) byte;\n"
                                           "declare d address data (.ar((1)));\nw = d;\n",
                                           (151, 150),
                                           "T.PLM:9:29: error: (1): the subscript of a location "
                                           "in a DATA or INITIAL value has nothing in parentheses; "
                                           "Intel's PL/M-80 V3.1 rejects it (ERROR #151, INVALID "
                                           "OPERAND IN RESTRICTED EXPRESSION, and #150, MISSING ')' "
                                           "AT END OF RESTRICTED SUBSCRIPT)"),
    "constant-list-in-a-subscript-in-data": ("declare ar (5) byte;\n"
                                             "declare d address data (.ar(.(1)));\nw = d;\n",
                                             (151, 150),
                                             "T.PLM:9:29: error: .(1): the subscript of a "
                                             "location in a DATA or INITIAL value is numbers "
                                             "only"),
    "built-in-in-a-subscript-in-a-constant-list": ("declare ar (5) byte;\n"
                                                   "w = .(.ar(low(1)), 7);\n",
                                                   (151, 150, 32, 210, 172),
                                                   "T.PLM:9:11: error: LOW(1): LOW is a built-in, "
                                                   "and a constant list holds constants only"),
    "variable-in-a-subscript-in-at": ("declare ar (5) byte;\ndeclare d byte at (.ar(b));\n"
                                      "c = d;\n", (151,),
                                      "T.PLM:9:24: error: B: B is a variable, and an AT address is "
                                      "a restricted expression"),
    "product-in-at": ("declare d byte at (2 * 3);\nb = d;\n", (146,),
                      "T.PLM:8:20: error: 2 * 3: of the operators, an AT address takes + and - "
                      "only; Intel's PL/M-80 V3.1 rejects it (ERROR #146, MISSING ')' AFTER 'AT' "
                      "RESTRICTED EXPRESSION)"),
    "location-second-in-at": ("declare ar (4) byte;\ndeclare d byte at (3 + .ar(1));\nb = d;\n",
                              (151, 146),
                              "T.PLM:9:24: error: .AR(1): an AT address is a location plus or "
                              "minus constants, the location first, or constants alone; Intel's "
                              "PL/M-80 V3.1 rejects it (ERROR #151, INVALID OPERAND IN RESTRICTED "
                              "EXPRESSION, and #146, MISSING ')' AFTER 'AT' RESTRICTED "
                              "EXPRESSION)"),
    "string-in-at": ("declare d byte at ('AB');\nb = d;\n", (151, 146),
                     "T.PLM:8:20: error: 'AB': an AT address has no string in it"),
    # A location with two subscripts, which uplm80 took for an element
    # further on: `.ar(1)(1)' of an ADDRESS array AR+4 (0.4.3: AR+2 in a
    # DATA list, AR+3 in an INITIAL one, AR+4 in an AT, and `.s.m(1)(1)'
    # refused).  V3.1 reads the location to its first subscript and stops
    # at the second as at an operator it does not take (#152, in an AT
    # #146), and gives that error too after one in a subscript.
    "two-subscripts-in-data": ("declare ar (3) address;\ndeclare d address data (.ar(1)(1));\n"
                               "w = d;\n", (152,),
                               "T.PLM:9:26: error: AR(1)(1): a location takes one subscript, "
                               "not two; Intel's PL/M-80 V3.1 rejects it (ERROR #152, MISSING "
                               "')' AFTER CONSTANT LIST)"),
    "two-subscripts-in-initial": ("declare ar (3) address;\n"
                                  "declare d address initial (.ar(1)(1));\nw = d;\n", (152,),
                                  "T.PLM:9:29: error: AR(1)(1): a location takes one subscript"),
    "two-subscripts-in-at": ("declare ar (3) address;\ndeclare d byte at (.ar(1)(1));\nb = d;\n",
                             (146,),
                             "T.PLM:9:21: error: AR(1)(1): a location takes one subscript, not "
                             "two; Intel's PL/M-80 V3.1 rejects it (ERROR #146, MISSING ')' "
                             "AFTER 'AT' RESTRICTED EXPRESSION)"),
    "two-subscripts-on-a-member-in-data": ("declare s structure (m (3) address);\n"
                                           "declare d address data (.s.m(1)(1));\nw = d;\n",
                                           (152,),
                                           "T.PLM:9:26: error: S.M(1)(1): a location takes one "
                                           "subscript, not two"),
    "two-subscripts-in-a-constant-list": ("declare ar (3) address;\nw = .(7, .ar(1)(1));\n",
                                          (152, 32, 209),
                                          "T.PLM:9:11: error: AR(1)(1): a location takes one "
                                          "subscript, not two"),
    "two-subscripts-in-a-subscript-in-data": ("declare ar (3) address;\n"
                                              "declare d address data (.ar(.ar(1)(1)));\n"
                                              "w = d;\n", (151, 150, 152),
                                              "T.PLM:9:29: error: .AR(1)(1): the subscript of a "
                                              "location in a DATA or INITIAL value is numbers "
                                              "only; Intel's PL/M-80 V3.1 rejects it (ERROR #151, "
                                              "INVALID OPERAND IN RESTRICTED EXPRESSION, and #152, "
                                              "MISSING ')' AFTER CONSTANT LIST, and #150, MISSING "
                                              "')' AT END OF RESTRICTED SUBSCRIPT)"),
    # A subscript on the location of what is not an array - a scalar, a
    # structure, a member, a procedure, a label - which uplm80 took for the
    # element that far past it, with a warning that named #127, V3.1's
    # error in an expression (`.w(1)' of an ADDRESS W+2, in 0.4.3's DATA
    # list W+1), or, of a procedure, refused naming #104.
    "subscripted-scalar-in-data": ("declare d address data (.w(1));\nw = d;\n", (149,),
                                   "T.PLM:8:26: error: W(1): W is not an array, and only an "
                                   "array's location takes a subscript; Intel's PL/M-80 V3.1 "
                                   "rejects it (ERROR #149, INVALID SUBSCRIPTING IN RESTRICTED "
                                   "REFERENCE)"),
    "subscripted-scalar-in-initial": ("declare d address initial (.w(1));\nw = d;\n", (149,),
                                      "T.PLM:8:29: error: W(1): W is not an array"),
    "subscripted-scalar-in-at": ("declare d byte at (.w(1));\nb = d;\n", (149,),
                                 "T.PLM:8:21: error: W(1): W is not an array"),
    "subscripted-structure-in-data": ("declare s structure (k byte, m address);\n"
                                      "declare d address data (.s(1));\nw = d;\n", (149,),
                                      "T.PLM:9:26: error: S(1): S is not an array"),
    "subscripted-member-in-at": ("declare s structure (k byte, m address);\n"
                                 "declare d byte at (.s.k(1));\nb = d;\n", (149,),
                                 "T.PLM:9:21: error: S.K(1): S.K is not an array"),
    "subscripted-scalar-then-a-variable-in-data": ("declare d (2) address data (.w(1), b);\n"
                                                   "w = d(0);\n", (149, 151),
                                                   "T.PLM:8:30: error: W(1): W is not an array"),
    "subscripted-scalar-in-a-constant-list": ("w = .(7, .w(1));\n", (149, 209),
                                              "T.PLM:8:11: error: W(1): W is not an array"),
    "twice-subscripted-scalar-in-data": ("declare d address data (.w(1)(1));\nw = d;\n",
                                         (149, 152),
                                         "T.PLM:8:26: error: W(1): W is not an array"),
    "subscripted-procedure-in-data": ("declare d address data (.g(1));\nw = d;\n", (149,),
                                      "T.PLM:8:26: error: G(1): G is a procedure, and only an "
                                      "array's location takes a subscript; Intel's PL/M-80 V3.1 "
                                      "rejects it (ERROR #149,"),
    "subscripted-procedure-in-at": ("declare d byte at (.g(1));\nb = d;\n", (149, 211),
                                    "T.PLM:8:21: error: .G: G is a procedure, and the location "
                                    "in an AT address is a variable's, or MEMORY's; Intel's "
                                    "PL/M-80 V3.1 rejects it (ERROR #149, INVALID SUBSCRIPTING IN "
                                    "RESTRICTED REFERENCE, and #211,"),
    "subscripted-label-in-data": ("declare d address data (.lb(1));\nlb: w = d;\n", (149,),
                                  "T.PLM:8:26: error: LB(1): LB is a label, and only an array's "
                                  "location takes a subscript"),
    # The location in an AT of a procedure, which uplm80 took for its
    # address, of a label and of a BASED variable, which it refused in
    # words of its own.
    "procedure-in-at": ("declare d byte at (.f);\nb = d;\n", (211,),
                        "T.PLM:8:21: error: .F: F is a procedure, and the location in an AT "
                        "address is a variable's, or MEMORY's; Intel's PL/M-80 V3.1 rejects it "
                        "(ERROR #211, INVALID IDENTIFIER IN 'AT' RESTRICTED REFERENCE)"),
    "procedure-plus-a-variable-in-at": ("declare d byte at (.f + b);\nc = d;\n", (211, 151),
                                        "T.PLM:8:21: error: .F: F is a procedure"),
    "label-in-at": ("declare d byte at (.lb);\nlb: b = d;\n", (211,),
                    "T.PLM:8:21: error: .LB: LB is a label, and the location in an AT address "
                    "is a variable's, or MEMORY's"),
    "based-in-at": ("declare bb based w byte;\ndeclare d byte at (.bb);\nb = d;\n", (212,),
                    "T.PLM:9:21: error: .BB: BB is BASED, and has no fixed address for an AT "
                    "address to name; Intel's PL/M-80 V3.1 rejects it (ERROR #212, INVALID "
                    "RESTRICTED REFERENCE IN 'AT', BASE ILLEGAL)"),
    "subscripted-based-in-at": ("declare bb based w byte;\ndeclare d byte at (.bb(1));\nb = d;\n",
                                (149, 212), "T.PLM:9:21: error: .BB: BB is BASED"),
    "based-member-in-at": ("declare bs based w structure (k byte, m address);\n"
                           "declare d byte at (.bs.m);\nb = d;\n", (212,),
                           "T.PLM:9:21: error: .BS: BS is BASED"),
}
# What V3.1 rejects and uplm80 compiles, with a warning, as programs written
# for it rely on it: tests/test_implicit_calls.plm's `callee$func()', and
# the tests' INITIALs in procedures (80un may use them too).
V31_WARNS = {
    "f()": ("b = f();\n", (102, 153),
            "T.PLM:8:5: warning: F(): PL/M-80 has no empty argument list, and this is taken "
            "for F, a call with no arguments; Intel's PL/M-80 V3.1 rejects it (ERROR #102"),
    "call g()": ("call g();\n", (102, 153),
                 "T.PLM:8:6: warning: G(): PL/M-80 has no empty argument list"),
    "initial-in-procedure": ("p: procedure;\n  declare k byte initial (7);\n  k = k + 1; "
                             "b = k;\nend p;\ncall p; call p;\n", (73,),
                             "T.PLM:9:11: warning: K: INITIAL in procedure P initializes the "
                             "variable once, when the program is loaded, not at each entry; "
                             "Intel's PL/M-80 V3.1 rejects it (ERROR #73"),
    "initial-in-do": ("do;\n  declare k byte initial (7);\n  b = k;\nend;\n", (73,),
                      "warning: K: INITIAL in a DO block initializes the variable once"),
    # 0.4.2's Known issues, 0.4.3: tests/test_optimizer_soundness.py and
    # tests/test_calls_and_loops.py test what uplm80 makes of these.
    "subscripted-scalar": ("b = c(1);\n", (127,),
                           "T.PLM:8:5: warning: C(1): C is not an array, and this is taken for the "
                           "element that far past C, as if C were an array; Intel's PL/M-80 V3.1 "
                           "rejects it (ERROR #127, INVALID SUBSCRIPT ON NON-ARRAY)"),
    "member-of-an-unsubscripted-array": ("declare s2(2) structure (m(2) byte);\ns2.m(1) = 3;\n",
                                         (133,),
                                         "T.PLM:9:1: warning: S2.M: S2 is an array, and this is "
                                         "taken for S2(0).M; Intel's PL/M-80 V3.1 rejects it "
                                         "(ERROR #133, ILLEGAL REFERENCE TO UNSUBSCRIPTED ARRAY)"),
}


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("name", sorted(V31_REJECTS))
def test_what_v31_rejects_is_an_error(opt, name):
    """A dimension of 0, `declare z (0) byte', of an array or a member; the
    address of a built-in but MEMORY, `.double'; empty parentheses after a
    built-in, `carry()'; and a PUBLIC or EXTERNAL variable or procedure
    in a procedure or a DO block (0.4.1's Known issues).  An END that
    names another block; a DO CASE with no case; `.h(1)'; anything in
    parentheses in a subscript of LENGTH, LAST or SIZE's argument; a
    procedure with no statements; `shl(w, 3)' of a scalar SHL; an array
    or a member array without a subscript; a procedure called before its
    declaration (0.4.2's).  uplm80 compiled each.  A built-in in a DATA or
    INITIAL list, an AT address or a constant list (0.4.3's), which -O1
    and up folded and -O0 refused, or took for something else; there a
    location with two subscripts, or one on what is not an array, and in
    an AT the location of a procedure, a label or a BASED variable.
    Intel's PL/M-80 V3.1 rejects each, with the errors the message names.
    No program of MP/M II, 80un, sample_code or tests/ has one (as tests/
    now)."""
    stmts, _, message = V31_REJECTS[name]
    err = _compile_error(PRELUDE + V31_DECLS + stmts + "end t;\n", opt)
    assert message in err, err


@pytest.mark.parametrize("name", sorted(V31_WARNS))
def test_what_v31_rejects_and_programs_rely_on_is_a_warning(name):
    """Empty parentheses after a procedure, `f()' and `call g()', which
    uplm80 takes for `f' and `g', and INITIAL in a procedure or a DO
    block, which initializes the variable once: V3.1 rejects them, and
    uplm80 warns, with V3.1's error, and compiles them as before."""
    stmts, _, message = V31_WARNS[name]
    r = _compile(PRELUDE + V31_DECLS + stmts + "end t;\n")
    assert r.returncode == 0, r.stderr
    assert message in r.stderr, r.stderr


# What Intel's PL/M-80 V3.1 takes of the forms above: an array without a
# subscript after a dot and in LENGTH, LAST and SIZE, whose subscripts may
# be expressions but for anything in parentheses; REENTRANT procedures that
# call each other before their declarations (MP/M II's SN.PLM); the address
# of a procedure declared later, in an INITIAL list and elsewhere; a call
# through an address with arguments.  V3.1 compiles the program to print
# what is expected (tests/test_intel_oracle.py builds it again), and
# uplm80 compiles it without a word.
V31_ALLOWS = """
declare (b, n) byte, (w, q) address;
declare a(4) byte, ab(6) byte, s structure (m(3) byte, k byte);
declare s2(2) structure (m(5) byte), sa(3) structure (z(2) byte);
declare hx(*) byte data ('0123');
declare tab(2) address initial (.later, .ev);
f: procedure byte; return 2; end f;
ev: procedure (x) byte reentrant;
  declare x byte;
  if x = 0 then return 1;
  return od(x - 1);
end ev;
od: procedure (x) byte reentrant;
  declare x byte;
  if x = 0 then return 0;
  return ev(x - 1);
end od;
later: procedure (x, y);
  declare (x, y) byte;
  n = x + y;
end later;
lp: do;
  w = .a; call ph(w - .a);
  w = length(a) + last(a) + size(a); call ph(w);
  w = .s.m - .s; call ph(w);
  w = size(s.m) + length(s2.m) + last(sa.z); call ph(w);
  call move(2, .hx, .a); call ph(a(1));
  b = 1;
  w = size(ab(b + 1)) + size(ab(f)) + size(sa(b).z) + size(ab(.w)); call ph(w);
  call ph(hx(2));
  call ph(ev(7)); call ph(od(7));
  q = tab(0); call q(3, 4); call ph(n);
end lp;
"""


def test_only_the_parsers_tree_is_held_to_what_v31_takes():
    """check_names holds the program as the parser gives it to what V3.1
    takes, at every level; resolve_names, which code generation runs on
    the optimizer's tree, whose procedure's statements may be gone, does
    not do it again."""
    src = "t: do;\np: procedure;\n  declare k byte;\nend p;\nend t;\n"
    with pytest.raises(CodeGenError, match="ERROR #174"):
        check_names([parse_source(src, "T.PLM")])
    resolve_names([parse_source(src, "T.PLM")])


def test_what_v31_takes_of_those_forms_is_compiled_without_a_word(capsys):
    _check(V31_ALLOWS, [0, 0xB, 0, 9, 0x31, 5, 0x32, 0, 1, 7])
    capsys.readouterr()
    assert Compiler(opt_level=0).compile(_PH_PRELUDE + V31_ALLOWS + "end t;\n", "T.PLM")
    assert "warning" not in capsys.readouterr().err


# What Intel's PL/M-80 V3.1 takes in a restricted expression, which has no
# built-in in it (V31_REJECTS): MEMORY's location, in a DATA or an INITIAL
# list as in an AT, and a constant list of sums and differences.  V3.1
# compiles the program to print what is expected (tests/test_intel_oracle.py
# builds it again).
V31_RESTRICTED = """
declare a(5) byte;
declare dm address data (.memory), dm3 address data (.memory(3));
declare im(3) address initial (.a, .memory - 1, .memory + 2);
declare atm byte at (.memory);
declare p address, c based p (4) byte;
call ph(dm - .memory); call ph(dm3 - .memory);
call ph(im(0) - .a); call ph(im(1) - .memory); call ph(im(2) - .memory);
call ph(.atm - .memory);
p = .(1 + 2, -1, 'a', 9 - 2);
call ph(c(0)); call ph(c(1)); call ph(c(2)); call ph(c(3));
"""


@pytest.mark.parametrize("opt", LEVELS)
def test_what_v31_takes_in_a_restricted_expression_is_laid_out(opt):
    """`.memory' in a DATA or INITIAL list is the linker's end of the
    program, as in an expression; it was `dw MEMORY', which um80 did not
    know.  And an expression in a constant list is its value at -O0 too,
    where nothing has folded it; it was left out, and `.(1 + 2, 7)' was
    `.(7)'."""
    asm = Compiler(opt_level=opt).compile(_PH_PRELUDE + V31_RESTRICTED + "end t;\n", "T.PLM")
    lines = [" ".join(line.split()) for line in asm.splitlines()]
    for want in ("dw __END__", "dw __END__+3", "dw (__END__-1)", "dw (__END__+2)"):
        assert want in lines, want
    at = lines.index("db 3")
    assert lines[at:at + 4] == ["db 3", "db 0FFH", "db 'a'", "db 7"], lines[at - 1:at + 4]


def test_what_v31_takes_in_a_restricted_expression_prints_what_it_prints():
    _check(V31_RESTRICTED, [0, 3, 0, 0xFFFF, 2, 0, 3, 0xFF, 0x61, 7])


@pytest.mark.parametrize("name", sorted(V31_REJECTS) + sorted(V31_WARNS))
def test_the_message_names_every_error_v31_gives(name):
    """A message that names Intel's PL/M-80 V3.1's errors names each
    error V3.1 gives for the program, and no other: in a constant list
    V3.1 gives #209, #32 or #172 besides #151 and #152, and after a
    parenthesis in a subscript of LENGTH's argument #125 besides #32.
    tests/test_intel_oracle.py checks that V3.1 gives those and no
    other."""
    stmts, errors, _ = {**V31_REJECTS, **V31_WARNS}[name]
    r = _compile(PRELUDE + V31_DECLS + stmts + "end t;\n", 0)
    line = next((l for l in r.stderr.splitlines() if "Intel's PL/M-80 V3.1 rejects it (" in l),
                None)
    if line is None:
        pytest.skip("the message names none of V3.1's errors")
    named = [int(n) for n in re.findall(r"#(\d+), ", line.split("rejects it (", 1)[1])]
    assert sorted(named) == sorted(errors), line


# Locations in a DATA or INITIAL list that Intel's PL/M-80 V3.1 takes, and
# code generation laid out wrongly or refused: an element of an ADDRESS
# array, a member, an element of an array of structures and a member of
# one, a variable declared further down, a subscript that is a sum at -O0;
# and in a constant list what a byte holds of V3.1's arithmetic.  V3.1
# compiles the program to print what is expected (tests/test_intel_oracle.py
# builds it again).
V31_LOCATIONS = """
declare arr (3) address, s structure (m (3) byte, k address);
declare sa (3) structure (x byte, y address);
declare da address data (.arr(2)), dk address data (.s.k), dm address data (.s.m(1));
declare dy address data (.sa(1).y), dsa address data (.sa(2)), dx address data (.sa(2).x + 1);
declare ia (2) address initial (.later(2), .arr(1 + 1));
declare later (3) address;
declare p address, c based p (3) byte;
call ph(da - .arr); call ph(dk - .s); call ph(dm - .s);
call ph(dy - .sa); call ph(dsa - .sa); call ph(dx - .sa);
call ph(ia(0) - .later); call ph(ia(1) - .arr);
p = .(0ffffh + 1, 300 - 100, -1 - 1);
call ph(c(0)); call ph(c(1)); call ph(c(2));
"""


def test_a_location_in_a_list_is_where_v31_puts_it():
    """The module's DATA is laid out before its other variables, and an
    INITIAL list may name a variable declared further down: the element
    of an array not yet laid out was taken for a byte, `.arr(2)' of an
    ADDRESS array ARR+2, and an array of structures' for a word; a
    member, `.s.k', was refused, and so was `.arr(1 + 1)' at -O0, where
    nothing had folded the subscript."""
    _check(V31_LOCATIONS, [4, 3, 1, 4, 6, 7, 4, 4, 0, 0xC8, 0xFE])


# A name the program declares that is also a built-in's, MEMORY or SIZE, in
# a DATA list at module level: the program's variable.  It was taken for
# the built-in's while the module's DATA was laid out, before the variable
# was: `.memory' for the end of the program, and `.size(2)' of a BYTE array
# for SIZE+4, as if SIZE were an ADDRESS one.  V3.1 prints what is expected
# (tests/test_intel_oracle.py).
V31_DECLARED_BUILTINS = {
    "arrays": ("""
declare memory (4) byte, size (3) byte;
declare dm address data (.memory), dm1 address data (.memory(1));
declare ds address data (.size(2)), ds1 address data (.size(2) + 1);
call ph(dm - .memory); call ph(dm1 - .memory); call ph(ds - .size); call ph(ds1 - .size);
""", [0, 1, 2, 3]),
    "scalar": ("""
declare memory address;
declare dm address data (.memory);
call ph(dm - .memory);
""", [0]),
}


@pytest.mark.parametrize("name", sorted(V31_DECLARED_BUILTINS))
def test_a_declared_memory_or_size_in_a_data_list_is_the_programs(name):
    body, expect = V31_DECLARED_BUILTINS[name]
    _check(body, expect)


# A location in a DATA list or an AT address of a procedure or a DO block
# that names a variable the block declares further down: that variable,
# as PL/M-80 scopes a name to its whole block (9.1), and as V3.1 prints
# what is expected (tests/test_intel_oracle.py).  Code generation looked
# the name up among what it had laid out so far, and at module level
# only after that: the block's MEMORY was the end of the program
# (`dw __END__', 0004), `.arr(2)' and `.size(2)' the module's arrays of
# the other type, `buf' in an AT the module's, a structure's member not
# found, and a name the module does not declare um80's undefined symbol;
# at module level an AT's `.memory(3)' was the end of the program too.
V31_BLOCK_LOCATIONS = {
    "procedure": ("""
declare arr (3) byte, size (3) address, buf (4) address;
p: procedure;
  declare dm address data (.memory), dm1 address data (.memory(1));
  declare da address data (.arr(2)), ds address data (.size(2)), dl address data (.later(1));
  declare dk address data (.s.k), dy address data (.sa(2).y);
  declare z byte at (.buf(3)), zm byte at (.memory(3));
  declare memory (4) byte, arr (3) address, size (3) byte, buf (4) byte, later (2) address;
  declare s structure (m (3) byte, k address), sa (3) structure (x byte, y address);
  call ph(dm - .memory); call ph(dm1 - .memory); call ph(da - .arr); call ph(ds - .size);
  call ph(dl - .later); call ph(dk - .s); call ph(dy - .sa); call ph(.z - .buf);
  call ph(.zm - .memory);
end p;
call p;
""", [0, 1, 4, 2, 2, 3, 7, 3, 3]),
    "do-block": ("""
declare arr (3) byte;
do;
  declare w address data (.arr(2)), wl address data (.later(1));
  declare z byte at (.later(3));
  declare arr (3) address, later (4) address;
  call ph(w - .arr); call ph(wl - .later); call ph(.z - .later);
end;
""", [4, 2, 6]),
    "module-at": ("""
declare z byte at (.memory(3));
declare memory (4) byte;
call ph(.z - .memory);
""", [3]),
}


@pytest.mark.parametrize("name", sorted(V31_BLOCK_LOCATIONS))
def test_a_location_names_its_blocks_declaration_further_down(name):
    body, expect = V31_BLOCK_LOCATIONS[name]
    _check(body, expect)


def test_an_initial_location_names_its_procedures_memory_further_down(capsys):
    """INITIAL in a procedure, which V3.1 rejects (#73) and uplm80 takes
    with a warning, resolves a location as DATA does: the procedure's own
    MEMORY, declared after it."""
    _check("""
p: procedure;
  declare w address initial (.memory(1));
  declare memory (4) byte;
  call ph(w - .memory);
end p;
call p;
""", [1])
    capsys.readouterr()


@pytest.mark.parametrize("opt", LEVELS)
def test_memory_in_a_list_of_a_module_that_does_not_declare_it(opt):
    """In a multi-file compile a module that does not declare MEMORY means
    the built-in in a DATA list too, as in an expression and an AT, when
    another module declares a MEMORY PUBLIC: its DATA had `dw MEMORY',
    the other module's variable, and its code the end of the program."""
    user = _BUILTINS_USER.replace(
        "declare w address, b byte;",
        "declare dm address data (.memory(2)), am byte at (.memory(2));\n"
        "declare w address, b byte;").replace(
        "call ph(own);", "call ph(own);\ncall ph(dm - .memory); call ph(.am - .memory);") % ""
    assert _run_modules([user, _BUILTINS_LIB], opt).endswith("79BC 0002 0002 ")


# More values than the declaration holds, each past its space a number a
# byte does not hold: Intel's PL/M-80 V3.1 gives #209 alone, checking
# nothing past the space (tests/test_intel_oracle.py).
PAST_THE_SPACE = {
    "structure": "declare d structure (p byte, q address, r byte) data ('ABCD', 300);\n"
                 "c = d.p;\n",
    "scalar": "declare d byte data (1, 300);\nc = d;\n",
}


@pytest.mark.parametrize("name", sorted(PAST_THE_SPACE))
def test_a_value_past_a_declarations_space_is_not_held_to_a_byte(name):
    """uplm80 lays such a value out after the declaration, as 0.4.3's
    Known issues have it (#209); the message said V3.1 rejects it for a
    BYTE's value over 255 (#210), which V3.1 does not give there."""
    r = _compile(PRELUDE + V31_DECLS + PAST_THE_SPACE[name] + "end t;\n", 0)
    assert r.returncode == 0, r.stderr
    assert "#210" not in r.stderr, r.stderr


@pytest.mark.parametrize("name", ["stackptr", "shl", "double"])
def test_the_location_of_a_built_in_in_a_list_is_uplm80s_own_refusal(name):
    """Intel's PL/M-80 V3.1 takes `data (.stackptr)' for an address of its
    own (tests/test_intel_oracle.py), and uplm80, which has none to give,
    refuses it, as it did; the message said V3.1 rejects it (#123), which
    V3.1 does in an expression only."""
    err = _compile_error(PRELUDE + f"declare d address data (.{name});\nend t;\n", 0)
    assert f"uplm80 has none to give {name.upper()} in a DATA or INITIAL list" in err, err
    assert "rejects" not in err, err


# The location of what has no fixed address in uplm80, in a DATA or INITIAL
# list or an AT: a REENTRANT procedure's local or parameter, on the stack
# at each call, and a BASED variable, at the address its base holds.  V3.1
# takes each for an address (tests/test_intel_oracle.py, the CHANGELOG's
# Known issues); uplm80 refuses each, a scalar in an AT as it did, and in a
# list since 0.4.4, where it was `dw X', which um80 did not know
# ("Undefined symbol").  Each: the statements, after PRELUDE and V31_DECLS;
# the name; why; and whether V3.1 compiles it (INITIAL in a procedure it
# does not, #73).
NO_FIXED_ADDRESS = {
    "reentrant-local": ("p: procedure reentrant;\n  declare x address;\n"
                        "  declare d address data (.x);\n  x = d;\nend p;\ncall p;\n",
                        "X", "a REENTRANT local", True),
    "reentrant-local-further-down": ("p: procedure reentrant;\n  declare d address data (.x);\n"
                                     "  declare x address;\n  x = d;\nend p;\ncall p;\n",
                                     "X", "a REENTRANT local", True),
    "reentrant-element": ("p: procedure reentrant;\n  declare x (3) address;\n"
                          "  declare d address data (.x(1));\n  x(0) = d;\nend p;\ncall p;\n",
                          "X", "a REENTRANT local", True),
    "reentrant-parameter": ("p: procedure (x) reentrant;\n  declare x address;\n"
                            "  declare d address data (.x);\n  x = d;\nend p;\ncall p(1);\n",
                            "X", "a REENTRANT local", True),
    "reentrant-local-in-a-do-block": ("p: procedure reentrant;\n  declare x address;\n  do;\n"
                                      "    declare d address data (.x);\n    x = d;\n  end;\n"
                                      "end p;\ncall p;\n", "X", "a REENTRANT local", True),
    "reentrant-initial": ("p: procedure reentrant;\n  declare x address;\n"
                          "  declare d address initial (.x);\n  x = d;\nend p;\ncall p;\n",
                          "X", "a REENTRANT local", False),
    "reentrant-at": ("p: procedure reentrant;\n  declare x address;\n"
                     "  declare z address at (.x);\n  x = z;\nend p;\ncall p;\n",
                     "X", "a REENTRANT local", True),
    "reentrant-element-at": ("p: procedure reentrant;\n  declare x (3) address;\n"
                             "  declare z address at (.x(1));\n  x(0) = z;\nend p;\ncall p;\n",
                             "X", "a REENTRANT local", True),
    "based-data": ("declare bb based w byte;\ndeclare d address data (.bb);\nw = d;\n",
                   "BB", "BASED", True),
    "based-initial": ("declare bb based w byte;\ndeclare d address initial (.bb);\nw = d;\n",
                      "BB", "BASED", True),
    "based-element": ("declare ba based w (3) byte;\ndeclare d address data (.ba(1));\nw = d;\n",
                      "BA", "BASED", True),
}


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("name", sorted(NO_FIXED_ADDRESS))
def test_a_location_with_no_fixed_address_is_uplm80s_own_refusal(opt, name):
    """uplm80 refuses, at every level, the location of what it gives no
    fixed address, a REENTRANT procedure's local or a BASED variable, in a
    list as in an AT; in a list it compiled to `dw X', and um80 failed, and
    in an AT so did an array or a structure, `at (.x(1))', `EQU X+2'."""
    stmts, var, why, _ = NO_FIXED_ADDRESS[name]
    err = _compile_error(PRELUDE + V31_DECLS + stmts + "end t;\n", opt)
    line = next(l for l in err.splitlines() if ": error: " in l)
    assert line.endswith(f"{var} has no fixed address ({why})"), err


def test_an_untyped_data_string_is_an_array():
    """uplm80 takes `declare hx data ('0123')', which V3.1 does not (ERROR
    #61), for an array of the string's bytes, as 80un's bas.plm has it:
    `hx(i)' is no subscript on a scalar."""
    r = _compile(PRELUDE + "declare hx data ('0123'), i byte;\ni = 1;\ncall pc(hx(i));\nend t;\n")
    assert r.returncode == 0 and "warning" not in r.stderr, r.stderr


def test_the_address_of_memory_is_no_error():
    """MEMORY is the one built-in with an address."""
    r = _compile(PRELUDE + V31_DECLS + "w = .memory; w = .memory(3);\nend t;\n")
    assert r.returncode == 0 and "warning" not in r.stderr, r.stderr


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("stmt, name, col", [
    ("y = nosuch + 1;", "NOSUCH", 5),
    ("call noproc;", "NOPROC", 6),
    ("call noproc2(1, 2);", "NOPROC2", 6),
    ("y = nofunc(3);", "NOFUNC", 5),
    ("y = .nowhere;", "NOWHERE", 6),
    ("if 0 then y = gone;", "GONE", 15),
    ("y = n;", "NN", 5),
])
def test_a_name_declared_nowhere_is_an_error(opt, stmt, name, col):
    """`y = nosuch + 1' compiled to `ld hl,(NOSUCH)', and only um80
    reported it, as an undefined symbol (0.3.6 the same); the optimizer
    dropped a use it could prove unreached, so that `if 0 then y = gone'
    compiled at -O1 and up.  Intel's PL/M-80 V3.1: ERROR #105, UNDECLARED
    IDENTIFIER, for each, and for a LITERALLY whose text is a name
    declared nowhere (`n literally 'nn''; its own name is declared again
    in `p', which the macro pass makes a declaration of NN there)."""
    src = PRELUDE + """declare n literally 'nn';
declare (x, y) address;
p: procedure; declare n byte; n = 1; end p;
""" + stmt + "\nend t;\n"
    err = _compile_error(src, opt)
    assert (f"T.PLM:8:{col}: error: {name} is not declared "
            "(Programming Manual 9800268B, 6.1)") in err, err


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("src, line, col", [
    ("p: procedure;\n  y = lit;\nend p;\ndeclare lit literally '5';\ncall p;", 7, 7),
    ("p: procedure;\n  do case y;\n    y = 1;\n    y = lit;\n  end;\nend p;\n"
     "declare lit literally '5';\ncall p;", 9, 9),
    ("p: procedure;\n  q: procedure;\n    y = lit;\n  end q;\n"
     "  declare lit literally '5';\n  call q;\nend p;\ncall p;", 8, 9),
])
def test_a_literally_used_before_its_declaration_is_an_error(opt, src, line, col):
    """A LITERALLY's text is "substituted for each occurrence of the
    identifier in subsequent text" (6.4), so a use of the name before the
    declaration is of a name not declared there.  uplm80 put the text in
    its place anyway: `y = lit' in a procedure the module's `declare lit
    literally '5'' follows compiled to `y = 5' (0.4.0 the same).  Intel's
    PL/M-80 V3.1 rejects each: ERROR #105, UNDECLARED IDENTIFIER."""
    err = _compile_error(PRELUDE + "declare y byte;\n" + src + "\nend t;\n", opt)
    assert (f"T.PLM:{line}:{col}: error: LIT is not declared here: a LITERALLY declared "
            "after it puts its text in place of LIT only in the text that follows the "
            "declaration (Programming Manual 9800268B, 6.4)") in err, err


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("decl", [
    "declare a (lit) byte;\ndeclare lit literally '3';",
    "declare a (nosuch) byte;",
    "declare n byte, a (n) byte;",
    "p: procedure;\n  declare lit literally '4';\n  y = lit;\nend p;\ndeclare a (lit) byte;",
    "declare s structure (m (lit) byte);\ndeclare lit literally '3';",
])
def test_a_dimension_that_is_not_a_number_is_an_error(opt, decl):
    """"A dimension specifier is a numeric constant in parentheses"
    (6.2.5), which a LITERALLY declared before it can give.  A name left
    there - a LITERALLY declared after it, or out of its scope, a variable,
    a name declared nowhere - made the array a scalar: `declare a (lit)
    byte' with `lit literally '3'' after it was one byte (0.4.0 the same).
    Intel's PL/M-80 V3.1 rejects each: ERROR #59, ILLEGAL DIMENSION
    ATTRIBUTE."""
    err = _compile_error(PRELUDE + "declare y byte;\n" + decl + "\ny = 1;\nend t;\n", opt)
    name = re.search(r"\((\w+)\) byte", decl).group(1).upper()
    assert (f"error: ({name}): the dimension of an array is a number, and {name} is not a "
            "LITERALLY declared before it whose text is one (Programming Manual 9800268B, "
            "6.2.5)") in err, err


@pytest.mark.parametrize("opt", LEVELS)
def test_a_literally_used_after_its_declaration_and_a_variable_before_its_own(opt):
    """A LITERALLY declared before a procedure is its text in it, and a
    variable declared after a procedure that uses it is the variable, in
    both compilers."""
    assert run_plm(PRELUDE + """declare lit literally '41h';
declare y byte;
p: procedure;
  y = lit;
  x = y + 1;
end p;
declare x byte;
call p;
call pc(y); call pc(x);
end t;
""", opt) == "AB"


def test_a_built_in_needs_no_declaration():
    r = _compile(PRELUDE + """declare (x, y) address, b (4) byte;
y = low(x) + high(x) + double(1) + shl(x, 1) + shr(x, 1) + rol(1, 1) + ror(1, 1)
  + scl(x, 1) + scr(x, 1) + length(b) + last(b) + size(b) + input(3) + dec(1)
  + memory(0) + stackptr + carry + zero + sign + parity;
call move(1, .x, .y); call time(1); output(3) = 1;
end t;
""")
    assert r.returncode == 0, r.stderr


_UNDECLARED_PARAMETER = ("and no DECLARE of the procedure declares it; a parameter is declared "
                         "a BYTE or an ADDRESS scalar, not BASED, by a DECLARE of its "
                         "procedure (Programming Manual 9800268B, 8.1.1)")
_PARAMETER_FORM = ("and a parameter is declared a BYTE or an ADDRESS scalar, not BASED and "
                   "with no other attribute (Programming Manual 9800268B, 8.1.1)")


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("proc, name", [
    ("p: procedure (a);\n  y = a;\nend p;", "A"),
    ("p: procedure (a, b);\n  declare a byte;\n  y = a + b;\nend p;", "B"),
    ("p: procedure (a);\n  y = 1;\nend p;", "A"),
    ("p: procedure (a) external;\nend p;", "A"),
    ("p: procedure (a) reentrant;\n  y = 1;\nend p;", "A"),
    ("p: procedure (a);\n  do;\n    declare a byte;\n    y = a;\n  end;\nend p;", "A"),
])
def test_a_parameter_no_declare_declares_is_an_error(opt, proc, name):
    """"Each formal parameter must be declared as a non-based scalar
    variable in a DECLARE statement preceding the first executable
    statement in the procedure body" (8.1.1).  A parameter only the
    PROCEDURE statement names was an ADDRESS, and `y = a + b' compiled to
    `ld (??AUTO+0),hl / ld a,l / ld (Y),a' (0.4.0 the same); a DO block's
    DECLARE of the name declares a variable of the block.  Intel's PL/M-80
    V3.1 rejects each: ERROR #25, UNDECLARED PARAMETER, and #105,
    UNDECLARED IDENTIFIER, at a use of it."""
    src = PRELUDE + "declare y byte;\n" + proc + "\nend t;\n"
    err = _compile_error(src, opt)
    col = proc.split("\n")[0].index(name.lower()) + 1
    assert (f"T.PLM:6:{col}: error: {name} is a parameter of P, "
            + _UNDECLARED_PARAMETER) in err, err


@pytest.mark.parametrize("opt", LEVELS)
@pytest.mark.parametrize("decl", [
    "a (3) byte",
    "a based w byte",
    "a label",
    "a structure (m byte)",
    "a byte public",
    "a byte initial (3)",
    "a byte at (.y)",
    "a byte data (1)",
    "a address external",
])
def test_a_parameter_declared_as_anything_but_a_scalar_is_an_error(opt, decl):
    """A parameter declared an array, BASED, a LABEL or a structure, or
    with PUBLIC, EXTERNAL, INITIAL, DATA or AT, compiled as one (0.4.0 the
    same).  Intel's PL/M-80 V3.1 rejects each: ERROR #76, CONFLICTING
    ATTRIBUTE WITH PARAMETER, #77, INVALID PARAMETER DECLARATION, BASE
    ILLEGAL, and #79, ILLEGAL PARAMETER TYPE, NOT BYTE OR ADDRESS."""
    src = (PRELUDE + "declare y byte, w address;\np: procedure (a);\n  declare " + decl
           + ";\n  y = 1;\nend p;\ncall p(1);\nend t;\n")
    err = _compile_error(src, opt)
    assert "T.PLM:7:11: error: A is a parameter of P, " + _PARAMETER_FORM in err, err


def test_a_parameter_declared_twice_is_an_error():
    """Intel's PL/M-80 V3.1: ERROR #78, DUPLICATE DECLARATION."""
    err = _compile_error(PRELUDE + """declare y byte;
q: procedure (a) byte;
  declare a byte;
  declare a byte;
  return a;
end q;
y = q(1);
end t;
""")
    assert "T.PLM:8:11: error: A is declared twice in the same block" in err, err


@pytest.mark.parametrize("opt", LEVELS)
def test_parameters_declared_with_locals_and_on_their_own(opt):
    """Every parameter declared once, as PL/M-80 asks, in any order, with
    locals in one factored declaration or on its own."""
    assert run_plm(PRELUDE + """declare y byte;
p: procedure (a, b, c) byte;
  declare (x, b) byte, c address;
  declare a byte;
  x = a + b;
  return x + low(c);
end p;
e: procedure (a) byte external;
  declare a byte;
end e;
y = p(20h, 21h, 0ffh);
call pc(y);
end t;
""", opt) == "@"


@pytest.mark.parametrize("opt", LEVELS)
def test_a_literallys_name_declared_again_in_an_inner_block(opt):
    """A LITERALLY's text is "substituted for each occurrence of the
    identifier in subsequent text" (6.4), in its scope, and that takes in
    a declaration of the name in an inner block: `n literally '5'' makes a
    procedure's `declare n byte' `declare 5 byte'.  Intel's PL/M-80 V3.1
    does the same, ERROR #48, ILLEGAL DECLARATION STATEMENT SYNTAX, and so
    does uplm80, which now says where the 5 came from."""
    err = _compile_error(PRELUDE + """declare n literally '5';
declare w address;
p: procedure;
  declare n byte;
  n = 3;
  w = n;
end p;
w = n;
end t;
""", opt)
    assert ("T.PLM:8:11: error: unexpected token 'NUMBER' '5'; expected one of: IDENT, LPAREN; "
            "that is the text of N, declared LITERALLY '5', which PL/M-80 puts in place of N "
            "wherever it occurs in the LITERALLY's scope (Programming Manual 9800268B, "
            "6.4)") in err, err


def test_the_note_names_the_literally_whose_text_the_token_is():
    """With `nn literally '5', n literally 'nn'', an inner `declare n
    byte' is `declare 5 byte' by way of NN; the note said the 5 was the
    text of N, declared LITERALLY 'nn'.  It is NN's, in N's text.  Intel's
    PL/M-80 V3.1: ERROR #48, ILLEGAL DECLARATION STATEMENT SYNTAX, near
    NN."""
    err = _compile_error(PRELUDE + """declare nn literally '5', n literally 'nn';
declare w address;
p: procedure;
  declare n byte;
  n = 3;
  w = n;
end p;
w = n;
end t;
""")
    assert ("T.PLM:8:11: error: unexpected token 'NUMBER' '5'; expected one of: IDENT, LPAREN; "
            "that is the text of NN, declared LITERALLY '5', in the text of N, declared "
            "LITERALLY 'nn', and PL/M-80 puts a LITERALLY's text in place of its name "
            "wherever it occurs in the LITERALLY's scope (Programming Manual 9800268B, "
            "6.4)") in err, err


@pytest.mark.parametrize("opt", LEVELS)
def test_a_literally_whose_text_is_a_name_declares_that_name_again(opt):
    """With `m literally 'w'', q's `declare m byte' declares a W of q's
    own, which q's `m = 7' sets, and p's `m = 9' still sets the module's W.
    As MP/M II's MPMLDR needs `mon1 literally 'ldmon1'' to make its `mon1:
    procedure external' LDMON1.  The program compiled by Intel's PL/M-80
    V3.1 prints 79 too."""
    assert run_plm(PRELUDE + """declare w address;
p: procedure;
  declare m literally 'w';
  q: procedure;
    declare m byte;
    m = 7;
    call pc('0' + m);
  end q;
  m = 9;
  call q;
  call pc('0' + low(m));
end p;
call p;
end t;
""", opt) == "79"
    asm = Compiler(opt_level=opt).compile(
        "t: do;\ndeclare mon1 literally 'ldmon1';\n"
        "mon1: procedure (f, a) external; declare f byte, a address; end mon1;\n"
        "declare b byte;\ncall mon1(b, 65);\nend t;\n", "<t>")
    assert asm is not None
    assert re.search(r"^\s*extrn\s+LDMON1\s*$", asm, re.I | re.M), asm
    assert re.search(r"^\s*call\s+LDMON1\s*$", asm, re.I | re.M), asm
