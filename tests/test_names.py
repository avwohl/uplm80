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
import subprocess
import tempfile

import pytest

from ._toolchain import compile_cmd, compiler_env, run_asm, run_plm, tools_missing

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
    an expression, a DATA and an INITIAL list and an AT, and is called
    through it, with an argument where it takes one."""
    src = PRELUDE + """
extp: procedure external; end extp;
declare q address;
declare s structure (p address);
declare mt (3) address data (.top, .extp, .pub);
declare mi address initial (.top);
declare atop byte at (.top);
top: procedure; call pc('T'); end top;
pub: procedure (w) public; declare w address; call pc(low(w)); end pub;
re: procedure (c) reentrant; declare c byte; call pc(c); end re;
outer: procedure;
  declare t (2) address data (.show, .inner2);
  declare r address initial (.show);
  declare atx byte at (.show);
  show: procedure; call pc('S'); end show;
  inner2: procedure (c); declare c byte; call pc(c); end inner2;
  q = .show; call q;
  q = t(0); call q;
  q = t(1); call q('I');
  q = r; call q;
  if .atx = .show then call pc('=');
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
if .atop = .top then call pc('=');
s.p = .extp; call s.p;
call pc('.');
end t;
"""
    assert run_plm(src, opt, extra_asm=EXTP) == "SSIS=S/TTEPTpRE=E."


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
    err = _compile_error(PRELUDE + "declare i byte;\np: procedure; end p;\ndeclare p byte;\nend t;\n")
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
                         "p: procedure public; end p;\nend t;\n")
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
    `f()' is still taken for `f' (V3.1 rejects that too)."""
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


def test_empty_parentheses_after_a_parameter_are_an_error():
    err = _compile_error(PRELUDE + "q: procedure (x) byte; declare x byte; return x(); end q;\n"
                         "call pc(q(1));\nend t;\n")
    assert "T.PLM:5:47: error: X(): X is a parameter" in err, err
