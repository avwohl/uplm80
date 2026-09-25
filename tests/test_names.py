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

from ._toolchain import compile_cmd, compiler_env, run_plm

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


@pytest.mark.parametrize("opt", (0, 3))
def test_the_address_of_a_procedures_label(opt):
    """`.there' in a procedure is its own label, `@P$THERE'; it was `THERE',
    which nothing defines."""
    assert run_plm(PRELUDE + """
declare a address;
p: procedure;
   declare q address;
   q = .there;
   goto there;
there:
   call pc('P');
end p;
a = .here;
call p;
here:
call pc('M');
end t;
""", opt) == "PM"


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


def test_a_call_through_an_address_with_two_arguments_warns():
    """A procedure private to its module takes all but its last argument in
    its own storage, which a call through an address cannot reach."""
    r = _compile(PRELUDE + """declare q address;
pub: procedure (a, b) public; declare (a, b) byte; call pc(a); call pc(b); end pub;
q = .pub;
call q(1, 2);
end t;
""")
    assert r.returncode == 0, r.stderr
    assert ("T.PLM:8:1: warning: a CALL through an address passes more than one argument "
            "only to a PUBLIC or REENTRANT procedure") in r.stderr, r.stderr

